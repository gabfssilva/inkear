"""The Core ML pipeline, in the shape the Swift side implements: tokenizer, prompt, the generation loop
with host-owned caches, and chunked codec decoding."""

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import coremltools as ct
import numpy as np
import sentencepiece as spm
import torch

from moss_nano_coreml import graphs, reference

F16 = np.float16

# coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML drops them later on its
# own queue without the GIL (apple/coremltools#2829). Every input stays referenced: per-call inputs
# here, and the large caches live in buffers owned by the pipeline for its whole life.
_PINNED: list[np.ndarray] = []


def _pin(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    _PINNED.extend(arrays.values())
    return arrays


MAX_FRAMES = 375
# Frames per text token of a usable take, from 2 voices x 2 sentences (28 and 59 tokens): good takes
# fell in 1.9-3.7. Sampling sometimes stops mid-sentence (1.3-1.7) or keeps babbling (4.5-6.4).
PLAUSIBLE_FRAMES_PER_TOKEN = (1.8, 4.2)


@dataclass(frozen=True)
class Quality:
    """RVQ depth generated per frame, and how many times an implausibly long or short take is redrawn."""

    depth: int
    retries: int


QUALITY = {
    "low": Quality(depth=12, retries=0),
    "normal": Quality(depth=16, retries=1),
    "high": Quality(depth=16, retries=2),
    "highest": Quality(depth=16, retries=4),
}


@dataclass(frozen=True)
class Settings:
    """Sampling knobs fed to the local decoder; defaults are upstream's."""

    depth: int = 16
    text_temperature: float = 1.0
    temperature: float = 0.8
    top_k: int = 25
    top_p: float = 0.95
    penalty: float = 1.2
    max_frames: int = MAX_FRAMES

    def row(self) -> np.ndarray:
        return np.array(
            [
                [
                    1 / self.text_temperature,
                    1 / self.temperature,
                    self.top_k,
                    self.top_p,
                    self.penalty,
                    1 / self.penalty,
                ]
            ],
            dtype=F16,
        )


@dataclass
class Timings:
    prefill: float = 0.0
    local: list[float] = field(default_factory=list)
    step: list[float] = field(default_factory=list)
    codec: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class Synthesis:
    codes: np.ndarray
    audio: np.ndarray
    ended: bool
    seconds: float
    timings: Timings

    @property
    def duration(self) -> float:
        return self.audio.shape[1] / reference.SAMPLE_RATE

    @property
    def rtf(self) -> float:
        return self.seconds / self.duration


class Tokenizer:
    """SentencePiece encoding and the upstream voice-clone prompt layout."""

    def __init__(self, directory: Path) -> None:
        self.processor = spm.SentencePieceProcessor(model_file=str(directory / "tokenizer.model"))
        config = json.loads((directory / "config.json").read_text())
        self.tokens: Mapping[str, int] = config["tokens"]
        self.template: Mapping[str, str] = config["prompt"]

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def prompt(self, text: str, voice: np.ndarray) -> np.ndarray:
        """Rows [n, 17]: text rows carry the audio pad id; reference rows carry the user slot id and codes."""
        t, tokens = self.template, self.tokens
        head = [
            tokens["im_start"],
            *self.encode(t["user_prefix"]),
            *self.encode(t["reference_prefix"]),
            tokens["audio_start"],
        ]
        tail = [
            tokens["audio_end"],
            *self.encode(t["after_reference"]),
            *self.encode(text),
            *self.encode(t["user_suffix"]),
            tokens["im_end"],
            *self.encode(t["assistant_prefix"]),
            tokens["im_start"],
            *self.encode(t["assistant_role"]),
            tokens["audio_start"],
        ]

        def text_rows(ids: Sequence[int]) -> np.ndarray:
            rows = np.full((len(ids), 17), graphs.AUDIO_PAD, dtype=np.int32)
            rows[:, 0] = ids
            return rows

        audio = np.concatenate([np.full((len(voice), 1), tokens["audio_user_slot"], np.int32), voice], 1)
        return np.concatenate([text_rows(head), audio.astype(np.int32), text_rows(tail)])


class Pipeline:
    def __init__(self, directory: Path, compute_units: ct.ComputeUnit) -> None:
        self.config = json.loads((directory / "config.json").read_text())
        self.tokenizer = Tokenizer(directory)
        compiled = {
            name: ct.models.utils.compile_model(str(directory / f"{name}.mlpackage"))
            for name in ("GlobalDecoder", "LocalDecoder", "CodecDecoder")
        }

        def load(package: str, function: str) -> ct.models.CompiledMLModel:
            return ct.models.CompiledMLModel(
                compiled[package], compute_units=compute_units, function_name=function
            )

        self.prefill = {
            rows: load("GlobalDecoder", f"prefill_{rows}") for rows in self.config["prefill_rows"]
        }
        self.step = {context: load("GlobalDecoder", f"step_{context}") for context in self.config["contexts"]}
        self.local = {
            depth: load("LocalDecoder", f"frame_{depth}") for depth in self.config["codebook_depths"]
        }
        chunk = self.config["chunk_frames"]
        self.codec = load("CodecDecoder", f"chunk_{chunk}")
        self.caches = {
            context: (
                np.zeros((graphs.LAYERS, graphs.HEADS, context, graphs.HEAD_DIM), F16),
                np.zeros((graphs.LAYERS, graphs.HEADS, context, graphs.HEAD_DIM), F16),
            )
            for context in self.config["contexts"]
        }
        self.codec_caches = {
            name: np.zeros(shape, F16) for name, shape in self.config["codec_caches"].items()
        }
        _PINNED.extend(
            [*(array for pair in self.caches.values() for array in pair), *self.codec_caches.values()]
        )

    def speak(self, text: str, voice: np.ndarray, quality: Quality, seed: int) -> list[Synthesis]:
        """Synthesize at `quality`: redraw (seed + 1, ...) while a take ends outside the plausible
        length or never ends. Returns every take; the last one is the result."""
        tokens = len(self.tokenizer.encode(text))
        shortest, longest = (bound * tokens for bound in PLAUSIBLE_FRAMES_PER_TOKEN)
        settings = Settings(depth=quality.depth, max_frames=min(MAX_FRAMES, int(longest) + 1))
        takes: list[Synthesis] = []
        while len(takes) <= quality.retries:
            takes.append(self.synthesize(text, voice, settings, seed + len(takes)))
            if takes[-1].ended and shortest <= len(takes[-1].codes) <= longest:
                break
        return takes

    def synthesize(self, text: str, voice: np.ndarray, settings: Settings, seed: int) -> Synthesis:
        rng = np.random.default_rng(seed)
        timings = Timings()
        started = time.perf_counter()
        rows = self.tokenizer.prompt(text, voice)
        hidden, length = self._prefill(rows, timings)
        decoder = _CodecStream(self, timings)
        seen = np.zeros((settings.depth, graphs.CODEBOOK_SIZE), F16)
        frames: list[np.ndarray] = []
        ended = False
        largest = max(self.config["contexts"])
        while len(frames) < settings.max_frames and length + len(frames) < largest:
            codes, ended = self._sample(hidden, seen, settings, rng, timings)
            if ended:
                break
            seen[np.arange(settings.depth), codes[: settings.depth]] = 1
            frames.append(codes)
            decoder.push(codes)
            hidden = self._step(codes, length + len(frames) - 1, timings)
        audio = decoder.finish()
        return Synthesis(
            codes=np.array(frames, dtype=np.int32).reshape(-1, graphs.CODEBOOKS),
            audio=audio,
            ended=ended,
            seconds=time.perf_counter() - started,
            timings=timings,
        )

    def _prefill(self, rows: np.ndarray, timings: Timings) -> tuple[np.ndarray, int]:
        length = len(rows)
        size = next(size for size in self.config["prefill_rows"] if size >= length)
        padded = np.full((size, 17), graphs.AUDIO_PAD, np.int32)
        padded[:, 0] = self.config["tokens"]["pad"]
        padded[:length] = rows
        start = time.perf_counter()
        out = self.prefill[size].predict(_pin(rows=padded))
        timings.prefill = time.perf_counter() - start
        for keys, values in self.caches.values():
            context = min(length, keys.shape[2])
            keys[:, :, :context] = out["keys"][:, :, :context]
            values[:, :, :context] = out["values"][:, :, :context]
        return out["hidden"][length - 1 : length], length

    def _sample(
        self,
        hidden: np.ndarray,
        seen: np.ndarray,
        settings: Settings,
        rng: np.random.Generator,
        timings: Timings,
    ) -> tuple[np.ndarray, bool]:
        uniform = rng.random((settings.depth + 1, graphs.CANDIDATES))
        gumbel = (-np.log(-np.log(np.clip(uniform, 1e-12, 1 - 1e-12)))).astype(F16)
        start = time.perf_counter()
        out = self.local[settings.depth].predict(
            _pin(
                hidden=hidden.astype(F16),
                seen=seen.copy(),
                gumbel=gumbel[1:].copy(),
                text_gumbel=gumbel[:1, :2].copy(),
                settings=settings.row(),
            )
        )
        timings.local.append(time.perf_counter() - start)
        codes = np.full(graphs.CODEBOOKS, graphs.AUDIO_PAD, np.int32)
        codes[: settings.depth] = np.rint(out["codes"][0]).astype(np.int32)
        return codes, bool(out["end"][0, 0] > 0.5)

    def _step(self, codes: np.ndarray, position: int, timings: Timings) -> np.ndarray:
        """Feed the frame just sampled at `position`; returns the hidden state that predicts the next one."""
        context = next(size for size in self.config["contexts"] if size >= position)
        keys, values = self.caches[context]
        mask = np.where(np.arange(context) < position, 0.0, graphs.MASKED).astype(F16)[None]
        cos, sin = (
            table.numpy().astype(F16)
            for table in graphs.rope_tables(torch.tensor([position]), graphs.HEAD_DIM)
        )
        row = np.concatenate([[self.config["tokens"]["audio_assistant_slot"]], codes]).astype(np.int32)[None]
        start = time.perf_counter()
        out = self.step[context].predict(
            _pin(row=row, key_cache=keys, value_cache=values, mask=mask, cos=cos, sin=sin)
        )
        timings.step.append(time.perf_counter() - start)
        for other_keys, other_values in self.caches.values():
            if position < other_keys.shape[2]:
                other_keys[:, :, position] = out["key"][:, :, 0]
                other_values[:, :, position] = out["value"][:, :, 0]
        return out["hidden"]


class _CodecStream:
    """Decodes frames in fixed chunks as they arrive; the last partial chunk is padded and trimmed."""

    def __init__(self, pipeline: Pipeline, timings: Timings) -> None:
        self.pipeline = pipeline
        self.timings = timings
        self.chunk = pipeline.config["chunk_frames"]
        self.window = pipeline.config["codec_window_frames"]
        self.pending: list[np.ndarray] = []
        self.decoded = 0
        self.audio: list[np.ndarray] = []
        for cache in pipeline.codec_caches.values():
            cache[...] = 0

    def push(self, codes: np.ndarray) -> None:
        self.pending.append(codes)
        if len(self.pending) == self.chunk:
            self._decode()

    def finish(self) -> np.ndarray:
        if self.pending:
            self._decode()
        if not self.audio:
            return np.zeros((2, 0), np.float32)
        return np.concatenate(self.audio, 1)

    def _decode(self) -> None:
        count = len(self.pending)
        codes = np.full((self.chunk, graphs.CODEBOOKS), graphs.AUDIO_PAD, F16)
        codes[:count] = np.array(self.pending)
        caches = self.pipeline.codec_caches
        past = np.array([[min(self.decoded, self.window)]], F16)
        start = time.perf_counter()
        out = self.pipeline.codec.predict(_pin(codes=codes, past_frames=past) | caches)
        self.timings.codec.append(time.perf_counter() - start)
        for name, cache in caches.items():
            np.copyto(cache, out[f"new_{name}"])
        self.audio.append(out["audio"][:, : count * reference.SAMPLES_PER_FRAME].astype(np.float32))
        self.decoded += count
        self.pending = []
