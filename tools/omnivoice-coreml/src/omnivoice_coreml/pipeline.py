"""The Core ML pipeline, step by step as the Swift side runs it: front end, unmasking loop, decoder."""

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
from omnivoice.utils.duration import RuleDurationEstimator
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from omnivoice_coreml.modules import CODEBOOKS, MASK_ID, Bucket

SAMPLE_RATE = 24_000
SAMPLES_PER_FRAME = 960
T_SHIFT = 0.1
LAYER_PENALTY = 5.0
POSITION_TEMPERATURE = 5.0

STYLE_START = "<|lang_start|>"
DENOISE = "<|denoise|>"


@dataclass(frozen=True)
class Voice:
    """A fixed reference prompt: codec tokens of a recording, its transcript and loudness."""

    name: str
    tokens: np.ndarray
    text: str
    rms: float

    @classmethod
    def load(cls, path: Path) -> "Voice":
        data = json.loads(path.read_text())
        return cls(
            name=data["name"], tokens=np.array(data["tokens"], np.int32), text=data["text"], rms=data["rms"]
        )


@dataclass(frozen=True)
class Conditioning:
    """The conditional sequence with its target frames still masked."""

    ids: np.ndarray
    audio_mask: np.ndarray
    target: int

    @property
    def length(self) -> int:
        return self.ids.shape[1]


def condition(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    language: str,
    voice: Voice | None = None,
    instruct: str | None = None,
    speed: float = 1.0,
) -> Conditioning:
    """Style, text, reference and masked target, laid out as `OmniVoice._prepare_inference_inputs`."""
    style = (DENOISE if voice else "") + f"{STYLE_START}{language}<|lang_end|>"
    style += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    combined = combine_text(text, voice.text if voice else None)
    ids = tokenizer(style, add_special_tokens=False).input_ids
    ids += tokenizer(f"<|text_start|>{combined}<|text_end|>", add_special_tokens=False).input_ids
    target = estimate_frames(text, voice, speed)
    reference = voice.tokens if voice else np.zeros((CODEBOOKS, 0), np.int32)
    rows = np.concatenate(
        [
            np.tile(np.array(ids, np.int32), (CODEBOOKS, 1)),
            reference,
            np.full((CODEBOOKS, target), MASK_ID, np.int32),
        ],
        axis=1,
    )
    audio_mask = np.zeros(rows.shape[1], np.float32)
    audio_mask[len(ids) :] = 1
    return Conditioning(ids=rows, audio_mask=audio_mask, target=target)


def combine_text(text: str, reference: str | None) -> str:
    combined = f"{reference.strip()} {text.strip()}" if reference else text.strip()
    combined = re.sub(r"[\r\n]+", "", combined).replace("（", "(").replace("）", ")")
    combined = re.sub(r"[ \t]+", " ", combined)
    return re.sub(r"(?<=[一-鿿])\s+|\s+(?=[一-鿿])", "", combined)


def estimate_frames(text: str, voice: Voice | None, speed: float = 1.0) -> int:
    """Target length from character weights, paced by the reference (or 25 frames per "Nice to meet you.")."""
    reference, frames = (voice.text, voice.tokens.shape[1]) if voice else ("Nice to meet you.", 25)
    estimate = _ESTIMATOR.estimate_duration(text, reference, frames)
    return max(1, int(estimate / speed))


_ESTIMATOR = RuleDurationEstimator()


def schedule(frames: int, steps: int) -> list[int]:
    """How many of the `frames`·8 masked codes each step reveals (time-shifted, as upstream)."""
    # In float64; upstream's float32 times move a code between neighbouring steps in ~1% of lengths.
    t = np.linspace(0, 1, steps + 1)
    t = T_SHIFT * t / (1 + (T_SHIFT - 1) * t)
    total, remaining, counts = frames * CODEBOOKS, frames * CODEBOOKS, []
    for step in range(steps):
        count = remaining if step == steps - 1 else min(math.ceil(total * (t[step + 1] - t[step])), remaining)
        counts.append(count)
        remaining -= count
    return counts


@dataclass(frozen=True)
class Layout:
    """A conditioning placed in a bucket: both segments right-aligned, padding in front."""

    bucket: Bucket
    input_ids: np.ndarray
    audio_mask: np.ndarray
    valid: np.ndarray
    target: int

    @classmethod
    def place(cls, conditioning: Conditioning, bucket: Bucket) -> "Layout":
        cond, target, length = bucket.cond, conditioning.target, conditioning.length
        ids = np.full((1, CODEBOOKS, bucket.sequence), MASK_ID, np.int32)
        audio_mask = np.zeros((1, bucket.sequence), np.float16)
        valid = np.zeros((1, bucket.sequence), np.float16)
        ids[0, :, cond - length : cond] = conditioning.ids
        audio_mask[0, cond - length : cond] = conditioning.audio_mask
        valid[0, cond - length : cond] = 1
        audio_mask[0, bucket.sequence - target :] = 1
        valid[0, bucket.sequence - target :] = 1
        return cls(bucket=bucket, input_ids=ids, audio_mask=audio_mask, valid=valid, target=target)

    def write(self, tokens: np.ndarray) -> None:
        """Put the current target codes [8, target] in both segments."""
        cond, end = self.bucket.cond, self.bucket.sequence
        self.input_ids[0, :, cond - self.target : cond] = tokens
        self.input_ids[0, :, end - self.target : end] = tokens

    def inputs(self) -> Mapping[str, np.ndarray]:
        return {"input_ids": self.input_ids, "audio_mask": self.audio_mask, "valid": self.valid}


type StepFunction = Callable[[Mapping[str, np.ndarray]], tuple[np.ndarray, np.ndarray]]


def unmask(
    step: StepFunction,
    layout: Layout,
    steps: int,
    rng: np.random.Generator | None,
    on_step: Callable[[int, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Iterative unmasking; `rng` None picks positions greedily (upstream's position_temperature=0)."""
    frames, window = layout.target, layout.bucket.target
    tokens = np.full((CODEBOOKS, frames), MASK_ID, np.int32)
    penalty = (LAYER_PENALTY * np.arange(CODEBOOKS, dtype=np.float32))[:, None]
    for index, count in enumerate(schedule(frames, steps)):
        if count == 0:
            continue
        predicted, confidence = step(layout.inputs())
        predicted = predicted[0, :, window - frames :]
        scores = confidence[0, :, window - frames :].astype(np.float32) - penalty
        if rng is not None:
            scores = scores / POSITION_TEMPERATURE + _gumbel(rng, scores.shape)
        scores[tokens != MASK_ID] = -np.inf
        chosen = np.argpartition(-scores.ravel(), count - 1)[:count]
        tokens.ravel()[chosen] = predicted.ravel()[chosen]
        layout.write(tokens)
        if on_step:
            on_step(index, tokens)
    return tokens


def _gumbel(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    u = rng.random(shape, dtype=np.float32)
    return -np.log(-np.log(u + 1e-10) + 1e-10)


def postprocess(audio: np.ndarray, voice: Voice | None) -> np.ndarray:
    """Collapse long pauses, trim the edges, match loudness, fade and pad (upstream's postprocess)."""
    audio = trim_silence(audio)
    if voice is None:
        peak = np.abs(audio).max(initial=0.0)
        audio = audio / peak * 0.5 if peak > 1e-6 else audio
    elif voice.rms < 0.1:
        audio = audio * voice.rms / 0.1
    fade = min(int(0.1 * SAMPLE_RATE), len(audio) // 2)
    audio = audio.copy()
    audio[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    audio[len(audio) - fade :] *= np.linspace(1, 0, fade, dtype=np.float32)
    pad = np.zeros(int(0.1 * SAMPLE_RATE), np.float32)
    return np.concatenate([pad, audio, pad])


SILENCE_DBFS = -50.0
CHUNK = SAMPLE_RATE // 100


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """10 ms chunks below -50 dBFS are silent; inner silences over 1 s keep 0.5 s per side, edges keep 0.1 s."""
    chunks = len(audio) // CHUNK
    if chunks == 0:
        return audio
    rms = np.sqrt(np.mean(audio[: chunks * CHUNK].reshape(chunks, CHUNK) ** 2, axis=1))
    loud = 20 * np.log10(np.maximum(rms, 1e-10)) >= SILENCE_DBFS
    if not loud.any():
        return audio
    keep = np.ones(chunks, bool)
    first, last = int(np.argmax(loud)), int(chunks - 1 - np.argmax(loud[::-1]))
    keep[: max(0, first - 10)] = False
    keep[last + 11 :] = False
    run_start = None
    for index in range(first, last + 1):
        if not loud[index]:
            run_start = index if run_start is None else run_start
        elif run_start is not None:
            if index - run_start > 100:
                keep[run_start + 50 : index - 50] = False
            run_start = None
    tail = audio[chunks * CHUNK :] if keep[-1] else audio[:0]
    return np.concatenate([audio[: chunks * CHUNK].reshape(chunks, CHUNK)[keep].ravel(), tail])


@dataclass(frozen=True)
class Speech:
    audio: np.ndarray
    tokens: np.ndarray
    frames: int
    bucket: Bucket

    @property
    def seconds(self) -> float:
        return len(self.audio) / SAMPLE_RATE


class Engine:
    """Loads the converted packages from `directory` and synthesizes with them."""

    def __init__(self, directory: Path, compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE) -> None:
        self.directory = directory
        self.compute_units = compute_units
        self.config = json.loads((directory / "omnivoice.json").read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(directory)
        self.buckets = tuple(Bucket(b["cond"], b["target"]) for b in self.config["step_buckets"])
        self.decoder_frames = tuple(self.config["decoder_frames"])
        self._steps: dict[Bucket, StepFunction] = {}
        self._decoders: dict[int, Callable[[np.ndarray, int], np.ndarray]] = {}

    def voice(self, name: str) -> Voice:
        return Voice.load(self.directory / "voices" / f"{name}.json")

    def bucket(self, conditioning: Conditioning) -> Bucket:
        fits = [b for b in self.buckets if b.cond >= conditioning.length and b.target >= conditioning.target]
        if not fits:
            raise ValueError(f"{conditioning.length} positions / {conditioning.target} frames fit no bucket")
        return min(fits, key=lambda b: b.sequence)

    def step(self, bucket: Bucket) -> StepFunction:
        if bucket in self._steps:
            return self._steps[bucket]
        model = self._model("step", bucket.name)

        def run(inputs: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            pinned = {name: value.copy() for name, value in inputs.items()}
            _PINNED.append(pinned)
            out = model.predict(pinned)
            return out["tokens"], out["scores"]

        self._steps[bucket] = run
        return run

    def decoder(self, frames: int) -> Callable[[np.ndarray, int], np.ndarray]:
        if frames in self._decoders:
            return self._decoders[frames]
        model = self._model("decoder", f"f{frames}")

        def run(tokens: np.ndarray, valid: int) -> np.ndarray:
            codes = np.zeros((1, CODEBOOKS, frames), np.int32)
            codes[0, :, :valid] = tokens[:, :valid]
            mask = (np.arange(frames) < valid).astype(np.float16)[None]
            inputs = {"codes": codes, "frame_mask": mask}
            _PINNED.append(inputs)
            return model.predict(inputs)["audio"][0, : valid * SAMPLES_PER_FRAME]

        self._decoders[frames] = run
        return run

    def decode(self, tokens: np.ndarray) -> np.ndarray:
        frames = tokens.shape[1]
        size = next((f for f in self.decoder_frames if f >= frames), None)
        if size is None:
            raise ValueError(f"{frames} frames exceed the largest decoder ({self.decoder_frames[-1]})")
        return self.decoder(size)(tokens, frames)

    def synthesize(
        self,
        text: str,
        language: str,
        voice: Voice | None = None,
        instruct: str | None = None,
        steps: int = 16,
        seed: int | None = 0,
    ) -> Speech:
        conditioning = condition(self.tokenizer, text, language, voice, instruct)
        bucket = self.bucket(conditioning)
        layout = Layout.place(conditioning, bucket)
        rng = None if seed is None else np.random.default_rng(seed)
        tokens = unmask(self.step(bucket), layout, steps, rng)
        audio = postprocess(self.decode(tokens), voice)
        return Speech(audio=audio, tokens=tokens, frames=conditioning.target, bucket=bucket)

    def _model(self, package: str, function: str) -> ct.models.MLModel:
        return ct.models.MLModel(
            str(self.directory / f"{package}.mlpackage"),
            compute_units=self.compute_units,
            function_name=function,
        )


# coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML releases them later
# on its own queue without the GIL, corrupting the Python heap (apple/coremltools#2829, fixed
# after 9.0). Each prediction gets fresh copies that nothing else touches, kept here for good, so
# the arrays the loop reuses (audio_mask, valid) are never part of that race.
_PINNED: list[Mapping[str, np.ndarray]] = []
