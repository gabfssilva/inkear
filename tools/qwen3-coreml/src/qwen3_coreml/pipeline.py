"""The Swift contract run from Python: text front end, generation loop and speech decoding.

The loop runs over `Stages`, implemented by the Core ML packages (`CoreMLStages`) and by the
PyTorch reference modules (`TorchStages`). Both draw from the same random numbers, so their codes
agree until numerical differences flip a draw.
"""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import coremltools as ct
import numpy as np
import torch
from qwen_tts import Qwen3TTSModel
from transformers import Qwen2TokenizerFast
from transformers.cache_utils import DynamicCache

_HIDE = -1e4


@dataclass(frozen=True)
class Config:
    """The parts of `config.json` the generation loop reads."""

    model: str
    default_voices: Mapping[str, str]
    slots: int
    prefill_tokens: int
    speech_frames: int
    samples_per_frame: int
    text_tokens: Mapping[str, int]
    codec_tokens: Mapping[str, int]
    languages: Mapping[str, int]
    voices: Mapping[str, Mapping[str, object]]
    temperature: float
    top_k: int
    repetition_penalty: float
    suppressed_from: int
    min_frames: int

    @staticmethod
    def read(directory: Path) -> "Config":
        raw = json.loads((directory / "config.json").read_text())
        talker = raw["sampling"]["talker"]
        return Config(
            model=raw["model"],
            default_voices=raw["default_voices"],
            slots=raw["slots"],
            prefill_tokens=raw["prefill_tokens"],
            speech_frames=raw["speech_frames"],
            samples_per_frame=raw["samples_per_frame"],
            text_tokens=raw["text_tokens"],
            codec_tokens=raw["codec_tokens"],
            languages=raw["languages"],
            voices=raw["voices"],
            temperature=talker["temperature"],
            top_k=talker["top_k"],
            repetition_penalty=talker["repetition_penalty"],
            suppressed_from=talker["suppressed_from"],
            min_frames=talker["min_frames"],
        )


@dataclass
class Timings:
    """Wall-clock seconds of every call, by stage."""

    calls: dict[str, list[float]] = field(default_factory=dict)

    def add(self, stage: str, seconds: float) -> None:
        self.calls.setdefault(stage, []).append(seconds)

    def total(self) -> float:
        return sum(sum(values) for values in self.calls.values())


class Stages(Protocol):
    """The models of the contract. Vectors are float32 `[1024]`; `hidden` is the talker's output."""

    timings: Timings

    def text(self, token: int) -> np.ndarray: ...
    def code(self, token: int) -> np.ndarray: ...
    def prefill(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def step(self, embed: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]: ...
    def predict(
        self, hidden: np.ndarray, first: np.ndarray, noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def speech(self, codes: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Assets:
    """A build directory's config, tokenizer, control embeddings and voices."""

    directory: Path
    config: Config
    tokenizer: Qwen2TokenizerFast
    tts_pad: np.ndarray
    tts_bos: np.ndarray
    tts_eos: np.ndarray

    @staticmethod
    def read(directory: Path) -> "Assets":
        return Assets(
            directory=directory,
            config=Config.read(directory),
            tokenizer=Qwen2TokenizerFast(
                vocab_file=str(directory / "vocab.json"), merges_file=str(directory / "merges.txt")
            ),
            tts_pad=np.load(directory / "tts_pad_embed.npy"),
            tts_bos=np.load(directory / "tts_bos_embed.npy"),
            tts_eos=np.load(directory / "tts_eos_embed.npy"),
        )

    def speaker(self, voice: str) -> np.ndarray:
        """The voice's x-vector."""
        return np.load(self.directory / str(self.config.voices[voice]["embedding"])).astype(np.float32)

    def prompt(self, text: str, language: str, voice: str, stages: Stages) -> np.ndarray:
        """The talker's prompt, `[tokens, 1024]`, in the checkpoints' non-streaming layout."""
        tokens, codec = self.config.text_tokens, self.config.codec_tokens
        rows = [stages.text(token) for token in (tokens["im_start"], tokens["assistant"], tokens["newline"])]
        for token in (
            codec["think"],
            codec["think_bos"],
            self.config.languages[language],
            codec["think_eos"],
        ):
            rows.append(self.tts_pad + stages.code(token))
        rows.append(self.tts_pad + self.speaker(voice))
        codec_pad = stages.code(codec["pad"])
        rows.append(self.tts_bos + codec_pad)
        rows.extend(stages.text(token) + codec_pad for token in self.tokenizer(text)["input_ids"])
        rows.append(self.tts_eos + codec_pad)
        rows.append(self.tts_pad + stages.code(codec["bos"]))
        return np.stack(rows)


@dataclass(frozen=True)
class Speech:
    audio: np.ndarray
    codes: np.ndarray  # [16, frames]
    finished: bool
    timings: Timings

    @property
    def seconds(self) -> float:
        return len(self.audio) / 24_000


def first_code(config: Config, logits: np.ndarray, history: list[int], rng: np.random.Generator) -> int:
    """The frame's first code: repetition penalty over earlier codes, control tokens masked (the end
    token too before `min_frames` frames), temperature, then a draw among the top k."""
    logits = logits.astype(np.float64)
    seen = np.unique(np.array(history, dtype=np.int64))
    penalty = config.repetition_penalty
    logits[seen] = np.where(logits[seen] > 0, logits[seen] / penalty, logits[seen] * penalty)
    suppressed = np.arange(config.suppressed_from, len(logits))
    if len(history) >= config.min_frames:
        suppressed = suppressed[suppressed != config.codec_tokens["eos"]]
    logits[suppressed] = -np.inf
    logits /= config.temperature
    top = np.argpartition(logits, -config.top_k)[-config.top_k :]
    weights = np.exp(logits[top] - logits[top].max())
    return int(top[rng.choice(len(top), p=weights / weights.sum())])


def synthesize(
    assets: Assets, stages: Stages, text: str, language: str, voice: str, rng: np.random.Generator
) -> Speech:
    config = assets.config
    embeds = assets.prompt(text, language, voice, stages)
    length = len(embeds)
    if length > config.prefill_tokens:
        raise ValueError(f"{length} prompt tokens exceed the prefill's {config.prefill_tokens}")
    logits, hidden = stages.prefill(embeds)
    limit = min(config.speech_frames, config.slots - length)
    frames: list[np.ndarray] = []
    history: list[int] = []
    finished = False
    for index in range(limit):
        token = first_code(config, logits, history, rng)
        if token == config.codec_tokens["eos"]:
            finished = True
            break
        history.append(token)
        first = stages.code(token)
        codes, embeds_sum = stages.predict(hidden, first, rng.gumbel(size=(15, 2048)).astype(np.float32))
        frames.append(np.concatenate(([token], codes)))
        if index + 1 == limit:
            break
        logits, hidden = stages.step(first + embeds_sum + assets.tts_pad, length + index)
    codes = np.array(frames, dtype=np.int64).T.reshape(16, -1)
    return Speech(audio=stages.speech(codes), codes=codes, finished=finished, timings=stages.timings)


class CoreMLStages:
    """The packages of a build directory, loaded for `compute_units`.

    coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML drops them later on
    its own queue without the GIL, corrupting the Python heap (apple/coremltools#2829). Every input
    is kept alive, untouched, in `pinned`.
    """

    def __init__(self, directory: Path, compute_units: ct.ComputeUnit) -> None:
        def load(name: str, function: str | None = None) -> ct.models.MLModel:
            path = str(directory / f"{name}.mlpackage")
            return ct.models.MLModel(path, compute_units=compute_units, function_name=function)

        self.config = Config.read(directory)
        self.text_model = load("TextProjector")
        self.code_model = load("CodeEmbedder")
        self.prefill_model = load("CodeDecoder", "prefill")
        self.decode_model = load("CodeDecoder", "decode")
        self.predictor_model = load("MultiCodeDecoder")
        self.speech_model = load("SpeechDecoder")
        self.pinned: list[Mapping[str, np.ndarray]] = []
        self.timings = Timings()
        self.state = self.decode_model.make_state()

    def reset(self) -> None:
        self.timings = Timings()
        self.state = self.decode_model.make_state()

    def _run(
        self, stage: str, model: ct.models.MLModel, inputs: Mapping[str, np.ndarray], stateful: bool = False
    ) -> Mapping[str, np.ndarray]:
        pinned = {name: np.ascontiguousarray(value) for name, value in inputs.items()}
        self.pinned.append(pinned)
        start = time.perf_counter()
        out = model.predict(pinned, state=self.state) if stateful else model.predict(pinned)
        self.timings.add(stage, time.perf_counter() - start)
        return out

    def _lookup(self, model: ct.models.MLModel, token: int) -> np.ndarray:
        out = self._run("lookup", model, {"input_ids": np.array([token], dtype=np.int32)})
        return out["input_embeds"].astype(np.float32).reshape(-1)

    def text(self, token: int) -> np.ndarray:
        return self._lookup(self.text_model, token)

    def code(self, token: int) -> np.ndarray:
        return self._lookup(self.code_model, token)

    def prefill(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        size, length = self.config.prefill_tokens, len(embeds)
        padded = np.zeros((1, 1024, 1, size), dtype=np.float16)
        padded[0, :, 0, :length] = embeds.T
        positions = np.arange(size)
        causal = np.where(positions[None, :] <= positions[:, None], 0, _HIDE).astype(np.float16)
        out = self._run("prefill", self.prefill_model, {"input_embeds": padded, "key_padding_mask": causal})
        for name in ("key_cache", "value_cache"):
            # coremltools writes states from float32 arrays only; the state itself is fp16.
            cache = np.zeros((1, out[name].shape[1], 1, self.config.slots), dtype=np.float32)
            cache[..., :length] = out[name][..., :length]
            self.state.write_state(name, cache)
        hidden = out["hidden_states"][0, :, 0, length - 1]
        return out["logits"][0, length - 1].astype(np.float32), hidden.astype(np.float32)

    def step(self, embed: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        slots = np.arange(self.config.slots)
        inputs = {
            "input_embeds": embed.astype(np.float16).reshape(1, 1024, 1, 1),
            "key_padding_mask": np.where(slots <= position, 0, _HIDE).astype(np.float16)[None],
            "kv_cache_update_mask": (slots == position).astype(np.float16)[None],
        }
        out = self._run("decode", self.decode_model, inputs, stateful=True)
        return out["logits"].astype(np.float32).reshape(-1), out["hidden_states"].astype(np.float32).reshape(
            -1
        )

    def predict(
        self, hidden: np.ndarray, first: np.ndarray, noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs = {
            "hidden_states": hidden.astype(np.float16).reshape(1, 1024, 1, 1),
            "code_embeds": first.astype(np.float16).reshape(1, 1024, 1, 1),
            "noise": noise.astype(np.float16),
        }
        out = self._run("predictor", self.predictor_model, inputs)
        codes = out["codes"].reshape(-1).round().astype(np.int64)
        return codes, out["codes_embeds"].astype(np.float32).reshape(-1)

    def speech(self, codes: np.ndarray) -> np.ndarray:
        capacity, count = self.config.speech_frames, codes.shape[1]
        if count > capacity:
            raise ValueError(f"{count} frames exceed the SpeechDecoder's {capacity}")
        padded = np.zeros((1, 16, capacity), dtype=np.int32)
        padded[0, :, :count] = codes
        audio = self._run("speech", self.speech_model, {"audio_codes": padded})["audio"].reshape(-1)
        return audio[: count * self.config.samples_per_frame].astype(np.float32)


class TorchStages:
    """The same stages computed by the reference's own modules, fp32 on the CPU.

    The predictor draws with the Gumbel-max rule of the MultiCodeDecoder: the argmax over the top-k
    logits of `logits / temperature + noise`.
    """

    def __init__(self, model: Qwen3TTSModel, temperature: float, top_k: int) -> None:
        self.talker = model.model.talker
        self.decoder = model.model.speech_tokenizer.model.decoder
        self.temperature = temperature
        self.top_k = top_k
        self.cache = DynamicCache()
        self.timings = Timings()

    def reset(self) -> None:
        self.cache = DynamicCache()
        self.timings = Timings()

    @torch.inference_mode()
    def text(self, token: int) -> np.ndarray:
        embedding = self.talker.model.text_embedding(torch.tensor([token]))
        return self.talker.text_projection(embedding)[0].numpy()

    @torch.inference_mode()
    def code(self, token: int) -> np.ndarray:
        return self.talker.model.codec_embedding(torch.tensor([token]))[0].numpy()

    @torch.inference_mode()
    def _talk(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = self.talker.model(
            inputs_embeds=torch.from_numpy(embeds)[None], past_key_values=self.cache, use_cache=True
        )
        hidden = out.last_hidden_state[0, -1]
        return self.talker.codec_head(hidden).numpy(), hidden.numpy()

    def prefill(self, embeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.cache = DynamicCache()
        return self._talk(embeds.astype(np.float32))

    def step(self, embed: np.ndarray, position: int) -> tuple[np.ndarray, np.ndarray]:
        return self._talk(embed.astype(np.float32)[None])

    @torch.inference_mode()
    def predict(
        self, hidden: np.ndarray, first: np.ndarray, noise: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        predictor = self.talker.code_predictor
        cache = DynamicCache()
        inputs = torch.from_numpy(np.stack((hidden, first)).astype(np.float32))[None]
        codes: list[int] = []
        total = torch.zeros(1024)
        for group in range(15):
            state = predictor.model(
                inputs_embeds=inputs, past_key_values=cache, use_cache=True
            ).last_hidden_state
            logits = predictor.lm_head[group](state[0, -1]).double()
            threshold = torch.topk(logits, self.top_k).values[-1]
            scores = logits / self.temperature + torch.from_numpy(noise[group]).double()
            code = int(torch.where(logits >= threshold, scores, -torch.inf).argmax())
            codes.append(code)
            embed = predictor.model.codec_embedding[group](torch.tensor([code]))
            total += embed[0]
            inputs = embed[None]
        return np.array(codes), total.numpy()

    @torch.inference_mode()
    def frame_embedding(self, frame: np.ndarray) -> np.ndarray:
        """The talker's input for a frame `[16]`: the sum of its codes' embeddings."""
        tables = self.talker.code_predictor.model.codec_embedding
        total = self.talker.model.codec_embedding(torch.tensor([int(frame[0])]))[0]
        for group, code in enumerate(frame[1:]):
            total = total + tables[group](torch.tensor([int(code)]))[0]
        return total.numpy()

    @torch.inference_mode()
    def speech(self, codes: np.ndarray) -> np.ndarray:
        return self.decoder(torch.from_numpy(codes)[None]).reshape(-1).numpy()
