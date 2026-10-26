"""The Core ML pipeline, reading only what build/chatterbox holds: the Swift contract, in Python."""

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import coremltools as ct
import numpy as np
from tokenizers import Tokenizer

from chatterbox_coreml import assets, s3gen, t3

_PUNCTUATION = (
    ("...", ", "),
    ("…", ", "),
    (":", ","),
    (" - ", ", "),
    (";", ", "),
    ("—", "-"),
    ("–", "-"),
    (" ,", ","),
    ("“", '"'),
    ("”", '"'),
    ("‘", "'"),
    ("’", "'"),
)


def normalize(text: str) -> str:
    """The Space's `punc_norm`."""
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in _PUNCTUATION:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    return text if text.endswith((".", "!", "?", "-", ",")) else text + "."


@dataclass(frozen=True)
class Voice:
    t3_cond: np.ndarray
    prompt_tokens: np.ndarray
    prompt_mel: np.ndarray
    spks: np.ndarray

    @staticmethod
    def load(directory: Path) -> "Voice":
        meta = json.loads((directory / "voice.json").read_text())
        return Voice(
            t3_cond=np.fromfile(directory / "t3_cond.f32", "<f4").reshape(meta["t3_cond"]),
            prompt_tokens=np.fromfile(directory / "prompt_tokens.i32", "<i4"),
            prompt_mel=np.fromfile(directory / "prompt_mel.f32", "<f4").reshape(meta["prompt_mel"]),
            spks=np.fromfile(directory / "spks.f32", "<f4"),
        )


@dataclass(frozen=True)
class Bundle:
    """build/chatterbox: packages, tables, the default voice and config.json."""

    root: Path = assets.BUILD

    @cached_property
    def config(self) -> Mapping[str, object]:
        return json.loads((self.root / "config.json").read_text())

    @cached_property
    def tokenizer(self) -> Tokenizer:
        return Tokenizer.from_file(str(self.root / "tokenizer.json"))

    def table(self, name: str) -> np.ndarray:
        return np.fromfile(self.root / "tables" / f"{name}.f16", "<f2").reshape(-1, t3.HIDDEN)

    @cached_property
    def tables(self) -> Mapping[str, np.ndarray]:
        return {name: self.table(name) for name in ("text_emb", "text_pos", "speech_emb", "speech_pos")}

    @cached_property
    def voice(self) -> Voice:
        return Voice.load(self.root / "voices" / assets.VOICE_NAME)

    @cached_property
    def noise(self) -> np.ndarray:
        return np.fromfile(self.root / "flow_noise.f32", "<f4").reshape(s3gen.MEL_CHANNELS, -1)


@dataclass
class Models:
    """Core ML functions loaded on first use, keyed by (package, function)."""

    bundle: Bundle
    compute_units: ct.ComputeUnit = ct.ComputeUnit.CPU_AND_NE
    loaded: dict[tuple[str, str], ct.models.MLModel] = field(default_factory=dict)

    def __call__(self, package: str, function: str) -> ct.models.MLModel:
        key = (package, function)
        if key not in self.loaded:
            self.loaded[key] = ct.models.MLModel(
                str(self.bundle.root / f"{package}.mlpackage"), compute_units=self.compute_units, function_name=function
            )
        return self.loaded[key]


def predict(model: ct.models.MLModel, inputs: Mapping[str, np.ndarray], state: object | None = None) -> np.ndarray:
    """The model's single output, as float32."""
    _PINNED.append(inputs)
    outputs = model.predict(dict(inputs), state=state) if state is not None else model.predict(dict(inputs))
    return np.asarray(next(iter(outputs.values())), dtype=np.float32)


# coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML drops them later
# on its own queue without the GIL, corrupting the Python heap (apple/coremltools#2829, fixed
# after 9.0). Keeping a reference to every input means that release never frees.
_PINNED: list[Mapping[str, np.ndarray]] = []


@dataclass(frozen=True)
class Timings:
    prefill: float
    decode: float
    flow: float
    vocoder: float

    @property
    def total(self) -> float:
        return self.prefill + self.decode + self.flow + self.vocoder


@dataclass(frozen=True)
class Synthesis:
    tokens: np.ndarray
    wav: np.ndarray
    timings: Timings

    @property
    def seconds(self) -> float:
        return len(self.wav) / 24_000


def text_ids(bundle: Bundle, text: str, language: str) -> list[int]:
    """[START] + tokenizer("[lang]" + text with spaces as [SPACE]) + [STOP]."""
    config = bundle.config["text"]
    encoded = bundle.tokenizer.encode(f"[{language}]" + normalize(text).replace(" ", "[SPACE]")).ids
    return [config["start"], *encoded, config["stop"]]


def prefill_embeds(bundle: Bundle, ids: list[int]) -> np.ndarray:
    """(2, L, 1024): conditioning, text (+ position), then the start-of-speech embedding twice.

    Row 1 is the unconditional guidance row: text embeddings zeroed, text positions kept.
    """
    tables, voice = bundle.tables, bundle.voice
    start = bundle.config["speech"]["start"]
    positions = tables["text_pos"][: len(ids)].astype(np.float32)
    text = tables["text_emb"][ids].astype(np.float32) + positions
    bos = tables["speech_emb"][start].astype(np.float32) + tables["speech_pos"][0].astype(np.float32)
    rows = [np.concatenate([voice.t3_cond, part, bos[None], bos[None]]) for part in (text, positions)]
    return np.stack(rows)


def sample(logits: np.ndarray, generated: list[int], rng: np.random.Generator, config: Mapping[str, float]) -> int:
    """CFG, repetition penalty, temperature, min-p, top-p, then a draw (the Space's order)."""
    scores = (logits[0] + config["cfg_weight"] * (logits[0] - logits[1])).astype(np.float64)
    seen = np.unique(generated)
    scores[seen] = np.where(scores[seen] < 0, scores[seen] * config["repetition_penalty"], scores[seen] / config["repetition_penalty"])
    scores /= config["temperature"]
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    scores[probs < config["min_p"] * probs.max()] = -np.inf
    order = np.argsort(scores)
    ascending = np.exp(scores[order] - scores[order[-1]])
    cumulative = np.cumsum(ascending / ascending.sum())
    remove = cumulative <= 1 - config["top_p"]
    remove[-1] = False
    scores[order[remove]] = -np.inf
    probs = np.exp(scores - scores.max())
    return int(rng.choice(len(probs), p=probs / probs.sum()))


def generate_tokens(models: Models, ids: list[int], rng: np.random.Generator) -> tuple[list[int], float, float]:
    """T3: prefill, then one decode call per token until the stop token. (tokens, prefill s, decode s)"""
    bundle = models.bundle
    config, speech = bundle.config["t3"], bundle.config["speech"]
    embeds = prefill_embeds(bundle, ids)
    length = embeds.shape[1]
    bucket = next(n for n in config["prefill_lengths"] if n >= length)
    start = time.perf_counter()
    padded = np.zeros((2, bucket, t3.HIDDEN), np.float16)
    padded[:, :length] = embeds
    cos, sin = t3.rope(np.arange(bucket))
    last = np.zeros((1, bucket), np.float16)
    last[0, length - 1] = 1
    prefill = models("t3", f"prefill_{bucket}")
    state = prefill.make_state()
    logits = predict(prefill, {"embeds": padded, "cos": cos.astype(np.float16), "sin": sin.astype(np.float16), "last": last}, state)
    prefill_seconds = time.perf_counter() - start

    start = time.perf_counter()
    tables = bundle.tables
    generated = [speech["start"]]
    for step in range(bundle.config["sampling"]["max_new_tokens"]):
        token = sample(logits, generated, rng, bundle.config["sampling"])
        generated.append(token)
        position = length + step
        if token == speech["stop"] or position >= t3.CONTEXT:
            break
        window = next(w for w in config["decode_windows"] if w > position)
        embed = tables["speech_emb"][token].astype(np.float32) + tables["speech_pos"][step + 1].astype(np.float32)
        cos, sin = t3.rope(np.array([position]))
        mask = np.full((1, window), t3.MASKED, np.float16)
        mask[0, : position + 1] = 0
        inputs = {
            "embed": embed[None].astype(np.float16),
            "cos": cos.astype(np.float16),
            "sin": sin.astype(np.float16),
            "position": np.array([position], np.int32),
            "mask": mask,
        }
        logits = predict(models("t3", f"decode_{window}"), inputs, state)
    tokens = [t for t in generated[1:] if t < speech["flow_vocab"]]
    return tokens, prefill_seconds, time.perf_counter() - start


def flow(models: Models, tokens: list[int], steps: int) -> np.ndarray:
    """S3Gen flow: (80, 2·len(tokens)) mel frames for the generated tokens."""
    bundle, voice = models.bundle, models.bundle.voice
    prompt = len(voice.prompt_tokens)
    total = prompt + len(tokens)
    n = next(n for n in bundle.config["flow"]["tokens"] if n >= total)
    frames = 2 * n
    ids = np.zeros((1, n), np.int32)
    ids[0, :prompt] = voice.prompt_tokens
    ids[0, prompt:total] = tokens
    token_mask = np.zeros((1, n), np.float16)
    token_mask[0, :total] = 1
    mu = predict(models("flow_encoder", f"n{n}"), {"tokens": ids, "mask": token_mask}).astype(np.float16)
    cond = np.zeros((1, s3gen.MEL_CHANNELS, frames), np.float16)
    cond[0, :, : voice.prompt_mel.shape[0]] = voice.prompt_mel.T
    mask = np.zeros((1, 1, frames), np.float16)
    mask[..., : 2 * total] = 1
    x = bundle.noise[None, :, :frames].astype(np.float16)
    schedule = s3gen.cosine_schedule(steps)
    step = models("flow_step", f"m{frames}")
    for k in range(steps):
        inputs = {
            "x": x,
            "mu": mu,
            "cond": cond,
            "spks": voice.spks[None].astype(np.float16),
            "mask": mask,
            "t_emb": s3gen.time_embedding(schedule[k]).astype(np.float16),
            "dt": np.array([schedule[k + 1] - schedule[k]], np.float16),
        }
        x = predict(step, inputs).astype(np.float16)
    return x[0, :, 2 * prompt : 2 * total].astype(np.float32)


def chunks(frames: int, size: int, context: int) -> list[tuple[int, int, int]]:
    """(input start, keep start, keep stop) covering [0, frames) with `size`-frame windows.

    Kept frames have `context` frames of real input on each side, or the utterance edge, which
    the models pad exactly as the reference does.
    """
    spans = []
    out = 0
    while out < frames:
        start = min(max(out - context, 0), max(frames - size, 0))
        stop = frames if start + size >= frames else start + size - context
        spans.append((start, out, stop))
        out = stop
    return spans


def _windows(models: Models, package: str, mel: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
    """Run a vocoder package over fixed windows of `mel` (80, T), keeping each window's trusted
    frames; `extra` is the per-sample source for HiFT. Returns (T · samples per input frame)."""
    config = models.bundle.config["vocoder"]
    size, context, hop = config["frames"], config["context"], config["samples_per_frame"]
    frames = mel.shape[1]
    padded = np.zeros((s3gen.MEL_CHANNELS, frames + size), np.float16)
    padded[:, :frames] = mel
    model = models(package, f"t{size}")
    rate = 1 if extra is None else hop
    out = np.zeros(frames * rate, np.float32)
    if extra is not None:
        source = np.zeros((frames + size) * hop, np.float16)
        source[: len(extra)] = extra
    for start, keep, stop in chunks(frames, size, context):
        inputs = {"mel": padded[None, :, start : start + size]}
        if extra is not None:
            inputs["source"] = source[None, None, start * hop : (start + size) * hop]
        window = predict(model, inputs).reshape(-1)
        out[keep * rate : stop * rate] = window[(keep - start) * rate : (stop - start) * rate]
    return out


def f0(models: Models, mel: np.ndarray) -> np.ndarray:
    """Per-frame f0 in Hz, (T,)."""
    return _windows(models, "f0", mel)


def hift(models: Models, mel: np.ndarray, excitation: np.ndarray) -> np.ndarray:
    """Waveform (480·T,) from mel and the harmonic source."""
    return _windows(models, "hift", mel, excitation)


def vocode(models: Models, mel: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """f0, then the harmonic source on the host (float64 phases), then HiFT."""
    config = models.bundle.config["vocoder"]
    excitation = s3gen.source(f0(models, mel), np.array(config["source_weight"]), config["source_bias"], rng)
    return hift(models, mel, excitation)


def finish(audio: np.ndarray, tokens: int) -> np.ndarray:
    """Drop the last token's audio and fade in the first 40 ms, as the reference does."""
    trim = 24_000 // 50
    fade = np.concatenate([np.zeros(trim), (np.cos(np.linspace(np.pi, 0, trim)) + 1) / 2]).astype(np.float32)
    audio = audio.copy()
    audio[: 2 * trim] *= fade[: len(audio[: 2 * trim])]
    return audio[: max(1, tokens - 1) * 960]


def synthesize(models: Models, text: str, language: str, steps: int, seed: int) -> Synthesis:
    rng = np.random.default_rng(seed)
    tokens, prefill_seconds, decode_seconds = generate_tokens(models, text_ids(models.bundle, text, language), rng)
    start = time.perf_counter()
    mel = flow(models, tokens, steps)
    flow_seconds = time.perf_counter() - start
    start = time.perf_counter()
    audio = vocode(models, mel, rng)
    vocoder_seconds = time.perf_counter() - start
    return Synthesis(
        np.array(tokens), finish(audio, len(tokens)), Timings(prefill_seconds, decode_seconds, flow_seconds, vocoder_seconds)
    )
