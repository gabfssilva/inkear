"""The generation loop the Swift side reimplements, over interchangeable stage backends."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from magpie_coreml.checkpoint import AUDIO_EOS, CODEBOOKS, FORBIDDEN, STACK
from magpie_coreml.modules import CONTEXT, TEXT

MAX_STEPS = 250  # max_decoder_steps 500 frames / frame stacking 2
MIN_FRAMES = 4  # EOS is forbidden while step * 2 < 4
PRIOR_EPSILON = 0.1
LOOKAHEAD = 6
SINK = 4  # attention_sink_threshold


@dataclass(frozen=True)
class Quality:
    """Sampling settings. None of them changes the work per frame: both CFG rows always run."""

    cfg_scale: float
    temperature: float
    attention_prior: bool


DEFAULT = Quality(cfg_scale=2.5, temperature=0.6, attention_prior=True)  # NeMo's inference_parameters


class Stages(Protocol):
    def encode(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray: ...

    def prefill(self, speaker: int, encoder_output: np.ndarray, mask: np.ndarray) -> None:
        """Run the context prefill and seed the step state with its caches."""

    def step(
        self, audio_emb: np.ndarray, position: int, mask: np.ndarray, prior: np.ndarray, cfg_scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> logits (16, 2024), h_last (2, 1, 768), alignment (256,)."""

    def sample(
        self, h_last: np.ndarray, gumbel: np.ndarray, temperature: float, cfg_scale: float, forbid_eos: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """-> codes (16,), audio_emb (1, 1, 768)."""

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """(frames, 8) codec codes -> (frames * 1024,) audio at 22.05 kHz."""


@dataclass
class Alignment:
    """Tracks the attended text position and builds the next cross-attention prior.

    Mirrors MagpieTTSModel.get_most_attended_text_timestep and construct_inference_prior for a
    single chunk, with the v2607 inference parameters.
    """

    text_length: int
    last: int = 1
    counts: dict[int, int] = field(default_factory=dict)

    def update(self, scores: np.ndarray) -> np.ndarray:
        start = self.last + 1 if self.counts.get(self.last, 0) >= SINK else self.last
        window = scores[start : min(start + LOOKAHEAD, self.text_length - 3)]
        attended = start + int(np.argmax(window)) if window.size else self.text_length - 1
        self.counts[attended] = self.counts.get(attended, 0) + 1
        self.last = attended
        prior = np.full(TEXT, PRIOR_EPSILON, np.float32)
        if self.text_length <= 5:
            prior[:] = 1
        else:
            prior[max(1, attended - 1)] = 1
            prior[attended] = 1
            for offset in range(1, LOOKAHEAD + 1):
                prior[min(attended + offset, self.text_length - 1)] = 1
        for position, count in self.counts.items():
            if count >= SINK:
                prior[: position + 1] = PRIOR_EPSILON
        return prior


def synthesize(
    stages: Stages,
    ids: list[int],
    speaker: int,
    quality: Quality,
    bos_emb: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    return stages.decode(generate(stages, ids, speaker, quality, bos_emb, rng).codes)


@dataclass(frozen=True)
class Generation:
    codes: np.ndarray  # (frames, 8) int32
    steps: int
    eos: bool


def generate(
    stages: Stages,
    ids: list[int],
    speaker: int,
    quality: Quality,
    bos_emb: np.ndarray,
    rng: np.random.Generator,
) -> Generation:
    """Autoregressive generation of codec frames for one text chunk."""
    if len(ids) > TEXT:
        raise ValueError(f"{len(ids)} text tokens exceed the {TEXT}-token encoder")
    tokens = np.zeros((1, TEXT), np.int32)
    tokens[0, : len(ids)] = ids
    mask = np.zeros((1, TEXT), np.float32)
    mask[0, : len(ids)] = 1
    stages.prefill(speaker, stages.encode(tokens, mask), mask)
    alignment = Alignment(text_length=len(ids))
    prior = np.ones(TEXT, np.float32)
    audio_emb = bos_emb
    frames: list[np.ndarray] = []
    for step in range(MAX_STEPS):
        forbid_eos = step * STACK < MIN_FRAMES
        logits, h_last, scores = stages.step(audio_emb, CONTEXT + step, mask, prior, quality.cfg_scale)
        if quality.attention_prior:
            prior = alignment.update(scores)
        gumbel = (
            rng.gumbel(size=(16, 2024)).astype(np.float32)
            if quality.temperature > 0
            else np.zeros((16, 2024), np.float32)
        )
        codes, audio_emb = stages.sample(
            h_last, gumbel, max(quality.temperature, 1.0e-3), quality.cfg_scale, forbid_eos
        )
        greedy = np.argmax(logits + _logit_bias(forbid_eos), axis=-1)
        stacked = codes.reshape(STACK, CODEBOOKS)
        ended = [
            i
            for i in range(STACK)
            if (stacked[i] == AUDIO_EOS).any() or (greedy.reshape(STACK, CODEBOOKS)[i] == AUDIO_EOS).any()
        ]
        keep = stacked[: ended[0]] if ended else stacked
        frames.extend(keep)
        if ended:
            return Generation(
                codes=np.array(frames, np.int32).reshape(-1, CODEBOOKS), steps=step + 1, eos=True
            )
    return Generation(codes=np.array(frames, np.int32).reshape(-1, CODEBOOKS), steps=MAX_STEPS, eos=False)


def _logit_bias(forbid_eos: bool) -> np.ndarray:
    bias = np.zeros(2024, np.float32)
    bias[list(FORBIDDEN)] = -np.inf
    if forbid_eos:
        bias[AUDIO_EOS] = -np.inf
    return bias


def latents(codes: np.ndarray) -> np.ndarray:
    """FSQ dequantization: (frames, 8) codes -> (1, 32, frames) codec latents.

    Each code splits into 4 digits with levels (8, 7, 6, 6), each mapped to (d - L // 2) / (L // 2).
    """
    levels = np.array([8, 7, 6, 6])
    base = np.array([1, 8, 56, 336])
    digits = (codes[..., None] // base) % levels  # (frames, 8, 4)
    half = levels // 2
    return ((digits - half) / half).reshape(len(codes), -1).T[None].astype(np.float32)
