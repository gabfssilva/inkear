"""T3, the Llama-520M speech-token generator, rebuilt for the Neural Engine.

The batch holds the two classifier-free guidance rows (conditional, unconditional), folded into
the head axis so every tensor stays at rank 4 or less. The KV cache is Core ML state shared by
the prefill and decode functions of one package.
"""

import numpy as np
import torch
from torch import nn

LAYERS = 30
HEADS = 16
HEAD_DIM = 64
HIDDEN = 1024
VOCAB = 8194
ROWS = 2
CONTEXT = 1024
EPS = 1e-5
ROPE_THETA = 500_000.0
MASKED = -3e4

# The residual stream carries an attention-sink outlier near 1.3e3, so mean(x²) overflows fp16.
# RMSNorm is scale invariant; normalising x / 32 keeps the sum of squares in range.
_NORM_SCALE = 1 / 32


def rope_inverse_frequencies() -> np.ndarray:
    """Llama 3 scaled RoPE frequencies (factor 8, low 1, high 4, original context 8192)."""
    inv = 1.0 / ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float64) / HEAD_DIM)
    factor, low, high, original = 8.0, 1.0, 4.0, 8192
    wavelength = 2 * np.pi / inv
    smooth = (original / wavelength - low) / (high - low)
    scaled = np.where(wavelength > original / low, inv / factor, inv)
    medium = (wavelength <= original / low) & (wavelength >= original / high)
    return np.where(medium, (1 - smooth) * inv / factor + smooth * inv, scaled)


def rope(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos and sin, (len(positions), HEAD_DIM), as HF Llama builds them."""
    angles = np.outer(positions.astype(np.float64), rope_inverse_frequencies())
    angles = np.concatenate([angles, angles], axis=-1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x = x * _NORM_SCALE
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS * _NORM_SCALE**2) * weight


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = HEAD_DIM // 2
    return x * cos + torch.cat([-x[..., half:], x[..., :half]], dim=-1) * sin


class _Layer(nn.Module):
    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        attention, mlp = layer.self_attn, layer.mlp
        self.input_norm = nn.Parameter(layer.input_layernorm.weight.detach().clone())
        self.post_norm = nn.Parameter(layer.post_attention_layernorm.weight.detach().clone())
        self.q = _copy(attention.q_proj)
        self.k = _copy(attention.k_proj)
        self.v = _copy(attention.v_proj)
        self.o = _copy(attention.o_proj)
        self.gate = _copy(mlp.gate_proj)
        self.up = _copy(mlp.up_proj)
        self.down = _copy(mlp.down_proj)

    def project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k, v as (ROWS * HEADS, tokens, HEAD_DIM), rotated; q carries the 1/sqrt(d) scale."""
        h = _rms_norm(x, self.input_norm)

        def heads(projection: nn.Linear) -> torch.Tensor:
            tokens = x.shape[1]
            return projection(h).view(ROWS, tokens, HEADS, HEAD_DIM).transpose(1, 2).reshape(
                ROWS * HEADS, tokens, HEAD_DIM
            )

        q = _rotate(heads(self.q), cos, sin) * HEAD_DIM**-0.5
        return q, _rotate(heads(self.k), cos, sin), heads(self.v)

    def finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        tokens = x.shape[1]
        merged = attended.view(ROWS, HEADS, tokens, HEAD_DIM).transpose(1, 2).reshape(ROWS, tokens, HIDDEN)
        x = x + self.o(merged)
        h = _rms_norm(x, self.post_norm)
        return x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))


def _copy(linear: nn.Linear) -> nn.Linear:
    copy = nn.Linear(linear.in_features, linear.out_features, bias=False)
    copy.weight = nn.Parameter(linear.weight.detach().clone())
    return copy


class _Model(nn.Module):
    def __init__(self, t3: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(layer) for layer in t3.tfmr.layers)
        self.norm = nn.Parameter(t3.tfmr.norm.weight.detach().clone())
        self.head = _copy(t3.speech_head)
        shape = (LAYERS, ROWS * HEADS, CONTEXT, HEAD_DIM)
        self.register_buffer("k_cache", torch.zeros(shape, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(shape, dtype=torch.float16))

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(_rms_norm(x, self.norm))


class Prefill(_Model):
    """Runs a padded context of `length` tokens from position 0 and fills the cache with it.

    Inputs: `embeds` (ROWS, length, HIDDEN), `cos`/`sin` (length, HEAD_DIM), `last` (1, length)
    one-hot at the final real token. Output: `logits` (ROWS, VOCAB) at that token. Padding sits
    after the real tokens, so the causal mask alone keeps it out of every real position.
    """

    def __init__(self, t3: nn.Module, length: int) -> None:
        super().__init__(t3)
        self.length = length
        causal = torch.triu(torch.full((length, length), MASKED), diagonal=1)
        self.register_buffer("causal", causal)

    def forward(self, embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, last: torch.Tensor):
        x = embeds
        for index, layer in enumerate(self.layers):
            q, k, v = layer.project(x, cos, sin)
            self.k_cache[index, :, : self.length, :] = k.half()
            self.v_cache[index, :, : self.length, :] = v.half()
            weights = torch.softmax(q @ k.transpose(1, 2) + self.causal, dim=-1)
            x = layer.finish(x, weights @ v)
        return self.logits((last @ x).squeeze(1))


class Decode(_Model):
    """Runs one token at cache slot `position` and returns its logits.

    Attention reads only the first `window` cache slots, so `position` must be below it.
    Inputs: `embed` (1, HIDDEN), shared by both rows; `cos`/`sin` (1, HEAD_DIM); `position` (1,)
    int32; `mask` (1, window), 0 up to and including `position` and MASKED after it.
    Output: `logits` (ROWS, VOCAB).
    """

    def __init__(self, t3: nn.Module, window: int) -> None:
        super().__init__(t3)
        self.window = window

    def forward(
        self, embed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position: torch.Tensor, mask: torch.Tensor
    ):
        x = embed.expand(ROWS, HIDDEN).unsqueeze(1)
        for index, layer in enumerate(self.layers):
            q, k, v = layer.project(x, cos, sin)
            self.k_cache[index : index + 1, :, position : position + 1, :] = k.half().unsqueeze(0)
            self.v_cache[index : index + 1, :, position : position + 1, :] = v.half().unsqueeze(0)
            keys = self.k_cache[index, :, : self.window, :].float()
            values = self.v_cache[index, :, : self.window, :].float()
            weights = torch.softmax(q @ keys.transpose(1, 2) + mask, dim=-1)
            x = layer.finish(x, weights @ values)
        return self.logits(x.squeeze(1))
