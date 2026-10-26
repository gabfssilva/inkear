"""qwen3-coreml's code predictor for the Neural Engine of Core AI.

Two changes keep it on the Neural Engine and its values right:
- The top-k threshold is found by bisection: MPSGraph runs `torch.topk` on the GPU, which iOS
  denies to apps in the background.
- Keys and values go to fixed 32-slot buffers through one-hot writes, with a mask over the slots
  not written yet. Grown by concatenation, at 2 to 17 slots, they come out wrong from the Neural
  Engine.
"""

import torch
import torch.nn.functional as F
from qwen3_coreml import modules
from qwen3_coreml.modules import (
    CODEBOOK,
    GROUPS,
    HEAD_DIM,
    HIDDEN,
    KV_HEADS,
    PREDICTOR_SCALE,
    PREDICTOR_SCALED_FROM,
    Predictor,
    _tokens_last,
    rotary,
)
from torch import Tensor

# Halving steps for the threshold: they narrow it to 1/65536 of the logits' range.
_STEPS = 16
_HIDE = -1e4
_SLOTS = 32


class NeuralEnginePredictor(Predictor):
    """`Predictor`, with the same inputs and outputs."""

    def __init__(self, talker: torch.nn.Module, temperature: float, top_k: int) -> None:
        super().__init__(talker, temperature, top_k)
        slots = torch.arange(_SLOTS)
        positions = torch.arange(GROUPS + 2)[:, None]
        self.register_buffer("slot", (slots[None] == positions).to(torch.float32))
        self.register_buffer("hidden", torch.where(slots[None] <= positions, 0.0, _HIDE))

    def forward(self, hidden_states: Tensor, code_embeds: Tensor, noise: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat((hidden_states, code_embeds), dim=-1)
        rotate = rotary(self.cos[..., :2], self.sin[..., :2])
        keys: list[Tensor] = []
        values: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            if index == PREDICTOR_SCALED_FROM:
                x = x * (1 / PREDICTOR_SCALE)
            q, k, v = layer.qkv(x, rotate)
            v = _tokens_last(v)
            x = layer.feed_forward(layer.attend(x, q, k, v, self.causal))
            keys.append(F.pad(k, (0, _SLOTS - 2)))
            values.append(F.pad(v, (0, 0, 0, _SLOTS - 2)))
        x = x[..., 1:]
        codes, embeds = [], []
        for group in range(GROUPS):
            hot = self._sample(
                F.conv2d(modules.unit(x, 1), self.heads[group]).view(1, CODEBOOK), noise[group : group + 1]
            )
            codes.append(torch.matmul(hot, self.index))
            embed = torch.matmul(hot, self.embeddings[group]).view(1, HIDDEN, 1, 1)
            embeds.append(embed)
            if group == GROUPS - 1:
                break
            position = group + 2
            rotate = rotary(self.cos[..., position : position + 1], self.sin[..., position : position + 1])
            slot = self.slot[position]
            mask = self.hidden[position].view(1, 1, 1, _SLOTS)
            x = embed
            for index, layer in enumerate(self.layers):
                if index == PREDICTOR_SCALED_FROM:
                    x = x * (1 / PREDICTOR_SCALE)
                q, k, v = layer.qkv(x, rotate)
                keys[index] = keys[index] + k * slot.view(1, 1, 1, _SLOTS)
                values[index] = values[index] + v.view(1, KV_HEADS, 1, HEAD_DIM) * slot.view(1, 1, _SLOTS, 1)
                x = layer.feed_forward(layer.attend(x, q, keys[index], values[index], mask))
        return torch.cat(codes, dim=1), torch.stack(embeds).sum(0)

    def _sample(self, logits: Tensor, noise: Tensor) -> Tensor:
        """`Predictor._sample` with the top-k threshold found by bisection: the largest value that
        at least `top_k` logits reach, to within 1/65536 of the logits' range."""
        low = logits.amin(-1, keepdim=True)
        high = logits.amax(-1, keepdim=True)
        for _ in range(_STEPS):
            middle = (low + high) / 2
            enough = ((logits >= middle).to(logits.dtype).sum(-1, keepdim=True) >= self.top_k).to(logits.dtype)
            low = low + enough * (middle - low)
            high = middle + enough * (high - middle)
        allowed = (logits >= low).to(logits.dtype)
        scores = logits * self.inverse_temperature + noise + (allowed - 1) * -_HIDE
        hot = (scores >= scores.max(-1, keepdim=True).values).to(logits.dtype)
        ranked = hot * self.rank
        return (ranked >= ranked.max(-1, keepdim=True).values).to(logits.dtype)
