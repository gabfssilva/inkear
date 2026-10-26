"""T3 for Core AI: chatterbox-coreml's T3 rearranged for the Neural Engine compiler of Core AI.

The compiler refuses two things chatterbox-coreml's Core ML graph does:
- RoPE on heads folded into the batch axis, (ROWS * HEADS, tokens, HEAD_DIM) ("The strides of the
  Reshape is not valid"). q, k and v stay (ROWS, HEADS, tokens, HEAD_DIM).
- A KV cache kept as state: it crashes on any write into a state at a position known only at run
  time. The cache is an input instead, which the caller keeps: prefill returns the keys and values
  of its positions, and decode those of its token, for the caller to write at its position.

Caches hold the two rows of layer i at [2i, 2i + 2): keys (LAYERS * ROWS, HEADS, HEAD_DIM, slots),
transposed for the attention product, and values (LAYERS * ROWS, HEADS, slots, HEAD_DIM).
"""

import torch
from chatterbox_coreml import t3
from chatterbox_coreml.t3 import HEAD_DIM, HEADS, HIDDEN, LAYERS, MASKED, ROWS
from torch import nn


def key_shape(slots: int) -> tuple[int, int, int, int]:
    return (LAYERS * ROWS, HEADS, HEAD_DIM, slots)


def value_shape(slots: int) -> tuple[int, int, int, int]:
    return (LAYERS * ROWS, HEADS, slots, HEAD_DIM)


class _Layer(t3._Layer):
    def project(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """q, k, v as (ROWS, HEADS, tokens, HEAD_DIM), rotated; q carries the 1/sqrt(d) scale."""
        h = t3._rms_norm(x, self.input_norm)
        tokens = x.shape[1]

        def heads(projection: nn.Linear) -> torch.Tensor:
            return projection(h).view(ROWS, tokens, HEADS, HEAD_DIM).transpose(1, 2)

        q = t3._rotate(heads(self.q), cos, sin) * HEAD_DIM**-0.5
        return q, t3._rotate(heads(self.k), cos, sin), heads(self.v)

    def finish(self, x: torch.Tensor, attended: torch.Tensor) -> torch.Tensor:
        merged = attended.transpose(1, 2).reshape(ROWS, x.shape[1], HIDDEN)
        x = x + self.o(merged)
        h = t3._rms_norm(x, self.post_norm)
        return x + self.down(nn.functional.silu(self.gate(h)) * self.up(h))


class Weights(nn.Module):
    """The transformer and the speech head, which every function of the package shares."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer(layer) for layer in model.tfmr.layers)
        self.norm = nn.Parameter(model.tfmr.norm.weight.detach().clone())
        self.head = t3._copy(model.speech_head)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(t3._rms_norm(x, self.norm))


class Prefill(nn.Module):
    """chatterbox-coreml's `t3.Prefill`, a padded context of `length` tokens from position 0, that
    also returns the keys and values of every position: `keys` key_shape(length) and `values`
    value_shape(length)."""

    def __init__(self, weights: Weights, length: int) -> None:
        super().__init__()
        self.weights = weights
        self.register_buffer("causal", torch.triu(torch.full((length, length), MASKED), diagonal=1))

    def forward(self, embeds: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, last: torch.Tensor):
        x = embeds
        keys, values = [], []
        for layer in self.weights.layers:
            q, k, v = layer.project(x, cos, sin)
            weights = torch.softmax(q @ k.transpose(2, 3) + self.causal, dim=-1)
            x = layer.finish(x, weights @ v)
            keys.append(k.transpose(2, 3))
            values.append(v)
        # As a product with `last`, the selection runs on the GPU.
        final = (x * last.view(1, -1, 1)).sum(1)
        return self.weights.logits(final), torch.cat(keys), torch.cat(values)


class Decode(nn.Module):
    """chatterbox-coreml's `t3.Decode` over a cache of `window` slots, given as inputs.

    Inputs: `embed` (1, HIDDEN); `cos`/`sin` (1, HEAD_DIM) at the token's position; `mask`
    (1, window), 0 before the token's position and MASKED from it on; `key_cache`
    key_shape(window) and `value_cache` value_shape(window). Outputs: `logits` (ROWS, VOCAB) and
    the token's `key` key_shape(1) and `value` value_shape(1), for the caller to write at its
    position. Each layer attends to the cache and to the token's own key and value apart.
    """

    def __init__(self, weights: Weights, window: int) -> None:
        super().__init__()
        self.weights = weights
        self.window = window

    def forward(
        self,
        embed: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ):
        x = embed.expand(ROWS, HIDDEN).unsqueeze(1)
        keys, values = [], []
        for index, layer in enumerate(self.weights.layers):
            q, k, v = layer.project(x, cos, sin)
            rows = slice(ROWS * index, ROWS * index + ROWS)
            scores = torch.cat([q @ key_cache[rows] + mask, (q * k).sum(-1, keepdim=True)], dim=-1)
            weights = torch.softmax(scores, dim=-1)
            x = layer.finish(x, weights[..., : self.window] @ value_cache[rows] + weights[..., self.window :] * v)
            keys.append(k.transpose(2, 3))
            values.append(v)
        return self.weights.logits(x.squeeze(1)), torch.cat(keys), torch.cat(values)
