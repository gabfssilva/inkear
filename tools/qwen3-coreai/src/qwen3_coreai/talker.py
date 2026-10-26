"""qwen3-coreml's talker step for Core AI, with the KV cache as an input the caller keeps.

The Neural Engine compiler of Core AI crashes on a model that writes into a state at a position
known only at run time, so the step takes the cache, attends to the token's own key and value
apart, and returns them for the caller to write at the token's slot.
"""

import torch
from qwen3_coreml import modules
from qwen3_coreml.modules import (
    _HIDDEN_RMS,
    HEAD_DIM,
    KV_HEADS,
    KV_WIDTH,
    Q_HEADS,
    _conv,
    _layers,
    rope_table,
    rotary,
)
from torch import Tensor, nn
from torch.nn import functional as F


class Decode(nn.Module):
    """One talker step over a cache of `slots` slots.

    Inputs: `input_embeds [1, 1024, 1, 1]`; `key_padding_mask [1, slots]`, 0 for the slots before
    the step's and -1e4 from it on; `position [1, slots]`, one-hot at the step's slot, which is
    also its rotary position; `key_cache`/`value_cache [1, 28672, 1, slots]`, channel
    `layer * 1024 + head * 128 + dim`. Outputs: `logits [1, 1, 3072]`, `hidden_states
    [1, 1024, 1, 1]`, and the step's `key`/`value [1, 28672, 1, 1]`.
    """

    def __init__(self, talker: nn.Module, slots: int) -> None:
        super().__init__()
        self.layers = _layers(talker.model.layers)
        norm = talker.model.norm.weight.detach().float()
        self.head = nn.Parameter(_conv(talker.codec_head, norm, _HIDDEN_RMS))
        self.register_buffer("norm", norm[None, :, None, None] * _HIDDEN_RMS)
        cos, sin = rope_table(slots, talker.config.rope_theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.slots = slots

    def forward(
        self,
        input_embeds: Tensor,
        key_padding_mask: Tensor,
        position: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        slots = self.slots
        cos = torch.matmul(position, self.cos).view(1, 1, HEAD_DIM, 1)
        sin = torch.matmul(position, self.sin).view(1, 1, HEAD_DIM, 1)
        rotate = rotary(cos, sin)
        mask = key_padding_mask.view(1, 1, 1, slots)
        x = input_embeds
        keys, values = [], []
        for index, layer in enumerate(self.layers):
            channels = slice(index * KV_WIDTH, (index + 1) * KV_WIDTH)
            q, k, v = layer.qkv(x, rotate)
            # Query heads 2h and 2h+1 share key head h.
            queries = q.view(1, KV_HEADS, 2, HEAD_DIM)
            cached = torch.matmul(queries, key_cache[:, channels].view(1, KV_HEADS, HEAD_DIM, slots)) + mask
            own = torch.matmul(queries, k)
            weights = torch.cat([cached, own], dim=-1).softmax(-1)
            cached_values = value_cache[:, channels].view(1, KV_HEADS, HEAD_DIM, slots).transpose(2, 3)
            attended = torch.matmul(weights[..., :slots], cached_values) + weights[..., slots:] * v.view(
                1, KV_HEADS, 1, HEAD_DIM
            )
            x = x + F.conv2d(attended.reshape(1, Q_HEADS * HEAD_DIM, 1, 1), layer.o)
            x = layer.feed_forward(x)
            keys.append(k.view(1, KV_WIDTH, 1, 1))
            values.append(v.view(1, KV_WIDTH, 1, 1))
        # Looked up when it runs, so that qwen3_coreai.norms can replace it.
        normalized = modules.unit(x, 1)
        logits = F.conv2d(normalized, self.head).view(1, 1, -1)
        return logits, normalized * self.norm, torch.cat(keys, dim=1), torch.cat(values, dim=1)
