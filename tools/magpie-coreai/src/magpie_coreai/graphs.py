"""magpie-coreml's stages rearranged for the Neural Engine of Core AI.

MPSGraph puts what the Neural Engine does not take on the GPU, which iOS denies to apps in the
background:
- Lookups by integer index run on the GPU, so the app reads the text embeddings, the speakers'
  context and the decoder's position embeddings from tables (`tables`).
- The Neural Engine compiler of Core AI crashes on writes into a state at a position known only
  at run time, so the decoder step takes its self-attention cache as inputs and returns the
  step's keys and values for the app to write (`DecoderStep`).
- The Neural Engine compiler fails on the prefill's 217 positions, so the prefill takes the
  context padded to PREFILL positions (`DecoderPrefill`).
- `torch.topk` runs on the GPU, so the local transformer finds the top-k threshold by bisection
  and the drawn code as the largest score (`LocalTransformer`).

Caches stack the two rows (conditional, unconditional) of layer i at [2i, 2i + 2): keys
(24, 12, 64, positions), transposed for the attention product, and values (24, 12, positions, 64).
"""

import torch
from magpie_coreml import modules
from magpie_coreml.checkpoint import CODES, VOCAB
from magpie_coreml.modules import (
    ALIGNMENT_LAYERS,
    CACHE,
    HEAD_DIM,
    HEADS,
    NEG,
    PRIOR_LAYERS,
    TEXT,
    TOP_K,
    D,
    _attend,
    _cross_kv,
    _cross_out,
    _cross_probs,
    _ffn,
    _self_qkv,
)
from nemo.collections.tts.models import MagpieTTSModel
from torch import Tensor, nn

# The speaker's context padded to a width the Neural Engine compiler takes.
PREFILL = 224

# Halving steps for the top-k threshold: they narrow it to 1/65536 of the logits' range.
_STEPS = 16


def tables(model: MagpieTTSModel) -> dict[str, Tensor]:
    """The rows the app looks up: text_embeddings (text vocabulary, 768), speakers (5 · 217, 768)
    and decoder_positions (positions, 768)."""
    return {
        "text_embeddings": model.text_embedding.weight,
        "speakers": model.baked_context_embedding.weight.reshape(-1, D),
        "decoder_positions": model.decoder.position_embeddings.weight,
    }


class TextEncoder(modules.TextEncoder):
    """`modules.TextEncoder` from the tokens' embeddings, embeds (1, 256, 768)."""

    def forward(self, embeds: Tensor, mask: Tensor) -> Tensor:
        x = embeds + self.encoder.position_embeddings.weight[:TEXT]
        for layer in self.encoder.layers:
            q, k, v = _self_qkv(layer, x)
            x = x + layer.self_attention.o_net(_attend(q, k, v, self.causal))
            h = layer.norm_pos_ff(x).transpose(1, 2)
            h = modules.F.gelu(
                modules.F.conv1d(modules.F.pad(h, (2, 0)), layer.pos_ff.proj.conv.weight), approximate="tanh"
            )
            h = modules.F.conv1d(modules.F.pad(h, (2, 0)), layer.pos_ff.o_net.conv.weight)
            x = x + h.transpose(1, 2)
        return self.encoder.norm_out(x) * mask.unsqueeze(-1)


class DecoderPrefill(nn.Module):
    """`modules.DecoderPrefill` from the speaker's context, context (1, PREFILL, 768): its 217
    rows, then zeros. Outputs keys and values of the PREFILL positions in the caches' layout, of
    which the first 217 are the context's (causal attention keeps them apart from the padding),
    and the text's cross-attention keys and values, cross_keys and cross_values (12, 1, 256,
    128)."""

    def __init__(self, model: MagpieTTSModel) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.register_buffer("causal", torch.triu(torch.full((PREFILL, PREFILL), NEG), diagonal=1))

    def forward(
        self, context: Tensor, encoder_output: Tensor, encoder_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = torch.cat([context, torch.zeros_like(context)]) + self.decoder.position_embeddings.weight[:PREFILL]
        text_mask = ((1 - encoder_mask) * NEG)[:, None, None, :]
        keys, values, cross_keys, cross_values = [], [], [], []
        for layer in self.decoder.layers:
            q, k, v = _self_qkv(layer, x)
            x = x + layer.self_attention.o_net(_attend(q, k, v, self.causal))
            xk, xv = _cross_kv(layer, encoder_output)
            cross = _cross_out(layer, _cross_probs(layer, x[:1], xk, text_mask), xv)
            x = x + torch.cat([cross, torch.zeros_like(cross)])
            x = x + _ffn(layer, x)
            keys.append(k.transpose(2, 3))
            values.append(v)
            cross_keys.append(xk)
            cross_values.append(xv)
        return torch.cat(keys), torch.cat(values), torch.cat(cross_keys), torch.cat(cross_values)


class DecoderStep(nn.Module):
    """`modules.DecoderStep` over caches given as inputs.

    Inputs: audio_emb (1, 1, 768), the previous step's embedding plus the position embedding of
    the step; mask (1, 480), 0 for the cache slots before the step and NEG from it on;
    encoder_mask, attn_prior (1, 256); cfg_scale (1,); key_cache (24, 12, 64, 480), value_cache
    (24, 12, 480, 64); cross_keys, cross_values (12, 1, 256, 128). Outputs: logits, h_last and
    alignment as `modules.DecoderStep`, and the step's key (24, 12, 64, 1) and value
    (24, 12, 1, 64) for the app to write at the step's slot. Each layer attends to the cache and
    to the step's own key and value apart.
    """

    def __init__(self, model: MagpieTTSModel) -> None:
        super().__init__()
        self.decoder = model.decoder
        self.final_proj = model.final_proj

    def forward(
        self,
        audio_emb: Tensor,
        mask: Tensor,
        encoder_mask: Tensor,
        attn_prior: Tensor,
        cfg_scale: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        cross_keys: Tensor,
        cross_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        x = audio_emb.expand(2, 1, D)
        mask = mask.view(1, 1, 1, CACHE)
        text_mask = ((1 - encoder_mask) * NEG)[:, None, None, :]
        prior = attn_prior[:, None, None, :]
        scale = HEAD_DIM**-0.5
        alignment, keys, values = [], [], []
        for i, layer in enumerate(self.decoder.layers):
            q, k, v = _self_qkv(layer, x)  # (2, 12, 1, 64)
            rows = slice(2 * i, 2 * i + 2)
            cached = torch.matmul(q, key_cache[rows]) * scale + mask
            own = (q * k).sum(-1, keepdim=True) * scale
            probs = torch.softmax(torch.cat([cached, own], dim=-1), dim=-1)
            attended = torch.matmul(probs[..., :CACHE], value_cache[rows]) + probs[..., CACHE:] * v
            x = x + layer.self_attention.o_net(attended.transpose(1, 2).flatten(2))
            cross = _cross_probs(layer, x[:1], cross_keys[i : i + 1], text_mask)
            if i in PRIOR_LAYERS:
                cross = cross * prior
                cross = cross / cross.sum(dim=-1, keepdim=True)
            if i in ALIGNMENT_LAYERS:
                alignment.append(cross)
            cross_out = _cross_out(layer, cross, cross_values[i : i + 1])
            x = x + torch.cat([cross_out, torch.zeros_like(cross_out)])
            x = x + _ffn(layer, x)
            keys.append(k.transpose(2, 3))
            values.append(v)
        h = self.decoder.norm_out(x)
        both = self.final_proj(h)
        logits = both[1:] + cfg_scale * (both[:1] - both[1:])
        return (
            logits.reshape(1, 1, CODES, VOCAB),
            h,
            torch.stack(alignment).mean(dim=0).reshape(1, -1),
            torch.cat(keys),
            torch.cat(values),
        )


class LocalTransformer(modules.LocalTransformer):
    """`modules.LocalTransformer` without `torch.topk`: the top-k threshold is the largest value
    that at least TOP_K logits reach, found by bisection, and the drawn code the largest of the
    kept scores, the larger id on an exact tie."""

    def forward(
        self, h_last: Tensor, gumbel: Tensor, temperature: Tensor, cfg_scale: Tensor, forbid_eos: Tensor
    ) -> tuple[Tensor, Tensor]:
        positions = self.transformer.position_embeddings.weight
        bias = self.forbidden + forbid_eos * self.eos
        empty = torch.zeros(2, HEADS, CODES, HEAD_DIM, dtype=h_last.dtype)
        keys = [empty for _ in self.transformer.layers]
        values = [empty for _ in self.transformer.layers]
        x = h_last + positions[0]
        codes, embedded = [], []
        for j in range(CODES):
            for i, layer in enumerate(self.transformer.layers):
                q, k, v = _self_qkv(layer, x)
                keys[i] = keys[i] + k * self.slots[j]
                values[i] = values[i] + v * self.slots[j]
                x = x + layer.self_attention.o_net(_attend(q, keys[i], values[i], self.visible[j]))
                x = x + _ffn(layer, x)
            both = self.heads[j](x[:, 0])
            logits = both[1:] + cfg_scale * (both[:1] - both[1:]) + bias
            low, high = logits.amin(-1, keepdim=True), logits.amax(-1, keepdim=True)
            for _ in range(_STEPS):
                middle = (low + high) / 2
                enough = ((logits >= middle).to(logits.dtype).sum(-1, keepdim=True) >= TOP_K).to(logits.dtype)
                low = low + enough * (middle - low)
                high = middle + enough * (high - middle)
            kept = (logits >= low).to(logits.dtype)
            scores = (logits / temperature + gumbel[j : j + 1]) * kept + (1 - kept) * NEG
            hot = (scores >= scores.amax(-1, keepdim=True)).to(logits.dtype)
            code = (hot * self.ids).amax(-1, keepdim=True)
            onehot = (self.ids == code).to(x.dtype)
            embedding = onehot @ self.embeddings[j].weight
            codes.append(code)
            embedded.append(embedding)
            x = (embedding + positions[j + 1]).expand(2, 1, D)
        return torch.cat(codes, dim=1), torch.stack(embedded).mean(dim=0)[None]


