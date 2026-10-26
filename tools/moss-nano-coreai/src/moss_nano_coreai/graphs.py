"""moss-nano-coreml's graphs rearranged for the Neural Engine of Core AI.

MPSGraph puts what the Neural Engine does not take on the GPU, which iOS denies to apps in the
background:
- Lookups by integer index run on the GPU, so the decoder takes the sum of its rows' embeddings,
  which the app reads from embeddings.f16 (`embedding_table`).
- `torch.topk` runs on the GPU, so the sampler finds the top-k and top-p thresholds by bisection
  and draws among the tokens above them, with noise per token instead of per rank.
- The local decoder keeps its keys and values in fixed buffers through one-hot writes: grown by
  concatenation, they come out wrong from the Neural Engine, as in qwen3-coreai's predictor.

The functions of a package share their weights through one module (`Decoder`, `Local`), which
the converter stores once.
"""

import torch
from moss_nano_coreml import graphs
from moss_nano_coreml.graphs import CODEBOOK_SIZE, CONTINUE_TOKEN, END_TOKEN, HEAD_DIM, HEADS, MASKED
from torch import Tensor, nn

# Halving steps for a threshold: they narrow it to 1/65536 of the scores' range.
_STEPS = 16
_SLOTS = 32


def install() -> None:
    """Makes moss-nano-coreml's RoPE rotate in the dtype of its input (`_rotate`) and its codec
    layers attend with one softmax over both key sets (`_attend`)."""
    graphs.rotate = _rotate
    graphs.attend = _attend


def _rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """`graphs.rotate` with its rotation matrix, a module constant `.half()` leaves in Float32, in
    the dtype of `x`."""
    return x * cos + (x @ graphs.SWAP_HALVES.to(x.dtype)) * sin


def _attend(
    q: Tensor, past_keys: Tensor, past_values: Tensor, past_mask: Tensor, keys: Tensor, values: Tensor, mask: Tensor
) -> Tensor:
    """`graphs.attend` as one softmax over the concatenated scores: merged by their log-sum-exp,
    the two softmaxes run on the GPU."""
    past = q @ past_keys.transpose(1, 2) + past_mask
    own = q @ keys.transpose(1, 2) + mask
    probs = torch.softmax(torch.cat([past, own], -1), -1)
    cached = past.shape[-1]
    return probs[..., :cached] @ past_values + probs[..., cached:] @ values


class _LayerNorm(nn.Module):
    """`nn.LayerNorm` over the last axis without overflowing Float16. The Neural Engine sums the
    squares of the residual stream in Float16, which overflows on the prompt's text rows. This
    divides the centered values by 32 times their mean magnitude first: a scaled value is then at
    most size / 32, and their mean square at most (size / 32)^2."""

    def __init__(self, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.weight = norm.weight
        self.bias = norm.bias
        self.eps = norm.eps

    def forward(self, x: Tensor) -> Tensor:
        share = 1 / x.shape[-1]
        centered = x - (x * share).sum(-1, keepdim=True)
        scale = 32 * torch.clamp((centered.abs() * share).sum(-1, keepdim=True), min=1e-4)
        scaled = centered / scale
        variance = (scaled * scaled * share).sum(-1, keepdim=True)
        return scaled / torch.sqrt(variance + self.eps / (scale * scale)) * self.weight + self.bias


def _safe_norms(module: nn.Module) -> None:
    """Replaces every `nn.LayerNorm` under `module` with `_LayerNorm`, in place."""
    for parent in list(module.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(parent, name, _LayerNorm(child))


def embedding_table(model: nn.Module) -> Tensor:
    """The rows `graphs.Embedding` sums, [16384 + 16 * 1025, 768]: the text table, then each
    codebook's with a zero row for the pad id."""
    return graphs.Embedding(model).table


class Decoder(nn.Module):
    """The global decoder's blocks and final norm."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(graphs.Block(block) for block in model.transformer.h)
        self.ln_f = model.transformer.ln_f
        _safe_norms(self)


class Prefill(nn.Module):
    """`graphs.GlobalPrefill` from the rows' embeddings, embeds [T, 768]."""

    def __init__(self, decoder: Decoder, length: int) -> None:
        super().__init__()
        self.decoder = decoder
        cos, sin = graphs.rope_tables(torch.arange(length), HEAD_DIM)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        # 4D: the Neural Engine's attention takes a mask broadcastable to [1, heads, T, T].
        self.register_buffer("causal", torch.triu(torch.full((length, length), MASKED), 1)[None, None])

    def forward(self, embeds: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = embeds
        keys, values = [], []
        for block in self.decoder.blocks:
            q, k, v = block.project(x, self.cos, self.sin)
            probs = torch.softmax(q[None] @ k[None].transpose(2, 3) + self.causal, -1)
            x = block.finish(x, (probs @ v[None])[0])
            keys.append(k)
            values.append(v)
        return self.decoder.ln_f(x), torch.stack(keys), torch.stack(values)


class Step(nn.Module):
    """`graphs.GlobalStep` from the row's embedding, embed [1, 768], with the key cache transposed,
    [layers, heads, 64, context]: transposed in the graph, its scores come out wrong from the Neural
    Engine once the cache holds the prompt's text rows."""

    def __init__(self, decoder: Decoder) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(
        self, embed: Tensor, key_cache: Tensor, value_cache: Tensor, mask: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        x = embed
        keys, values = [], []
        context = key_cache.shape[3]
        for layer, block in enumerate(self.decoder.blocks):
            q, k, v = block.project(x, cos, sin)
            scores = torch.cat([q @ key_cache[layer] + mask, (q * k).sum(-1, keepdim=True)], -1)
            probs = torch.softmax(scores, -1)
            x = block.finish(x, probs[..., :context] @ value_cache[layer] + probs[..., context:] * v)
            keys.append(k)
            values.append(v)
        return self.decoder.ln_f(x), torch.stack(keys), torch.stack(values)


class Local(nn.Module):
    """The local decoder's block, final norm and tables."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        local = model.local_transformer
        self.block = graphs.Block(local.h[0])
        self.ln_f = local.ln_f
        _safe_norms(self)
        wte = model.transformer.wte.weight.detach()
        self.register_buffer("continue_embedding", wte[CONTINUE_TOKEN : CONTINUE_TOKEN + 1].clone())
        self.register_buffer("text_head", wte[[CONTINUE_TOKEN, END_TOKEN]].clone())
        self.audio = nn.Parameter(
            torch.stack([table.weight.detach() for table in model.audio_embeddings]), requires_grad=False
        )


class Sampler(nn.Module):
    """`graphs.LocalSampler` with the same inputs but `gumbel`, which is [codebooks, 1024]: noise
    per token."""

    def __init__(self, local: Local, codebooks: int) -> None:
        super().__init__()
        self.local = local
        self.codebooks = codebooks
        cos, sin = graphs.rope_tables(torch.arange(codebooks + 1), HEAD_DIM)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("ids", torch.arange(CODEBOOK_SIZE, dtype=torch.float32))
        slots = torch.arange(_SLOTS)
        positions = torch.arange(codebooks + 1)[:, None]
        self.register_buffer("slot", (slots[None] == positions).float())
        self.register_buffer("hidden", torch.where(slots[None] <= positions, 0.0, MASKED))
        # Position 1 projects the constant continue embedding. MPSGraph runs a branch of constants
        # alone on the GPU, so its q, k and v are computed here.
        with torch.no_grad():
            first = local.block.project(local.continue_embedding, cos[1:2].to(local.audio.dtype), sin[1:2].to(local.audio.dtype))
        for name, tensor in zip(("first_q", "first_k", "first_v"), first, strict=True):
            self.register_buffer(name, tensor.unsqueeze(0).clone())

    def forward(
        self, hidden: Tensor, seen: Tensor, gumbel: Tensor, text_gumbel: Tensor, settings: Tensor
    ) -> tuple[Tensor, Tensor]:
        local = self.local
        keys = torch.zeros(1, HEADS, _SLOTS, HEAD_DIM, dtype=hidden.dtype)
        values = torch.zeros(1, HEADS, _SLOTS, HEAD_DIM, dtype=hidden.dtype)

        def advance(x: Tensor, position: int) -> Tensor:
            nonlocal keys, values
            cos, sin = self.cos[position : position + 1], self.sin[position : position + 1]
            if position == 1:
                q, k, v = self.first_q, self.first_k, self.first_v
            else:
                q, k, v = (t.unsqueeze(0) for t in local.block.project(x, cos, sin))
            slot = self.slot[position].view(1, 1, _SLOTS, 1)
            keys = keys + k * slot
            values = values + v * slot
            probs = torch.softmax(q @ keys.transpose(2, 3) + self.hidden[position], -1)
            return local.ln_f(local.block.finish(x, (probs @ values).squeeze(0)))

        text_logits = advance(hidden, 0) @ local.text_head.T * settings[:, 0:1] + text_gumbel
        end = (text_logits[:, 1:2] > text_logits[:, 0:1]).to(hidden.dtype)

        x = local.continue_embedding
        codes: list[Tensor] = []
        for channel in range(self.codebooks):
            table = local.audio[channel]
            logits = advance(x, channel + 1) @ table.T
            code = self.sample(logits, seen[channel : channel + 1], gumbel[channel : channel + 1], settings)
            codes.append(code)
            x = (self.ids[None] == code).to(code.dtype) @ table
        return end, torch.cat(codes, -1)

    def sample(self, logits: Tensor, seen: Tensor, gumbel: Tensor, settings: Tensor) -> Tensor:
        """logits/seen/gumbel [1, 1024] -> the drawn token id [1, 1] as a float.

        Top-k keeps the tokens at or above the largest score that `top_k` tokens reach; top-p then
        keeps those whose better tokens carry at most `top_p` of the kept mass, as upstream. Each
        threshold is found to within 1/65536 of the scores' range.
        """
        inv_temperature, top_k, top_p = settings[:, 1:2], settings[:, 2:3], settings[:, 3:4]
        penalty, inv_penalty = settings[:, 4:5], settings[:, 5:6]
        penalized = logits + seen * (
            torch.clamp(logits, max=0.0) * (penalty - 1.0)
            + torch.clamp(logits, min=0.0) * (inv_penalty - 1.0)
        )
        scores = penalized * inv_temperature
        floor, peak = scores.amin(-1, keepdim=True), scores.amax(-1, keepdim=True)
        low, high = floor, peak
        for _ in range(_STEPS):
            middle = (low + high) / 2
            enough = ((scores >= middle).to(scores.dtype).sum(-1, keepdim=True) >= top_k).to(scores.dtype)
            low = low + enough * (middle - low)
            high = middle + enough * (high - middle)
        top = (scores >= low).to(scores.dtype)
        weights = torch.exp(scores - peak) * top
        mass = weights / weights.sum(-1, keepdim=True)
        # The lowest threshold whose better tokens carry at most top_p, which lies in (low, high].
        low, high = floor, peak
        for _ in range(_STEPS):
            middle = (low + high) / 2
            within = ((mass * (scores > middle).to(scores.dtype)).sum(-1, keepdim=True) <= top_p).to(scores.dtype)
            high = high + within * (middle - high)
            low = middle + within * (low - middle)
        kept = top * (scores > low).to(scores.dtype)
        drawn = scores + gumbel + (kept - 1.0) * -MASKED
        hot = (drawn >= drawn.max(-1, keepdim=True).values).to(scores.dtype)
        # The largest id among exact ties.
        return (hot * self.ids[None]).amax(-1, keepdim=True)


class _CodecStage(graphs.CodecStage):
    """`graphs.CodecStage` given its validity mask, [1, cache], 0 for the slots that hold a key
    and MASKED for the rest: computed from `past_frames` in the graph, it runs on the GPU. Returns
    the chunk's keys and values, [layers, heads, tokens, 64], for the caller to append to the
    caches: an output that copies part of an input runs on the GPU."""

    def forward(self, x: Tensor, keys: Tensor, values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        masks = (self.window_mask + valid, self.causal)
        rope = (self.cos, self.sin, self.cache_cos, self.cache_sin)
        x = self.input_proj(x)
        new_keys, new_values = [], []
        for index, layer in enumerate(self.layers):
            x, k, v = layer(x, keys[index], values[index], rope, masks)
            new_keys.append(k)
            new_values.append(v)
        return self.output_proj(x), torch.stack(new_keys), torch.stack(new_values)


def _unpatch(x: Tensor, patch: int) -> Tensor:
    """`graphs.unpatch` as a stack of the patch's columns: with the transpose, coreai-torch
    commutes the next projection into it for one of its uses and fails to type the residual sum."""
    length, width = x.shape
    columns = x.reshape(length, width // patch, patch).unbind(-1)
    return torch.stack(columns, 1).reshape(length * patch, width // patch)


class Codec(graphs.CodecDecoder):
    """`graphs.CodecDecoder` with its one-hot codes in the model's dtype, taking each stage's
    validity mask in place of `past_frames` and returning each stage's chunk keys and values in
    place of the updated caches (`_CodecStage`). Slot j of a stage's cache, of `cache` slots and
    `rate` tokens per frame, holds a key once `cache - j <= past_frames * rate`."""

    def __init__(self, codec: nn.Module, chunk_frames: int) -> None:
        super().__init__(codec, chunk_frames)
        modules = list(codec.decoder)
        self.stages = nn.ModuleList(
            _CodecStage(modules[index], rate, chunk_frames)
            for index, rate in zip((1, 3, 5, 7), self.RATES, strict=True)
        )
        _safe_norms(self.stages)

    def forward(self, codes: Tensor, *inputs: Tensor) -> tuple[Tensor, ...]:
        valid, caches = inputs[: len(self.stages)], inputs[len(self.stages) :]
        onehot = (codes[:, :, None] == self.ids).to(codes.dtype)
        latent = torch.einsum("fqc,qcd->fqd", onehot, self.codebooks)
        used = onehot.sum(-1)
        x = self.output_proj(torch.einsum("fqd,qde->fe", latent, self.code_proj) + used @ self.code_bias)
        updated: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            x = _unpatch(x, self.patches[index])
            x, keys, values = stage(x, caches[2 * index], caches[2 * index + 1], valid[index])
            updated += [keys, values]
        samples = _unpatch(x, self.patches[-1])
        return (samples.reshape(-1, 2).T, *updated)
