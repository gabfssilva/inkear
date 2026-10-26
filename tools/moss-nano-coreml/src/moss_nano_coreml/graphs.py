"""ANE-oriented rewrites of MOSS-TTS-Nano and its codec, built from the upstream weights.

Every graph is stateless with static shapes: Core ML states fail to load on the CPU and Neural
Engine on macOS 27 (error -14), so the key/value caches travel as inputs and outputs and the host
owns them. RoPE angles never reach fp16 arithmetic: decode steps get cos/sin from the host, and
graphs whose positions are fixed bake them in as constants.

Query/key head dimensions are permuted from GPT-2's interleaved RoPE pairs to split halves, which
leaves every dot product unchanged and turns `rotate_half` into a fixed signed permutation.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

HIDDEN = 768
HEADS = 12
HEAD_DIM = 64
LAYERS = 12
CODEBOOKS = 16
CODEBOOK_SIZE = 1024
TEXT_VOCAB = 16384
AUDIO_PAD = 1024
CONTINUE_TOKEN = 9
END_TOKEN = 7
MASKED = -30000.0
# Sampling candidates per codebook; top_k must not exceed it.
CANDIDATES = 32


def half_layout(head_dim: int) -> Tensor:
    """Index that reorders one head's interleaved RoPE pairs into [even dims, odd dims]."""
    return torch.cat([torch.arange(0, head_dim, 2), torch.arange(1, head_dim, 2)])


def rope_tables(positions: Tensor, head_dim: int, base: float = 10000.0) -> tuple[Tensor, Tensor]:
    """cos/sin in split-half layout, [len(positions), head_dim], computed in float64."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    angles = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    return torch.cat([angles.cos(), angles.cos()], -1).float(), torch.cat(
        [angles.sin(), angles.sin()], -1
    ).float()


def _swap_halves(dim: int) -> Tensor:
    """x @ swap == concat(-x[d/2:], x[:d/2]), i.e. rotate_half in split-half layout."""
    half = dim // 2
    swap = torch.zeros(dim, dim)
    swap[torch.arange(half) + half, torch.arange(half)] = -1.0
    swap[torch.arange(half), torch.arange(half) + half] = 1.0
    return swap


# Built at import so tracing sees a constant. A matmul by it runs far faster on the Neural Engine
# than slicing the innermost axis.
SWAP_HALVES = _swap_halves(HEAD_DIM)


def rotate(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """RoPE in split-half layout (head_dim 64 for both the decoder and the codec)."""
    return x * cos + (x @ SWAP_HALVES) * sin


def attend(
    q: Tensor,
    past_keys: Tensor,
    past_values: Tensor,
    past_mask: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor,
) -> Tensor:
    """Softmax attention over two key sets (a cache and the current rows) without concatenating them.

    Each set gets its own max-shifted softmax; the two are merged with weights from their
    log-sum-exp. Concatenating scores and slicing probabilities costs the Neural Engine twice as
    much, and a shared max taken across both sets makes its compiler reject the graph (error 22).
    A fully masked cache gets weight sigmoid(-30000) = 0.

    q [h, n, d]; past_* [h, m, d]; keys/values [h, k, d]; masks are additive and broadcast to scores.
    """
    past = q @ past_keys.transpose(1, 2) + past_mask
    own = q @ keys.transpose(1, 2) + mask
    past_peak, own_peak = past.max(-1, keepdim=True).values, own.max(-1, keepdim=True).values
    past, own = torch.exp(past - past_peak), torch.exp(own - own_peak)
    past_total, own_total = past.sum(-1, keepdim=True), own.sum(-1, keepdim=True)
    weight = torch.sigmoid(past_peak + torch.log(past_total) - own_peak - torch.log(own_total))
    return weight * (past @ past_values / past_total) + (1.0 - weight) * (own @ values / own_total)


class Block(nn.Module):
    """One GPT-2 block of the upstream decoder with q/k permuted to split halves and q pre-scaled."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.ln_1 = source.ln_1
        self.ln_2 = source.ln_2
        self.c_proj = source.attn.c_proj
        self.fc_in = source.mlp.fc_in
        self.fc_out = source.mlp.fc_out
        order = half_layout(HEAD_DIM)
        rows = torch.cat([head * HEAD_DIM + order for head in range(HEADS)])
        weight = source.attn.c_attn.weight.detach()
        bias = source.attn.c_attn.bias.detach()
        scale = 1.0 / math.sqrt(HEAD_DIM)
        self.q = nn.Linear(HIDDEN, HIDDEN)
        self.k = nn.Linear(HIDDEN, HIDDEN)
        self.v = nn.Linear(HIDDEN, HIDDEN)
        with torch.no_grad():
            self.q.weight.copy_(weight[:HIDDEN][rows] * scale)
            self.q.bias.copy_(bias[:HIDDEN][rows] * scale)
            self.k.weight.copy_(weight[HIDDEN : 2 * HIDDEN][rows])
            self.k.bias.copy_(bias[HIDDEN : 2 * HIDDEN][rows])
            self.v.weight.copy_(weight[2 * HIDDEN :])
            self.v.bias.copy_(bias[2 * HIDDEN :])

    def project(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """x [T, 768] -> q, k, v as [heads, T, head_dim], q and k rotated."""
        h = self.ln_1(x)
        length = x.shape[0]

        def heads(t: Tensor) -> Tensor:
            return t.reshape(length, HEADS, HEAD_DIM).transpose(0, 1)

        return rotate(heads(self.q(h)), cos, sin), rotate(heads(self.k(h)), cos, sin), heads(self.v(h))

    def finish(self, x: Tensor, attended: Tensor) -> Tensor:
        """Residual, output projection and MLP; attended is [heads, T, head_dim]."""
        x = x + self.c_proj(attended.transpose(0, 1).reshape(x.shape[0], HIDDEN))
        return x + self.fc_out(F.gelu(self.fc_in(self.ln_2(x)), approximate="tanh"))


class Embedding(nn.Module):
    """Row embedding: text table plus the 16 audio tables, each with a zero row for the pad id."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        audio = [
            torch.cat([table.weight.detach(), torch.zeros(1, HIDDEN)]) for table in model.audio_embeddings
        ]
        self.table = nn.Parameter(
            torch.cat([model.transformer.wte.weight.detach(), *audio]), requires_grad=False
        )
        offsets = [0] + [TEXT_VOCAB + channel * (CODEBOOK_SIZE + 1) for channel in range(CODEBOOKS)]
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int32))

    def forward(self, rows: Tensor) -> Tensor:
        """rows [T, 17] int32 -> [T, 768]."""
        return F.embedding((rows + self.offsets).long(), self.table).sum(1)


class GlobalPrefill(nn.Module):
    """Causal pass over a whole prompt padded to `length` rows, from position 0.

    Rows past the real prompt are padding; causality keeps them from affecting earlier rows.
    """

    def __init__(self, model: nn.Module, length: int) -> None:
        super().__init__()
        self.embedding = Embedding(model)
        self.blocks = nn.ModuleList(Block(block) for block in model.transformer.h)
        self.ln_f = model.transformer.ln_f
        cos, sin = rope_tables(torch.arange(length), HEAD_DIM)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("causal", torch.triu(torch.full((length, length), MASKED), 1))

    def forward(self, rows: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """rows [T, 17] int32 -> hidden [T, 768], keys and values [layers, heads, T, head_dim]."""
        x = self.embedding(rows)
        keys, values = [], []
        for block in self.blocks:
            q, k, v = block.project(x, self.cos, self.sin)
            probs = torch.softmax(q @ k.transpose(1, 2) + self.causal, -1)
            x = block.finish(x, probs @ v)
            keys.append(k)
            values.append(v)
        return self.ln_f(x), torch.stack(keys), torch.stack(values)


class GlobalStep(nn.Module):
    """One new row against a cache of `context` positions; the host appends the returned k/v."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.embedding = Embedding(model)
        self.blocks = nn.ModuleList(Block(block) for block in model.transformer.h)
        self.ln_f = model.transformer.ln_f

    def forward(
        self, row: Tensor, key_cache: Tensor, value_cache: Tensor, mask: Tensor, cos: Tensor, sin: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """row [1, 17]; caches [layers, heads, C, head_dim]; mask [1, C] additive; cos/sin [1, head_dim].

        Returns hidden [1, 768] and this row's keys and values [layers, heads, 1, head_dim].
        """
        x = self.embedding(row)
        keys, values = [], []
        for layer, block in enumerate(self.blocks):
            q, k, v = block.project(x, cos, sin)
            # One own key: a softmax over the concatenated scores is cheaper here than `attend`.
            cached = q @ key_cache[layer].transpose(1, 2) + mask
            probs = torch.softmax(torch.cat([cached, (q * k).sum(-1, keepdim=True)], -1), -1)
            context = key_cache.shape[2]
            x = block.finish(x, probs[..., :context] @ value_cache[layer] + probs[..., context:] * v)
            keys.append(k)
            values.append(v)
        return self.ln_f(x), torch.stack(keys), torch.stack(values)


class LocalSampler(nn.Module):
    """One frame of the local decoder with sampling in the graph: continue/end, then `codebooks` codes.

    Matches upstream `_iter_generation_events` with Gumbel-max sampling: given i.i.d. Gumbel noise g,
    the drawn token is argmax over the kept candidates of (score / temperature + g), the same
    distribution as upstream's softmax + multinomial. Candidates are the CANDIDATES best scores
    (`topk`); top-k keeps the first k of them and top-p those whose better candidates carry at most
    top_p of the mass, which keeps the first one crossing it, as upstream. Noise is indexed by
    candidate rank, which keeps it i.i.d.
    """

    def __init__(self, model: nn.Module, codebooks: int = CODEBOOKS) -> None:
        super().__init__()
        local = model.local_transformer
        self.codebooks = codebooks
        self.block = Block(local.h[0])
        self.ln_f = local.ln_f
        wte = model.transformer.wte.weight.detach()
        self.register_buffer("continue_embedding", wte[CONTINUE_TOKEN : CONTINUE_TOKEN + 1].clone())
        self.register_buffer("text_head", wte[[CONTINUE_TOKEN, END_TOKEN]].clone())
        # All 16 tables in every depth variant, so multifunction packages store them once.
        self.audio = nn.Parameter(
            torch.stack([table.weight.detach() for table in model.audio_embeddings]), requires_grad=False
        )
        cos, sin = rope_tables(torch.arange(codebooks + 1), HEAD_DIM)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        self.register_buffer("ids", torch.arange(CODEBOOK_SIZE, dtype=torch.float32))
        self.register_buffer("ranks", torch.arange(CANDIDATES, dtype=torch.float32)[None])
        self.register_buffer("better", torch.triu(torch.ones(CANDIDATES, CANDIDATES), 1))

    def forward(
        self, hidden: Tensor, seen: Tensor, gumbel: Tensor, text_gumbel: Tensor, settings: Tensor
    ) -> tuple[Tensor, Tensor]:
        """hidden [1, 768]; seen [codebooks, 1024] 0/1; gumbel [codebooks, CANDIDATES]; text_gumbel [1, 2];
        settings [1, 6] = (1/text_temperature, 1/temperature, top_k, top_p, penalty, 1/penalty).

        Returns end [1, 1] (1 when the model ends speech) and codes [1, codebooks] as floats.
        """
        keys: list[Tensor] = []
        values: list[Tensor] = []

        def advance(x: Tensor, position: int) -> Tensor:
            cos, sin = self.cos[position : position + 1], self.sin[position : position + 1]
            q, k, v = self.block.project(x, cos, sin)
            keys.append(k)
            values.append(v)
            probs = torch.softmax(q @ torch.cat(keys, 1).transpose(1, 2), -1)
            return self.ln_f(self.block.finish(x, probs @ torch.cat(values, 1)))

        text_logits = advance(hidden, 0) @ self.text_head.T * settings[:, 0:1] + text_gumbel
        end = (text_logits[:, 1:2] > text_logits[:, 0:1]).float()

        x = self.continue_embedding
        codes: list[Tensor] = []
        for channel in range(self.codebooks):
            table = self.audio[channel]
            logits = advance(x, channel + 1) @ table.T
            code = self.sample(logits, seen[channel : channel + 1], gumbel[channel : channel + 1], settings)
            codes.append(code)
            x = (self.ids[None] == code).float() @ table
        return end, torch.cat(codes, -1)

    def sample(self, logits: Tensor, seen: Tensor, gumbel: Tensor, settings: Tensor) -> Tensor:
        """logits/seen [1, 1024], gumbel [1, CANDIDATES] -> drawn token id [1, 1] as a float.

        The argmax is a max followed by the largest rank reaching it: reduce_argmax yields int32,
        which the Neural Engine does not run. The tie-break only matters for exact fp16 ties.
        """
        inv_temperature, top_k, top_p = settings[:, 1:2], settings[:, 2:3], settings[:, 3:4]
        penalty, inv_penalty = settings[:, 4:5], settings[:, 5:6]
        penalized = logits + seen * (
            torch.clamp(logits, max=0.0) * (penalty - 1.0)
            + torch.clamp(logits, min=0.0) * (inv_penalty - 1.0)
        )
        scores, indices = torch.topk(penalized * inv_temperature, CANDIDATES, -1)
        kept = (self.ranks < top_k).float()
        weights = torch.exp(scores - scores[:, :1]) * kept
        kept = kept * ((weights / weights.sum(-1, keepdim=True)) @ self.better <= top_p).float()
        drawn = scores + gumbel + (kept - 1.0) * -MASKED
        reached = (drawn >= drawn.max(-1, keepdim=True).values).float()
        rank = (reached * self.ranks).max(-1, keepdim=True).values
        return ((self.ranks == rank).float() * indices.float()).sum(-1, keepdim=True)


CODEC_HEADS = 4
CODEC_HEAD_DIM = 64
CODEC_WIDTH = 256
FRAMES_PER_SECOND = 12.5


class CodecLayer(nn.Module):
    """One causal sliding-window layer of the codec decoder, q/k in split-half layout, q pre-scaled."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.norm1 = source.norm1
        self.norm2 = source.norm2
        self.out_proj = source.self_attn.out_proj
        self.ffn_in = source.ffn[0]
        self.ffn_out = source.ffn[2]
        self.register_buffer("scale1", source.layer_scale_1.scale.detach().clone())
        self.register_buffer("scale2", source.layer_scale_2.scale.detach().clone())
        weight = source.self_attn.in_proj.weight.detach()
        order = half_layout(CODEC_HEAD_DIM)
        rows = torch.cat([head * CODEC_HEAD_DIM + order for head in range(CODEC_HEADS)])
        self.q = nn.Linear(CODEC_WIDTH, CODEC_WIDTH, bias=False)
        self.k = nn.Linear(CODEC_WIDTH, CODEC_WIDTH, bias=False)
        self.v = nn.Linear(CODEC_WIDTH, CODEC_WIDTH, bias=False)
        with torch.no_grad():
            self.q.weight.copy_(weight[:CODEC_WIDTH][rows] / math.sqrt(CODEC_HEAD_DIM))
            self.k.weight.copy_(weight[CODEC_WIDTH : 2 * CODEC_WIDTH][rows])
            self.v.weight.copy_(weight[2 * CODEC_WIDTH :])

    def forward(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        rope: tuple[Tensor, Tensor, Tensor, Tensor],
        masks: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """x [n, 256]; keys/values [heads, cache, 64] unrotated. Returns x and this chunk's raw k, v."""
        length = x.shape[0]

        def heads(t: Tensor) -> Tensor:
            return t.reshape(length, CODEC_HEADS, CODEC_HEAD_DIM).transpose(0, 1)

        h = self.norm1(x)
        q, k, v = heads(self.q(h)), heads(self.k(h)), heads(self.v(h))
        cos, sin, cache_cos, cache_sin = rope
        q_rot, k_rot = rotate(q, cos, sin), rotate(k, cos, sin)
        attended = attend(q_rot, rotate(keys, cache_cos, cache_sin), values, masks[0], k_rot, v, masks[1])
        x = x + self.scale1 * self.out_proj(attended.transpose(0, 1).reshape(length, CODEC_WIDTH))
        x = x + self.scale2 * self.ffn_out(F.gelu(self.ffn_in(self.norm2(x))))
        return x, k, v


class CodecStage(nn.Module):
    """input_proj -> causal layers with a `window`-token context -> output_proj, at `rate` tokens/frame."""

    def __init__(self, source: nn.Module, rate: int, chunk_frames: int) -> None:
        super().__init__()
        self.input_proj = source.input_proj
        self.output_proj = source.output_proj
        self.layers = nn.ModuleList(CodecLayer(layer) for layer in source.transformer.layers)
        self.rate = rate
        window = source.transformer.layers[0].self_attn.context
        self.cache = window - 1
        tokens = rate * chunk_frames
        cos, sin = rope_tables(torch.arange(tokens), CODEC_HEAD_DIM)
        cache_cos, cache_sin = rope_tables(torch.arange(-self.cache, 0), CODEC_HEAD_DIM)
        for name, table in {"cos": cos, "sin": sin, "cache_cos": cache_cos, "cache_sin": cache_sin}.items():
            self.register_buffer(name, table)
        query = torch.arange(tokens)[:, None]
        slot = torch.arange(self.cache)[None, :]
        # Cache slot j holds the key `cache - j` tokens before the chunk; query t sees it while it is
        # inside the window of `window` tokens ending at t, i.e. from slot t on.
        self.register_buffer("window_mask", torch.where(slot >= query, 0.0, MASKED))
        self.register_buffer("causal", torch.triu(torch.full((tokens, tokens), MASKED), 1))
        self.register_buffer("distance", (self.cache - torch.arange(self.cache, dtype=torch.float32))[None])

    def forward(
        self, x: Tensor, keys: Tensor, values: Tensor, past_frames: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """x [n, in]; keys/values [layers, heads, cache, 64]; past_frames [1, 1] frames decoded so far (capped)."""
        valid = torch.where(self.distance <= past_frames * self.rate, 0.0, MASKED)
        masks = (self.window_mask + valid, self.causal)
        rope = (self.cos, self.sin, self.cache_cos, self.cache_sin)
        x = self.input_proj(x)
        new_keys, new_values = [], []
        for index, layer in enumerate(self.layers):
            x, k, v = layer(x, keys[index], values[index], rope, masks)
            new_keys.append(torch.cat([keys[index], k], 1)[:, -self.cache :])
            new_values.append(torch.cat([values[index], v], 1)[:, -self.cache :])
        return self.output_proj(x), torch.stack(new_keys), torch.stack(new_values)


def unpatch(x: Tensor, patch: int) -> Tensor:
    """Upstream PatchedPretransform.decode in time-major layout: [T, d * patch] -> [T * patch, d]."""
    length, width = x.shape
    return x.reshape(length, width // patch, patch).transpose(1, 2).reshape(length * patch, width // patch)


class CodecDecoder(nn.Module):
    """`chunk_frames` frames of codes -> stereo audio, carrying each stage's sliding-window cache."""

    RATES = (4, 8, 16, 32)

    def __init__(self, codec: nn.Module, chunk_frames: int) -> None:
        super().__init__()
        quantizer = codec.quantizer
        quantizers = quantizer.quantizers
        self.register_buffer("codebooks", torch.stack([q.codebook.weight.detach() for q in quantizers]))
        self.register_buffer(
            "code_proj", torch.stack([q.out_proj.weight.detach()[:, :, 0].T for q in quantizers])
        )
        self.register_buffer("code_bias", torch.stack([q.out_proj.bias.detach() for q in quantizers]))
        self.output_proj = nn.Linear(512, 768)
        with torch.no_grad():
            self.output_proj.weight.copy_(quantizer.output_proj.weight.detach()[:, :, 0])
            self.output_proj.bias.copy_(quantizer.output_proj.bias.detach())
        self.register_buffer("ids", torch.arange(CODEBOOK_SIZE, dtype=torch.float32))
        modules = list(codec.decoder)
        self.stages = nn.ModuleList(
            CodecStage(modules[index], rate, chunk_frames)
            for index, rate in zip((1, 3, 5, 7), self.RATES, strict=True)
        )
        self.patches = [modules[index].patch_size for index in (0, 2, 4, 6, 8)]

    def forward(self, codes: Tensor, past_frames: Tensor, *caches: Tensor) -> tuple[Tensor, ...]:
        """codes [F, 16] as floats, 1024 for an unused codebook (its bias is left out, as upstream decodes
        with fewer quantizers); past_frames [1, 1] = frames decoded before this chunk, capped at the
        125-frame window; caches = keys, values per stage.

        Returns audio [2, F * 3840] and the updated caches in the same order.
        """
        onehot = (codes[:, :, None] == self.ids).float()
        latent = torch.einsum("fqc,qcd->fqd", onehot, self.codebooks)
        used = onehot.sum(-1)
        x = self.output_proj(torch.einsum("fqd,qde->fe", latent, self.code_proj) + used @ self.code_bias)
        updated: list[Tensor] = []
        for index, stage in enumerate(self.stages):
            x = unpatch(x, self.patches[index])
            x, keys, values = stage(x, caches[2 * index], caches[2 * index + 1], past_frames)
            updated += [keys, values]
        samples = unpatch(x, self.patches[-1])
        return (samples.reshape(-1, 2).T, *updated)
