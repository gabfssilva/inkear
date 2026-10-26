"""Qwen3-TTS decoders rewritten for the Neural Engine.

Activations are channels-first, `[1, channels, 1, tokens]`, and every projection is a 1x1
convolution. A KV cache is `[1, layers * 1024, 1, slots]`, channel `layer * 1024 + head * 128 + dim`.

The Neural Engine pays a fixed cost per operation, which dominates single-token steps, so the
layers are written with as few operations as the math allows: norm weights are folded into the
projections, an RMS norm is two reductions' worth of ops, and rotary embeddings are elementwise.
"""

import copy
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HIDDEN = 1024
HEAD_DIM = 128
KV_HEADS = 8
Q_HEADS = 16
KV_WIDTH = KV_HEADS * HEAD_DIM
CODEBOOK = 2048
GROUPS = 15

# The code predictor's layer 2 MLP product reaches 1.8e5 and the residual stream 6.3e4, past fp16;
# from that layer on the stream is carried divided by 64. Earlier, its inputs are embeddings
# around 1e-2, which the division would push into fp16 subnormals.
PREDICTOR_SCALE = 64.0
PREDICTOR_SCALED_FROM = 2
_HIDE = -1e4
_PADDING_EPS = 1e-4
Rotation = Callable[[Tensor], Tensor]


def unit(x: Tensor, dim: int, eps: float = 0.0) -> Tensor:
    """`x / (|x| + eps)` over `dim`: the RMS norm, divided by `sqrt(size)`, with no epsilon by default.

    The model's epsilon (1e-6) is negligible next to the activations. `eps` only keeps padded,
    all-zero tokens finite. The Neural Engine's L2 reduction neither overflows on the residual
    streams' ~1e3 peaks nor underflows on 1e-4 inputs.
    """
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True)
    return x / (norm + eps if eps else norm)


_HIDDEN_RMS = HIDDEN**0.5
_HEAD_RMS = HEAD_DIM**0.5


def unit_eps(x: Tensor, eps: float) -> Tensor:
    """`x / sqrt(|x|^2 + size * eps)` over channels: `unit` with the RMS norm's epsilon kept exactly.

    The epsilon enters as one extra constant channel of the L2 reduction, which stays overflow-free.
    """
    pad = torch.full_like(x[:, :1], (x.shape[1] * eps) ** 0.5)
    return x / torch.linalg.vector_norm(torch.cat((x, pad), dim=1), dim=1, keepdim=True)


def _conv(linear: nn.Linear, columns: Tensor | None = None, scale: float = 1.0) -> Tensor:
    weight = linear.weight.detach().float() * scale
    if columns is not None:
        weight = weight * columns.detach().float()[None, :]
    return weight[:, :, None, None]


def rope_table(slots: int, theta: float) -> tuple[Tensor, Tensor]:
    """cos and sin of every slot, `[slots, 128]`, in the half-split layout of `rotate_half`."""
    inv_freq = 1.0 / theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float64) / HEAD_DIM)
    freqs = torch.arange(slots, dtype=torch.float64)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().float(), emb.sin().float()


def rotary(cos: Tensor, sin: Tensor) -> Rotation:
    """Rotation of heads `[1, H, 128, S]` with per-token `cos`/`sin` `[1, 1, 128, S]`.

    Elementwise on purpose: as a matmul by a constant rotation matrix, the Neural Engine rejects
    the whole attention block and Core ML moves it to the CPU.
    """
    half = HEAD_DIM // 2
    return lambda x: x * cos + torch.cat((-x[:, :, half:], x[:, :, :half]), dim=2) * sin


class Layer(nn.Module):
    """One Qwen3 decoder layer. The norm weights are folded into the projections that follow them.

    `residual_scale` divides the residual stream: dividing the input, the attention output and the
    MLP product leaves every norm output unchanged.
    """

    def __init__(self, layer: nn.Module, residual_scale: float) -> None:
        super().__init__()
        attn, mlp = layer.self_attn, layer.mlp
        pre, post = layer.input_layernorm.weight, layer.post_attention_layernorm.weight
        self.q = nn.Parameter(_conv(attn.q_proj, pre, _HIDDEN_RMS))
        self.k = nn.Parameter(_conv(attn.k_proj, pre, _HIDDEN_RMS))
        # The residual scale rides on the values, not on o_proj: o_proj / 64 would push many weights
        # below 6e-5, into fp16 subnormals, which the Neural Engine flushes to zero.
        self.v = nn.Parameter(_conv(attn.v_proj, pre, _HIDDEN_RMS / residual_scale))
        # The attention scale rides on the query norm.
        q_norm = attn.q_norm.weight.detach().float() * _HEAD_RMS * HEAD_DIM**-0.5
        self.q_norm = nn.Parameter(q_norm[None, None, :, None])
        self.k_norm = nn.Parameter((attn.k_norm.weight.detach().float() * _HEAD_RMS)[None, None, :, None])
        self.o = nn.Parameter(_conv(attn.o_proj))
        self.gate = nn.Parameter(_conv(mlp.gate_proj, post, _HIDDEN_RMS))
        self.up = nn.Parameter(_conv(mlp.up_proj, post, _HIDDEN_RMS / residual_scale))
        self.down = nn.Parameter(_conv(mlp.down_proj))

    def qkv(self, x: Tensor, rotate: Rotation, eps: float = 0.0) -> tuple[Tensor, Tensor, Tensor]:
        """Queries `[1, 16, 128, S]`, keys and values `[1, 8, 128, S]`, rotated to their positions."""
        tokens = x.shape[-1]
        h = unit(x, 1, eps)
        q = F.conv2d(h, self.q).view(1, Q_HEADS, HEAD_DIM, tokens)
        k = F.conv2d(h, self.k).view(1, KV_HEADS, HEAD_DIM, tokens)
        v = F.conv2d(h, self.v).view(1, KV_HEADS, HEAD_DIM, tokens)
        return rotate(unit(q, 2, eps) * self.q_norm), rotate(unit(k, 2, eps) * self.k_norm), v

    def attend(self, x: Tensor, q: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None) -> Tensor:
        """`x` plus attention of `q` over `keys [1, 8, 128, T]` and `values [1, 8, T, 128]`.

        `mask` is additive, `[1, 1, 2S, T]` with rows ordered (group, token).
        """
        tokens = q.shape[-1]
        # Query heads 2h and 2h+1 share key head h: rows ordered (head, group, token).
        if tokens == 1:
            queries = q.view(1, KV_HEADS, 2, HEAD_DIM)
        else:
            queries = q.view(1, KV_HEADS, 2, HEAD_DIM, tokens).transpose(3, 4)
            queries = queries.reshape(1, KV_HEADS, 2 * tokens, HEAD_DIM)
        scores = torch.matmul(queries, keys)
        if mask is not None:
            scores = scores + mask
        attended = torch.matmul(scores.softmax(-1), values)
        if tokens > 1:
            attended = attended.view(1, KV_HEADS, 2, tokens, HEAD_DIM).transpose(3, 4)
        return x + F.conv2d(attended.reshape(1, Q_HEADS * HEAD_DIM, 1, tokens), self.o)

    def feed_forward(self, x: Tensor, eps: float = 0.0) -> Tensor:
        h = unit(x, 1, eps)
        return x + F.conv2d(F.silu(F.conv2d(h, self.gate)) * F.conv2d(h, self.up), self.down)


def _layers(layers: nn.ModuleList, scaled_from: int | None = None) -> nn.ModuleList:
    return nn.ModuleList(
        Layer(layer, PREDICTOR_SCALE if scaled_from is not None and index >= scaled_from else 1.0)
        for index, layer in enumerate(layers)
    )


def _tokens_last(values: Tensor) -> Tensor:
    """Values `[1, 8, 128, S]` as `[1, 8, S, 128]`."""
    return values.transpose(2, 3)


class TalkerStep(nn.Module):
    """CodeDecoder `decode`: one talker step over the KV cache, kept as Core ML state.

    Inputs: `input_embeds [1, 1024, 1, 1]`; `key_padding_mask [1, slots]`, 0 for slots up to and
    including the step's, -1e4 after; `kv_cache_update_mask [1, slots]`, one-hot at the step's
    slot, which is also its rotary position. States `key_cache`/`value_cache [1, 28672, 1, slots]`.
    Outputs: `logits [1, 1, 3072]` and `hidden_states [1, 1024, 1, 1]`.

    The new cache is written as the last operation and never read back in the same prediction:
    on macOS 27, Core ML fails to plan (error -14) any program that reads a state after writing it.
    """

    def __init__(self, talker: nn.Module, slots: int) -> None:
        super().__init__()
        config = talker.config
        self.layers = _layers(talker.model.layers)
        norm = talker.model.norm.weight.detach().float()
        self.head = nn.Parameter(_conv(talker.codec_head, norm, _HIDDEN_RMS))
        # Plain tensors, not buffers: the converter takes every buffer of a stateful module for state.
        self.norm = norm[None, :, None, None] * _HIDDEN_RMS
        self.cos, self.sin = rope_table(slots, config.rope_theta)
        width = len(self.layers) * KV_WIDTH
        self.register_buffer("key_cache", torch.zeros(1, width, 1, slots))
        self.register_buffer("value_cache", torch.zeros(1, width, 1, slots))

    def forward(
        self, input_embeds: Tensor, key_padding_mask: Tensor, kv_cache_update_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        slots = kv_cache_update_mask.shape[-1]
        cos = torch.matmul(kv_cache_update_mask, self.cos).view(1, 1, HEAD_DIM, 1)
        sin = torch.matmul(kv_cache_update_mask, self.sin).view(1, 1, HEAD_DIM, 1)
        rotate = rotary(cos, sin)
        mask = key_padding_mask.view(1, 1, 1, slots)
        update = kv_cache_update_mask.view(1, 1, 1, slots)
        keep = 1 - update
        x = input_embeds
        keys, values = [], []
        for index, layer in enumerate(self.layers):
            channels = slice(index * KV_WIDTH, (index + 1) * KV_WIDTH)
            q, k, v = layer.qkv(x, rotate)
            key = self.key_cache[:, channels] * keep + k.view(1, KV_WIDTH, 1, 1) * update
            value = self.value_cache[:, channels] * keep + v.view(1, KV_WIDTH, 1, 1) * update
            x = layer.attend(
                x,
                q,
                key.view(1, KV_HEADS, HEAD_DIM, slots),
                _tokens_last(value.view(1, KV_HEADS, HEAD_DIM, slots)),
                mask,
            )
            x = layer.feed_forward(x)
            keys.append(key)
            values.append(value)
        normalized = unit(x, 1)
        logits = F.conv2d(normalized, self.head).view(1, 1, -1)
        self.key_cache[:] = torch.cat(keys, dim=1)
        self.value_cache[:] = torch.cat(values, dim=1)
        return logits, normalized * self.norm


class TalkerPrefill(nn.Module):
    """CodeDecoder `prefill`: the talker over a prompt of up to `tokens` positions from slot 0.

    Inputs: `input_embeds [1, 1024, 1, tokens]`, the prompt padded at the end;
    `key_padding_mask [tokens, tokens]`, causal, 0 visible and -1e4 hidden.
    Outputs: `logits [1, tokens, 3072]`, `hidden_states [1, 1024, 1, tokens]`, and the cache
    columns of every position, `key_cache`/`value_cache [1, 28672, 1, tokens]`.
    """

    def __init__(self, talker: nn.Module, tokens: int) -> None:
        super().__init__()
        config = talker.config
        self.layers = _layers(talker.model.layers)
        norm = talker.model.norm.weight.detach().float()
        self.head = nn.Parameter(_conv(talker.codec_head, norm, _HIDDEN_RMS))
        self.register_buffer("norm", norm[None, :, None, None] * _HIDDEN_RMS)
        cos, sin = rope_table(tokens, config.rope_theta)
        self.register_buffer("cos", cos.T.reshape(1, 1, HEAD_DIM, tokens))
        self.register_buffer("sin", sin.T.reshape(1, 1, HEAD_DIM, tokens))

    def forward(
        self, input_embeds: Tensor, key_padding_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = input_embeds.shape[-1]
        mask = key_padding_mask.view(1, 1, tokens, tokens).repeat(1, 1, 2, 1)
        rotate = rotary(self.cos, self.sin)
        x = input_embeds
        keys, values = [], []
        for layer in self.layers:
            q, k, v = layer.qkv(x, rotate, _PADDING_EPS)
            x = layer.feed_forward(layer.attend(x, q, k, _tokens_last(v), mask), _PADDING_EPS)
            keys.append(k.reshape(1, KV_WIDTH, 1, tokens))
            values.append(v.reshape(1, KV_WIDTH, 1, tokens))
        normalized = unit(x, 1, _PADDING_EPS)
        logits = F.conv2d(normalized, self.head).view(1, -1, tokens).transpose(1, 2)
        return logits, normalized * self.norm, torch.cat(keys, dim=1), torch.cat(values, dim=1)


class Predictor(nn.Module):
    """MultiCodeDecoder: the code predictor's 15 steps of one frame, sampling included.

    Inputs: `hidden_states [1, 1024, 1, 1]`, the talker's; `code_embeds [1, 1024, 1, 1]`, the
    talker's codec embedding of the frame's first code; `noise [15, 2048]`, standard Gumbel noise.
    Outputs: `codes [1, 15]`, codebooks 1-15 as whole numbers in fp16, and `codes_embeds
    [1, 1024, 1, 1]`, the sum of the predictor's embeddings of those codes.

    Each code is the argmax of `logits / temperature + noise` over the `top_k` largest logits,
    which draws from their softmax at that temperature (Gumbel-max).
    """

    def __init__(self, talker: nn.Module, temperature: float, top_k: int) -> None:
        super().__init__()
        predictor = talker.code_predictor
        theta = predictor.config.rope_theta
        self.layers = _layers(predictor.model.layers, PREDICTOR_SCALED_FROM)
        norm = predictor.model.norm.weight
        self.heads = nn.ParameterList(
            nn.Parameter(_conv(head, norm, _HIDDEN_RMS)) for head in predictor.lm_head
        )
        self.embeddings = nn.ParameterList(
            nn.Parameter(table.weight.detach().float()) for table in predictor.model.codec_embedding
        )
        cos, sin = rope_table(GROUPS + 1, theta)
        self.register_buffer("cos", cos.T.reshape(1, 1, HEAD_DIM, GROUPS + 1))
        self.register_buffer("sin", sin.T.reshape(1, 1, HEAD_DIM, GROUPS + 1))
        self.register_buffer("causal", torch.tensor([[0.0, _HIDE], [0.0, 0.0]]).repeat(2, 1).view(1, 1, 4, 2))
        self.register_buffer("rank", torch.arange(CODEBOOK, 0, -1, dtype=torch.float32).view(1, CODEBOOK))
        self.register_buffer("index", torch.arange(CODEBOOK, dtype=torch.float32).view(CODEBOOK, 1))
        self.inverse_temperature = 1 / temperature
        self.top_k = top_k

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
            keys.append(k)
            values.append(v)
        x = x[..., 1:]
        codes, embeds = [], []
        for group in range(GROUPS):
            hot = self._sample(
                F.conv2d(unit(x, 1), self.heads[group]).view(1, CODEBOOK), noise[group : group + 1]
            )
            codes.append(torch.matmul(hot, self.index))
            embed = torch.matmul(hot, self.embeddings[group]).view(1, HIDDEN, 1, 1)
            embeds.append(embed)
            if group == GROUPS - 1:
                break
            position = group + 2
            rotate = rotary(self.cos[..., position : position + 1], self.sin[..., position : position + 1])
            x = embed
            for index, layer in enumerate(self.layers):
                if index == PREDICTOR_SCALED_FROM:
                    x = x * (1 / PREDICTOR_SCALE)
                q, k, v = layer.qkv(x, rotate)
                keys[index] = torch.cat((keys[index], k), dim=-1)
                values[index] = torch.cat((values[index], v.view(1, KV_HEADS, 1, HEAD_DIM)), dim=2)
                x = layer.feed_forward(layer.attend(x, q, keys[index], values[index], None))
        return torch.cat(codes, dim=1), torch.stack(embeds).sum(0)

    def _sample(self, logits: Tensor, noise: Tensor) -> Tensor:
        """One-hot `[1, 2048]` of the Gumbel-max draw over the top-k logits; ties go to the lowest index."""
        threshold = torch.topk(logits, self.top_k, dim=-1).values[:, -1:]
        allowed = (logits >= threshold).to(logits.dtype)
        scores = logits * self.inverse_temperature + noise + (allowed - 1) * -_HIDE
        hot = (scores >= scores.max(-1, keepdim=True).values).to(logits.dtype)
        # fp16 ties are rare but real; keep only the tied entry with the lowest index.
        ranked = hot * self.rank
        return (ranked >= ranked.max(-1, keepdim=True).values).to(logits.dtype)


class ShuffleUpsample(nn.Module):
    """A causal `ConvTranspose1d(kernel=2s, stride=s)`, trimmed by `s` on the right, as a two-tap
    causal convolution to `s` times the channels followed by an interleave.

    Output `t * s + r` is `W[:, :, r] x[t] + W[:, :, r + s] x[t - 1]`. On the Neural Engine the
    decoder's transposed convolutions are wrong (stride 8: 49% RMS error against the reference,
    where the CPU in fp16 gives 0.9%) or not supported (stride 5); convolutions are exact.
    """

    def __init__(self, transposed: nn.ConvTranspose1d) -> None:
        super().__init__()
        weight = transposed.weight.detach().float()  # [in, out, 2s]
        stride = transposed.stride[0]
        inputs, outputs, kernel = weight.shape
        assert kernel == 2 * stride
        taps = torch.stack(
            (weight[:, :, stride:], weight[:, :, :stride]), dim=-1
        )  # [in, out, s, 2]: x[t-1], x[t]
        self.weight = nn.Parameter(taps.permute(1, 2, 0, 3).reshape(outputs * stride, inputs, 2))
        self.bias = nn.Parameter(transposed.bias.detach().float().repeat_interleave(stride))
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        batch, _, length = x.shape
        y = F.conv1d(F.pad(x, (1, 0)), self.weight, self.bias)
        return y.view(batch, -1, self.stride, length).transpose(2, 3).reshape(batch, -1, length * self.stride)


class SpeechDecoder(nn.Module):
    """SpeechDecoder: the 12 Hz speech tokenizer's decoder over a fixed number of frames.

    Input `audio_codes [1, 16, frames]` (int32, codebook-major); output `audio [1, 1, frames * 1920]`
    at 24 kHz. The decoder is causal, so frames padded at the end leave the audio of the frames
    before them untouched.

    The Neural Engine takes tensors up to 65536 wide, and the last two upsampling blocks are wider
    (80000 and 240000 for 125 frames). They run on `rows` overlapping segments stacked on the batch
    axis: each later segment starts `HALO` samples early, more than the 28 samples of history the
    remaining layers see, and the first starts at time 0 with the decoder's own zero padding.
    """

    HALO = 32
    ROWS = 4

    def __init__(self, decoder: nn.Module, frames: int) -> None:
        super().__init__()
        quantizer = decoder.quantizer
        first, rest = quantizer.rvq_first, quantizer.rvq_rest

        def projected(rvq: nn.Module) -> list[Tensor]:
            projection = rvq.output_proj.weight.detach().float()[:, :, 0]
            tables = []
            for layer in rvq.vq.layers:
                book = layer._codebook
                usage = book.cluster_usage.detach().float().clamp(min=book.epsilon)
                tables.append(book.embedding_sum.detach().float() / usage[:, None] @ projection.T)
            return tables

        tables = projected(first) + projected(rest)
        self.register_buffer("codebooks", torch.cat(tables))
        self.register_buffer(
            "offsets", (torch.arange(len(tables), dtype=torch.int32) * tables[0].shape[0])[:, None]
        )
        self.pre_conv = decoder.pre_conv
        transformer = decoder.pre_transformer
        config = transformer.config
        self.heads = config.num_attention_heads
        self.eps = config.rms_norm_eps
        self.head_dim = config.head_dim
        rms = config.hidden_size**0.5
        self.input_proj = nn.Parameter(transformer.input_proj.weight.detach().float()[:, :, None, None])
        self.input_bias = nn.Parameter(transformer.input_proj.bias.detach().float()[None, :, None, None])
        self.layers = nn.ModuleList()
        for layer in transformer.layers:
            attn, mlp = layer.self_attn, layer.mlp
            pre, post = layer.input_layernorm.weight, layer.post_attention_layernorm.weight
            attention_scale = layer.self_attn_layer_scale.scale.detach().float()[:, None, None, None]
            mlp_scale = layer.mlp_layer_scale.scale.detach().float()[:, None, None, None]
            block = nn.Module()
            block.qkv = nn.Parameter(
                torch.cat([_conv(proj, pre, rms) for proj in (attn.q_proj, attn.k_proj, attn.v_proj)])
            )
            block.o = nn.Parameter(_conv(attn.o_proj) * attention_scale)
            block.gate = nn.Parameter(_conv(mlp.gate_proj, post, rms))
            block.up = nn.Parameter(_conv(mlp.up_proj, post, rms))
            block.down = nn.Parameter(_conv(mlp.down_proj) * mlp_scale)
            self.layers.append(block)
        self.output_proj = nn.Parameter(_conv(transformer.output_proj, transformer.norm.weight, rms))
        self.output_bias = nn.Parameter(transformer.output_proj.bias.detach().float()[None, :, None, None])
        inv_freq = 1.0 / config.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float64) / self.head_dim
        )
        freqs = torch.arange(frames, dtype=torch.float64)[:, None] * inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos().float().T.reshape(1, 1, self.head_dim, frames))
        self.register_buffer("sin", emb.sin().float().T.reshape(1, 1, self.head_dim, frames))
        position = torch.arange(frames)
        window = position[None, :] > position[:, None] - config.sliding_window
        visible = (position[None, :] <= position[:, None]) & window
        self.register_buffer("mask", torch.where(visible, 0.0, _HIDE).view(1, 1, frames, frames))
        self.upsample = decoder.upsample
        # Input conv, four upsampling blocks (strides 8, 5, 4, 3), activation, output conv.
        blocks = list(decoder.decoder)
        for index in range(1, 5):
            blocks[index] = copy.deepcopy(blocks[index])
            blocks[index].block[1] = ShuffleUpsample(blocks[index].block[1].conv)
        self.narrow = nn.Sequential(*blocks[:3])
        self.wide = nn.Sequential(*blocks[3:])
        self.wide_rate = int(torch.tensor(config.upsample_rates[2:]).prod())

    def forward(self, audio_codes: Tensor) -> Tensor:
        hidden = F.embedding(audio_codes[0] + self.offsets, self.codebooks).sum(0)  # [frames, 512]
        return self.decode(hidden.T.unsqueeze(0))

    def decode(self, codes_sum: Tensor) -> Tensor:
        """`forward` from the sum of the codes' embeddings, `[1, 512, frames]`."""
        frames = codes_sum.shape[-1]
        hidden = self.pre_conv(codes_sum)  # [1, 1024, frames]
        x = F.conv2d(hidden.view(1, -1, 1, frames), self.input_proj) + self.input_bias
        width = self.heads * self.head_dim
        half = self.head_dim // 2
        for block in self.layers:
            qkv = F.conv2d(unit_eps(x, self.eps), block.qkv)
            q, k, v = (part.view(1, self.heads, self.head_dim, frames) for part in qkv.split(width, dim=1))
            q = q * self.cos + torch.cat((-q[:, :, half:], q[:, :, :half]), dim=2) * self.sin
            k = k * self.cos + torch.cat((-k[:, :, half:], k[:, :, :half]), dim=2) * self.sin
            scores = torch.matmul(q.transpose(2, 3), k) * self.head_dim**-0.5 + self.mask
            attended = torch.matmul(scores.softmax(-1), v.transpose(2, 3))  # [1, heads, frames, head_dim]
            x = x + F.conv2d(attended.transpose(2, 3).reshape(1, width, 1, frames), block.o)
            h = unit_eps(x, self.eps)
            x = x + F.conv2d(F.silu(F.conv2d(h, block.gate)) * F.conv2d(h, block.up), block.down)
        hidden = (F.conv2d(unit_eps(x, self.eps), self.output_proj) + self.output_bias).view(1, -1, frames)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        hidden = self.narrow(hidden)
        length = hidden.shape[-1]
        assert length % self.ROWS == 0
        row = length // self.ROWS
        segments = [hidden[..., : row + self.HALO]] + [
            hidden[..., index * row - self.HALO : (index + 1) * row] for index in range(1, self.ROWS)
        ]
        wide = self.wide(torch.cat(segments)).clamp(min=-1, max=1)  # [rows, 1, (row + halo) * rate]
        kept = row * self.wide_rate
        skip = self.HALO * self.wide_rate
        return torch.cat(
            [wide[:1, :, :kept]] + [wide[i : i + 1, :, skip:] for i in range(1, self.ROWS)], dim=-1
        )
