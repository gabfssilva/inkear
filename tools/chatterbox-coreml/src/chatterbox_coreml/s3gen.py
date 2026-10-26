"""S3Gen (token-to-mel flow and HiFT vocoder) rebuilt for fixed shapes and the Neural Engine.

Every stage takes right-padded inputs plus a 0/1 mask. Padded positions are re-zeroed where a
layer would otherwise let them reach valid ones (the encoder's right-looking lookahead conv);
attention masks padded keys. The flow then matches the unpadded reference at every valid frame.
"""

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

MEL_CHANNELS = 80
TOKEN_MEL_RATIO = 2
SAMPLES_PER_FRAME = 480
CFG_RATE = 0.7
MASKED = -3e4
TIME_EMBEDDING_DIM = 320
HARMONICS = 9


def time_embedding(t: float) -> np.ndarray:
    """The estimator's sinusoidal timestep embedding (scale 1000), (1, 320) in fp32.

    Its angles reach 1000 rad, beyond what fp16 resolves, so it is computed outside the model.
    """
    half = TIME_EMBEDDING_DIM // 2
    frequencies = np.exp(np.arange(half, dtype=np.float64) * -(math.log(10000) / (half - 1)))
    angles = 1000.0 * t * frequencies
    return np.concatenate([np.sin(angles), np.cos(angles)])[None].astype(np.float32)


def cosine_schedule(steps: int) -> np.ndarray:
    """The flow's timesteps t_0 = 0 … t_steps = 1 on the cosine schedule."""
    return 1 - np.cos(np.linspace(0, 1, steps + 1) * 0.5 * np.pi)
_QUERY_CHUNK = 256


def _conv2d(layer: nn.Module, scale: float = 1.0) -> nn.Conv2d:
    """A Linear or Conv1d as the equivalent Conv2d over (B, C, 1, S), weights times `scale`."""
    weight = layer.weight.detach()
    kernel = weight.shape[2] if weight.dim() == 3 else 1
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], (1, kernel), bias=layer.bias is not None)
    conv.weight = nn.Parameter(weight.reshape(weight.shape[0], weight.shape[1], 1, kernel) * scale)
    if layer.bias is not None:
        conv.bias = nn.Parameter(layer.bias.detach() * scale)
    return conv


def _channel_norm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.permute(0, 2, 3, 1), norm.normalized_shape, norm.weight, norm.bias, norm.eps).permute(0, 3, 1, 2)


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
    """One head: query (B, C, 1, Q), key (B, K, 1, C), value (B, D, 1, K), key_bias (B, K, 1, 1).

    Keys sit on the channel axis and queries go in chunks of `_QUERY_CHUNK`, the fastest
    arrangement found on the Neural Engine (1.4x over one (B, H, S, D) batched matmul). The key
    mask rides in the matmul as one more channel, a 1 on the query side and the bias on the key
    side, which saves a full-size add per chunk (another 9%).
    """
    rows, _, _, length = query.shape
    query = torch.cat([query, torch.ones(rows, 1, 1, length, dtype=query.dtype)], dim=1)
    key = torch.cat([key, key_bias], dim=3)
    chunks = []
    for chunk in query.split(_QUERY_CHUNK, 3):
        weights = torch.einsum("bchq,bkhc->bkhq", chunk, key).softmax(dim=1)
        chunks.append(torch.einsum("bkhq,bchk->bchq", weights, value))
    return torch.cat(chunks, dim=3)


class _RelativeAttention(nn.Module):
    """ESPnet relative-position self-attention (sinusoidal relative encoding) without rel_shift.

    The position term (q + v)·W_pos·pe(i − j) is bilinear in sinusoids of i and of j, since
    sin((i − j)ω) and cos((i − j)ω) expand into products of sin and cos of iω and jω. Each head then
    scores with one matmul of [q + u, rotate(W_posᵀ(q + v))] against [k, cos(jω), sin(jω)] instead
    of ESPnet's pad-reshape-slice skew, which the Neural Engine runs slowly. The result is the same.
    """

    def __init__(self, attention: nn.Module, length: int) -> None:
        super().__init__()
        self.heads, self.width = attention.h, attention.d_k
        model = self.heads * self.width
        scale = self.width**-0.5
        self.q = _conv2d(attention.linear_q, scale)
        self.k, self.v, self.o = _conv2d(attention.linear_k), _conv2d(attention.linear_v), _conv2d(attention.linear_out)
        self.register_buffer("bias_u", attention.pos_bias_u.detach().reshape(1, model, 1, 1) * scale)
        self.register_buffer("bias_v", attention.pos_bias_v.detach().reshape(1, model, 1, 1) * scale)
        # Per head, W_posᵀ: (model, width), its rows reordered as sin coefficients, then cos ones.
        position = attention.linear_pos.weight.detach().view(self.heads, self.width, model).transpose(1, 2)
        position = torch.cat([position[:, 0::2], position[:, 1::2]], dim=1)
        self.register_buffer("position", position.reshape(self.heads, model, self.width, 1, 1))
        frequencies = torch.exp(torch.arange(0, model, 2, dtype=torch.float64) * -(math.log(10000.0) / model))
        angles = torch.outer(torch.arange(length, dtype=torch.float64), frequencies)
        sin, cos = torch.sin(angles).float(), torch.cos(angles).float()
        self.register_buffer("sin", sin.T.reshape(1, -1, 1, length))
        self.register_buffer("cos", cos.T.reshape(1, -1, 1, length))
        self.register_buffer("keys", torch.cat([cos, sin], dim=1).reshape(1, length, 1, model))

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q(x), self.k(x), self.v(x)
        half = self.sin.shape[1]
        heads = []
        for h in range(self.heads):
            part = slice(h * self.width, (h + 1) * self.width)
            c = F.conv2d(q[:, part] + self.bias_v[:, part], self.position[h])
            a, b = c[:, :half], c[:, half:]
            query = torch.cat([q[:, part] + self.bias_u[:, part], a * self.sin + b * self.cos, b * self.sin - a * self.cos], dim=1)
            key = torch.cat([k[:, part].transpose(1, 3), self.keys], dim=3)
            heads.append(_attend(query, key, v[:, part], key_bias))
        return self.o(torch.cat(heads, dim=1))


class _Conformer(nn.Module):
    def __init__(self, layer: nn.Module, length: int) -> None:
        super().__init__()
        self.norm_mha, self.norm_ff = layer.norm_mha, layer.norm_ff
        self.attention = _RelativeAttention(layer.self_attn, length)
        self.w1, self.w2 = _conv2d(layer.feed_forward.w_1), _conv2d(layer.feed_forward.w_2)

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(_channel_norm(self.norm_mha, x), key_bias)
        return x + self.w2(F.silu(self.w1(_channel_norm(self.norm_ff, x))))


class FlowEncoder(nn.Module):
    """Speech tokens to the flow's conditioning mel `mu`.

    Inputs: `tokens` (1, N) int32, prompt tokens then generated ones, right-padded with 0;
    `mask` (1, N), 1 at real tokens. Output: `mu` (1, 80, 2N).
    """

    def __init__(self, flow: nn.Module, tokens: int) -> None:
        super().__init__()
        encoder = flow.encoder
        self.embedding = flow.input_embedding
        self.embed, self.embed_norm = _conv2d(encoder.embed.out[0]), encoder.embed.out[1]
        self.up_embed, self.up_embed_norm = _conv2d(encoder.up_embed.out[0]), encoder.up_embed.out[1]
        self.xscale = encoder.embed.pos_enc.xscale
        lookahead = encoder.pre_lookahead_layer
        self.lookahead1, self.lookahead2 = _conv2d(lookahead.conv1), _conv2d(lookahead.conv2)
        self.lookahead = lookahead.pre_lookahead_len
        self.layers = nn.ModuleList(_Conformer(layer, tokens) for layer in encoder.encoders)
        self.upsample = _conv2d(encoder.up_layer.conv)
        self.stride = encoder.up_layer.stride
        self.up_layers = nn.ModuleList(_Conformer(layer, self.stride * tokens) for layer in encoder.up_encoders)
        self.after_norm = encoder.after_norm
        self.projection = _conv2d(flow.encoder_proj)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encode(self.embedding(tokens.long()).transpose(1, 2).unsqueeze(2), mask)

    def encode(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """`forward` from the tokens' embeddings, (1, 512, 1, N)."""
        x = embeds * mask.view(1, 1, 1, -1)
        # Re-zeroing after the embedding keeps padding out of the lookahead conv's reach.
        x = _channel_norm(self.embed_norm, self.embed(x)) * self.xscale * mask.view(1, 1, 1, -1)
        x = x + self.lookahead2(F.pad(F.leaky_relu(self.lookahead1(F.pad(x, (0, self.lookahead)))), (2, 0)))
        key_bias = ((1 - mask) * MASKED).view(1, -1, 1, 1)
        for layer in self.layers:
            x = layer(x, key_bias)
        x = self.upsample(F.pad(x.repeat_interleave(self.stride, dim=3), (2 * self.stride, 0)))
        up_mask = mask.repeat_interleave(self.stride, dim=1)
        x = _channel_norm(self.up_embed_norm, self.up_embed(x)) * self.xscale * up_mask.view(1, 1, 1, -1)
        up_bias = ((1 - up_mask) * MASKED).view(1, -1, 1, 1)
        for layer in self.up_layers:
            x = layer(x, up_bias)
        return self.projection(_channel_norm(self.after_norm, x)).squeeze(2)


class _CausalBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        conv, _, self.norm, _, _ = block.block
        self.conv = _conv2d(conv)
        self.pad = conv.causal_padding[0]

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv(F.pad(x * mask, (self.pad, 0)))
        return F.mish(_channel_norm(self.norm, h)) * mask


class _Resnet(nn.Module):
    def __init__(self, resnet: nn.Module) -> None:
        super().__init__()
        self.block1, self.block2 = _CausalBlock(resnet.block1), _CausalBlock(resnet.block2)
        self.time = resnet.mlp[1]
        self.residual = _conv2d(resnet.res_conv)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask) + self.time(F.mish(t)).view(t.shape[0], -1, 1, 1)
        return self.block2(h, mask) + self.residual(x * mask)


class _Transformer(nn.Module):
    """diffusers' BasicTransformerBlock over (B, C, 1, S)."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        attention = block.attn1
        self.heads = attention.heads
        self.norm1, self.norm3 = block.norm1, block.norm3
        self.q = _conv2d(attention.to_q, attention.scale)
        self.k, self.v, self.o = _conv2d(attention.to_k), _conv2d(attention.to_v), _conv2d(attention.to_out[0])
        self.ff1, self.ff2 = _conv2d(block.ff.net[0].proj), _conv2d(block.ff.net[2])

    def forward(self, x: torch.Tensor, key_bias: torch.Tensor) -> torch.Tensor:
        h = _channel_norm(self.norm1, x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        width = q.shape[1] // self.heads
        heads = [
            _attend(qh, kh, vh, key_bias)
            for qh, kh, vh in zip(q.split(width, 1), k.transpose(1, 3).split(width, 3), v.split(width, 1), strict=True)
        ]
        x = x + self.o(torch.cat(heads, dim=1))
        return x + self.ff2(F.gelu(self.ff1(_channel_norm(self.norm3, x))))


class FlowStep(nn.Module):
    """One Euler step of the flow ODE with classifier-free guidance (rate 0.7) folded in.

    Inputs: `x`, `mu`, `cond` (1, 80, M); `spks` (1, 80); `mask` (1, 1, M); `t_emb` (1, 320) from
    `time_embedding(t)`; `dt` (1,). Output: `x_next` = x + dt · (1.7 v(cond) − 0.7 v(uncond)).
    The two guidance rows run as a batch in the (B, C, 1, S) layout the Neural Engine prefers.
    """

    def __init__(self, estimator: nn.Module) -> None:
        super().__init__()
        self.time = estimator.time_mlp
        (resnet, transformers, downsample), = estimator.down_blocks
        self.down = _Resnet(resnet)
        self.down_transformers = nn.ModuleList(_Transformer(b) for b in transformers)
        self.downsample = _conv2d(downsample)
        self.mid = nn.ModuleList(_Resnet(resnet) for resnet, _ in estimator.mid_blocks)
        self.mid_transformers = nn.ModuleList(
            nn.ModuleList(_Transformer(b) for b in transformers) for _, transformers in estimator.mid_blocks
        )
        (resnet, transformers, upsample), = estimator.up_blocks
        self.up = _Resnet(resnet)
        self.up_transformers = nn.ModuleList(_Transformer(b) for b in transformers)
        self.upsample = _conv2d(upsample)
        self.final = _CausalBlock(estimator.final_block)
        self.projection = _conv2d(estimator.final_proj)

    def forward(self, x, mu, cond, spks, mask, t_emb, dt):
        frames = x.shape[-1]
        rows_mask = torch.cat([mask, mask]).unsqueeze(2)
        key_bias = ((1 - rows_mask) * MASKED).view(2, frames, 1, 1)
        t = self.time(t_emb).expand(2, -1)
        h = torch.cat(
            [
                torch.cat([x, x]),
                torch.cat([mu, torch.zeros_like(mu)]),
                torch.cat([spks, torch.zeros_like(spks)]).unsqueeze(-1).expand(2, MEL_CHANNELS, frames),
                torch.cat([cond, torch.zeros_like(cond)]),
            ],
            dim=1,
        ).unsqueeze(2)

        def stage(resnet: _Resnet, transformers: nn.ModuleList, h: torch.Tensor) -> torch.Tensor:
            h = resnet(h, rows_mask, t)
            for block in transformers:
                h = block(h, key_bias)
            return h

        skip = stage(self.down, self.down_transformers, h)
        h = self.downsample(F.pad(skip * rows_mask, (self.downsample.kernel_size[1] - 1, 0)))
        for resnet, transformers in zip(self.mid, self.mid_transformers, strict=True):
            h = stage(resnet, transformers, h)
        h = stage(self.up, self.up_transformers, torch.cat([h, skip], dim=1))
        h = self.upsample(F.pad(h * rows_mask, (self.upsample.kernel_size[1] - 1, 0)))
        v = (self.projection(self.final(h, rows_mask) * rows_mask) * rows_mask).squeeze(2)
        guided = (1 + CFG_RATE) * v[:1] - CFG_RATE * v[1:]
        return x + dt.view(1, 1, 1) * guided


class F0(nn.Module):
    """HiFT's f0 predictor. Input `mel` (1, 80, T); output `f0` (1, T) in Hz."""

    def __init__(self, hift: nn.Module) -> None:
        super().__init__()
        self.predictor = hift.f0_predictor

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.predictor(mel)


class _PhaseConv(nn.Module):
    """A dilated conv run undilated over its `dilation` interleaved phases, stacked as a batch.

    The Neural Engine rejects dilation 5; the result is identical.
    """

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        self.dilation = conv.dilation[0]
        self.conv = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size[0], padding=conv.padding[0] // self.dilation)
        self.conv.load_state_dict(conv.state_dict())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels, length = x.shape[1], x.shape[2]
        x = F.pad(x, (0, -length % self.dilation))
        phases = x.view(channels, -1, self.dilation).permute(2, 0, 1)
        y = self.conv(phases)
        return y.permute(1, 2, 0).reshape(1, y.shape[1], -1)[:, :, :length]


class _PolyphaseConv(nn.Module):
    """A strided conv whose kernel spans whole strides, run as an unstrided conv over the
    `stride` input phases folded into channels. The Neural Engine rejects kernel 30, stride 15.
    """

    def __init__(self, conv: nn.Conv1d) -> None:
        super().__init__()
        self.stride, self.padding = conv.stride[0], conv.padding[0]
        out_channels, in_channels, kernel = conv.weight.shape
        self.taps = kernel // self.stride
        self.conv = nn.Conv1d(in_channels * self.stride, out_channels, self.taps)
        weight = conv.weight.detach().view(out_channels, in_channels, self.taps, self.stride).permute(0, 1, 3, 2)
        self.conv.weight = nn.Parameter(weight.reshape(out_channels, in_channels * self.stride, self.taps))
        self.conv.bias = nn.Parameter(conv.bias.detach().clone())
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels, length = x.shape[1], x.shape[2]
        frames = (length + 2 * self.padding - self.kernel) // self.stride + 1
        span = self.stride * (frames + self.taps - 1)
        x = F.pad(x, (self.padding, span - length - self.padding))
        return self.conv(x.view(channels, -1, self.stride).permute(0, 2, 1).reshape(1, channels * self.stride, -1))


class _PolyphaseTransposedConv(nn.Module):
    """A transposed conv as one ordinary conv that emits every output phase as channels, then
    interleaves them, giving stride · L outputs for L inputs.

    The Neural Engine rejects kernel 11 at stride 5 and kernel 7 at stride 3, and runs the
    transposed convs it accepts with far less precision: HiFT's first upsampler alone drops from
    63 to 35 dB SNR, and the waveform to 21 dB.
    """

    def __init__(self, conv: nn.ConvTranspose1d) -> None:
        super().__init__()
        stride, kernel, padding = conv.stride[0], conv.kernel_size[0], conv.padding[0]
        weight = conv.weight.detach()  # (in, out, kernel)
        taps = {(r, k): (r + padding - k) // stride for r in range(stride) for k in range(kernel) if (r + padding - k) % stride == 0}
        low, high = min(taps.values()), max(taps.values())
        phases = torch.zeros(stride, weight.shape[1], weight.shape[0], high - low + 1)
        for (r, k), offset in taps.items():
            phases[r, :, :, offset - low] = weight[:, :, k].T
        self.stride, self.out_channels = stride, weight.shape[1]
        self.conv = nn.Conv1d(weight.shape[0], stride * weight.shape[1], high - low + 1, bias=conv.bias is not None)
        self.conv.weight = nn.Parameter(phases.reshape(stride * weight.shape[1], weight.shape[0], -1))
        if conv.bias is not None:
            self.conv.bias = nn.Parameter(conv.bias.detach().repeat(stride))
        self.padding = (-low, high)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(F.pad(x, self.padding))  # (1, stride·out, L), phase-major
        length = y.shape[-1]
        return y.view(self.stride, self.out_channels, length).permute(1, 2, 0).reshape(1, self.out_channels, -1)


class _Snake(nn.Module):
    """Snake, x + sin²(αx)/α, with 1/α kept in fp16 range.

    Some channels have |α| near 1e-5, where 1/α overflows fp16 and Core ML keeps the product in
    fp32, off the Neural Engine. Below `floor` the term is replaced by its limit α·x², which is
    off by (αx)²/3 relative to a term that is itself near zero.
    """

    def __init__(self, snake: nn.Module, floor: float = 1e-3) -> None:
        super().__init__()
        alpha = snake.alpha.detach()
        small = alpha.abs() < floor
        self.register_buffer("alpha", alpha.view(1, -1, 1))
        self.register_buffer("inverse", torch.where(small, 0.0, 1.0 / (alpha + snake.no_div_by_zero)).view(1, -1, 1))
        self.register_buffer("limit", torch.where(small, alpha, 0.0).view(1, -1, 1))
        self.linear = bool(small.any())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.inverse * torch.sin(x * self.alpha).pow(2)
        return y + self.limit * x * x if self.linear else y


def _ane_friendly(hift: nn.Module) -> nn.Module:
    hift = copy.deepcopy(hift)
    for module in hift.modules():
        if parametrize.is_parametrized(module):
            for name in list(module.parametrizations.keys()):
                parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    for blocks in (hift.resblocks, hift.source_resblocks):
        for block in blocks:
            block.convs1 = nn.ModuleList(_PhaseConv(c) if c.dilation[0] > 4 else c for c in block.convs1)
            block.activations1 = nn.ModuleList(_Snake(a) for a in block.activations1)
            block.activations2 = nn.ModuleList(_Snake(a) for a in block.activations2)
    hift.source_downs = nn.ModuleList(
        _PolyphaseConv(c) if c.kernel_size[0] > 16 else c for c in hift.source_downs
    )
    hift.ups = nn.ModuleList(
        _PolyphaseTransposedConv(c) for c in hift.ups
    )
    return hift


class HiFT(nn.Module):
    """HiFT's waveform decoder with a matmul STFT and iSTFT (n_fft 16, hop 4).

    Inputs: `mel` (1, 80, T); `source` (1, 1, 480·T), the harmonic source from `source(f0)`.
    Output: `audio` (1, 480·T).
    """

    def __init__(self, hift: nn.Module) -> None:
        super().__init__()
        self.hift = _ane_friendly(hift)
        n_fft, hop = hift.istft_params["n_fft"], hift.istft_params["hop_len"]
        self.n_fft, self.hop = n_fft, hop
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
        bins = n_fft // 2 + 1
        angles = 2 * math.pi * torch.outer(torch.arange(bins, dtype=torch.float64), torch.arange(n_fft, dtype=torch.float64)) / n_fft
        self.register_buffer("analysis", torch.cat([torch.cos(angles), -torch.sin(angles)]).mul(window).unsqueeze(1).float())
        scale = torch.full((bins, 1), 2.0, dtype=torch.float64)
        scale[0] = scale[-1] = 1.0
        synthesis = torch.cat([torch.cos(angles), -torch.sin(angles)]) * torch.cat([scale, scale]) / n_fft
        overlap_add = nn.ConvTranspose1d(2 * bins, 1, n_fft, hop, padding=n_fft // 2, bias=False)
        overlap_add.weight = nn.Parameter((synthesis * window).unsqueeze(1).float())
        self.overlap_add = _PolyphaseTransposedConv(overlap_add)
        self.register_buffer("window_square", (window**2).view(1, 1, -1).float())

    def forward(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        hift = self.hift
        pad = self.n_fft // 2
        spectrum = F.conv1d(F.pad(source, (pad, pad), mode="reflect"), self.analysis, stride=self.hop)
        x = hift.conv_pre(mel)
        for i in range(hift.num_upsamples):
            x = hift.ups[i](F.leaky_relu(x, hift.lrelu_slope))
            if i == hift.num_upsamples - 1:
                x = hift.reflection_pad(x)
            x = x + hift.source_resblocks[i](hift.source_downs[i](spectrum))
            x = sum(hift.resblocks[i * hift.num_kernels + j](x) for j in range(hift.num_kernels)) / hift.num_kernels
        x = hift.conv_post(F.leaky_relu(x))
        bins = self.n_fft // 2 + 1
        magnitude = torch.clamp(torch.exp(x[:, :bins]), max=1e2)
        phase = torch.sin(x[:, bins:])
        frames = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        length = self.hop * (frames.shape[-1] - 1)
        audio = self.overlap_add(frames)[:, 0, :length]
        ones = torch.ones(1, 1, frames.shape[-1], dtype=frames.dtype)
        norm = F.conv_transpose1d(ones, self.window_square, stride=self.hop)[:, 0, pad : pad + length]
        return torch.clamp(audio / norm.clamp(min=1e-7), -hift.audio_limit, hift.audio_limit)


def source(f0: np.ndarray, weight: np.ndarray, bias: float, rng: np.random.Generator) -> np.ndarray:
    """HiFT's neural-source excitation for per-frame `f0` (T,), as (480·T,) float32.

    Phases accumulate in float64: in fp16 the running phase would lose all precision within a
    second. `weight` (9,) and `bias` are the source module's `l_linear` parameters.
    """
    sample_f0 = np.repeat(f0.astype(np.float64), SAMPLES_PER_FRAME)
    harmonics = np.arange(1, HARMONICS + 1, dtype=np.float64)[:, None]
    phase = np.cumsum(sample_f0[None] * harmonics / 24_000, axis=1) % 1
    offsets = rng.uniform(-np.pi, np.pi, size=(HARMONICS, 1))
    offsets[0] = 0
    sines = 0.1 * np.sin(2 * np.pi * phase + offsets)
    voiced = (sample_f0 > 10).astype(np.float64)[None]
    noise = (voiced * 0.003 + (1 - voiced) * 0.1 / 3) * rng.standard_normal(sines.shape)
    excitation = sines * voiced + noise
    return np.tanh(np.einsum("h,hs->s", weight.astype(np.float64), excitation) + bias).astype(np.float32)
