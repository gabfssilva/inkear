"""Traceable rewrites of the OmniVoice backbone step and the audio decoder, shaped for the Neural Engine."""

import itertools
import math
import operator
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from omnivoice.models.omnivoice import OmniVoice
from torch import nn
from transformers import HiggsAudioV2TokenizerModel

MASK_ID = 1024
CODEBOOKS = 8
VOCAB = 1025
GUIDANCE = 2.0
# Added to the scores of padding keys; finite, so fp16 never sees inf - inf.
BLOCKED = -1e4


@dataclass(frozen=True)
class Bucket:
    """A backbone step specialised for `cond` conditional positions and `target` audio frames.

    The step runs one sequence of `cond + target` positions: the conditional segment
    (style, text, reference, target) right-aligned in `[0, cond)` and the unconditional
    segment (target only) right-aligned in `[cond, cond + target)`. Right alignment puts the
    target frames of both segments at the end of the windows `[cond - target, cond)` and
    `[cond, cond + target)`, which are contiguous.
    """

    cond: int
    target: int

    @property
    def sequence(self) -> int:
        return self.cond + self.target

    @property
    def segments(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (0, self.cond), (self.cond, self.sequence)

    @property
    def name(self) -> str:
        return f"c{self.cond}_t{self.target}"


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over the channel axis of [1, C, 1, S], computed on `x / max|x|`.

    The residual stream reaches ~1.3e4, so its mean square (~2.7e5) overflows fp16; RMSNorm is
    scale invariant, so normalising by the max first changes nothing but the eps term.
    """
    peak = x.abs().amax(1, keepdim=True)
    unit = x / peak
    variance = unit.pow(2).mean(1, keepdim=True) + (math.sqrt(eps) / peak).pow(2)
    return unit * torch.rsqrt(variance) * weight


class Layer(nn.Module):
    """One bidirectional Qwen3 decoder layer in the Neural Engine's [1, C, 1, S] layout.

    Attention runs per head and per segment with the keys on the channel axis and the softmax
    over it, as in apple/ml-ane-transformers: ~30% faster per layer than [1, heads, S, d]
    matmuls with a softmax over the last axis.
    """

    def __init__(self, layer: nn.Module, eps: float) -> None:
        super().__init__()
        attention = layer.self_attn
        self.q, self.k, self.v, self.o = (
            _pointwise(attention.q_proj),
            _pointwise(attention.k_proj),
            _pointwise(attention.v_proj),
            _pointwise(attention.o_proj),
        )
        self.gate, self.up, self.down = (
            _pointwise(layer.mlp.gate_proj),
            _pointwise(layer.mlp.up_proj),
            _pointwise(layer.mlp.down_proj),
        )
        self.register_buffer("q_norm", attention.q_norm.weight.detach().view(1, -1, 1, 1))
        self.register_buffer("k_norm", attention.k_norm.weight.detach().view(1, -1, 1, 1))
        self.register_buffer("input_norm", layer.input_layernorm.weight.detach().view(1, -1, 1, 1))
        self.register_buffer("post_norm", layer.post_attention_layernorm.weight.detach().view(1, -1, 1, 1))
        self.heads = attention.config.num_attention_heads
        self.kv_heads = attention.config.num_key_value_heads
        self.head_dim = attention.head_dim
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        biases: tuple[torch.Tensor, ...],
        bucket: Bucket,
    ) -> torch.Tensor:
        h = rms_norm(x, self.input_norm, self.eps)
        x = x + self.attend(h, cos, sin, biases, bucket)
        h = rms_norm(x, self.post_norm, self.eps)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def attend(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        biases: tuple[torch.Tensor, ...],
        bucket: Bucket,
    ) -> torch.Tensor:
        d = self.head_dim
        q, k, v = self.q(x), self.k(x), self.v(x)

        def head(projection: torch.Tensor, index: int, weight: torch.Tensor) -> torch.Tensor:
            return _rotate(
                rms_norm(projection[:, d * index : d * (index + 1)], weight, self.eps), cos, sin, d // 2
            )

        keys = [head(k, j, self.k_norm) for j in range(self.kv_heads)]
        values = [v[:, d * j : d * (j + 1)] for j in range(self.kv_heads)]
        group = self.heads // self.kv_heads
        outputs = []
        for i in range(self.heads):
            query, key, value = head(q, i, self.q_norm), keys[i // group], values[i // group]
            # The segments never attend to each other, so each gets its own attention.
            outputs.append(
                torch.cat(
                    [
                        _attention(query[..., a:b], key[..., a:b], value[..., a:b], bias, d)
                        for (a, b), bias in zip(bucket.segments, biases, strict=True)
                    ],
                    dim=3,
                )
            )
        return self.o(torch.cat(outputs, dim=1))


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, d: int) -> torch.Tensor:
    """q, k, v [1, d, 1, n]; `bias` [1, n, 1, 1] blocks padding keys. Scores are [1, keys, 1, queries]."""
    scores = torch.einsum("bchq,bkhc->bkhq", q, k.transpose(1, 3)) * d**-0.5 + bias
    return torch.einsum("bkhq,bchk->bchq", scores.softmax(dim=1), v)


def _log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """log_softmax spelled out: the Neural Engine's own computes log(softmax), which underflows in
    fp16 and returns -45440 for log-probabilities below ~-17, and guidance turns those into winners."""
    shifted = x - x.amax(dim, keepdim=True)
    return shifted - torch.log(torch.exp(shifted).sum(dim, keepdim=True))


def _pointwise(linear: nn.Linear) -> nn.Conv2d:
    conv = nn.Conv2d(linear.in_features, linear.out_features, 1, bias=False)
    conv.weight.data = linear.weight.detach().view(linear.out_features, linear.in_features, 1, 1)
    return conv


class Step(nn.Module):
    """One unmasking step: backbone over both segments, then guided log-probabilities per frame.

    Inputs: `input_ids` [1, 8, S] int32, `audio_mask` [1, S] (1 at audio positions) and
    `valid` [1, S] (1 at real positions, 0 at padding). Outputs over the target window,
    [1, 8, target]: `tokens`, the most likely non-mask code, and `scores`, its guided
    log-probability. With `float_tokens`, `tokens` is Float16 and comes from no integer op:
    Core AI mislabels the integer argmax of the Neural Engine.
    """

    def __init__(self, model: OmniVoice, bucket: Bucket, float_tokens: bool = False) -> None:
        super().__init__()
        self.float_tokens = float_tokens
        llm = model.llm
        eps = llm.config.rms_norm_eps
        self.bucket = bucket
        self.text_embedding = llm.embed_tokens
        self.audio_embedding = model.audio_embeddings
        self.audio_heads = _pointwise(model.audio_heads)
        self.layers = nn.ModuleList(Layer(layer, eps) for layer in llm.layers)
        self.register_buffer("norm", llm.norm.weight.detach().view(1, -1, 1, 1))
        self.eps = eps
        self.register_buffer(
            "offsets", (torch.arange(CODEBOOKS) * VOCAB).view(1, CODEBOOKS, 1).to(torch.int32)
        )
        positions = torch.arange(bucket.sequence, dtype=torch.float32)
        cos, sin = llm.rotary_emb(positions.view(1, 1, -1), positions.view(1, -1).long())
        self.register_buffer("cos", cos[0].T.reshape(1, -1, 1, bucket.sequence))
        self.register_buffer("sin", sin[0].T.reshape(1, -1, 1, bucket.sequence))

    def forward(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio = audio_mask.view(1, -1, 1)
        text = self.text_embedding(input_ids[:, 0])
        sound = self.audio_embedding(
            input_ids * audio_mask.to(torch.int32).view(1, 1, -1) + self.offsets
        ).sum(1)
        x = (text + audio * (sound - text)).transpose(1, 2).unsqueeze(2)
        bias = ((1 - valid) * BLOCKED).view(1, -1, 1, 1)
        biases = tuple(bias[:, a:b] for a, b in self.bucket.segments)
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, biases, self.bucket)
        cond, target = self.bucket.cond, self.bucket.target
        window = rms_norm(x[..., cond - target : cond + target], self.norm, self.eps)
        logits = self.audio_heads(window).view(1, CODEBOOKS, VOCAB, 2 * target)
        c = _log_softmax(logits[..., :target], dim=2)
        u = _log_softmax(logits[..., target:], dim=2)
        guided = _log_softmax(c + GUIDANCE * (c - u), dim=2)[:, :, :MASK_ID]
        if self.float_tokens:
            scores = guided.amax(dim=2)
            # The last position holding the maximum, each position counted from 1.
            positions = torch.arange(1, MASK_ID + 1, dtype=guided.dtype).view(1, 1, -1, 1)
            hits = (guided == scores.unsqueeze(2)).to(guided.dtype)
            return (hits * positions).amax(dim=2) - 1, scores
        scores, tokens = guided.max(dim=2)
        # Nothing may follow the argmax: an int32 op after it makes the Neural Engine compiler
        # reject the whole program ("Cannot validate the MIL program").
        return tokens.to(torch.int32), scores


class Decoder(nn.Module):
    """Audio codes [1, 8, frames] int32 to a 24 kHz waveform [1, frames·960].

    The Neural Engine rejects tensors longer than ~16k along a dimension, and the waveform of a
    sentence is ~200k samples. So the signal is folded into rows of `ROW_FRAMES` frames,
    [1, C, rows, width], and every convolution takes the samples it needs across a row edge
    from the neighbouring rows (`_halo`), which keeps the result identical to the 1-D decoder.
    `frame_mask` [1, frames] zeroes the codes past the utterance, as the reference decoder's
    zero padding would.
    """

    def __init__(self, tokenizer: HiggsAudioV2TokenizerModel, frames: int) -> None:
        super().__init__()
        if frames % ROW_FRAMES:
            raise ValueError(f"{frames} frames is not a multiple of {ROW_FRAMES}")
        quantizers = tokenizer.quantizer.quantizers
        self.codebooks = nn.ParameterList(q.codebook.embed for q in quantizers)
        self.projections = nn.ModuleList(q.project_out for q in quantizers)
        self.fc2 = tokenizer.fc2
        decoder = tokenizer.acoustic_decoder
        rows = frames // ROW_FRAMES
        widths = itertools.accumulate(
            (b.conv_t1.stride[0] for b in decoder.block), operator.mul, initial=ROW_FRAMES
        )
        self.conv1 = _Conv(decoder.conv1, rows)
        self.blocks = nn.Sequential(
            *(_Block(block, rows, width) for block, width in zip(decoder.block, widths, strict=False))
        )
        self.snake = _Snake(decoder.snake1.alpha)
        self.conv2 = _Conv(decoder.conv2, rows)
        self.rows = rows

    def forward(self, codes: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
        quantized = sum(
            projection(F.embedding(codes[:, i], codebook))
            for i, (codebook, projection) in enumerate(zip(self.codebooks, self.projections, strict=True))
        )
        acoustic = self.fc2(quantized) * frame_mask.view(1, -1, 1)
        x = acoustic.view(1, self.rows, ROW_FRAMES, -1).permute(0, 3, 1, 2)
        x = self.blocks(self.conv1(x))
        return self.conv2(self.snake(x)).view(1, -1)


ROW_FRAMES = 16


class _Snake(nn.Module):
    """DAC's snake, x + sin(αx)²/α, rewritten so no constant leaves the fp16 range (|α| reaches 4e-7)."""

    def __init__(self, alpha: torch.Tensor) -> None:
        super().__init__()
        a = alpha.detach().view(1, -1, 1, 1)
        self.register_buffer("alpha", a)
        self.register_buffer("sign", torch.where(a < 0, -1.0, 1.0))
        self.register_buffer("scale", (a.abs() + 1e-9).rsqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sign * (torch.sin(self.alpha * x) * self.scale).pow(2)


class _Conv(nn.Module):
    """A same-padded Conv1d applied to folded rows."""

    def __init__(self, conv: nn.Conv1d, rows: int) -> None:
        super().__init__()
        self.rows = rows
        self.halo = conv.dilation[0] * (conv.kernel_size[0] - 1) // 2
        self.conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, (1, conv.kernel_size[0]), dilation=(1, conv.dilation[0])
        )
        self.conv.weight.data = conv.weight.detach().unsqueeze(2)
        self.conv.bias.data = conv.bias.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_halo(x, self.halo, self.rows) if self.halo else x)


class _Upsample(nn.Module):
    """DAC's ConvTranspose1d (kernel 2s, stride s, padding ⌈s/2⌉, output padding s mod 2) in polyphase form.

    Output sample m·s + r depends on inputs m-1, m and m+1 only, so the transposed conv is a
    kernel-3 conv with s·C_out channels, one group per phase r, interleaved along the row.
    """

    def __init__(self, conv: nn.ConvTranspose1d, rows: int, width: int) -> None:
        super().__init__()
        stride, padding = conv.stride[0], conv.padding[0]
        weight = conv.weight.detach().transpose(0, 1)  # [C_out, C_in, 2s]
        out_channels, in_channels = weight.shape[:2]
        taps = torch.zeros(stride, out_channels, in_channels, 3)
        for phase in range(stride):
            k = phase + padding
            if k < stride:
                taps[phase, :, :, 0], taps[phase, :, :, 1] = weight[..., k + stride], weight[..., k]
            else:
                taps[phase, :, :, 1], taps[phase, :, :, 2] = weight[..., k], weight[..., k - stride]
        self.conv = nn.Conv2d(in_channels, stride * out_channels, (1, 3))
        self.conv.weight.data = taps.reshape(stride * out_channels, in_channels, 1, 3)
        self.conv.bias.data = conv.bias.detach().repeat(stride)
        self.stride, self.out_channels, self.rows, self.width = stride, out_channels, rows, width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phases = self.conv(_halo(x, 1, self.rows)).view(self.stride, self.out_channels, self.rows, self.width)
        return phases.permute(1, 2, 3, 0).reshape(1, self.out_channels, self.rows, self.width * self.stride)


class _Residual(nn.Module):
    def __init__(self, unit: nn.Module, rows: int) -> None:
        super().__init__()
        self.snake1, self.conv1 = _Snake(unit.snake1.alpha), _Conv(unit.conv1, rows)
        self.snake2, self.conv2 = _Snake(unit.snake2.alpha), _Conv(unit.conv2, rows)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.snake2(self.conv1(self.snake1(x))))


class _Block(nn.Module):
    def __init__(self, block: nn.Module, rows: int, width: int) -> None:
        super().__init__()
        self.snake = _Snake(block.snake1.alpha)
        self.upsample = _Upsample(block.conv_t1, rows, width)
        self.units = nn.Sequential(
            *(_Residual(unit, rows) for unit in (block.res_unit1, block.res_unit2, block.res_unit3))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.units(self.upsample(self.snake(x)))


def _halo(x: torch.Tensor, size: int, rows: int) -> torch.Tensor:
    """Extend each of the `rows` rows with `size` samples from the rows around it (zeros past the ends)."""
    padded = F.pad(x, (0, 0, 1, 1))
    return torch.cat([padded[:, :, :rows, -size:], x, padded[:, :, 2:, :size]], dim=3)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, half: int) -> torch.Tensor:
    """RoPE on the channel axis of one head, [1, d, 1, S]."""
    return x * cos + torch.cat((-x[:, half:], x[:, :half]), dim=1) * sin
