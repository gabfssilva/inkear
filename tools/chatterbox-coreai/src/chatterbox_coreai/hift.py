"""HiFT for Core AI: chatterbox-coreml's HiFT with every fold of time into channels or batch at an
aligned length.

MPSGraph moves a padded tensor reshaped to a length that is not a multiple of 32 off the Neural
Engine, to the GPU, which iOS denies to apps in the background. The polyphase and dilated
convolutions pad their input further, to such a length, and crop what that adds.
"""

import math

import torch
from chatterbox_coreml import s3gen
from torch import nn
from torch.nn import functional as F

ALIGN = 32


def _aligned(length: int) -> int:
    return math.ceil(length / ALIGN) * ALIGN


class _PolyphaseConv(nn.Module):
    def __init__(self, source: s3gen._PolyphaseConv) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.source
        channels, length = x.shape[1], x.shape[2]
        frames = (length + 2 * s.padding - s.kernel) // s.stride + 1
        folded = _aligned(frames + s.taps - 1)
        x = F.pad(x, (s.padding, s.stride * folded - length - s.padding))
        y = s.conv(x.view(channels, folded, s.stride).permute(0, 2, 1).reshape(1, channels * s.stride, folded))
        return y[:, :, :frames]


class _PhaseConv(nn.Module):
    def __init__(self, source: s3gen._PhaseConv) -> None:
        super().__init__()
        self.source = source

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.source
        channels, length = x.shape[1], x.shape[2]
        folded = _aligned(math.ceil(length / s.dilation))
        x = F.pad(x, (0, s.dilation * folded - length))
        y = s.conv(x.view(channels, folded, s.dilation).permute(2, 0, 1))
        return y.permute(1, 2, 0).reshape(1, y.shape[1], -1)[:, :, :length]


def for_neural_engine(hift: s3gen.HiFT) -> s3gen.HiFT:
    """`hift` with its polyphase and dilated convolutions folding at aligned lengths, in place."""
    inner = hift.hift
    inner.source_downs = nn.ModuleList(
        _PolyphaseConv(c) if isinstance(c, s3gen._PolyphaseConv) else c for c in inner.source_downs
    )
    for block in (*inner.resblocks, *inner.source_resblocks):
        block.convs1 = nn.ModuleList(_PhaseConv(c) if isinstance(c, s3gen._PhaseConv) else c for c in block.convs1)
    return hift


class HiFT(nn.Module):
    """chatterbox-coreml's `s3gen.HiFT` with even widths: the Neural Engine compiler fails on the
    odd ones HiFT has, 120 · T + 1 spectrum frames and as many samples after its reflection pad
    ("'memref.transpose' op result type ... is not equivalent to the canonical transposed input
    type"), and MPSGraph then runs the function on the GPU.

    A spectrum frame more, over a hop of zeros, and a zero frame appended to the last stage change
    the last ~10 ms of the window, which the caller drops: chatterbox-coreml keeps only the middle
    of each window, and drops the last token's audio at the end of a text.
    """

    def __init__(self, model: s3gen.HiFT, frames: int) -> None:
        super().__init__()
        self.model = for_neural_engine(model)
        # The overlap-add's window normalization depends only on the size, and MPSGraph runs it on
        # the GPU when the model computes it.
        pad, hop, length = model.n_fft // 2, model.hop, frames * s3gen.SAMPLES_PER_FRAME
        spectrum_frames = length // hop + 2
        ones = torch.ones(1, 1, spectrum_frames)
        norm = F.conv_transpose1d(ones, model.window_square, stride=hop)[:, 0, pad : pad + length]
        self.register_buffer("inverse_norm", 1 / norm.clamp(min=1e-7))

    def forward(self, mel: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        model, hift = self.model, self.model.hift
        pad = model.n_fft // 2
        length = source.shape[-1]
        # One hop of zeros more makes the convolution give the even frame count directly.
        padded = F.pad(F.pad(source, (pad, pad), mode="reflect"), (0, model.hop))
        spectrum = F.conv1d(padded, model.analysis, stride=model.hop)
        x = hift.conv_pre(mel)
        for i in range(hift.num_upsamples):
            x = hift.ups[i](F.leaky_relu(x, hift.lrelu_slope))
            if i == hift.num_upsamples - 1:
                x = F.pad(hift.reflection_pad(x), (0, 1))
            x = x + hift.source_resblocks[i](hift.source_downs[i](spectrum))
            x = sum(hift.resblocks[i * hift.num_kernels + j](x) for j in range(hift.num_kernels)) / hift.num_kernels
        x = hift.conv_post(F.leaky_relu(x))
        bins = model.n_fft // 2 + 1
        magnitude = torch.clamp(torch.exp(x[:, :bins]), max=1e2)
        phase = torch.sin(x[:, bins:])
        frames = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        audio = model.overlap_add(frames)[:, 0, :length]
        return torch.clamp(audio * self.inverse_norm, -hift.audio_limit, hift.audio_limit)


class F0(nn.Module):
    """chatterbox-coreml's `s3gen.F0` without its final `abs`, which the caller applies: any
    operation after the predictor's one-channel head makes MPSGraph run the whole function on the
    GPU. Input `mel` (1, 80, T); output `f0` (1, T), signed.
    """

    def __init__(self, hift: nn.Module) -> None:
        super().__init__()
        predictor = hift.f0_predictor
        self.condnet = nn.Sequential(*(_baked(m) if isinstance(m, nn.Conv1d) else m for m in predictor.condnet))
        self.head = nn.Conv1d(predictor.classifier.in_features, 1, 1)
        self.head.weight = nn.Parameter(predictor.classifier.weight.detach()[:, :, None].clone())
        self.head.bias = nn.Parameter(predictor.classifier.bias.detach().clone())

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.head(self.condnet(mel)).view(1, -1)


def _baked(conv: nn.Conv1d) -> nn.Conv1d:
    """A copy of `conv` whose weight is the one its parametrization, a weight norm, computes."""
    baked = nn.Conv1d(
        conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0]
    )
    baked.weight = nn.Parameter(conv.weight.detach().clone())
    baked.bias = nn.Parameter(conv.bias.detach().clone())
    return baked
