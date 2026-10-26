"""Replacements for the norms of qwen3-coreml's modules that run on the Neural Engine of Core AI.

Core ML runs `torch.linalg.vector_norm` as the Neural Engine's own L2 reduction, which handles
the residual streams' ~1e3 peaks and inputs near 1e-4. coreai-torch lowers it to
`sqrt(sum(x * x))` in Float16, which overflows once the sum passes 65504, as it does in the code
predictor, and MPSGraph runs part of it on the GPU, as it does any division by a maximum.

These divide `x` by 32 times its mean magnitude first, which leaves the result unchanged. The
mean sums |x| / size, so it never exceeds the largest magnitude; since |x| is at most size times
the mean, a scaled element is at most size / 32 and the sum of squares at most 1024. The mean has
a floor, so that all-zero padding tokens scale by a finite factor.
"""

import torch
from qwen3_coreml import modules
from torch import Tensor

_SCALE = 32.0
_FLOOR = 1e-4


def _inverse_scale(x: Tensor, dim: int) -> Tensor:
    """32 times the mean magnitude of `x` over `dim`, at least 32 · 1e-4."""
    mean = (x.abs() * (1 / x.shape[dim])).sum(dim, keepdim=True)
    return _SCALE * torch.clamp(mean, min=_FLOOR)


def unit(x: Tensor, dim: int, eps: float = 0.0) -> Tensor:
    """`x / (|x| + eps)` over `dim`, as `modules.unit`."""
    inverse = _inverse_scale(x, dim)
    scaled = x / inverse
    norm = (scaled * scaled).sum(dim, keepdim=True).sqrt()
    return scaled / (norm + eps / inverse if eps else norm)


def unit_eps(x: Tensor, eps: float) -> Tensor:
    """`x / sqrt(|x|^2 + size * eps)` over channels, as `modules.unit_eps`."""
    inverse = _inverse_scale(x, 1)
    scaled = x / inverse
    padding = (x.shape[1] * eps) ** 0.5 / inverse
    return scaled / ((scaled * scaled).sum(1, keepdim=True) + padding * padding).sqrt()


def install() -> None:
    """Makes qwen3-coreml's modules use these norms; their layers look them up when they run."""
    modules.unit = unit
    modules.unit_eps = unit_eps
