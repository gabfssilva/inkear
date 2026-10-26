"""Int8 weights for Core AI, as qwen3-coreml's `int8_weights` makes them for Core ML."""

import torch
from coreai_torch._compression.custom_layers import WeightDequantizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from torch import nn
from torch.nn.utils import parametrize

# Core ML's linear quantizer leaves smaller weights, such as the per-head norms, in Float16.
_THRESHOLD = 2048


def int8_weights(module: nn.Module) -> None:
    """Quantizes the convolution weights of `module` to int8, symmetric per output channel, in
    place. Activations stay Float16. The per-frame models read all their weights at every step, so
    halving them cuts the step time on the Neural Engine by about a third.

    Tables multiplied by a one-hot row, such as the code predictor's embeddings, stay Float16:
    MPSGraph runs their int8 dequantization on the GPU.
    """
    for owner in list(module.modules()):
        for name, parameter in list(owner.named_parameters(recurse=False)):
            if parameter.dim() != 4 or parameter.numel() <= _THRESHOLD:
                continue
            weight = parameter.detach().float()
            rows = tuple(range(1, weight.dim()))
            scale = weight.abs().amax(dim=rows, keepdim=True).clamp(min=1e-12) / 127
            quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
            dequantize = wrap_for_parametrization(WeightDequantizeModule)(quantized, scale.to(parameter.dtype))
            parametrize.register_parametrization(owner, name, dequantize, unsafe=True)
