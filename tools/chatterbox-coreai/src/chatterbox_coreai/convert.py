"""chatterbox-coreai-convert: the Core AI models of the functions the app loads, in build/aimodel.

Each package of chatterbox-coreml becomes an .aimodel holding only the functions Chatterbox.swift
loads, under the same names: every loaded function holds its own copy of the weights. T3's linear
weights are palettized to 4 bits per group of 16 output channels with the k-means of coremltools,
as chatterbox-coreml does. `coreai-aot` then compiles each package for every iPhone architecture. The table the app reads
in place of a function goes to build/assets.
"""

import argparse
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from chatterbox_coreml import reference, s3gen
from chatterbox_coreml.convert import VOCODER_FRAMES
from coreai_torch import TorchConverter, get_decomp_table
from coreai_torch._compression.custom_layers import PalettizeModule
from coreai_torch._compression.utils import wrap_for_parametrization
from coremltools.optimize.coreml._quantization_passes import palettize_weights
from torch import nn
from torch.nn.utils import parametrize

from chatterbox_coreai import hift, t3

BUILD = Path(__file__).resolve().parents[2] / "build" / "aimodel"
ASSETS = BUILD.parent / "assets"
PREFILL_LENGTH = 384
DECODE_WINDOWS = (384, 768, 1024)
FLOW_TOKENS = 640
PACKAGES = ("t3", "flow_encoder", "flow_step", "f0", "hift")

F16 = torch.float16


@dataclass(frozen=True)
class Function:
    module: nn.Module
    inputs: Mapping[str, torch.Tensor]
    outputs: Sequence[str]


def _elu(
    x: torch.Tensor, alpha: float = 1.0, scale: float = 1.0, input_scale: float = 1.0
) -> torch.Tensor:
    return scale * (torch.relu(x) + alpha * (torch.exp(torch.clamp(x * input_scale, max=0)) - 1))


def _mish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.tanh(torch.log(1 + torch.exp(x)))


def decompositions() -> dict[torch._ops.OpOverload, Callable[..., torch.Tensor]]:
    """coreai-torch's decompositions, with elu and mish written in operations the Neural Engine
    runs: MPSGraph places PyTorch's own decompositions of them, and of softplus, on the GPU."""
    table = get_decomp_table()
    table[torch.ops.aten.elu.default] = _elu
    table[torch.ops.aten.mish.default] = _mish
    return table


def save(path: Path, functions: Mapping[str, Function]) -> None:
    """Writes `functions` to `path`, an .aimodel with one entrypoint per function."""
    table = decompositions()
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    for name, function in functions.items():
        with torch.inference_mode():
            exported = torch.export.export(
                function.module.eval(), args=tuple(function.inputs.values())
            ).run_decompositions(table)
        converter.add_exported_program(
            exported,
            input_names=list(function.inputs),
            output_names=list(function.outputs),
            entrypoint_name=name,
        )
    if path.exists():
        shutil.rmtree(path)
    converter.to_coreai().save_asset(path)


def palettize(module: nn.Module) -> None:
    """Replaces the weight of each linear layer of `module` whose rows split into groups of 16 by
    a 4-bit lookup table per group; the others, such as T3's head of 8194 rows, stay Float16."""
    for linear in [m for m in module.modules() if isinstance(m, nn.Linear)]:
        params = palettize_weights.blockwise_compress(
            linear.weight.detach().float().numpy(), "KMEANS", 4, [16, 0]
        )
        if params is None:
            continue
        palette = wrap_for_parametrization(PalettizeModule)(
            torch.from_numpy(params.indices), torch.from_numpy(params.lut.astype(np.float16))
        )
        parametrize.register_parametrization(linear, "weight", palette, unsafe=True)


def _zeros(*shape: int, dtype: torch.dtype = F16) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype)


def t3_functions() -> dict[str, Function]:
    weights = t3.Weights(reference.model().t3).half()
    palettize(weights)
    prefill = Function(
        t3.Prefill(weights, PREFILL_LENGTH).half(),
        {
            "embeds": _zeros(t3.ROWS, PREFILL_LENGTH, t3.HIDDEN),
            "cos": _zeros(PREFILL_LENGTH, t3.HEAD_DIM),
            "sin": _zeros(PREFILL_LENGTH, t3.HEAD_DIM),
            "last": _zeros(1, PREFILL_LENGTH),
        },
        ["logits", "keys", "values"],
    )

    def decode(window: int) -> Function:
        inputs = {
            "embed": _zeros(1, t3.HIDDEN),
            "cos": _zeros(1, t3.HEAD_DIM),
            "sin": _zeros(1, t3.HEAD_DIM),
            "mask": _zeros(1, window),
            "key_cache": _zeros(*t3.key_shape(window)),
            "value_cache": _zeros(*t3.value_shape(window)),
        }
        return Function(t3.Decode(weights, window).half(), inputs, ["logits", "key", "value"])

    return {f"prefill_{PREFILL_LENGTH}": prefill} | {f"decode_{w}": decode(w) for w in DECODE_WINDOWS}


class _FlowEncoder(nn.Module):
    """`s3gen.FlowEncoder` from the tokens' embeddings, (1, 512, 1, N), which the app looks up in
    tables/flow_emb.f16: MPSGraph runs a lookup by integer index on the GPU."""

    def __init__(self, encoder: s3gen.FlowEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(embeds, mask)


def flow_encoder_functions() -> dict[str, Function]:
    encoder = s3gen.FlowEncoder(reference.model().s3gen.flow, FLOW_TOKENS)
    width = encoder.embedding.embedding_dim
    inputs = {"embeds": _zeros(1, width, 1, FLOW_TOKENS), "mask": _zeros(1, FLOW_TOKENS)}
    return {f"n{FLOW_TOKENS}": Function(_FlowEncoder(encoder).half(), inputs, ["mu"])}


def export_assets() -> None:
    """The flow's token embeddings, (6561, 512) Float16, which the app looks up itself."""
    tables = ASSETS / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    weight = reference.model().s3gen.flow.input_embedding.weight.detach()
    weight.numpy().astype("<f2").tofile(tables / "flow_emb.f16")


def flow_step_functions() -> dict[str, Function]:
    frames = s3gen.TOKEN_MEL_RATIO * FLOW_TOKENS
    module = s3gen.FlowStep(reference.model().s3gen.flow.decoder.estimator).half()
    mel = (1, s3gen.MEL_CHANNELS, frames)
    inputs = {
        "x": _zeros(*mel),
        "mu": _zeros(*mel),
        "cond": _zeros(*mel),
        "spks": _zeros(1, s3gen.MEL_CHANNELS),
        "mask": _zeros(1, 1, frames),
        "t_emb": _zeros(1, s3gen.TIME_EMBEDDING_DIM),
        "dt": _zeros(1),
    }
    return {f"m{frames}": Function(module, inputs, ["x_next"])}


def f0_functions() -> dict[str, Function]:
    module = hift.F0(reference.model().s3gen.mel2wav).half()
    inputs = {"mel": _zeros(1, s3gen.MEL_CHANNELS, VOCODER_FRAMES)}
    return {f"t{VOCODER_FRAMES}": Function(module, inputs, ["f0"])}


def hift_functions() -> dict[str, Function]:
    module = hift.HiFT(s3gen.HiFT(reference.model().s3gen.mel2wav), VOCODER_FRAMES).half()
    inputs = {
        "mel": _zeros(1, s3gen.MEL_CHANNELS, VOCODER_FRAMES),
        "source": _zeros(1, 1, s3gen.SAMPLES_PER_FRAME * VOCODER_FRAMES),
    }
    return {f"t{VOCODER_FRAMES}": Function(module, inputs, ["audio"])}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", choices=PACKAGES, action="append")
    args = parser.parse_args()
    if args.package is None or "flow_encoder" in args.package:
        export_assets()
    builders: Mapping[str, Callable[[], dict[str, Function]]] = {
        "t3": t3_functions,
        "flow_encoder": flow_encoder_functions,
        "flow_step": flow_step_functions,
        "f0": f0_functions,
        "hift": hift_functions,
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    for package in args.package or PACKAGES:
        started = time.monotonic()
        path = BUILD / f"{package}.aimodel"
        functions = builders[package]()
        save(path, functions)
        print(f"{path}: {', '.join(functions)} in {time.monotonic() - started:.0f} s", flush=True)
