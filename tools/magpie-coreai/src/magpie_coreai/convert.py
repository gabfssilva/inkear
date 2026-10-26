"""magpie-coreai-convert: Magpie TTS Multilingual 357M as Core AI models for the Neural Engine, in
build/aimodel.

The stages are magpie-coreml's, rearranged in `graphs` so that every function runs on the Neural
Engine alone. The decoder's prefill and step share one package, so that its weights are stored
once. build/assets holds the tables the app looks up (`graphs.tables`); magpie-coreml's convert
writes the rest the app reads: magpie.json, the front end, the BOS embedding and the codec, which
stays a Core ML package because the ahead-of-time compiler gets it wrong. `coreai-aot` then
compiles each package for every iPhone architecture.
"""

import argparse
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from coreai_torch import TorchConverter, get_decomp_table
from magpie_coreml import checkpoint
from magpie_coreml.checkpoint import CODES, VOCAB
from magpie_coreml.modules import CACHE, CROSS_DIM, HEAD_DIM, HEADS, TEXT, D
from nemo.collections.tts.models import MagpieTTSModel
from torch import Tensor, nn

from magpie_coreai import graphs

BUILD = Path(__file__).resolve().parents[2] / "build" / "aimodel"
ASSETS = BUILD.parent / "assets"
PACKAGES = ("text_encoder", "decoder", "local_transformer")
LAYERS = 12
F16 = torch.float16


@dataclass(frozen=True)
class Function:
    module: nn.Module
    inputs: Mapping[str, Tensor]
    outputs: Sequence[str]


def save(path: Path, functions: Mapping[str, Function]) -> None:
    """Writes `functions` to `path`, an .aimodel with one entrypoint per function."""
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    for name, function in functions.items():
        with torch.inference_mode():
            exported = torch.export.export(
                function.module.eval(), args=tuple(function.inputs.values())
            ).run_decompositions(get_decomp_table())
        converter.add_exported_program(
            exported,
            input_names=list(function.inputs),
            output_names=list(function.outputs),
            entrypoint_name=name,
        )
    if path.exists():
        shutil.rmtree(path)
    converter.to_coreai().save_asset(path)


def _zeros(*shape: int) -> Tensor:
    return torch.zeros(shape, dtype=F16)


def text_encoder(model: MagpieTTSModel) -> dict[str, Function]:
    inputs = {"embeds": _zeros(1, TEXT, D), "mask": _zeros(1, TEXT)}
    return {"main": Function(graphs.TextEncoder(model).half(), inputs, ["encoder_output"])}


def decoder(model: MagpieTTSModel) -> dict[str, Function]:
    rows = 2 * LAYERS
    cross = {"cross_keys": _zeros(LAYERS, 1, TEXT, CROSS_DIM), "cross_values": _zeros(LAYERS, 1, TEXT, CROSS_DIM)}
    prefill = graphs.DecoderPrefill(model).half()
    step = graphs.DecoderStep(model).half()
    return {
        "prefill": Function(
            prefill,
            {"context": _zeros(1, graphs.PREFILL, D), "encoder_output": _zeros(1, TEXT, D), "encoder_mask": _zeros(1, TEXT)},
            ["keys", "values", "cross_keys", "cross_values"],
        ),
        "step": Function(
            step,
            {
                "audio_emb": _zeros(1, 1, D),
                "mask": _zeros(1, CACHE),
                "encoder_mask": _zeros(1, TEXT),
                "attn_prior": _zeros(1, TEXT),
                "cfg_scale": _zeros(1),
                "key_cache": _zeros(rows, HEADS, HEAD_DIM, CACHE),
                "value_cache": _zeros(rows, HEADS, CACHE, HEAD_DIM),
                **cross,
            },
            ["logits", "h_last", "alignment", "key", "value"],
        ),
    }


def local_transformer(model: MagpieTTSModel) -> dict[str, Function]:
    inputs = {
        "h_last": _zeros(2, 1, D),
        "gumbel": _zeros(CODES, VOCAB),
        "temperature": _zeros(1),
        "cfg_scale": _zeros(1),
        "forbid_eos": _zeros(1),
    }
    return {"main": Function(graphs.LocalTransformer(model).half(), inputs, ["codes", "audio_emb"])}


def export_assets(model: MagpieTTSModel) -> None:
    """`graphs.tables`, Float16 and row-major, as <name>.f16."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name, table in graphs.tables(model).items():
        table.detach().float().numpy().astype("<f2").tofile(ASSETS / f"{name}.f16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", choices=(*PACKAGES, "assets"), action="append")
    args = parser.parse_args()
    model = checkpoint.load().model
    builders: Mapping[str, Callable[[MagpieTTSModel], dict[str, Function]]] = {
        "text_encoder": text_encoder,
        "decoder": decoder,
        "local_transformer": local_transformer,
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    selected = args.package or (*PACKAGES, "assets")
    if "assets" in selected:
        export_assets(model)
        print(f"{ASSETS}: tables", flush=True)
    for package in (p for p in PACKAGES if p in selected):
        started = time.monotonic()
        path = BUILD / f"{package}.aimodel"
        functions = builders[package](model)
        save(path, functions)
        print(f"{path}: {', '.join(functions)} in {time.monotonic() - started:.0f} s", flush=True)
