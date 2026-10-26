"""moss-nano-coreai-convert: MOSS-TTS-Nano as Core AI models for the Neural Engine, in build/aimodel.

The packages and functions are moss-nano-coreml's, rearranged in `graphs` so that every function
runs on the Neural Engine alone. `coreai-aot` then compiles each package for every iPhone
architecture. build/assets/embeddings.f16 holds the decoder's embeddings, which the app sums;
moss-nano-coreml's convert writes the rest the app reads: config.json, the tokenizer and the
voices.
"""

import argparse
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from coreai_torch import TorchConverter, get_decomp_table
from moss_nano_coreml import graphs as coreml_graphs
from moss_nano_coreml import reference
from moss_nano_coreml.convert import CHUNK_FRAMES, CODEBOOK_DEPTHS, CONTEXTS, PREFILL_ROWS, codec_caches
from torch import Tensor, nn

from moss_nano_coreai import graphs

BUILD = Path(__file__).resolve().parents[2] / "build" / "aimodel"
ASSETS = BUILD.parent / "assets"
PACKAGES = ("GlobalDecoder", "LocalDecoder", "CodecDecoder")
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


def global_decoder(source: reference.Reference) -> dict[str, Function]:
    decoder = graphs.Decoder(source.model).half()
    cache = (coreml_graphs.LAYERS, coreml_graphs.HEADS)
    head = coreml_graphs.HEAD_DIM
    steps = {
        f"step_{context}": Function(
            graphs.Step(decoder).half(),
            {
                "embed": _zeros(1, coreml_graphs.HIDDEN),
                "key_cache": _zeros(*cache, head, context),
                "value_cache": _zeros(*cache, context, head),
                "mask": _zeros(1, context),
                "cos": _zeros(1, head),
                "sin": _zeros(1, head),
            },
            ["hidden", "key", "value"],
        )
        for context in CONTEXTS
    }
    prefills = {
        f"prefill_{rows}": Function(
            graphs.Prefill(decoder, rows).half(),
            {"embeds": _zeros(rows, coreml_graphs.HIDDEN)},
            ["hidden", "keys", "values"],
        )
        for rows in PREFILL_ROWS
    }
    return steps | prefills


def local_decoder(source: reference.Reference) -> dict[str, Function]:
    local = graphs.Local(source.model).half()
    return {
        f"frame_{depth}": Function(
            graphs.Sampler(local, depth).half(),
            {
                "hidden": _zeros(1, coreml_graphs.HIDDEN),
                "seen": _zeros(depth, coreml_graphs.CODEBOOK_SIZE),
                "gumbel": _zeros(depth, coreml_graphs.CODEBOOK_SIZE),
                "text_gumbel": _zeros(1, 2),
                "settings": _zeros(1, 6),
            },
            ["end", "codes"],
        )
        for depth in CODEBOOK_DEPTHS
    }


def codec_decoder(source: reference.Reference) -> dict[str, Function]:
    caches = codec_caches(source.codec)
    codec = graphs.Codec(source.codec, CHUNK_FRAMES).half()
    inputs = {
        "codes": _zeros(CHUNK_FRAMES, coreml_graphs.CODEBOOKS),
        **{f"valid_{index}": _zeros(1, stage.cache) for index, stage in enumerate(codec.stages)},
        **{name: _zeros(*shape) for name, shape in caches},
    }
    return {
        f"chunk_{CHUNK_FRAMES}": Function(
            codec,
            inputs,
            ["audio", *(f"chunk_{name}" for name, _ in caches)],
        )
    }


def export_assets(source: reference.Reference) -> None:
    """embeddings.f16: `graphs.embedding_table`, Float16 and row-major."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    table = graphs.embedding_table(source.model)
    table.detach().float().numpy().astype("<f2").tofile(ASSETS / "embeddings.f16")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--package", choices=(*PACKAGES, "assets"), action="append")
    args = parser.parse_args()
    graphs.install()
    source = reference.load()
    builders: Mapping[str, Callable[[reference.Reference], dict[str, Function]]] = {
        "GlobalDecoder": global_decoder,
        "LocalDecoder": local_decoder,
        "CodecDecoder": codec_decoder,
    }
    BUILD.mkdir(parents=True, exist_ok=True)
    selected = args.package or (*PACKAGES, "assets")
    if "assets" in selected:
        export_assets(source)
        print(f"{ASSETS}: embeddings", flush=True)
    for package in (p for p in PACKAGES if p in selected):
        started = time.monotonic()
        path = BUILD / f"{package}.aimodel"
        functions = builders[package](source)
        save(path, functions)
        print(f"{path}: {', '.join(functions)} in {time.monotonic() - started:.0f} s", flush=True)
