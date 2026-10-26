import argparse
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import coremltools as ct
import numpy as np
import onnx
import onnxslim
import torch
from onnx import defs
from onnx2torch import convert as onnx_to_torch
from onnx2torch.node_converters.registry import add_converter, get_converter

from supertonic_coreml import assets as assets_module
from supertonic_coreml.assets import STAGES, Assets, StageName
from supertonic_coreml.pipeline import Bucket, Stage, Stages

BUILD = Path(__file__).resolve().parents[2] / "build" / "supertonic-3"

# Latent frames per text id run from about 0.8 to 1.45 (short utterances, spelled-out numbers).
BUCKETS = (Bucket(64, 96), Bucket(128, 192), Bucket(256, 384))

# Core ML runs the duration predictor on the CPU at either precision; fp32 keeps durations exact.
PRECISIONS: Mapping[StageName, ct.precision] = {
    "duration_predictor": ct.precision.FLOAT32,
    "text_encoder": ct.precision.FLOAT16,
    "vector_estimator": ct.precision.FLOAT16,
    "vocoder": ct.precision.FLOAT16,
}

# The upstream graphs are opset 19. These ops only gained dtypes or modes since the versions
# onnx2torch implements, and the graphs use none of them.
for _op, _implemented in {
    "Cast": 13,
    "Equal": 13,
    "Pad": 13,
    "Reshape": 14,
    "Shape": 15,
    "Split": 13,
}.items():
    add_converter(operation_type=_op, version=defs.get_schema(_op, 19).since_version)(
        get_converter(domain="", operation_type=_op, version=_implemented)
    )


def convert(
    stage: StageName,
    bucket: Bucket,
    assets: Assets,
    directory: Path,
    precision: ct.precision,
) -> Path:
    """Convert one stage to an ML Program with every input fixed to `bucket`."""
    model = onnx.load(assets.onnx(stage))
    shapes = _input_shapes(model, bucket)
    for value in model.graph.input:
        for dim, size in zip(value.type.tensor_type.shape.dim, shapes[value.name], strict=True):
            dim.Clear()
            dim.dim_value = size
    model = onnxslim.slim(model)
    examples = tuple(_example(value, shapes[value.name]) for value in model.graph.input)
    traced = torch.jit.trace(onnx_to_torch(model).eval(), examples)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name=value.name, shape=example.shape, dtype=_coreml_dtype(example))
            for value, example in zip(model.graph.input, examples, strict=True)
        ],
        outputs=[ct.TensorType(name=output.name) for output in model.graph.output],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = directory / f"{stage}.mlpackage"
    mlmodel.save(str(path))
    return path


def package(
    stage: StageName,
    buckets: Sequence[Bucket],
    assets: Assets,
    directory: Path,
    precision: ct.precision,
) -> Path:
    """Convert `stage` once per bucket into one package, a function per bucket sharing weights.

    Functions are named after `Bucket.name`; the first bucket is the default function.
    """
    descriptor = ct.utils.MultiFunctionDescriptor()
    with tempfile.TemporaryDirectory() as scratch:
        for bucket in buckets:
            single = Path(scratch) / bucket.name
            single.mkdir()
            descriptor.add_function(
                str(convert(stage, bucket, assets, single, precision)),
                src_function_name="main",
                target_function_name=bucket.name,
            )
        descriptor.default_function_name = buckets[0].name
        path = directory / f"{stage}.mlpackage"
        ct.utils.save_multifunction(descriptor, str(path))
    return path


def stages(directory: Path, compute_units: ct.ComputeUnit, bucket: Bucket | None = None) -> Stages:
    """Load every stage from `directory`, picking `bucket`'s function from multifunction packages."""

    def stage(name: StageName) -> Stage:
        model = ct.models.MLModel(
            str(directory / f"{name}.mlpackage"),
            compute_units=compute_units,
            function_name=bucket and bucket.name,
        )
        spec = model.get_spec()
        description = (
            {function.name: function for function in spec.description.functions}[bucket.name]
            if bucket
            else spec.description
        )
        output = description.output[0].name
        int_inputs = {
            feature.name
            for feature in description.input
            if feature.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32
        }

        def run(inputs: Mapping[str, np.ndarray]) -> np.ndarray:
            pinned = {
                key: np.array(value, dtype=np.int32 if key in int_inputs else np.float32, order="C")
                for key, value in inputs.items()
            }
            _PINNED.append(pinned)
            return model.predict(pinned)[output]

        return run

    return {name: stage(name) for name in STAGES}


# coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML drops them later
# on its own queue without the GIL, corrupting the Python heap (apple/coremltools#2829, fixed
# after 9.0). Keeping a reference to every input, untouched, means that release never frees.
_PINNED: list[Mapping[str, np.ndarray]] = []


def select(ids: int, frames: int | None = None) -> Bucket | None:
    """The smallest bucket that pads `ids` text ids and, when given, `frames` latent frames.

    A bucket must leave at least one position of padding: padded and unpadded inputs give
    different outputs, and the packages are meant for padded ones.
    """
    return next(
        (
            bucket
            for bucket in BUCKETS
            if bucket.text_length > ids and (frames is None or bucket.latent_length > frames)
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Convert Supertonic 3 to Core ML packages, with the assets they need, in {BUILD}"
    )
    parser.add_argument("--stage", choices=STAGES, action="append")
    args = parser.parse_args()
    BUILD.mkdir(parents=True, exist_ok=True)
    assets = assets_module.load()
    for name in ("tts.json", "unicode_indexer.json"):
        shutil.copy(assets.root / "onnx" / name, BUILD / name)
    shutil.copytree(assets.root / "voice_styles", BUILD / "voice_styles", dirs_exist_ok=True)
    for stage in args.stage or STAGES:
        print(f"{stage}: {package(stage, BUCKETS, assets, BUILD, PRECISIONS[stage])}", flush=True)


def _input_shapes(model: onnx.ModelProto, bucket: Bucket) -> Mapping[str, tuple[int, ...]]:
    sizes = {"batch_size": 1, "text_length": bucket.text_length, "latent_length": bucket.latent_length}
    return {
        value.name: tuple(dim.dim_value or sizes[dim.dim_param] for dim in value.type.tensor_type.shape.dim)
        for value in model.graph.input
    }


def _example(value: onnx.ValueInfoProto, shape: tuple[int, ...]) -> torch.Tensor:
    if value.type.tensor_type.elem_type == onnx.TensorProto.INT64:
        return torch.zeros(shape, dtype=torch.int64)
    return torch.ones(shape, dtype=torch.float32)


def _coreml_dtype(example: torch.Tensor) -> type[np.generic]:
    return np.int32 if example.dtype == torch.int64 else np.float32
