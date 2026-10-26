import argparse
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.models.compute_plan import MLComputePlan

from supertonic_coreml import assets as assets_module
from supertonic_coreml import coreml
from supertonic_coreml.assets import STAGES, StageName
from supertonic_coreml.pipeline import Bucket, Stages, onnx_stages, record, synthesize

SAMPLE = (
    "No meio do caminho tinha uma pedra. Tinha uma pedra no meio do caminho, "
    "e eu nunca me esquecerei desse acontecimento."
)
SHORT_SAMPLE = "A viagem era curta."


@dataclass(frozen=True)
class Placement:
    """Share of a stage's estimated cost per preferred compute device."""

    cost: Mapping[str, float]
    ops: Mapping[str, int]


def placement(package: Path, compute_units: ct.ComputeUnit, function: str = "main") -> Placement:
    # MLComputePlan only plans the default function, so plan a copy where `function` is the default.
    descriptor = ct.utils.MultiFunctionDescriptor(str(package))
    descriptor.default_function_name = function
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / package.name
        ct.utils.save_multifunction(descriptor, str(copy))
        # The compiled .mlmodelc is a temporary that lives only as long as the MLModel.
        model = ct.models.MLModel(str(copy), compute_units=compute_units, function_name=function)
        plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=compute_units)

    def operations() -> Iterator[tuple[str, float]]:
        for operation in plan.model_structure.program.functions[function].block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            cost = plan.get_estimated_cost_for_mlprogram_operation(operation)
            if usage is not None:
                yield type(usage.preferred_compute_device).__name__, cost.weight if cost else 0.0

    placed = tuple(operations())
    cost = Counter[str]()
    for device, weight in placed:
        cost[device] += weight
    total = sum(cost.values()) or 1.0
    return Placement(
        cost={device: weight / total for device, weight in cost.items()},
        ops=Counter(device for device, _ in placed),
    )


def stage_latency(
    stages: Stages, inputs: Mapping[StageName, Mapping[str, np.ndarray]], runs: int
) -> Mapping[StageName, float]:
    def median(run: Callable[[], object]) -> float:
        run()
        return float(np.median([_timed(run) for _ in range(runs)]))

    return {name: median(lambda name=name: stages[name](inputs[name])) for name in STAGES}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Report placement and latency of the packages in {coreml.BUILD}"
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    assets = assets_module.load()
    voice = assets.voice("F1")

    def measure(stages: Stages, text: str, bucket: Bucket) -> tuple[Mapping[StageName, float], float]:
        """Per-stage median latency, and the total for one utterance with `args.steps` steps."""
        recorded = record(stages)
        synthesize(recorded.stages, assets, text, "pt", voice, steps=args.steps, bucket=bucket)
        latency = stage_latency(stages, recorded.inputs, args.runs)
        return latency, sum(latency.values()) + (args.steps - 1) * latency["vector_estimator"]

    print("neural engine share of estimated cost, CPU_AND_NE")
    for name in STAGES:
        shares = (
            placement(coreml.BUILD / f"{name}.mlpackage", ct.ComputeUnit.CPU_AND_NE, bucket.name).cost
            for bucket in coreml.BUCKETS
        )
        cells = "  ".join(
            f"{bucket.name} {share.get('MLNeuralEngineComputeDevice', 0.0):6.1%}"
            for bucket, share in zip(coreml.BUCKETS, shares, strict=True)
        )
        print(f"  {name:20s} {cells}")

    print(f"latency per bucket on CPU_AND_NE, median of {args.runs} (ms); rtf of an utterance filling it")
    for bucket in coreml.BUCKETS:
        stages = coreml.stages(coreml.BUILD, ct.ComputeUnit.CPU_AND_NE, bucket)
        latency, total = measure(stages, SHORT_SAMPLE, bucket)
        capacity = bucket.latent_length * assets.samples_per_frame / assets.sample_rate
        cells = "  ".join(f"{name} {seconds * 1e3:6.1f}" for name, seconds in latency.items())
        print(f"  {bucket.name:10s} {cells}  rtf {total / capacity:.3f}")

    bucket = coreml.BUCKETS[1]
    backends: Mapping[str, Callable[[], Stages]] = {
        "onnx cpu": lambda: onnx_stages(assets),
        "coreml cpu": lambda: coreml.stages(coreml.BUILD, ct.ComputeUnit.CPU_ONLY, bucket),
        "coreml cpu+ane": lambda: coreml.stages(coreml.BUILD, ct.ComputeUnit.CPU_AND_NE, bucket),
    }
    print(f"rtf of the sample sentence in {bucket.name}, by backend")
    for label, load in backends.items():
        stages = load()
        _, total = measure(stages, SAMPLE, bucket)
        seconds = synthesize(stages, assets, SAMPLE, "pt", voice, steps=args.steps, bucket=bucket).seconds
        print(f"  {label:15s} {total / seconds:.3f}")


def _timed(run: Callable[[], object]) -> float:
    start = time.perf_counter()
    run()
    return time.perf_counter() - start
