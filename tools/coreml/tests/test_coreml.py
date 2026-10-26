from collections.abc import Mapping
from pathlib import Path

import coremltools as ct
import numpy as np
import pytest
from conftest import Reference, make_reference, relative_error

from supertonic_coreml import coreml
from supertonic_coreml.assets import STAGES, Assets, StageName, Voice
from supertonic_coreml.metrics import log_spectral_distance
from supertonic_coreml.pipeline import Stages, synthesize
from supertonic_coreml.report import placement

TOLERANCE: Mapping[StageName, float] = {
    "duration_predictor": 1e-4,
    "text_encoder": 1e-2,
    "vector_estimator": 1e-2,
    "vocoder": 5e-2,
}


@pytest.fixture
def packaged(packages: Path, reference: Reference) -> Stages:
    return coreml.stages(packages, ct.ComputeUnit.CPU_AND_NE, reference.bucket)


def describe_convert():
    def when_fp32():
        @pytest.fixture(scope="module")
        def exact_reference(onnx: Stages, assets: Assets, voice: Voice) -> Reference:
            return make_reference(coreml.BUCKETS[1], onnx, assets, voice)

        @pytest.fixture(scope="module")
        def fp32_cpu(tmp_path_factory: pytest.TempPathFactory, assets: Assets) -> Stages:
            directory = tmp_path_factory.mktemp("fp32")
            for stage in STAGES:
                coreml.convert(stage, coreml.BUCKETS[1], assets, directory, ct.precision.FLOAT32)
            return coreml.stages(directory, ct.ComputeUnit.CPU_ONLY)

        @pytest.mark.parametrize("stage", STAGES)
        def it_reproduces_onnx(
            stage: StageName, fp32_cpu: Stages, exact_reference: Reference, assets: Assets
        ):
            actual = fp32_cpu[stage](exact_reference.calls.inputs[stage])

            assert relative_error(*_valid(stage, actual, exact_reference, assets)) < 1e-4


def describe_package():
    def when_run_on_the_neural_engine():
        @pytest.mark.parametrize("stage", STAGES)
        def it_stays_within_tolerance_per_stage(
            stage: StageName, packaged: Stages, reference: Reference, assets: Assets
        ):
            actual = packaged[stage](reference.calls.inputs[stage])

            assert relative_error(*_valid(stage, actual, reference, assets)) < TOLERANCE[stage]

        def it_sounds_closer_to_onnx_than_another_noise_seed(
            packaged: Stages, onnx: Stages, reference: Reference, assets: Assets, voice: Voice
        ):
            text, bucket = reference.text, reference.bucket

            other_noise = synthesize(onnx, assets, text, "pt", voice, bucket=bucket, noise_seed=1)
            converted = synthesize(packaged, assets, text, "pt", voice, bucket=bucket)

            natural = log_spectral_distance(other_noise.wav, reference.speech.wav)
            assert log_spectral_distance(converted.wav, reference.speech.wav) < natural / 3

        @pytest.mark.parametrize("stage", ["vector_estimator", "vocoder"])
        def it_places_the_heavy_stages_on_the_neural_engine(
            stage: StageName, packages: Path, reference: Reference
        ):
            package = packages / f"{stage}.mlpackage"

            plan = placement(package, ct.ComputeUnit.CPU_AND_NE, reference.bucket.name)

            assert plan.cost.get("MLNeuralEngineComputeDevice", 0.0) > 0.95


def _valid(
    stage: StageName, actual: np.ndarray, reference: Reference, assets: Assets
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the bucket padding, which the pipeline discards, off `actual` and the reference."""
    frames = int(reference.calls.inputs["vector_estimator"]["latent_mask"].sum())
    match stage:
        case "duration_predictor":
            length = None
        case "text_encoder":
            length = int(reference.calls.inputs["text_encoder"]["text_mask"].sum())
        case "vector_estimator":
            length = frames
        case "vocoder":
            length = frames * assets.samples_per_frame
    return actual[..., :length], reference.calls.outputs[stage][..., :length]
