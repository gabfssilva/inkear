from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from supertonic_coreml import assets as assets_module
from supertonic_coreml import coreml
from supertonic_coreml.assets import STAGES, Assets, Voice
from supertonic_coreml.pipeline import Bucket, Recording, Speech, Stages, onnx_stages, record, synthesize

TEXTS: Mapping[Bucket, str] = {
    coreml.BUCKETS[0]: "A viagem era curta, e os versos não eram maus.",
    coreml.BUCKETS[1]: (
        "No meio do caminho tinha uma pedra. Tinha uma pedra no meio do caminho, "
        "e eu nunca me esquecerei desse acontecimento."
    ),
    coreml.BUCKETS[2]: (
        "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central "
        "um rapaz aqui do bairro, que eu conheço de vista e de chapéu."
    ),
}


@dataclass(frozen=True)
class Reference:
    """ONNX run of `text` in `bucket`: the first call to each stage, and the speech."""

    text: str
    bucket: Bucket
    calls: Recording
    speech: Speech


@pytest.fixture(scope="session")
def assets() -> Assets:
    return assets_module.load()


@pytest.fixture(scope="session")
def voice(assets: Assets) -> Voice:
    return assets.voice("F1")


@pytest.fixture(scope="session")
def onnx(assets: Assets) -> Stages:
    return onnx_stages(assets)


@pytest.fixture(scope="session", params=coreml.BUCKETS, ids=lambda bucket: bucket.name)
def reference(request: pytest.FixtureRequest, onnx: Stages, assets: Assets, voice: Voice) -> Reference:
    return make_reference(request.param, onnx, assets, voice)


@pytest.fixture(scope="session")
def packages(tmp_path_factory: pytest.TempPathFactory, assets: Assets) -> Path:
    directory = tmp_path_factory.mktemp("packages")
    for stage in STAGES:
        coreml.package(stage, coreml.BUCKETS, assets, directory, coreml.PRECISIONS[stage])
    return directory


def make_reference(bucket: Bucket, onnx: Stages, assets: Assets, voice: Voice) -> Reference:
    calls = record(onnx)
    speech = synthesize(calls.stages, assets, TEXTS[bucket], "pt", voice, bucket=bucket)
    return Reference(text=TEXTS[bucket], bucket=bucket, calls=calls, speech=speech)


def relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))
