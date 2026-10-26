"""Reference runs of the Core ML packages, for checking ports of the pipeline against them.

Each case is a JSON manifest plus raw little-endian float32 arrays, next to the packages.

The runs use the CPU only: the Neural Engine computes in fp16 differently from the CPU, and the
ports are checked on machines without one, such as the iOS simulator.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import coremltools as ct
import numpy as np

from supertonic_coreml import assets as assets_module
from supertonic_coreml import coreml
from supertonic_coreml.pipeline import encode, record, synthesize
from supertonic_coreml.text import to_ids

FIXTURES = coreml.BUILD.parent / "fixtures"
STEPS = 8
SPEED = 1.05


@dataclass(frozen=True)
class Case:
    name: str
    text: str
    lang: str
    voice: str


CASES = (
    Case("short-pt", "A viagem era curta, e os versos não eram maus.", "pt", "F1"),
    Case(
        "caminho-pt",
        "No meio do caminho tinha uma pedra. Tinha uma pedra no meio do caminho, "
        "e eu nunca me esquecerei desse acontecimento.",
        "pt",
        "M1",
    ),
    Case(
        "long-en",
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the "
        "age of foolishness, it was the epoch of belief, it was the epoch of incredulity.",
        "en",
        "F2",
    ),
)


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    assets = assets_module.load()
    for case in CASES:
        voice = assets.voice(case.voice)
        ids = to_ids(case.text, case.lang, assets.indexer)
        text_bucket = coreml.select(len(ids))
        if text_bucket is None:
            raise ValueError(f"{case.name}: {len(ids)} ids fit no bucket")
        encoded = encode(case.text, case.lang, assets, text_bucket.text_length)
        duration_predictor = coreml.stages(coreml.BUILD, ct.ComputeUnit.CPU_ONLY, text_bucket)[
            "duration_predictor"
        ]
        seconds = float(
            duration_predictor(
                {"text_ids": encoded.text_ids, "text_mask": encoded.text_mask, "style_dp": voice.dp}
            )[0]
        )
        frames = -(-int(seconds / SPEED * assets.sample_rate) // assets.samples_per_frame)
        bucket = coreml.select(len(ids), frames)
        if bucket is None:
            raise ValueError(f"{case.name}: {frames} frames fit no bucket")
        calls = record(coreml.stages(coreml.BUILD, ct.ComputeUnit.CPU_ONLY, bucket))
        speech = synthesize(
            calls.stages, assets, case.text, case.lang, voice, steps=STEPS, speed=SPEED, bucket=bucket
        )
        arrays = {
            "noise": calls.inputs["vector_estimator"]["noisy_latent"][0, :, :frames],
            "text_emb": calls.outputs["text_encoder"],
            "latent": calls.inputs["vocoder"]["latent"],
            "wav": speech.wav,
        }
        manifest = {
            "text": case.text,
            "lang": case.lang,
            "voice": case.voice,
            "steps": STEPS,
            "speed": SPEED,
            "bucket": {"text_length": bucket.text_length, "latent_length": bucket.latent_length},
            "ids": ids.tolist(),
            "seconds": speech.seconds,
            "frames": frames,
            "arrays": {
                name: _write(FIXTURES / f"{case.name}.{name}.f32", array) for name, array in arrays.items()
            },
        }
        (FIXTURES / f"{case.name}.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(
            f"{case.name}: bucket {bucket.name}, {len(ids)} ids, {frames} frames, {speech.seconds:.2f}s",
            flush=True,
        )


def _write(path: Path, array: np.ndarray) -> dict[str, object]:
    path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes())
    return {"file": path.name, "shape": list(array.shape)}
