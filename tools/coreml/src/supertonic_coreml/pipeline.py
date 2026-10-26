from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

from supertonic_coreml.assets import STAGES, Assets, StageName, Voice
from supertonic_coreml.text import to_ids

type Inputs = Mapping[str, np.ndarray]
type Stage = Callable[[Inputs], np.ndarray]
type Stages = Mapping[StageName, Stage]


@dataclass(frozen=True)
class Bucket:
    """Fixed input lengths a converted model is specialised for."""

    text_length: int
    latent_length: int

    @property
    def name(self) -> str:
        return f"t{self.text_length}_l{self.latent_length}"


@dataclass(frozen=True)
class Encoded:
    text_ids: np.ndarray
    text_mask: np.ndarray


@dataclass(frozen=True)
class Speech:
    wav: np.ndarray
    seconds: float


def onnx_stages(assets: Assets) -> Stages:
    def stage(name: StageName) -> Stage:
        session = ort.InferenceSession(assets.onnx(name), providers=["CPUExecutionProvider"])
        return lambda inputs: session.run(None, dict(inputs))[0]

    return {name: stage(name) for name in STAGES}


@dataclass(frozen=True)
class Recording:
    stages: Stages
    inputs: Mapping[StageName, Inputs]
    outputs: Mapping[StageName, np.ndarray]


def record(stages: Stages) -> Recording:
    """Wrap `stages` so the first call to each stage is kept in `inputs` and `outputs`."""
    inputs: dict[StageName, Inputs] = {}
    outputs: dict[StageName, np.ndarray] = {}

    def recorder(name: StageName) -> Stage:
        def run(values: Inputs) -> np.ndarray:
            output = stages[name](values)
            if name not in inputs:
                inputs[name], outputs[name] = values, output
            return output

        return run

    return Recording(stages={name: recorder(name) for name in STAGES}, inputs=inputs, outputs=outputs)


def encode(text: str, lang: str, assets: Assets, length: int | None = None) -> Encoded:
    ids = to_ids(text, lang, assets.indexer)
    length = length or len(ids)
    if len(ids) > length:
        raise ValueError(f"text has {len(ids)} ids, bucket holds {length}")
    return Encoded(text_ids=_pad(ids, length)[None], text_mask=_mask(len(ids), length))


def synthesize(
    stages: Stages,
    assets: Assets,
    text: str,
    lang: str,
    voice: Voice,
    *,
    noise_seed: int = 0,
    steps: int = 8,
    speed: float = 1.05,
    bucket: Bucket | None = None,
) -> Speech:
    """Synthesize one utterance, padding every input to `bucket` when given.

    The initial noise depends only on `noise_seed` and the utterance length, so padded and
    unpadded runs start from the same latent.
    """
    encoded = encode(text, lang, assets, bucket and bucket.text_length)
    text_inputs = {"text_ids": encoded.text_ids, "text_mask": encoded.text_mask}
    seconds = float(stages["duration_predictor"]({**text_inputs, "style_dp": voice.dp})[0]) / speed
    text_emb = stages["text_encoder"]({**text_inputs, "style_ttl": voice.ttl})

    samples = int(seconds * assets.sample_rate)
    frames = -(-samples // assets.samples_per_frame)
    padded_frames = bucket.latent_length if bucket else frames
    if frames > padded_frames:
        raise ValueError(f"utterance needs {frames} latent frames, bucket holds {padded_frames}")
    noise = np.random.default_rng(noise_seed).standard_normal((assets.latent_channels, frames), np.float32)
    latent = _pad(noise, padded_frames)[None]
    total = np.full(1, steps, np.float32)
    for step in range(steps):
        latent = stages["vector_estimator"](
            {
                "noisy_latent": latent,
                "text_emb": text_emb,
                "style_ttl": voice.ttl,
                "latent_mask": _mask(frames, padded_frames),
                "text_mask": encoded.text_mask,
                "current_step": np.full(1, step, np.float32),
                "total_step": total,
            }
        )
    wav = stages["vocoder"]({"latent": latent})
    return Speech(wav=wav[0, :samples], seconds=seconds)


def _pad(values: np.ndarray, length: int) -> np.ndarray:
    return np.pad(values, [(0, 0)] * (values.ndim - 1) + [(0, length - values.shape[-1])])


def _mask(valid: int, length: int) -> np.ndarray:
    return (np.arange(length) < valid).astype(np.float32).reshape(1, 1, length)
