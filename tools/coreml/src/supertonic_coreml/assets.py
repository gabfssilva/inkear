import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import numpy as np
from huggingface_hub import snapshot_download

REPO = "supertone-oss-archive/supertonic-3"
REVISION = "aafc6e32416a594460b32413efc49d7fe4ce6d46"

type StageName = Literal["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]
STAGES: tuple[StageName, ...] = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")


@dataclass(frozen=True)
class Voice:
    ttl: np.ndarray
    dp: np.ndarray


@dataclass(frozen=True)
class Assets:
    root: Path
    sample_rate: int
    samples_per_frame: int
    latent_channels: int
    indexer: Sequence[int]

    def onnx(self, stage: StageName) -> Path:
        return self.root / "onnx" / f"{stage}.onnx"

    def voice(self, name: str) -> Voice:
        style = json.loads((self.root / "voice_styles" / f"{name}.json").read_text())

        def tensor(key: str) -> np.ndarray:
            return np.asarray(style[key]["data"], dtype=np.float32).reshape(style[key]["dims"])

        return Voice(ttl=tensor("style_ttl"), dp=tensor("style_dp"))


@cache
def load() -> Assets:
    root = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["onnx/*", "voice_styles/*"]))
    config = json.loads((root / "onnx" / "tts.json").read_text())
    compress = config["ttl"]["chunk_compress_factor"]
    return Assets(
        root=root,
        sample_rate=config["ae"]["sample_rate"],
        samples_per_frame=config["ae"]["base_chunk_size"] * compress,
        latent_channels=config["ttl"]["latent_dim"] * compress,
        indexer=json.loads((root / "onnx" / "unicode_indexer.json").read_text()),
    )
