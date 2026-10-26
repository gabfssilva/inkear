from collections.abc import Mapping
from functools import cache
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
from coremltools.models.model import MLState

from magpie_coreml.checkpoint import SAMPLES_PER_FRAME, Checkpoint
from magpie_coreml.modules import (
    CACHE,
    CONTEXT,
    HEAD_DIM,
    HEADS,
    DecoderPrefill,
    DecoderStep,
    LocalTransformer,
    TextEncoder,
)
from magpie_coreml.pipeline import latents

CODEC_FRAMES = 32  # frames per codec call; longer windows push activations past what the ANE plans


class TorchStages:
    """The traceable modules run eagerly in PyTorch fp32: the conversion source, checked against NeMo.
    Decoding goes through NeMo's codec."""

    def __init__(self, checkpoint: Checkpoint):
        model = checkpoint.model
        self.codec = model._codec_helper
        self.encoder = TextEncoder(model).eval()
        self.prefiller = DecoderPrefill(model).eval()
        self.stepper = DecoderStep(model, torch.float32).eval()
        self.sampler = LocalTransformer(model).eval()
        self.layers = len(model.decoder.layers)
        self.cross: tuple[torch.Tensor, ...] = ()

    def encode(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.encoder(torch.from_numpy(tokens), torch.from_numpy(mask)).numpy()

    def prefill(self, speaker: int, encoder_output: np.ndarray, mask: np.ndarray) -> None:
        out = self.prefiller(
            torch.tensor([speaker], dtype=torch.int32),
            torch.from_numpy(encoder_output),
            torch.from_numpy(mask),
        )
        n = self.layers
        for i in range(n):
            for name, value in (("k", out[i]), ("v", out[n + i])):
                state = getattr(self.stepper, f"sa_{name}_{i}")
                state.zero_()
                state[:, :, :CONTEXT] = value
        self.cross = out[2 * n :]

    def step(
        self, audio_emb: np.ndarray, position: int, mask: np.ndarray, prior: np.ndarray, cfg_scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, h, alignment = self.stepper(
            torch.from_numpy(audio_emb),
            torch.tensor([position], dtype=torch.int32),
            torch.from_numpy(mask),
            torch.from_numpy(prior[None]),
            torch.tensor([cfg_scale]),
            *self.cross,
        )
        return logits[0, 0].numpy(), h.numpy(), alignment[0].numpy()

    def sample(
        self, h_last: np.ndarray, gumbel: np.ndarray, temperature: float, cfg_scale: float, forbid_eos: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        codes, audio_emb = self.sampler(
            torch.from_numpy(h_last),
            torch.from_numpy(gumbel),
            torch.tensor([temperature]),
            torch.tensor([cfg_scale]),
            torch.tensor([float(forbid_eos)]),
        )
        return codes[0].numpy().astype(np.int32), audio_emb.numpy()

    def decode(self, codes: np.ndarray) -> np.ndarray:
        tokens = torch.from_numpy(codes.T[None].astype(np.int64))
        return self.codec.codes_to_audio(tokens, torch.tensor([len(codes)]))[0][0].numpy()


class CoreMLStages:
    """The converted packages, run through Core ML."""

    def __init__(self, directory: Path, compute_units: ct.ComputeUnit):
        def load(name: str) -> ct.models.MLModel:
            return ct.models.MLModel(str(directory / f"{name}.mlpackage"), compute_units=compute_units)

        self.encoder = load("text_encoder")
        self.prefiller = load("decoder_prefill")
        self.stepper = load("decoder_step")
        self.sampler = load("local_transformer")
        self.codec = load("nanocodec_decoder")
        self.histories = {
            feature.name: tuple(feature.type.multiArrayType.shape)
            for feature in self.codec.get_spec().description.input
            if feature.name.startswith("history_")
        }
        self.state = self.stepper.make_state()
        self.cross: dict[str, np.ndarray] = {}

    def encode(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return _predict(self.encoder, {"tokens": tokens, "mask": mask})["encoder_output"]

    def prefill(self, speaker: int, encoder_output: np.ndarray, mask: np.ndarray) -> None:
        out = _predict(
            self.prefiller,
            {
                "speaker_idx": np.array([speaker], np.int32),
                "encoder_output": encoder_output,
                "encoder_mask": mask,
            },
        )
        self.state = self.stepper.make_state()
        for name, value in out.items():
            if name.startswith("sa_"):
                cache = np.zeros((2, HEADS, CACHE, HEAD_DIM), np.float32)  # write_state takes fp32 only
                cache[:, :, :CONTEXT] = value
                self.state.write_state(name, cache)
        self.cross = {name: value for name, value in out.items() if name.startswith("xa_")}

    def step(
        self, audio_emb: np.ndarray, position: int, mask: np.ndarray, prior: np.ndarray, cfg_scale: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = _predict(
            self.stepper,
            {
                "audio_emb": audio_emb,
                "position": np.array([position], np.int32),
                "encoder_mask": mask,
                "attn_prior": prior[None],
                "cfg_scale": np.array([cfg_scale], np.float32),
                **self.cross,
            },
            self.state,
        )
        return out["logits"][0, 0], out["h_last"], out["alignment"][0]

    def sample(
        self, h_last: np.ndarray, gumbel: np.ndarray, temperature: float, cfg_scale: float, forbid_eos: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        out = _predict(
            self.sampler,
            {
                "h_last": h_last,
                "gumbel": gumbel,
                "temperature": np.array([temperature], np.float32),
                "cfg_scale": np.array([cfg_scale], np.float32),
                "forbid_eos": np.array([float(forbid_eos)], np.float32),
            },
        )
        return out["codes"][0].astype(np.int32), out["audio_emb"]

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Codec calls of CODEC_FRAMES frames, zero latents after the end, histories carried over."""
        frames = latents(codes)
        count = frames.shape[-1]
        frames = np.pad(frames, ((0, 0), (0, 0), (0, -count % CODEC_FRAMES)))
        histories = {name: np.zeros(shape, np.float16) for name, shape in self.histories.items()}
        audio = []
        for start in range(0, frames.shape[-1], CODEC_FRAMES):
            out = _predict(self.codec, {"latents": frames[..., start : start + CODEC_FRAMES], **histories})
            audio.append(out["audio"][0])
            histories = {name: out[f"{name}_out"] for name in histories}
        return np.concatenate(audio)[: count * SAMPLES_PER_FRAME]


# coremltools 9.0 hands numpy inputs to Core ML without copying, and Core ML drops them later on
# its own queue without the GIL, corrupting the Python heap (apple/coremltools#2829, fixed after
# 9.0). Keeping a reference to every input means that release never frees.
_PINNED: list[Mapping[str, np.ndarray]] = []

_DTYPES = {
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32: np.float32,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16: np.float16,
    ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32: np.int32,
}


def _predict(
    model: ct.models.MLModel, inputs: Mapping[str, np.ndarray], state: MLState | None = None
) -> dict[str, np.ndarray]:
    """Predict with every input cast to its declared type and kept alive."""
    types = _input_types(model)
    pinned = {name: np.ascontiguousarray(value, dtype=types[name]) for name, value in inputs.items()}
    _PINNED.append(pinned)
    return model.predict(pinned, state=state)


@cache
def _input_types(model: ct.models.MLModel) -> Mapping[str, type[np.generic]]:
    return {
        feature.name: _DTYPES[feature.type.multiArrayType.dataType]
        for feature in model.get_spec().description.input
    }
