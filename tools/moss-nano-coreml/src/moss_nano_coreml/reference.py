"""The upstream PyTorch model and codec, loaded from pinned Hugging Face revisions."""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM

MODEL_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M"
MODEL_REVISION = "44502f80dbf9743528fa921cc544d662c685ebec"
CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano"
CODEC_REVISION = "6aa02b01e445cc585582cf0ba480bc3ea6c8dd68"
SAMPLE_RATE = 48000
SAMPLES_PER_FRAME = 3840


@dataclass(frozen=True)
class Sampling:
    """Upstream `infer.py` defaults; the text layer only chooses between continue and end."""

    text_temperature: float = 1.0
    audio_temperature: float = 0.8
    audio_top_k: int = 25
    audio_top_p: float = 0.95
    audio_repetition_penalty: float = 1.2


@dataclass(frozen=True)
class Reference:
    model: nn.Module
    codec: nn.Module
    root: Path

    def tokenizer(self) -> nn.Module:
        return self.model._load_text_tokenizer()

    def encode_audio(self, path: Path) -> np.ndarray:
        """Reference audio -> codes [frames, 16], as upstream voice cloning prepares it."""
        waveform, rate = self.model._load_reference_audio(str(path), SAMPLE_RATE, 2)
        with torch.no_grad():
            encoded = self.model._call_audio_encode(self.codec, waveform, rate)
        return self.model._normalize_audio_codes(encoded).numpy().astype(np.int32)

    def prompt(self, text: str, voice: np.ndarray) -> np.ndarray:
        """Voice-clone prompt rows [rows, 17]."""
        rows, _ = self.model.build_inference_input_ids(
            text=text,
            text_tokenizer=self.tokenizer(),
            mode="voice_clone",
            prompt_audio_codes=torch.from_numpy(voice).long(),
        )
        return rows[0].numpy().astype(np.int32)

    def generate(
        self, text: str, voice: np.ndarray, seed: int, sampling: Sampling, max_frames: int = 375, nq: int = 16
    ) -> np.ndarray:
        """Sampled codes [frames, 16] from the upstream generation loop (pad id past `nq`)."""
        rows = torch.from_numpy(self.prompt(text, voice)).long()[None]
        torch.manual_seed(seed)
        output = self.model.generate(
            rows,
            torch.ones(rows.shape[:2], dtype=torch.bool),
            nq=nq,
            max_new_frames=max_frames,
            do_sample=True,
            text_temperature=sampling.text_temperature,
            text_top_p=1.0,
            text_top_k=50,
            audio_temperature=sampling.audio_temperature,
            audio_top_p=sampling.audio_top_p,
            audio_top_k=sampling.audio_top_k,
            audio_repetition_penalty=sampling.audio_repetition_penalty,
        )
        return output.audio_token_ids[0].numpy().astype(np.int32)

    def decode(self, codes: np.ndarray, nq: int = 16) -> np.ndarray:
        """Codes [frames, 16] -> stereo audio [2, samples] at 48 kHz."""
        with torch.no_grad():
            decoded = self.model._call_audio_decode(
                self.codec, torch.from_numpy(codes).long(), SAMPLE_RATE, nq
            )
        audio, _ = self.model._extract_waveform_and_sample_rate(decoded, fallback_sample_rate=SAMPLE_RATE)
        return audio.numpy()


@cache
def load() -> Reference:
    root = Path(snapshot_download(MODEL_REPO, revision=MODEL_REVISION))
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True).eval()
    model._set_attention_implementation("sdpa")
    codec_root = snapshot_download(CODEC_REPO, revision=CODEC_REVISION)
    codec = AutoModel.from_pretrained(codec_root, trust_remote_code=True).eval()
    return Reference(model=model, codec=codec, root=root)
