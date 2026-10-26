"""Evaluation of synthesized speech: Whisper transcripts, WER, UTMOS, pt-BR accent, speaker similarity."""

import re
import unicodedata
from collections.abc import Sequence
from functools import cache
from pathlib import Path

import mlx_whisper
import numpy as np
import soundfile as sf
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from omnivoice.eval.models.utmos import UTMOS22Strong
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    WavLMForXVector,
)

WHISPER = "mlx-community/whisper-large-v3-turbo"
ACCENT = "lgris/portuguese-accent-classifier"
SPEAKER = "microsoft/wavlm-base-plus-sv"


def transcribe(path: Path, language: str) -> str:
    return mlx_whisper.transcribe(str(path), path_or_hf_repo=WHISPER, language=language)["text"].strip()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate after lowercasing and dropping punctuation."""
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1] / max(len(ref), 1)


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s'-]", " ", text).split()


@torch.inference_mode()
def utmos(path: Path) -> float:
    """UTMOS22 strong learner: predicted MOS, 1 to 5."""
    return float(_utmos()(_load(path, 16_000)[None], 16_000))


@cache
def _utmos() -> UTMOS22Strong:
    model = UTMOS22Strong()
    checkpoint = hf_hub_download("k2-fsa/TTS_eval_models", "mos/utmos22_strong_step7459_v1.pt")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()


@torch.inference_mode()
def brazilian(path: Path) -> float:
    """Probability that the speech has a Brazilian (vs European) Portuguese accent, mean over 5 s windows."""
    extractor, model = _accent()
    audio = _load(path, 16_000).numpy()
    window = 5 * 16_000
    windows = [audio[i : i + window] for i in range(0, max(1, len(audio) - window // 2), window)]
    probabilities = []
    for chunk in (w for w in windows if len(w) >= 16_000):
        inputs = extractor(chunk, sampling_rate=16_000, return_tensors="pt")
        probabilities.append(torch.softmax(model(**inputs).logits, -1)[0, 0].item())
    return float(np.mean(probabilities))


@cache
def _accent() -> tuple[Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification]:
    return (
        Wav2Vec2FeatureExtractor.from_pretrained(ACCENT),
        Wav2Vec2ForSequenceClassification.from_pretrained(ACCENT).eval(),
    )


def speaker_similarity(paths: Sequence[Path], reference: Path) -> list[float]:
    """Cosine similarity of WavLM x-vectors between each of `paths` and `reference`."""
    anchor = _embedding(reference)
    return [float(torch.cosine_similarity(_embedding(p), anchor, dim=-1)) for p in paths]


@torch.inference_mode()
def _embedding(path: Path) -> torch.Tensor:
    extractor, model = _speaker()
    inputs = extractor(_load(path, 16_000).numpy(), sampling_rate=16_000, return_tensors="pt")
    return model(**inputs).embeddings[0]


@cache
def _speaker() -> tuple[Wav2Vec2FeatureExtractor, WavLMForXVector]:
    return Wav2Vec2FeatureExtractor.from_pretrained(SPEAKER), WavLMForXVector.from_pretrained(SPEAKER).eval()


def _load(path: Path, rate: int) -> torch.Tensor:
    audio, original = sf.read(path, dtype="float32", always_2d=True)
    return torchaudio.functional.resample(torch.from_numpy(audio.mean(axis=1)), original, rate)


def log_spectral_distance(actual: np.ndarray, expected: np.ndarray, floor_db: float = 80.0) -> float:
    """Mean log-spectral distance in dB over the common length, ignoring bins `floor_db` below the peak."""

    def spectrum(wav: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(wav, 2048)[::512] * np.hanning(2048)
        return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)) + 1e-9)

    length = min(len(actual), len(expected))
    a, e = spectrum(actual[:length]), spectrum(expected[:length])
    floor = e.max() - floor_db
    return float(np.sqrt(((np.maximum(a, floor) - np.maximum(e, floor)) ** 2).mean(axis=1)).mean())
