import re
import unicodedata
from functools import cache

import mlx_whisper
import numpy as np
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

WHISPER = "mlx-community/whisper-large-v3-turbo"
ACCENT = "lgris/portuguese-accent-classifier"


def words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def wer(hypothesis: str, reference: str) -> float:
    """Word error rate after lowercasing and dropping punctuation."""
    h, r = words(hypothesis), words(reference)
    row = list(range(len(h) + 1))
    for i, word in enumerate(r, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(h, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1] / max(len(r), 1)


def transcribe(audio: np.ndarray, rate: int, language: str) -> str:
    """Whisper large-v3-turbo (MLX) on the mono mix."""
    pcm = resample_poly(audio.mean(0), 16000 // 1000, rate // 1000).astype(np.float32)
    return str(mlx_whisper.transcribe(pcm, path_or_hf_repo=WHISPER, language=language)["text"]).strip()


@cache
def _accent() -> tuple[AutoFeatureExtractor, Wav2Vec2ForSequenceClassification]:
    return AutoFeatureExtractor.from_pretrained(ACCENT), Wav2Vec2ForSequenceClassification.from_pretrained(
        ACCENT
    ).eval()


def brazilian(audio: np.ndarray, rate: int) -> float:
    """Mean P(pt_br) of lgris/portuguese-accent-classifier (pt_br vs pt_pt) over 5 s windows."""
    extractor, model = _accent()
    pcm = resample_poly(audio.mean(0), 16000 // 1000, rate // 1000).astype(np.float32)
    window = 5 * 16000
    probabilities = []
    for start in range(0, max(len(pcm) - 16000, 1), window):
        inputs = extractor(pcm[start : start + window], sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            probabilities.append(float(torch.softmax(model(**inputs).logits, -1)[0, 0]))
    return float(np.mean(probabilities))


def snr(actual: np.ndarray, expected: np.ndarray) -> float:
    length = min(actual.shape[-1], expected.shape[-1])
    a, e = actual[..., :length], expected[..., :length]
    return float(10 * np.log10((e**2).sum() / ((a - e) ** 2).sum()))


def relative(actual: np.ndarray, expected: np.ndarray) -> float:
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    return float(np.linalg.norm(a - e) / np.linalg.norm(e))


def cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    a, e = np.asarray(actual, np.float64).ravel(), np.asarray(expected, np.float64).ravel()
    return float(a @ e / np.linalg.norm(a) / np.linalg.norm(e))
