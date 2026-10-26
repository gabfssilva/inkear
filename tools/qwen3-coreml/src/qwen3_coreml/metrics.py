"""Intelligibility and naturalness proxies: Whisper large-v3-turbo transcripts and UTMOS22."""

import re
import unicodedata
from functools import cache

import librosa
import mlx.core as mx
import mlx_whisper
import numpy as np
import torch
from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
from mlx_whisper.decoding import detect_language
from mlx_whisper.load_models import load_model
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.whisper import Whisper

WHISPER = "mlx-community/whisper-large-v3-turbo"
_WHISPER_RATE = 16_000


def _resampled(audio: np.ndarray, rate: int) -> np.ndarray:
    return librosa.resample(audio.astype(np.float32), orig_sr=rate, target_sr=_WHISPER_RATE)


def transcribe(audio: np.ndarray, rate: int, language: str) -> str:
    result = mlx_whisper.transcribe(
        _resampled(audio, rate), path_or_hf_repo=WHISPER, language=language, temperature=0.0
    )
    return str(result["text"]).strip()


@cache
def _whisper() -> Whisper:
    return load_model(WHISPER)


def language_probability(audio: np.ndarray, rate: int, language: str) -> float:
    """Whisper's probability that the first 30 s of `audio` are in `language`."""
    model = _whisper()
    mel = pad_or_trim(log_mel_spectrogram(_resampled(audio, rate), model.dims.n_mels), N_FRAMES, axis=-2)
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    _, probabilities = detect_language(model, mel.astype(mx.float16)[None], tokenizer)
    return float(probabilities[0][language])


def _words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    return re.sub(r"[^\w\s]", " ", text).split()


def wer(hypothesis: str, reference: str) -> float:
    """Word error rate, ignoring case and punctuation."""
    hyp, ref = _words(hypothesis), _words(reference)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1] / max(len(ref), 1)


@cache
def _utmos() -> torch.nn.Module:
    return torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()


def mos(audio: np.ndarray, rate: int) -> float:
    """UTMOS22 (strong) predicted MOS, 1-5. Trained on English; a rough proxy elsewhere."""
    with torch.inference_mode():
        return float(_utmos()(torch.from_numpy(_resampled(audio, rate))[None], _WHISPER_RATE)[0])


def log_spectral_distance(actual: np.ndarray, expected: np.ndarray, floor_db: float = 80.0) -> float:
    """Mean log-spectral distance in dB, ignoring bins `floor_db` below the loudest one."""

    def spectrum(wav: np.ndarray) -> np.ndarray:
        frames = np.lib.stride_tricks.sliding_window_view(wav, 2048)[::512] * np.hanning(2048)
        return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=-1)) + 1e-9)

    length = min(len(actual), len(expected))
    a, e = spectrum(actual[:length]), spectrum(expected[:length])
    floor = e.max() - floor_db
    return float(np.sqrt(((np.maximum(a, floor) - np.maximum(e, floor)) ** 2).mean(axis=1)).mean())
