"""Checkpoints, reference voices and the sample texts."""

import hashlib
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
from qwen_tts import Qwen3TTSModel

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "build" / "cache"

BASE = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SAMPLE_RATE = 24_000

TEXTS = {
    "pt": (
        "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz "
        "aqui do bairro, que eu conheço de vista e de chapéu."
    ),
    "en": (
        "The quick brown fox jumps over the lazy dog, and then it runs back into the forest before "
        "anyone notices."
    ),
}
# Codec language names of the checkpoints' `codec_language_id`, by ISO 639-1 code.
LANGUAGES = {"pt": "portuguese", "en": "english"}


@dataclass(frozen=True)
class VoiceSource:
    """A stretch of a public-domain recording whose speaker embedding becomes a Base voice."""

    name: str
    language: str
    description: str
    url: str
    sha256: str
    start: float
    end: float
    transcript: str
    license: str = "Public Domain Mark 1.0 (LibriVox), https://creativecommons.org/publicdomain/mark/1.0/"


VOICES = (
    VoiceSource(
        name="thiago",
        language="pt",
        description="Male, Brazilian Portuguese; LibriVox reader Thiago, Dom Casmurro ch. LXI",
        url="https://archive.org/download/dom_casmurro_2102_librivox/domcasmurro_13_assis.mp3",
        sha256="68df6dc4f5077b2f32f75682dc3b79c8c516eb34f70d59273b414f2f6b82bd2f",
        start=25.86,
        end=43.92,
        transcript=(
            "O mais foi muito. Vi saírem os primeiros dias da separação, duros e opacos, sem embargo "
            "das palavras de conforto que me deram os padres e os seminaristas, e as de minha mãe e "
            "tio Cosme, trazidas por José Dias ao seminário."
        ),
    ),
    VoiceSource(
        name="peter",
        language="en",
        description="Male, English; LibriVox reader Peter Yearsley, Alice's Adventures in Wonderland ch. I",
        url="https://archive.org/download/alice_wonderland_0711_librivox/alice_01_carroll_64kb.mp3",
        sha256="9cf058756aa0a2311f63e06ad94110d37b0ab6a2dfe33f49f51f9c495a954fd7",
        start=16.18,
        end=37.14,
        transcript=(
            "Alice was beginning to get very tired of sitting by her sister on the bank, and of having "
            "nothing to do. Once or twice she had peeped into the book her sister was reading, but it had "
            "no pictures or conversations in it. And what is the use of a book, thought Alice, without "
            "pictures or conversation?"
        ),
    ),
)


def load(repo: str) -> Qwen3TTSModel:
    """The PyTorch reference, fp32 on the CPU."""
    return Qwen3TTSModel.from_pretrained(repo, dtype=torch.float32, device_map="cpu")


def reference_audio(source: VoiceSource) -> np.ndarray:
    """The voice's stretch of its recording, mono at 24 kHz, downloaded once into the cache."""
    path = CACHE / Path(source.url).name
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source.url) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise ValueError(f"{path} does not match the recorded sha256")
    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)[int(source.start * rate) : int(source.end * rate)]
    return librosa.resample(audio, orig_sr=rate, target_sr=SAMPLE_RATE)


def speaker_embedding(model: Qwen3TTSModel, source: VoiceSource) -> np.ndarray:
    """The Base checkpoint's x-vector of the voice, `[1024]` float32."""
    with torch.inference_mode():
        embedding = model.model.extract_speaker_embedding(reference_audio(source), SAMPLE_RATE)
    return embedding.float().numpy()
