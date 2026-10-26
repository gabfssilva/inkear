"""Reference prompts: public recordings cut to a few seconds and encoded once with the codec."""

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from moss_nano_coreml import reference

UPSTREAM_AUDIO = "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTS-Nano/8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb/assets/audio"
LIBRIVOX = "https://archive.org/download/dom_casmurro_2102_librivox"
PUBLIC_DOMAIN = "Public Domain Mark 1.0 (LibriVox), https://creativecommons.org/publicdomain/mark/1.0/"
UPSTREAM_LICENSE = (
    "Apache-2.0, as part of the OpenMOSS/MOSS-TTS-Nano repository; the speakers are not identified upstream"
)


@dataclass(frozen=True)
class Voice:
    name: str
    language: str
    description: str
    url: str
    sha256: str
    start: float
    end: float
    transcript: str
    license: str


VOICES = (
    Voice(
        name="rachel",
        language="pt-BR",
        description="Female, Brazilian Portuguese; LibriVox reader Rachel Moraes, Dom Casmurro ch. VI",
        url=f"{LIBRIVOX}/domcasmurro_02_assis.mp3",
        sha256="11736fd7fcaed4788e8f248f57b34a285cd802ecb640fb054815a316e1f51bec",
        start=26.62,
        end=35.95,
        transcript="Tio Cosme vivia com minha mãe desde que ela enviuvou. Já então era viúvo, como prima Justina."
        " Era a casa dos três viúvos.",
        license=PUBLIC_DOMAIN,
    ),
    Voice(
        name="thiago",
        language="pt-BR",
        description="Male, Brazilian Portuguese; LibriVox reader Thiago, Dom Casmurro ch. LXI",
        url=f"{LIBRIVOX}/domcasmurro_13_assis.mp3",
        sha256="68df6dc4f5077b2f32f75682dc3b79c8c516eb34f70d59273b414f2f6b82bd2f",
        start=81.70,
        end=88.95,
        transcript="E contava-me tudo isso cheio de uma admiração lacrimosa. Tio Cosme também se enternecia muito.",
        license=PUBLIC_DOMAIN,
    ),
    Voice(
        name="ava",
        language="en",
        description="Female, English; upstream preset 'Ava' (assets/audio/en_2.wav)",
        url=f"{UPSTREAM_AUDIO}/en_2.wav",
        sha256="",
        start=0.0,
        end=0.0,
        transcript="",
        license=UPSTREAM_LICENSE,
    ),
    Voice(
        name="adam",
        language="en",
        description="Male, English; upstream preset 'Adam' (assets/audio/en_4.wav)",
        url=f"{UPSTREAM_AUDIO}/en_4.wav",
        sha256="",
        start=0.0,
        end=0.0,
        transcript="",
        license=UPSTREAM_LICENSE,
    ),
)


def prepare(voice: Voice, downloads: Path, directory: Path) -> np.ndarray:
    """Download, cut and encode `voice`; writes <name>.wav (48 kHz stereo) and <name>.json (codes + source)."""
    source = downloads / voice.url.rsplit("/", 1)[1]
    if not source.exists():
        downloads.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(voice.url, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if voice.sha256 and digest != voice.sha256:
        raise ValueError(f"{source}: sha256 {digest}, expected {voice.sha256}")
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    if voice.end > voice.start:
        audio = audio[int(voice.start * rate) : int(voice.end * rate)]
    audio = resample_poly(audio, reference.SAMPLE_RATE // 100, rate // 100, axis=0).astype(np.float32)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    fade = int(0.01 * reference.SAMPLE_RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
    audio[:fade] *= ramp
    audio[-fade:] *= ramp[::-1]
    directory.mkdir(parents=True, exist_ok=True)
    wav = directory / f"{voice.name}.wav"
    sf.write(wav, audio, reference.SAMPLE_RATE, subtype="PCM_16")
    codes = reference.load().encode_audio(wav)
    record = asdict(voice) | {"sha256": digest, "frames": len(codes), "codes": codes.tolist()}
    (directory / f"{voice.name}.json").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    return codes


def load(directory: Path, name: str) -> np.ndarray:
    return np.array(json.loads((directory / f"{name}.json").read_text())["codes"], dtype=np.int32)
