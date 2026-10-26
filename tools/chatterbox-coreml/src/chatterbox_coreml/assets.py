import sys
import urllib.request
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "chatterbox"
CACHE = ROOT / "build" / "cache"

# The pt-BR single-language pack ships only T3 and S3Gen; its demo Space supplies the inference
# code and takes the voice encoder, tokenizer and built-in conditionals from the base repository.
PACK = ("ResembleAI/Chatterbox-Multilingual-pt-br", "b3952f18bc2eaa72b9bd7c17d2c4653bcad4770d")
BASE = ("ResembleAI/chatterbox", "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18")
SPACE = ("ResembleAI/Chatterbox-Multilingual-TTS-pt-br", "9e515821e826e207cd617a0fdd0223899ed108ea")

# The Space's default reference voice (`DEFAULT_CONFIG["audio"]` in its app.py).
VOICE_NAME = "pt_br_f2"
VOICE_URL = (
    "https://storage.googleapis.com/chatterbox-demo-samples/mtl-v3-single-language-prompts/pt-br/pt_br_f2.wav"
)

T3_FILE = "t3_pt_br.safetensors"
S3GEN_FILE = "s3gen_v3.pt"
TOKENIZER_FILE = "grapheme_mtl_merged_expanded_v1.json"


@dataclass(frozen=True)
class Checkpoint:
    """A directory laid out as the Space's `ChatterboxTTS.from_local` expects."""

    directory: Path
    voice_wav: Path

    @property
    def tokenizer(self) -> Path:
        return self.directory / TOKENIZER_FILE


@cache
def checkpoint() -> Checkpoint:
    directory = CACHE / "checkpoint"
    directory.mkdir(parents=True, exist_ok=True)
    files = {
        T3_FILE: hf_hub_download(PACK[0], T3_FILE, revision=PACK[1]),
        S3GEN_FILE: hf_hub_download(PACK[0], S3GEN_FILE, revision=PACK[1]),
        **{name: hf_hub_download(BASE[0], name, revision=BASE[1]) for name in ("ve.pt", TOKENIZER_FILE)},
    }
    for name, source in files.items():
        link = directory / name
        if not link.exists():
            link.symlink_to(source)
    voice = CACHE / f"{VOICE_NAME}.wav"
    if not voice.exists():
        urllib.request.urlretrieve(VOICE_URL, voice)
    return Checkpoint(directory, voice)


@cache
def import_space() -> None:
    """Make the Space's `chatterbox` package importable."""
    sys.path.insert(0, str(Path(snapshot_download(SPACE[0], repo_type="space", revision=SPACE[1])) / "chatterbox/src"))
