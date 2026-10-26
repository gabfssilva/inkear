from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.tts.models import MagpieTTSModel

REPO = "nvidia/magpie_tts_multilingual_357m"
# main at the v2607 release plus the GGUF commits, which do not touch the .nemo file.
REVISION = "446bff55e1fe3d3aa29c771574e7d2b0fa322990"

SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1024
CODEBOOKS = 8
STACK = 2  # frame_stacking_factor: one decoder step predicts two codec frames
CODES = 16  # CODEBOOKS * STACK tokens per decoder step
VOCAB = 2024  # 2016 codec codes + 8 special tokens per codebook
AUDIO_BOS = 2016
AUDIO_EOS = 2017
# Special tokens that are never sampled: BOS, context BOS/EOS, MASK and the three reserved ids.
FORBIDDEN = (2016, 2018, 2019, 2020, 2021, 2022, 2023)
SPEAKERS = {"Aria": 0, "Jason": 1, "John": 2, "Leo": 3, "Sofia": 4}
TOKENIZERS = {"pt": "portuguese_Brazilian_phoneme", "en": "english_phoneme"}


@dataclass(frozen=True)
class Checkpoint:
    model: MagpieTTSModel
    path: Path


def load() -> Checkpoint:
    """The v2607 model on the CPU, in eval mode, with a deterministic text front end.

    The released config sets `phoneme_probability: 0.8`, a training-time augmentation that spells
    one word in five as graphemes at random. NeMo means to switch it off for inference
    (`setup_tokenizers(mode='test')` calls `set_phone_prob(1.0)`), but that method is a context
    manager the call never enters, so do_tts stays random. Clearing it gives the dictionary
    pronunciation for every known word, which is what the Swift front end reproduces.
    """
    path = Path(hf_hub_download(REPO, "magpie_tts_multilingual_357m.nemo", revision=REVISION))
    model = MagpieTTSModel.restore_from(str(path), map_location="cpu").eval()
    for name in TOKENIZERS.values():
        model.tokenizer.tokenizers[name].g2p.phoneme_probability = None
    torch.set_grad_enabled(False)
    return Checkpoint(model=model, path=path)


def text_ids(model: MagpieTTSModel, text: str, language: str) -> list[int]:
    """Token ids for `text`, ending with the text EOS id, as NeMo's single-chunk inference builds them."""
    return model.tokenizer.encode(text=text, tokenizer_name=TOKENIZERS[language]) + [model.eos_id]
