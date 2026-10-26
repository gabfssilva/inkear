"""The pt-BR pack's own PyTorch inference (the demo Space's code), on the CPU in fp32."""

from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np
import torch

from chatterbox_coreml import assets

SAMPLE_RATE = 24_000
SAMPLES_PER_TOKEN = 960
EXAGGERATION = 0.5


class Model(Protocol):
    """The parts of the Space's `ChatterboxTTS` this package uses."""

    t3: torch.nn.Module
    s3gen: torch.nn.Module
    tokenizer: object
    conds: object

    def prepare_conditionals(self, wav_fpath: str, exaggeration: float = 0.5) -> None: ...


@dataclass(frozen=True)
class Synthesis:
    tokens: np.ndarray
    wav: np.ndarray

    @property
    def seconds(self) -> float:
        return len(self.wav) / SAMPLE_RATE


@cache
def model() -> Model:
    """The pack with its default voice prepared at the default exaggeration.

    The flow decoder draws its initial noise once, at construction, so the seed fixes it.
    """
    checkpoint = assets.checkpoint()
    assets.import_space()
    from chatterbox.tts import ChatterboxTTS

    torch.manual_seed(0)
    tts = ChatterboxTTS.from_local(checkpoint.directory, "cpu")
    tts.prepare_conditionals(str(checkpoint.voice_wav), exaggeration=EXAGGERATION)
    return tts


def synthesize(text: str, language: str, seed: int) -> Synthesis:
    """`ChatterboxTTS.generate` with the Space's defaults, minus the Perth watermark."""
    tts = model()
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    from chatterbox.tts import punc_norm

    hp = tts.t3.hp
    ids = tts.tokenizer.text_to_tokens(punc_norm(text), language_id=language)
    ids = torch.nn.functional.pad(torch.cat([ids, ids]), (1, 0), value=hp.start_text_token)
    ids = torch.nn.functional.pad(ids, (0, 1), value=hp.stop_text_token)
    torch.manual_seed(seed)
    with torch.inference_mode():
        tokens = tts.t3.inference(
            t3_cond=tts.conds.t3, text_tokens=ids, max_new_tokens=1000, temperature=0.8, cfg_weight=0.5
        )[0]
        tokens = drop_invalid_tokens(tokens)
        wav, _ = tts.s3gen.inference(speech_tokens=tokens, ref_dict=tts.conds.gen)
    wav = wav.squeeze(0).numpy()
    return Synthesis(tokens.numpy(), wav[: max(1, len(tokens) - 1) * SAMPLES_PER_TOKEN])
