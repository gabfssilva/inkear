"""CharsiuG2P word-by-word, run through FluidAudio's Core ML pair exactly as MultilingualG2PModel does."""

import re
import unicodedata
from functools import cache
from pathlib import Path

import coremltools as ct
import numpy as np

PREFIX = "<por-bz>: "
OFFSET = 3
EOS = 1


class Charsiu:
    def __init__(self, directory: Path) -> None:
        units = ct.ComputeUnit.CPU_ONLY
        self.encoder = ct.models.CompiledMLModel(str(directory / "MultilingualG2PEncoder.mlmodelc"), units)
        self.decoder = ct.models.CompiledMLModel(str(directory / "MultilingualG2PDecoder.mlmodelc"), units)
        self.word = cache(self._word)

    def _word(self, word: str) -> str:
        ids = np.array([[b + OFFSET for b in (PREFIX + word).encode()]], dtype=np.int32)
        mask = np.ones_like(ids)
        hidden = self.encoder.predict({"input_ids": ids, "attention_mask": mask})["last_hidden_state"]
        out: list[int] = []
        for _ in range(128):
            decoder_ids = np.array([[0] + out], dtype=np.int32)
            logits = self.decoder.predict(
                {
                    "decoder_input_ids": decoder_ids,
                    "encoder_hidden_states": hidden,
                    "encoder_attention_mask": mask,
                }
            )["logits"]
            best = int(np.argmax(logits[0, -1]))
            if best == EOS:
                break
            out.append(best)
        return bytes(t - OFFSET for t in out if 0 <= t - OFFSET <= 255).decode("utf-8", errors="ignore")


TIES = {"t͡ʃ": "ʧ", "d͡ʒ": "ʤ", "t͡s": "ʦ", "d͡z": "ʣ"}
IPA_VOWELS = set("aeiouɐɛɔəɪʊæyɨ")


def to_espeak(ipa: str, stressed: int, nuclei: int) -> str:
    """CharsiuG2P por-bz IPA in the espeak-ng pt-br conventions Kokoro was trained on, with stress marks."""
    s = unicodedata.normalize("NFD", ipa).replace("ˈ", "").replace("ˌ", "").replace(".", "")
    for old, new in TIES.items():
        s = s.replace(unicodedata.normalize("NFD", old), new)
    s = s.replace("͡", "")
    s = re.sub(r"[ʁχhɣ]", "x", s)
    # Charsiu writes coda r as a fricative; espeak pt-br as a tap plus schwa, or r at the end.
    s = re.sub(r"(?<=[aeiouɐɛɔ\u0303])x(?=[^aeiouɐɛɔəɪʊjw])", "ɾ", s)
    s = re.sub(r"(?<=[aeiouɐɛɔ\u0303])x$", "r", s)
    s = re.sub(r"(?<=[pbtdkɡfv])ɾ", "r", s)
    s = re.sub(r"ɾ$", "r", s)
    s = re.sub(r"ɾ(?=[^aeiouɐɛɔəɪʊ̃jw])", "ɾə", s)
    tilde = "̃"
    # nasal diphthongs and nasal vowels before consonants
    s = s.replace(f"ɐ{tilde}w{tilde}", "ɐ̃ʊ̃").replace(f"ɐ{tilde}w", "ɐ̃ʊ̃")
    s = s.replace(f"e{tilde}j{tilde}", "Aŋ").replace("ẽj", "Aŋ").replace(f"o{tilde}j{tilde}", "õj")
    s = s.replace(f"ɐ{tilde}j{tilde}", "ɐ̃y")
    s = re.sub(
        f"([eiouɐ]){tilde}(?=[pb])",
        lambda m: {"e": "A", "ɐ": "ɐ̃"}.get(m[1], m[1] + ("̃" if m[1] == "u" else "")) + "m",
        s,
    )
    s = re.sub(
        f"([eiouɐ]){tilde}(?![ʊjw])", lambda m: {"e": "A", "ɐ": "ɐ̃", "u": "ũ"}.get(m[1], m[1]) + "ŋ", s
    )
    s = s.replace("ŋm", "m").replace("ŋŋ", "ŋ")
    s = s.replace("aj", "I").replace("aw", "W").replace("ej", "A").replace("oj", "oɪ").replace("ew", "eʊ")
    s = s.replace("ɛw", "ɛʊ").replace("iw", "iʊ").replace("uj", "uɪ")
    # final unstressed vowels
    s = re.sub(r"ɐ(s?)$", r"æ\1", s)
    if stressed < nuclei - 1:
        s = re.sub(r"i(s?)$", r"y\1", s)
        s = re.sub(r"u(s?)$", r"ʊ\1", s)
    # stress before the vowel of the stressed nucleus
    out, k = "", 0
    marks = {stressed: "ˈ"}
    if stressed >= 2 or (stressed == 1 and stressed < nuclei - 1):
        marks[0] = "ˌ"
    for j in range(2, stressed - 1, 2):
        marks[j] = "ˌ"
    previous_vowel = False
    for c in s:
        vowel = c in IPA_VOWELS or c in "AIOWYæ"
        if vowel and not previous_vowel:
            out += marks.get(k, "")
            k += 1
        if c != tilde:
            previous_vowel = vowel
        out += c
    return out


def phonemize(text: str, charsiu: "Charsiu") -> str:
    from kokoro_pt import rules

    def word(part: str, level: str) -> str:
        if level != "primary" or part in rules.WORDS or not part.isalpha():
            return rules.phonemize_word(part, level)
        us = rules.units(part)
        groups = rules.nuclei(us)
        if not groups:
            return rules.phonemize_word(part, level)
        return to_espeak(charsiu.word(part), rules.stressed(part, us, groups), len(groups))

    return rules.phonemize(text, word)
