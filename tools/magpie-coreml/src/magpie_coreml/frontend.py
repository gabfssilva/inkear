"""Text front end: exported as plain files for Swift, and reimplemented here from those files only,
so tests can hold it to NeMo's tokenizer.

Steps, per language (pt = NeMo's portuguese_Brazilian_phoneme, en = english_phoneme):

1. pt: NFC, then U+2019 -> '. en: drop combining marks after NFD (strips accents), then
   U+2019 -> ', U+201C/U+201D -> ".
2. Split into runs with WORD_PATTERN: a word (letters, inner hyphens/apostrophes), a |...| run
   (unused here), or a run of anything else. en lowercases words at this point.
3. Each run, uppercased: no letter or digit -> its characters, one symbol each; en only: a
   heteronym -> graphemes; a dictionary word -> its first pronunciation, one symbol per Unicode
   scalar; en only: not in the dictionary but ending in 's or s with the stem in it -> the stem's
   pronunciation + s (stem ends in T), ɪ z ('s after S) or z; else, if it has hyphens, each part
   by these rules joined by "-"; else graphemes (grapheme_prefix + each character).
4. Keep symbols present in `tokens` (unknown ones are dropped), then drop trailing spaces;
   pt wraps the result in one space on each side. Repeated spaces are kept, as NeMo does.
5. id = offset + index in `tokens`; append the text EOS id.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import WORD_CHARS_ALL
from nemo.collections.tts.models import MagpieTTSModel

from magpie_coreml.checkpoint import TOKENIZERS

WORD_PATTERN = rf"([{WORD_CHARS_ALL}]+(?:[{WORD_CHARS_ALL}\-']*[{WORD_CHARS_ALL}]+)*)|(\|[^|]*\|)|([^{WORD_CHARS_ALL}|]+)"
EN_WORD_PATTERN = r"([A-Za-z]+(?:[A-Za-z\-']*[A-Za-z]+)*)|(\|[^|]*\|)|([^A-Za-z|]+)"
CHARS = re.compile(rf"[{WORD_CHARS_ALL}\d]")


def export(model: MagpieTTSModel, directory: Path) -> None:
    """Write tokenizer_<lang>.json, lexicon_<lang>.tsv (WORD<TAB>pronunciation) and, for en,
    heteronyms_en.txt."""
    for language, name in TOKENIZERS.items():
        tokenizer = model.tokenizer.tokenizers[name]
        g2p = tokenizer.g2p
        config = {
            "tokenizer": name,
            "offset": model.tokenizer.tokenizer_offsets[name],
            "eos": model.eos_id,
            "tokens": tokenizer.tokens,
            "punctuation": list(tokenizer.punct_list),
            "space": tokenizer.tokens[tokenizer.space],
            "pad_with_space": tokenizer.pad_with_space,
            "grapheme_prefix": g2p.grapheme_prefix,
            "word_pattern": EN_WORD_PATTERN if language == "en" else WORD_PATTERN,
            "lowercase_words": language == "en",
        }
        (directory / f"tokenizer_{language}.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=1)
        )
        with (directory / f"lexicon_{language}.tsv").open("w", encoding="utf-8") as lexicon:
            for word, prons in sorted(g2p.phoneme_dict.items()):
                lexicon.write(f"{word}\t{''.join(prons[0])}\n")
        if g2p.heteronyms:
            (directory / f"heteronyms_{language}.txt").write_text("\n".join(sorted(g2p.heteronyms)) + "\n")


@dataclass(frozen=True)
class Frontend:
    language: str
    offset: int
    eos: int
    ids: dict[str, int]
    space: str
    pad_with_space: bool
    prefix: str
    pattern: re.Pattern[str]
    lowercase: bool
    lexicon: dict[str, str]
    heteronyms: frozenset[str]

    @classmethod
    def load(cls, directory: Path, language: str) -> "Frontend":
        config = json.loads((directory / f"tokenizer_{language}.json").read_text())
        lines = (directory / f"lexicon_{language}.tsv").read_text(encoding="utf-8").splitlines()
        heteronyms = directory / f"heteronyms_{language}.txt"
        return cls(
            language=language,
            offset=config["offset"],
            eos=config["eos"],
            ids={token: index for index, token in enumerate(config["tokens"])},
            space=config["space"],
            pad_with_space=config["pad_with_space"],
            prefix=config["grapheme_prefix"],
            pattern=re.compile(config["word_pattern"]),
            lowercase=config["lowercase_words"],
            lexicon=dict(line.split("\t") for line in lines),
            heteronyms=frozenset(heteronyms.read_text().split()) if heteronyms.exists() else frozenset(),
        )

    def encode(self, text: str) -> list[int]:
        symbols: list[str] = []
        for word, unchanged, other in self.pattern.findall(self._normalize(text)):
            if unchanged:
                symbols += [self.prefix + part for part in unchanged[1:-1].split(" ")]
            else:
                symbols += self._word(word.lower() if word and self.lowercase else word or other)
        kept = [symbol for symbol in symbols if symbol in self.ids]
        while kept and kept[-1] == self.space:
            kept.pop()
        if self.pad_with_space:
            kept = [self.space, *kept, self.space]
        return [self.offset + self.ids[symbol] for symbol in kept] + [self.eos]

    def _normalize(self, text: str) -> str:
        if self.language == "en":
            text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
            text = text.replace("’", "'").replace("“", '"').replace("”", '"')
        else:
            text = unicodedata.normalize("NFC", text).replace("’", "'")
        return unicodedata.normalize("NFC", text)

    def _word(self, word: str) -> list[str]:
        word = word.upper()
        if CHARS.search(word) is None:
            return list(word)
        if word in self.heteronyms:
            return self._graphemes(word)
        if found := self._lookup(word):
            return found
        parts = word.split("-")
        if len(parts) == 1:
            return self._graphemes(word)
        symbols: list[str] = []
        for part in parts:
            symbols += [*self._word(part), "-"]
        return symbols[:-1]

    def _lookup(self, word: str) -> list[str]:
        """Dictionary pronunciation, with NeMo's en-US rules for 's and s suffixes."""
        if word in self.lexicon:
            return list(self.lexicon[word])
        if self.language != "en" or self.heteronyms and word in self.heteronyms:
            return []
        for suffix, voiced in (("'S", ["ɪ", "z"]), ("S", ["z"])):
            stem = word[: -len(suffix)]
            if len(word) > len(suffix) and word.endswith(suffix) and stem in self.lexicon:
                ending = (
                    ["s"]
                    if stem.endswith("T")
                    else voiced
                    if suffix == "'S" and stem.endswith("S")
                    else ["z"]
                )
                return list(self.lexicon[stem]) + ending
        return []

    def _graphemes(self, word: str) -> list[str]:
        return [self.prefix + character for character in word]
