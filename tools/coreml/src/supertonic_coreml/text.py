"""Text preprocessing, ported from the upstream reference (`py/helper.py`)."""

import re
from collections.abc import Sequence
from unicodedata import normalize

import numpy as np

LANGUAGES = frozenset(
    [
        "en",
        "ko",
        "ja",
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "es",
        "et",
        "fi",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "lt",
        "lv",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "tr",
        "uk",
        "vi",
        "na",
    ]
)


def to_ids(text: str, lang: str, indexer: Sequence[int]) -> np.ndarray:
    """Return the model's character ids for `text`, shape ``(len,)``.

    Characters outside the vocabulary are dropped: upstream maps them to -1, which the
    embedding Gather reads as the last row.
    """
    ids = (indexer[ord(c)] for c in preprocess(text, lang))
    return np.fromiter((i for i in ids if i >= 0), dtype=np.int64)


def preprocess(text: str, lang: str) -> str:
    if lang not in LANGUAGES:
        raise ValueError(f"unsupported language: {lang}")
    text = _EMOJI.sub("", normalize("NFKD", text))
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    text = re.sub(r" ([,.!?;:'])", r"\1", text)
    text = re.sub(r"\"{2,}", '"', text)
    text = re.sub(r"'{2,}", "'", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


_EMOJI = re.compile(
    "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff☀-⛿✀-➿\U0001f1e6-\U0001f1ff]+"
)

_REPLACEMENTS = (
    *((dash, "-") for dash in "–‑—"),
    *((quote, '"') for quote in "“”"),
    *((quote, "'") for quote in "‘’´`"),
    *((symbol, " ") for symbol in "_[]|/#→←"),
    ("@", " at "),
    ("e.g.,", "for example, "),
    ("i.e.,", "that is, "),
)
