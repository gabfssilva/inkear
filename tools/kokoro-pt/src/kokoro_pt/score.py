"""Phoneme error rate of a phonemizer against espeak-ng pt-br, both reduced to Kokoro's vocabulary."""

import json
import sys
import unicodedata
from collections.abc import Callable
from pathlib import Path

VOCAB = set(json.loads((Path(__file__).parent / "vocab.json").read_text()))


def kokoro(phonemes: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", phonemes) if c in VOCAB or c == " ")


def distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ca != cb))
    return row[-1]


def per(rows: list[tuple[str, str]], phonemize: Callable[[str], str], strip: str = "") -> float:
    errors = total = 0
    for text, reference in rows:
        ref = kokoro(reference).translate({ord(c): None for c in strip})
        hyp = kokoro(phonemize(text)).translate({ord(c): None for c in strip})
        errors += distance(hyp, ref)
        total += len(ref)
    return errors / total


def load(path: Path) -> list[tuple[str, str]]:
    return [tuple(line.split("\t")) for line in path.read_text().splitlines()]  # type: ignore[misc]


if __name__ == "__main__":
    from kokoro_pt import rules

    rows = load(Path(sys.argv[1]))
    print(f"PER {per(rows, rules.phonemize):.4f}  without stress {per(rows, rules.phonemize, 'ˈˌ'):.4f}")
