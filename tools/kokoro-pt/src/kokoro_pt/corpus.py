"""Picks Brazilian Portuguese sentences from the corpus PDFs for the phonemizer comparison."""

import re
import sys
from pathlib import Path

import pymupdf

DOCUMENTS = [
    "pt-book-dom-casmurro.pdf",
    "pt-book-bras-cubas.pdf",
    "pt-book-o-mulato.pdf",
    "pt-book-pro-git.pdf",
    "pt-paper-csp.pdf",
    "pt-paper-rbef.pdf",
    "pt-paper-rbh.pdf",
]
FIXED = [
    (
        "Uma noite destas, vindo da cidade para o Engenho Novo, encontrei no trem da Central um rapaz "
        "aqui do bairro, que eu conheço de vista e de chapéu."
    ),
    "Capítulo primeiro – Do título.",
]
PER_DOCUMENT = 5


def sentences(path: Path) -> list[str]:
    with pymupdf.open(path) as document:
        pages = [page.get_text() for page in document]
    text = " ".join(pages[len(pages) // 10 :])
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s+", " ", text)
    found = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ—–])", text)
    return [
        s.strip()
        for s in found
        if 50 <= len(s) <= 220
        and re.fullmatch(r"[\wÀ-ÿ ,.;:!?—–\-()\"'“”‘’]+", s.strip())
        and not re.search(r"[0-9]|http|www|\b[A-Z]{3,}\b|\b(?:the|and|of)\b", s)
    ]


def main() -> None:
    corpus = Path(sys.argv[1])
    out = Path(sys.argv[2])
    picked = list(FIXED)
    for name in DOCUMENTS:
        found = sentences(corpus / name)
        step = max(1, len(found) // PER_DOCUMENT)
        picked += found[step // 2 :: step][:PER_DOCUMENT]
    out.write_text("\n".join(picked) + "\n")
    print(f"{len(picked)} sentences -> {out}")


if __name__ == "__main__":
    main()
