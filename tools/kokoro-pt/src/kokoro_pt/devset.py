"""Writes a development set of corpus sentences, disjoint from the test sentences, with espeak-ng phonemes."""

import random
import sys
from pathlib import Path

from misaki import espeak

from kokoro_pt.corpus import DOCUMENTS, sentences


def main() -> None:
    corpus, test, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    excluded = set(test.read_text().splitlines())
    pool = sorted({s for name in DOCUMENTS for s in sentences(corpus / name)} - excluded)
    picked = random.Random(7).sample(pool, 600)
    g2p = espeak.EspeakG2P(language="pt-br")
    out.write_text("".join(f"{s}\t{g2p(s)[0]}\n" for s in picked))
    print(f"{len(picked)} of {len(pool)} sentences -> {out}")


if __name__ == "__main__":
    main()
