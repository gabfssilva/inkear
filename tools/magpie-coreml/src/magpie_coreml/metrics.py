import re

import numpy as np


def words(text: str) -> list[str]:
    """Lowercased words, punctuation dropped."""
    return re.findall(r"[\w']+", text.lower())


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Word-level edit distance over the reference length."""
    hyp, ref = words(hypothesis), words(reference)
    row = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, 1):
        previous, row[0] = row[0], i
        for j, other in enumerate(hyp, 1):
            previous, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, previous + (word != other))
    return row[-1] / max(len(ref), 1)


def snr_db(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(10 * np.log10((expected**2).sum() / ((actual - expected) ** 2).sum()))
