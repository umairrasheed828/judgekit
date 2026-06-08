from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class BiasReport:
    metric: str
    correlation: float
    n: int

    def summary(self) -> str:
        return f"{self.metric}: r={self.correlation:+.2f} over n={self.n}"


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based); tied values share the mean of their positions."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(a.shape[0], dtype=float)
    n = a.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and bool(sorted_a[j] == sorted_a[i]):
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0  # average rank for the tie group
        i = j
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation = Pearson on average ranks."""
    xr = _rankdata(x)
    yr = _rankdata(y)
    xr -= xr.mean()
    yr -= yr.mean()
    denom = float(np.sqrt((xr**2).sum() * (yr**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((xr * yr).sum() / denom)


def verbosity_bias(outputs: Sequence[str], scores: Sequence[int]) -> BiasReport:
    """Rank-correlation between output length and score.

    A strongly positive correlation suggests the judge may reward length
    regardless of quality (verbosity bias). It is a diagnostic signal, not
    proof: length can also legitimately track quality.
    """
    lengths = np.asarray([len(o) for o in outputs], dtype=float)
    r = _spearman(lengths, np.asarray(scores, dtype=float))
    return BiasReport(
        metric="verbosity (length vs score)", correlation=r, n=len(outputs)
    )
