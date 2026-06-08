from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 5000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean. Returns (mean, lower, upper)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    resamples = arr[rng.integers(0, arr.size, size=(n_resamples, arr.size))]
    means = resamples.mean(axis=1)
    alpha = (1 - confidence) / 2
    return (
        float(arr.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Agreement between two raters, corrected for agreement expected by chance."""
    ra, rb = np.asarray(a), np.asarray(b)
    if ra.size == 0:
        return 0.0
    po = float(np.mean(ra == rb))  # observed agreement
    pe = float(sum(np.mean(ra == c) * np.mean(rb == c) for c in np.union1d(ra, rb)))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
