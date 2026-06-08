from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    avg_confidence: float
    accuracy: float


def brier_score(confidences: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error between confidence and outcome (a proper scoring rule)."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(labels, dtype=float)
    return float(np.mean((c - y) ** 2))


def reliability_bins(
    confidences: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> list[ReliabilityBin]:
    """Bucket predictions by confidence; report avg confidence vs actual accuracy."""
    c = np.asarray(confidences, dtype=float)
    y = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (c >= lo) & (c < hi) if i < n_bins - 1 else (c >= lo) & (c <= hi)
        count = int(mask.sum())
        if count:
            bins.append(
                ReliabilityBin(
                    lo, hi, count, float(c[mask].mean()), float(y[mask].mean())
                )
            )
        else:
            bins.append(ReliabilityBin(lo, hi, 0, 0.0, 0.0))
    return bins


def expected_calibration_error(
    confidences: Sequence[float], labels: Sequence[int], n_bins: int = 10
) -> float:
    """ECE: weighted average gap between confidence and accuracy across bins."""
    n = len(confidences)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in reliability_bins(confidences, labels, n_bins):
        if b.count:
            ece += (b.count / n) * abs(b.avg_confidence - b.accuracy)
    return ece


_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(confidences: Sequence[float], labels: Sequence[int]) -> float:
    """Fit a temperature T > 0 that recalibrates confidences (minimises NLL)."""
    z = _logit(np.asarray(confidences, dtype=float))
    y = np.asarray(labels, dtype=float)
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.05, 20.0, 200):
        nll = _nll(_sigmoid(z / float(t)), y)
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def apply_temperature(confidences: Sequence[float], temperature: float) -> list[float]:
    """Rescale confidences by a fitted temperature (ranking unchanged)."""
    z = _logit(np.asarray(confidences, dtype=float))
    return [float(x) for x in _sigmoid(z / temperature)]
