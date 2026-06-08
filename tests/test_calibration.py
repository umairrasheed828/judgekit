from judgekit.calibration import (
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def test_brier_perfect() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_brier_known() -> None:
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25


def test_ece_overconfident() -> None:
    # always 0.9 confident, always wrong -> ECE ~ 0.9
    assert round(expected_calibration_error([0.9, 0.9, 0.9], [0, 0, 0]), 2) == 0.9


def test_ece_well_calibrated() -> None:
    # says 0.5, right half the time -> gap 0
    assert expected_calibration_error([0.5, 0.5], [1, 0]) == 0.0


def test_bins_count_conserved() -> None:
    bins = reliability_bins([0.1, 0.4, 0.95], [0, 1, 1])
    assert sum(b.count for b in bins) == 3
