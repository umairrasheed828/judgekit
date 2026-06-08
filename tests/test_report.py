from judgekit.calibration import calibration_report


def test_report_overconfident() -> None:
    rep = calibration_report([0.9] * 10, [1] * 7 + [0] * 3)
    assert rep.n == 10
    assert rep.temperature > 1.0
    assert rep.ece_after < rep.ece  # recalibration helped
    assert "ECE" in rep.summary()


def test_report_well_calibrated() -> None:
    rep = calibration_report([0.5, 0.5], [1, 0])
    assert rep.ece == 0.0
