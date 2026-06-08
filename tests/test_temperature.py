from judgekit.calibration import (
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
)


def test_temperature_reduces_ece_when_overconfident() -> None:
    conf = [0.9] * 10
    labels = [1] * 7 + [0] * 3  # 70% accurate, but 90% confident -> overconfident
    before = expected_calibration_error(conf, labels)
    t = fit_temperature(conf, labels)
    after = expected_calibration_error(apply_temperature(conf, t), labels)
    assert t > 1.0  # softens the overconfidence
    assert after < before  # calibration improved


def test_apply_temperature_identity() -> None:
    conf = [0.3, 0.6, 0.8]
    out = apply_temperature(conf, 1.0)
    assert all(abs(a - b) < 1e-3 for a, b in zip(conf, out))
