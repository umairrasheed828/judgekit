from judgekit.bias import verbosity_bias


def test_verbosity_strong_positive() -> None:
    outputs = ["a", "aa", "aaa", "aaaa", "aaaaa"]
    scores = [1, 2, 3, 4, 5]  # longer -> higher
    assert verbosity_bias(outputs, scores).correlation > 0.9


def test_verbosity_none_when_score_constant() -> None:
    outputs = ["a", "aa", "aaa", "aaaa"]
    scores = [3, 3, 3, 3]
    assert verbosity_bias(outputs, scores).correlation == 0.0
