from judgekit.stats import bootstrap_ci, cohens_kappa


def test_bootstrap_ci_brackets_mean() -> None:
    mean, lo, hi = bootstrap_ci([4, 4, 5, 3, 4, 5, 4])
    assert lo <= mean <= hi


def test_bootstrap_ci_deterministic() -> None:
    assert bootstrap_ci([1, 2, 3, 4, 5]) == bootstrap_ci([1, 2, 3, 4, 5])


def test_kappa_perfect_agreement() -> None:
    assert cohens_kappa([1, 2, 3, 1, 2], [1, 2, 3, 1, 2]) == 1.0


def test_kappa_chance_level() -> None:
    # agree exactly as often as chance predicts -> kappa = 0
    assert cohens_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == 0.0
