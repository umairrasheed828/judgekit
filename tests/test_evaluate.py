from judgekit import Judgment, Sample, evaluate


class ConstJudge:
    """A trivial judge that proves any object with .score() plugs in."""

    def __init__(self, value: int) -> None:
        self.value = value

    def score(self, sample: Sample) -> Judgment:
        return Judgment(scores={"quality": self.value}, rationale="")


def test_evaluate_aggregates() -> None:
    samples = [Sample(input=f"q{i}", output="a") for i in range(4)]
    result = evaluate(ConstJudge(3), samples)
    assert len(result.judgments) == 4
    assert result.mean_scores() == {"quality": 3.0}


def test_mean_scores_handles_empty() -> None:
    assert evaluate(ConstJudge(3), []).mean_scores() == {}
