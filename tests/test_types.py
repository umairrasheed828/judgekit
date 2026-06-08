from judgekit import Axis, Judgment, Sample


def test_sample_fields() -> None:
    s = Sample(input="q", output="a")
    assert s.input == "q"
    assert s.context is None


def test_axis_defaults() -> None:
    ax = Axis(name="faithfulness", description="grounded in context")
    assert (ax.min_score, ax.max_score) == (1, 5)


def test_judgment() -> None:
    j = Judgment(scores={"faithfulness": 5}, rationale="ok")
    assert j.scores["faithfulness"] == 5
