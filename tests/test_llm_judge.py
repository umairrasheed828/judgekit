from judgekit import Axis, Judge, LLMJudge, Sample


def test_parses_scores() -> None:
    judge = LLMJudge(
        [Axis("faithfulness", "grounded"), Axis("relevance", "on topic")],
        lambda _p: '{"scores": {"faithfulness": 5, "relevance": 4}, "rationale": "ok"}',
    )
    j = judge.score(Sample(input="q", output="a"))
    assert j.scores == {"faithfulness": 5, "relevance": 4}
    assert j.rationale == "ok"


def test_clamps_out_of_range() -> None:
    judge = LLMJudge(
        [Axis("quality", "overall")],
        lambda _p: '{"scores": {"quality": 9}, "rationale": ""}',
    )
    assert judge.score(Sample(input="q", output="a")).scores["quality"] == 5


def test_implements_protocol() -> None:
    judge = LLMJudge([Axis("q", "d")], lambda _p: '{"scores":{"q":3},"rationale":""}')
    assert isinstance(judge, Judge)
