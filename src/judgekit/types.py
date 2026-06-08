from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    """One unit to be judged."""

    input: str  # the question/prompt the system was given
    output: str  # the system's response, which we judge
    context: str | None = None  # source material the output should be faithful to
    reference: str | None = None  # a gold/reference answer, if one exists


@dataclass(frozen=True)
class Axis:
    """One evaluation dimension, e.g. faithfulness."""

    name: str
    description: str
    min_score: int = 1
    max_score: int = 5


@dataclass
class Judgment:
    """A judge's verdict on a sample: a score per axis, plus reasoning."""

    scores: dict[str, int]  # axis name -> score
    rationale: str = ""
