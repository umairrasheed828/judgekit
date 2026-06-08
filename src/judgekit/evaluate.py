from dataclasses import dataclass
from statistics import mean

from judgekit.base import Judge
from judgekit.types import Sample


@dataclass
class EvalResult:
    samples: list
    judgments: list

    def mean_scores(self) -> dict[str, float]:
        """Average score per axis across all judgments."""
        if not self.judgments:
            return {}
        axes = self.judgments[0].scores.keys()
        return {ax: mean(j.scores[ax] for j in self.judgments) for ax in axes}


def evaluate(judge: Judge, samples: list[Sample]) -> EvalResult:
    """Run a judge over many samples and collect the results."""
    judgments = [judge.score(s) for s in samples]
    return EvalResult(samples=samples, judgments=judgments)
