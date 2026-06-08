from typing import Protocol, runtime_checkable

from judgekit.types import Judgment, Sample


@runtime_checkable
class Judge(Protocol):
    """Anything that can score a Sample. Implement `score` to make a new judge."""

    def score(self, sample: Sample) -> Judgment: ...
