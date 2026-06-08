import json
from collections.abc import Callable

from judgekit.types import Axis, Judgment, Sample

Completer = Callable[[str], str]

_PROMPT = """You are a strict evaluator. Score the OUTPUT on each axis below using
whole integers in the given range. Judge each axis independently.

Axes:
{axes}

Return ONLY a JSON object of the form:
{{"scores": {{"<axis name>": <int>, ...}}, "rationale": "<one or two sentences>"}}

INPUT:
{input}

OUTPUT:
{output}
{extra}"""


def _format_axes(axes: list[Axis]) -> str:
    return "\n".join(
        f"- {a.name} ({a.min_score}-{a.max_score}): {a.description}" for a in axes
    )


def _extract_json(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in judge output: {text!r}")
    return json.loads(text[start : end + 1])


class LLMJudge:
    """A judge that asks an LLM to score a sample on configurable axes.

    Provider-agnostic: pass any `complete(prompt) -> str` function.
    """

    def __init__(self, axes: list[Axis], complete: Completer) -> None:
        self.axes = axes
        self.complete = complete

    def score(self, sample: Sample) -> Judgment:
        extra = ""
        if sample.context is not None:
            extra += f"\nCONTEXT (source of truth):\n{sample.context}"
        if sample.reference is not None:
            extra += f"\nREFERENCE (gold answer):\n{sample.reference}"
        prompt = _PROMPT.format(
            axes=_format_axes(self.axes),
            input=sample.input,
            output=sample.output,
            extra=extra,
        )
        data = _extract_json(self.complete(prompt))
        scores = {}
        for a in self.axes:
            val = int(data["scores"][a.name])
            scores[a.name] = max(a.min_score, min(a.max_score, val))  # clamp to range
        return Judgment(scores=scores, rationale=str(data.get("rationale", "")))
