# judgekit

Calibrated, rigorous evaluation for LLM judges — a small, dependency-light Python library.

Most "LLM-as-judge" setups score an output and stop. `judgekit` adds the part that makes
evaluation *trustworthy*: it measures whether your judge is **calibrated**, **recalibrates**
it, checks it for **bias**, and reports results with **statistical rigor**.

## Install
\`\`\`bash
pip install judgekit          # core (zero heavy dependencies)
pip install judgekit[llm]     # + OpenAI adapter
\`\`\`

## Quickstart
\`\`\`python
from judgekit import Axis, LLMJudge, Sample, openai_complete

judge = LLMJudge(
    axes=[Axis("faithfulness", "claims supported by the context")],
    complete=openai_complete(),        # or any complete(prompt) -> str function
)
j = judge.score(Sample(
    input="What is RAG?",
    output="RAG retrieves documents and grounds the answer in them.",
    context="Retrieval-augmented generation conditions generation on fetched documents.",
))
print(j.scores)   # e.g. {'faithfulness': 5}
\`\`\`

## What makes it different: calibration
An LLM judge is just another model — usually overconfident. `judgekit` measures and fixes that:
\`\`\`python
from judgekit import calibration_report

report = calibration_report(confidences, outcomes)  # confidences in [0,1], outcomes in {0,1}
print(report.summary())
# n=200  ECE=0.180  Brier=0.210  T=2.40  ECE_after=0.040
\`\`\`
- **ECE / Brier / reliability bins** quantify miscalibration.
- **Temperature scaling** (`fit_temperature`, `apply_temperature`) recalibrates with one fitted scalar.

## Rigor & bias
\`\`\`python
from judgekit import bootstrap_ci, cohens_kappa, verbosity_bias

bootstrap_ci(scores)                # mean with a 95% confidence interval
cohens_kappa(judge_scores, humans)  # chance-corrected judge/human agreement
verbosity_bias(outputs, scores)     # does the judge just reward longer answers?
\`\`\`

## Design
- **Provider-agnostic** — `LLMJudge` takes any `complete(prompt) -> str`; OpenAI is an optional extra.
- **Pluggable** — anything with a `score(sample) -> Judgment` method is a `Judge`.
- **Light core** — core types are stdlib dataclasses; heavy deps (numpy, LLM SDK) are isolated.

## Development
\`\`\`bash
uv sync --dev
uv run ruff check . && uv run mypy src && uv run pytest
uv run python examples/quickstart.py   # end-to-end demo (needs OPENAI_API_KEY)
\`\`\`

## Status
Core judge API, calibration engine (ECE / Brier / temperature scaling), statistical rigor
(bootstrap CI, Cohen's kappa), and verbosity-bias detection are implemented and tested.
Planned: position-bias detection for pairwise judges, a results dashboard, and a PyPI release.