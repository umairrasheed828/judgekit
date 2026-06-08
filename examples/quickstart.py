"""End-to-end demo of judgekit with a real LLM.

uv sync --extra llm           # install the OpenAI adapter
# put OPENAI_API_KEY in a .env file
uv run python examples/quickstart.py
"""

from dotenv import load_dotenv

from judgekit import (
    Axis,
    LLMJudge,
    Sample,
    calibration_report,
    openai_complete,
    verbosity_bias,
)

load_dotenv()

CONTEXT = "Retrieval-augmented generation fetches documents and conditions generation on them."


def main() -> None:
    axes = [
        Axis("faithfulness", "every claim is supported by the context"),
        Axis("relevance", "the output answers the question"),
    ]
    judge = LLMJudge(axes, openai_complete())

    samples = [
        Sample(
            "What is RAG?",
            "RAG retrieves documents and grounds the answer in them.",
            CONTEXT,
        ),
        Sample("What is RAG?", "RAG is a fabric softener used in laundry.", CONTEXT),
    ]

    print("== Judgments ==")
    outputs, faithfulness = [], []
    for s in samples:
        j = judge.score(s)
        print(f"  {s.output[:42]!r:46} -> {j.scores}")
        outputs.append(s.output)
        faithfulness.append(j.scores["faithfulness"])

    print("\n== Bias (diagnostic) ==")
    print(" ", verbosity_bias(outputs, faithfulness).summary())

    print("\n== Calibration (illustrative confidence vs outcome) ==")
    confidences = [0.95, 0.9, 0.92, 0.88, 0.91, 0.93, 0.9, 0.94, 0.89, 0.9]
    outcomes = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]  # ~40% correct but ~91% confident
    print(" ", calibration_report(confidences, outcomes).summary())


if __name__ == "__main__":
    main()
