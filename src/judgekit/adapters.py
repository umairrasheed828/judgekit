from collections.abc import Callable


def openai_complete(
    model: str = "gpt-4o-mini", temperature: float = 0.0
) -> Callable[[str], str]:
    """A completion function backed by OpenAI.

    Requires the optional 'llm' extra (`pip install judgekit[llm]`) and an
    OPENAI_API_KEY environment variable.
    """
    from openai import OpenAI  # lazy import: the core library needs no openai

    client = OpenAI()

    def complete(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    return complete
