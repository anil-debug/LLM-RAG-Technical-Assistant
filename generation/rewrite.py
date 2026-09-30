"""Rewrite a follow-up into a standalone retrieval query.

Conversation history is not the retrieval query. A follow-up such as
"How is authentication handled?" is embedded only after it has been rewritten
with the missing subject. The generator still receives the original history.
"""

from generation.prompt import REWRITE_SYSTEM
from generation.provider import LLMProvider


def rewrite_query(question: str, history: list[dict[str, str]], llm: LLMProvider | None) -> str:
    """Return ``question`` unchanged when there is no history.

    When history exists, the provider must return a short standalone query.
    An empty or very long reply falls back to the original question so a
    rewriter failure does not drop the user's words.
    """
    if not history:
        return question.strip()
    if llm is None:
        return question.strip()
    messages = [{"role": "system", "content": REWRITE_SYSTEM}]
    for turn in history:
        role = turn.get("role", "user")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": turn.get("content", "")})
    messages.append(
        {
            "role": "user",
            "content": "Rewrite this follow-up into a standalone search query:\n" + question.strip(),
        }
    )
    result = llm.generate(messages, temperature=0.0, max_tokens=64)
    rewritten = result.text.strip().strip('"').strip()
    first_line = rewritten.splitlines()[0].strip() if rewritten else ""
    if not first_line or len(first_line) > 400:
        return question.strip()
    return first_line
