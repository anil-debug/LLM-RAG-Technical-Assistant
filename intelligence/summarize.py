"""Abstractive summaries through the existing LLM provider.

The summary is evidence-constrained text. Callers may index it as a chunk
with ``metadata['kind'] == 'summary'`` and can drop those chunks by turning
``INTEL_SUMMARIZE`` off and re-ingesting.
"""

from generation.provider import LLMProvider

SUMMARY_SYSTEM = (
    "Summarize the document using only the text in the user message. "
    "Do not add versions, codes, or procedures that the text does not state. "
    "Write at most six sentences."
)


def summarize_document(title: str, text: str, llm: LLMProvider, *, max_chars: int = 6000) -> str:
    clipped = text[:max_chars]
    result = llm.generate(
        [
            {"role": "system", "content": SUMMARY_SYSTEM},
            {"role": "user", "content": f"Title: {title}\n\n{clipped}"},
        ],
        temperature=0.0,
        max_tokens=280,
    )
    return result.text.strip()
