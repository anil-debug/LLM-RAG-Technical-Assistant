"""LLM provider interface.

Ollama and vLLM both speak ``/v1/chat/completions``, so they share
``OpenAICompatibleProvider``. A local PyTorch model implements the same
``generate`` method in ``TransformersLocalProvider``.
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


class LLMProvider(Protocol):
    """One chat completion. Implementations must not retrieve documents."""

    model_name: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        """Return the model text and token counts when the server reports them."""
