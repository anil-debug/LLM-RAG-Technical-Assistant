"""Test doubles. These are not semantic models and must not be quoted as quality scores."""

import hashlib

import numpy as np

from core.text import tokenize
from generation.provider import GenerationResult


class HashingEmbedder:
    """Bag-of-tokens unit vector. Shared tokens move vectors together.

    The mapping is SHA-256 based so it is stable across processes. It does not
    estimate semantic similarity.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dimension = dim
        self.model_name = "hashing-test-double"

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            vector[0] = 1.0
            norm = 1.0
        return vector / norm


class FakeLLM:
    """Scripted chat model. ``reply`` may be a string or a function of messages."""

    def __init__(self, reply: str | object = "The provided documents do not contain enough information to answer this question.") -> None:
        self.reply = reply
        self.model_name = "fake-llm"
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages, *, temperature: float, max_tokens: int) -> GenerationResult:
        del temperature, max_tokens
        self.calls.append(messages)
        text = self.reply(messages) if callable(self.reply) else self.reply
        return GenerationResult(
            text=text,
            model=self.model_name,
            prompt_tokens=1,
            completion_tokens=1,
            latency_ms=0.1,
        )
