"""Bi-encoder interface.

``SentenceTransformerEmbedder`` loads Hugging Face weights on first use.
Tests inject another object with the same two methods. The hashing embedder
is intentionally not in this module: it is a test double, not a model.
"""

from typing import Protocol

import numpy as np

from core.device import resolve_device
from core.errors import ModelUnavailableError
from embeddings.config import EmbeddingConfig
from embeddings.embed import apply_prefix, l2_normalize


class EmbeddingModel(Protocol):
    """Encode queries and passages into one vector space."""

    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return one normalized row per passage."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return one normalized query vector."""


class SentenceTransformerEmbedder:
    """sentence-transformers bi-encoder with explicit prefixes and normalization.

    The library tokenizes, runs the transformer, and pools token states into
    one vector. This class chooses the prefix, the batch size, and L2
    normalization. It does not implement search.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.model_name = config.model_name
        self.dimension = config.dimension
        self._model = None

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefixed = [apply_prefix(text, self.config.passage_prefix) for text in texts]
        return self._encode(prefixed)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([apply_prefix(text, self.config.query_prefix)])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load()
        try:
            vectors = model.encode(
                texts,
                batch_size=self.config.batch_size,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise ModelUnavailableError(f"Embedding failed for {self.model_name}: {exc}") from exc
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.dimension:
            raise ModelUnavailableError(
                f"{self.model_name} returned dimension {array.shape[1]}, "
                f"configured dimension is {self.dimension}."
            )
        return l2_normalize(array)

    def _load(self):
        if self._model is not None:
            return self._model
        device = resolve_device(self.config.device)
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=device)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load embedding model {self.model_name}. "
                "Download it first or check the model id. "
                f"Underlying error: {exc}"
            ) from exc
        return self._model


class ModelTokenCounter:
    """Count tokens with the embedding model's tokenizer.

    Chunk budgets should use this counter when the tokenizer can be loaded.
    ``RegexTokenCounter`` remains available for offline tests.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None

    def count(self, text: str) -> int:
        if not text.strip():
            return 0
        from core.errors import TokenizerUnavailableError

        try:
            tokenizer = self._load()
            return len(tokenizer.encode(text, add_special_tokens=False))
        except TokenizerUnavailableError:
            raise
        except Exception as exc:
            raise TokenizerUnavailableError(
                f"Could not tokenize with {self.model_name}: {exc}"
            ) from exc

    def _load(self):
        if self._tokenizer is not None:
            return self._tokenizer
        from core.errors import TokenizerUnavailableError

        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as exc:
            raise TokenizerUnavailableError(
                f"Could not load tokenizer for {self.model_name}: {exc}"
            ) from exc
        return self._tokenizer
