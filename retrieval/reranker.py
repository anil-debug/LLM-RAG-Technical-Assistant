"""Cross-encoder reranking.

The bi-encoder never sees the query and the chunk in one sequence. The
cross-encoder does, which is why it can reorder a few dozen candidates more
accurately. It is too expensive to run on the whole corpus, so it only sees
the fused candidate window. Disable it with ``RERANKER_ENABLED=false``.
"""

from collections.abc import Callable

import numpy as np

from core.device import resolve_device
from core.errors import ModelUnavailableError
from retrieval.types import Hit, copy_hit


class CrossEncoderReranker:
    """Score query/chunk pairs with ``BAAI/bge-reranker-base`` by default."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str = "cpu",
        predict: Callable | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._predict = predict
        self._model = None

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits or top_n <= 0:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = np.asarray(self._score(pairs), dtype=np.float32).reshape(-1)
        if len(scores) != len(hits):
            raise ModelUnavailableError("Reranker returned a different number of scores than candidates.")
        order = np.argsort(-scores, kind="stable")
        ranked: list[Hit] = []
        for position in order[:top_n]:
            index = int(position)
            ranked.append(copy_hit(hits[index], rerank_score=float(scores[index])))
        return ranked

    def _score(self, pairs: list[tuple[str, str]]):
        if self._predict is not None:
            return self._predict(pairs)
        model = self._load()
        return model.predict(pairs, batch_size=8, show_progress_bar=False)

    def _load(self):
        if self._model is not None:
            return self._model
        device = resolve_device(self.device)
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=device)
        except Exception as exc:
            raise ModelUnavailableError(
                f"Could not load reranker {self.model_name}: {exc}"
            ) from exc
        return self._model
