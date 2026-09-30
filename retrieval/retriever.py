"""Semantic search: embed the query, then cosine top-k."""

import numpy as np

from retrieval.types import Hit


def semantic_search(query: str, *, embedder, store, k: int) -> list[Hit]:
    """Return the nearest stored chunks for ``query``.

    The embedder applies its own query prefix. The store compares normalized
    vectors, so the score is a cosine similarity.
    """
    if k <= 0 or not query.strip():
        return []
    vector = np.asarray(embedder.embed_query(query), dtype=np.float32)
    return store.search_semantic(vector, model_name=embedder.model_name, k=k)
