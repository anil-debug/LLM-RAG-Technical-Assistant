"""Reciprocal rank fusion and the hybrid retrieval function.

Semantic scores and BM25 scores are not on the same scale. Fusion uses rank:

    RRF(d) = sum_i 1 / (k + rank_i(d))

with k = 60. A chunk that only the keyword list found still receives a score.
"""

import logging
import time

from core.logging_config import log_event
from core.text import tokenize
from retrieval.bm25 import BM25Index
from retrieval.retriever import semantic_search
from retrieval.types import Hit, copy_hit

LOGGER = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    semantic_hits: list[Hit],
    lexical_hits: list[Hit],
    *,
    k: int = 60,
) -> list[Hit]:
    """Fuse two ranked lists. Ties break on chunk id so tests stay stable."""
    merged: dict[str, Hit] = {}
    for rank, hit in enumerate(semantic_hits, start=1):
        current = merged.get(hit.chunk_id)
        if current is None:
            current = copy_hit(hit, fusion_score=0.0, semantic_rank=rank)
            merged[hit.chunk_id] = current
        else:
            current.semantic_score = hit.semantic_score
            current.semantic_rank = rank
        current.fusion_score = float(current.fusion_score or 0.0) + 1.0 / (k + rank)
    for rank, hit in enumerate(lexical_hits, start=1):
        current = merged.get(hit.chunk_id)
        if current is None:
            current = copy_hit(hit, fusion_score=0.0, bm25_rank=rank)
            merged[hit.chunk_id] = current
        else:
            current.bm25_score = hit.bm25_score
            current.bm25_rank = rank
            if not current.text:
                current.text = hit.text
        current.fusion_score = float(current.fusion_score or 0.0) + 1.0 / (k + rank)
    fused = list(merged.values())
    fused.sort(key=lambda hit: (-float(hit.fusion_score or 0.0), hit.chunk_id))
    return fused


def boost_entity_matches(hits: list[Hit], query: str, amount: float = 1.0) -> list[Hit]:
    """Add a fixed bonus when a stored entity surface also appears in the query.

    The bonus is applied to the BM25 score before fusion ranks are used.
    Callers re-rank after the bonus. Chunks ingested with entity extraction
    off have no entity metadata, so this function leaves them unchanged.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return hits
    adjusted: list[Hit] = []
    for hit in hits:
        entities = hit.metadata.get("entities") or []
        matched = False
        for entity in entities:
            if set(tokenize(str(entity.get("text", "")))) & query_terms:
                matched = True
                break
        score = float(hit.bm25_score or 0.0)
        if matched:
            score += amount
        adjusted.append(copy_hit(hit, bm25_score=score))
    adjusted.sort(key=lambda hit: (-float(hit.bm25_score or 0.0), hit.chunk_id))
    return [copy_hit(hit, bm25_rank=rank) for rank, hit in enumerate(adjusted, start=1)]


def hybrid_retrieve(
    query: str,
    *,
    embedder: object,
    store: object,
    candidates: int,
    top_n: int,
    rrf_k: int,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    reranker: object | None = None,
    reranker_enabled: bool = False,
    entity_boost: bool = False,
) -> list[Hit]:
    """Run semantic search, BM25, fusion, and an optional reranker.

    BM25 is rebuilt from the chunks currently in ``store``. That is simple and
    correct for this corpus size. It rereads every chunk on each query.
    """
    started = time.perf_counter()
    semantic_hits = semantic_search(query, embedder=embedder, store=store, k=candidates)
    embedding_ms = (time.perf_counter() - started) * 1000
    lexical_started = time.perf_counter()
    bm25 = BM25Index(store.all_chunks(), k1=bm25_k1, b=bm25_b)
    lexical_hits = bm25.search(query, candidates)
    if entity_boost:
        lexical_hits = boost_entity_matches(lexical_hits, query)
    bm25_ms = (time.perf_counter() - lexical_started) * 1000
    fused = reciprocal_rank_fusion(semantic_hits, lexical_hits, k=rrf_k)
    window = fused[:candidates]
    rerank_started = time.perf_counter()
    if reranker_enabled and reranker is not None:
        ranked = reranker.rerank(query, window, top_n)
    else:
        ranked = window[:top_n]
        for hit in ranked:
            hit.rerank_score = None
    reranker_ms = (time.perf_counter() - rerank_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    log_event(
        LOGGER,
        "hybrid retrieval",
        model_name=getattr(embedder, "model_name", None),
        top_k=top_n,
        embedding_ms=round(embedding_ms, 3),
        bm25_ms=round(bm25_ms, 3),
        reranker_ms=round(reranker_ms, 3),
        retrieval_ms=round(total_ms, 3),
        candidates=len(window),
    )
    return ranked
