"""Reciprocal rank fusion uses rank, not the raw score scales."""

from retrieval.hybrid_search import reciprocal_rank_fusion
from retrieval.types import Hit


def _hit(chunk_id: str, **scores) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id="d",
        filename="f.md",
        title="t",
        text=chunk_id,
        section=None,
        page_start=None,
        page_end=None,
        **scores,
    )


def test_a_document_ranked_high_on_both_lists_wins():
    semantic = [_hit("a", semantic_score=0.2), _hit("b", semantic_score=0.99)]
    lexical = [_hit("b", bm25_score=1.0), _hit("c", bm25_score=50.0)]
    fused = reciprocal_rank_fusion(semantic, lexical, k=60)
    assert fused[0].chunk_id == "b"
    assert fused[0].semantic_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[1].chunk_id in {"a", "c"}


def test_keyword_only_hit_is_kept():
    semantic = [_hit("a", semantic_score=0.9)]
    lexical = [_hit("code", bm25_score=3.0)]
    fused = reciprocal_rank_fusion(semantic, lexical, k=60)
    ids = [hit.chunk_id for hit in fused]
    assert "code" in ids
    code = next(hit for hit in fused if hit.chunk_id == "code")
    assert code.bm25_score == 3.0
    assert code.semantic_score is None
