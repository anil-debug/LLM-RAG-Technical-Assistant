"""Cross-encoder reranking with an injected scoring function."""

import pytest

from core.errors import ModelUnavailableError
from retrieval.reranker import CrossEncoderReranker
from retrieval.types import Hit


def _hit(chunk_id: str, text: str) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id="d",
        filename="troubleshooting.md",
        title="Troubleshooting",
        text=text,
        section="Errors",
        page_start=1,
        page_end=1,
    )


def test_reranker_reorders_by_injected_scores() -> None:
    hits = [_hit("low", "unrelated cooling prose"), _hit("high", "Error 0x1F means SYS_FAN1 stalled")]

    def predict(pairs: list[tuple[str, str]]) -> list[float]:
        assert pairs[0][0] == "0x1F"
        return [0.1, 0.9]

    ranked = CrossEncoderReranker(predict=predict).rerank("0x1F", hits, top_n=1)
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "high"
    assert ranked[0].rerank_score == pytest.approx(0.9)


def test_reranker_rejects_a_short_score_vector() -> None:
    hits = [_hit("a", "one"), _hit("b", "two")]
    reranker = CrossEncoderReranker(predict=lambda _pairs: [0.2])
    with pytest.raises(ModelUnavailableError):
        reranker.rerank("query", hits, top_n=2)


def test_reranker_empty_window() -> None:
    assert CrossEncoderReranker(predict=lambda _pairs: []).rerank("query", [], top_n=8) == []
