"""BM25 prefers rare exact identifiers over nearby prose."""

from retrieval.bm25 import BM25Index
from retrieval.types import StoredChunk


def _chunk(chunk_id: str, text: str) -> StoredChunk:
    return StoredChunk(
        chunk_id=chunk_id,
        document_id="d",
        filename="doc.md",
        title="doc",
        text=text,
        section=None,
        page_start=None,
        page_end=None,
        embed_text=text,
        token_count=1,
        prev_chunk_id=None,
        next_chunk_id=None,
    )


def test_exact_error_code_beats_a_nearby_paragraph():
    index = BM25Index(
        [
            _chunk("exact", "Error 0x1F means SYS_FAN1 stalled."),
            _chunk("near", "The cooling fan subsystem uses a management controller and firmware."),
        ]
    )
    hits = index.search("0x1F", k=2)
    assert hits[0].chunk_id == "exact"
    assert hits[0].bm25_score > 0
    assert all(hit.chunk_id != "near" or hit.bm25_score < hits[0].bm25_score for hit in hits)


def test_version_and_sensor_tokens_match_case_insensitively():
    index = BM25Index([_chunk("a", "Apply BMC firmware 01.73.12 before replacing SYS_FAN1.")])
    assert index.search("01.73.12", k=1)[0].chunk_id == "a"
    assert index.search("sys_fan1", k=1)[0].chunk_id == "a"


def test_empty_query_returns_nothing():
    index = BM25Index([_chunk("a", "BMC")])
    assert index.search("   ", k=5) == []
