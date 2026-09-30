"""Memory store round-trip and cosine ordering. Postgres is opt-in."""

import os

import numpy as np
import pytest

from embeddings.embed import l2_normalize
from ingestion.models import Chunk, Document
from retrieval.vector_store import MemoryVectorStore, PostgresVectorStore


def _document() -> Document:
    from datetime import UTC, datetime

    return Document(
        id="doc-1",
        filename="guide.md",
        source="guide.md",
        media_type="markdown",
        checksum="abc",
        title="Guide",
        text="Error 0x1F means SYS_FAN1 stalled.",
        pages=None,
        created_at=datetime.now(UTC),
    )


def _chunk(text: str, index: int = 0) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id="doc-1",
        chunk_index=index,
        text=text,
        embed_text=text,
        section="Fan",
        page_start=None,
        page_end=None,
        token_count=4,
    )


def test_memory_store_orders_by_cosine_and_replaces_checksum():
    store = MemoryVectorStore()
    document = _document()
    chunks = [_chunk("alpha"), _chunk("beta", 1)]
    chunks[1].id = "chunk-1"
    vectors = l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    store.upsert(document, chunks, vectors, "test-model")
    hits = store.search_semantic(np.array([1.0, 0.0], dtype=np.float32), model_name="test-model", k=2)
    assert hits[0].chunk_id == "chunk-0"
    assert hits[0].semantic_score > hits[1].semantic_score
    replacement = _document()
    replacement.id = "doc-2"
    store.upsert(replacement, [_chunk("gamma")], l2_normalize(np.array([[1.0, 0.0]])), "test-model")
    assert [row["id"] for row in store.list_documents()] == ["doc-2"]


def test_delete_missing_document():
    store = MemoryVectorStore()
    with pytest.raises(Exception):
        store.delete("missing")


@pytest.mark.skipif(os.environ.get("RUN_POSTGRES") != "1", reason="Set RUN_POSTGRES=1 to exercise pgvector.")
def test_postgres_round_trip():
    store = PostgresVectorStore(os.environ["DATABASE_URL"], dimension=2)
    document = _document()
    chunk = _chunk("pgvector row")
    vectors = l2_normalize(np.array([[0.3, 0.4]], dtype=np.float32))
    store.upsert(document, [chunk], vectors, "test-model")
    hits = store.search_semantic(np.array([0.3, 0.4], dtype=np.float32), model_name="test-model", k=1)
    assert hits[0].chunk_id == "chunk-0"
    store.delete(document.id)
