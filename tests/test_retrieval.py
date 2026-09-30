"""Semantic search and hybrid retrieval with the hashing test double."""

from datetime import UTC, datetime

from embeddings.embed import l2_normalize
from ingestion.models import Chunk, Document
from retrieval.hybrid_search import boost_entity_matches, hybrid_retrieve
from retrieval.retriever import semantic_search
from retrieval.types import Hit
from retrieval.vector_store import MemoryVectorStore
from tests.fakes import HashingEmbedder


def _document(doc_id: str, filename: str, text: str) -> tuple[Document, Chunk]:
    document = Document(
        id=doc_id,
        filename=filename,
        source=filename,
        media_type="markdown",
        checksum=doc_id,
        title=filename,
        text=text,
        pages=None,
        created_at=datetime.now(UTC),
    )
    chunk = Chunk(
        id=f"{doc_id}-c",
        document_id=doc_id,
        chunk_index=0,
        text=text,
        embed_text=text,
        section="Body",
        page_start=None,
        page_end=None,
        token_count=4,
    )
    return document, chunk


def test_semantic_search_ranks_shared_tokens_first() -> None:
    embedder = HashingEmbedder()
    store = MemoryVectorStore()
    fan, fan_chunk = _document("fan", "troubleshooting.md", "Error 0x1F means SYS_FAN1 stalled")
    vlan, vlan_chunk = _document("vlan", "networking-guide.md", "Management traffic uses a dedicated gateway")
    matrix = l2_normalize(embedder.embed_documents([fan_chunk.embed_text, vlan_chunk.embed_text]))
    store.upsert(fan, [fan_chunk], matrix[0:1], embedder.model_name)
    store.upsert(vlan, [vlan_chunk], matrix[1:2], embedder.model_name)
    hits = semantic_search("0x1F SYS_FAN1", embedder=embedder, store=store, k=1)
    assert hits[0].filename == "troubleshooting.md"
    assert hits[0].semantic_score is not None
    assert hits[0].semantic_rank == 1


def test_hybrid_keeps_exact_identifier_from_bm25(settings) -> None:
    embedder = HashingEmbedder()
    store = MemoryVectorStore()
    fan, fan_chunk = _document("fan", "troubleshooting.md", "Error 0x1F means SYS_FAN1 stalled")
    other, other_chunk = _document("other", "readme-notes.txt", "Use the hostname when DNS is missing")
    matrix = embedder.embed_documents([fan_chunk.embed_text, other_chunk.embed_text])
    store.upsert(fan, [fan_chunk], matrix[0:1], embedder.model_name)
    store.upsert(other, [other_chunk], matrix[1:2], embedder.model_name)
    hits = hybrid_retrieve(
        "0x1F",
        embedder=embedder,
        store=store,
        candidates=5,
        top_n=2,
        rrf_k=settings.rrf_k,
        reranker_enabled=False,
    )
    assert any("0x1F" in hit.text for hit in hits)
    fan_hit = next(hit for hit in hits if hit.filename == "troubleshooting.md")
    assert fan_hit.bm25_score is not None
    assert fan_hit.fusion_score is not None


def test_entity_boost_promotes_a_matching_surface() -> None:
    hits = [
        Hit(
            chunk_id="a",
            document_id="d",
            filename="a.md",
            title="A",
            text="general cooling note",
            section=None,
            page_start=None,
            page_end=None,
            bm25_score=1.5,
        ),
        Hit(
            chunk_id="b",
            document_id="d",
            filename="b.md",
            title="B",
            text="sensor row",
            section=None,
            page_start=None,
            page_end=None,
            bm25_score=1.0,
            metadata={"entities": [{"label": "identifier", "text": "SYS_FAN1"}]},
        ),
    ]
    boosted = boost_entity_matches(hits, "SYS_FAN1 status")
    assert boosted[0].chunk_id == "b"
    assert boosted[0].bm25_score == 2.0
    assert boosted[0].bm25_rank == 1
