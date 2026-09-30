"""Retrieval records. Scores stay optional because each stage fills its own."""

from dataclasses import dataclass, field, replace


@dataclass
class StoredChunk:
    """A chunk plus the document fields needed to cite it."""

    chunk_id: str
    document_id: str
    filename: str
    title: str
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    embed_text: str
    token_count: int
    prev_chunk_id: str | None
    next_chunk_id: str | None
    metadata: dict = field(default_factory=dict)


@dataclass
class Hit:
    """One retrieved chunk and the scores that put it in the candidate set."""

    chunk_id: str
    document_id: str
    filename: str
    title: str
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    semantic_score: float | None = None
    bm25_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, chunk: StoredChunk, **scores: float | int | None) -> "Hit":
        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            filename=chunk.filename,
            title=chunk.title,
            text=chunk.text,
            section=chunk.section,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            metadata=dict(chunk.metadata),
            **scores,
        )


def copy_hit(hit: Hit, **changes: object) -> Hit:
    return replace(hit, **changes)
