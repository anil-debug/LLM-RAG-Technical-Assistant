"""Normalized documents and chunks shared by the rest of the pipeline."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Page:
    """One PDF page after text extraction."""

    page_number: int
    text: str


@dataclass
class Document:
    """A source file after loading, before or after cleaning.

    ``text`` is the normalized full text. PDF documents also keep ``pages`` so
    later chunks can cite a page number. ``checksum`` is the SHA-256 of the
    original bytes and is filled in by the loader wrapper.
    """

    id: str
    filename: str
    source: str
    media_type: str
    checksum: str
    title: str
    text: str
    pages: list[Page] | None
    created_at: datetime
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """The retrieval unit. ``embed_text`` is what the bi-encoder encodes."""

    id: str
    document_id: str
    chunk_index: int
    text: str
    embed_text: str
    section: str | None
    page_start: int | None
    page_end: int | None
    token_count: int
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    metadata: dict = field(default_factory=dict)
