"""Turn a cleaned document into citation-sized chunks."""

from uuid import uuid4

from core.text import RegexTokenCounter
from ingestion.chunking.strategies import (
    Piece,
    fixed_windows,
    log_pieces,
    recursive_windows,
    structure_pieces,
    token_windows,
)
from ingestion.models import Chunk, Document, Page


def chunk_document(
    document: Document,
    *,
    strategy: str = "structure",
    counter: RegexTokenCounter | None = None,
    target_tokens: int = 480,
    overlap_tokens: int = 72,
    fixed_chars: int = 2000,
    fixed_overlap_chars: int = 200,
) -> list[Chunk]:
    """Chunk ``document``.

    Logs use event boundaries when the caller asked for the structure strategy,
    because a half event is not a useful retrieval unit. Other strategies are
    selected explicitly.
    """
    counter = counter or RegexTokenCounter()
    resolved = "log" if document.media_type == "log" and strategy == "structure" else strategy
    pieces = _pieces(
        document,
        resolved,
        counter,
        target_tokens,
        overlap_tokens,
        fixed_chars,
        fixed_overlap_chars,
    )
    chunks: list[Chunk] = []
    for index, piece in enumerate(pieces):
        page = piece.page_start
        if page is None:
            page = _locate_page(piece.text, document.pages)
        section = piece.section
        embed_text = f"{section}\n{piece.text}" if section else piece.text
        chunks.append(
            Chunk(
                id=str(uuid4()),
                document_id=document.id,
                chunk_index=index,
                text=piece.text,
                embed_text=embed_text,
                section=section,
                page_start=page,
                page_end=piece.page_end if piece.page_end is not None else page,
                token_count=counter.count(piece.text),
                metadata={},
            )
        )
    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk.prev_chunk_id = chunks[index - 1].id
        if index + 1 < len(chunks):
            chunk.next_chunk_id = chunks[index + 1].id
    return chunks


def _pieces(
    document: Document,
    strategy: str,
    counter: RegexTokenCounter,
    target: int,
    overlap: int,
    fixed_chars: int,
    fixed_overlap: int,
) -> list[Piece]:
    if strategy == "structure":
        return structure_pieces(document, counter, target, overlap)
    if strategy == "log":
        return log_pieces(document.text, counter, target, overlap)
    if strategy == "fixed":
        return [
            Piece(text, None, None, None)
            for text in fixed_windows(document.text, fixed_chars, fixed_overlap)
        ]
    if strategy == "token":
        return [Piece(text, None, None, None) for text in token_windows(document.text, target, overlap)]
    if strategy == "recursive":
        return [
            Piece(text, None, None, None)
            for text in recursive_windows(document.text, counter, target, overlap)
        ]
    raise ValueError(f"Unknown chunk strategy '{strategy}'.")


def _locate_page(text: str, pages: list[Page] | None) -> int | None:
    if not pages:
        return None
    probe = text.strip()[:80]
    if not probe:
        return None
    for page in pages:
        if probe in page.text:
            return page.page_number
    return None
