"""Load, clean, annotate, and chunk one file.

Document-intelligence hooks are optional. With every flag off, this function
only loads, cleans, extracts metadata, and chunks.
"""

from pathlib import Path

from core.settings import Settings
from core.text import RegexTokenCounter
from ingestion.chunking.chunker import chunk_document
from ingestion.cleaning import clean_document
from ingestion.loaders import load_file
from ingestion.metadata.extractor import extract_metadata
from ingestion.models import Chunk, Document


def prepare_document(
    path: Path,
    settings: Settings | None = None,
    counter: RegexTokenCounter | None = None,
    *,
    llm: object | None = None,
    embedder: object | None = None,
) -> tuple[Document, list[Chunk]]:
    """Return a cleaned document and its chunks.

    ``llm`` and ``embedder`` are used only when the matching intelligence flag
    is enabled. They are typed loosely here so this module does not import the
    model classes at import time.
    """
    settings = settings or Settings(_env_file=None)
    counter = counter or RegexTokenCounter()
    document = extract_metadata(clean_document(load_file(path)))
    if _intelligence_requested(settings):
        from intelligence.pipeline import annotate_document

        document = annotate_document(document, settings, llm=llm, embedder=embedder)
    chunks = chunk_document(
        document,
        strategy=settings.chunk_strategy,
        counter=counter,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        fixed_chars=settings.fixed_chunk_chars,
        fixed_overlap_chars=settings.fixed_chunk_overlap_chars,
    )
    if settings.intel_entities or settings.intel_section_classifier or settings.intel_summarize:
        from intelligence.pipeline import annotate_chunks

        chunks = annotate_chunks(document, chunks, settings, llm=llm, embedder=embedder)
    return document, chunks


def _intelligence_requested(settings: Settings) -> bool:
    return any(
        (
            settings.intel_entities,
            settings.intel_document_classifier,
            settings.intel_summarize,
        )
    )
