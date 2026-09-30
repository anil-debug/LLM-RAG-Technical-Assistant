"""Run whichever document-intelligence modules are enabled.

Each flag is independent. Disabled modules do not read a model and do not
write annotations.
"""

from uuid import uuid4

from core.errors import ClassifierNotReadyError, ModelUnavailableError
from core.settings import Settings
from ingestion.models import Chunk, Document
from intelligence.entities import extract_entities
from intelligence.summarize import summarize_document


def annotate_document(document: Document, settings: Settings, *, llm=None, embedder=None) -> Document:
    """Document-level entities, classification, and an optional summary."""
    notes = dict(document.metadata.get("intelligence") or {})
    if settings.intel_entities:
        notes["entities"] = [entity.as_dict() for entity in extract_entities(document.text)]
    if settings.intel_document_classifier:
        notes["document_type"] = _classify_document(document, settings, embedder)
    if settings.intel_summarize:
        if llm is None:
            raise ModelUnavailableError("INTEL_SUMMARIZE is on and no LLM provider was supplied.")
        notes["summary"] = summarize_document(document.title, document.text, llm)
    document.metadata["intelligence"] = notes
    return document


def annotate_chunks(
    document: Document,
    chunks: list[Chunk],
    settings: Settings,
    *,
    llm=None,
    embedder=None,
) -> list[Chunk]:
    """Per-chunk entities and section labels, plus an optional summary chunk."""
    del llm
    if settings.intel_entities:
        for chunk in chunks:
            chunk.metadata["entities"] = [entity.as_dict() for entity in extract_entities(chunk.text)]
    if settings.intel_section_classifier:
        labels = _classify_sections(chunks, settings, embedder)
        for chunk, label in zip(chunks, labels, strict=True):
            chunk.metadata["section_label"] = label
    summary = (document.metadata.get("intelligence") or {}).get("summary")
    if settings.intel_summarize and summary:
        chunks.append(
            Chunk(
                id=str(uuid4()),
                document_id=document.id,
                chunk_index=len(chunks),
                text=summary,
                embed_text=f"Summary\n{summary}",
                section="Summary",
                page_start=None,
                page_end=None,
                token_count=max(1, len(summary.split())),
                metadata={"kind": "summary"},
            )
        )
        if len(chunks) > 1:
            chunks[-1].prev_chunk_id = chunks[-2].id
            chunks[-2].next_chunk_id = chunks[-1].id
    return chunks


def _classify_document(document: Document, settings: Settings, embedder) -> str:
    if embedder is None:
        raise ModelUnavailableError("Document classification needs an embedding model.")
    from intelligence.document_classifier import load_document_classifier

    try:
        classifier = load_document_classifier(settings.document_classifier_path, embedder.dimension)
    except ClassifierNotReadyError:
        raise
    vector = embedder.embed_documents([document.text[:4000]])
    return classifier.predict(vector)[0]


def _classify_sections(chunks: list[Chunk], settings: Settings, embedder) -> list[str]:
    if embedder is None:
        raise ModelUnavailableError("Section classification needs an embedding model.")
    if not chunks:
        return []
    from intelligence.section_classifier import load_section_classifier

    classifier = load_section_classifier(settings.section_classifier_path, embedder.dimension)
    vectors = embedder.embed_documents([chunk.embed_text for chunk in chunks])
    return classifier.predict(vectors)
