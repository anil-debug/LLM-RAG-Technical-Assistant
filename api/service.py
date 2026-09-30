"""Ingestion and query operations shared by the HTTP routes."""

import time
from pathlib import Path

from core.context import set_document_id
from core.errors import EmptyDocumentError, UnsupportedMediaTypeError, UploadTooLargeError
from core.logging_config import log_event
from generation.answer import answer_question
from ingestion.pipeline import prepare_document
from retrieval.hybrid_search import hybrid_retrieve

import logging

LOGGER = logging.getLogger(__name__)


def save_upload(container, filename: str, data: bytes) -> dict:
    """Validate an upload, store the bytes under ``data_dir``, and index them."""
    safe_name = Path(filename).name
    if safe_name in {"", ".", ".."}:
        raise UnsupportedMediaTypeError("The upload needs a filename.")
    suffix = Path(safe_name).suffix.lower()
    if suffix not in container.settings.allowed_suffixes:
        raise UnsupportedMediaTypeError(f"Unsupported file type '{suffix}'.")
    if not data:
        raise EmptyDocumentError("The upload is empty.")
    if len(data) > container.settings.max_upload_bytes:
        raise UploadTooLargeError(
            f"Upload is {len(data)} bytes. The limit is {container.settings.max_upload_bytes}."
        )
    destination_dir = container.settings.data_dir / "documents"
    destination_dir.mkdir(parents=True, exist_ok=True)
    # Keep the suffix so the loader can see the type. The uuid blocks two
    # uploads of the same filename from sharing one temporary path.
    from uuid import uuid4

    temporary = destination_dir / f"upload-{uuid4().hex}-{safe_name}"
    temporary.write_bytes(data)
    try:
        document, chunks = prepare_document(
            temporary,
            container.settings,
            container.token_counter,
            llm=container.llm,
            embedder=container.embedder,
        )
    finally:
        if temporary.exists():
            temporary.unlink()
    document.filename = safe_name
    target_dir = destination_dir / document.id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    target.write_bytes(data)
    document.source = str(target)
    set_document_id(document.id)
    embeddings = None
    model_name = None
    if chunks:
        embeddings = container.embedder.embed_documents([chunk.embed_text for chunk in chunks])
        model_name = container.embedder.model_name
    container.store.upsert(document, chunks, embeddings, model_name)
    log_event(LOGGER, "document indexed", document_id=document.id, chunks=len(chunks), model_name=model_name)
    return {
        "document_id": document.id,
        "filename": document.filename,
        "chunks": len(chunks),
        "checksum": document.checksum,
    }


def search(container, query: str, top_k: int | None) -> dict:
    started = time.perf_counter()
    limit = top_k or container.settings.rerank_top_n
    hits = hybrid_retrieve(
        query,
        embedder=container.embedder,
        store=container.store,
        candidates=container.settings.retrieval_candidates,
        top_n=limit,
        rrf_k=container.settings.rrf_k,
        bm25_k1=container.settings.bm25_k1,
        bm25_b=container.settings.bm25_b,
        reranker=container.reranker,
        reranker_enabled=container.settings.reranker_enabled,
        entity_boost=container.settings.intel_entities,
    )
    elapsed = (time.perf_counter() - started) * 1000
    log_event(
        LOGGER,
        "search complete",
        model_name=container.embedder.model_name,
        top_k=limit,
        retrieval_ms=round(elapsed, 3),
        bm25_docs=len(container.store.all_chunks()),
    )
    return {"query": query, "hits": hits, "timings_ms": {"retrieval_ms": round(elapsed, 3)}}


def chat(container, message: str, conversation_id: str | None) -> dict:
    conversation_id = container.store.ensure_conversation(conversation_id)
    history = [
        {"role": item["role"], "content": item["content"]}
        for item in container.store.list_messages(conversation_id)
    ]
    result = answer_question(
        message,
        history=history,
        embedder=container.embedder,
        store=container.store,
        llm=container.llm,
        counter=container.token_counter,
        settings=container.settings,
        reranker=container.reranker,
        conversation_id=conversation_id,
    )
    container.store.add_message(conversation_id, "user", message, [])
    container.store.add_message(
        conversation_id,
        "assistant",
        result.answer,
        [citation.__dict__ for citation in result.citations],
    )
    result.conversation_id = conversation_id
    return result
