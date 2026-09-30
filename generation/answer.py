"""Citations and the end-to-end answer function."""

import re
import time
from dataclasses import dataclass, field

from core.logging_config import log_event
from generation.context import ContextPack, build_context
from generation.prompt import build_messages
from generation.provider import LLMProvider
from generation.rewrite import rewrite_query
from retrieval.hybrid_search import hybrid_retrieve
from retrieval.types import Hit

import logging

LOGGER = logging.getLogger(__name__)
_CITATION = re.compile(r"\[(\d+)\]")
_SOURCES = re.compile(r"\n+Sources:\n.*\Z", re.DOTALL)


@dataclass
class Citation:
    index: int
    chunk_id: str
    document_id: str
    filename: str
    title: str
    section: str | None
    page_start: int | None
    page_end: int | None


@dataclass
class AnswerResult:
    answer: str
    rewritten_query: str
    citations: list[Citation]
    rejected_citation_ids: list[int]
    hits: list[Hit]
    context: str
    timings_ms: dict[str, float]
    model_name: str
    conversation_id: str | None = None
    metadata: dict = field(default_factory=dict)


def extract_citation_ids(text: str) -> list[int]:
    """Return bracket ids in order of first appearance."""
    found: list[int] = []
    for match in _CITATION.finditer(text):
        value = int(match.group(1))
        if value not in found:
            found.append(value)
    return found


def resolve_citations(text: str, pack: ContextPack) -> tuple[list[Citation], list[int]]:
    """Keep ids that were packed and reject the rest."""
    by_index = {block.index: block for block in pack.blocks}
    valid: list[Citation] = []
    rejected: list[int] = []
    for citation_id in extract_citation_ids(text):
        block = by_index.get(citation_id)
        if block is None:
            rejected.append(citation_id)
            continue
        hit = block.hit
        valid.append(
            Citation(
                index=citation_id,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                filename=hit.filename,
                title=hit.title,
                section=hit.section,
                page_start=hit.page_start,
                page_end=hit.page_end,
            )
        )
    return valid, rejected


def render_answer(model_text: str, citations: list[Citation], rejected: list[int]) -> str:
    """Drop invalid citation markers and append a source list we computed."""
    body = _SOURCES.sub("", model_text).strip()

    def replace(match: re.Match[str]) -> str:
        value = int(match.group(1))
        if value in rejected:
            return ""
        return match.group(0)

    body = _CITATION.sub(replace, body)
    body = re.sub(r"[ \t]{2,}", " ", body).strip()
    if not citations:
        return body
    lines = ["Sources:"]
    for citation in citations:
        section = citation.section or "untitled section"
        page = f" — page {citation.page_start}" if citation.page_start is not None else ""
        lines.append(
            f"[{citation.index}] {citation.title} — {section}{page} "
            f"— {citation.filename} — chunk {citation.chunk_id}"
        )
    return body + "\n\n" + "\n".join(lines)


def answer_question(
    question: str,
    *,
    history: list[dict[str, str]],
    embedder,
    store,
    llm: LLMProvider,
    counter,
    settings,
    reranker=None,
    conversation_id: str | None = None,
) -> AnswerResult:
    """Rewrite, retrieve, pack context, generate, and validate citations."""
    started = time.perf_counter()
    rewritten = rewrite_query(question, history, llm if history else None)
    rewrite_ms = _elapsed(started)

    retrieval_started = time.perf_counter()
    hits = hybrid_retrieve(
        rewritten,
        embedder=embedder,
        store=store,
        candidates=settings.retrieval_candidates,
        top_n=settings.rerank_top_n,
        rrf_k=settings.rrf_k,
        bm25_k1=settings.bm25_k1,
        bm25_b=settings.bm25_b,
        reranker=reranker,
        reranker_enabled=settings.reranker_enabled,
        entity_boost=settings.intel_entities,
    )
    retrieval_ms = _elapsed(retrieval_started)
    pack = build_context(
        hits,
        counter=counter,
        budget=settings.context_token_budget,
        chunks_by_id=store.chunks_by_id(),
    )
    messages = build_messages(question, history, pack.prompt_text)
    generation_started = time.perf_counter()
    generated = llm.generate(
        messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    generation_ms = _elapsed(generation_started)
    citations, rejected = resolve_citations(generated.text, pack)
    answer = render_answer(generated.text, citations, rejected)
    total_ms = _elapsed(started)
    timings = {
        "rewrite_ms": round(rewrite_ms, 3),
        "retrieval_ms": round(retrieval_ms, 3),
        "llm_ms": round(generation_ms, 3),
        "total_ms": round(total_ms, 3),
    }
    log_event(
        LOGGER,
        "chat complete",
        model_name=generated.model,
        top_k=settings.rerank_top_n,
        **timings,
    )
    return AnswerResult(
        answer=answer,
        rewritten_query=rewritten,
        citations=citations,
        rejected_citation_ids=rejected,
        hits=hits,
        context=pack.prompt_text,
        timings_ms=timings,
        model_name=generated.model,
        conversation_id=conversation_id,
    )


def _elapsed(started: float) -> float:
    return (time.perf_counter() - started) * 1000
