"""Search and chat routes."""

from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from api.auth import Principal, get_principal
from api.schemas.models import (
    ChatRequest,
    ChatResponse,
    CitationModel,
    HitModel,
    SearchRequest,
    SearchResponse,
)
from api.service import chat, search
from core.context import reset_query_id, set_query_id
from generation.answer import Citation
from retrieval.types import Hit

router = APIRouter(tags=["query"])


def _hit_model(hit: Hit) -> HitModel:
    return HitModel(
        chunk_id=hit.chunk_id,
        document_id=hit.document_id,
        filename=hit.filename,
        title=hit.title,
        text=hit.text,
        section=hit.section,
        page_start=hit.page_start,
        page_end=hit.page_end,
        semantic_score=hit.semantic_score,
        bm25_score=hit.bm25_score,
        semantic_rank=hit.semantic_rank,
        bm25_rank=hit.bm25_rank,
        fusion_score=hit.fusion_score,
        rerank_score=hit.rerank_score,
    )


@router.post("/search", response_model=SearchResponse)
def search_route(
    body: SearchRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> SearchResponse:
    del principal
    token = set_query_id(uuid4().hex)
    try:
        result = search(request.app.state.container, body.query, body.top_k)
    finally:
        reset_query_id(token)
    return SearchResponse(
        query=result["query"],
        hits=[_hit_model(hit) for hit in result["hits"]],
        timings_ms=result["timings_ms"],
    )


@router.post("/chat", response_model=ChatResponse)
def chat_route(
    body: ChatRequest,
    request: Request,
    principal: Principal = Depends(get_principal),
) -> ChatResponse:
    del principal
    token = set_query_id(uuid4().hex)
    try:
        result = chat(request.app.state.container, body.message, body.conversation_id)
    finally:
        reset_query_id(token)
    return ChatResponse(
        conversation_id=result.conversation_id or "",
        rewritten_query=result.rewritten_query,
        answer=result.answer,
        citations=[_citation_model(item) for item in result.citations],
        rejected_citation_ids=result.rejected_citation_ids,
        hits=[_hit_model(hit) for hit in result.hits],
        context=result.context,
        timings_ms=result.timings_ms,
        model_name=result.model_name,
    )


def _citation_model(citation: Citation) -> CitationModel:
    return CitationModel(
        index=citation.index,
        chunk_id=citation.chunk_id,
        document_id=citation.document_id,
        filename=citation.filename,
        title=citation.title,
        section=citation.section,
        page_start=citation.page_start,
        page_end=citation.page_end,
    )
