"""Liveness and readiness."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.schemas.models import HealthResponse, ReadyResponse
from core.errors import DatabaseUnavailableError

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request):
    container = request.app.state.container
    settings = container.settings
    try:
        container.store.ping()
    except DatabaseUnavailableError as exc:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": str(exc)})
    return ReadyResponse(
        status="ready",
        vector_backend=settings.vector_backend,
        embedding_model=settings.embedding_model,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model if settings.llm_provider == "openai_compatible" else settings.transformers_model,
        reranker_enabled=settings.reranker_enabled,
    )
