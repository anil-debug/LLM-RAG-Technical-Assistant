"""HTTP request and response models."""

from pydantic import BaseModel, Field


class DocumentSummary(BaseModel):
    id: str
    filename: str
    media_type: str
    title: str
    checksum: str
    chunk_count: int
    created_at: str


class DocumentCreated(BaseModel):
    document_id: str
    filename: str
    chunks: int
    checksum: str


class HitModel(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    title: str
    text: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    semantic_score: float | None = None
    bm25_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class SearchResponse(BaseModel):
    query: str
    hits: list[HitModel]
    timings_ms: dict[str, float]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None


class CitationModel(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    filename: str
    title: str
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    rewritten_query: str
    answer: str
    citations: list[CitationModel]
    rejected_citation_ids: list[int]
    hits: list[HitModel]
    context: str
    timings_ms: dict[str, float]
    model_name: str


class ErrorBody(BaseModel):
    code: str
    message: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    vector_backend: str
    embedding_model: str
    llm_provider: str
    llm_model: str
    reranker_enabled: bool
