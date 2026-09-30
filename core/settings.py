"""Process configuration loaded from the environment and an optional ``.env`` file."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

KNOWN_EMBEDDING_DIMS = {
    "BAAI/bge-base-en-v1.5": 768,
    "intfloat/e5-base-v2": 768,
}


class Settings(BaseSettings):
    """Runtime settings.

    Environment variables use the field name in upper case, for example
    ``LLM_BASE_URL``. A value already set in the environment wins over ``.env``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"

    vector_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"

    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768
    embedding_batch_size: int = 16
    embedding_device: str = "cpu"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    embedding_passage_prefix: str = ""

    chunk_target_tokens: int = 480
    chunk_overlap_tokens: int = 72
    chunk_strategy: Literal["fixed", "token", "recursive", "structure", "log"] = "structure"
    chunk_tokenizer: Literal["model", "regex"] = "regex"
    fixed_chunk_chars: int = 2000
    fixed_chunk_overlap_chars: int = 200

    retrieval_candidates: int = 30
    rerank_top_n: int = 8
    context_token_budget: int = 3000
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"

    llm_provider: Literal["openai_compatible", "transformers"] = "openai_compatible"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: str = "ollama"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    llm_timeout_seconds: float = 120.0

    transformers_model: str = ""
    transformers_device: str = "cpu"
    transformers_dtype: str = "float32"

    intel_entities: bool = False
    intel_section_classifier: bool = False
    intel_document_classifier: bool = False
    intel_summarize: bool = False
    section_classifier_path: Path = Path("models/section_classifier.pt")
    document_classifier_path: Path = Path("models/document_classifier.pt")

    max_upload_bytes: int = 20 * 1024 * 1024
    data_dir: Path = Path("data")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    dataset_version: str = "sample-v1"

    allowed_suffixes: tuple[str, ...] = Field(
        default=(".pdf", ".md", ".markdown", ".txt", ".html", ".htm", ".log")
    )

    @model_validator(mode="after")
    def check_ranges(self) -> "Settings":
        expected = KNOWN_EMBEDDING_DIMS.get(self.embedding_model)
        if expected is not None and self.embedding_dim != expected:
            raise ValueError(
                f"{self.embedding_model} uses {expected} dimensions, "
                f"not {self.embedding_dim}."
            )
        if self.chunk_overlap_tokens >= self.chunk_target_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_target_tokens.")
        if self.chunk_target_tokens < 32:
            raise ValueError("chunk_target_tokens must be at least 32.")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive.")
        return self
