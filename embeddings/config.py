"""Embedding model names, dimensions, and query/passage prefixes.

BGE v1.5 expects a retrieval instruction on the query only. E5 expects
``query:`` and ``passage:`` prefixes. The adapter owns that difference so
retrieval code can pass raw text.
"""

from dataclasses import dataclass

from core.settings import Settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

PRESETS: dict[str, tuple[str, str, int]] = {
    "BAAI/bge-base-en-v1.5": (BGE_QUERY_PREFIX, "", 768),
    "intfloat/e5-base-v2": (E5_QUERY_PREFIX, E5_PASSAGE_PREFIX, 768),
}


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    dimension: int
    query_prefix: str
    passage_prefix: str
    batch_size: int
    device: str


def config_from_settings(settings: Settings) -> EmbeddingConfig:
    return EmbeddingConfig(
        model_name=settings.embedding_model,
        dimension=settings.embedding_dim,
        query_prefix=settings.embedding_query_prefix,
        passage_prefix=settings.embedding_passage_prefix,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def preset_config(model_name: str, *, batch_size: int = 16, device: str = "cpu") -> EmbeddingConfig:
    """Build a config for a known benchmark model."""
    if model_name not in PRESETS:
        raise KeyError(f"No embedding preset for {model_name}.")
    query_prefix, passage_prefix, dimension = PRESETS[model_name]
    return EmbeddingConfig(
        model_name=model_name,
        dimension=dimension,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
        batch_size=batch_size,
        device=device,
    )
