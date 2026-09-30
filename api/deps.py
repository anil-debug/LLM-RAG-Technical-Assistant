"""Wire settings, stores, and model adapters.

Model weights are not downloaded here. Embeddings and the reranker load on
first use and raise ``ModelUnavailableError`` if the weights are missing.
"""

from dataclasses import dataclass

from core.settings import Settings
from core.text import RegexTokenCounter
from embeddings.config import config_from_settings
from embeddings.embedding_model import ModelTokenCounter, SentenceTransformerEmbedder
from generation.openai_compatible import OpenAICompatibleProvider
from generation.transformers_local import TransformersLocalProvider
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import MemoryVectorStore, PostgresVectorStore


@dataclass
class Container:
    settings: Settings
    store: MemoryVectorStore | PostgresVectorStore
    embedder: SentenceTransformerEmbedder
    llm: OpenAICompatibleProvider | TransformersLocalProvider
    reranker: CrossEncoderReranker | None
    token_counter: RegexTokenCounter | ModelTokenCounter


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings()
    if settings.vector_backend == "postgres":
        store: MemoryVectorStore | PostgresVectorStore = PostgresVectorStore(
            settings.database_url,
            dimension=settings.embedding_dim,
        )
    else:
        store = MemoryVectorStore()
    if settings.chunk_tokenizer == "model":
        counter: RegexTokenCounter | ModelTokenCounter = ModelTokenCounter(settings.embedding_model)
    else:
        counter = RegexTokenCounter()
    if settings.llm_provider == "transformers":
        llm: OpenAICompatibleProvider | TransformersLocalProvider = TransformersLocalProvider(
            settings.transformers_model,
            device=settings.transformers_device,
            dtype=settings.transformers_dtype,
        )
    else:
        llm = OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    reranker = (
        CrossEncoderReranker(settings.reranker_model, device=settings.reranker_device)
        if settings.reranker_enabled
        else None
    )
    return Container(
        settings=settings,
        store=store,
        embedder=SentenceTransformerEmbedder(config_from_settings(settings)),
        llm=llm,
        reranker=reranker,
        token_counter=counter,
    )
