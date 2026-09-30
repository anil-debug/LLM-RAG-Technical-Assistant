"""Index the synthetic AstraRack sample without a neural embedding model.

BM25 evaluation uses this index. Semantic evaluation supplies its own embedder
and must not quote the hashing test double as a quality score.
"""

from pathlib import Path

import numpy as np

from core.settings import Settings
from core.text import RegexTokenCounter
from ingestion.pipeline import prepare_document
from retrieval.vector_store import MemoryVectorStore

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
GOLDEN = ROOT / "evaluation" / "datasets" / "golden.jsonl"


def sample_files(root: Path | None = None) -> list[Path]:
    """Manuals and the SEL log. Label files are not documents."""
    base = root or SAMPLE
    files = sorted((base / "manuals").glob("*")) + sorted((base / "logs").glob("*"))
    return [path for path in files if path.is_file()]


def index_documents(
    settings: Settings,
    *,
    embedder=None,
    strategy: str | None = None,
) -> MemoryVectorStore:
    """Chunk every sample file into a memory store.

    When ``embedder`` is omitted, chunks are stored without vectors. BM25 does
    not need them. A caller that passes an embedder is asking for semantic
    search and owns the decision to download weights.
    """
    current = settings.model_copy(update={"chunk_strategy": strategy}) if strategy else settings
    store = MemoryVectorStore()
    counter = RegexTokenCounter()
    for path in sample_files():
        document, chunks = prepare_document(path, current, counter, embedder=embedder)
        vectors = None
        model_name = None
        if embedder is not None and chunks:
            vectors = np.asarray(embedder.embed_documents([chunk.embed_text for chunk in chunks]))
            model_name = embedder.model_name
        store.upsert(document, chunks, vectors, model_name)
    return store
