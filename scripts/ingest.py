"""Index files into the configured vector store.

The memory backend lives in this process only. Use VECTOR_BACKEND=postgres
when the index must outlive the command. ``--skip-embed`` stores chunks for
BM25 and skips the embedding download. Semantic search then returns no hits.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.deps import build_container
from core.logging_config import setup_logging
from core.settings import Settings
from evaluation.corpus import sample_files
from ingestion.pipeline import prepare_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents.")
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--skip-embed", action="store_true")
    args = parser.parse_args()
    settings = Settings()
    setup_logging(settings.log_level)
    if settings.vector_backend == "memory":
        print("VECTOR_BACKEND=memory: this index disappears when the process exits.")
    container = build_container(settings)
    paths = args.paths or sample_files()
    for path in paths:
        document, chunks = prepare_document(
            path,
            settings,
            container.token_counter,
            llm=container.llm,
            embedder=container.embedder,
        )
        vectors = None
        model_name = None
        if chunks and not args.skip_embed:
            vectors = container.embedder.embed_documents([chunk.embed_text for chunk in chunks])
            model_name = container.embedder.model_name
        container.store.upsert(document, chunks, vectors, model_name)
        print(f"{document.id} {document.filename} chunks={len(chunks)} embedded={model_name or 'no'}")


if __name__ == "__main__":
    main()
