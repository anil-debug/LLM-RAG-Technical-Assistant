"""Run local evaluations and write timestamped JSON reports.

BM25, chunk-strategy, entity, and entity-boost modes execute in this process.
They do not call a neural embedding model or an LLM.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings
from evaluation.report import git_revision, machine_info, write_report
from evaluation.runners import (
    evaluate_bm25,
    evaluate_chunk_strategies,
    evaluate_entities,
    evaluate_entity_boost,
)

MODES = {
    "bm25": ("bm25_retrieval", evaluate_bm25),
    "chunk-strategies": ("chunk_strategy", evaluate_chunk_strategies),
    "entities": ("entity_extraction", evaluate_entities),
    "entity-boost": ("intelligence_ablation", evaluate_entity_boost),
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        names = ", ".join(MODES)
        raise SystemExit(f"Usage: python scripts/evaluate.py <{names}>")
    prefix, function = MODES[sys.argv[1]]
    settings = Settings()
    payload = function(settings) if sys.argv[1] != "entities" else function()
    payload["git_revision"] = git_revision(ROOT)
    payload["machine"] = machine_info()
    payload["dataset_version"] = settings.dataset_version
    payload.setdefault(
        "configuration",
        {
            "chunk_strategy": settings.chunk_strategy,
            "chunk_target_tokens": settings.chunk_target_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "bm25_k1": settings.bm25_k1,
            "bm25_b": settings.bm25_b,
            "rrf_k": settings.rrf_k,
        },
    )
    path = write_report(ROOT / "evaluation" / "reports", prefix, payload)
    print(f"{payload['status']} {path}")


if __name__ == "__main__":
    main()
