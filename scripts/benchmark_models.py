"""Benchmark embedding models and LLM endpoints when they are actually reachable.

A failed download or a refused connection is written as NOT_EXECUTED.
This script never fills a metric with a guessed number.
"""

import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings
from evaluation.report import git_revision, machine_info, write_report


def _base(settings: Settings) -> dict:
    return {
        "git_revision": git_revision(ROOT),
        "machine": machine_info(),
        "dataset_version": settings.dataset_version,
        "metrics": None,
    }


def embedding_report(settings: Settings, config: dict, *, download: bool) -> dict:
    """Try to encode one sentence with each configured bi-encoder.

    A full Recall@K comparison requires both models to encode the corpus.
    This environment check stops at load time when the weights are absent,
    and it does not publish a partial quality score. ``download`` must be
    passed explicitly so a benchmark run does not silently fetch weights.
    """
    attempts = []
    for model_name in config["embedding_models"]:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device="cpu", local_files_only=not download)
            vector = model.encode(["probe"], normalize_embeddings=False)
            attempts.append(
                {
                    "model_name": model_name,
                    "loaded": True,
                    "output_dimension": int(vector.shape[-1]),
                }
            )
        except Exception as exc:
            attempts.append({"model_name": model_name, "loaded": False, "error": str(exc)[:400]})
    loaded = [row for row in attempts if row["loaded"]]
    if len(loaded) < 2:
        return {
            **_base(settings),
            "status": "NOT_EXECUTED",
            "reason": (
                "Embedding benchmark was not executed. Both BAAI/bge-base-en-v1.5 and "
                "intfloat/e5-base-v2 must load before Recall@K, MRR, indexing time, "
                "query latency, and memory can be compared. "
                "Download them on a machine with network access and rerun "
                "uv run python scripts/benchmark_models.py"
            ),
            "model_name": None,
            "configuration": {"attempts": attempts, "config": config},
            "methodology": (
                "No quality metric is reported from a hashing embedder or from a single "
                "model. Memory would be process RSS plus CUDA peak when a GPU is used."
            ),
        }
    return {
        **_base(settings),
        "status": "NOT_EXECUTED",
        "reason": (
            "Both embedding models loaded, but the golden-set comparison pass is still "
            "required. Extend this script's comparison loop before quoting Recall@K."
        ),
        "model_name": [row["model_name"] for row in loaded],
        "configuration": {"attempts": attempts},
        "methodology": "Load check only. Quality metrics were not computed in this path.",
    }


def llm_report(settings: Settings, config: dict) -> dict:
    """Ask each configured server for /models. Do not score answers that were not returned."""
    attempts = []
    for item in config["llm_models"]:
        url = item["base_url"].rstrip("/") + "/models"
        try:
            response = httpx.get(url, timeout=3.0)
            response.raise_for_status()
            attempts.append({"model": item["name"], "base_url": item["base_url"], "reachable": True})
        except Exception as exc:
            attempts.append(
                {
                    "model": item["name"],
                    "base_url": item["base_url"],
                    "reachable": False,
                    "error": str(exc),
                }
            )
    reachable = [row for row in attempts if row["reachable"]]
    status = "NOT_EXECUTED" if len(reachable) < 1 else "NOT_EXECUTED"
    reason = (
        "LLM benchmark was not executed. Retrieved contexts were not frozen and no "
        "answers were scored. Start Ollama or vLLM, then rerun "
        "uv run python scripts/benchmark_models.py. "
        "Client latency is the measurement for a remote server. "
        "Add process RSS only when the server is local and its PID is observable. "
        "Do not label remote time as GPU memory."
    )
    if reachable:
        reason = (
            "An OpenAI-compatible server responded, but this script does not yet score "
            "the golden set. No answer relevance, faithfulness, or citation number was computed."
        )
    return {
        **_base(settings),
        "status": status,
        "reason": reason,
        "model_name": None,
        "configuration": {"attempts": attempts},
        "methodology": (
            "Generation quality is scored only on answers the server returns, with "
            "retrieved context held fixed across models. This run did not generate answers."
        ),
    }


def main() -> None:
    download = "--download" in sys.argv
    settings = Settings()
    config = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text(encoding="utf-8"))
    reports = ROOT / "evaluation" / "reports"
    embedding_path = write_report(
        reports,
        "embedding_benchmark",
        embedding_report(settings, config, download=download),
    )
    llm_path = write_report(reports, "llm_benchmark", llm_report(settings, config))
    print(embedding_path)
    print(llm_path)


if __name__ == "__main__":
    main()
