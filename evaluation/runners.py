"""Local evaluation runners.

Neural and LLM metrics stay null unless the corresponding model actually
answers. BM25, chunk-strategy, and entity numbers come from this process.
"""

import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.settings import Settings
from evaluation.corpus import GOLDEN, SAMPLE, index_documents
from evaluation.datasets import load_jsonl
from evaluation.retrieval_eval import average_scores, score_question
from intelligence.entities import extract_entities, precision_recall, unique_keys
from retrieval.bm25 import BM25Index
from retrieval.hybrid_search import boost_entity_matches


def process_rss_bytes() -> int | None:
    """Resident set size from ``/proc/self/status``. This is not GPU memory."""
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    for line in status.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def evaluate_bm25(settings: Settings, *, k: int = 10) -> dict:
    """Score the golden set with BM25 only.

    Multi-turn rows are scored on their labeled standalone question. The raw
    follow-up is not the retrieval string. Unanswerable rows are excluded.
    """
    items = load_jsonl(GOLDEN)
    store = index_documents(settings)
    index = BM25Index(store.all_chunks(), k1=settings.bm25_k1, b=settings.bm25_b)
    rows: list[dict] = []
    by_tag: dict[str, list[dict]] = defaultdict(list)
    latencies: list[float] = []
    for item in items:
        started = time.perf_counter()
        hits = index.search(item["question"], k)
        latencies.append((time.perf_counter() - started) * 1000)
        scored = score_question(hits, item)
        if scored is None:
            continue
        tag = (item.get("tags") or ["untagged"])[0]
        row = {"id": item["id"], "tag": tag, **scored}
        rows.append(row)
        by_tag[tag].append(scored)
    return {
        "status": "EXECUTED",
        "retriever": "bm25",
        "model_name": None,
        "chunk_count": len(store.all_chunks()),
        "questions_scored": len(rows),
        "questions_excluded_unanswerable": sum(1 for item in items if item.get("unanswerable")),
        "metrics": average_scores([row for row in rows]),
        "by_tag": {tag: average_scores(group) for tag, group in sorted(by_tag.items())},
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "method": "time.perf_counter around BM25Index.search. Embedding and LLM time are not included.",
        },
        "memory": {
            "rss_bytes": process_rss_bytes(),
            "method": "/proc/self/status VmRSS after the BM25 loop. Process RSS, not CUDA memory.",
        },
        "methodology": (
            "Recall@K is the fraction of expected_any_substrings that appear in top-k hits "
            "whose filename is in expected_files. Precision@K uses denominator k. "
            "MRR uses the first hit that is in an expected file and contains a labeled substring. "
            "Multi-turn items use the labeled standalone question. "
            "This run did not embed queries and did not call an LLM."
        ),
    }


def evaluate_chunk_strategies(settings: Settings) -> dict:
    """Compare chunkers with the same BM25 scorer so the chunker is the only change."""
    strategies = ("fixed", "token", "recursive", "structure")
    results = {}
    for strategy in strategies:
        store = index_documents(settings, strategy=strategy)
        index = BM25Index(store.all_chunks(), k1=settings.bm25_k1, b=settings.bm25_b)
        rows = []
        for item in load_jsonl(GOLDEN):
            hits = index.search(item["question"], 10)
            scored = score_question(hits, item)
            if scored is not None:
                rows.append(scored)
        summary = average_scores(rows)
        summary["chunk_count"] = len(store.all_chunks())
        results[strategy] = summary
    return {
        "status": "EXECUTED",
        "model_name": None,
        "metrics": results,
        "methodology": (
            "Each strategy re-chunks the same sample files. Retrieval is BM25 with identical "
            "k1, b, and k=10. No embedding model is involved, so this does not rank bi-encoders."
        ),
    }


def evaluate_entities(path: Path | None = None) -> dict:
    """Score the hand-labeled entity passages. ARX-77 is a labeled miss."""
    label_path = path or (SAMPLE / "labels" / "entities.jsonl")
    rows = []
    found_all: set[tuple[str, str]] = set()
    gold_all: set[tuple[str, str]] = set()
    for item in load_jsonl(label_path):
        found = unique_keys(extract_entities(item["text"]))
        gold = {(row["label"], row["text"].casefold()) for row in item["gold"]}
        found_all |= found
        gold_all |= gold
        rows.append({"id": item["id"], **precision_recall(found, gold)})
    micro = precision_recall(found_all, gold_all)
    return {
        "status": "EXECUTED",
        "model_name": None,
        "metrics": {"micro": micro, "per_passage": rows},
        "methodology": (
            "Gold labels were written by hand for a few passages, including tracking code ARX-77, "
            "which the patterns are not supposed to match. Precision and recall use label plus "
            "casefolded surface. This is not a learned NER model."
        ),
    }


def evaluate_entity_boost(settings: Settings) -> dict:
    """Ablate the +1.0 entity bonus on BM25 rank. Do not assume it helps."""
    plain = index_documents(settings.model_copy(update={"intel_entities": False}))
    enriched = index_documents(settings.model_copy(update={"intel_entities": True}))
    plain_index = BM25Index(plain.all_chunks(), k1=settings.bm25_k1, b=settings.bm25_b)
    rich_index = BM25Index(enriched.all_chunks(), k1=settings.bm25_k1, b=settings.bm25_b)

    def score(index: BM25Index, boost: bool) -> dict:
        rows = []
        for item in load_jsonl(GOLDEN):
            hits = index.search(item["question"], 30)
            if boost:
                hits = boost_entity_matches(hits, item["question"])[:10]
            else:
                hits = hits[:10]
            scored = score_question(hits, item)
            if scored is not None:
                rows.append(scored)
        return average_scores(rows)

    baseline = score(plain_index, boost=False)
    boosted = score(rich_index, boost=True)
    return {
        "status": "EXECUTED",
        "model_name": None,
        "metrics": {"bm25": baseline, "bm25_entity_boost": boosted},
        "methodology": (
            "Both arms use BM25. The boosted arm adds 1.0 to the BM25 score when a stored entity "
            "surface overlaps the query, then re-sorts. Section classification, document "
            "classification, and summarization are not in this comparison. An improvement is "
            "not assumed."
        ),
    }
