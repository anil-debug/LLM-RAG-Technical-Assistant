"""Score a ranked list against one golden question."""

from evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from retrieval.types import Hit


def score_question(hits: list[Hit], item: dict, ks: tuple[int, ...] = (5, 10)) -> dict | None:
    """Return retrieval metrics, or None when the question is unanswerable.

    Unanswerable questions have no relevant chunk. They are scored on the
    generation side, by whether the model abstains.
    """
    if item.get("unanswerable"):
        return None
    expected = list(item.get("expected_files") or [])
    substrings = list(item.get("expected_any_substrings") or [])
    if not expected:
        return None
    return {
        "recall": {str(k): recall_at_k(hits, expected, k, substrings) for k in ks},
        "precision": {str(k): precision_at_k(hits, expected, substrings, k) for k in ks},
        "mrr": mean_reciprocal_rank(hits, expected, substrings),
    }


def average_scores(rows: list[dict]) -> dict:
    """Macro-average recall, precision, and MRR. Empty input yields an empty dict."""
    if not rows:
        return {}
    recall_keys = rows[0]["recall"].keys()
    precision_keys = rows[0]["precision"].keys()
    count = len(rows)
    return {
        "count": count,
        "recall": {key: sum(row["recall"][key] for row in rows) / count for key in recall_keys},
        "precision": {
            key: sum(row["precision"][key] for row in rows) / count for key in precision_keys
        },
        "mrr": sum(row["mrr"] for row in rows) / count,
    }
