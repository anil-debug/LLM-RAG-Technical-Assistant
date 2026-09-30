"""Deterministic generation checks against labeled facts and citations."""

from generation.prompt import ABSTENTION
from evaluation.metrics import citation_correct, claim_supported, labeled_fact_coverage


def score_answer(item: dict, answer: str, cited_filenames: list[str], cited_texts: list[str], rejected: list[int]) -> dict:
    """Score one answer. Fields are proxies, not an LLM judge."""
    unanswerable = bool(item.get("unanswerable"))
    substrings = list(item.get("expected_any_substrings") or [])
    expected_files = list(item.get("expected_files") or [])
    abstained = ABSTENTION.casefold() in answer.casefold()
    if unanswerable:
        relevance = 1.0 if abstained else 0.0
        faithfulness = 1.0 if abstained and not cited_filenames else 0.0
        fact_coverage = None
    else:
        relevance = labeled_fact_coverage(answer, substrings)
        faithfulness = claim_supported(answer, cited_texts, substrings)
        fact_coverage = relevance
    return {
        "answer_relevance_proxy": relevance,
        "faithfulness_proxy": faithfulness,
        "labeled_fact_coverage": fact_coverage,
        "citation_accuracy": citation_correct(
            cited_filenames,
            rejected,
            expected_files,
            unanswerable=unanswerable,
        ),
        "abstained": abstained,
        "methodology": (
            "answer_relevance_proxy is labeled-fact coverage for answerable questions "
            "and abstention for unanswerable questions. faithfulness_proxy requires each "
            "labeled fact in the answer to appear in a cited chunk. These are not LLM-judge scores."
        ),
    }


def average_generation(rows: list[dict]) -> dict:
    if not rows:
        return {}
    count = len(rows)
    return {
        "count": count,
        "answer_relevance_proxy": sum(row["answer_relevance_proxy"] for row in rows) / count,
        "faithfulness_proxy": sum(row["faithfulness_proxy"] for row in rows) / count,
        "citation_accuracy": sum(1.0 if row["citation_accuracy"] else 0.0 for row in rows) / count,
    }
