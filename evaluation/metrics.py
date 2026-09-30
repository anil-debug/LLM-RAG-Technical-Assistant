"""Retrieval and answer metrics.

Recall, precision, and MRR are computed from labeled files and substrings.
Answer metrics in this module are deterministic proxies. An LLM judge is a
separate optional pass and must be labeled as a judge when it is used.
"""

from retrieval.types import Hit


def is_relevant(hit: Hit, expected_files: list[str], substrings: list[str]) -> bool:
    """A hit is relevant when its file is expected and it contains a labeled fact."""
    if expected_files and hit.filename not in expected_files:
        return False
    if not substrings:
        return hit.filename in expected_files
    text = hit.text.casefold()
    return any(item.casefold() in text for item in substrings)


def recall_at_k(
    hits: list[Hit],
    expected_files: list[str],
    k: int,
    substrings: list[str] | None = None,
) -> float:
    """Fraction of labeled facts present in relevant top-k hits.

    A fact counts as retrieved when its substring appears in a top-k hit whose
    file is one of ``expected_files``. When ``substrings`` is omitted, recall
    is the fraction of expected files that appear at least once in the top k.
    """
    if substrings:
        blob = "\n".join(
            hit.text for hit in hits[:k] if not expected_files or hit.filename in expected_files
        ).casefold()
        return sum(1 for item in substrings if item.casefold() in blob) / len(substrings)
    if not expected_files:
        raise ValueError("recall_at_k requires expected files or substrings.")
    found = {hit.filename for hit in hits[:k]}
    return len(found & set(expected_files)) / len(set(expected_files))


def precision_at_k(
    hits: list[Hit],
    expected_files: list[str],
    substrings: list[str],
    k: int,
) -> float:
    """Fraction of the top k slots that are relevant.

    The denominator is k. Missing slots, when fewer than k hits were returned,
    count as not relevant.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    good = sum(1 for hit in hits[:k] if is_relevant(hit, expected_files, substrings))
    return good / k


def mean_reciprocal_rank(
    hits: list[Hit],
    expected_files: list[str],
    substrings: list[str],
) -> float:
    """1 / rank of the first relevant hit, or 0 when none is relevant."""
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, expected_files, substrings):
            return 1.0 / rank
    return 0.0


def labeled_fact_coverage(answer: str, substrings: list[str]) -> float:
    """Fraction of labeled fact strings present in the answer."""
    if not substrings:
        return 1.0
    text = answer.casefold()
    return sum(1 for item in substrings if item.casefold() in text) / len(substrings)


def citation_correct(
    cited_filenames: list[str],
    rejected_ids: list[int],
    expected_files: list[str],
    *,
    unanswerable: bool,
) -> bool:
    """True when every cited file was expected and no id was rejected.

    An unanswerable question is correct only when the answer cites nothing.
    """
    if rejected_ids:
        return False
    if unanswerable:
        return not cited_filenames
    if not cited_filenames:
        return False
    expected = set(expected_files)
    return all(name in expected for name in cited_filenames)


def claim_supported(answer: str, cited_texts: list[str], substrings: list[str]) -> float:
    """Fraction of labeled substrings that appear in both the answer and a cited chunk.

    This is a proxy for faithfulness on labeled facts. It is not a general
    entailment judgment.
    """
    if not substrings:
        return 1.0
    cited = "\n".join(cited_texts).casefold()
    answer_text = answer.casefold()
    supported = 0
    for item in substrings:
        needle = item.casefold()
        if needle in answer_text and needle in cited:
            supported += 1
    return supported / len(substrings)
