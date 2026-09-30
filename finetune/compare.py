"""Four-way comparison scaffold.

Arms, when a run is actually executed:

- base: the instruct model with no retrieved context
- base_rag: the same model with retrieved context
- finetuned: the LoRA adapter with no retrieved context
- finetuned_rag: the adapter with retrieved context

This module does not invent those scores. Without an adapter and a reachable
generator the report status is NOT_EXECUTED.
"""

from evaluation.generation_eval import average_generation, score_answer


ARMS = ("base", "base_rag", "finetuned", "finetuned_rag")


def score_arm(items: list[dict], answers: list[dict]) -> dict:
    """Score answers that were actually produced.

    ``answers`` entries need ``answer``, ``cited_filenames``, ``cited_texts``,
    and ``rejected``. The caller measured latency separately.
    """
    rows = [
        score_answer(
            item,
            answer["answer"],
            answer.get("cited_filenames") or [],
            answer.get("cited_texts") or [],
            answer.get("rejected") or [],
        )
        for item, answer in zip(items, answers, strict=True)
    ]
    return average_generation(rows)


def comparison_status(*, adapter_ready: bool, generator_ready: bool) -> dict:
    """Return NOT_EXECUTED until both an adapter and a generator exist."""
    if adapter_ready and generator_ready:
        return {
            "status": "READY",
            "reason": "Adapter and generator are available. Run the comparison to fill metrics.",
            "metrics": None,
        }
    missing = []
    if not adapter_ready:
        missing.append("no LoRA adapter has been trained")
    if not generator_ready:
        missing.append("no LLM generator is reachable")
    return {
        "status": "NOT_EXECUTED",
        "reason": (
            "The four-way comparison was not executed: "
            + "; ".join(missing)
            + ". Arms would be base, base+RAG, fine-tuned, and fine-tuned+RAG "
            "on the golden set, which is disjoint from the training questions."
        ),
        "metrics": None,
        "arms": list(ARMS),
    }
