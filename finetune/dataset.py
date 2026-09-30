"""Build a LoRA training set that does not reuse golden question strings."""

import json
from pathlib import Path

from evaluation.datasets import load_jsonl
from finetune.facts import FACTS, WRAPPERS

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "evaluation" / "datasets" / "golden.jsonl"


def build_pairs() -> list[dict[str, str]]:
    """Expand each fact with the wrappers. Raise if a question copies the golden set."""
    blocked = {item["question"].strip() for item in load_jsonl(GOLDEN)}
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for fact in FACTS:
        for wrapper in WRAPPERS:
            question = wrapper.format(q=fact["prompt"]).strip()
            if question in blocked:
                raise ValueError(f"Training question collides with the golden set: {question}")
            if question in seen:
                continue
            seen.add(question)
            pairs.append(
                {
                    "question": question,
                    "answer": fact["answer"],
                    "source": fact["source"],
                }
            )
    return pairs


def write_jsonl(path: Path) -> int:
    pairs = build_pairs()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(pair, ensure_ascii=True) for pair in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(pairs)
