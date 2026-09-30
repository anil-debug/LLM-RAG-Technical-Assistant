"""Metric definitions, the golden set, and the local BM25 runner."""

from pathlib import Path

from evaluation.datasets import load_jsonl
from evaluation.generation_eval import score_answer
from evaluation.metrics import citation_correct, precision_at_k, recall_at_k
from evaluation.report import write_report
from evaluation.retrieval_eval import score_question
from evaluation.runners import evaluate_bm25, evaluate_entities
from finetune.compare import comparison_status, score_arm
from finetune.dataset import build_pairs
from finetune.train import lora_config, training_plan
from retrieval.types import Hit


def _hit(filename: str, text: str, chunk_id: str = "c") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document_id="d",
        filename=filename,
        title=filename,
        text=text,
        section=None,
        page_start=None,
        page_end=None,
    )


def test_recall_is_fact_recall_not_file_overlap() -> None:
    hits = [_hit("firmware-update.md", "The chassis ships with many pages of unrelated overview.")]
    assert recall_at_k(hits, ["firmware-update.md"], 5, ["01.73.12"]) == 0.0
    hits[0].text = "Apply BMC firmware 01.73.12 before BIOS 2.8.4."
    assert recall_at_k(hits, ["firmware-update.md"], 5, ["01.73.12", "2.8.4"]) == 1.0
    assert precision_at_k(hits, ["firmware-update.md"], ["01.73.12"], 5) == 1 / 5


def test_unanswerable_questions_are_not_retrieval_rows() -> None:
    item = {"unanswerable": True, "expected_files": [], "expected_any_substrings": []}
    assert score_question([_hit("troubleshooting.md", "no default BIOS password")], item) is None


def test_citation_and_abstention_proxies() -> None:
    item = {
        "unanswerable": True,
        "expected_files": [],
        "expected_any_substrings": [],
    }
    abstained = score_answer(
        item,
        "The provided documents do not contain enough information to answer this question.",
        [],
        [],
        [],
    )
    assert abstained["answer_relevance_proxy"] == 1.0
    assert abstained["citation_accuracy"] is True
    answered = score_answer(item, "The password is secret.", ["troubleshooting.md"], ["secret"], [])
    assert answered["faithfulness_proxy"] == 0.0
    assert citation_correct(["firmware-update.md"], [], ["firmware-update.md"], unanswerable=False)
    assert not citation_correct(["other.md"], [], ["firmware-update.md"], unanswerable=False)


def test_golden_set_covers_the_required_tags() -> None:
    rows = load_jsonl(Path("evaluation/datasets/golden.jsonl"))
    assert len(rows) >= 40
    tags = {tag for row in rows for tag in row["tags"]}
    assert {"exact-identifier", "technical", "multi-hop", "unanswerable", "multi-turn"} <= tags
    assert sum(1 for row in rows if row["unanswerable"]) >= 5
    assert sum(1 for row in rows if "multi-turn" in row["tags"]) >= 5


def test_bm25_runner_executes_on_the_sample(settings) -> None:
    report = evaluate_bm25(settings)
    assert report["status"] == "EXECUTED"
    assert report["questions_scored"] >= 40
    recall_10 = report["metrics"]["recall"]["10"]
    assert 0.0 <= recall_10 <= 1.0
    assert report["metrics"]["mrr"] > 0.0
    entities = evaluate_entities()
    misses = entities["metrics"]["per_passage"][0]["false_negatives"]
    assert ("tracking_code", "arx-77") in misses


def test_training_pairs_are_disjoint_from_golden_questions() -> None:
    pairs = build_pairs()
    assert len(pairs) >= 200
    golden = {row["question"] for row in load_jsonl(Path("evaluation/datasets/golden.jsonl"))}
    assert not {pair["question"] for pair in pairs} & golden
    plan = training_plan()
    assert plan["examples"] == len(pairs)
    assert plan["formula"] == "W' = W + B A"
    assert lora_config().r == 8
    assert comparison_status(adapter_ready=False, generator_ready=False)["status"] == "NOT_EXECUTED"


def test_score_arm_averages_only_supplied_answers() -> None:
    items = [
        {
            "unanswerable": False,
            "expected_files": ["firmware-update.md"],
            "expected_any_substrings": ["01.73.12"],
        }
    ]
    answers = [
        {
            "answer": "Apply 01.73.12 [1].",
            "cited_filenames": ["firmware-update.md"],
            "cited_texts": ["Apply BMC firmware 01.73.12."],
            "rejected": [],
        }
    ]
    assert score_arm(items, answers)["citation_accuracy"] == 1.0


def test_report_keeps_null_metrics(tmp_path: Path) -> None:
    path = write_report(
        tmp_path,
        "llm_benchmark",
        {
            "status": "NOT_EXECUTED",
            "reason": "Ollama is not running.",
            "model_name": None,
            "metrics": None,
            "dataset_version": "sample-v1",
            "methodology": "No request was sent.",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "NOT_EXECUTED" in text
    assert '"metrics": null' in text
