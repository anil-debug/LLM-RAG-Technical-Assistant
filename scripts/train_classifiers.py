"""Train the linear document and section heads.

The heads need frozen embeddings. When the embedding weights are not on disk,
this script writes NOT_EXECUTED. The training loop itself is covered by
tests/test_intelligence.py on synthetic features.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.settings import Settings
from evaluation.report import git_revision, machine_info, write_report


def main() -> None:
    settings = Settings()
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            settings.embedding_model,
            device=settings.embedding_device,
            local_files_only=True,
        )
        status = "NOT_EXECUTED"
        reason = (
            f"{settings.embedding_model} loaded, but this script does not fit the heads "
            "until labeled section and document vectors are encoded. "
            "No accuracy number was computed."
        )
    except Exception as exc:
        status = "NOT_EXECUTED"
        reason = (
            "Classifier training was not executed because the embedding model "
            f"could not be loaded ({str(exc)[:400]}). "
            "Unit tests fit LinearClassifier on synthetic features. "
            "After the weights exist, encode data/sample/labels/sections.jsonl and "
            "documents.jsonl, call intelligence.linear_classifier.train_classifier, "
            "and save the state dict to the configured .pt paths."
        )
    payload = {
        "status": status,
        "reason": reason,
        "git_revision": git_revision(ROOT),
        "machine": machine_info(),
        "dataset_version": settings.dataset_version,
        "model_name": settings.embedding_model,
        "metrics": None,
        "configuration": {
            "section_labels": "intelligence.section_classifier.SECTION_LABELS",
            "document_classifier_path": str(settings.document_classifier_path),
            "section_classifier_path": str(settings.section_classifier_path),
        },
        "methodology": (
            "A linear layer on frozen embeddings, Adam, cross-entropy. "
            "The encoder is not updated. No accuracy is reported until that loop runs "
            "on real embeddings."
        ),
    }
    path = write_report(ROOT / "evaluation" / "reports", "classifier_training", payload)
    print(f"{status} {path}")


if __name__ == "__main__":
    main()
