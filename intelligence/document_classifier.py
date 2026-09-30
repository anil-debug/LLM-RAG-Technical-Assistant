"""Document-level labels from a pooled embedding.

A handful of manuals is a weak accuracy statistic. The training loop is real,
and any reported accuracy must include the number of documents it was measured
on. This module does not invent that number.
"""

from pathlib import Path

from core.errors import ClassifierNotReadyError
from intelligence.linear_classifier import LinearClassifier, TrainedClassifier

DOCUMENT_LABELS = [
    "manual",
    "api_reference",
    "troubleshooting",
    "release_notes",
    "firmware_notes",
    "log",
]


def load_document_classifier(path: Path, dim: int) -> TrainedClassifier:
    import torch

    if not path.exists():
        raise ClassifierNotReadyError(
            f"No document classifier weights at {path}. "
            "Train them with scripts/train_classifiers.py after embeddings exist."
        )
    model = LinearClassifier(dim, len(DOCUMENT_LABELS))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return TrainedClassifier(model=model, labels=list(DOCUMENT_LABELS))
