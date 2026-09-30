"""Section labels for chunks.

The head is trained on frozen chunk embeddings. Heading text is a feature of
the embedding the caller supplies, not a separate rule engine. If no weights
are loaded, the pipeline refuses to pretend it has a label.
"""

from pathlib import Path

from core.errors import ClassifierNotReadyError
from intelligence.linear_classifier import LinearClassifier, TrainedClassifier

SECTION_LABELS = [
    "overview",
    "procedure",
    "reference",
    "error_definition",
    "warning",
    "log_event",
    "other",
]


def load_section_classifier(path: Path, dim: int) -> TrainedClassifier:
    import torch

    if not path.exists():
        raise ClassifierNotReadyError(
            f"No section classifier weights at {path}. "
            "Train them with scripts/train_classifiers.py after embeddings exist."
        )
    model = LinearClassifier(dim, len(SECTION_LABELS))
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return TrainedClassifier(model=model, labels=list(SECTION_LABELS))
