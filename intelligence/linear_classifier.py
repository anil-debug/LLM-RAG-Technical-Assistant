"""A linear classifier on frozen vectors.

The encoder is not updated. Training fits ``y = softmax(x W^T + b)`` with
Adam and cross-entropy. That is the smallest PyTorch model that can be
inspected end to end: one matrix and one bias.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class LinearClassifier(nn.Module):
    """One linear layer. ``dim`` is the embedding size and ``n_labels`` the classes."""

    def __init__(self, dim: int, n_labels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, n_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


@dataclass
class TrainedClassifier:
    model: LinearClassifier
    labels: list[str]

    def predict(self, features: np.ndarray) -> list[str]:
        self.model.eval()
        with torch.inference_mode():
            tensor = torch.tensor(np.asarray(features, dtype=np.float32))
            if tensor.ndim == 1:
                tensor = tensor.reshape(1, -1)
            indexes = torch.argmax(self.model(tensor), dim=1).tolist()
        return [self.labels[index] for index in indexes]


def train_classifier(
    features: np.ndarray,
    label_indexes: np.ndarray,
    labels: list[str],
    *,
    epochs: int = 80,
    lr: float = 0.1,
    seed: int = 0,
) -> TrainedClassifier:
    """Fit a linear head. Features are treated as constants, not as a graph."""
    torch.manual_seed(seed)
    matrix = np.asarray(features, dtype=np.float32)
    targets = np.asarray(label_indexes, dtype=np.int64)
    model = LinearClassifier(matrix.shape[1], len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    inputs = torch.tensor(matrix)
    gold = torch.tensor(targets)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), gold)
        loss.backward()
        optimizer.step()
    model.eval()
    return TrainedClassifier(model=model, labels=list(labels))
