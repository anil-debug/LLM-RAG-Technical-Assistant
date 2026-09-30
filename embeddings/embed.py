"""Vector helpers that do not load a model.

Cosine similarity of L2-normalized vectors is their dot product. Retrieval
stores the normalized vectors and reports that dot product as the score.
"""

import numpy as np


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Return rows scaled to unit length. A zero row becomes a one-hot at index 0."""
    matrix = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.maximum(norms, 1e-12)
    normalized = matrix / safe
    zero_rows = norms.reshape(-1) < 1e-12
    if np.any(zero_rows):
        normalized[zero_rows] = 0.0
        normalized[zero_rows, 0] = 1.0
    return normalized


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity. Inputs are normalized inside this function."""
    a = l2_normalize(left)[0]
    b = l2_normalize(right)[0]
    return float(np.dot(a, b))


def apply_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text
    return f"{prefix}{text}"
