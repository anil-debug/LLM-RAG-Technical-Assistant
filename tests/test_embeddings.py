"""Embedding helpers: normalization, cosine, and prefixes. No weight download."""

import numpy as np

from embeddings.config import preset_config
from embeddings.embed import apply_prefix, cosine_similarity, l2_normalize
from embeddings.embedding_model import SentenceTransformerEmbedder


def test_unit_vectors_have_cosine_equal_to_dot_product():
    left = np.array([1.0, 0.0, 0.0])
    right = np.array([1.0, 1.0, 0.0])
    normalized = l2_normalize(np.stack([left, right]))
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, 1.0)
    assert cosine_similarity(left, right) == np.dot(normalized[0], normalized[1])


def test_prefixes_differ_for_bge_and_e5():
    bge = preset_config("BAAI/bge-base-en-v1.5")
    e5 = preset_config("intfloat/e5-base-v2")
    assert bge.dimension == e5.dimension == 768
    assert apply_prefix("fan", bge.query_prefix).startswith("Represent this sentence")
    assert apply_prefix("fan", e5.query_prefix) == "query: fan"
    assert apply_prefix("fan", e5.passage_prefix) == "passage: fan"
    assert apply_prefix("fan", bge.passage_prefix) == "fan"


def test_embedder_prefixes_queries_and_normalizes():
    config = preset_config("BAAI/bge-base-en-v1.5")
    embedder = SentenceTransformerEmbedder(config)
    seen: dict[str, list[str]] = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            seen["texts"] = list(texts)
            assert kwargs["normalize_embeddings"] is False
            rows = []
            for index, _text in enumerate(texts):
                row = np.zeros(config.dimension, dtype=np.float32)
                row[index % config.dimension] = 3.0
                rows.append(row)
            return np.stack(rows)

    embedder._model = FakeModel()
    query = embedder.embed_query("0x1F")
    assert seen["texts"] == [config.query_prefix + "0x1F"]
    assert np.isclose(np.linalg.norm(query), 1.0)
    docs = embedder.embed_documents(["a", "b"])
    assert docs.shape == (2, 768)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
