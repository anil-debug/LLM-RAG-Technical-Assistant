"""Okapi BM25 over in-memory chunk text.

Score for term t in chunk D:

    IDF(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |D| / avgdl))

IDF is the Robertson-Sparck Jones form:

    log(1 + (N - n_t + 0.5) / (n_t + 0.5))

This index is rebuilt from chunk text. It is the right size for the sample
corpus and for a few thousand chunks. Past that, move term statistics into
PostgreSQL or a search engine. ``ts_rank`` is not used here, because it is
not BM25.
"""

import math
from collections import Counter

from core.text import tokenize
from retrieval.types import Hit, StoredChunk


class BM25Index:
    """BM25 over a fixed list of chunks."""

    def __init__(self, chunks: list[StoredChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tf: list[Counter[str]] = []
        self.doc_len: list[int] = []
        document_frequency: Counter[str] = Counter()
        for chunk in self.chunks:
            tokens = tokenize(chunk.text)
            counts = Counter(tokens)
            self.doc_tf.append(counts)
            self.doc_len.append(len(tokens))
            document_frequency.update(counts.keys())
        self.df = document_frequency
        self.n = len(self.chunks)
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 0.0

    def idf(self, term: str) -> float:
        n_t = self.df.get(term, 0)
        return math.log(1.0 + (self.n - n_t + 0.5) / (n_t + 0.5))

    def score_document(self, index: int, terms: list[str]) -> float:
        if self.avgdl == 0:
            return 0.0
        tf_map = self.doc_tf[index]
        length = self.doc_len[index]
        score = 0.0
        for term in terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * length / self.avgdl)
            score += self.idf(term) * numerator / denominator
        return score

    def search(self, query: str, k: int) -> list[Hit]:
        terms = tokenize(query)
        if not terms or not self.chunks or k <= 0:
            return []
        scored: list[tuple[float, StoredChunk]] = []
        for index, chunk in enumerate(self.chunks):
            score = self.score_document(index, terms)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        hits: list[Hit] = []
        for rank, (score, chunk) in enumerate(scored[:k], start=1):
            hits.append(Hit.from_chunk(chunk, bm25_score=score, bm25_rank=rank))
        return hits
