"""Reusable lexical retrieval over document text, with metadata kept outside."""

from collections.abc import Sequence

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split whitespace; retain clinical terms and punctuation."""
    return text.lower().split()


class BM25Retriever:
    """Score documents in input order using Okapi BM25."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        tokenized = [tokenize(document) for document in documents]
        if not tokenized or not any(tokenized):
            raise ValueError("BM25 requires a nonempty tokenized corpus.")
        self.index = BM25Okapi(tokenized, k1=k1, b=b)
        self.corpus_size = len(tokenized)
        self.vocabulary_size = len(self.index.idf)
        self.empty_document_count = sum(not tokens for tokens in tokenized)

    def score(self, query: str) -> np.ndarray:
        """Return one score per document, without metadata filtering."""
        return self.index.get_scores(tokenize(query))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return document positions and scores; break ties by input order."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        scores = self.score(query)
        positions = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(position), float(scores[position])) for position in positions]
