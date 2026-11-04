"""Reusable lexical retrieval over document text, with metadata kept outside."""

import re
from collections.abc import Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Lowercase and split whitespace; retain clinical terms and punctuation."""
    return text.lower().split()


# Generic English function words and query filler (no clinical terms).
FILLER_WORDS = frozenset("""
a an the of for in on at to by from with without and or as is are be been was were
that which what who whose this these those it its into about any all
trial trials study studies involving using find show me please
""".split())


def tokenize_v2(text: str) -> list[str]:
    """
    Tokenizer v2: lowercase, split on every non-alphanumeric character
    (so "Keytruda(R)", "(NSCLC)," and "pembrolizumab/cemiplimab" yield clean
    terms), and drop a small list of generic filler words.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in FILLER_WORDS]


TOKENIZERS: dict[str, Callable[[str], list[str]]] = {"v1": tokenize, "v2": tokenize_v2}


class BM25Retriever:
    """Score documents in input order using Okapi BM25."""

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75,
                 tokenizer: str = "v1"):
        self.tokenizer_version = tokenizer
        self._tokenize = TOKENIZERS[tokenizer]
        tokenized = [self._tokenize(document) for document in documents]
        if not tokenized or not any(tokenized):
            raise ValueError("BM25 requires a nonempty tokenized corpus.")
        self.index = BM25Okapi(tokenized, k1=k1, b=b)
        self.corpus_size = len(tokenized)
        self.vocabulary_size = len(self.index.idf)
        self.empty_document_count = sum(not tokens for tokens in tokenized)

    def score(self, query: str) -> np.ndarray:
        """Return one score per document, without metadata filtering."""
        return self.index.get_scores(self._tokenize(query))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return document positions and scores; break ties by input order."""
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        scores = self.score(query)
        positions = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(position), float(scores[position])) for position in positions]
