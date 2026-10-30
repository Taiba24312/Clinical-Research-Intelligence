"""
Dense embedding utilities for ClinicalTrials.gov trial documents.

Uses BAAI/bge-small-en-v1.5 to convert trial text into dense vectors.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseEmbedder:
    """Generate normalized dense embeddings for trial documents and queries."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

    def encode_documents(
        self,
        documents: Sequence[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode trial documents into normalized embeddings."""

        embeddings = self.model.encode(
            list(documents),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a user query into one normalized embedding."""

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""

        return self.model.get_sentence_embedding_dimension()