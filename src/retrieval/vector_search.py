"""
Reusable GPU vector search using FAISS.

Expects normalized embeddings so inner product is equivalent
to cosine similarity.
"""

from __future__ import annotations

from typing import Sequence

import faiss
import numpy as np


class VectorSearcher:
    """Search normalized dense embeddings using GPU FAISS."""

    def __init__(
        self,
        embeddings: np.ndarray,
        *,
        device: int = 0,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain NaN or infinite values."
            )

        self.dimension = embeddings.shape[1]
        self.corpus_size = embeddings.shape[0]

        cpu_index = faiss.IndexFlatIP(
            self.dimension
        )

        self.gpu_resources = faiss.StandardGpuResources()

        self.index = faiss.index_cpu_to_gpu(
            self.gpu_resources,
            device,
            cpu_index,
        )

        self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return document positions and similarity scores."""

        if top_k < 1:
            raise ValueError(
                "top_k must be positive."
            )

        if top_k > 2048:
            raise ValueError(
                "GPU FAISS search supports top_k <= 2048 "
                "for this index."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (
            1,
            self.dimension,
        ):
            raise ValueError(
                "Query embedding has an unexpected shape."
            )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains NaN or infinite values."
            )

        scores, positions = self.index.search(
            query_embedding,
            top_k,
        )

        return [
            (
                int(position),
                float(score),
            )
            for position, score in zip(
                positions[0],
                scores[0],
            )
        ]