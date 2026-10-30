"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from collections.abc import Sequence


class HybridRetriever:
    """Fuse lexical and dense rankings using Reciprocal Rank Fusion."""

    def __init__(self, *, fusion_k: int = 60) -> None:
        if fusion_k < 1:
            raise ValueError("fusion_k must be positive.")

        self.fusion_k = fusion_k

    def fuse(
        self,
        lexical_results: Sequence[tuple[int, float]],
        dense_results: Sequence[tuple[int, float]],
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """Fuse BM25 and dense rankings using RRF."""

        if top_k < 1:
            raise ValueError("top_k must be positive.")

        lexical_ranks = {
            position: rank
            for rank, (position, _) in enumerate(
                lexical_results,
                start=1,
            )
        }

        dense_ranks = {
            position: rank
            for rank, (position, _) in enumerate(
                dense_results,
                start=1,
            )
        }

        candidate_positions = (
            set(lexical_ranks)
            | set(dense_ranks)
        )

        fused_results = []

        for position in candidate_positions:
            score = 0.0

            if position in lexical_ranks:
                rank = lexical_ranks[position]
                score += 1.0 / (self.fusion_k + rank)

            if position in dense_ranks:
                rank = dense_ranks[position]
                score += 1.0 / (self.fusion_k + rank)

            fused_results.append(
                (position, score)
            )

        # Deterministic tie-breaking by document position.
        fused_results.sort(
            key=lambda item: (-item[1], item[0])
        )

        return fused_results[:top_k]