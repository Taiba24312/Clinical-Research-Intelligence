"""
Hybrid retrieval experiment using BM25 + Dense Retrieval + RRF.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_search import VectorSearcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUERY = "Recruiting NSCLC trials involving pembrolizumab"

KNOWN_RELEVANT = {
    "NCT07061379",
    "NCT06195527",
}


def find_rank(results, nct_id, df):
    """Return rank and score for a trial ID."""
    for rank, (position, score) in enumerate(
        results,
        start=1,
    ):
        if df.iloc[position]["nct_id"] == nct_id:
            return rank, score

    return None, None


def main():
    # ------------------------------------------------------------------
    # 1. Load processed trials
    # ------------------------------------------------------------------

    df = pd.read_parquet(
        PROJECT_ROOT / "data/processed/trials.parquet"
    )

    df = add_trial_documents(df).reset_index(drop=True)

    documents = df["document"].tolist()

    print(f"Corpus size: {len(df)}")

    # ------------------------------------------------------------------
    # 2. BM25 retrieval
    # ------------------------------------------------------------------

    print("\nBuilding BM25...")
    bm25 = BM25Retriever(documents)

    bm25_results = bm25.search(
        QUERY,
        top_k=100,
    )

    print(
        f"BM25 candidates: {len(bm25_results)}"
    )

    # ------------------------------------------------------------------
    # 3. Dense retrieval
    # ------------------------------------------------------------------

    print("\nGenerating dense embeddings...")

    embedder = DenseEmbedder()

    embeddings = embedder.encode_documents(
        documents,
        batch_size=64,
    )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    # ------------------------------------------------------------------
    # 4. GPU vector search
    # ------------------------------------------------------------------

    print("\nBuilding GPU vector index...")

    vector_searcher = VectorSearcher(
        embeddings,
        device=0,
    )

    query_embedding = embedder.encode_query(
        QUERY
    )

    dense_results = vector_searcher.search(
        query_embedding,
        top_k=100,
    )

    print(
        f"Dense candidates: {len(dense_results)}"
    )

    # ------------------------------------------------------------------
    # 5. Hybrid RRF fusion
    # ------------------------------------------------------------------

    print("\nRunning RRF fusion...")

    hybrid = HybridRetriever(
        fusion_k=60,
    )

    hybrid_results = hybrid.fuse(
        lexical_results=bm25_results,
        dense_results=dense_results,
        top_k=50,
    )

    # ------------------------------------------------------------------
    # 6. Display top 10
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("HYBRID RETRIEVAL — TOP 10")
    print("=" * 100)

    for rank, (position, score) in enumerate(
        hybrid_results[:10],
        start=1,
    ):
        row = df.iloc[position]

        print(
            f"{rank:2d} | "
            f"{row['nct_id']} | "
            f"{score:.6f} | "
            f"{row['status']} | "
            f"{row['title']}"
        )

    # ------------------------------------------------------------------
    # 7. Compare known relevant trials
    # ------------------------------------------------------------------

    print("\n" + "=" * 100)
    print("KNOWN Q01 RELEVANT TRIALS")
    print("=" * 100)

    for nct_id in sorted(KNOWN_RELEVANT):

        bm25_rank, bm25_score = find_rank(
            bm25_results,
            nct_id,
            df,
        )

        dense_rank, dense_score = find_rank(
            dense_results,
            nct_id,
            df,
        )

        hybrid_rank, hybrid_score = find_rank(
            hybrid_results,
            nct_id,
            df,
        )

        print(f"\n{nct_id}")

        print(
            f"  BM25   : rank={bm25_rank}, "
            f"score={bm25_score}"
        )

        print(
            f"  Dense  : rank={dense_rank}, "
            f"score={dense_score}"
        )

        print(
            f"  Hybrid : rank={hybrid_rank}, "
            f"score={hybrid_score}"
        )

    # ------------------------------------------------------------------
    # 8. Basic sanity checks
    # ------------------------------------------------------------------

    assert len(hybrid_results) == 50

    positions = [
        position
        for position, _ in hybrid_results
    ]

    scores = [
        score
        for _, score in hybrid_results
    ]

    assert len(set(positions)) == 50
    assert np.isfinite(scores).all()

    print("\n" + "=" * 100)
    print("HYBRID RETRIEVAL TEST: PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()