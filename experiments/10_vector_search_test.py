from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.vector_search import VectorSearcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Load corpus
    df = pd.read_parquet(
        PROJECT_ROOT / "data/processed/trials.parquet"
    )

    df = add_trial_documents(df).reset_index(drop=True)

    documents = df["document"].tolist()

    # Create embeddings
    print("Generating embeddings...")

    embedder = DenseEmbedder()

    embeddings = embedder.encode_documents(
        documents,
        batch_size=64,
    )

    # Build reusable vector searcher
    print("Building VectorSearcher...")

    searcher = VectorSearcher(
        embeddings,
        device=0,
    )

    # Encode Q01
    query = "Recruiting NSCLC trials involving pembrolizumab"

    query_embedding = embedder.encode_query(query)

    # Search
    results = searcher.search(
        query_embedding,
        top_k=10,
    )

    # Display results
    print("\n" + "=" * 80)
    print("VECTOR SEARCH TEST")
    print("=" * 80)

    for rank, (position, score) in enumerate(
        results,
        start=1,
    ):
        row = df.iloc[position]

        print(
            f"{rank:2d} | "
            f"{row['nct_id']} | "
            f"{score:.6f} | "
            f"{row['title']} | "
            f"{row['status']}"
        )

    # Basic validation
    assert len(results) == 10

    positions = [position for position, _ in results]
    scores = [score for _, score in results]

    assert len(set(positions)) == 10
    assert np.isfinite(scores).all()

    print("\nVectorSearcher test: PASSED")


if __name__ == "__main__":
    main()