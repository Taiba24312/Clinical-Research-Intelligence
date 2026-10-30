"""
Compare BM25, Dense, and Hybrid retrieval on the development queries.

This is a retrieval comparison experiment, not the final evaluation.
Final relevance judgments will be created later through pooling.
"""

import json
from pathlib import Path

import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_search import VectorSearcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BM25_TOP_K = 100
DENSE_TOP_K = 100
HYBRID_TOP_K = 50


def get_trial_info(df, results, top_k):
    """Convert retrieval positions into readable trial records."""

    output = []

    for rank, (position, score) in enumerate(
        results[:top_k],
        start=1,
    ):
        row = df.iloc[position]

        output.append(
            {
                "rank": rank,
                "nct_id": row["nct_id"],
                "score": score,
                "title": row["title"],
                "status": row["status"],
            }
        )

    return output


def main():
    # ---------------------------------------------------------
    # 1. Load corpus
    # ---------------------------------------------------------

    corpus_path = (
        PROJECT_ROOT
        / "data/processed/trials.parquet"
    )

    df = pd.read_parquet(corpus_path)

    if len(df) != 4695:
        raise ValueError(
            f"Expected 4695 trials, found {len(df)}."
        )

    if df["nct_id"].duplicated().any():
        raise ValueError(
            "Duplicate NCT IDs found."
        )

    df = add_trial_documents(
        df
    ).reset_index(drop=True)

    documents = df["document"].tolist()

    print("=" * 100)
    print("RETRIEVAL DEV-SET COMPARISON")
    print("=" * 100)

    print(
        f"\nCorpus size: {len(df)}"
    )

    # ---------------------------------------------------------
    # 2. Load evaluation queries
    # ---------------------------------------------------------

    evaluation_path = (
        PROJECT_ROOT
        / "data/evaluation/queries.json"
    )

    with evaluation_path.open(
        encoding="utf-8"
    ) as handle:
        queries = json.load(handle)

    print(
        f"Development queries: {len(queries)}"
    )

    # ---------------------------------------------------------
    # 3. Build BM25
    # ---------------------------------------------------------

    print("\nBuilding BM25...")
    bm25 = BM25Retriever(documents)

    # ---------------------------------------------------------
    # 4. Build Dense retrieval
    # ---------------------------------------------------------

    print("\nGenerating dense embeddings...")

    embedder = DenseEmbedder()

    embeddings = embedder.encode_documents(
        documents,
        batch_size=64,
    )

    print(
        f"Embedding matrix: {embeddings.shape}"
    )

    print("\nBuilding GPU vector index...")

    vector_searcher = VectorSearcher(
        embeddings,
        device=0,
    )

    # ---------------------------------------------------------
    # 5. Hybrid
    # ---------------------------------------------------------

    hybrid = HybridRetriever(
        fusion_k=60
    )

    # ---------------------------------------------------------
    # 6. Run every query
    # ---------------------------------------------------------

    all_results = []

    for query_item in queries:

        query_id = query_item["query_id"]
        query_text = query_item["query"]

        print("\n" + "=" * 100)
        print(
            f"{query_id}: {query_text}"
        )
        print("=" * 100)

        # -----------------------------
        # BM25
        # -----------------------------

        bm25_results = bm25.search(
            query_text,
            top_k=BM25_TOP_K,
        )

        # -----------------------------
        # Dense
        # -----------------------------

        query_embedding = (
            embedder.encode_query(
                query_text
            )
        )

        dense_results = (
            vector_searcher.search(
                query_embedding,
                top_k=DENSE_TOP_K,
            )
        )

        # -----------------------------
        # Hybrid
        # -----------------------------

        hybrid_results = hybrid.fuse(
            lexical_results=bm25_results,
            dense_results=dense_results,
            top_k=HYBRID_TOP_K,
        )

        # -----------------------------
        # Display Top 5
        # -----------------------------

        for method, results in [
            ("BM25", bm25_results),
            ("DENSE", dense_results),
            ("HYBRID", hybrid_results),
        ]:

            print(f"\n{method} TOP 5")

            for item in get_trial_info(
                df,
                results,
                top_k=5,
            ):
                print(
                    f"{item['rank']:2d} | "
                    f"{item['nct_id']} | "
                    f"{item['status']} | "
                    f"{item['title']}"
                )

        # -----------------------------
        # Save complete results
        # -----------------------------

        for method, results, candidate_k in [
            (
                "bm25",
                bm25_results,
                BM25_TOP_K,
            ),
            (
                "dense",
                dense_results,
                DENSE_TOP_K,
            ),
            (
                "hybrid",
                hybrid_results,
                HYBRID_TOP_K,
            ),
        ]:

            trial_results = get_trial_info(
                df,
                results,
                candidate_k,
            )

            for item in trial_results:

                all_results.append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "method": method,
                        "rank": item["rank"],
                        "nct_id": item["nct_id"],
                        "score": item["score"],
                        "title": item["title"],
                        "status": item["status"],
                    }
                )

    # ---------------------------------------------------------
    # 7. Save results
    # ---------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "data/evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "dev_retrieval_results.csv"
    )

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 100)
    print("DEV-SET COMPARISON COMPLETE")
    print("=" * 100)

    print(
        f"\nSaved results to:\n{output_path}"
    )

    print(
        f"\nTotal result rows: "
        f"{len(results_df)}"
    )

    print(
        "\nBM25 candidates/query: "
        f"{BM25_TOP_K}"
    )

    print(
        "Dense candidates/query: "
        f"{DENSE_TOP_K}"
    )

    print(
        "Hybrid candidates/query: "
        f"{HYBRID_TOP_K}"
    )


if __name__ == "__main__":
    main()