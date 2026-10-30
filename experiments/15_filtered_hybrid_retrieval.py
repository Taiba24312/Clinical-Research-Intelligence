
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.preprocessing.trial_representation import add_trial_documents
from src.query_understanding.parser import parse_query
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embeddings import DenseEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.metadata_filter import MetadataFilter


DATA_PATH = Path("data/processed/trials.parquet")
QUERIES_PATH = Path("data/evaluation/queries.json")
OUTPUT_PATH = Path("data/evaluation/filtered_hybrid_candidates.csv")

TOP_K_RETRIEVAL = 100
TOP_K_HYBRID = 50


def main() -> None:
    # 1. Load corpus and evaluation queries
    trials = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    print("=" * 80)
    print("FILTERED HYBRID RETRIEVAL")
    print("=" * 80)
    print(f"Corpus size: {len(trials)}")
    print(f"Queries: {len(queries)}")

    if trials.empty:
        raise ValueError("The processed trial corpus is empty.")

    # 2. Build one searchable document per trial
    # add_trial_documents returns a DataFrame containing a 'document' column.
    trials = add_trial_documents(trials)
    documents = trials["document"].tolist()

    if len(documents) != len(trials):
        raise ValueError("Document count does not match trial count.")

    # 3. Build BM25 lexical retrieval index
    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever(documents)
    print(f"BM25 corpus size: {bm25.corpus_size}")

    # 4. Generate dense document embeddings
    print("\nGenerating document embeddings...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    embedder = DenseEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        device=device,
    )

    embeddings = embedder.encode_documents(documents, batch_size=64)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.shape[0] != len(trials):
        raise ValueError("Embedding count does not match trial count.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Document embeddings contain NaN or infinite values.")

    # Embeddings are normalized by the existing DenseEmbedder.
    embeddings_tensor = torch.from_numpy(embeddings).to(device)

    # 5. Initialize metadata filtering and RRF fusion
    metadata_filter = MetadataFilter(trials)
    hybrid = HybridRetriever(fusion_k=60)

    all_results: list[dict] = []

    # 6. Retrieve candidates for every evaluation query
    for item in queries:
        query_id = item["query_id"]
        raw_query = item["query"]

        parsed = parse_query(raw_query)
        semantic_query = parsed["semantic_query"]
        filters = parsed["filters"]

        print("\n" + "-" * 80)
        print(f"{query_id}: {raw_query}")
        print(f"Semantic query: {semantic_query}")
        print(f"Metadata filters: {filters}")

        # Apply structured constraints before ranking.
        allowed_positions = metadata_filter.apply(filters)

        print(f"Trials satisfying filters: {len(allowed_positions)}")

        if len(allowed_positions) == 0:
            print("No trials satisfy the supplied filters.")
            continue

        allowed_array = np.asarray(allowed_positions, dtype=np.int64)

        # ----- BM25 retrieval over allowed trials -----
        start = time.perf_counter()

        # Score the full corpus, then rank only filter-eligible positions.
        # This preserves BM25's corpus-wide IDF statistics.
        bm25_scores = bm25.score(semantic_query)
        allowed_bm25_scores = bm25_scores[allowed_array]

        actual_k = min(TOP_K_RETRIEVAL, len(allowed_positions))
        bm25_order = np.argsort(
            -allowed_bm25_scores,
            kind="stable",
        )[:actual_k]

        lexical_results = [
            (
                int(allowed_array[i]),
                float(allowed_bm25_scores[i]),
            )
            for i in bm25_order
        ]

        bm25_ms = (time.perf_counter() - start) * 1000

        # ----- Dense retrieval over allowed trials -----
        start = time.perf_counter()

        query_embedding = embedder.encode_query(semantic_query)
        query_tensor = torch.as_tensor(
            np.asarray(query_embedding, dtype=np.float32),
            device=device,
        ).reshape(-1)

        if query_tensor.numel() != embeddings_tensor.shape[1]:
            raise ValueError(
                f"{query_id}: query/document embedding dimensions do not match."
            )

        allowed_embeddings = embeddings_tensor[allowed_array]

        # Normalized vectors + dot product = cosine similarity.
        dense_scores = allowed_embeddings @ query_tensor

        dense_k = min(TOP_K_RETRIEVAL, len(allowed_positions))
        dense_top_scores, dense_top_indices = torch.topk(
            dense_scores,
            k=dense_k,
        )

        dense_top_scores = dense_top_scores.cpu().numpy()
        dense_top_indices = dense_top_indices.cpu().numpy()

        dense_results = [
            (
                int(allowed_array[local_index]),
                float(score),
            )
            for local_index, score in zip(
                dense_top_indices,
                dense_top_scores,
            )
        ]

        dense_ms = (time.perf_counter() - start) * 1000

        # ----- Reciprocal Rank Fusion -----
        hybrid_results = hybrid.fuse(
            lexical_results,
            dense_results,
            top_k=TOP_K_HYBRID,
        )

        final_positions = [position for position, _ in hybrid_results]
        allowed_set = set(allowed_positions)

        filter_violations = [
            position
            for position in final_positions
            if position not in allowed_set
        ]

        if filter_violations:
            raise AssertionError(
                f"{query_id}: metadata filter violations at positions "
                f"{filter_violations}"
            )

        print(f"BM25 latency: {bm25_ms:.2f} ms")
        print(f"Dense latency: {dense_ms:.2f} ms")
        print(f"Hybrid candidates: {len(hybrid_results)}")
        print(f"Filter violations: {len(filter_violations)}")

        # Save every returned candidate (up to Top-50), not only the Top-5.
        # This output will be used by the reranking step.
        for rank, (position, rrf_score) in enumerate(
            hybrid_results,
            start=1,
        ):
            trial = trials.iloc[position]

            all_results.append(
                {
                    "query_id": query_id,
                    "rank": rank,
                    "position": position,
                    "nct_id": trial["nct_id"],
                    "title": trial["title"],
                    "status": trial["status"],
                    "phase": trial["phase"],
                    "sex": trial["sex"],
                    "study_type": trial["study_type"],
                    "min_age": trial["min_age"],
                    "max_age": trial["max_age"],
                    "rrf_score": rrf_score,
                    "semantic_query": semantic_query,
                    "filters": json.dumps(filters, sort_keys=True),
                }
            )

        # Display Top-5 for inspection; the CSV retains Top-50.
        print("\nTop-5 hybrid results:")
        for rank, (position, rrf_score) in enumerate(
            hybrid_results[:5],
            start=1,
        ):
            trial = trials.iloc[position]
            print(
                f"{rank}. {trial['nct_id']} | "
                f"{trial['status']} | "
                f"{trial['phase']} | "
                f"RRF={rrf_score:.6f} | "
                f"{trial['title']}"
            )

    # 7. Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"Queries processed: {len(queries)}")
    print(f"Saved candidate rows: {len(results_df)}")
    print(f"Output: {OUTPUT_PATH}")

    if not results_df.empty:
        print("\nCandidates per query:")
        print(results_df.groupby("query_id").size().to_string())


if __name__ == "__main__":
    main()
