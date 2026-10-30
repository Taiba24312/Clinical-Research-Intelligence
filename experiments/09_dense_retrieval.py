"""
Dense retrieval experiment using BGE-small + FAISS.

This experiment:
1. Loads the processed 4,695-trial corpus.
2. Creates dense embeddings for all trial documents.
3. Builds a FAISS inner-product index.
4. Runs Q01.
5. Reports top-ranked trials.
6. Reports Q01 sanity-check metrics.
7. Measures embedding and retrieval latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
import torch
import faiss
import numpy as np
import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents
from src.retrieval.embeddings import DenseEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWN_RELEVANT_IDS = {
    "NCT07061379",
    "NCT06195527",
}


def recall_at_k(ranked_ids, relevant_ids, k):
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    retrieved = ranked_ids[:k]
    return sum(nct_id in relevant_ids for nct_id in retrieved) / k


def reciprocal_rank(ranked_ids, relevant_ids):
    for rank, nct_id in enumerate(ranked_ids, start=1):
        if nct_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def main():
    # ---------------------------------------------------------
    # 1. Load processed corpus
    # ---------------------------------------------------------
    corpus_path = PROJECT_ROOT / "data/processed/trials.parquet"

    df = pd.read_parquet(corpus_path)

    if len(df) != 4695:
        raise ValueError(
            f"Expected 4,695 unique trials, but found {len(df)}."
        )

    if df["nct_id"].duplicated().any():
        raise ValueError("Duplicate NCT IDs found in corpus.")

    # ---------------------------------------------------------
    # 2. Create searchable trial documents
    # ---------------------------------------------------------
    df = add_trial_documents(df).reset_index(drop=True)

    documents = df["document"].tolist()

    print("=" * 80)
    print("DENSE RETRIEVAL EXPERIMENT")
    print("=" * 80)

    print("\nCorpus size:", len(documents))

    # ---------------------------------------------------------
    # 3. Load embedding model
    # ---------------------------------------------------------
    print("\nLoading BGE-small model...")

    embedder = DenseEmbedder()

    print("Model:", embedder.model_name)
    print("Embedding dimension:", embedder.dimension)

    # ---------------------------------------------------------
    # 4. Generate embeddings
    # ---------------------------------------------------------
    print("\nGenerating document embeddings...")

    embedding_start = perf_counter()

    embeddings = embedder.encode_documents(
        documents,
        batch_size=64,
    )

    embedding_time_ms = (
        perf_counter() - embedding_start
    ) * 1000

    print(
        f"\nEmbedding generation time: "
        f"{embedding_time_ms / 1000:.3f} seconds"
    )

    print("Embedding matrix shape:", embeddings.shape)
    print("Embedding dtype:", embeddings.dtype)

    # ---------------------------------------------------------
    # 5. Validate embeddings
    # ---------------------------------------------------------
    if embeddings.shape != (len(df), embedder.dimension):
        raise ValueError(
            "Unexpected embedding matrix shape."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # ---------------------------------------------------------
    # 6. Build FAISS index
    # ---------------------------------------------------------
    print("\nBuilding FAISS index...")

    index_start = perf_counter()

    # Embeddings are normalized, so inner product
    # is equivalent to cosine similarity.
     # Build the FAISS index on CPU first, then move it to GPU.
    cpu_index = faiss.IndexFlatIP(embedder.dimension)

    # Use GPU 0 (RTX PRO 6000 Blackwell) for vector search.
    gpu_resources = faiss.StandardGpuResources()

    index = faiss.index_cpu_to_gpu(
        gpu_resources,
        0,
        cpu_index,
    )

    index.add(embeddings)

    index_time_ms = (
        perf_counter() - index_start
    ) * 1000

    print(
        f"FAISS index creation time: "
        f"{index_time_ms:.3f} ms"
    )

    print("FAISS index size:", index.ntotal)

    # ---------------------------------------------------------
    # 7. Load Q01
    # ---------------------------------------------------------
    evaluation_path = (
        PROJECT_ROOT / "data/evaluation/queries.json"
    )

    with evaluation_path.open(encoding="utf-8") as handle:
        queries = json.load(handle)

    query = next(
        item
        for item in queries
        if item["query_id"] == "Q01"
    )

    query_text = query["query"]

    print("\nQuery:")
    print(query_text)

    # ---------------------------------------------------------
    # 8. Encode query
    # ---------------------------------------------------------
    query_embedding = embedder.encode_query(
        query_text
    )

    query_embedding = query_embedding.reshape(1, -1)

    # ---------------------------------------------------------
    # 9. Dense retrieval
    # ---------------------------------------------------------
    print("\nSearching FAISS index...")

    search_start = perf_counter()

    scores, positions = index.search(
        query_embedding,
        10,
    )

    search_latency_ms = (
        perf_counter() - search_start
    ) * 1000

    # ---------------------------------------------------------
    # 10. Display top-10
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 10 DENSE RETRIEVAL RESULTS")
    print("=" * 80)

    ranked_ids = []

    for rank, (position, score) in enumerate(
        zip(positions[0], scores[0]),
        start=1,
    ):
        nct_id = df.iloc[position]["nct_id"]
        ranked_ids.append(nct_id)

        row = df.iloc[position]

        print(
            f"{rank:2d} | "
            f"{nct_id} | "
            f"{score:.6f} | "
            f"{row['title']} | "
            f"{row['status']}"
        )

    # ---------------------------------------------------------
    # 11. Known relevant trial ranks
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("KNOWN Q01 RELEVANT TRIAL RANKS")
    print("=" * 80)

    # FAISS GPU cannot return k > 2048 in one search.
    # For exact evaluation over the full 4,695-trial corpus,
    # compute all cosine similarities directly on the GPU.

    document_embeddings_gpu = torch.from_numpy(
        embeddings
    ).to("cuda:0")

    query_embedding_gpu = torch.from_numpy(
        query_embedding
    ).to("cuda:0")

    full_similarity_scores = (
        document_embeddings_gpu @ query_embedding_gpu.T
    ).squeeze(1)

    sorted_positions = torch.argsort(
        full_similarity_scores,
        descending=True,
    )

    full_positions = sorted_positions.cpu().numpy()

    full_scores = full_similarity_scores[
        sorted_positions
    ].cpu().numpy()

    full_ranked_ids = [
        df.iloc[position]["nct_id"]
        for position in full_positions
    ]

    for nct_id in sorted(KNOWN_RELEVANT_IDS):
        rank = full_ranked_ids.index(nct_id) + 1

        score = full_scores[
            full_ranked_ids.index(nct_id)
        ]

        print(
            f"{nct_id} | "
            f"score={score:.6f} | "
            f"rank={rank}"
        )

    # ---------------------------------------------------------
    # 12. Sanity-check metrics
    # ---------------------------------------------------------
    recall_5 = recall_at_k(
        full_ranked_ids,
        KNOWN_RELEVANT_IDS,
        5,
    )

    recall_10 = recall_at_k(
        full_ranked_ids,
        KNOWN_RELEVANT_IDS,
        10,
    )

    precision_5 = precision_at_k(
        full_ranked_ids,
        KNOWN_RELEVANT_IDS,
        5,
    )

    mrr = reciprocal_rank(
        full_ranked_ids,
        KNOWN_RELEVANT_IDS,
    )

    print("\n" + "=" * 80)
    print("Q01 DENSE RETRIEVAL SANITY-CHECK METRICS")
    print("=" * 80)

    print(f"Recall@5:    {recall_5:.4f}")
    print(f"Recall@10:   {recall_10:.4f}")
    print(f"Precision@5: {precision_5:.4f}")
    print(f"MRR:         {mrr:.4f}")
    print(f"Search latency: {search_latency_ms:.3f} ms")

    print("\n" + "=" * 80)
    print("DENSE RETRIEVAL EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()