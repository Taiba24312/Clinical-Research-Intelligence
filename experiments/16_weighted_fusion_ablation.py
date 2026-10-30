
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

OUTPUT_PATH = Path("data/evaluation/weighted_fusion_candidates.csv")
METRICS_PATH = Path("data/evaluation/weighted_fusion_metrics.csv")

TOP_K_RETRIEVAL = 100
TOP_K_FUSION = 50
FUSION_K = 60

# Provisional relevance labels available for Q01 only.
KNOWN_RELEVANT = {
    "Q01": {"NCT06195527", "NCT07061379"}
}

# The first configuration is the existing equal-weight RRF baseline.
CONFIGS = {
    "rrf_baseline_50_50": (0.50, 0.50),
    "lexical_heavy_75_25": (0.75, 0.25),
    "dense_heavy_25_75": (0.25, 0.75),
}


def weighted_fuse(
    lexical_results: list[tuple[int, float]],
    dense_results: list[tuple[int, float]],
    *,
    lexical_weight: float,
    dense_weight: float,
    top_k: int = 50,
    fusion_k: int = 60,
) -> list[tuple[int, float]]:
    """Fuse ranked lists using weighted Reciprocal Rank Fusion."""

    if lexical_weight < 0 or dense_weight < 0:
        raise ValueError("Fusion weights cannot be negative.")

    if lexical_weight + dense_weight == 0:
        raise ValueError("At least one fusion weight must be positive.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    lexical_ranks = {
        position: rank
        for rank, (position, _) in enumerate(lexical_results, start=1)
    }
    dense_ranks = {
        position: rank
        for rank, (position, _) in enumerate(dense_results, start=1)
    }

    candidate_positions = set(lexical_ranks) | set(dense_ranks)
    fused_scores = {}

    for position in candidate_positions:
        score = 0.0

        if position in lexical_ranks:
            score += lexical_weight / (
                fusion_k + lexical_ranks[position]
            )

        if position in dense_ranks:
            score += dense_weight / (
                fusion_k + dense_ranks[position]
            )

        fused_scores[position] = score

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return ranked[:top_k]


def calculate_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> dict[str, float]:
    """Calculate preliminary retrieval metrics for known labels."""

    metrics = {}

    for k in (5, 10):
        top_ids = ranked_ids[:k]
        hits = sum(nct_id in relevant_ids for nct_id in top_ids)

        metrics[f"recall_at_{k}"] = (
            hits / len(relevant_ids) if relevant_ids else 0.0
        )
        metrics[f"precision_at_{k}"] = (
            hits / k
        )

    reciprocal_rank = 0.0

    for rank, nct_id in enumerate(ranked_ids, start=1):
        if nct_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    metrics["mrr"] = reciprocal_rank
    return metrics


def main() -> None:
    # 1. Load corpus and queries
    trials = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if trials.empty:
        raise ValueError("Processed trial corpus is empty.")

    print("=" * 80)
    print("WEIGHTED FUSION ABLATION")
    print("=" * 80)
    print(f"Corpus size: {len(trials)}")
    print(f"Queries: {len(queries)}")
    print(f"Configurations: {len(CONFIGS)}")

    # 2. Build searchable documents
    trials = add_trial_documents(trials)
    documents = trials["document"].tolist()

    # 3. Build retrieval indexes once for all experiments
    print("\nBuilding BM25 index...")
    bm25 = BM25Retriever(documents)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    print("Generating dense document embeddings...")

    embedder = DenseEmbedder(
        model_name="BAAI/bge-small-en-v1.5",
        device=device,
    )

    embeddings = np.asarray(
        embedder.encode_documents(documents, batch_size=64),
        dtype=np.float32,
    )

    if embeddings.shape[0] != len(trials):
        raise ValueError("Embedding count does not match corpus size.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    embeddings_tensor = torch.from_numpy(embeddings).to(device)

    metadata_filter = MetadataFilter(trials)
    baseline_hybrid = HybridRetriever(fusion_k=FUSION_K)

    all_results = []
    all_metrics = []

    # 4. Retrieve the same lexical and dense candidates per query.
    # Only the fusion weights will change between configurations.
    for item in queries:
        query_id = item["query_id"]
        raw_query = item["query"]

        parsed = parse_query(raw_query)
        semantic_query = parsed["semantic_query"]
        filters = parsed["filters"]

        allowed_positions = metadata_filter.apply(filters)

        print("\n" + "-" * 80)
        print(f"{query_id}: {raw_query}")
        print(f"Filters: {filters}")
        print(f"Eligible trials: {len(allowed_positions)}")

        if len(allowed_positions) == 0:
            print("No trials satisfy these filters.")
            continue

        allowed_array = np.asarray(
            allowed_positions,
            dtype=np.int64,
        )

        # BM25: preserve corpus-wide IDF, rank only eligible trials.
        start = time.perf_counter()

        bm25_scores = bm25.score(semantic_query)
        allowed_bm25_scores = bm25_scores[allowed_array]

        retrieval_k = min(
            TOP_K_RETRIEVAL,
            len(allowed_positions),
        )

        lexical_order = np.argsort(
            -allowed_bm25_scores,
            kind="stable",
        )[:retrieval_k]

        lexical_results = [
            (
                int(allowed_array[i]),
                float(allowed_bm25_scores[i]),
            )
            for i in lexical_order
        ]

        bm25_ms = (time.perf_counter() - start) * 1000

        # Dense retrieval: rank only eligible trial embeddings.
        start = time.perf_counter()

        query_embedding = embedder.encode_query(semantic_query)
        query_tensor = torch.as_tensor(
            np.asarray(query_embedding, dtype=np.float32),
            device=device,
        ).reshape(-1)

        allowed_embeddings = embeddings_tensor[allowed_array]
        dense_scores = allowed_embeddings @ query_tensor

        dense_order = torch.argsort(
            dense_scores,
            descending=True,
            stable=True,
        )[:retrieval_k]

        dense_results = [
            (
                int(allowed_array[local_index]),
                float(dense_scores[local_index].item()),
            )
            for local_index in dense_order.cpu().tolist()
        ]

        dense_ms = (time.perf_counter() - start) * 1000

        # 5. Evaluate each fusion configuration on identical candidates.
        rankings_for_query = {}

        for config_name, (lexical_weight, dense_weight) in CONFIGS.items():

            if config_name == "rrf_baseline_50_50":
                # Use the existing implementation as the baseline.
                hybrid_results = baseline_hybrid.fuse(
                    lexical_results,
                    dense_results,
                    top_k=TOP_K_FUSION,
                )
            else:
                hybrid_results = weighted_fuse(
                    lexical_results,
                    dense_results,
                    lexical_weight=lexical_weight,
                    dense_weight=dense_weight,
                    top_k=TOP_K_FUSION,
                    fusion_k=FUSION_K,
                )

            final_positions = [
                position for position, _ in hybrid_results
            ]

            allowed_set = set(allowed_positions)

            violations = [
                position
                for position in final_positions
                if position not in allowed_set
            ]

            if violations:
                raise AssertionError(
                    f"{query_id}/{config_name}: "
                    f"metadata filter violations: {violations}"
                )

            ranked_ids = [
                str(trials.iloc[position]["nct_id"])
                for position in final_positions
            ]

            rankings_for_query[config_name] = ranked_ids

            # Save Top-50 results for downstream reranking.
            for rank, (position, fusion_score) in enumerate(
                hybrid_results,
                start=1,
            ):
                trial = trials.iloc[position]

                all_results.append(
                    {
                        "query_id": query_id,
                        "config": config_name,
                        "lexical_weight": lexical_weight,
                        "dense_weight": dense_weight,
                        "rank": rank,
                        "position": position,
                        "nct_id": trial["nct_id"],
                        "title": trial["title"],
                        "status": trial["status"],
                        "phase": trial["phase"],
                        "sex": trial["sex"],
                        "study_type": trial["study_type"],
                        "fusion_score": fusion_score,
                        "semantic_query": semantic_query,
                        "filters": json.dumps(
                            filters,
                            sort_keys=True,
                        ),
                    }
                )

            # Compute metrics only where manual relevance labels exist.
            if query_id in KNOWN_RELEVANT:
                metrics = calculate_metrics(
                    ranked_ids,
                    KNOWN_RELEVANT[query_id],
                )

                all_metrics.append(
                    {
                        "query_id": query_id,
                        "config": config_name,
                        "lexical_weight": lexical_weight,
                        "dense_weight": dense_weight,
                        "bm25_ms": bm25_ms,
                        "dense_ms": dense_ms,
                        **metrics,
                    }
                )

            print(
                f"{config_name}: "
                f"{len(hybrid_results)} candidates, "
                f"filter violations=0"
            )

        # Show whether the two known Q01 trials changed rank.
        if query_id in KNOWN_RELEVANT:
            print("\nKnown Q01 relevance-label ranks:")

            for config_name, ranked_ids in rankings_for_query.items():
                print(f"\n{config_name}")

                for relevant_id in sorted(KNOWN_RELEVANT[query_id]):
                    if relevant_id in ranked_ids:
                        rank = ranked_ids.index(relevant_id) + 1
                        print(f"  {relevant_id}: rank {rank}")
                    else:
                        print(f"  {relevant_id}: not in Top-{TOP_K_FUSION}")

    # 6. Save all experiment results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(all_metrics)

    results_df.to_csv(OUTPUT_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\n" + "=" * 80)
    print("WEIGHTED FUSION ABLATION COMPLETE")
    print("=" * 80)
    print(f"Candidate rows: {len(results_df)}")
    print(f"Metrics rows: {len(metrics_df)}")
    print(f"Candidates saved to: {OUTPUT_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")

    if not results_df.empty:
        print("\nCandidates per query/configuration:")
        print(
            results_df.groupby(["query_id", "config"])
            .size()
            .to_string()
        )

    if not metrics_df.empty:
        print("\nPreliminary Q01 metrics:")
        print(
            metrics_df[
                [
                    "config",
                    "recall_at_5",
                    "recall_at_10",
                    "precision_at_5",
                    "precision_at_10",
                    "mrr",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
