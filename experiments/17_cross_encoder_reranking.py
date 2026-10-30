
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from src.preprocessing.trial_representation import add_trial_documents


DATA_PATH = Path("data/processed/trials.parquet")
QUERIES_PATH = Path("data/evaluation/queries.json")
CANDIDATES_PATH = Path("data/evaluation/weighted_fusion_candidates.csv")

OUTPUT_PATH = Path("data/evaluation/reranked_candidates.csv")
METRICS_PATH = Path("data/evaluation/reranking_metrics.csv")

MODEL_NAME = "BAAI/bge-reranker-base"
BASELINE_CONFIG = "rrf_baseline_50_50"
BATCH_SIZE = 16
MAX_LENGTH = 512

KNOWN_RELEVANT = {
    "Q01": {"NCT06195527", "NCT07061379"}
}


def calculate_metrics(
    ranked_ids: list[str],
    relevant_ids: set[str],
) -> dict[str, float]:
    """Calculate preliminary metrics using known relevance labels."""

    metrics = {}

    for k in (5, 10):
        top_ids = ranked_ids[:k]
        hits = sum(nct_id in relevant_ids for nct_id in top_ids)

        metrics[f"recall_at_{k}"] = (
            hits / len(relevant_ids) if relevant_ids else 0.0
        )
        metrics[f"precision_at_{k}"] = hits / k

    reciprocal_rank = 0.0

    for rank, nct_id in enumerate(ranked_ids, start=1):
        if nct_id in relevant_ids:
            reciprocal_rank = 1.0 / rank
            break

    metrics["mrr"] = reciprocal_rank
    return metrics


def main() -> None:
    print("=" * 80)
    print("CROSS-ENCODER RERANKING")
    print("=" * 80)

    # 1. Load the corpus, queries, and fusion candidates.
    trials = pd.read_parquet(DATA_PATH).reset_index(drop=True)

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    candidates = pd.read_csv(CANDIDATES_PATH)

    if trials.empty or candidates.empty:
        raise ValueError("Corpus or candidate file is empty.")

    required_columns = {
        "query_id", "config", "rank", "position", "nct_id"
    }
    missing = required_columns - set(candidates.columns)

    if missing:
        raise ValueError(
            f"Candidate file is missing columns: {sorted(missing)}"
        )

    # 2. Keep the baseline RRF Top-50 only.
    candidates = candidates[
        candidates["config"] == BASELINE_CONFIG
    ].copy()

    candidates = candidates.sort_values(
        ["query_id", "rank"],
        kind="stable",
    )

    if candidates.empty:
        raise ValueError(
            f"No candidates found for configuration {BASELINE_CONFIG}."
        )

    if candidates.duplicated(["query_id", "nct_id"]).any():
        raise ValueError("Duplicate trial IDs found within a query.")

    # 3. Rebuild documents using the original corpus row positions.
    trials = add_trial_documents(trials)
    documents = trials["document"].tolist()

    # Guard against accidental mismatch between saved positions and corpus.
    for row in candidates.itertuples(index=False):
        position = int(row.position)

        if not 0 <= position < len(trials):
            raise ValueError(
                f"Invalid corpus position {position} for {row.nct_id}."
            )

        actual_nct_id = str(trials.iloc[position]["nct_id"])

        if actual_nct_id != str(row.nct_id):
            raise ValueError(
                f"Corpus mismatch at position {position}: "
                f"expected {row.nct_id}, found {actual_nct_id}."
            )

    query_texts = {
        item["query_id"]: item["query"]
        for item in queries
    }

    unknown_queries = set(candidates["query_id"]) - set(query_texts)

    if unknown_queries:
        raise ValueError(
            f"Queries missing from queries.json: {sorted(unknown_queries)}"
        )

    print(f"Corpus size: {len(trials)}")
    print(f"Queries with candidates: {candidates['query_id'].nunique()}")
    print(f"Baseline candidate rows: {len(candidates)}")

    # 4. Load the pretrained cross-encoder.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {MODEL_NAME} on {device}...")

    model = CrossEncoder(
        MODEL_NAME,
        device=device,
        max_length=MAX_LENGTH,
    )

    all_results = []
    all_metrics = []

    # 5. Rerank each query's Top-50 candidates.
    for query_id, group in candidates.groupby(
        "query_id",
        sort=False,
    ):
        group = group.sort_values("rank", kind="stable").copy()
        raw_query = query_texts[query_id]

        # Use the original natural-language query for reranking.
        pairs = [
            (raw_query, documents[int(position)])
            for position in group["position"]
        ]

        start = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
        )

        reranking_ms = (time.perf_counter() - start) * 1000
        group["cross_encoder_score"] = [
            float(score) for score in scores
        ]

        # Preserve original rank for comparison.
        group["original_rank"] = group["rank"].astype(int)

        # Stable sorting makes ties deterministic.
        group = group.sort_values(
            ["cross_encoder_score", "original_rank"],
            ascending=[False, True],
            kind="stable",
        ).reset_index(drop=True)

        group["reranked_rank"] = range(1, len(group) + 1)
        group["query_text"] = raw_query
        group["reranking_latency_ms"] = reranking_ms

        # Save each query's reranked Top-50.
        for row in group.itertuples(index=False):
            all_results.append(
                {
                    "query_id": query_id,
                    "query_text": raw_query,
                    "nct_id": row.nct_id,
                    "title": trials.iloc[int(row.position)]["title"],
                    "original_rank": row.original_rank,
                    "reranked_rank": row.reranked_rank,
                    "rrf_score": group.loc[group["nct_id"] == row.nct_id, "fusion_score"].iloc[0],
                    "cross_encoder_score": row.cross_encoder_score,
                    "status": row.status,
                    "phase": row.phase,
                    "sex": row.sex,
                    "study_type": row.study_type,
                    "position": int(row.position),
                    "reranking_latency_ms": reranking_ms,
                }
            )

        # 6. Compare original and reranked metrics where labels exist.
        original_group = candidates[
            candidates["query_id"] == query_id
        ].sort_values("rank", kind="stable")

        original_ids = original_group["nct_id"].astype(str).tolist()
        reranked_ids = group["nct_id"].astype(str).tolist()

        if query_id in KNOWN_RELEVANT:
            relevant_ids = KNOWN_RELEVANT[query_id]

            original_metrics = calculate_metrics(
                original_ids, relevant_ids
            )
            reranked_metrics = calculate_metrics(
                reranked_ids, relevant_ids
            )

            all_metrics.append(
                {
                    "query_id": query_id,
                    "config": BASELINE_CONFIG,
                    "method": "original_rrf",
                    **original_metrics,
                    "reranking_latency_ms": 0.0,
                }
            )

            all_metrics.append(
                {
                    "query_id": query_id,
                    "config": BASELINE_CONFIG,
                    "method": "cross_encoder_reranked",
                    **reranked_metrics,
                    "reranking_latency_ms": reranking_ms,
                }
            )

            print(f"\n{query_id}: {raw_query}")
            print(f"Reranking latency: {reranking_ms:.2f} ms")

            for method_name, ranked_ids in [
                ("Original RRF", original_ids),
                ("Cross-encoder", reranked_ids),
            ]:
                print(f"\n{method_name} relevant-trial ranks:")

                for nct_id in sorted(relevant_ids):
                    if nct_id in ranked_ids:
                        rank = ranked_ids.index(nct_id) + 1
                        print(f"  {nct_id}: rank {rank}")
                    else:
                        print(f"  {nct_id}: not in Top-50")

            print("\nTop-5 after reranking:")
            for row in group.head(5).itertuples(index=False):
                print(
                    f"{row.reranked_rank}. {row.nct_id} | "
                    f"score={row.cross_encoder_score:.4f} | "
                    f"{trials.iloc[int(row.position)]['title']}"
                )
        else:
            print(
                f"{query_id}: reranked {len(group)} candidates "
                f"in {reranking_ms:.2f} ms"
            )

    # 7. Save candidates and evaluation metrics.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(all_metrics)

    results_df.to_csv(OUTPUT_PATH, index=False)
    metrics_df.to_csv(METRICS_PATH, index=False)

    print("\n" + "=" * 80)
    print("CROSS-ENCODER RERANKING COMPLETE")
    print("=" * 80)
    print(f"Saved candidate rows: {len(results_df)}")
    print(f"Saved metrics rows: {len(metrics_df)}")
    print(f"Candidates: {OUTPUT_PATH}")
    print(f"Metrics: {METRICS_PATH}")

    if not metrics_df.empty:
        print("\nPreliminary Q01 metric comparison:")
        print(
            metrics_df[
                [
                    "method",
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
