
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/trials.parquet")
RRF_PATH = Path("data/evaluation/weighted_fusion_candidates.csv")
RERANKED_PATH = Path("data/evaluation/reranked_candidates.csv")

OUTPUT_PATH = Path("data/evaluation/q01_manual_relevance_review.csv")

QUERY_ID = "Q01"
TOP_K = 10

KNOWN_RELEVANT = {"NCT06195527", "NCT07061379"}


def main():
    trials = pd.read_parquet(DATA_PATH)

    rrf = pd.read_csv(RRF_PATH)
    reranked = pd.read_csv(RERANKED_PATH)

    # Original equal-weight RRF Top-10.
    rrf = rrf[
        (rrf["query_id"] == QUERY_ID)
        & (rrf["config"] == "rrf_baseline_50_50")
    ].sort_values("rank")

    rrf = rrf.head(TOP_K)[
        ["nct_id", "rank"]
    ].rename(columns={"rank": "original_rrf_rank"})

    # Cross-encoder Top-10.
    reranked_q = reranked[
        reranked["query_id"] == QUERY_ID
    ].sort_values("reranked_rank")

    reranked_q = reranked_q.head(TOP_K)[
        ["nct_id", "reranked_rank", "cross_encoder_score"]
    ]

    # Union of candidates, retaining each system's rank.
    pooled = rrf.merge(reranked_q, on="nct_id", how="outer")

    # Add actual trial details from the processed corpus.
    detail_columns = [
        "nct_id",
        "title",
        "conditions",
        "interventions",
        "summary",
        "status",
        "phase",
    ]

    details = trials[detail_columns].copy()
    details["nct_id"] = details["nct_id"].astype(str)
    pooled["nct_id"] = pooled["nct_id"].astype(str)

    review = pooled.merge(details, on="nct_id", how="left")

    # Keep existing labels if this script is rerun.
    if OUTPUT_PATH.exists():
        previous = pd.read_csv(OUTPUT_PATH)
        if {"nct_id", "relevance_label", "notes"}.issubset(previous.columns):
            previous = previous[
                ["nct_id", "relevance_label", "notes"]
            ]
            review = review.merge(
                previous,
                on="nct_id",
                how="left",
            )
        else:
            review["relevance_label"] = pd.NA
            review["notes"] = ""
    else:
        review["relevance_label"] = pd.NA
        review["notes"] = ""

    # Seed only the two previously established positive labels.
    # Preserve any existing manual label.
    for nct_id in KNOWN_RELEVANT:
        mask = review["nct_id"] == nct_id
        review.loc[
            mask & review["relevance_label"].isna(),
            "relevance_label",
        ] = 1

    review["relevance_label"] = review["relevance_label"].astype("Int64")

    review = review.sort_values(
        ["original_rrf_rank", "reranked_rank"],
        na_position="last",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(OUTPUT_PATH, index=False)

    print(f"Unique pooled trials: {len(review)}")
    print(f"Saved review file: {OUTPUT_PATH}")
    print("\nFill relevance_label manually:")
    print("1 = Relevant to the complete query")
    print("0 = Not relevant to the complete query")
    print("Leave blank = Not yet judged")
    print("\nCandidate list:")
    print(
        review[
            [
                "nct_id",
                "original_rrf_rank",
                "reranked_rank",
                "title",
                "relevance_label",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
