"""BM25 baseline experiment over the complete ClinicalTrials.gov corpus.

This experiment:
1. Loads the processed 4,695-trial corpus.
2. Builds BM25 over searchable trial documents.
3. Runs Q01.
4. Reports top-ranked trials.
5. Reports Q01 sanity-check retrieval metrics.
6. Measures BM25 scoring latency.

IMPORTANT:
The Q01 relevance labels currently contain only two known relevant trials.
Therefore, the reported metrics are sanity-check metrics, NOT final benchmark
metrics for the project.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents
from src.retrieval.bm25 import BM25Retriever, tokenize


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Currently known relevant trials for Q01.
# These are initial/manual labels, not the final pooled evaluation set.
KNOWN_RELEVANT_IDS = {
    "NCT07061379",
    "NCT06195527",
}


def recall_at_k(ranked_ids, relevant_ids, k):
    """Fraction of known relevant documents retrieved in top-k."""
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(ranked_ids, relevant_ids, k):
    """Fraction of top-k retrieved documents that are relevant."""
    retrieved = ranked_ids[:k]
    return sum(nct_id in relevant_ids for nct_id in retrieved) / k


def reciprocal_rank(ranked_ids, relevant_ids):
    """Reciprocal rank of the first relevant document."""
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

    # Sanity checks
    if len(df) != 4695:
        raise ValueError(
            f"Expected 4,695 unique trials, but found {len(df)}."
        )

    if df["nct_id"].duplicated().any():
        raise ValueError("Duplicate NCT IDs found in the corpus.")

    # ---------------------------------------------------------
    # 2. Create searchable trial documents
    # ---------------------------------------------------------
    #
    # Searchable fields:
    #   TITLE
    #   CONDITIONS
    #   INTERVENTIONS
    #   SUMMARY
    #   ELIGIBILITY
    #
    # Structured metadata such as status, phase, age etc.
    # remains outside BM25.
    #
    df = add_trial_documents(df).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Load Q01
    # ---------------------------------------------------------
    evaluation_path = PROJECT_ROOT / "data/evaluation/queries.json"

    with evaluation_path.open(encoding="utf-8") as handle:
        queries = json.load(handle)

    query = next(
        item
        for item in queries
        if item["query_id"] == "Q01"
    )

    query_text = query["query"]

    # ---------------------------------------------------------
    # 4. Build BM25 index
    # ---------------------------------------------------------
    print("=" * 80)
    print("BM25 BASELINE EXPERIMENT")
    print("=" * 80)

    print("\nQuery:")
    print(query_text)

    print("\nCorpus size:")
    print(len(df))

    print("\nBuilding BM25 index...")

    index_start = perf_counter()

    retriever = BM25Retriever(
        df["document"].tolist()
    )

    index_time_ms = (
        perf_counter() - index_start
    ) * 1000

    print(
        f"Index creation time: {index_time_ms:.3f} ms"
    )

    print(
        f"Vocabulary size: {retriever.vocabulary_size}"
    )

    print(
        f"Empty tokenized documents: "
        f"{retriever.empty_document_count}"
    )

    # ---------------------------------------------------------
    # 5. Inspect query tokenization
    # ---------------------------------------------------------
    query_tokens = tokenize(query_text)

    print("\nQuery tokens:")
    print(query_tokens)

    oov_tokens = [
        token
        for token in query_tokens
        if token not in retriever.index.idf
    ]

    print("\nOut-of-vocabulary query tokens:")

    if oov_tokens:
        print(oov_tokens)
    else:
        print("None")

    # ---------------------------------------------------------
    # 6. Score complete corpus
    # ---------------------------------------------------------
    #
    # BM25 scoring itself is CPU-based.
    # This is expected; GPU will become useful for Dense
    # Retrieval and Cross-Encoder reranking.
    #
    print("\nScoring complete corpus...")

    start = perf_counter()

    scores = retriever.score(query_text)

    latency_ms = (
        perf_counter() - start
    ) * 1000

    # Validate scores
    if len(scores) != len(df):
        raise ValueError(
            "Number of BM25 scores does not match corpus size."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "BM25 produced invalid scores."
        )

    print(
        f"BM25 scoring latency: {latency_ms:.3f} ms"
    )

    # ---------------------------------------------------------
    # 7. Rank entire corpus
    # ---------------------------------------------------------
    #
    # We rank the complete corpus so that Recall@K and MRR
    # can be calculated.
    #
    ranked_results = retriever.search(
        query_text,
        len(df)
    )

    ranked_ids = [
        df.iloc[position]["nct_id"]
        for position, _ in ranked_results
    ]

    # ---------------------------------------------------------
    # 8. Display Top-10
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("TOP 10 BM25 RESULTS")
    print("(Full corpus; NO metadata filtering)")
    print("=" * 80)

    for rank, (position, score) in enumerate(
        ranked_results[:10],
        start=1
    ):
        row = df.iloc[position]

        print(
            f"{rank:2d} | "
            f"{row['nct_id']} | "
            f"{score:.6f} | "
            f"{row['title']} | "
            f"{row['status']}"
        )

    # ---------------------------------------------------------
    # 9. Known relevant trial scores
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("KNOWN Q01 RELEVANT TRIAL SCORES")
    print("=" * 80)

    for nct_id in sorted(KNOWN_RELEVANT_IDS):

        positions = df.index[
            df["nct_id"].eq(nct_id)
        ]

        if len(positions) != 1:
            raise ValueError(
                f"Known relevant trial missing or duplicated: "
                f"{nct_id}"
            )

        position = positions[0]

        score = scores[position]

        rank = ranked_ids.index(nct_id) + 1

        print(
            f"{nct_id} | "
            f"score={score:.6f} | "
            f"rank={rank}"
        )

    # ---------------------------------------------------------
    # 10. Q01 sanity-check metrics
    # ---------------------------------------------------------
    relevant_ids = KNOWN_RELEVANT_IDS

    recall_5 = recall_at_k(
        ranked_ids,
        relevant_ids,
        5
    )

    recall_10 = recall_at_k(
        ranked_ids,
        relevant_ids,
        10
    )

    precision_5 = precision_at_k(
        ranked_ids,
        relevant_ids,
        5
    )

    mrr = reciprocal_rank(
        ranked_ids,
        relevant_ids
    )

    print("\n" + "=" * 80)
    print("Q01 BM25 SANITY-CHECK METRICS")
    print("=" * 80)

    print(
        f"Recall@5:    {recall_5:.4f}"
    )

    print(
        f"Recall@10:   {recall_10:.4f}"
    )

    print(
        f"Precision@5: {precision_5:.4f}"
    )

    print(
        f"MRR:         {mrr:.4f}"
    )

    print(
        f"Latency:     {latency_ms:.3f} ms"
    )

    # ---------------------------------------------------------
    # 11. Explain baseline limitations
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE OBSERVATIONS")
    print("=" * 80)

    print(
        "\n1. BM25 is lexical retrieval."
    )

    print(
        "   It rewards exact/near-exact token matches."
    )

    print(
        "\n2. BM25 does NOT understand structured constraints."
    )

    print(
        "   For example, 'Recruiting' is treated as a text token."
    )

    print(
        "   BM25 does not enforce status == RECRUITING."
    )

    print(
        "\n3. Structured metadata filters will therefore be"
        " handled separately in the final architecture."
    )

    print(
        "\n4. The current tokenizer is intentionally simple:"
    )

    print(
        "   lowercase + whitespace splitting."
    )

    print(
        "   No stopword removal or stemming."
    )

    print(
        "\n5. This creates medical-tokenization problems."
    )

    print(
        "   Example: 'Pembrolizumab/Cemiplimab' may remain"
        " one token instead of matching 'pembrolizumab'."
    )

    print(
        "\n6. BM25 may rank COMPLETED trials highly if they"
        " contain strong lexical matches."
    )

    print(
        "\n7. Dense retrieval will be added in Step 7 to"
        " capture semantic similarity and synonyms."
    )

    # ---------------------------------------------------------
    # 12. Important evaluation warning
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("EVALUATION WARNING")
    print("=" * 80)

    print(
        "\nThe current metrics use only the two known Q01 "
        "relevant trials:"
    )

    print(
        sorted(KNOWN_RELEVANT_IDS)
    )

    print(
        "\nThese are SANITY-CHECK metrics only."
    )

    print(
        "They are NOT the final project benchmark."
    )

    print(
        "\nFinal evaluation will use the roadmap's pooled "
        "relevance judgments across retrieval systems."
    )

    print(
        "That evaluation will be completed after BM25, "
        "Dense, Hybrid and Reranker candidates are available."
    )

    print("\n" + "=" * 80)
    print("BM25 EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()