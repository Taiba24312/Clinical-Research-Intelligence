from pathlib import Path

import pandas as pd

from src.retrieval.metadata_filter import MetadataFilter
from src.query_understanding.parser import parse_query


DATA_PATH = Path("data/processed/trials.parquet")
QUERIES_PATH = Path("data/evaluation/queries.json")


def main():
    trials = pd.read_parquet(DATA_PATH)

    print("=" * 80)
    print("METADATA FILTER TEST")
    print("=" * 80)

    print(f"\nCorpus size: {len(trials)}")

    metadata_filter = MetadataFilter(trials)

    # -------------------------
    # Test each evaluation query
    # -------------------------
    import json

    with QUERIES_PATH.open() as f:
        queries = json.load(f)

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]

        parsed = parse_query(query)

        filters = parsed["filters"]

        positions = metadata_filter.apply(filters)

        filtered = trials.iloc[positions]

        print("\n" + "-" * 80)
        print(query_id)
        print(f"Query   : {query}")
        print(f"Filters : {filters}")
        print(f"Matches : {len(filtered)}")

        if len(filtered) > 0:
            print(
                filtered[
                    [
                        "nct_id",
                        "status",
                        "phase",
                        "sex",
                        "study_type",
                        "min_age",
                        "max_age",
                    ]
                ]
                .head(5)
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()