from pathlib import Path
import json

from src.query_understanding.parser import parse_query


QUERIES_PATH = Path("data/evaluation/queries.json")


def main():
    with QUERIES_PATH.open() as f:
        queries = json.load(f)

    total = 0
    correct = 0

    field_total = {}
    field_correct = {}

    print("=" * 80)
    print("QUERY UNDERSTANDING EVALUATION")
    print("=" * 80)

    for item in queries:
        query_id = item["query_id"]
        query = item["query"]
        expected = item["filters"]

        result = parse_query(query)

        predicted = result["filters"]

        print(f"\n{query_id}")
        print(f"Query     : {query}")
        print(f"Semantic  : {result['semantic_query']}")
        print(f"Expected  : {expected}")
        print(f"Predicted : {predicted}")

        all_fields = set(expected) | set(predicted)

        for field in all_fields:
            field_total[field] = field_total.get(field, 0) + 1

            if expected.get(field) == predicted.get(field):
                field_correct[field] = (
                    field_correct.get(field, 0) + 1
                )

        if expected == predicted:
            correct += 1

        total += 1

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nExact query-level accuracy: {correct}/{total} "
          f"= {correct / total:.2%}")

    print("\nField-level accuracy:")

    for field in sorted(field_total):
        accuracy = (
            field_correct.get(field, 0)
            / field_total[field]
        )

        print(
            f"  {field:15s}: "
            f"{field_correct.get(field, 0)}/{field_total[field]}"
            f" = {accuracy:.2%}"
        )


if __name__ == "__main__":
    main()
    