"""
Tests for the structured-constraint parser v2 (M2 Task 1).

Committed BEFORE the parser is implemented. Tests that need
src/query_understanding/parser_v2.py are skipped until it exists; the
independence and old-parser checks run now.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import json
import re
import unittest
from pathlib import Path

from src.query_understanding.parser import parse_query

try:
    from src.query_understanding.parser_v2 import parse_query_v2
except ImportError:          # not implemented yet
    parse_query_v2 = None


ROOT = Path(__file__).resolve().parents[1]
PHRASINGS = json.loads((ROOT / "tests/data/parser_phrasings_v1.json").read_text())["phrasings"]
GOLDEN_OLD = ROOT / "tests/data/parser_v1_golden_outputs.json"
SPEC_FILES = [ROOT / "data/evaluation/query_specs_v1.json",
              ROOT / "data/evaluation/query_specs_v1.1.json"]

CONSTRAINT_KEYS = {"status", "phase", "sex", "study_type", "age_range"}
LIST_KEYS = {"status", "phase", "sex", "study_type"}


def ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def spec_texts() -> dict[str, str]:
    texts = {}
    for path in SPEC_FILES:
        for item in json.loads(path.read_text()):
            texts[item["query_id"]] = item["text"]
    return texts


class TestPhrasingList(unittest.TestCase):

    def test_ids_unique_and_schema_valid(self):
        ids = [p["id"] for p in PHRASINGS]
        self.assertEqual(len(ids), len(set(ids)))
        for p in PHRASINGS:
            with self.subTest(id=p["id"]):
                self.assertLessEqual(set(p["expect"]["constraints"]), CONSTRAINT_KEYS)
                self.assertLessEqual(set(p["expect"].get("unresolved", [])), CONSTRAINT_KEYS)

    def test_phrasings_are_independent_of_query_specs(self):
        """No phrasing shares a 4-word sequence with any evaluation query."""
        queries = spec_texts()
        overlaps = [
            (p["id"], query_id, " ".join(sorted(ngrams(p["text"]) & ngrams(text))[0]))
            for p in PHRASINGS for query_id, text in queries.items()
            if ngrams(p["text"]) & ngrams(text)
        ]
        self.assertEqual(overlaps, [])


class TestOldParserFrozen(unittest.TestCase):
    """The old parser must keep producing exactly its recorded outputs."""

    def test_old_parser_outputs_unchanged(self):
        golden = json.loads(GOLDEN_OLD.read_text())
        for text, expected in golden["outputs"].items():
            with self.subTest(text=text):
                self.assertEqual(parse_query(text), expected)


def as_comparable(constraints: dict) -> dict:
    return {k: (set(v) if k in LIST_KEYS else list(v)) for k, v in constraints.items()}


@unittest.skipIf(parse_query_v2 is None, "parser_v2 not implemented yet")
class TestParserV2(unittest.TestCase):

    def test_constraints(self):
        for p in PHRASINGS:
            with self.subTest(id=p["id"], text=p["text"]):
                parsed = parse_query_v2(p["text"])
                got = as_comparable(parsed.constraints)
                if "status_all_except" in p:
                    status = got.pop("status", set())
                    for excluded in p["status_all_except"]:
                        self.assertNotIn(excluded, status)
                    self.assertLessEqual({"RECRUITING", "NOT_YET_RECRUITING",
                                          "ACTIVE_NOT_RECRUITING", "TERMINATED"}, status)
                self.assertEqual(got, as_comparable(p["expect"]["constraints"]))

    def test_unresolved_reported_with_matched_text(self):
        for p in PHRASINGS:
            with self.subTest(id=p["id"], text=p["text"]):
                parsed = parse_query_v2(p["text"])
                types = {item["constraint"] for item in parsed.unresolved}
                self.assertEqual(types, set(p["expect"].get("unresolved", [])))
                for item in parsed.unresolved:
                    self.assertTrue(item["text"])
                    self.assertIn(item["text"].lower(), p["text"].lower())
                    self.assertNotIn(item["constraint"], parsed.constraints)

    def test_secondary_status_section(self):
        for p in PHRASINGS:
            with self.subTest(id=p["id"], text=p["text"]):
                parsed = parse_query_v2(p["text"])
                self.assertEqual(list(parsed.secondary_status),
                                 p["expect"].get("secondary_status", []))
                if parsed.secondary_status:
                    self.assertEqual(parsed.constraints.get("status"), ["RECRUITING"])

    def test_semantic_query_keeps_concepts_and_drops_cues(self):
        for p in PHRASINGS:
            with self.subTest(id=p["id"], text=p["text"]):
                words = set(re.findall(r"[a-z0-9]+", parse_query_v2(p["text"]).semantic_query.lower()))
                for word in p.get("semantic_keeps", []):
                    self.assertIn(word, words)
                for word in p.get("semantic_drops", []):
                    self.assertNotIn(word, words)

    def test_deterministic(self):
        for p in PHRASINGS:
            self.assertEqual(parse_query_v2(p["text"]), parse_query_v2(p["text"]))


if __name__ == "__main__":
    unittest.main()
