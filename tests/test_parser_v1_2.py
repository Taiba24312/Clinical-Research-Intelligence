"""Tests for parser v1.2 (age groups) and MetadataFilter max_age."""

import json
import re
import unittest
from pathlib import Path

import pandas as pd

from src.evaluation.retrieval_metrics import constraint_violations
from src.query_understanding.parser_v1_1 import parse_query_v1_1
from src.query_understanding.parser_v1_2 import parse_query_v1_2
from src.retrieval.metadata_filter import MetadataFilter


ROOT = Path(__file__).resolve().parents[1]

# (own phrasing, expected filters, words that must not remain in semantic_query)
CASES = [
    ("Paediatric asthma research", {"max_age": 18}, ["paediatric"]),
    ("Migraine studies in kids", {"max_age": 18}, ["kids"]),
    ("Children with sickle cell anemia", {"max_age": 18}, ["children"]),
    ("Teenagers with depression", {"min_age": 12, "max_age": 18}, ["teenagers"]),
    ("Adolescent scoliosis research", {"min_age": 12, "max_age": 18}, ["adolescent"]),
    ("Infants with RSV bronchiolitis", {"max_age": 2}, ["infants"]),
    ("Elderly people with insomnia", {"min_age": 65}, ["elderly"]),
    ("Older adults with glaucoma", {"min_age": 65}, ["older", "adults"]),
    ("Osteoporosis research for people aged 70 and above", {"min_age": 70}, ["70"]),
    ("COPD studies in patients over 60", {"min_age": 60}, ["60"]),
    ("Asthma research in patients under 12", {"max_age": 12}, ["12"]),
    ("Asthma studies including both children and adults", {}, ["children", "adults"]),
    ("Recruiting gout research for adults", {"status": ["RECRUITING"], "min_age": 18}, ["adults"]),
    ("Gout research with over 100 participants", {}, []),
]


def ngrams(text, n=4):
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class TestParserV12(unittest.TestCase):

    def test_age_filters_and_semantic_query(self):
        for text, expected, dropped in CASES:
            with self.subTest(text=text):
                parsed = parse_query_v1_2(text)
                self.assertEqual(parsed["filters"], expected)
                words = set(re.findall(r"[a-z0-9]+", parsed["semantic_query"]))
                for word in dropped:
                    self.assertNotIn(word, words)

    def test_v1_1_unchanged(self):
        self.assertEqual(parse_query_v1_1("Paediatric asthma research")["filters"], {})

    def test_phrasings_independent_of_query_specs(self):
        texts = [q["text"] for path in ("query_specs_v1.json", "query_specs_v1.1.json")
                 for q in json.loads((ROOT / "data/evaluation" / path).read_text())]
        self.assertEqual([(c[0], t) for c in CASES for t in texts if ngrams(c[0]) & ngrams(t)], [])


@unittest.skipUnless((ROOT / "data/processed/trials.parquet").exists(), "corpus not available")
class TestAgeFilterMatchesEvaluationChecker(unittest.TestCase):
    """The retrieval filter and the evaluation checker must select the same trials."""

    def test_agreement_on_full_corpus(self):
        trials = pd.read_parquet(ROOT / "data/processed/trials.parquet")
        metadata_filter = MetadataFilter(trials)
        records = trials.to_dict("index")
        for filters, age_range in [({"max_age": 18}, [0, 18]), ({"min_age": 12, "max_age": 18}, [12, 18]),
                                   ({"max_age": 2}, [0, 2]), ({"min_age": 65}, [65, None])]:
            with self.subTest(filters=filters):
                expected = [i for i, r in records.items()
                            if not constraint_violations(r, {"age_range": age_range})]
                self.assertEqual(metadata_filter.apply(filters), expected)


if __name__ == "__main__":
    unittest.main()
