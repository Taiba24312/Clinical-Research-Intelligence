"""
Unit tests for parser v1.1 (src/query_understanding/parser_v1_1.py) and the
MetadataFilter hours/minutes age-unit fix. Phrasings are written for these
tests and share no 4-word sequence with any evaluation query.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import json
import re
import unittest
from pathlib import Path

import pandas as pd

from src.query_understanding.parser import parse_query
from src.query_understanding.parser_v1_1 import parse_query_v1_1
from src.retrieval.metadata_filter import MetadataFilter


ROOT = Path(__file__).resolve().parents[1]

# (phrasing, expected filters, words that must not remain in semantic_query)
CASES = [
    ("Psoriasis research not yet recruiting", {"status": ["NOT_YET_RECRUITING"]}, ["recruiting", "yet"]),
    ("Not-yet-recruiting gout studies", {"status": ["NOT_YET_RECRUITING"]}, ["recruiting"]),
    ("Lupus studies currently enrolling", {"status": ["RECRUITING"]}, ["enrolling"]),
    ("Asthma research enrolling now", {"status": ["RECRUITING"]}, ["enrolling"]),
    ("Migraine studies not recruiting", {}, ["recruiting"]),
    ("Phase II asthma research", {"phase": ["PHASE2"]}, ["phase", "ii"]),
    ("Phase IV statin research", {"phase": ["PHASE4"]}, ["phase", "iv"]),
    ("phase iii psoriasis studies", {"phase": ["PHASE3"]}, ["iii"]),
    ("Phase I gout research", {"phase": ["PHASE1"]}, ["phase"]),
    ("Phase 4 glaucoma research", {"phase": ["PHASE4"]}, ["phase"]),
    ("Non-interventional psoriasis research", {"study_type": ["OBSERVATIONAL"]}, ["interventional"]),
    ("Noninterventional asthma research", {"study_type": ["OBSERVATIONAL"]}, ["noninterventional"]),
    ("Gout research excluding observational designs",
     {"study_type": ["INTERVENTIONAL", "EXPANDED_ACCESS"]}, ["observational"]),
    ("Non-observational lupus research", {"study_type": ["INTERVENTIONAL", "EXPANDED_ACCESS"]}, ["observational"]),
    ("Observational asthma research", {"study_type": ["OBSERVATIONAL"]}, ["observational"]),
    ("Recruiting gout research for adults", {"status": ["RECRUITING"], "min_age": 18}, ["recruiting", "adults"]),
]


def ngrams(text, n=4):
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("'", ""))
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


class TestParserV11(unittest.TestCase):

    def test_filters_and_semantic_query(self):
        for text, expected, dropped in CASES:
            with self.subTest(text=text):
                parsed = parse_query_v1_1(text)
                got = {k: (sorted(v) if isinstance(v, list) else v) for k, v in parsed["filters"].items()}
                want = {k: (sorted(v) if isinstance(v, list) else v) for k, v in expected.items()}
                self.assertEqual(got, want)
                words = set(re.findall(r"[a-z0-9]+", parsed["semantic_query"]))
                for word in dropped:
                    self.assertNotIn(word, words)

    def test_old_parser_still_has_old_behaviour(self):
        # v1 is kept unchanged for reproducibility (full snapshot in test_parser_v2.py).
        self.assertEqual(parse_query("Psoriasis research not yet recruiting")["filters"],
                         {"status": ["RECRUITING"]})

    def test_phrasings_independent_of_query_specs(self):
        texts = [q["text"] for path in ("query_specs_v1.json", "query_specs_v1.1.json")
                 for q in json.loads((ROOT / "data/evaluation" / path).read_text())]
        overlaps = [(case[0], t) for case in CASES for t in texts if ngrams(case[0]) & ngrams(t)]
        self.assertEqual(overlaps, [])


class TestAgeUnits(unittest.TestCase):

    def test_hours_and_minutes_are_converted(self):
        years = MetadataFilter._age_to_years(pd.Series(["6 Hours", "30 Minutes", "2 Years", "6 Months"]))
        self.assertAlmostEqual(years[0], 6 / 8766)
        self.assertAlmostEqual(years[1], 30 / 525960)
        self.assertAlmostEqual(years[2], 2.0)
        self.assertAlmostEqual(years[3], 0.5)


if __name__ == "__main__":
    unittest.main()
