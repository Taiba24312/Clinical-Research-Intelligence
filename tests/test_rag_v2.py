"""Tests for the deterministic parts of RAG v2 (no model needed)."""

import unittest

from src.generation.rag_v2 import PROMPTS, derive_grade, extract_json, quote_found


class TestQuoteCheck(unittest.TestCase):
    TEXT = "Title: Pembrolizumab in “advanced” NSCLC\nSummary: Patients   receive KEYTRUDA® every 3 weeks."

    def test_found_despite_case_quotes_and_spacing(self):
        self.assertTrue(quote_found("pembrolizumab in \"advanced\" nsclc", self.TEXT))
        self.assertTrue(quote_found("Patients receive KEYTRUDA® every 3 weeks", self.TEXT))

    def test_invented_or_too_short_quote_rejected(self):
        self.assertFalse(quote_found("Patients receive nivolumab every 3 weeks", self.TEXT))
        self.assertFalse(quote_found("NSCLC", self.TEXT))
        self.assertFalse(quote_found("", self.TEXT))


class TestParsingAndGrades(unittest.TestCase):

    def test_extract_json(self):
        self.assertEqual(extract_json('Sure: {"disease_match": "exact", "evidence": "x"} done'),
                         {"disease_match": "exact", "evidence": "x"})
        self.assertIsNone(extract_json("no json here"))
        self.assertIsNone(extract_json("{not valid json}"))

    def test_grade_rule_matches_rubric(self):
        self.assertEqual(derive_grade("exact", "exact"), 2)
        self.assertEqual(derive_grade("exact", "not_applicable"), 2)
        self.assertEqual(derive_grade("broader", "exact"), 1)
        self.assertEqual(derive_grade("exact", "class_level"), 1)
        self.assertEqual(derive_grade("exact", "absent"), 0)
        self.assertEqual(derive_grade("not_applicable", "not_applicable"), 0)

    def test_every_prompt_version_complete(self):
        for version, prompts in PROMPTS.items():
            self.assertEqual(set(prompts), {"concepts", "verify", "answer"}, version)


if __name__ == "__main__":
    unittest.main()
