"""Tests for BM25 tokenizer v2 and trial document v2."""

import unittest

import pandas as pd

from src.preprocessing.trial_representation import add_trial_documents, build_trial_document
from src.retrieval.bm25 import BM25Retriever, tokenize, tokenize_v2


class TestTokenizerV2(unittest.TestCase):

    def test_splits_on_punctuation_and_symbols(self):
        self.assertEqual(tokenize_v2("KEYTRUDA® (MK-3475),"), ["keytruda", "mk", "3475"])
        self.assertEqual(tokenize_v2("Pemetrexed+Pembrolizumab/Cemiplimab"),
                         ["pemetrexed", "pembrolizumab", "cemiplimab"])
        self.assertEqual(tokenize_v2("(NSCLC)."), ["nsclc"])

    def test_drops_filler_words_only(self):
        self.assertEqual(tokenize_v2("Trials of insulin for the treatment of diabetes"),
                         ["insulin", "treatment", "diabetes"])

    def test_old_tokenizer_unchanged(self):
        self.assertEqual(tokenize("KEYTRUDA® (MK-3475),"), ["keytruda®", "(mk-3475),"])

    def test_brand_name_matches_only_with_v2(self):
        # Three documents: with two, a term in one of them has BM25 IDF log(1.5/1.5) = 0.
        docs = ["Interventions: KEYTRUDA® in lung cancer", "Interventions: metformin in diabetes",
                "Interventions: aspirin in stroke"]
        v1, v2 = BM25Retriever(docs), BM25Retriever(docs, tokenizer="v2")
        self.assertEqual(v1.score("keytruda").tolist(), [0.0, 0.0, 0.0])
        self.assertGreater(v2.score("keytruda")[0], 0.0)


class TestDocumentV2(unittest.TestCase):

    def row(self, **extra):
        base = {"title": "T", "conditions": "C", "interventions": "I", "summary": "S",
                "eligibility": "E", "official_title": "", "intervention_other_names": "",
                "condition_mesh": "", "intervention_mesh": "", "keywords": ""}
        base.update(extra)
        return pd.DataFrame([base])

    def test_new_fields_before_summary_and_empty_fields_omitted(self):
        doc = add_trial_documents(self.row(intervention_other_names="Keytruda; MK-3475"),
                                  version="v2")["document"][0]
        self.assertIn("INTERVENTION OTHER NAMES: Keytruda; MK-3475", doc)
        self.assertLess(doc.index("OTHER NAMES"), doc.index("SUMMARY"))
        self.assertNotIn("OFFICIAL TITLE", doc)

    def test_v1_document_unchanged(self):
        frame = self.row(intervention_other_names="Keytruda")
        self.assertEqual(add_trial_documents(frame)["document"][0], build_trial_document(frame.iloc[0]))
        self.assertNotIn("Keytruda", add_trial_documents(frame)["document"][0])


if __name__ == "__main__":
    unittest.main()
