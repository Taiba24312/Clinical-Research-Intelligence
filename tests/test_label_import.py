"""
Unit tests for src/evaluation/label_import.py.

Run from the project root:
    python -m unittest discover -s tests -v
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from src.evaluation.judgments import load_query_specs, parse_query_specs, prepare_qrels
from src.evaluation.label_import import import_labels, read_label_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HEADERS = ["row_id", "query_id", "NCT ID", "disease_match", "intervention_match",
           "mention_role", "notes", "title"]

SPECS = parse_query_specs([
    {"query_id": "QA", "text": "drug in disease", "split": "dev", "has_answer": True,
     "core_concepts": {"disease": {"concept": "d"}, "intervention": {"concept": "i"}},
     "structured_constraints": {}},
    {"query_id": "QB", "text": "disease only", "split": "dev", "has_answer": True,
     "core_concepts": {"disease": {"concept": "d"}, "intervention": None},
     "structured_constraints": {}},
])


def template(rows):
    """rows: (row_id, query_id, nct_id) -> template with not_applicable prefilled."""
    out = []
    for row_id, query_id, nct_id in rows:
        has_drug = SPECS[query_id].has_intervention
        out.append({"row_id": row_id, "query_id": query_id, "nct_id": nct_id,
                    "disease_match": "", "intervention_match": "" if has_drug else "not_applicable",
                    "mention_role": "", "notes": "", "labeler": "", "label_date": ""})
    return pd.DataFrame(out)


def write_sheet(path, rows):
    """rows: dicts with HEADERS keys (missing keys -> blank)."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Labels"
    sheet.append(HEADERS)
    for row in rows:
        sheet.append([row.get(h, None) for h in HEADERS])
    workbook.create_sheet("Instructions")
    workbook.save(path)


def row(row_id, query_id, nct_id, disease=None, intervention=None, role=None, notes=None):
    if intervention is None and not SPECS[query_id].has_intervention:
        intervention = "not_applicable"
    return {"row_id": row_id, "query_id": query_id, "NCT ID": nct_id,
            "disease_match": disease, "intervention_match": intervention,
            "mention_role": role, "notes": notes, "title": "t"}


class TestImportLabels(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sheet.xlsx"
        self.template = template([("QA-001", "QA", "N1"), ("QA-002", "QA", "N2"),
                                  ("QA-003", "QA", "N3"), ("QB-001", "QB", "N4")])

    def tearDown(self):
        self.tmp.cleanup()

    def run_import(self, rows, tmpl=None):
        write_sheet(self.path, rows)
        return import_labels(read_label_sheet(self.path),
                             self.template if tmpl is None else tmpl, SPECS, "AJ", "2026-10-31")

    def test_valid_labels_normalized_and_graded(self):
        qrels, report = self.run_import([
            row("QA-001", "QA", "N1", "exact", "Class Level"),      # typed with spaces/caps
            row("QA-002", "QA", "N2", " exact ", "exact", notes="UNCERTAIN: dose unclear"),
            row("QA-003", "QA", "N3", "exact", "absent_or_different", "prior_therapy"),
            row("QB-001", "QB", "N4", "broader"),
        ])
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(len(qrels), 4)
        self.assertEqual(qrels.loc[qrels.row_id == "QA-001", "intervention_match"].item(), "class_level")
        self.assertEqual(set(qrels["labeler"]), {"AJ"})
        self.assertEqual(set(qrels["label_date"]), {"2026-10-31"})
        self.assertEqual([u["row_id"] for u in report.uncertain], ["QA-002"])
        self.assertEqual(report.per_query["QA"], {"rows": 3, "judged": 3,
                                                  "grade_2": 1, "grade_1": 1, "grade_0": 1})
        prepared = prepare_qrels(qrels, SPECS)              # accepted by the Step 2 loader
        self.assertEqual(prepared.attrs["pending_rows"], 0)

    def test_pending_and_partial_rows(self):
        qrels, report = self.run_import([
            row("QA-001", "QA", "N1"),                          # pending
            row("QA-002", "QA", "N2", "exact"),                 # partial: drug blank
            row("QA-003", "QA", "N3", notes="UNCERTAIN: later"),  # partial: notes only
            row("QB-001", "QB", "N4", "exact"),
        ])
        self.assertTrue(report.ok, report.errors)
        self.assertEqual(qrels["row_id"].tolist(), ["QB-001"])
        self.assertEqual(report.partial, ["QA-002", "QA-003"])

    def test_invalid_and_locked_values_are_errors(self):
        _, report = self.run_import([
            row("QA-001", "QA", "N1", "maybe", "exact"),
            row("QA-002", "QA", "N2", "exact", "not_applicable"),
            row("QA-003", "QA", "N3", "exact", "exact", role="sometimes"),
            row("QB-001", "QB", "N4", "exact", intervention="exact"),   # locked cell changed
        ])
        self.assertEqual(sorted(e["row_id"] for e in report.errors),
                         ["QA-001", "QA-002", "QA-003", "QB-001"])

    def test_inconsistent_mention_role_is_error(self):
        _, report = self.run_import([
            row("QA-001", "QA", "N1", "exact", "exact", "prior_therapy"),
            row("QA-002", "QA", "N2"), row("QA-003", "QA", "N3"), row("QB-001", "QB", "N4"),
        ])
        self.assertEqual([e["row_id"] for e in report.errors], ["QA-001"])
        self.assertIn("absent_or_different", report.errors[0]["error"])

    def test_template_mismatch_is_error(self):
        _, report = self.run_import([
            row("QA-001", "QA", "N9", "exact", "exact"),           # NCT changed
            row("QA-002", "QA", "N2"),
            row("QB-001", "QB", "N4"),
            row("QX-001", "QA", "N7"),                             # not in template
        ])                                                         # QA-003 missing
        errors = {e["row_id"]: e["error"] for e in report.errors}
        self.assertIn("key changed", errors["QA-001"])
        self.assertEqual(errors["QA-003"], "row missing from sheet")
        self.assertEqual(errors["QX-001"], "row not in template")

    def test_warning_when_complete_answerable_query_has_no_relevant(self):
        _, report = self.run_import([
            row("QA-001", "QA", "N1"), row("QA-002", "QA", "N2"), row("QA-003", "QA", "N3"),
            row("QB-001", "QB", "N4", "absent_or_different"),
        ])
        self.assertTrue(any(w.startswith("QB") for w in report.warnings))
        self.assertFalse(any(w.startswith("QA") for w in report.warnings))   # QA incomplete

    def test_no_warning_for_query_reusing_judgments(self):
        items = [
            {"query_id": "QB", "text": "disease only", "split": "dev", "has_answer": True,
             "core_concepts": {"disease": {"concept": "d"}, "intervention": None},
             "structured_constraints": {}},
            {"query_id": "QC", "text": "paraphrase", "split": "dev", "has_answer": True,
             "core_concepts": {"disease": {"concept": "d"}, "intervention": None},
             "structured_constraints": {}, "judgments_shared_with": "QB"},
        ]
        specs = parse_query_specs(items)
        tmpl = pd.DataFrame([{"row_id": "QC-001", "query_id": "QC", "nct_id": "N5",
                              "disease_match": "", "intervention_match": "not_applicable",
                              "mention_role": "", "notes": "", "labeler": "", "label_date": ""}])
        write_sheet(self.path, [{"row_id": "QC-001", "query_id": "QC", "NCT ID": "N5",
                                 "disease_match": "absent_or_different",
                                 "intervention_match": "not_applicable"}])
        _, report = import_labels(read_label_sheet(self.path), tmpl, specs, "AJ", "2026-11-02")
        self.assertEqual(report.warnings, [])


SHEET_V11 = PROJECT_ROOT / "data/evaluation/labeling_sheet_v1.1.xlsx"
TEMPLATE_V11 = PROJECT_ROOT / "data/evaluation/qrels_v1.1_template.csv"
SPECS_V11 = PROJECT_ROOT / "data/evaluation/query_specs_v1.1.json"


@unittest.skipUnless(SHEET_V11.exists() and TEMPLATE_V11.exists(), "v1.1 files not available")
class TestRealBlankSheet(unittest.TestCase):

    def test_blank_v11_sheet_imports_as_all_pending(self):
        template_frame = pd.read_csv(TEMPLATE_V11, dtype=str, keep_default_na=False)
        qrels, report = import_labels(read_label_sheet(SHEET_V11), template_frame,
                                      load_query_specs(SPECS_V11), "AJ", "2026-10-31")
        self.assertTrue(report.ok, report.errors[:5])
        self.assertEqual(len(qrels), 0)
        self.assertEqual(report.partial, [])
        self.assertEqual(sum(q["rows"] for q in report.per_query.values()), 474)


if __name__ == "__main__":
    unittest.main()
