"""
Import a filled labeling sheet (.xlsx) into the qrels format.

The sheet is checked against the qrels template it was built from
(data/evaluation/qrels_v1.1_template.csv): every row_id must still be
present with the same query_id and nct_id, and pre-filled not_applicable
cells must be unchanged. All problems are collected in one pass so the
labeler can fix them together.

Row states:
    judged   every applicable concept field has a valid value
    pending  no label typed yet (blank) - allowed, reported
    partial  some applicable fields (or only mention_role / notes) filled,
             others blank - reported, skipped
    error    invalid value, inconsistent mention_role, or a template mismatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.evaluation.judgments import (
    DISEASE_VALUES,
    INTERVENTION_VALUES,
    MENTION_ROLES,
    QRELS_COLUMNS,
    QuerySpec,
    derive_grade,
)


LABEL_FIELDS = ("disease_match", "intervention_match", "mention_role", "notes")
KEY_FIELDS = ("row_id", "query_id", "NCT ID")
ALLOWED = {
    "disease_match": set(DISEASE_VALUES),
    "intervention_match": set(INTERVENTION_VALUES),
    "mention_role": set(MENTION_ROLES),
}


@dataclass
class ImportReport:
    errors: list[dict] = field(default_factory=list)
    partial: list[str] = field(default_factory=list)
    uncertain: list[dict] = field(default_factory=list)
    per_query: dict[str, dict] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_errors": len(self.errors),
            "errors": self.errors,
            "partial_rows": self.partial,
            "uncertain_rows": self.uncertain,
            "warnings": self.warnings,
            "per_query": self.per_query,
        }


def read_label_sheet(path: str | Path, sheet_name: str = "Labels") -> pd.DataFrame:
    """Read the key and label columns of a labeling sheet as strings."""

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook[sheet_name].iter_rows(values_only=True)
        headers = [str(h).strip() if h is not None else "" for h in next(rows)]
        missing = [h for h in KEY_FIELDS + LABEL_FIELDS if h not in headers]
        if missing:
            raise ValueError(f"Labeling sheet is missing columns: {missing}")

        index = {h: headers.index(h) for h in KEY_FIELDS + LABEL_FIELDS}
        records = []
        for values in rows:
            if all(v is None for v in values):
                continue
            records.append({h: "" if values[i] is None else str(values[i]).strip()
                            for h, i in index.items()})
    finally:
        workbook.close()
    return pd.DataFrame(records, columns=list(index)).rename(columns={"NCT ID": "nct_id"})


def _normalize(value: str, column: str) -> str:
    """Accept dropdown values typed with stray case/whitespace; keep notes as-is."""
    if column == "notes":
        return value
    return value.strip().lower().replace(" ", "_")


def import_labels(
    sheet: pd.DataFrame,
    template: pd.DataFrame,
    specs: dict[str, QuerySpec],
    labeler: str,
    label_date: str,
) -> tuple[pd.DataFrame, ImportReport]:
    """
    Validate sheet labels against the template and return
    (judged qrels in QRELS_COLUMNS + row_id, report).
    """

    report = ImportReport()
    template = template.fillna("").astype(str)

    # ---- template integrity -------------------------------------------
    sheet_ids, template_ids = set(sheet["row_id"]), set(template["row_id"])
    for row_id in sorted(template_ids - sheet_ids):
        report.errors.append({"row_id": row_id, "error": "row missing from sheet"})
    for row_id in sorted(sheet_ids - template_ids):
        report.errors.append({"row_id": row_id, "error": "row not in template"})
    for row_id in sheet.loc[sheet["row_id"].duplicated(), "row_id"]:
        report.errors.append({"row_id": row_id, "error": "duplicate row_id in sheet"})

    expected = template.set_index("row_id")
    judged_rows = []

    for row in sheet.drop_duplicates("row_id").itertuples(index=False):
        if row.row_id not in expected.index:
            continue
        ref = expected.loc[row.row_id]
        if (row.query_id, row.nct_id) != (ref["query_id"], ref["nct_id"]):
            report.errors.append({
                "row_id": row.row_id,
                "error": f"key changed: sheet {row.query_id}/{row.nct_id}, "
                         f"template {ref['query_id']}/{ref['nct_id']}",
            })
            continue

        values = {c: _normalize(getattr(row, c), c) for c in LABEL_FIELDS}

        bad = False
        for column in ("disease_match", "intervention_match"):
            prefilled = ref[column] == "not_applicable"
            if prefilled and values[column] != "not_applicable":
                report.errors.append({"row_id": row.row_id,
                                      "error": f"{column}: locked not_applicable was changed "
                                               f"to {values[column]!r}"})
                bad = True
            elif not prefilled and values[column] == "not_applicable":
                report.errors.append({"row_id": row.row_id,
                                      "error": f"{column}: not_applicable is not allowed here"})
                bad = True
            elif values[column] and values[column] not in ALLOWED[column]:
                report.errors.append({"row_id": row.row_id,
                                      "error": f"{column}: invalid value {getattr(row, column)!r}"})
                bad = True
        if values["mention_role"] not in ALLOWED["mention_role"]:
            report.errors.append({"row_id": row.row_id,
                                  "error": f"mention_role: invalid value {row.mention_role!r}"})
            bad = True
        if bad:
            continue

        applicable = [c for c in ("disease_match", "intervention_match")
                      if ref[c] != "not_applicable"]
        filled = [c for c in applicable if values[c]]
        if not filled and not values["mention_role"] and not values["notes"]:
            continue                                   # pending
        if len(filled) < len(applicable):
            report.partial.append(row.row_id)
            continue

        try:
            grade = derive_grade(values["disease_match"], values["intervention_match"],
                                 values["mention_role"])
        except ValueError as error:
            report.errors.append({"row_id": row.row_id, "error": str(error)})
            continue

        if values["notes"].upper().startswith("UNCERTAIN"):
            report.uncertain.append({"row_id": row.row_id, "query_id": row.query_id,
                                     "nct_id": row.nct_id, "notes": values["notes"]})

        judged_rows.append({
            "row_id": row.row_id, "query_id": row.query_id, "nct_id": row.nct_id,
            "disease_match": values["disease_match"],
            "intervention_match": values["intervention_match"],
            "mention_role": values["mention_role"], "notes": values["notes"],
            "labeler": labeler, "label_date": label_date, "grade": grade,
        })

    qrels = pd.DataFrame(judged_rows, columns=["row_id"] + QRELS_COLUMNS + ["grade"])

    # ---- per-query summary --------------------------------------------
    for query_id, group in template.groupby("query_id"):
        judged = qrels[qrels["query_id"] == query_id]
        grades = judged["grade"].value_counts().to_dict()
        report.per_query[query_id] = {
            "rows": len(group),
            "judged": len(judged),
            "grade_2": int(grades.get(2, 0)),
            "grade_1": int(grades.get(1, 0)),
            "grade_0": int(grades.get(0, 0)),
        }
        complete = len(judged) == len(group)
        if complete and specs[query_id].has_answer and grades.get(2, 0) + grades.get(1, 0) == 0:
            report.warnings.append(
                f"{query_id} is marked has_answer=true but no pooled trial was judged "
                "grade >= 1; check the labels or the has_answer flag.")

    return qrels.drop(columns=["grade"]), report
