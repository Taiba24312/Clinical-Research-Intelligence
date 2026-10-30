"""
M1 Step 4b - labeling pool and sheet v1.1.

Derived from v1 by removing the rows of queries that query_specs_v1.1.json
marks as zero-answer (Q16 after Experiment 23). Every remaining row is copied
unchanged from labeling_sheet_v1.xlsx, keeping its row_id and position, so
v1 and v1.1 rows correspond one-to-one (re-shuffling would change the order
and row_ids of later queries, because v1 used one sequential seeded shuffle).
Freed labels become a reserve for judging new trials retrieved by improved
systems later. v1 files are not modified.

Run from the project root:
    PYTHONPATH=$PWD python experiments/24_build_labeling_sheet_v1_1.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from src.evaluation.judgments import load_query_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "data/evaluation"
SPECS_V11 = EVAL / "query_specs_v1.1.json"

V1 = {"sheet": EVAL / "labeling_sheet_v1.xlsx", "provenance": EVAL / "pool_v1_provenance.csv",
      "template": EVAL / "qrels_v1_template.csv", "summary": EVAL / "pool_v1_summary.json"}
V11 = {"sheet": EVAL / "labeling_sheet_v1.1.xlsx", "provenance": EVAL / "pool_v1.1_provenance.csv",
       "template": EVAL / "qrels_v1.1_template.csv", "summary": EVAL / "pool_v1.1_summary.json"}

BUDGET = 500

_spec = importlib.util.spec_from_file_location(
    "exp22", PROJECT_ROOT / "experiments/22_build_pools_and_labeling_sheet.py")
exp22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp22)


def main() -> None:
    for path in V11.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    specs = load_query_specs(SPECS_V11)
    removed_queries = sorted(q for q, s in specs.items() if not s.has_answer)

    # Rows exactly as they appear in the v1 sheet (order, row_id, content).
    sheet = load_workbook(V1["sheet"])["Labels"]
    headers = [cell.value for cell in sheet[1]]
    rows = []
    for r in range(2, sheet.max_row + 1):
        row = {h: sheet.cell(r, c).value for c, h in enumerate(headers, start=1)}
        row = {k: ("" if v is None else v) for k, v in row.items()}
        row["link"] = sheet.cell(r, headers.index("NCT ID") + 1).hyperlink.target
        rows.append(row)
    kept = [row for row in rows if row["query_id"] not in removed_queries]

    exp22.INSTRUCTIONS = (
        exp22.INSTRUCTIONS
        .replace("query set v1", "query set v1.1")
        .replace("docs/labeling_guide_v1.md", "docs/labeling_guide_v1.1.md")
        .replace("docs/labeling_rubric_v1.md", "docs/labeling_rubric_v1.1.md")
    )
    exp22.write_sheet(kept, specs, None, V11["sheet"])

    provenance = pd.read_csv(V1["provenance"])
    template = pd.read_csv(V1["template"], dtype=str, keep_default_na=False)
    provenance[~provenance["query_id"].isin(removed_queries)].to_csv(V11["provenance"], index=False)
    template[~template["query_id"].isin(removed_queries)].to_csv(V11["template"], index=False)

    v1_summary = json.loads(V1["summary"].read_text())
    per_query = {q: d for q, d in v1_summary["per_query"].items() if q not in removed_queries}
    total = sum(d["to_label"] for d in per_query.values())
    removed_rows = {q: v1_summary["per_query"][q]["to_label"]
                    for q in removed_queries if q in v1_summary["per_query"]}
    answerable = {split: sorted(q for q, s in specs.items() if s.split == split and s.has_answer)
                  for split in ("dev", "test")}

    summary = {
        "derived_from": "pool_v1 / labeling_sheet_v1.xlsx (rows copied unchanged)",
        "query_specs": "data/evaluation/query_specs_v1.1.json",
        "rubric": "docs/labeling_rubric_v1.1.md",
        "budget": BUDGET,
        "total_to_label": total,
        "reserve": BUDGET - total,
        "reserve_purpose": "judge new trials retrieved by improved systems later; no new queries",
        "removed_rows_by_query": removed_rows,
        "zero_answer_queries_not_pooled": removed_queries,
        "answerable_queries": {split: {"count": len(qs), "queries": qs} for split, qs in answerable.items()},
        "per_query": per_query,
    }
    V11["summary"].write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Removed queries: {removed_queries} -> rows removed: {removed_rows}")
    print(f"Rows to label: {total} / {BUDGET} | reserve: {BUDGET - total}")
    print(f"Answerable: dev {len(answerable['dev'])}, test {len(answerable['test'])}")
    for path in V11.values():
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
