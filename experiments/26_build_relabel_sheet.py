"""
M1 Step 5 - consistency re-label sample (about 10% of the labeler's judgments).

Sample: per query, max(1, round(10% of its judged rows)) rows from
qrels_v1.1.csv (the labeler's own judgments, not copied paraphrase rows),
drawn with a fixed seed. Rows get new ids (R-001, ...) so earlier labels
cannot be looked up by row_id; the mapping is kept in a separate key file
that the labeler should not open until the re-label is done.

Outputs (never overwritten), in data/evaluation/relabel_v1.1/:
    labeling_sheet_v1.1_relabel.xlsx   blank label cells, rows shuffled per query
    relabel_v1.1_template.csv          qrels template for the importer
    relabel_v1.1_key.csv               relabel_id -> original row_id (NOT for the labeler)

Run from the project root:
    PYTHONPATH=$PWD python experiments/26_build_relabel_sheet.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from src.evaluation.judgments import load_query_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "data/evaluation"
QRELS = EVAL / "qrels_v1.1.csv"
BLANK_SHEET = EVAL / "labeling_sheet_v1.1.xlsx"
TEMPLATE = EVAL / "qrels_v1.1_template.csv"
SPECS = EVAL / "query_specs_v1.1.json"

OUT_DIR = EVAL / "relabel_v1.1"
SHEET_OUT = OUT_DIR / "labeling_sheet_v1.1_relabel.xlsx"
TEMPLATE_OUT = OUT_DIR / "relabel_v1.1_template.csv"
KEY_OUT = OUT_DIR / "relabel_v1.1_key.csv"

SEED = 20261102
FRACTION = 0.10
OPEN_ON = "2026-11-09"

_spec = importlib.util.spec_from_file_location(
    "exp22", PROJECT_ROOT / "experiments/22_build_pools_and_labeling_sheet.py")
exp22 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp22)


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    qrels = pd.read_csv(QRELS, dtype=str, keep_default_na=False)
    template = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False).set_index("row_id")

    rng = np.random.default_rng(SEED)
    sampled = []
    for query_id in sorted(qrels["query_id"].unique(),
                           key=lambda q: (0 if specs[q].split == "dev" else 1, q)):
        rows = sorted(qrels.loc[qrels["query_id"] == query_id, "row_id"])
        k = max(1, round(FRACTION * len(rows)))
        sampled.extend(rows[i] for i in rng.choice(len(rows), size=k, replace=False))

    # Blank-sheet rows (no earlier labels) for the sampled row_ids.
    sheet = load_workbook(BLANK_SHEET)["Labels"]
    headers = [c.value for c in sheet[1]]
    by_id = {}
    for r in range(2, sheet.max_row + 1):
        row = {h: ("" if sheet.cell(r, c).value is None else sheet.cell(r, c).value)
               for c, h in enumerate(headers, start=1)}
        row["link"] = sheet.cell(r, headers.index("NCT ID") + 1).hyperlink.target
        by_id[row["row_id"]] = row

    rows, key, relabel_template = [], [], []
    for i, row_id in enumerate(sampled, start=1):
        relabel_id = f"R-{i:03d}"
        row = dict(by_id[row_id], row_id=relabel_id)
        rows.append(row)
        ref = template.loc[row_id]
        key.append({"relabel_id": relabel_id, "row_id": row_id,
                    "query_id": ref["query_id"], "nct_id": ref["nct_id"]})
        relabel_template.append({"row_id": relabel_id, "query_id": ref["query_id"],
                                 "nct_id": ref["nct_id"], "disease_match": ref["disease_match"],
                                 "intervention_match": ref["intervention_match"],
                                 "mention_role": "", "notes": "", "labeler": "", "label_date": ""})

    exp22.INSTRUCTIONS = (
        f"CONSISTENCY RE-LABEL - please open on or after {OPEN_ON}.\n"
        "Label these rows from scratch with rubric v1.1 (docs/labeling_rubric_v1.1.md).\n"
        "Do not look at your earlier labels or at relabel_v1.1_key.csv until you are done.\n\n"
        + exp22.INSTRUCTIONS
        .replace("query set v1", "query set v1.1")
        .replace("docs/labeling_guide_v1.md", "docs/labeling_guide_v1.1.md")
        .replace("docs/labeling_rubric_v1.md", "docs/labeling_rubric_v1.1.md")
    )
    exp22.write_sheet(rows, specs, None, SHEET_OUT)
    pd.DataFrame(key).to_csv(KEY_OUT, index=False)
    pd.DataFrame(relabel_template).to_csv(TEMPLATE_OUT, index=False)

    counts = pd.DataFrame(key)["query_id"].value_counts().sort_index()
    print(f"Sampled {len(sampled)} of {len(qrels)} judged rows (seed {SEED}):")
    print(counts.to_string())
    for path in (SHEET_OUT, TEMPLATE_OUT, KEY_OUT):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
