"""
M1 Step 5 - import the filled labeling sheet into qrels v1.1.

Validates the sheet against qrels_v1.1_template.csv and query_specs_v1.1.json
(src/evaluation/label_import.py). A timestamped JSON report is always
written. The qrels file is written only when there are no errors, and an
existing qrels file is never overwritten. Pending (blank) rows are allowed,
so a partly finished sheet can be checked at any time.

Usage (from the project root):
    PYTHONPATH=$PWD python experiments/25_import_labels.py \
        --sheet data/evaluation/labeling_sheet_v1.1_labeled.xlsx --labeler AJ
Options:
    --date YYYY-MM-DD   label date recorded in qrels (default: today)
    --check-only        validate and report, never write qrels
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import load_qrels, load_query_specs
from src.evaluation.label_import import import_labels, read_label_sheet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "data/evaluation"
SPECS = EVAL / "query_specs_v1.1.json"
TEMPLATE = EVAL / "qrels_v1.1_template.csv"
QRELS_OUT = EVAL / "qrels_v1.1.csv"
REPORT_DIR = EVAL / "label_import_reports"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--labeler", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    date.fromisoformat(args.date)
    specs = load_query_specs(SPECS)
    template = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False)
    sheet = read_label_sheet(args.sheet)

    qrels, report = import_labels(sheet, template, specs, args.labeler, args.date)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = REPORT_DIR / f"import_report_{stamp}.json"
    payload = {"sheet": str(args.sheet), "labeler": args.labeler, "label_date": args.date,
               "rows_in_template": len(template), "rows_judged": len(qrels),
               **report.as_dict()}
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    totals = pd.DataFrame(report.per_query).T
    print(totals.to_string())
    print(f"\nJudged {len(qrels)} / {len(template)} rows | partial: {len(report.partial)} "
          f"| UNCERTAIN: {len(report.uncertain)} | errors: {len(report.errors)}")
    for error in report.errors[:30]:
        print(f"  ERROR {error['row_id']}: {error['error']}")
    if len(report.errors) > 30:
        print(f"  ... {len(report.errors) - 30} more errors in the report")
    for row_id in report.partial[:30]:
        print(f"  PARTIAL {row_id}")
    for warning in report.warnings:
        print(f"  WARNING {warning}")
    print(f"Report: {report_path}")

    if not report.ok:
        print("\nNot writing qrels: fix the errors above and re-run.")
        return 1
    if args.check_only:
        return 0
    if QRELS_OUT.exists():
        print(f"\n{QRELS_OUT} already exists; not overwriting. Remove or rename it deliberately.")
        return 1

    qrels.to_csv(QRELS_OUT, index=False)
    reloaded = load_qrels(QRELS_OUT, specs)          # final check with the Step 2 loader
    print(f"\nSaved {QRELS_OUT} ({len(reloaded)} judged rows accepted by the loader).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
