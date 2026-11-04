"""
Import the labeled test sheet as qrels v1.3 and recompute the TEST results.

  1. Verify the frozen configuration hashes (nothing may have changed).
  2. Import data/evaluation/test_labels_v1/ labels (all 11 rows must be
     judged; same validation as experiments/25).
  3. qrels v1.3 = qrels v1.2 + these judgments (no overlapping pairs).
  4. Re-score the SAVED test runs of Experiment 35 (data/runs/final_test/)
     for both pipelines with qrels v1.2 (must reproduce the original numbers)
     and qrels v1.3. Retrieval is not re-run, so the frozen pipeline's
     rankings cannot change.

Outputs (never overwritten): data/evaluation/qrels_v1.3.csv and
data/evaluation/results/final_test_qrels_v1.3/{summary.csv, paired.csv,
per_query.csv, import_report.json, config.json}

Usage (from the project root):
    PYTHONPATH=$PWD python experiments/37_import_test_labels_and_recompute.py \
        --sheet data/evaluation/test_labels_v1/labeling_sheet_test_v1_labeled.xlsx --labeler AJ
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import (
    QRELS_COLUMNS, expand_shared_judgments, load_qrels, load_query_specs, prepare_qrels, prepare_run,
)
from src.evaluation.label_import import import_labels, read_label_sheet
from src.evaluation.retrieval_metrics import evaluate_run, macro_summary, paired_comparison


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/evaluation/frozen_config_final.json"
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS_V12 = ROOT / "data/evaluation/qrels_v1.2.csv"
TEMPLATE = ROOT / "data/evaluation/test_labels_v1/qrels_test_v1_template.csv"
RUN_DIR = ROOT / "data/runs/final_test"
PARQUET = ROOT / "data/processed/trials.parquet"
ORIGINAL_SUMMARY = ROOT / "data/evaluation/results/final_test/summary.csv"

METRICS = ["ndcg@10", "cond_ndcg@10", "p_strict@10", "p_lenient@10", "mrr_strict", "mrr_lenient",
           "judged@10", "violation_rate@10"]
PAIR_METRICS = ["ndcg@10", "cond_ndcg@10", "p_lenient@10", "mrr_lenient", "violation_rate@10"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(qrels: pd.DataFrame, specs, test: list[str]):
    run = prepare_run(pd.concat([pd.read_csv(RUN_DIR / f"rrf.{s}.csv") for s in ("baseline", "final")],
                                ignore_index=True))
    qrels = expand_shared_judgments(qrels, specs)
    result = evaluate_run(run, qrels[qrels["query_id"].isin(test)], specs,
                          pd.read_parquet(PARQUET), ks=(5, 10), query_ids=test)
    summary = macro_summary(result.per_query, metrics=METRICS)
    paired = pd.DataFrame([paired_comparison(result.per_query, "final", "baseline", m) for m in PAIR_METRICS])
    return result, summary, paired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sheet", required=True, type=Path)
    parser.add_argument("--labeler", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--qrels-out", type=Path, default=ROOT / "data/evaluation/qrels_v1.3.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/evaluation/results/final_test_qrels_v1.3")
    args = parser.parse_args()
    for path in (args.qrels_out, args.out_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    frozen = json.loads(FROZEN.read_text())
    changed = [p for group in ("data_sha256", "code_sha256")
               for p, digest in frozen[group].items() if sha256(ROOT / p) != digest]
    if changed:
        raise RuntimeError(f"Frozen files changed: {changed}")

    specs = load_query_specs(SPECS)
    test = sorted(q for q, s in specs.items() if s.split == "test")

    # ---- import and merge ------------------------------------------------
    template = pd.read_csv(TEMPLATE, dtype=str, keep_default_na=False)
    new, report = import_labels(read_label_sheet(args.sheet), template, specs, args.labeler, args.date)
    if not report.ok or len(new) != len(template):
        print(json.dumps(report.as_dict(), indent=1)[:3000])
        print(f"\nNot importing: {len(report.errors)} errors, {len(new)}/{len(template)} rows judged.")
        return 1
    v12 = pd.read_csv(QRELS_V12, dtype=str, keep_default_na=False)
    overlap = set(zip(v12.query_id, v12.nct_id)) & set(zip(new.query_id, new.nct_id))
    if overlap:
        raise RuntimeError(f"New labels overlap existing judgments: {sorted(overlap)}")
    v13 = pd.concat([v12, new[v12.columns.tolist()]], ignore_index=True)
    prepare_qrels(v13[QRELS_COLUMNS], specs)                      # full validation before writing

    # ---- recompute ---------------------------------------------------------
    _, summary_v12, paired_v12 = evaluate(load_qrels(QRELS_V12, specs), specs, test)
    original = pd.read_csv(ORIGINAL_SUMMARY)
    merged = summary_v12.merge(original, on=["system", "metric"], suffixes=("", "_orig"))
    if not (merged["mean"] - merged["mean_orig"]).abs().fillna(0).lt(1e-12).all():
        raise RuntimeError("Re-scoring with qrels v1.2 does not reproduce the original test results.")

    args.out_dir.mkdir(parents=True)
    v13.to_csv(args.qrels_out, index=False)
    qrels_v13 = load_qrels(args.qrels_out, specs)
    result_v13, summary_v13, paired_v13 = evaluate(qrels_v13, specs, test)

    table = (summary_v12[["system", "metric", "mean"]].rename(columns={"mean": "qrels_v1.2"})
             .merge(summary_v13[["system", "metric", "mean", "ci_low", "ci_high", "n_queries"]]
                    .rename(columns={"mean": "qrels_v1.3", "ci_low": "v1.3_ci_low", "ci_high": "v1.3_ci_high"}),
                    on=["system", "metric"]))
    table.to_csv(args.out_dir / "summary.csv", index=False)
    paired = pd.concat([paired_v12.assign(qrels="v1.2"), paired_v13.assign(qrels="v1.3")])
    paired.to_csv(args.out_dir / "paired.csv", index=False)
    result_v13.per_query.to_csv(args.out_dir / "per_query.csv", index=False)
    (args.out_dir / "import_report.json").write_text(json.dumps(report.as_dict(), indent=2) + "\n")
    (args.out_dir / "config.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sheet": str(args.sheet), "labeler": args.labeler, "label_date": args.date,
        "qrels_v1.3": str(args.qrels_out), "qrels_v1.3_sha256": sha256(args.qrels_out),
        "rows_added": len(new), "runs_rescored": "data/runs/final_test (Experiment 35, not re-run)",
        "frozen_hashes_verified": True, "v1.2_rescoring_reproduces_original": True,
    }, indent=2) + "\n")

    pd.set_option("display.width", 220)
    print(table.pivot(index="metric", columns="system", values=["qrels_v1.2", "qrels_v1.3"])
          .reindex(METRICS).round(3).to_string())
    print("\nPaired (final - baseline):")
    print(paired[["qrels", "metric", "n_queries", "mean_diff", "ci_low", "ci_high", "p_value",
                  "wins_a", "ties", "wins_b"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nNew labels by grade:", new.assign(
        grade=[g for g in prepare_qrels(new[QRELS_COLUMNS], specs)["grade"]]).groupby(
        ["query_id", "grade"]).size().to_dict())
    print(f"\nSaved {args.qrels_out} and {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
