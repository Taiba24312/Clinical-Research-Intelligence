"""
M1 Step 5 - baseline retrieval metrics on the DEV queries.

Systems: the four real baseline runs (parser filters) from data/runs/v1:
bm25_ws, dense, rrf, rrf_ce_raw. Pooling-only runs are not evaluated here.
Qrels: data/evaluation/qrels_v1.1.csv plus judgments shared with the
paraphrase queries (Q22 <- Q01, Q23 <- Q07). Test queries are not evaluated.

Outputs (never overwritten), in data/evaluation/results/baseline_v1.1_dev/:
    per_query.csv, summary.csv (all metrics, bootstrap CIs), summary_main.csv,
    paired.csv, violations.csv, zero_answer.csv, config.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/27_baseline_dev_evaluation.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import (
    expand_shared_judgments, load_qrels, load_query_specs, prepare_run,
)
from src.evaluation.retrieval_metrics import (
    DEFAULT_N_BOOT, DEFAULT_SEED, evaluate_run, macro_summary, paired_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL = PROJECT_ROOT / "data/evaluation"
SPECS = EVAL / "query_specs_v1.1.json"
QRELS = EVAL / "qrels_v1.1.csv"
RUN_DIR = PROJECT_ROOT / "data/runs/v1"
PARQUET = PROJECT_ROOT / "data/processed/trials.parquet"
OUT_DIR = EVAL / "results/baseline_v1.1_dev"

SYSTEMS = ["bm25_ws", "dense", "rrf", "rrf_ce_raw"]
MODE = "parser_filters"
KS = (5, 10, 20)
SPLIT = "dev"

MAIN_METRICS = ["ndcg@10", "cond_ndcg@10", "p_strict@10", "p_lenient@10",
                "r_strict@20", "r_lenient@20", "mrr_strict", "mrr_lenient",
                "judged@10", "violation_rate@10"]
PAIRS = [("rrf", "bm25_ws"), ("rrf", "dense"), ("rrf_ce_raw", "rrf")]
PAIR_METRICS = ["ndcg@10", "mrr_lenient", "p_lenient@10"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    dev_ids = sorted(q for q, s in specs.items() if s.split == SPLIT)

    qrels = load_qrels(QRELS, specs)
    qrels = expand_shared_judgments(qrels, specs)
    qrels = qrels[qrels["query_id"].isin(dev_ids)]

    run = prepare_run(pd.concat(
        [pd.read_csv(RUN_DIR / f"{system}.{MODE}.csv") for system in SYSTEMS],
        ignore_index=True,
    ))
    run["system"] = run["system"].str.replace(f".{MODE}", "", regex=False)
    run = run[run["query_id"].isin(dev_ids)]

    result = evaluate_run(run, qrels, specs, pd.read_parquet(PARQUET), ks=KS, query_ids=dev_ids)

    summary = macro_summary(result.per_query)
    main_table = (summary[summary["metric"].isin(MAIN_METRICS)]
                  .assign(order=lambda f: f["metric"].map(MAIN_METRICS.index))
                  .sort_values(["order", "system"]).drop(columns="order"))
    paired = pd.DataFrame([paired_comparison(result.per_query, a, b, m)
                           for a, b in PAIRS for m in PAIR_METRICS])

    result.per_query.to_csv(OUT_DIR / "per_query.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    main_table.to_csv(OUT_DIR / "summary_main.csv", index=False)
    paired.to_csv(OUT_DIR / "paired.csv", index=False)
    result.violations.to_csv(OUT_DIR / "violations.csv", index=False)
    result.zero_answer.to_csv(OUT_DIR / "zero_answer.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True).stdout.strip()
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_before_run": commit,
        "split": SPLIT, "query_ids": dev_ids,
        "systems": {s: f"data/runs/v1/{s}.{MODE}.csv" for s in SYSTEMS},
        "ks": KS, "bootstrap": {"n_boot": DEFAULT_N_BOOT, "seed": DEFAULT_SEED},
        "gain": "2**grade - 1",
        "qrels": {"file": str(QRELS.relative_to(PROJECT_ROOT)), "sha256": sha256(QRELS),
                  "judged_rows_dev_after_sharing": len(qrels),
                  "shared": {q: s.judgments_shared_with for q, s in specs.items()
                             if s.judgments_shared_with}},
        "query_specs_sha256": sha256(SPECS),
        "corpus_parquet_sha256": sha256(PARQUET),
    }
    (OUT_DIR / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print(main_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPaired comparisons:")
    print(paired[["metric", "system_a", "system_b", "n_queries", "mean_diff", "ci_low",
                  "ci_high", "p_value", "wins_a", "ties", "wins_b"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
