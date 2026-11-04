"""
FINAL TEST EVALUATION - run once.

Baseline pipeline vs frozen final pipeline (src/config.py,
data/evaluation/frozen_config_final.json) on the 10 TEST queries
(7 answerable + 3 zero-answer), qrels v1.2. Before running, every frozen
SHA-256 is checked against the files on disk; the script refuses to run if
the output folder already exists.

Outputs: data/runs/final_test/{rrf.baseline.csv, rrf.final.csv}
         data/evaluation/results/final_test/
             summary.csv (bootstrap 95% CIs), paired.csv, per_query.csv,
             violations.csv, zero_answer.csv, unjudged_top10.csv, config.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/35_final_test_evaluation.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import BASELINE_PIPELINE, FINAL_PIPELINE
from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs, prepare_run
from src.evaluation.retrieval_metrics import (
    constraint_violations, evaluate_run, macro_summary, paired_comparison,
)
from src.retrieval.pipeline import FUSION_TOP_K, TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/evaluation/frozen_config_final.json"
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
PARQUET = ROOT / "data/processed/trials.parquet"
SAVED_BASELINE = ROOT / "data/runs/v1/rrf.parser_filters.csv"
RUN_DIR = ROOT / "data/runs/final_test"
OUT_DIR = ROOT / "data/evaluation/results/final_test"

METRICS = ["ndcg@10", "cond_ndcg@10", "p_strict@10", "p_lenient@10", "mrr_strict", "mrr_lenient",
           "judged@10", "violation_rate@10"]
PAIR_METRICS = ["ndcg@10", "cond_ndcg@10", "p_lenient@10", "mrr_lenient", "violation_rate@10"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for folder in (RUN_DIR, OUT_DIR):
        if folder.exists():
            raise FileExistsError(f"{folder} exists: the test evaluation runs only once.")

    frozen = json.loads(FROZEN.read_text())
    mismatched = [p for group in ("data_sha256", "code_sha256")
                  for p, digest in frozen[group].items() if sha256(ROOT / p) != digest]
    if mismatched or frozen["final_pipeline"] != FINAL_PIPELINE:
        raise RuntimeError(f"Frozen configuration changed: {mismatched}")

    specs = load_query_specs(SPECS)
    test = sorted(q for q, s in specs.items() if s.split == "test")

    runs, engines = [], {}
    for name, config in (("baseline", BASELINE_PIPELINE), ("final", FINAL_PIPELINE)):
        engine = TrialSearchEngine(**config)
        engines[name] = engine.describe()
        rows = []
        for query_id in test:
            result = engine.search(specs[query_id].text, top_k=FUSION_TOP_K)
            rows += [{"query_id": query_id, "system": name, "nct_id": t["nct_id"],
                      "rank": t["rank"], "score": t["score"]} for t in result.trials]
        runs.append(pd.DataFrame(rows))
        del engine

    saved = pd.read_csv(SAVED_BASELINE)
    baseline = runs[0]
    reproduces = all(
        saved[saved.query_id == q].sort_values("rank").nct_id.tolist()
        == baseline[baseline.query_id == q].sort_values("rank").nct_id.tolist() for q in test)

    run = prepare_run(pd.concat(runs, ignore_index=True))
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    qrels = qrels[qrels["query_id"].isin(test)]
    metadata = pd.read_parquet(PARQUET)
    result = evaluate_run(run, qrels, specs, metadata, ks=(5, 10), query_ids=test)

    summary = macro_summary(result.per_query, metrics=METRICS)
    paired = pd.DataFrame([paired_comparison(result.per_query, "final", "baseline", m) for m in PAIR_METRICS])
    # Everything is computed; only now create the output folders and write.
    RUN_DIR.mkdir(parents=True)
    OUT_DIR.mkdir(parents=True)
    for frame in runs:
        frame.to_csv(RUN_DIR / f"rrf.{frame['system'].iloc[0]}.csv", index=False)
    result.per_query.to_csv(OUT_DIR / "per_query.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    paired.to_csv(OUT_DIR / "paired.csv", index=False)
    result.violations.to_csv(OUT_DIR / "violations.csv", index=False)
    result.zero_answer.to_csv(OUT_DIR / "zero_answer.csv", index=False)

    judged = set(zip(qrels.query_id, qrels.nct_id))
    records = metadata.set_index("nct_id").to_dict("index")
    unjudged = [{"system": s, "query_id": q, "nct_id": n, "rank": r}
                for frame in runs for q, n, r, s in zip(frame.query_id, frame.nct_id, frame["rank"], frame.system)
                if r <= 10 and specs[q].has_answer and (q, n) not in judged
                and not constraint_violations(records[n], specs[q].structured_constraints)]
    pd.DataFrame(unjudged, columns=["system", "query_id", "nct_id", "rank"]).to_csv(
        OUT_DIR / "unjudged_top10.csv", index=False)

    (OUT_DIR / "config.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                     text=True).stdout.strip(),
        "frozen_config": str(FROZEN.relative_to(ROOT)), "frozen_hashes_verified": True,
        "split": "test", "queries": test, "engines": engines,
        "baseline_reproduces_saved_v1_run": reproduces,
    }, indent=2) + "\n")

    pd.set_option("display.width", 220)
    table = summary.pivot(index="metric", columns="system", values=["mean", "ci_low", "ci_high"]).reindex(METRICS)
    print(table.round(3).to_string())
    print("\nn queries per metric:", summary.groupby("metric")["n_queries"].first().reindex(METRICS).to_dict())
    print("\nPaired (final - baseline):")
    print(paired[["metric", "n_queries", "mean_diff", "ci_low", "ci_high", "p_value", "wins_a", "ties",
                  "wins_b"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPer-query nDCG@10 / violation@10:")
    print(result.per_query.pivot(index="query_id", columns="system",
                                 values=["ndcg@10", "cond_ndcg@10", "violation_rate@10", "judged@10"]).round(2).to_string())
    print("\nZero-answer queries:")
    print(result.zero_answer[["system", "query_id", "n_returned", "best_score"]].to_string(index=False))
    print(f"\nUnjudged constraint-satisfying top-10 trials: "
          f"{pd.DataFrame(unjudged).groupby('system').size().to_dict() if unjudged else {}}")
    print(f"Baseline reproduces saved v1 run on test: {reproduces}")


if __name__ == "__main__":
    main()
