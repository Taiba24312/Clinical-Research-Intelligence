"""
Parser v1.1 on the DEV queries: RRF with the fixed parser vs the old
pipeline (parser v1) and the gold-filter ceiling. No other change or tuning.

Outputs (never overwritten):
    data/runs/v1.1_parser_fix_dev/rrf.parser_v1_1.csv (+ run_config.json)
    data/evaluation/results/parser_fix_v1.1_dev/{per_query.csv, summary.csv,
        unjudged_new_top10.csv, config.json}

Run from the project root:
    PYTHONPATH=$PWD python experiments/28_parser_fix_dev_eval.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs, prepare_run
from src.evaluation.retrieval_metrics import constraint_violations, evaluate_run, paired_comparison
from src.retrieval.pipeline import FUSION_TOP_K, TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
PARQUET = ROOT / "data/processed/trials.parquet"
OLD_RUN = ROOT / "data/runs/v1/rrf.parser_filters.csv"
GOLD_RUN = ROOT / "data/runs/v1/rrf.gold_filters.csv"
RUN_DIR = ROOT / "data/runs/v1.1_parser_fix_dev"
OUT_DIR = ROOT / "data/evaluation/results/parser_fix_v1.1_dev"

SYSTEMS = {"rrf.parser_v1": "old pipeline", "rrf.parser_v1_1": "fixed parser",
           "rrf.gold_filters": "gold-filter ceiling (pooling-only)"}
METRICS = ["ndcg@10", "cond_ndcg@10", "judged@10", "violation_rate@10"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for folder in (RUN_DIR, OUT_DIR):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    dev = sorted(q for q, s in specs.items() if s.split == "dev")

    engine = TrialSearchEngine(parser_version="v1.1")
    rows, parsed_log = [], {}
    for query_id in dev:
        result = engine.search(specs[query_id].text, top_k=FUSION_TOP_K)
        parsed_log[query_id] = {"semantic_query": result.semantic_query, "filters": result.filters,
                                "gold_constraints": specs[query_id].structured_constraints}
        rows += [{"query_id": query_id, "system": "rrf.parser_v1_1", "nct_id": t["nct_id"],
                  "rank": t["rank"], "score": t["score"]} for t in result.trials]
    new_run = pd.DataFrame(rows)
    new_run.to_csv(RUN_DIR / "rrf.parser_v1_1.csv", index=False)
    (RUN_DIR / "run_config.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "src/retrieval/pipeline.py TrialSearchEngine (parser v1.1, RRF top-50, no reranker)",
        "device": engine.device, "split": "dev", "queries": parsed_log,
        "corpus_parquet_sha256": engine.corpus_sha256,
    }, indent=2) + "\n")

    old = pd.read_csv(OLD_RUN).assign(system="rrf.parser_v1")
    gold = pd.read_csv(GOLD_RUN)
    run = prepare_run(pd.concat([old, new_run, gold], ignore_index=True))
    run = run[run["query_id"].isin(dev)]

    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    qrels = qrels[qrels["query_id"].isin(dev)]
    metadata = pd.read_parquet(PARQUET)
    result = evaluate_run(run, qrels, specs, metadata, ks=(10,), query_ids=dev)
    per_query = result.per_query
    per_query.to_csv(OUT_DIR / "per_query.csv", index=False)

    summary = []
    for system, label in SYSTEMS.items():
        group = per_query[per_query["system"] == system]
        row = {"system": system, "description": label}
        for metric in METRICS:
            values = group[metric].dropna()
            row[metric] = values.mean()
            row[f"n_{metric}"] = len(values)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    paired = pd.DataFrame([paired_comparison(per_query, "rrf.parser_v1_1", "rrf.parser_v1", m)
                           for m in ("ndcg@10", "cond_ndcg@10")])
    paired.to_csv(OUT_DIR / "paired.csv", index=False)

    # Newly retrieved top-10 trials that satisfy the constraints but are unjudged.
    judged = set(zip(qrels["query_id"], qrels["nct_id"]))
    records = metadata.set_index("nct_id").to_dict("index")
    old_top = {(q, n) for q, n, r in zip(old.query_id, old.nct_id, old["rank"]) if r <= 10}
    unjudged = [
        {"query_id": q, "nct_id": n, "rank": r}
        for q, n, r in zip(new_run.query_id, new_run.nct_id, new_run["rank"])
        if r <= 10 and (q, n) not in old_top and (q, n) not in judged
        and not constraint_violations(records[n], specs[q].structured_constraints)
    ]
    pd.DataFrame(unjudged, columns=["query_id", "nct_id", "rank"]).to_csv(
        OUT_DIR / "unjudged_new_top10.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT_DIR / "config.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit_before_run": commit,
        "split": "dev", "queries": dev, "systems": SYSTEMS,
        "qrels_sha256": sha256(QRELS), "query_specs_sha256": sha256(SPECS),
    }, indent=2) + "\n")

    wide = per_query.pivot(index="query_id", columns="system", values=["ndcg@10", "violation_rate@10"])
    changed = [q for q in dev if parsed_log[q]["filters"] != {}
               and json.dumps(parsed_log[q]["filters"], sort_keys=True) !=
               json.dumps(__import__("src.query_understanding.parser", fromlist=["parse_query"])
                          .parse_query(specs[q].text)["filters"], sort_keys=True)]
    worse = [q for q in dev
             if wide.loc[q, ("ndcg@10", "rrf.parser_v1_1")] < wide.loc[q, ("ndcg@10", "rrf.parser_v1")] - 1e-9]

    pd.set_option("display.width", 200)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPer-query (dev queries whose parse changed):")
    print(wide.loc[changed].round(3).to_string())
    print("\nPaired v1.1 vs v1:")
    print(paired[["metric", "n_queries", "mean_diff", "ci_low", "ci_high", "p_value",
                  "wins_a", "ties", "wins_b"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nDev queries that got worse (nDCG@10): {worse or 'none'}")
    print(f"Newly retrieved, constraint-satisfying, unjudged top-10 trials: {len(unjudged)}")
    print(f"Saved: {RUN_DIR} and {OUT_DIR}")


if __name__ == "__main__":
    main()
