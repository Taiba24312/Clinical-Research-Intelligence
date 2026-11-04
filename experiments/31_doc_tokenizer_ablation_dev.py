"""
Dev ablation: document v2 and BM25 tokenizer v2, alone and together.

All configurations: parser v1.1, RRF (BM25 + bge-small, k=60), top-50, no
reranker. DEV queries only.
    A  doc v1, tokenizer v1   (current pipeline with the fixed parser)
    B  doc v2, tokenizer v1
    C  doc v1, tokenizer v2
    D  doc v2, tokenizer v2

Outputs (never overwritten):
    data/runs/v2_dev_ablation/rrf.{A,B,C,D}.csv + run_config.json
    data/evaluation/results/doc_tokenizer_ablation_dev/
        summary.csv, per_query.csv, paired_vs_A.csv, unjudged_top10.csv, config.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/31_doc_tokenizer_ablation_dev.py
"""

from __future__ import annotations

import gc
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs, prepare_run
from src.evaluation.retrieval_metrics import constraint_violations, evaluate_run, paired_comparison
from src.retrieval.pipeline import FUSION_TOP_K, TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
PARQUET = ROOT / "data/processed/trials.parquet"
RUN_DIR = ROOT / "data/runs/v2_dev_ablation"
OUT_DIR = ROOT / "data/evaluation/results/doc_tokenizer_ablation_dev"

CONFIGS = {
    "A": {"doc_version": "v1", "tokenizer": "v1"},
    "B": {"doc_version": "v2", "tokenizer": "v1"},
    "C": {"doc_version": "v1", "tokenizer": "v2"},
    "D": {"doc_version": "v2", "tokenizer": "v2"},
}
METRICS = ["ndcg@10", "cond_ndcg@10", "p_lenient@10", "mrr_lenient", "judged@10", "violation_rate@10"]
FOCUS = ["Q01", "Q11", "Q22"]


def main() -> None:
    for folder in (RUN_DIR, OUT_DIR):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    dev = sorted(q for q, s in specs.items() if s.split == "dev")

    runs, engine_configs = [], {}
    for name, options in CONFIGS.items():
        engine = TrialSearchEngine(parser_version="v1.1", **options)
        engine_configs[name] = engine.describe()
        rows = []
        for query_id in dev:
            result = engine.search(specs[query_id].text, top_k=FUSION_TOP_K)
            rows += [{"query_id": query_id, "system": name, "nct_id": t["nct_id"],
                      "rank": t["rank"], "score": t["score"]} for t in result.trials]
        frame = pd.DataFrame(rows)
        frame.to_csv(RUN_DIR / f"rrf.{name}.csv", index=False)
        runs.append(frame)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        print(f"config {name} {options}: done")
    (RUN_DIR / "run_config.json").write_text(json.dumps(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "split": "dev",
         "configs": CONFIGS, "engines": engine_configs}, indent=2) + "\n")

    run = prepare_run(pd.concat(runs, ignore_index=True))
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    qrels = qrels[qrels["query_id"].isin(dev)]
    metadata = pd.read_parquet(PARQUET)
    result = evaluate_run(run, qrels, specs, metadata, ks=(10,), query_ids=dev)
    per_query = result.per_query
    per_query.to_csv(OUT_DIR / "per_query.csv", index=False)

    summary = per_query.groupby("system")[METRICS].mean().reindex(list(CONFIGS))
    summary.insert(0, "n_ndcg_queries", per_query.groupby("system")["ndcg@10"].count())
    summary.to_csv(OUT_DIR / "summary.csv")
    paired = pd.DataFrame([paired_comparison(per_query, name, "A", metric)
                           for name in ("B", "C", "D") for metric in ("ndcg@10", "cond_ndcg@10")])
    paired.to_csv(OUT_DIR / "paired_vs_A.csv", index=False)

    judged = set(zip(qrels["query_id"], qrels["nct_id"]))
    records = metadata.set_index("nct_id").to_dict("index")
    unjudged = []
    for frame in runs:
        top = frame[frame["rank"] <= 10]
        for q, n, r, s in zip(top.query_id, top.nct_id, top["rank"], top.system):
            if (q, n) not in judged and not constraint_violations(records[n], specs[q].structured_constraints):
                unjudged.append({"system": s, "query_id": q, "nct_id": n, "rank": r})
    unjudged = pd.DataFrame(unjudged, columns=["system", "query_id", "nct_id", "rank"])
    unjudged.to_csv(OUT_DIR / "unjudged_top10.csv", index=False)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT_DIR / "config.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit_before_run": commit,
        "split": "dev", "queries": dev, "configs": CONFIGS, "parser": "v1.1",
        "qrels_sha256": hashlib.sha256(QRELS.read_bytes()).hexdigest()}, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print("\n" + summary.round(3).to_string())
    print("\nPaired vs A:")
    print(paired[["metric", "system_a", "n_queries", "mean_diff", "ci_low", "ci_high", "p_value",
                  "wins_a", "ties", "wins_b"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    wide = per_query.pivot(index="query_id", columns="system", values="ndcg@10")[list(CONFIGS)]
    print("\nPer-query nDCG@10 (queries that change in any config):")
    print(wide[(wide.max(axis=1) - wide.min(axis=1)) > 1e-9].round(3).to_string())
    print("\nFocus queries (nDCG@10 / judged@10):")
    print(per_query[per_query.query_id.isin(FOCUS)].pivot(
        index="query_id", columns="system", values=["ndcg@10", "judged@10"]).round(2).to_string())
    print("\nUnjudged constraint-satisfying top-10 trials per config:", unjudged.groupby("system").size().to_dict(),
          "| distinct (query, trial) pairs across B/C/D:",
          len(set(zip(unjudged[unjudged.system != "A"].query_id, unjudged[unjudged.system != "A"].nct_id))))


if __name__ == "__main__":
    main()
