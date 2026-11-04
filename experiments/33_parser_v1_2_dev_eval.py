"""
Parser v1.2 (age groups) on DEV, with the chosen retrieval configuration C
(document v1 + BM25 tokenizer v2, RRF). Compared with config C + parser
v1.1 from Experiment 31. Only the parser differs.

Outputs (never overwritten):
    data/runs/v2_dev_parser_v1_2/rrf.C_parser_v1_2.csv + run_config.json
    data/evaluation/results/parser_v1_2_dev/{summary.csv, per_query.csv}

Run from the project root:
    PYTHONPATH=$PWD python experiments/33_parser_v1_2_dev_eval.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs, prepare_run
from src.evaluation.retrieval_metrics import evaluate_run
from src.retrieval.pipeline import FUSION_TOP_K, TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
PARQUET = ROOT / "data/processed/trials.parquet"
RUN_C_V11 = ROOT / "data/runs/v2_dev_ablation/rrf.C.csv"
RUN_DIR = ROOT / "data/runs/v2_dev_parser_v1_2"
OUT_DIR = ROOT / "data/evaluation/results/parser_v1_2_dev"
METRICS = ["ndcg@10", "cond_ndcg@10", "p_lenient@10", "mrr_lenient", "judged@10", "violation_rate@10"]


def main() -> None:
    for folder in (RUN_DIR, OUT_DIR):
        if folder.exists() and any(folder.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {folder}")
        folder.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    dev = sorted(q for q, s in specs.items() if s.split == "dev")
    engine = TrialSearchEngine(parser_version="v1.2", doc_version="v1", tokenizer="v2")
    rows = []
    for query_id in dev:
        result = engine.search(specs[query_id].text, top_k=FUSION_TOP_K)
        rows += [{"query_id": query_id, "system": "C_parser_v1_2", "nct_id": t["nct_id"],
                  "rank": t["rank"], "score": t["score"]} for t in result.trials]
    new = pd.DataFrame(rows)
    new.to_csv(RUN_DIR / "rrf.C_parser_v1_2.csv", index=False)
    (RUN_DIR / "run_config.json").write_text(json.dumps(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "split": "dev",
         "engine": engine.describe()}, indent=2) + "\n")

    old = pd.read_csv(RUN_C_V11).assign(system="C_parser_v1_1")
    run = prepare_run(pd.concat([old, new], ignore_index=True))
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    result = evaluate_run(run, qrels[qrels["query_id"].isin(dev)], specs,
                          pd.read_parquet(PARQUET), ks=(10,), query_ids=dev)
    result.per_query.to_csv(OUT_DIR / "per_query.csv", index=False)
    summary = result.per_query.groupby("system")[METRICS].mean()
    summary.to_csv(OUT_DIR / "summary.csv")

    print(summary.round(3).to_string())
    pq = result.per_query
    print("\nQ18:")
    print(pq[pq.query_id == "Q18"][["system", "ndcg@10", "cond_ndcg@10", "judged@10",
                                    "violation_rate@10"]].round(3).to_string(index=False))
    changed = pq.pivot(index="query_id", columns="system", values="ndcg@10")
    print("\nQueries whose nDCG@10 changed:",
          changed[(changed["C_parser_v1_2"] - changed["C_parser_v1_1"]).abs() > 1e-9].round(3).to_dict("index"))


if __name__ == "__main__":
    main()
