"""
RAG v2 on DEV (same 16 queries as Experiment 34: 15 answerable + Q24),
compared with RAG v1 (Experiment 34 outputs, re-scored with the same code).

Retrieval: frozen FINAL_PIPELINE, top-5 context. Model and GPU as in
Experiment 34 (Qwen3-4B-Instruct-2507, cuda:1). Qrels v1.3.

Usage: PYTHONPATH=$PWD python experiments/38_rag_v2_dev.py --prompt-version v2.0
Outputs (never overwritten): data/evaluation/results/rag_v2_dev/<version>/
    outputs.json, answers_per_query.csv, verifications.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import FINAL_PIPELINE
from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs
from src.evaluation.rag_metrics import (
    GOLD_MAP, context_grades, score_answer, split_v1_citations, summarize_answers, verification_agreement,
)
from src.generation.context_builder import TOP_K_CONTEXT
from src.generation.llm import LocalLLM
from src.generation.rag_v2 import PROMPTS, RAGv2
from src.retrieval.pipeline import TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.3.csv"
V1_OUTPUTS = ROOT / "data/evaluation/results/rag_eval_v1_dev/rag_outputs.json"
OUT_ROOT = ROOT / "data/evaluation/results/rag_v2_dev"
DEVICE = "cuda:1"


def load_gold(specs):
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    grade = {(q, n): int(g) for q, n, g in zip(qrels.query_id, qrels.nct_id, qrels.grade)}
    labels = {(q, n): (GOLD_MAP[d], GOLD_MAP[i]) for q, n, d, i in
              zip(qrels.query_id, qrels.nct_id, qrels.disease_match, qrels.intervention_match)}
    return grade, labels


def score_v1(outputs: list[dict], specs, grade_map, records) -> list[dict]:
    rows = []
    for o in outputs:
        spec = specs[o["query_id"]]
        trials = [{"nct_id": c["nct_id"]} for c in o["context"]]
        grades = context_grades(spec, trials, grade_map, records)
        match, partial = split_v1_citations(o["answer"], o["says_no_match"])
        cited = list(dict.fromkeys(re.findall(r"NCT\d{8}", o["answer"])))      # as in Experiment 34
        rows.append(score_answer(o["query_id"], spec.has_answer, grades, cited, match, partial,
                                 o["says_no_match"], o["latency_s"]))
    return rows


def run_v2(rag: RAGv2, engine, query_ids, specs, grade_map, label_map, records):
    outputs, answer_rows, verification_rows = [], [], []
    for query_id in query_ids:
        spec = specs[query_id]
        trials = engine.search(spec.text, top_k=TOP_K_CONTEXT).trials
        result = rag.run(spec.text, trials)
        grades = context_grades(spec, trials, grade_map, records)
        by_grade = {v["nct_id"]: v["verified_grade"] for v in result["verifications"]}
        as_match = [n for n in result["cited"] if by_grade.get(n) == 2]
        as_partial = [n for n in result["cited"] if by_grade.get(n) == 1]
        answer_rows.append(score_answer(query_id, spec.has_answer, grades, result["cited"], as_match,
                                        as_partial, result["says_no_match"], result["latency_s"])
                           | {"cited_unverified": len(result["cited_unverified"])})
        for v in result["verifications"]:
            gold = label_map.get((query_id, v["nct_id"])) if spec.has_answer else None
            gold_grade = grades[v["nct_id"]]
            verification_rows.append({
                "query_id": query_id, "nct_id": v["nct_id"],
                "model_disease": v["disease_match"], "model_intervention": v["intervention_match"],
                "model_grade": str(v["model_grade"]), "verified_grade": str(v["verified_grade"]),
                "quote_found": v["quote_found"], "dropped": v["dropped"], "valid_labels": v["valid_labels"],
                "gold_disease": gold[0] if gold else None,
                "gold_intervention": gold[1] if gold else None,
                "gold_grade": None if gold_grade is None else str(gold_grade),
                "evidence": v["evidence"],
            })
        outputs.append({"query_id": query_id, "query": spec.text,
                        "context": [{"nct_id": t["nct_id"], "title": t["title"],
                                     "grade_for_checking": grades[t["nct_id"]]} for t in trials],
                        **{k: result[k] for k in ("prompt_version", "concepts", "answer", "cited",
                                                  "invalid_citations", "cited_unverified",
                                                  "says_no_match", "latency_s")},
                        "verifications": [{k: v[k] for k in v if k != "raw"} | {"raw": v["raw"]}
                                          for v in result["verifications"]]})
        print(f"{query_id}: concepts={result['concepts']} verified="
              f"{[(v['nct_id'][-4:], v['verified_grade']) for v in result['verifications']]} "
              f"no_match={result['says_no_match']} ({result['latency_s']} s)")
    return outputs, answer_rows, verification_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", required=True, choices=sorted(PROMPTS))
    args = parser.parse_args()
    out_dir = OUT_ROOT / args.prompt_version
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {out_dir}")

    specs = load_query_specs(SPECS)
    dev = sorted(q for q, s in specs.items() if s.split == "dev")
    grade_map, label_map = load_gold(specs)
    engine = TrialSearchEngine(device=DEVICE, **FINAL_PIPELINE)
    records = engine.trials.drop(columns=["document"]).set_index("nct_id").to_dict("index")
    rag = RAGv2(LocalLLM(device=DEVICE), prompt_version=args.prompt_version)

    v1_outputs = json.loads(V1_OUTPUTS.read_text())["outputs"]
    v1_rows = score_v1(v1_outputs, specs, grade_map, records)
    outputs, v2_rows, verification_rows = run_v2(rag, engine, dev, specs, grade_map, label_map, records)

    verifications = pd.DataFrame(verification_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "prompt_version": args.prompt_version,
        "split": "dev", "queries": dev, "device": DEVICE, "retrieval": FINAL_PIPELINE, "qrels": QRELS.name,
        "rag_v1": summarize_answers(v1_rows),
        "rag_v2": summarize_answers(v2_rows) | {
            "cited_unverified": int(pd.DataFrame(v2_rows)["cited_unverified"].sum()),
            "verifications": len(verifications),
            "quotes_not_found": int((~verifications["quote_found"]).sum()),
            "verifications_dropped": int(verifications["dropped"].sum()),
            "invalid_labels": int((~verifications["valid_labels"]).sum()),
        },
        "verification_agreement": verification_agreement(verifications),
    }
    out_dir.mkdir(parents=True)
    (out_dir / "outputs.json").write_text(json.dumps({"prompts": PROMPTS[args.prompt_version],
                                                      "outputs": outputs}, indent=2, ensure_ascii=False) + "\n")
    pd.DataFrame(v2_rows).assign(version="v2").pipe(
        lambda f: pd.concat([pd.DataFrame(v1_rows).assign(version="v1"), f])).to_csv(
        out_dir / "answers_per_query.csv", index=False)
    verifications.to_csv(out_dir / "verifications.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    keys = ["citations", "citation_validity", "match_cited", "match_relevant_share", "partial_cited",
            "partial_relevant_share", "no_match_answers", "no_match_correct", "no_match_false",
            "claims_without_relevant", "grade2_not_cited", "median_latency_s"]
    print("\n" + pd.DataFrame({"v1": summary["rag_v1"], "v2": summary["rag_v2"]}).loc[keys].to_string())
    print("\nv2 quote check:", {k: summary["rag_v2"][k] for k in
                                ("verifications", "quotes_not_found", "verifications_dropped", "invalid_labels")})
    for name, stats in summary["verification_agreement"].items():
        print(f"\nAgreement {name}: n={stats['n']} accuracy={stats['accuracy']:.3f} kappa={stats['cohen_kappa']}")
        print(pd.DataFrame(stats["confusion (rows = labeler, columns = model)"]).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
