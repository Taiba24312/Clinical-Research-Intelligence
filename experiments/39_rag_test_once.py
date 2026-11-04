"""
FINAL RAG TEST EVALUATION - run once.

RAG v1 vs frozen RAG v2 (prompt v2.2) on the 10 TEST queries (7 answerable
+ 3 zero-answer), frozen retrieval, qrels v1.3. Before running, every hash
in data/evaluation/frozen_rag_v2.json is checked; the script refuses to run
if its output folder exists, and writes only after everything is computed.

Outputs: data/evaluation/results/rag_test/
    outputs.json, answers_per_query.csv, verifications.csv, summary.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/39_rag_test_once.py
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import FINAL_PIPELINE
from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs
from src.evaluation.rag_metrics import (
    GOLD_MAP, context_grades, score_answer, split_v1_citations, summarize_answers, verification_agreement,
)
from src.generation.context_builder import TOP_K_CONTEXT
from src.generation.llm import LocalLLM, answer_query
from src.generation.rag_v2 import DEFAULT_PROMPT_VERSION, RAGv2
from src.retrieval.pipeline import TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data/evaluation/frozen_rag_v2.json"
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.3.csv"
OUT_DIR = ROOT / "data/evaluation/results/rag_test"


def main() -> None:
    if OUT_DIR.exists():
        raise FileExistsError(f"{OUT_DIR} exists: the RAG test evaluation runs only once.")
    frozen = json.loads(FROZEN.read_text())
    changed = [p for p, h in frozen["sha256"].items()
               if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h]
    if changed or frozen["rag_v2"]["prompt_version"] != DEFAULT_PROMPT_VERSION:
        raise RuntimeError(f"Frozen RAG configuration changed: {changed}")
    device = frozen["device"]

    specs = load_query_specs(SPECS)
    test = sorted(q for q, s in specs.items() if s.split == "test")
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    grade_map = {(q, n): int(g) for q, n, g in zip(qrels.query_id, qrels.nct_id, qrels.grade)}
    label_map = {(q, n): (GOLD_MAP[d], GOLD_MAP[i]) for q, n, d, i in
                 zip(qrels.query_id, qrels.nct_id, qrels.disease_match, qrels.intervention_match)}

    engine = TrialSearchEngine(device=device, **FINAL_PIPELINE)
    records = engine.trials.drop(columns=["document"]).set_index("nct_id").to_dict("index")
    llm = LocalLLM(device=device)
    rag = RAGv2(llm)

    outputs, v1_rows, v2_rows, verification_rows = [], [], [], []
    for query_id in test:
        spec = specs[query_id]
        trials = engine.search(spec.text, top_k=TOP_K_CONTEXT).trials
        grades = context_grades(spec, trials, grade_map, records)

        v1 = answer_query(spec.text, trials, llm)
        match, partial = split_v1_citations(v1["answer"], v1["says_no_match"])
        v1_rows.append(score_answer(query_id, spec.has_answer, grades, v1["cited"], match, partial,
                                    v1["says_no_match"], v1["latency_s"]))

        v2 = rag.run(spec.text, trials)
        by_grade = {v["nct_id"]: v["verified_grade"] for v in v2["verifications"]}
        v2_rows.append(score_answer(query_id, spec.has_answer, grades, v2["cited"],
                                    [n for n in v2["cited"] if by_grade.get(n) == 2],
                                    [n for n in v2["cited"] if by_grade.get(n) == 1],
                                    v2["says_no_match"], v2["latency_s"])
                       | {"cited_unverified": len(v2["cited_unverified"])})
        for v in v2["verifications"]:
            gold = label_map.get((query_id, v["nct_id"])) if spec.has_answer else None
            verification_rows.append({
                "query_id": query_id, "nct_id": v["nct_id"],
                "model_disease": v["disease_match"], "model_intervention": v["intervention_match"],
                "model_grade": str(v["model_grade"]), "verified_grade": str(v["verified_grade"]),
                "quote_found": v["quote_found"], "dropped": v["dropped"], "valid_labels": v["valid_labels"],
                "gold_disease": gold[0] if gold else None, "gold_intervention": gold[1] if gold else None,
                "gold_grade": None if grades[v["nct_id"]] is None else str(grades[v["nct_id"]]),
                "evidence": v["evidence"],
            })
        outputs.append({"query_id": query_id, "query": spec.text, "has_answer": spec.has_answer,
                        "context": [{"nct_id": t["nct_id"], "title": t["title"],
                                     "grade_for_checking": grades[t["nct_id"]]} for t in trials],
                        "rag_v1": {k: v1[k] for k in ("answer", "cited", "says_no_match", "latency_s")},
                        "rag_v2": {k: v2[k] for k in ("concepts", "answer", "cited", "cited_unverified",
                                                      "says_no_match", "latency_s")}
                                  | {"verifications": [{k: x[k] for k in x if k != "raw"}
                                                       for x in v2["verifications"]]}})
        print(f"{query_id}: v1 no_match={v1['says_no_match']} cited={len(v1['cited'])} | "
              f"v2 no_match={v2['says_no_match']} cited={len(v2['cited'])} "
              f"verified={[(v['nct_id'][-4:], v['verified_grade']) for v in v2['verifications']]}")

    verifications = pd.DataFrame(verification_rows)
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "split": "test", "queries": test,
        "frozen_hashes_verified": True, "rag_v2_prompt_version": DEFAULT_PROMPT_VERSION, "qrels": QRELS.name,
        "rag_v1": summarize_answers(v1_rows),
        "rag_v2": summarize_answers(v2_rows) | {
            "cited_unverified": int(pd.DataFrame(v2_rows)["cited_unverified"].sum()),
            "verifications": len(verifications),
            "quotes_not_found": int((~verifications["quote_found"]).sum()),
            "verifications_dropped": int(verifications["dropped"].sum()),
        },
        "verification_agreement": verification_agreement(verifications),
    }

    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "outputs.json").write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + "\n")
    pd.concat([pd.DataFrame(v1_rows).assign(version="v1"), pd.DataFrame(v2_rows).assign(version="v2")]).to_csv(
        OUT_DIR / "answers_per_query.csv", index=False)
    verifications.to_csv(OUT_DIR / "verifications.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    keys = ["citations", "citation_validity", "match_cited", "match_relevant_share", "match_unjudged",
            "partial_cited", "partial_relevant_share", "no_match_answers", "no_match_correct", "no_match_false",
            "queries_without_relevant_in_context", "claims_without_relevant", "grade2_not_cited",
            "median_latency_s"]
    print("\n" + pd.DataFrame({"v1": summary["rag_v1"], "v2": summary["rag_v2"]}).loc[keys].to_string())
    print("\nv2 quote check:", {k: summary["rag_v2"][k] for k in ("verifications", "quotes_not_found",
                                                                   "verifications_dropped", "cited_unverified")})
    for name, stats in summary["verification_agreement"].items():
        print(f"\nAgreement {name}: n={stats['n']} accuracy={stats['accuracy']:.3f} kappa={stats['cohen_kappa']}")
        print(pd.DataFrame(stats["confusion (rows = labeler, columns = model)"]).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
