"""
Grounded RAG answers for 5 DEV queries (Q01, Q11, Q24, Q03, Q20).

Retrieval: TrialSearchEngine (parser v1.1, RRF), top-5 trials as context.
Generation: Qwen3-4B-Instruct-2507, greedy decoding (src/generation).
For checking only (not shown to the model), each context trial gets its
relevance grade from qrels v1.2; Q24 is a verified zero-answer query.

Outputs (never overwritten): data/evaluation/results/rag_v1_dev/
    rag_outputs.json, rag_outputs.md

Run from the project root:
    PYTHONPATH=$PWD python experiments/29_rag_dev_examples.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs
from src.generation.context_builder import SYSTEM_PROMPT, TOP_K_CONTEXT
from src.generation.llm import DEFAULT_MODEL, LocalLLM, answer_query
from src.retrieval.pipeline import TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
OUT_DIR = ROOT / "data/evaluation/results/rag_v1_dev"
QUERIES = ["Q01", "Q11", "Q24", "Q03", "Q20"]


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    assert all(specs[q].split == "dev" for q in QUERIES)
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    grade = {(q, n): int(g) for q, n, g in zip(qrels.query_id, qrels.nct_id, qrels.grade)}

    engine = TrialSearchEngine(parser_version="v1.1")
    llm = LocalLLM(device=engine.device)

    outputs = []
    for query_id in QUERIES:
        spec = specs[query_id]
        result = engine.search(spec.text, top_k=TOP_K_CONTEXT)
        generated = answer_query(spec.text, result.trials, llm)

        def label(nct_id: str) -> str:
            if not spec.has_answer:
                return "0 (verified zero-answer query)"
            value = grade.get((query_id, nct_id))
            return "unjudged" if value is None else str(value)

        outputs.append({
            "query_id": query_id, "query": spec.text,
            "parsed_filters": result.filters, "semantic_query": result.semantic_query,
            "context": [{"nct_id": t["nct_id"], "title": t["title"], "status": t["status"],
                         "phase": t["phase"], "grade_for_checking": label(t["nct_id"])}
                        for t in result.trials],
            **generated,
            "cited_grades": {n: label(n) for n in generated["cited"]},
        })
        print(f"\n=== {query_id}: {spec.text}\n{generated['answer']}\n"
              f"cited={generated['cited']} invalid={generated['invalid_citations']} "
              f"no_match={generated['says_no_match']} ({generated['latency_s']} s)")

    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": DEFAULT_MODEL,
               "decoding": "greedy, max_new_tokens=350", "retrieval": "parser v1.1 + RRF, top-5",
               "system_prompt": SYSTEM_PROMPT, "outputs": outputs}
    (OUT_DIR / "rag_outputs.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = ["# RAG outputs on 5 dev queries", "",
             f"Model: `{DEFAULT_MODEL}` (greedy). Retrieval: parser v1.1 + RRF, top-5 as context. "
             "Grades come from qrels v1.2 and were NOT shown to the model.", ""]
    for o in outputs:
        lines += [f"## {o['query_id']}: {o['query']}", "",
                  f"Parsed filters: `{json.dumps(o['parsed_filters'])}`", "",
                  "| # | NCT ID | Title | Status | Phase | Grade (for checking) |",
                  "|---|---|---|---|---|---|"]
        for i, c in enumerate(o["context"], start=1):
            lines.append(f"| {i} | {c['nct_id']} | {c['title'][:80]} | {c['status']} | {c['phase']} | {c['grade_for_checking']} |")
        lines += ["", "**Answer:**", "", o["answer"], "",
                  f"Cited: {o['cited']} · grades of cited trials: {o['cited_grades']} · "
                  f"invalid citations: {o['invalid_citations']} · says no match: {o['says_no_match']} · "
                  f"{o['latency_s']} s", ""]
    (OUT_DIR / "rag_outputs.md").write_text("\n".join(lines) + "\n")
    print(f"\nSaved {OUT_DIR}")


if __name__ == "__main__":
    main()
