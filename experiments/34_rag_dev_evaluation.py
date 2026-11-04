"""
Small RAG evaluation on DEV: the 15 answerable dev queries + Q24 (zero-answer).

Retrieval: FINAL_PIPELINE (src/config.py), top-5 as context. Generation and
prompt unchanged since Experiment 29 (Qwen3-4B-Instruct-2507, greedy).

Automatic measures (grades from qrels v1.2, never shown to the model):
  * citation validity  - every cited NCT ID is in the context
  * citation relevance - share of cited trials with grade >= 1, separately
                         for trials presented as matches and as "Partial
                         matches"; unjudged citations counted separately
  * no-match handling  - correct "no match" (no grade >= 1 trial in the
                         context), false "no match" (there was one), and
                         answers that claim matches when none exist
  * missed grade-2     - grade-2 trials in the context that were not cited

Outputs (never overwritten): data/evaluation/results/rag_eval_v1_dev/
    rag_outputs.json, rag_outputs.md, per_query.csv, summary.json

Run from the project root:
    PYTHONPATH=$PWD python experiments/34_rag_dev_evaluation.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import FINAL_GENERATION, FINAL_PIPELINE
from src.evaluation.judgments import expand_shared_judgments, load_qrels, load_query_specs
from src.evaluation.retrieval_metrics import constraint_violations
from src.generation.context_builder import SYSTEM_PROMPT, TOP_K_CONTEXT
from src.generation.llm import LocalLLM, answer_query
from src.retrieval.pipeline import TrialSearchEngine


ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
QRELS = ROOT / "data/evaluation/qrels_v1.2.csv"
OUT_DIR = ROOT / "data/evaluation/results/rag_eval_v1_dev"


def split_citations(answer: str, says_no_match: bool) -> tuple[list[str], list[str], list[str]]:
    """(presented as matches, presented as partial matches, mentioned in a no-match answer)."""
    ids = lambda text: list(dict.fromkeys(re.findall(r"NCT\d{8}", text)))
    if says_no_match:
        return [], [], ids(answer)
    parts = re.split(r"partial matches?\s*:?", answer, maxsplit=1, flags=re.IGNORECASE)
    main = ids(parts[0])
    partial = [n for n in ids(parts[1])] if len(parts) > 1 else []
    return main, [n for n in partial if n not in main], []


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs = load_query_specs(SPECS)
    queries = sorted(q for q, s in specs.items() if s.split == "dev")       # 15 answerable + Q24
    qrels = expand_shared_judgments(load_qrels(QRELS, specs), specs)
    grade = {(q, n): int(g) for q, n, g in zip(qrels.query_id, qrels.nct_id, qrels.grade)}

    engine = TrialSearchEngine(**FINAL_PIPELINE)
    records = engine.trials.drop(columns=["document"]).set_index("nct_id").to_dict("index")
    llm = LocalLLM(device=engine.device)

    outputs, rows = [], []
    for query_id in queries:
        spec = specs[query_id]
        result = engine.search(spec.text, top_k=TOP_K_CONTEXT)
        generated = answer_query(spec.text, result.trials, llm)

        def g(nct_id: str):
            if not spec.has_answer:
                return 0
            if constraint_violations(records[nct_id], spec.structured_constraints):
                return 0
            return grade.get((query_id, nct_id))                         # None = unjudged

        context_grades = {t["nct_id"]: g(t["nct_id"]) for t in result.trials}
        main_ids, partial_ids, mentioned = split_citations(generated["answer"], generated["says_no_match"])
        relevant_in_context = [n for n, v in context_grades.items() if v is not None and v >= 1]
        unjudged_in_context = [n for n, v in context_grades.items() if v is None]

        row = {
            "query_id": query_id, "has_answer": spec.has_answer,
            "n_cited": len(generated["cited"]), "invalid_citations": len(generated["invalid_citations"]),
            "match_cited": len(main_ids),
            "match_cited_relevant": sum((context_grades.get(n) or 0) >= 1 for n in main_ids),
            "match_cited_unjudged": sum(context_grades.get(n, 0) is None for n in main_ids),
            "partial_cited": len(partial_ids),
            "partial_cited_relevant": sum((context_grades.get(n) or 0) >= 1 for n in partial_ids),
            "says_no_match": generated["says_no_match"],
            "relevant_in_context": len(relevant_in_context),
            "unjudged_in_context": len(unjudged_in_context),
            "grade2_not_cited": [n for n, v in context_grades.items() if v == 2 and n not in generated["cited"]],
            "latency_s": generated["latency_s"],
        }
        row["no_match_correct"] = generated["says_no_match"] and not relevant_in_context
        row["no_match_false"] = generated["says_no_match"] and bool(relevant_in_context)
        row["claims_match_without_relevant"] = (not generated["says_no_match"] and bool(main_ids)
                                                and not relevant_in_context)
        rows.append(row)
        outputs.append({"query_id": query_id, "query": spec.text, "filters": result.filters,
                        "context": [{"nct_id": t["nct_id"], "title": t["title"], "conditions": t["conditions"],
                                     "interventions": t["interventions"], "status": t["status"],
                                     "phase": t["phase"], "grade_for_checking": context_grades[t["nct_id"]]}
                                    for t in result.trials],
                        "answer": generated["answer"], "cited_as_match": main_ids,
                        "cited_as_partial": partial_ids, "mentioned_in_no_match": mentioned, **row})
        print(f"{query_id}: cited={generated['cited']} no_match={generated['says_no_match']} "
              f"relevant_in_ctx={len(relevant_in_context)}")

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT_DIR / "per_query.csv", index=False)
    cited_total = int(frame["n_cited"].sum())
    match_total = int(frame["match_cited"].sum())
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": FINAL_PIPELINE, "generation": FINAL_GENERATION,
        "queries": len(frame),
        "citations_total": cited_total,
        "citation_validity": 1 - frame["invalid_citations"].sum() / cited_total if cited_total else None,
        "cited_as_match_total": match_total,
        "cited_as_match_relevant_share": frame["match_cited_relevant"].sum() / match_total if match_total else None,
        "cited_as_match_unjudged": int(frame["match_cited_unjudged"].sum()),
        "cited_as_partial_total": int(frame["partial_cited"].sum()),
        "cited_as_partial_relevant": int(frame["partial_cited_relevant"].sum()),
        "no_match_answers": int(frame["says_no_match"].sum()),
        "no_match_correct": int(frame["no_match_correct"].sum()),
        "no_match_false": int(frame["no_match_false"].sum()),
        "queries_without_relevant_in_context": int((frame["relevant_in_context"] == 0).sum()),
        "claims_match_without_relevant": int(frame["claims_match_without_relevant"].sum()),
        "grade2_in_context_not_cited": int(frame["grade2_not_cited"].map(len).sum()),
        "median_latency_s": float(frame["latency_s"].median()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    (OUT_DIR / "rag_outputs.json").write_text(json.dumps(
        {"system_prompt": SYSTEM_PROMPT, "outputs": outputs}, indent=2, ensure_ascii=False, default=str) + "\n")

    lines = ["# RAG evaluation on dev (15 answerable + Q24)", "",
             f"Pipeline {FINAL_PIPELINE}; model {FINAL_GENERATION['model']}; grades from qrels v1.2 "
             "(not shown to the model; None = unjudged).", ""]
    for o in outputs:
        lines += [f"## {o['query_id']}: {o['query']}", "", "| NCT ID | Title | Grade |", "|---|---|---|"]
        lines += [f"| {c['nct_id']} | {c['title'][:90]} | {c['grade_for_checking']} |" for c in o["context"]]
        lines += ["", o["answer"], "",
                  f"As match: {o['cited_as_match']} · partial: {o['cited_as_partial']} · "
                  f"no-match mention: {o['mentioned_in_no_match']} · says no match: {o['says_no_match']}", ""]
    (OUT_DIR / "rag_outputs.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
