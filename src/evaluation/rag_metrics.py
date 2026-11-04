"""
Scoring for RAG answers (v1 and v2) and for RAG v2's per-trial verification.

Context grades come from qrels; a trial that violates the query's structured
constraints, or any trial of a zero-answer query, is grade 0; an unjudged
trial is None.

Answer measures (per query):
    citation validity  - cited NCT IDs that were in the context
    match / partial    - cited trials presented as full matches / partial
                         matches, and how many of them have grade >= 1
    no match           - the answer presents no trial as a (partial) match;
                         correct if no context trial has grade >= 1, false otherwise
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.evaluation.retrieval_metrics import constraint_violations


GOLD_MAP = {"exact": "exact", "broader": "broader", "class_level": "class_level",
            "absent_or_different": "absent", "not_applicable": "not_applicable"}


def context_grades(spec, trials: list[dict], grade_map: dict, records: dict) -> dict[str, int | None]:
    grades = {}
    for trial in trials:
        nct_id = trial["nct_id"]
        if not spec.has_answer or constraint_violations(records[nct_id], spec.structured_constraints):
            grades[nct_id] = 0
        else:
            grades[nct_id] = grade_map.get((spec.query_id, nct_id))
    return grades


def split_v1_citations(answer: str, says_no_match: bool) -> tuple[list[str], list[str]]:
    """RAG v1: (presented as matches, presented as partial matches); none if 'no match'."""
    ids = lambda text: list(dict.fromkeys(re.findall(r"NCT\d{8}", text)))
    if says_no_match:
        return [], []
    parts = re.split(r"partial matches?\s*:?", answer, maxsplit=1, flags=re.IGNORECASE)
    main = ids(parts[0])
    partial = [n for n in (ids(parts[1]) if len(parts) > 1 else []) if n not in main]
    return main, partial


def score_answer(query_id: str, has_answer: bool, grades: dict[str, int | None], cited: list[str],
                 as_match: list[str], as_partial: list[str], says_no_match: bool,
                 latency_s: float) -> dict[str, Any]:
    relevant = [n for n, g in grades.items() if g is not None and g >= 1]
    rel = lambda ids: sum((grades.get(n) or 0) >= 1 for n in ids)
    unjudged = lambda ids: sum(n in grades and grades[n] is None for n in ids)
    return {
        "query_id": query_id, "has_answer": has_answer,
        "n_cited": len(cited), "invalid_citations": sum(n not in grades for n in cited),
        "match_cited": len(as_match), "match_relevant": rel(as_match), "match_unjudged": unjudged(as_match),
        "partial_cited": len(as_partial), "partial_relevant": rel(as_partial),
        "says_no_match": says_no_match,
        "relevant_in_context": len(relevant),
        "no_match_correct": says_no_match and not relevant,
        "no_match_false": says_no_match and bool(relevant),
        "claims_without_relevant": (not says_no_match) and bool(as_match) and not relevant,
        "grade2_not_cited": sum(g == 2 and n not in cited for n, g in grades.items()),
        "latency_s": latency_s,
    }


def summarize_answers(rows: list[dict]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    cited = int(frame["n_cited"].sum())
    match = int(frame["match_cited"].sum())
    partial = int(frame["partial_cited"].sum())
    return {
        "queries": len(frame),
        "citations": cited,
        "citation_validity": (cited - int(frame["invalid_citations"].sum())) / cited if cited else None,
        "match_cited": match,
        "match_relevant_share": int(frame["match_relevant"].sum()) / match if match else None,
        "match_unjudged": int(frame["match_unjudged"].sum()),
        "partial_cited": partial,
        "partial_relevant_share": int(frame["partial_relevant"].sum()) / partial if partial else None,
        "no_match_answers": int(frame["says_no_match"].sum()),
        "no_match_correct": int(frame["no_match_correct"].sum()),
        "no_match_false": int(frame["no_match_false"].sum()),
        "queries_without_relevant_in_context": int((frame["relevant_in_context"] == 0).sum()),
        "claims_without_relevant": int(frame["claims_without_relevant"].sum()),
        "grade2_not_cited": int(frame["grade2_not_cited"].sum()),
        "median_latency_s": float(frame["latency_s"].median()),
        "mean_latency_s": float(frame["latency_s"].mean()),
    }


def verification_agreement(rows: pd.DataFrame) -> dict[str, Any]:
    """
    rows: one per (query, context trial) with columns model_disease, model_intervention,
    model_grade, verified_grade, gold_disease, gold_intervention, gold_grade (gold may be NaN).
    """
    out: dict[str, Any] = {}
    for name, model_col, gold_col in [("disease", "model_disease", "gold_disease"),
                                      ("intervention", "model_intervention", "gold_intervention"),
                                      ("grade_before_quote_check", "model_grade", "gold_grade"),
                                      ("grade_after_quote_check", "verified_grade", "gold_grade")]:
        data = rows.dropna(subset=[gold_col])
        if name == "intervention":
            data = data[data[gold_col] != "not_applicable"]
        if data.empty:
            continue
        model = data[model_col].astype(str)
        gold = data[gold_col].astype(str)
        out[name] = {
            "n": int(len(data)),
            "accuracy": float((model == gold).mean()),
            "cohen_kappa": float(cohen_kappa_score(gold, model)) if gold.nunique() > 1 or model.nunique() > 1 else None,
            "confusion (rows = labeler, columns = model)": pd.crosstab(gold, model).to_dict(),
        }
    return out
