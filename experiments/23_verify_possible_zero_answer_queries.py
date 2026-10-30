"""
M1 Step 4b - exhaustive zero-answer check for Q14, Q16, Q20.

Same method as Experiment 20 (Q03, Q24-Q26), reusing its search function:
  * only trials satisfying the query's structured constraints;
  * all processed and raw-only fields (Experiment 19 field set);
  * tiers: exact, broader (a disease category containing the query
    condition, rubric v1.1 R2), related (context only);
  * POSSIBLE ANSWER = every core concept has an exact/broader match that is
    not in exclusion criteria, has no negation cue, and is not only in MeSH
    ancestors. FLAGGED ONLY = such matches exist only in those contexts.

Verdict: verified_zero if there is no possible answer; otherwise
needs_review, with the evidence saved for each possible answer.

Run from the project root:
    PYTHONPATH=$PWD python experiments/23_verify_possible_zero_answer_queries.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import load_query_specs
from src.evaluation.retrieval_metrics import constraint_violations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/clinical_trials_raw.json"
PARQUET_PATH = PROJECT_ROOT / "data/processed/trials.parquet"
SPECS_PATH = PROJECT_ROOT / "data/evaluation/query_specs_v1.json"

EVIDENCE_OUT = PROJECT_ROOT / "data/evaluation/zero_answer_check_v1_1_evidence.csv"
SUMMARY_OUT = PROJECT_ROOT / "data/evaluation/zero_answer_check_v1_1_summary.json"


def load_experiment(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp19 = load_experiment("exp19", "19_corpus_freeze_and_q01_ceiling.py")
exp20 = load_experiment("exp20", "20_verify_zero_answer_queries.py")

D = r"[\s\-]*"

CONCEPT_PATTERNS = {
    "Q14": {
        "disease": {
            "exact": [("HFrEF", r"hfref"),
                      ("reduced ejection fraction", r"reduced\s+ejection\s+fraction"),
                      ("systolic heart failure", r"systolic\s+heart\s+failure"),
                      ("LVEF threshold <= 40%",
                       r"(?:lvef|ejection\s+fraction)\s*(?:of\s*)?(?:≤|<=|<|less\s+than|below|under)"
                       r"\s*(?:3\d|40)\s*%")],
            "broader": [("heart failure (unspecified ejection fraction)", r"heart\s+failure")],
            "related": [("HFpEF / HFmrEF",
                         r"hfpef|hfmref|preserved\s+ejection|mildly\s+reduced\s+ejection")],
        },
    },
    "Q16": {
        "disease": {
            "exact": [("amyotrophic lateral sclerosis", r"amyotrophic\s+lateral\s+sclerosis"),
                      ("ALS", r"\bals\b"), ("Lou Gehrig's disease", r"lou\s+gehrig")],
            "broader": [("motor neuron disease", r"motor\s+neur(?:on|one)\s+disease"),
                        ("neuromuscular disease", r"neuromuscular\s+disease"),
                        ("neurodegenerative disease", rf"neuro{D}degenerative\s+disease")],
            "related": [("other motor neuron disorders",
                         r"spinal\s+muscular\s+atrophy|primary\s+lateral\s+sclerosis|frontotemporal")],
        },
    },
    "Q20": {
        "disease": {
            "exact": [("Parkinson's disease", r"parkinson(?!ism)")],
            "broader": [("parkinsonism (mixed)", r"parkinsonism"),
                        ("movement disorders", r"movement\s+disorder"),
                        ("neurodegenerative disease", rf"neuro{D}degenerative\s+disease")],
            "related": [("atypical parkinsonism",
                         r"progressive\s+supranuclear|multiple\s+system\s+atrophy|lewy\s+bod")],
        },
    },
}


def main() -> None:
    for path in (EVIDENCE_OUT, SUMMARY_OUT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    specs = load_query_specs(SPECS_PATH)
    metadata = pd.read_parquet(PARQUET_PATH).set_index("nct_id")
    records = metadata.to_dict("index")
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    fields_by_trial = {r["nct_id"]: exp19.raw_fields(r["study"]) for r in raw}

    summary, frames = {}, []
    for query_id, concepts in CONCEPT_PATTERNS.items():
        spec = specs[query_id]
        required = [c for c in ("disease", "intervention") if spec.core_concepts.get(c) is not None]
        if sorted(required) != sorted(concepts):
            raise ValueError(f"{query_id}: patterns do not cover concepts {required}")

        eligible = [n for n in metadata.index
                    if not constraint_violations(records[n], spec.structured_constraints)]
        frame = pd.DataFrame([row for n in eligible
                              for row in exp20.search_trial(n, fields_by_trial[n], concepts)])

        possible, flagged_only, any_mention = [], [], []
        if not frame.empty:
            frame["clean_strong"] = (
                frame["tier"].isin(["exact", "broader"])
                & ~frame["in_exclusion_section"]
                & (frame["negation_cues"] == "")
                & (frame["field"] != "condition_mesh_ancestors")
            )
            for nct_id, group in frame.groupby("nct_id"):
                any_mention.append(nct_id)
                strong = group[group["tier"].isin(["exact", "broader"])]
                if set(group.loc[group["clean_strong"], "concept"]) >= set(required):
                    possible.append(nct_id)
                elif set(strong["concept"]) >= set(required):
                    flagged_only.append(nct_id)

        # Context: exact matches regardless of the structured constraints.
        exact_any = sorted(
            n for n, fields in fields_by_trial.items()
            if not exp20.search_trial(n, {k: v for k, v in fields.items()
                                          if k != "condition_mesh_ancestors"},
                                      {c: {"exact": t["exact"]} for c, t in concepts.items()}) == []
        )
        exact_any_status = metadata.loc[exact_any, ["status", "phase"]].astype(str).agg(" / ".join, axis=1)

        summary[query_id] = {
            "text": spec.text,
            "structured_constraints": spec.structured_constraints,
            "trials_satisfying_constraints": len(eligible),
            "trials_with_any_concept_mention": len(any_mention),
            "possible_answers": possible,
            "flagged_only_matches": flagged_only,
            "trials_with_exact_match_any_constraints": len(exact_any),
            "exact_match_trials_status_phase": exact_any_status.value_counts().to_dict(),
            "verdict": "needs_review" if possible else "verified_zero",
            "patterns": {c: {t: dict(v) for t, v in tiers.items()} for c, tiers in concepts.items()},
        }
        if not frame.empty:
            frame.insert(0, "query_id", query_id)
            frame["possible_answer"] = frame["nct_id"].isin(possible)
            frame["flagged_only"] = frame["nct_id"].isin(flagged_only)
            frame["status"] = frame["nct_id"].map(metadata["status"])
            frame["phase"] = frame["nct_id"].map(metadata["phase"])
            frame["title"] = frame["nct_id"].map(metadata["title"])
            frames.append(frame)

        print(f"\n{query_id}: {spec.text}  constraints={spec.structured_constraints}")
        print(f"  constraint-satisfying trials       : {len(eligible)}")
        print(f"  with any concept mention           : {len(any_mention)}")
        print(f"  exact match, any status/phase      : {len(exact_any)} -> "
              f"{summary[query_id]['exact_match_trials_status_phase']}")
        print(f"  possible answers                   : {possible or 'none'}")
        print(f"  flagged-only (exclusion/negated/MeSH ancestor): {len(flagged_only)}")
        print(f"  verdict                            : {summary[query_id]['verdict']}")

    pd.concat(frames, ignore_index=True).to_csv(EVIDENCE_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {EVIDENCE_OUT}\nSaved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
