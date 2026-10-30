"""
M1 Step 3 - exhaustive verification of zero-answer candidate queries.

For each query with zero_answer_check == "pending" in query_specs_v1.json:
  1. keep only trials that satisfy the query's structured constraints
     (src.evaluation.retrieval_metrics.constraint_violations);
  2. search all processed and raw-only fields (same field set and text
     normalization as Experiment 19) for each core concept at several tiers:
       exact    - the query concept or a synonym / brand / code name
       broader  - a broader population (disease) that may include the query's
       class    - an UNNAMED drug class (intervention); named class members
                  are "different drugs" under rubric R3 and are not searched
       related  - context terms worth a human look, never enough on their own
  3. a trial is a POSSIBLE ANSWER if every core concept has an exact or
     broader/class match somewhere (before any judgment of mention role).

Verdicts:
  verified_zero  - no possible answer among constraint-satisfying trials
  needs_review   - at least one possible answer; the user decides

This script only reports; it does not edit query_specs_v1.json.

Run from the project root:
    PYTHONPATH=$PWD python experiments/20_verify_zero_answer_queries.py
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import load_query_specs
from src.evaluation.retrieval_metrics import constraint_violations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/clinical_trials_raw.json"
PARQUET_PATH = PROJECT_ROOT / "data/processed/trials.parquet"
SPECS_PATH = PROJECT_ROOT / "data/evaluation/query_specs_v1.json"

EVIDENCE_OUT = PROJECT_ROOT / "data/evaluation/zero_answer_check_v1_evidence.csv"
SUMMARY_OUT = PROJECT_ROOT / "data/evaluation/zero_answer_check_v1_summary.json"

# Reuse Experiment 19's field extraction, normalization and context flags.
_spec = importlib.util.spec_from_file_location(
    "exp19", PROJECT_ROOT / "experiments/19_corpus_freeze_and_q01_ceiling.py"
)
exp19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp19)

D = r"[\s\-]*"

# query_id -> concept -> tier -> list of (variant, pattern)
CONCEPT_PATTERNS = {
    "Q03": {
        "disease": {
            "exact": [("Alzheimer", r"alzheimer")],
            "broader": [("dementia", r"dementia"),
                        ("neurodegenerative disease", rf"neuro{D}degenerat")],
            "related": [("cognitive impairment / decline",
                         r"cognitive\s+(?:impairment|decline|disorder)|\bMCI\b"),
                        ("amyloid", r"amyloid")],
        },
    },
    "Q24": {
        "disease": {
            "exact": [("Alzheimer", r"alzheimer")],
            "broader": [("dementia", r"dementia")],
            "related": [("cognitive impairment", r"cognitive\s+impairment|\bMCI\b")],
        },
        "intervention": {
            "exact": [("lecanemab", r"lecanemab"), ("Leqembi", r"leqembi"),
                      ("BAN2401", rf"ban{D}2401")],
            "class": [("anti-amyloid antibody (unnamed)",
                       rf"anti{D}amyloid|amyloid{D}(?:beta|β)?\s*(?:directed\s+)?"
                       r"(?:monoclonal\s+)?antibod")],
            "related": [("other named anti-amyloid antibody",
                         r"aducanumab|donanemab|gantenerumab|solanezumab|crenezumab")],
        },
    },
    "Q25": {
        "disease": {
            "exact": [("hypertrophic cardiomyopathy",
                       rf"hypertrophic{D}(?:obstructive\s+)?cardiomyopath|\bo?HCM\b")],
            "broader": [("cardiomyopathy (any)", r"cardiomyopath")],
        },
        "intervention": {
            "exact": [("mavacamten", r"mavacamten"), ("Camzyos", r"camzyos"),
                      ("MYK-461", rf"myk{D}461")],
            "class": [("cardiac myosin inhibitor (unnamed)",
                       rf"(?:cardiac\s+)?myosin{D}inhibit")],
            "related": [("other named myosin inhibitor", r"aficamten|ck{D}274")],
        },
    },
    "Q26": {
        "intervention": {
            "exact": [("tirzepatide", r"tirzepatide"), ("Mounjaro", r"mounjaro"),
                      ("Zepbound", r"zepbound"), ("LY3298176", rf"ly{D}3298176")],
            "class": [("dual GIP/GLP-1 agonist (unnamed)",
                       rf"gip{D}/?{D}glp|glp{D}1{D}/{D}gip|dual{D}(?:incretin|agonist)"
                       r"|twincretin")],
            "related": [("GLP-1 receptor agonist", rf"glp{D}1")],
        },
    },
}


def search_trial(nct_id: str, fields: dict[str, str], concepts: dict) -> list[dict]:
    rows = []
    for concept, tiers in concepts.items():
        for tier, variants in tiers.items():
            for variant, pattern in variants:
                compiled = re.compile(pattern, re.IGNORECASE)
                for field, text in fields.items():
                    for found in compiled.finditer(text):
                        start, end = found.span()
                        segment = exp19.segment_around(text, start, end)
                        rows.append({
                            "nct_id": nct_id,
                            "concept": concept,
                            "tier": tier,
                            "variant": variant,
                            "field": field,
                            "matched_text": found.group(0),
                            "snippet": exp19.snippet_around(text, start, end),
                            "in_exclusion_section": field == "eligibility_exclusion",
                            "negation_cues": "; ".join(sorted({
                                m.group(0).lower()
                                for m in exp19.NEGATION_CUE.finditer(segment)
                            })),
                            "prior_therapy_cues": "; ".join(sorted({
                                m.group(0).lower()
                                for m in exp19.PRIOR_THERAPY_CUE.finditer(segment)
                            })),
                        })
    return rows


def main() -> None:
    for path in (EVIDENCE_OUT, SUMMARY_OUT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    specs = load_query_specs(SPECS_PATH)
    raw_specs = {item["query_id"]: item for item in json.loads(SPECS_PATH.read_text())}
    pending = [q for q, item in raw_specs.items() if item["zero_answer_check"] == "pending"]
    if set(pending) != set(CONCEPT_PATTERNS):
        raise ValueError(f"Pattern table {sorted(CONCEPT_PATTERNS)} != pending {pending}")

    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(PARQUET_PATH).set_index("nct_id")
    records = metadata.to_dict("index")
    fields_by_trial = {r["nct_id"]: exp19.raw_fields(r["study"]) for r in raw}

    all_rows, summary = [], {}

    for query_id in pending:
        spec = specs[query_id]
        concepts = CONCEPT_PATTERNS[query_id]
        required = [c for c in ("disease", "intervention")
                    if spec.core_concepts.get(c) is not None]
        if sorted(required) != sorted(concepts):
            raise ValueError(f"{query_id}: patterns do not cover concepts {required}")

        eligible = [
            nct for nct in metadata.index
            if not constraint_violations(records[nct], spec.structured_constraints)
        ]

        rows = []
        for nct_id in eligible:
            rows.extend(search_trial(nct_id, fields_by_trial[nct_id], concepts))
        frame = pd.DataFrame(rows)

        # A strong match must be exact/broader/class, outside exclusion
        # criteria, without a negation cue, and not only in MeSH ancestors
        # (ancestor terms such as "Neurodegenerative Diseases" do not show
        # that the trial population includes the query condition).
        possible, flagged_only, any_mention = [], [], []
        if not frame.empty:
            frame["clean_strong"] = (
                frame["tier"].isin(["exact", "broader", "class"])
                & ~frame["in_exclusion_section"]
                & (frame["negation_cues"] == "")
                & (frame["field"] != "condition_mesh_ancestors")
            )
            for nct_id, group in frame.groupby("nct_id"):
                any_mention.append(nct_id)
                strong = group[group["tier"].isin(["exact", "broader", "class"])]
                clean = group[group["clean_strong"]]
                if set(clean["concept"]) >= set(required):
                    possible.append(nct_id)
                elif set(strong["concept"]) >= set(required):
                    flagged_only.append(nct_id)

        # Context: concept matches among trials that FAIL the constraints.
        exact_any_status = sum(
            1 for nct_id, fields in fields_by_trial.items()
            if all(
                any(re.search(p, " ".join(fields.values()), re.IGNORECASE)
                    for _, p in concepts[c]["exact"])
                for c in required
            )
        )

        verdict = "needs_review" if possible else "verified_zero"
        summary[query_id] = {
            "text": spec.text,
            "structured_constraints": spec.structured_constraints,
            "trials_satisfying_constraints": len(eligible),
            "trials_with_any_concept_mention": len(any_mention),
            "possible_answers": possible,
            "flagged_only_matches": flagged_only,
            "trials_with_exact_match_for_all_concepts_ignoring_constraints": exact_any_status,
            "verdict": verdict,
            "patterns": {c: {t: dict(v) for t, v in tiers.items()}
                         for c, tiers in concepts.items()},
        }

        if not frame.empty:
            frame.insert(0, "query_id", query_id)
            frame["possible_answer"] = frame["nct_id"].isin(possible)
            frame["flagged_only"] = frame["nct_id"].isin(flagged_only)
            frame["status"] = frame["nct_id"].map(metadata["status"])
            frame["phase"] = frame["nct_id"].map(metadata["phase"])
            frame["title"] = frame["nct_id"].map(metadata["title"])
            all_rows.append(frame)

        print(f"\n{query_id}: {spec.text}")
        print(f"  constraint-satisfying trials: {len(eligible)}")
        print(f"  with any concept mention    : {len(any_mention)}")
        print(f"  exact match for all concepts, any status/phase/age: {exact_any_status}")
        print(f"  possible answers            : {possible or 'none'}")
        print(f"  matches only in exclusion/negated/ancestor context: {len(flagged_only)}")
        print(f"  verdict                     : {verdict}")

    evidence = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    evidence.to_csv(EVIDENCE_OUT, index=False)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"\nSaved: {EVIDENCE_OUT}\nSaved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
