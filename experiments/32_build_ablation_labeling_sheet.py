"""
Labeling sheet for unjudged dev trials in the top-10 of ablation configs C
and D (Experiment 31). Zero-answer queries are skipped (their trials are
grade 0 by verification). Same bias rules as the main sheet: no system,
rank or score shown; rows shuffled within each query with a fixed seed;
neutral evidence snippets; locked not_applicable cells.

Outputs (never overwritten), in data/evaluation/ablation_labels_v1/:
    labeling_sheet_ablation_v1.xlsx, qrels_ablation_v1_template.csv,
    ablation_v1_key.csv (which config retrieved each trial - NOT for the labeler)

Run from the project root:
    PYTHONPATH=$PWD python experiments/32_build_ablation_labeling_sheet.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.judgments import QRELS_COLUMNS, load_query_specs


ROOT = Path(__file__).resolve().parents[1]
UNJUDGED = ROOT / "data/evaluation/results/doc_tokenizer_ablation_dev/unjudged_top10.csv"
SPECS = ROOT / "data/evaluation/query_specs_v1.1.json"
RAW = ROOT / "data/raw/clinical_trials_raw.json"
OUT_DIR = ROOT / "data/evaluation/ablation_labels_v1"
SEED = 20261104
MAX_ROWS = 40


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {OUT_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exp19 = load("exp19", "19_corpus_freeze_and_q01_ceiling.py")
    exp22 = load("exp22", "22_build_pools_and_labeling_sheet.py")

    specs = load_query_specs(SPECS)
    unjudged = pd.read_csv(UNJUDGED)
    unjudged = unjudged[unjudged["system"].isin(["C", "D"])]
    unjudged = unjudged[unjudged["query_id"].map(lambda q: specs[q].has_answer)]
    key = (unjudged.groupby(["query_id", "nct_id"])
           .agg(configs=("system", lambda s: ",".join(sorted(set(s)))), best_rank=("rank", "min"))
           .reset_index())
    if len(key) > MAX_ROWS:
        raise RuntimeError(f"{len(key)} rows exceed the limit of {MAX_ROWS}")

    studies = {r["nct_id"]: r["study"] for r in json.loads(RAW.read_text(encoding="utf-8"))}
    rng = np.random.default_rng(SEED)
    sheet_rows, template, key_rows = [], [], []
    for query_id in sorted(key["query_id"].unique()):
        spec = specs[query_id]
        disease, intervention = spec.core_concepts.get("disease"), spec.core_concepts.get("intervention")
        terms = exp22.QUERY_TERMS[query_id]
        nct_ids = sorted(key.loc[key["query_id"] == query_id, "nct_id"])
        for i, idx in enumerate(rng.permutation(len(nct_ids)), start=1):
            nct_id = nct_ids[idx]
            row_id = f"A{query_id}-{i:02d}"
            fields = exp19.raw_fields(studies[nct_id])
            d_val = "" if disease else "not_applicable"
            i_val = "" if intervention else "not_applicable"
            sheet_rows.append({
                "row_id": row_id, "query_id": query_id, "NCT ID": nct_id,
                "link": f"https://clinicaltrials.gov/study/{nct_id}",
                "disease_match": d_val, "intervention_match": i_val, "mention_role": "", "notes": "",
                "query": spec.text,
                "disease concept": exp22.concept_label(disease),
                "intervention concept": exp22.concept_label(intervention),
                "title": fields["brief_title"], "official title": fields["official_title"],
                "conditions": fields["conditions"],
                "interventions (with other names)": exp22.interventions_text(studies[nct_id]),
                "disease evidence (where search terms occur)": (
                    exp22.evidence_text(exp22.find_terms(fields, terms["disease"]))
                    if "disease" in terms else "not applicable"),
                "intervention evidence (where search terms occur)": (
                    exp22.evidence_text(exp22.find_terms(fields, terms["intervention"]))
                    if "intervention" in terms else "not applicable"),
                "brief summary": exp22.cap(fields["brief_summary"]),
                "eligibility criteria": exp22.cap(studies[nct_id]["protocolSection"]
                                                  .get("eligibilityModule", {}).get("eligibilityCriteria", "")),
            })
            template.append({"row_id": row_id, "query_id": query_id, "nct_id": nct_id,
                             "disease_match": d_val, "intervention_match": i_val,
                             "mention_role": "", "notes": "", "labeler": "", "label_date": ""})
            match = key[(key.query_id == query_id) & (key.nct_id == nct_id)].iloc[0]
            key_rows.append({"row_id": row_id, "query_id": query_id, "nct_id": nct_id,
                             "configs": match["configs"], "best_rank": int(match["best_rank"])})

    exp22.INSTRUCTIONS = (
        "ABLATION LABELING (39-ish rows): new trials retrieved by improved configurations.\n"
        "Same rubric as before: docs/labeling_rubric_v1.1.md and docs/labeling_guide_v1.1.md.\n\n"
        + exp22.INSTRUCTIONS.replace("query set v1", "query set v1.1")
        .replace("docs/labeling_guide_v1.md", "docs/labeling_guide_v1.1.md")
        .replace("docs/labeling_rubric_v1.md", "docs/labeling_rubric_v1.1.md"))
    exp22.write_sheet(sheet_rows, specs, None, OUT_DIR / "labeling_sheet_ablation_v1.xlsx")
    pd.DataFrame(template, columns=["row_id"] + QRELS_COLUMNS).to_csv(
        OUT_DIR / "qrels_ablation_v1_template.csv", index=False)
    pd.DataFrame(key_rows).to_csv(OUT_DIR / "ablation_v1_key.csv", index=False)
    print(f"{len(sheet_rows)} rows:", pd.DataFrame(key_rows).groupby("query_id").size().to_dict())


if __name__ == "__main__":
    main()
