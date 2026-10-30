"""
Step 1 of Milestone M1.

Part A - Corpus freeze (corpus v1):
    Record hashes, fetch parameters, fetch-date evidence and corpus
    statistics in data/corpus_v1_manifest.json, and verify the read-only
    backup of the raw JSON.

Part B - Q01 ceiling verification:
    Find every RECRUITING (and, separately, NOT_YET_RECRUITING) trial that
    could be an NSCLC + pembrolizumab trial, searching processed AND
    raw-only fields with variant-robust patterns. Every match is saved with
    its field, matched text, a ~20-word snippet and heuristic context flags
    (exclusion section, negation cue, prior-therapy cue).

    This script does NOT assign relevance labels.

Nothing existing is modified. Output files are never overwritten.

Run from the project root:
    PYTHONPATH=$PWD python experiments/19_corpus_freeze_and_q01_ceiling.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing.processor import extract_trial


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = PROJECT_ROOT / "data/raw/clinical_trials_raw.json"
PARQUET_PATH = PROJECT_ROOT / "data/processed/trials.parquet"
BUILD_SCRIPT = PROJECT_ROOT / "experiments/02_build_corpus.py"
API_CLIENT = PROJECT_ROOT / "src/ingestion/clinicaltrials_api.py"
PROCESSOR = PROJECT_ROOT / "src/preprocessing/processor.py"

BACKUP_PATH = Path(
    "/mnt/drive/home/melange/clinical_rag/backups/"
    "clinical-research-intelligence/corpus_v1/clinical_trials_raw.json"
)

MANIFEST_PATH = PROJECT_ROOT / "data/corpus_v1_manifest.json"

EVAL_DIR = PROJECT_ROOT / "data/evaluation"
TRIALS_OUT = EVAL_DIR / "q01_ceiling_v1_trials.csv"
EVIDENCE_OUT = EVAL_DIR / "q01_ceiling_v1_evidence.csv"
SUMMARY_OUT = EVAL_DIR / "q01_ceiling_v1_summary.json"

FUSION_CANDIDATES = EVAL_DIR / "weighted_fusion_candidates.csv"
RERANKED_CANDIDATES = EVAL_DIR / "reranked_candidates.csv"

ORIGINAL_POSITIVES = {"NCT06195527", "NCT07061379"}
STATUSES_OF_INTEREST = ["RECRUITING", "NOT_YET_RECRUITING"]


# =============================================================
# Shared helpers
# =============================================================

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def refuse_overwrite(*paths: Path) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing outputs: {existing}"
        )


# =============================================================
# Part A - Corpus freeze
# =============================================================

def read_build_constants() -> dict:
    """Read fetch parameters from 02_build_corpus.py without executing it."""

    tree = ast.parse(BUILD_SCRIPT.read_text(encoding="utf-8"))
    constants = {}

    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {
                "DOMAIN_QUERIES",
                "TARGET_PER_DOMAIN",
            }:
                constants[target.id] = ast.literal_eval(node.value)

    # Call-site arguments that are not module constants.
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    constants["page_size"] = int(
        re.search(r"page_size\s*=\s*(\d+)", source).group(1)
    )
    constants["timeout"] = int(
        re.search(r"timeout\s*=\s*(\d+)", source).group(1)
    )
    constants["max_retries"] = int(
        re.search(r"max_retries\s*=\s*(\d+)", source).group(1)
    )
    return constants


def reconstruct_domain_statistics(raw: list, domain_order: list) -> dict:
    """
    Rebuild 02_build_corpus.py's per-domain statistics from source_domains.

    source_domains lists every domain whose query returned the trial, in
    fetch order, so source_domains[0] is the domain that first added it.
    """

    stats = {}
    for domain in domain_order:
        returned = sum(domain in r["source_domains"] for r in raw)
        new_unique = sum(r["source_domains"][0] == domain for r in raw)
        stats[domain] = {
            "returned_by_query": returned,
            "new_unique": new_unique,
            "duplicates": returned - new_unique,
            "only_this_domain": sum(
                r["source_domains"] == [domain] for r in raw
            ),
        }
    return stats


def verify_parquet_reproducible(raw: list, parquet: pd.DataFrame) -> bool:
    """Re-run the current processor in memory and compare to trials.parquet."""

    rebuilt = pd.DataFrame([extract_trial(trial) for trial in raw])
    rebuilt = rebuilt.astype(parquet.dtypes.to_dict())
    try:
        pd.testing.assert_frame_equal(
            rebuilt.reset_index(drop=True),
            parquet.reset_index(drop=True),
        )
        return True
    except AssertionError as error:
        print(f"Parquet differs from processor output: {error}")
        return False


def build_manifest(raw: list, parquet: pd.DataFrame) -> dict:
    constants = read_build_constants()
    domain_order = list(constants["DOMAIN_QUERIES"])

    raw_sha = sha256_of(RAW_PATH)
    backup_sha = sha256_of(BACKUP_PATH) if BACKUP_PATH.exists() else None

    version_holders = Counter(
        r["study"]["derivedSection"]["miscInfoModule"].get("versionHolder")
        for r in raw
    )

    stats = reconstruct_domain_statistics(raw, domain_order)

    return {
        "corpus_version": "v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "created_by": "experiments/19_corpus_freeze_and_q01_ceiling.py",
        "files": {
            "raw_json": {
                "path": str(RAW_PATH.relative_to(PROJECT_ROOT)),
                "sha256": raw_sha,
                "size_bytes": RAW_PATH.stat().st_size,
                "record_count": len(raw),
                "tracked_in_git": False,
            },
            "processed_parquet": {
                "path": str(PARQUET_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_of(PARQUET_PATH),
                "size_bytes": PARQUET_PATH.stat().st_size,
                "row_count": len(parquet),
                "columns": parquet.columns.tolist(),
                "unique_nct_ids": int(parquet["nct_id"].nunique()),
                "row_order_matches_raw": (
                    parquet["nct_id"].tolist()
                    == [r["nct_id"] for r in raw]
                ),
                "reproducible_from_raw_with_current_processor": (
                    verify_parquet_reproducible(raw, parquet)
                ),
                "tracked_in_git": True,
            },
        },
        "backup": {
            "path": str(BACKUP_PATH),
            "sha256": backup_sha,
            "matches_raw": backup_sha == raw_sha,
            "permissions": "file 0444, directory 0555",
            "note": (
                "Same physical disk (/mnt/drive) as the project: protects "
                "against accidental deletion/overwrite, not disk failure."
            ),
        },
        "fetch_parameters": {
            "source": "experiments/02_build_corpus.py (read via ast/regex)",
            "api_endpoint": "https://clinicaltrials.gov/api/v2/studies",
            "query_parameter": "query.cond",
            "domain_queries": constants["DOMAIN_QUERIES"],
            "max_studies_per_domain": constants["TARGET_PER_DOMAIN"],
            "page_size": constants["page_size"],
            "timeout_seconds": constants["timeout"],
            "max_retries": constants["max_retries"],
            "sort": "not specified (API default ordering)",
            "fields": "not specified (full study records)",
            "other_filters": "none",
            "deduplication": "by NCT ID, first domain in fetch order wins",
            "caveat": (
                "Parameters are read from the current script. There is no "
                "record proving this exact script version performed the "
                "fetch; its mtime precedes the raw file mtime by ~3 minutes."
            ),
            "script_sha256": {
                "experiments/02_build_corpus.py": sha256_of(BUILD_SCRIPT),
                "src/ingestion/clinicaltrials_api.py": sha256_of(API_CLIENT),
                "src/preprocessing/processor.py": sha256_of(PROCESSOR),
            },
        },
        "fetch_date": {
            "recorded": False,
            "statement": (
                "The fetch date was not recorded anywhere in the project "
                "(no logs, no timestamp in the JSON). Evidence below."
            ),
            "evidence": {
                "ctgov_data_version_holder": dict(version_holders),
                "raw_json_mtime_utc": utc_time(RAW_PATH.stat().st_mtime),
                "build_script_mtime_utc": utc_time(
                    BUILD_SCRIPT.stat().st_mtime
                ),
                "processed_parquet_mtime_utc": utc_time(
                    PARQUET_PATH.stat().st_mtime
                ),
                "max_last_update_post_date": parquet["last_update"].max(),
                "server_file_birth_note": (
                    "Files were created on this server on 2026-10-12 with "
                    "preserved mtimes, i.e. copied from another machine. "
                    "Original timezone of the mtime is unknown."
                ),
            },
            "best_estimate": (
                "On or after 2026-10-07 (ClinicalTrials.gov data version) "
                "and most likely 2026-10-08 (raw file mtime). Not verified."
            ),
        },
        "statistics": {
            "per_domain_fetch": stats,
            "fetched_total": sum(
                s["returned_by_query"] for s in stats.values()
            ),
            "duplicates_removed": sum(
                s["duplicates"] for s in stats.values()
            ),
            "domain_combinations": {
                "+".join(key): value
                for key, value in Counter(
                    tuple(sorted(r["source_domains"])) for r in raw
                ).most_common()
            },
            "status": parquet["status"].value_counts(dropna=False).to_dict(),
            "study_type": (
                parquet["study_type"].value_counts(dropna=False).to_dict()
            ),
        },
        "known_issues": [
            "torch 2.14.1 is a CUDA 13.0 build while faiss-gpu 1.12.0 is a "
            "CUDA 12.8 build. FAISS is not in the current pipeline; re-test "
            "GPU FAISS if it is used again.",
            "Corpus is a sample: at most 1000 trials per domain query, API "
            "default order. Do not expand v1; expansion is a later, "
            "separately versioned step.",
        ],
    }


# =============================================================
# Part B - Q01 ceiling verification
# =============================================================

DASH = r"[\s\-]*"

# Each concept tier is a list of (variant_name, pattern).
PATTERNS = {
    "disease_explicit": [
        ("NSCLC", r"nsclc"),
        ("non-small cell", rf"non{DASH}small{DASH}cell"),
    ],
    "disease_subtype": [
        ("lung adenocarcinoma",
         r"(?:lung|pulmonary)\s+adenocarcinoma"
         r"|adenocarcinoma\s+of\s+(?:the\s+)?lung"),
        ("lung squamous cell carcinoma",
         rf"squamous{DASH}cell\s+(?:carcinoma|cancer)\s+of\s+(?:the\s+)?lung"
         rf"|(?:lung|pulmonary)\s+squamous"
         rf"|squamous(?:{DASH}cell)?\s+lung"),
        ("large cell lung carcinoma",
         rf"large{DASH}cell\s+(?:lung\s+)?(?:carcinoma|cancer)"),
    ],
    "disease_broader": [
        ("lung cancer (unspecified)",
         r"lung\s+(?:cancer|carcinoma|neoplasm|tumou?r|malignan)"),
        ("solid tumor", r"solid\s+(?:tumou?r|malignan|neoplasm|cancer)"),
    ],
    "drug_explicit": [
        ("pembrolizumab", r"pembrolizumab"),
        ("lambrolizumab (former name)", r"lambrolizumab"),
        ("Keytruda", r"keytruda"),
        ("MK-3475", rf"mk{DASH}3475"),
        ("SCH 900475", rf"sch{DASH}900475"),
        ("pembro (abbreviation)", r"\bpembro\b"),
    ],
    "drug_class": [
        ("anti-PD-(L)1", rf"anti{DASH}pd{DASH}l?{DASH}1"),
        ("PD-(L)1 inhibitor/blockade",
         rf"pd{DASH}l?{DASH}1\s+(?:inhibit|blockade|block|antibod|therap)"),
        ("immune checkpoint", rf"(?:immune{DASH})?checkpoint\s+inhibit"),
        ("immunotherapy", r"immuno[\s\-]*therap"),
    ],
}

COMPILED = {
    tier: [(name, re.compile(pattern, re.IGNORECASE))
           for name, pattern in variants]
    for tier, variants in PATTERNS.items()
}

NEGATION_CUE = re.compile(
    r"\b(?:no|not|without|exclud\w*|ineligible|prohibit\w*|disallow\w*"
    r"|except|other\s+than)\b",
    re.IGNORECASE,
)
PRIOR_THERAPY_CUE = re.compile(
    r"\b(?:prior|previous(?:ly)?|history\s+of|pre-?treated|received"
    r"|refractory|resistant|progress\w*|fail\w*|after)\b",
    re.IGNORECASE,
)
SEGMENT_BREAK = re.compile(r"\n|\*|•|(?<=[.;])\s")
EXCLUSION_HEADER = re.compile(r"exclusion\s+criteria", re.IGNORECASE)

# Fields that also appear in the processed trial document used by
# BM25 / dense / cross-encoder. Everything else is raw-only.
PROCESSED_FIELDS = {
    "brief_title",
    "conditions",
    "intervention_names",
    "brief_summary",
    "eligibility_inclusion",
    "eligibility_exclusion",
    "eligibility_unsectioned",
}


def normalize(text: str) -> str:
    """NFKC-normalize and map Unicode dashes / no-break spaces to ASCII."""

    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[‐-―−]", "-", text)
    return text.replace(" ", " ")


def raw_fields(study: dict) -> dict[str, str]:
    """Collect searchable fields, including raw-only ones, from one record."""

    protocol = study["protocolSection"]
    derived = study.get("derivedSection", {})
    ident = protocol.get("identificationModule", {})
    description = protocol.get("descriptionModule", {})
    conditions = protocol.get("conditionsModule", {})
    arms_module = protocol.get("armsInterventionsModule", {})
    interventions = arms_module.get("interventions", [])
    arms = arms_module.get("armGroups", [])

    def mesh_terms(module: str, key: str) -> str:
        return " | ".join(
            item["term"] for item in derived.get(module, {}).get(key, [])
        )

    eligibility = protocol.get("eligibilityModule", {}).get(
        "eligibilityCriteria", ""
    )
    header = EXCLUSION_HEADER.search(eligibility)
    if header:
        inclusion, exclusion, unsectioned = (
            eligibility[: header.start()],
            eligibility[header.start():],
            "",
        )
    else:
        inclusion, exclusion, unsectioned = "", "", eligibility

    fields = {
        "brief_title": ident.get("briefTitle", ""),
        "official_title": ident.get("officialTitle", ""),
        "acronym": ident.get("acronym", ""),
        "conditions": " | ".join(conditions.get("conditions", [])),
        "keywords": " | ".join(conditions.get("keywords", [])),
        "condition_mesh": mesh_terms("conditionBrowseModule", "meshes"),
        "condition_mesh_ancestors": mesh_terms(
            "conditionBrowseModule", "ancestors"
        ),
        "intervention_names": " | ".join(
            item.get("name", "") for item in interventions
        ),
        "intervention_other_names": " | ".join(
            name for item in interventions
            for name in item.get("otherNames", [])
        ),
        "intervention_descriptions": " | ".join(
            item.get("description", "") for item in interventions
        ),
        "intervention_mesh": mesh_terms("interventionBrowseModule", "meshes"),
        "arm_labels": " | ".join(arm.get("label", "") for arm in arms),
        "arm_descriptions": " | ".join(
            arm.get("description", "") for arm in arms
        ),
        "brief_summary": description.get("briefSummary", ""),
        "detailed_description": description.get("detailedDescription", ""),
        "eligibility_inclusion": inclusion,
        "eligibility_exclusion": exclusion,
        "eligibility_unsectioned": unsectioned,
    }
    return {name: normalize(value) for name, value in fields.items()}


def segment_around(text: str, start: int, end: int) -> str:
    """Return the bullet / sentence containing a match."""

    seg_start, seg_end = 0, len(text)
    for boundary in SEGMENT_BREAK.finditer(text):
        if boundary.end() <= start:
            seg_start = boundary.end()
        elif boundary.start() >= end:
            seg_end = boundary.start()
            break
    return text[seg_start:seg_end]


def snippet_around(text: str, start: int, end: int, words: int = 10) -> str:
    before = " ".join(text[:start].split()[-words:])
    after = " ".join(text[end:].split()[:words])
    return f"{before} [[{text[start:end]}]] {after}".strip()


def find_matches(nct_id: str, fields: dict[str, str]) -> list[dict]:
    matches = []
    for tier, variants in COMPILED.items():
        for variant, pattern in variants:
            for field, text in fields.items():
                for found in pattern.finditer(text):
                    start, end = found.span()
                    segment = segment_around(text, start, end)
                    preceding = text[max(0, start - 4):start].lower()

                    negation = sorted({
                        cue.group(0).lower()
                        for cue in NEGATION_CUE.finditer(segment)
                    })
                    if preceding.endswith(("non-", "non ")) and tier.startswith("disease"):
                        negation.append("non- prefix")

                    prior = sorted({
                        cue.group(0).lower()
                        for cue in PRIOR_THERAPY_CUE.finditer(segment)
                    })

                    matches.append({
                        "nct_id": nct_id,
                        "tier": tier,
                        "variant": variant,
                        "field": field,
                        "field_in_processed_document": field in PROCESSED_FIELDS,
                        "matched_text": found.group(0),
                        "snippet": snippet_around(text, start, end),
                        "in_exclusion_section": field == "eligibility_exclusion",
                        "negation_cues": "; ".join(negation),
                        "prior_therapy_cues": "; ".join(prior),
                    })
    return matches


def is_clean(match: dict) -> bool:
    """Heuristic: not in exclusion criteria and no negation cue."""

    return not match["in_exclusion_section"] and not match["negation_cues"]


def summarize_trial(nct_id: str, matches: list[dict]) -> dict:
    by_tier = {tier: [m for m in matches if m["tier"] == tier] for tier in PATTERNS}

    def fields_of(items):
        return "; ".join(sorted({m["field"] for m in items}))

    def variants_of(items):
        return "; ".join(sorted({m["variant"] for m in items}))

    disease = by_tier["disease_explicit"] + by_tier["disease_subtype"]
    drug = by_tier["drug_explicit"]

    return {
        "nct_id": nct_id,
        "disease_explicit": bool(by_tier["disease_explicit"]),
        "disease_subtype_only": (
            not by_tier["disease_explicit"] and bool(by_tier["disease_subtype"])
        ),
        "disease_broader": bool(by_tier["disease_broader"]),
        "drug_explicit": bool(drug),
        "drug_class_mention": bool(by_tier["drug_class"]),
        "disease_has_clean_match": any(is_clean(m) for m in disease),
        "drug_has_clean_match": any(is_clean(m) for m in drug),
        "drug_all_matches_prior_therapy_context": (
            bool(drug) and all(m["prior_therapy_cues"] for m in drug)
        ),
        "disease_fields": fields_of(disease),
        "disease_variants": variants_of(disease),
        "drug_fields": fields_of(drug),
        "drug_variants": variants_of(drug),
        "broader_disease_variants": variants_of(by_tier["disease_broader"]),
        "drug_class_variants": variants_of(by_tier["drug_class"]),
        "drug_only_in_raw_fields": (
            bool(drug) and not any(m["field_in_processed_document"] for m in drug)
        ),
        "disease_only_in_raw_fields": (
            bool(disease)
            and not any(m["field_in_processed_document"] for m in disease)
        ),
    }


def match_group(row: pd.Series) -> str:
    disease = row["disease_explicit"] or row["disease_subtype_only"]
    if disease and row["drug_explicit"]:
        return "both" if row["disease_explicit"] else "both_via_subtype"
    if row["drug_explicit"]:
        return "drug_and_broader_disease" if row["disease_broader"] else "drug_only"
    if disease:
        return "disease_and_drug_class" if row["drug_class_mention"] else "disease_only"
    return "none"


def existing_ranks() -> pd.DataFrame:
    """Q01 ranks from Exp 16 (all fusion configs) and Exp 17 (reranker)."""

    fusion = pd.read_csv(FUSION_CANDIDATES)
    fusion = fusion[fusion["query_id"] == "Q01"]
    ranks = fusion.pivot(index="nct_id", columns="config", values="rank")
    ranks.columns = [f"exp16_{column}_rank" for column in ranks.columns]

    reranked = pd.read_csv(RERANKED_CANDIDATES)
    reranked = reranked[reranked["query_id"] == "Q01"].set_index("nct_id")
    ranks["exp17_cross_encoder_rank"] = reranked["reranked_rank"]
    return ranks


def legacy_audit_match(fields: dict[str, str]) -> tuple[bool, bool]:
    """Reproduce the audit's earlier substring check for comparison."""

    text = " ".join(fields.values())
    disease = re.search(
        r"nsclc|non[- ]?small[- ]?cell lung|carcinoma, non-small-cell lung",
        text, re.IGNORECASE,
    )
    drug = re.search(r"pembrolizumab|keytruda|mk-?3475", text, re.IGNORECASE)
    return bool(disease), bool(drug)


def run_q01_ceiling(raw: list, parquet: pd.DataFrame) -> dict:
    status_of = dict(zip(parquet["nct_id"], parquet["status"]))
    title_of = dict(zip(parquet["nct_id"], parquet["title"]))
    conditions_of = dict(zip(parquet["nct_id"], parquet["conditions"]))

    evidence_rows, trial_rows, legacy = [], [], {}

    for record in raw:
        nct_id = record["nct_id"]
        fields = raw_fields(record["study"])
        matches = find_matches(nct_id, fields)
        legacy[nct_id] = legacy_audit_match(fields)

        summary = summarize_trial(nct_id, matches)
        summary.update(
            status=status_of[nct_id],
            title=title_of[nct_id],
            conditions=conditions_of[nct_id],
        )
        trial_rows.append(summary)
        evidence_rows.extend(matches)

    trials = pd.DataFrame(trial_rows)
    trials["match_group"] = trials.apply(match_group, axis=1)
    evidence = pd.DataFrame(evidence_rows)

    # ---------------- counts ----------------
    def counts(frame: pd.DataFrame) -> dict:
        disease = frame["disease_explicit"] | frame["disease_subtype_only"]
        drug = frame["drug_explicit"]
        clean_both = (
            frame["disease_has_clean_match"] & frame["drug_has_clean_match"]
        )
        return {
            "trials": len(frame),
            "disease_explicit": int(frame["disease_explicit"].sum()),
            "disease_explicit_or_subtype": int(disease.sum()),
            "drug_explicit": int(drug.sum()),
            "both_explicit_disease": int(
                (frame["disease_explicit"] & drug).sum()
            ),
            "both_including_subtype": int((disease & drug).sum()),
            "both_with_clean_evidence_for_each": int(
                (disease & drug & clean_both).sum()
            ),
            "drug_and_broader_disease_only": int(
                (frame["match_group"] == "drug_and_broader_disease").sum()
            ),
            "disease_and_drug_class_only": int(
                (frame["match_group"] == "disease_and_drug_class").sum()
            ),
        }

    by_status = {
        status: counts(trials[trials["status"] == status])
        for status in STATUSES_OF_INTEREST
    }
    by_status["ALL_STATUSES"] = counts(trials)

    recruiting_ids = trials.loc[trials["status"] == "RECRUITING", "nct_id"]
    legacy_counts = {
        "disease": int(sum(legacy[n][0] for n in recruiting_ids)),
        "drug": int(sum(legacy[n][1] for n in recruiting_ids)),
        "both": int(sum(all(legacy[n]) for n in recruiting_ids)),
    }

    # ---------------- per-trial output ----------------
    in_scope = trials[
        trials["status"].isin(STATUSES_OF_INTEREST)
        & (trials["match_group"] != "none")
    ].copy()

    ranks = existing_ranks()
    in_scope = in_scope.join(ranks, on="nct_id")
    in_scope["original_q01_positive"] = in_scope["nct_id"].isin(ORIGINAL_POSITIVES)

    group_order = {
        "both": 0, "both_via_subtype": 1, "drug_and_broader_disease": 2,
        "drug_only": 3, "disease_and_drug_class": 4, "disease_only": 5,
    }
    in_scope = in_scope.sort_values(
        ["status", "match_group", "nct_id"],
        key=lambda column: column.map(group_order) if column.name == "match_group" else column,
    )

    leading = ["nct_id", "status", "match_group", "original_q01_positive", "title", "conditions"]
    in_scope = in_scope[leading + [c for c in in_scope.columns if c not in leading]]

    scope_evidence = evidence[evidence["nct_id"].isin(in_scope["nct_id"])]

    return {
        "trials": in_scope,
        "evidence": scope_evidence,
        "counts": by_status,
        "legacy_audit_counts_recruiting": legacy_counts,
    }


# =============================================================
# Main
# =============================================================

def main() -> None:
    refuse_overwrite(MANIFEST_PATH, TRIALS_OUT, EVIDENCE_OUT, SUMMARY_OUT)

    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    parquet = pd.read_parquet(PARQUET_PATH)

    # ---------------- Part A ----------------
    manifest = build_manifest(raw, parquet)

    if not manifest["backup"]["matches_raw"]:
        raise RuntimeError("Backup hash does not match the raw JSON.")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("PART A - CORPUS v1 MANIFEST")
    print("=" * 80)
    print(f"raw sha256     : {manifest['files']['raw_json']['sha256']}")
    print(f"backup sha256  : {manifest['backup']['sha256']} "
          f"(match={manifest['backup']['matches_raw']})")
    print(f"parquet sha256 : {manifest['files']['processed_parquet']['sha256']}")
    print("parquet reproducible from raw:",
          manifest["files"]["processed_parquet"]
          ["reproducible_from_raw_with_current_processor"])
    print("per-domain:", json.dumps(manifest["statistics"]["per_domain_fetch"], indent=1))
    print(f"Saved: {MANIFEST_PATH}")

    # ---------------- Part B ----------------
    result = run_q01_ceiling(raw, parquet)

    summary = {
        "corpus_version": "v1",
        "corpus_parquet_sha256": manifest["files"]["processed_parquet"]["sha256"],
        "query": "Recruiting NSCLC trials involving pembrolizumab",
        "method": {
            "text_normalization": "NFKC; Unicode dashes -> '-'; NBSP -> space",
            "matching": "case-insensitive regex search (no word-boundary "
                        "requirement except for 'pembro'), so attached "
                        "punctuation such as '(NSCLC),' or 'KEYTRUDA®' matches",
            "patterns": PATTERNS,
            "fields_searched": sorted(raw_fields(raw[0]["study"]).keys()),
            "fields_in_processed_document": sorted(PROCESSED_FIELDS),
            "eligibility_split": "text before the first 'Exclusion Criteria' "
                                 "header = inclusion, from it onward = exclusion",
            "context_flags": {
                "negation_cues": NEGATION_CUE.pattern,
                "prior_therapy_cues": PRIOR_THERAPY_CUE.pattern,
                "scope": "the bullet/sentence containing the match",
                "caveat": "heuristic flags for human review, not labels",
            },
        },
        "counts": result["counts"],
        "legacy_audit_counts_recruiting": result["legacy_audit_counts_recruiting"],
        "outputs": [str(TRIALS_OUT.relative_to(PROJECT_ROOT)),
                    str(EVIDENCE_OUT.relative_to(PROJECT_ROOT))],
    }

    result["trials"].to_csv(TRIALS_OUT, index=False)
    result["evidence"].to_csv(EVIDENCE_OUT, index=False)
    SUMMARY_OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("PART B - Q01 CEILING VERIFICATION")
    print("=" * 80)
    print(json.dumps(result["counts"], indent=1))
    print("Legacy audit method (RECRUITING):", result["legacy_audit_counts_recruiting"])
    print(f"\nSaved: {TRIALS_OUT}\nSaved: {EVIDENCE_OUT}\nSaved: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
