"""
M1 Step 4b - build judgment pools and the labeling sheet.

Pool per answerable query (has_answer = true):
  1. retrieval pool: union of the top-d trials of each run in data/runs/v1
     (both filter modes), d = 10 for rrf and rrf_ce_semq, 5 for bm25_ws,
     bm25_punct, dense and rrf_ce_raw;
  2. remove trials violating the query's structured constraints (they are
     grade 0 by code and are never labeled);
  3. add up to 3 concept candidates: constraint-satisfying trials with a
     clean core-term match for every core concept (not in exclusion
     criteria, no negation cue, not only in MeSH ancestors), strongest
     evidence first, ties broken by a seeded random key;
  4. paraphrase queries keep only trials not already pooled for the query
     whose judgments they reuse (judgments_shared_with);
  5. overrun rule if the total exceeds the budget: secondary lists 5 -> 3,
     largest pools first; then concept candidates capped at 1.

Outputs (never overwritten):
  data/evaluation/pool_v1_provenance.csv   which system/rank produced each trial
                                           (NOT for the labeler)
  data/evaluation/pool_v1_summary.json     sizes, settings, overrun decisions
  data/evaluation/qrels_v1_template.csv    Step 2 qrels format, labels blank
  data/evaluation/labeling_sheet_v1.xlsx   the labeling workbook

The sheet shows no system names, ranks or scores; trial order within each
query is shuffled with a fixed seed. Evidence cells list where search terms
occur, without saying which tier a term belongs to.

Run from the project root:
    PYTHONPATH=$PWD python experiments/22_build_pools_and_labeling_sheet.py
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.worksheet.datavalidation import DataValidation

from src.evaluation.judgments import QRELS_COLUMNS, load_query_specs
from src.evaluation.retrieval_metrics import constraint_violations


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/clinical_trials_raw.json"
PARQUET_PATH = PROJECT_ROOT / "data/processed/trials.parquet"
SPECS_PATH = PROJECT_ROOT / "data/evaluation/query_specs_v1.json"
RUN_DIR = PROJECT_ROOT / "data/runs/v1"

OUT_DIR = PROJECT_ROOT / "data/evaluation"
PROVENANCE_OUT = OUT_DIR / "pool_v1_provenance.csv"
SUMMARY_OUT = OUT_DIR / "pool_v1_summary.json"
TEMPLATE_OUT = OUT_DIR / "qrels_v1_template.csv"
SHEET_OUT = OUT_DIR / "labeling_sheet_v1.xlsx"

SEED = 20261030
BUDGET = 500
POOL_DEPTHS = {"rrf": 10, "rrf_ce_semq": 10,
               "bm25_ws": 5, "bm25_punct": 5, "dense": 5, "rrf_ce_raw": 5}
REDUCED_SECONDARY_DEPTH = 3
MAX_CONCEPT_CANDIDATES = 3
FILTER_MODES = ("parser_filters", "gold_filters")

_spec = importlib.util.spec_from_file_location(
    "exp19", PROJECT_ROOT / "experiments/19_corpus_freeze_and_q01_ceiling.py")
exp19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp19)


# =============================================================
# Search terms (case-insensitive regex). "core" terms select concept
# candidates; core + related terms are shown as evidence, unlabeled.
# =============================================================

D = r"[\s\-]*"
NOT_PRE = r"(?<!pre)(?<!pre-)"

TERMS = {
    "nsclc": {"core": [r"nsclc", rf"non{D}small{D}cell"],
              "related": [r"lung\s+(?:cancer|carcinoma|neoplasm|tumou?r)",
                          r"(?:lung|pulmonary)\s+adenocarcinoma|adenocarcinoma\s+of\s+(?:the\s+)?lung",
                          r"solid\s+tumou?r"]},
    "pembrolizumab": {"core": [r"pembrolizumab", r"keytruda", rf"mk{D}3475", r"lambrolizumab"],
                      "related": [rf"anti{D}pd{D}l?{D}1", rf"pd{D}l?{D}1\s+(?:inhibit|blockade|antibod)",
                                  rf"checkpoint{D}inhibit", r"immuno\s*therap"]},
    "breast_cancer": {"core": [r"breast\s+(?:cancer|carcinoma|neoplasm|tumou?r)", r"\btnbc\b",
                               r"mammary\s+carcinoma"],
                      "related": [r"solid\s+tumou?r"]},
    "alzheimer": {"core": [r"alzheimer", r"dementia"],
                  "related": [r"cognitive\s+(?:impairment|decline)", r"\bmci\b", rf"neuro{D}degenerat"]},
    "diabetes": {"core": [rf"{NOT_PRE}diabet", rf"\bt[12]{D}dm?\b"],
                 "related": [r"pre-?diabet", r"insulin\s+resistance", r"\bhba1c\b"]},
    "cancer_any": {"core": [r"cancer", r"carcinoma", r"tumou?r", r"neoplasm", r"malignan", r"lymphoma",
                            r"leuka?emia", r"melanoma", r"sarcoma", r"myeloma", r"glioblastoma|glioma"],
                   "related": []},
    "immunotherapy": {"core": [r"immuno\s*therap", rf"checkpoint{D}inhibit", rf"anti{D}pd{D}l?{D}1",
                               rf"anti{D}ctla{D}4",
                               r"nivolumab|pembrolizumab|atezolizumab|durvalumab|avelumab|cemiplimab"
                               r"|ipilimumab|tremelimumab|dostarlimab|tislelizumab|sintilimab"
                               r"|camrelizumab|toripalimab",
                               rf"\bcar{D}t\b", r"chimeric\s+antigen\s+receptor",
                               r"(?:cancer|tumou?r)\s+vaccine", r"bispecific", r"oncolytic"],
                      "related": [r"tumou?r[\s\-]infiltrating\s+lymphocyte", r"interleukin|\bil-?2\b",
                                  r"interferon", r"\bnk\s+cell"]},
    "cvd": {"core": [r"cardiovascular", r"heart\s+(?:disease|failure)", r"coronary",
                     r"myocardial\s+infarction", r"atrial\s+fibrillation", r"stroke", r"hypertension",
                     r"cardiomyopath", r"arrhythmi", r"atheroscler", r"angina"],
            "related": [r"peripheral\s+arter", r"venous\s+thrombo|thromboembol|deep\s+vein",
                        r"dyslipid|hypercholesterol", r"obes"]},
    "metastatic": {"core": [r"metasta"],
                   "related": [r"advanced", r"stage\s+iv\b", r"unresectable", r"locally\s+advanced"]},
    "solid_tumor": {"core": [r"solid\s+(?:tumou?r|malignan|neoplasm|cancer)", r"carcinoma", r"sarcoma",
                             r"melanoma", r"glioma",
                             r"(?:lung|breast|colorectal|colon|rectal|prostate|pancrea\w*|gastric|ovarian"
                             r"|liver|hepatocellular|kidney|renal|bladder|cervical|endometrial|thyroid"
                             r"|esophag\w*|head\s+and\s+neck)\s+cancer"],
                    "related": [r"lymphoma", r"leuka?emia", r"myeloma"]},
    "targeted_therapy": {"core": [r"targeted\s+therap", rf"kinase{D}inhibit", r"\btkis?\b",
                                  r"\b\w+(?:tinib|fenib|lisib|ciclib|rasib)\b",
                                  r"antibody[\s\-]drug\s+conjugate|\badcs?\b",
                                  r"\b\w+(?:tecan|vedotin|mertansine|emtansine)\b",
                                  r"monoclonal\s+antibod"],
                         "related": [r"\b\w+parib\b", r"radioligand|lutetium|\blu-?177\b"]},
    "infectious": {"core": [r"infect", r"\bhiv\b", r"tubercul", r"hepatitis", r"covid|sars-?cov-?2",
                            r"influenza", r"malaria", r"sepsis|septic", r"pneumonia",
                            r"bacter(?:ial|emia|aemia)", r"\bviral\b|virus", r"fungal", r"vaccin"],
                   "related": [r"long\s+covid|post-?covid"]},
    "lung_cancer": {"core": [r"lung\s+(?:cancer|carcinoma|neoplasm|tumou?r|adenocarcinoma)", r"nsclc",
                             r"\bsclc\b", rf"non{D}small{D}cell", rf"small{D}cell\s+lung"],
                    "related": [r"solid\s+tumou?r"]},
    "obesity": {"core": [r"obes", r"overweight", rf"weight{D}(?:loss|reduction|management)"],
                "related": [r"\bbmi\b", r"body\s+weight", r"type\s*(?:2|ii)\s+diabet|\bt2dm?\b"]},
    "semaglutide": {"core": [r"semaglutide", r"ozempic", r"wegovy", r"rybelsus"],
                    "related": [rf"glp{D}1", r"liraglutide|dulaglutide|tirzepatide|exenatide"]},
    "t2dm": {"core": [r"type\s*(?:2|ii)\s+diabet", rf"\bt2{D}dm?\b", r"non-?insulin[\s\-]dependent"],
             "related": [rf"{NOT_PRE}diabet"]},
    "sglt2": {"core": [rf"sglt{D}2", r"gliflozin"], "related": []},
    "hfref": {"core": [r"hfref", r"reduced\s+ejection\s+fraction", r"systolic\s+heart\s+failure"],
              "related": [r"heart\s+failure", r"hfpef|preserved\s+ejection", r"ejection\s+fraction"]},
    "afib": {"core": [r"atrial\s+fibrillation", r"\ba-?fib\b"],
             "related": [r"atrial\s+flutter", r"\baf\b"]},
    "anticoagulants": {"core": [r"anticoagul", r"warfarin", r"apixaban", r"rivaroxaban", r"dabigatran",
                                r"edoxaban", r"heparin", r"enoxaparin", r"\b[dn]oacs?\b",
                                r"vitamin\s+k\s+antagonist"],
                       "related": [r"antiplatelet|aspirin|clopidogrel|ticagrelor",
                                   r"left\s+atrial\s+appendage"]},
    "als": {"core": [r"amyotrophic\s+lateral\s+sclerosis", r"\bals\b"],
            "related": [r"motor\s+neur(?:on|one)\s+disease"]},
    "melanoma": {"core": [r"melanoma"], "related": [r"solid\s+tumou?r"]},
    "epilepsy": {"core": [r"epilep", r"dravet", r"lennox", r"infantile\s+spasm"],
                 "related": [r"seizure", r"convuls"]},
    "hiv": {"core": [r"\bhiv\b", r"human\s+immunodeficiency\s+virus"], "related": []},
    "prep": {"core": [r"pre-?\s*exposure\s+prophylaxis", r"\bprep\b"],
             "related": [r"emtricitabine|tenofovir|cabotegravir|lenacapavir|truvada|descovy|dapivirine",
                         r"post-?\s*exposure"]},
    "parkinson": {"core": [r"parkinson"],
                  "related": [r"multiple\s+system\s+atrophy|progressive\s+supranuclear"]},
    "tuberculosis": {"core": [r"tubercul", r"\btb\b"], "related": [r"non-?tuberculous"]},
}

QUERY_TERMS = {
    "Q01": {"disease": "nsclc", "intervention": "pembrolizumab"},
    "Q02": {"disease": "breast_cancer"},
    "Q03": {"disease": "alzheimer"},
    "Q04": {"disease": "diabetes"},
    "Q05": {"disease": "breast_cancer"},
    "Q06": {"disease": "cancer_any", "intervention": "immunotherapy"},
    "Q07": {"disease": "cvd"},
    "Q08": {"disease": "metastatic"},
    "Q09": {"disease": "solid_tumor", "intervention": "targeted_therapy"},
    "Q10": {"disease": "infectious"},
    "Q11": {"disease": "lung_cancer", "intervention": "pembrolizumab"},
    "Q12": {"disease": "obesity", "intervention": "semaglutide"},
    "Q13": {"disease": "t2dm", "intervention": "sglt2"},
    "Q14": {"disease": "hfref"},
    "Q15": {"disease": "afib", "intervention": "anticoagulants"},
    "Q16": {"disease": "als"},
    "Q17": {"disease": "melanoma"},
    "Q18": {"disease": "epilepsy"},
    "Q19": {"disease": "hiv", "intervention": "prep"},
    "Q20": {"disease": "parkinson"},
    "Q21": {"disease": "tuberculosis"},
    "Q22": {"disease": "nsclc", "intervention": "pembrolizumab"},
    "Q23": {"disease": "cvd"},
}

COMPILED = {
    name: {tier: [re.compile(p, re.IGNORECASE) for p in patterns]
           for tier, patterns in groups.items()}
    for name, groups in TERMS.items()
}

# Fields shown as evidence, in reading order (MeSH ancestors are omitted:
# they describe the MeSH hierarchy, not the trial).
FIELD_LABELS = {
    "brief_title": "Title",
    "official_title": "Official title",
    "acronym": "Acronym",
    "conditions": "Conditions",
    "keywords": "Keywords",
    "condition_mesh": "Condition MeSH",
    "intervention_names": "Interventions",
    "intervention_other_names": "Intervention other names",
    "intervention_mesh": "Intervention MeSH",
    "arm_labels": "Arm labels",
    "arm_descriptions": "Arm descriptions",
    "intervention_descriptions": "Intervention descriptions",
    "brief_summary": "Brief summary",
    "detailed_description": "Detailed description",
    "eligibility_inclusion": "Eligibility - inclusion criteria",
    "eligibility_unsectioned": "Eligibility (no inclusion/exclusion headers)",
    "eligibility_exclusion": "Eligibility - EXCLUSION criteria",
}
PRIOR_CUE = re.compile(
    r"\b(?:prior|previous(?:ly)?|history\s+of|pre-?treated|received|refractory\s+to"
    r"|resistant\s+to|failed|failure\s+of)\b", re.IGNORECASE)
MAX_SNIPPETS_PER_FIELD, MAX_SNIPPETS_PER_CONCEPT = 2, 8
EXCEL_CELL_LIMIT = 32000


# =============================================================
# Matching
# =============================================================

def find_terms(fields: dict[str, str], term_set: str, tiers=("core", "related")) -> list[dict]:
    hits = []
    for field in FIELD_LABELS:
        text = fields.get(field, "")
        if not text:
            continue
        for tier in tiers:
            for pattern in COMPILED[term_set][tier]:
                for found in pattern.finditer(text):
                    start, end = found.span()
                    segment = exp19.segment_around(text, start, end)
                    hits.append({
                        "field": field, "tier": tier, "start": start,
                        "snippet": exp19.snippet_around(text, start, end, words=12),
                        "exclusion": field == "eligibility_exclusion",
                        "negated": bool(exp19.NEGATION_CUE.search(segment)),
                        "prior": bool(PRIOR_CUE.search(segment)),
                    })
    return hits


def evidence_text(hits: list[dict]) -> str:
    """Neutral evidence cell: where terms occur, by field, with context tags."""
    if not hits:
        return "No search-term matches found in any field."
    lines, total = [], 0
    for field, label in FIELD_LABELS.items():
        seen, field_hits = set(), sorted(
            (h for h in hits if h["field"] == field), key=lambda h: h["start"])
        for hit in field_hits:
            if hit["snippet"] in seen:
                continue
            seen.add(hit["snippet"])
            tags = []
            if hit["exclusion"]:
                tags.append("in EXCLUSION criteria")
            if hit["prior"]:
                tags.append("prior-therapy wording nearby")
            tag = f"  ({'; '.join(tags)})" if tags else ""
            lines.append(f"- {label}: ...{hit['snippet']}...{tag}")
            total += 1
            if len(seen) >= MAX_SNIPPETS_PER_FIELD or total >= MAX_SNIPPETS_PER_CONCEPT:
                break
        if total >= MAX_SNIPPETS_PER_CONCEPT:
            break
    shown_fields = {h["field"] for h in hits}
    more = len(hits) - total
    if more > 0:
        lines.append(f"(+{more} more matches; fields with matches: "
                     f"{', '.join(FIELD_LABELS[f] for f in FIELD_LABELS if f in shown_fields)})")
    return "\n".join(lines)


def candidate_strength(fields: dict[str, str], term_sets: dict[str, str]) -> int | None:
    """Number of (concept, field) pairs with a clean core match; None if a concept lacks one."""
    strength = 0
    for term_set in term_sets.values():
        clean_fields = {
            h["field"] for h in find_terms(fields, term_set, tiers=("core",))
            if not h["exclusion"] and not h["negated"]
        }
        if not clean_fields:
            return None
        strength += len(clean_fields)
    return strength


# =============================================================
# Pools
# =============================================================

def load_runs() -> dict[tuple[str, str], dict[str, list[str]]]:
    """(system, mode) -> query_id -> ranked nct_ids."""
    runs = {}
    for system in POOL_DEPTHS:
        for mode in FILTER_MODES:
            frame = pd.read_csv(RUN_DIR / f"{system}.{mode}.csv").sort_values(["query_id", "rank"])
            runs[(system, mode)] = {
                q: g["nct_id"].tolist() for q, g in frame.groupby("query_id")}
    return runs


def build_pools(specs, runs, records, fields_by_trial, candidates_by_query,
                secondary_depth: dict[str, int], candidate_cap: int):
    """Return pools {query_id: {nct_id: [sources]}} and per-query size details."""
    pools, details = {}, {}
    order = sorted(q for q, s in specs.items() if s.has_answer and q in QUERY_TERMS)
    # Queries whose judgments are reused must be built before their paraphrases.
    order.sort(key=lambda q: 1 if raw_specs[q].get("judgments_shared_with") else 0)

    for query_id in order:
        spec = specs[query_id]
        sources: dict[str, list[str]] = {}
        for (system, mode), ranked_by_query in runs.items():
            depth = POOL_DEPTHS[system]
            if depth < 10:
                depth = secondary_depth.get(query_id, depth)
            for rank, nct_id in enumerate(ranked_by_query.get(query_id, [])[:depth], start=1):
                sources.setdefault(nct_id, []).append(f"{system}.{mode}@{rank}")

        raw_size = len(sources)
        violators = [n for n in sources
                     if constraint_violations(records[n], spec.structured_constraints)]
        for nct_id in violators:
            del sources[nct_id]
        after_violators = len(sources)

        added = 0
        for nct_id, strength in candidates_by_query[query_id]:
            if added >= candidate_cap:
                break
            if nct_id not in sources:
                sources[nct_id] = [f"concept_candidate(strength={strength})"]
                added += 1

        shared = raw_specs[query_id].get("judgments_shared_with")
        reused = 0
        if shared:
            for nct_id in list(sources):
                if nct_id in pools[shared]:
                    del sources[nct_id]
                    reused += 1

        pools[query_id] = sources
        details[query_id] = {
            "split": spec.split,
            "retrieval_pool": raw_size,
            "constraint_violators_removed": len(violators),
            "after_violator_removal": after_violators,
            "concept_candidates_added": added,
            "reused_from_" + shared if shared else "reused_from_other_query": reused,
            "to_label": len(sources),
            "secondary_depth": secondary_depth.get(query_id, 5),
        }
    return pools, details


# =============================================================
# Labeling sheet
# =============================================================

GREY = PatternFill("solid", start_color="D9D9D9")
HEADER = PatternFill("solid", start_color="1F4E78")
LABEL_HEADER = PatternFill("solid", start_color="C65911")
WRAP = Alignment(wrap_text=True, vertical="top")

COLUMNS = [  # (header, width, is_label_column)
    ("row_id", 9, False), ("query_id", 7, False), ("NCT ID", 13, False),
    ("disease_match", 18, True), ("intervention_match", 18, True), ("mention_role", 17, True),
    ("notes", 28, True),
    ("query", 32, False), ("disease concept", 22, False), ("intervention concept", 22, False),
    ("title", 40, False), ("official title", 40, False), ("conditions", 28, False),
    ("interventions (with other names)", 36, False),
    ("disease evidence (where search terms occur)", 60, False),
    ("intervention evidence (where search terms occur)", 60, False),
    ("brief summary", 70, False), ("eligibility criteria", 70, False),
]

INSTRUCTIONS = """Clinical trial relevance labeling - corpus v1, query set v1
Full guide: docs/labeling_guide_v1.md ; full rubric: docs/labeling_rubric_v1.md

For each row, judge the trial against the query's core concepts only:
  disease_match:       exact | broader | absent_or_different
  intervention_match:  exact | class_level | absent_or_different
  mention_role (optional): focus | prior_therapy | concurrent_therapy | excluded | background
  notes: free text; write "UNCERTAIN: <reason>" when unsure.

Grey cells already say not_applicable (the query has no such concept); they are locked.
Do NOT judge status, phase, sex, age or study type: these were checked automatically,
and trials that violate them were removed before this sheet was built.
Age rule = "admits": a trial counts for "pediatric" or "over 65" if its age range includes that group.

Evidence columns only show where the search terms occur (by field, with context tags).
They are not suggestions. A missing snippet does not prove a concept is absent -
open the ClinicalTrials.gov link when in doubt.

Q22 reuses the judgments of Q01, and Q23 those of Q07: only trials that were not
already listed under Q01/Q07 appear for Q22/Q23. Judge them with the same meaning.

Rows are shuffled within each query. Dev queries come first, then test queries.
Leave a cell blank if you have not judged the row yet; blank = not yet judged.
The sheet is protected only to keep grey cells fixed (no password): Review > Unprotect Sheet.
"""


def interventions_text(study: dict) -> str:
    items = study["protocolSection"].get("armsInterventionsModule", {}).get("interventions", [])
    parts = []
    for item in items:
        text = f"{item.get('name', '')} [{item.get('type', '')}]"
        if item.get("otherNames"):
            text += f" (other names: {'; '.join(item['otherNames'])})"
        parts.append(text)
    return "\n".join(parts) or "(none listed)"


def cap(text: str) -> str:
    text = text or ""
    return text if len(text) <= EXCEL_CELL_LIMIT else text[:EXCEL_CELL_LIMIT] + " ...[truncated]"


def concept_label(concept: dict | None) -> str:
    return "(none - not applicable)" if concept is None else concept["concept"]


def write_sheet(rows: list[dict], specs, raw_specs, path: Path) -> None:
    workbook = Workbook()
    guide = workbook.active
    guide.title = "Instructions"
    for i, line in enumerate(INSTRUCTIONS.splitlines(), start=1):
        guide.cell(row=i, column=1, value=line)
    guide.column_dimensions["A"].width = 110
    guide["A1"].font = Font(bold=True, size=13)

    sheet = workbook.create_sheet("Labels")
    for col, (header, width, is_label) in enumerate(COLUMNS, start=1):
        cell = sheet.cell(row=1, column=col, value=header)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = LABEL_HEADER if is_label else HEADER
        cell.alignment = WRAP
        sheet.column_dimensions[cell.column_letter].width = width

    for r, row in enumerate(rows, start=2):
        values = [row[h] for h, _, _ in COLUMNS]
        for col, value in enumerate(values, start=1):
            cell = sheet.cell(row=r, column=col, value=value)
            cell.alignment = WRAP
            header, _, is_label = COLUMNS[col - 1]
            if is_label and value != "not_applicable":
                cell.protection = Protection(locked=False)
            if value == "not_applicable":
                cell.fill = GREY
        link = sheet.cell(row=r, column=3)
        link.hyperlink = row["link"]
        link.font = Font(color="0563C1", underline="single")
        sheet.row_dimensions[r].height = 170

    last = len(rows) + 1
    validations = {
        "D": "exact,broader,absent_or_different",
        "E": "exact,class_level,absent_or_different",
        "F": "focus,prior_therapy,concurrent_therapy,excluded,background",
    }
    for column, options in validations.items():
        dv = DataValidation(type="list", formula1=f'"{options}"', allow_blank=True,
                            showErrorMessage=True, errorTitle="Invalid value",
                            error=f"Choose one of: {options}")
        sheet.add_data_validation(dv)
        dv.add(f"{column}2:{column}{last}")

    sheet.freeze_panes = "H2"
    sheet.auto_filter.ref = f"A1:{sheet.cell(row=1, column=len(COLUMNS)).column_letter}{last}"
    sheet.protection.sheet = True
    sheet.protection.formatColumns = False
    sheet.protection.formatRows = False
    sheet.protection.autoFilter = False

    terms = workbook.create_sheet("Search terms")
    terms.append(["query_id", "concept", "search terms (case-insensitive regular expressions)"])
    for query_id, mapping in QUERY_TERMS.items():
        for concept, term_set in mapping.items():
            patterns = TERMS[term_set]["core"] + TERMS[term_set]["related"]
            terms.append([query_id, concept, " | ".join(patterns)])
    terms.column_dimensions["C"].width = 150

    queries = workbook.create_sheet("Queries")
    queries.append(["query_id", "split", "query", "disease concept", "intervention concept",
                    "structured constraints (checked automatically)", "rows to label"])
    counts = pd.Series([r["query_id"] for r in rows]).value_counts()
    for query_id, spec in specs.items():
        queries.append([query_id, spec.split, spec.text,
                        concept_label(spec.core_concepts.get("disease")),
                        concept_label(spec.core_concepts.get("intervention")),
                        json.dumps(spec.structured_constraints),
                        int(counts.get(query_id, 0))])
    for column, width in zip("ABCDEFG", (9, 7, 60, 35, 35, 55, 13)):
        queries.column_dimensions[column].width = width

    workbook.move_sheet("Labels", offset=-1)
    workbook.active = 0
    workbook.save(path)


# =============================================================
# Main
# =============================================================

def main() -> None:
    for path in (PROVENANCE_OUT, SUMMARY_OUT, TEMPLATE_OUT, SHEET_OUT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    global raw_specs
    specs = load_query_specs(SPECS_PATH)
    raw_specs = {s["query_id"]: s for s in json.loads(SPECS_PATH.read_text())}
    pooled_queries = [q for q, s in specs.items() if s.has_answer]
    missing = set(pooled_queries) - set(QUERY_TERMS)
    if missing:
        raise ValueError(f"No search terms for pooled queries: {sorted(missing)}")

    metadata = pd.read_parquet(PARQUET_PATH).set_index("nct_id")
    records = metadata.to_dict("index")
    studies = {r["nct_id"]: r["study"] for r in json.loads(RAW_PATH.read_text(encoding="utf-8"))}
    fields_by_trial = {n: exp19.raw_fields(s) for n, s in studies.items()}
    runs = load_runs()

    # Concept candidates, ranked once per query (strongest first, seeded ties).
    rng = np.random.default_rng(SEED)
    candidates_by_query = {}
    for query_id in pooled_queries:
        spec = specs[query_id]
        scored = []
        for nct_id, fields in fields_by_trial.items():
            if constraint_violations(records[nct_id], spec.structured_constraints):
                continue
            strength = candidate_strength(fields, QUERY_TERMS[query_id])
            if strength is not None:
                scored.append((nct_id, strength, rng.random()))
        scored.sort(key=lambda item: (-item[1], item[2]))
        candidates_by_query[query_id] = [(n, s) for n, s, _ in scored]
        print(f"{query_id}: {len(scored)} constraint-satisfying trials with clean core matches")

    # Build pools, then apply the overrun rule if needed.
    secondary_depth: dict[str, int] = {}
    candidate_cap = MAX_CONCEPT_CANDIDATES
    pools, details = build_pools(specs, runs, records, fields_by_trial, candidates_by_query,
                                 secondary_depth, candidate_cap)
    initial_total = sum(d["to_label"] for d in details.values())
    overrun_steps = []
    while sum(d["to_label"] for d in details.values()) > BUDGET:
        open_queries = [q for q in details if q not in secondary_depth]
        if open_queries:
            largest = max(open_queries, key=lambda q: (details[q]["to_label"], q))
            secondary_depth[largest] = REDUCED_SECONDARY_DEPTH
            overrun_steps.append(f"secondary depth 5->3 for {largest}")
        elif candidate_cap > 1:
            candidate_cap = 1
            overrun_steps.append("concept candidates capped at 1")
        else:
            raise RuntimeError("Over budget after applying the full overrun rule.")
        pools, details = build_pools(specs, runs, records, fields_by_trial, candidates_by_query,
                                     secondary_depth, candidate_cap)
    total = sum(d["to_label"] for d in details.values())

    # Provenance (not for the labeler) and qrels template.
    provenance, template, sheet_rows = [], [], []
    ordered = sorted(pools, key=lambda q: (0 if specs[q].split == "dev" else 1, q))
    shuffle_rng = np.random.default_rng(SEED)
    for query_id in ordered:
        spec = specs[query_id]
        disease = spec.core_concepts.get("disease")
        intervention = spec.core_concepts.get("intervention")
        nct_ids = sorted(pools[query_id])
        nct_ids = [nct_ids[i] for i in shuffle_rng.permutation(len(nct_ids))]
        for i, nct_id in enumerate(nct_ids, start=1):
            row_id = f"{query_id}-{i:03d}"
            provenance.append({"row_id": row_id, "query_id": query_id, "nct_id": nct_id,
                               "sources": "; ".join(pools[query_id][nct_id])})
            d_val = "" if disease else "not_applicable"
            i_val = "" if intervention else "not_applicable"
            template.append({"row_id": row_id, "query_id": query_id, "nct_id": nct_id,
                             "disease_match": d_val, "intervention_match": i_val,
                             "mention_role": "", "notes": "", "labeler": "", "label_date": ""})
            fields = fields_by_trial[nct_id]
            study = studies[nct_id]
            terms = QUERY_TERMS[query_id]
            sheet_rows.append({
                "row_id": row_id, "query_id": query_id, "NCT ID": nct_id,
                "link": f"https://clinicaltrials.gov/study/{nct_id}",
                "disease_match": d_val, "intervention_match": i_val,
                "mention_role": "", "notes": "",
                "query": spec.text,
                "disease concept": concept_label(disease),
                "intervention concept": concept_label(intervention),
                "title": fields["brief_title"],
                "official title": fields["official_title"],
                "conditions": fields["conditions"],
                "interventions (with other names)": interventions_text(study),
                "disease evidence (where search terms occur)": (
                    evidence_text(find_terms(fields, terms["disease"])) if "disease" in terms
                    else "not applicable"),
                "intervention evidence (where search terms occur)": (
                    evidence_text(find_terms(fields, terms["intervention"])) if "intervention" in terms
                    else "not applicable"),
                "brief summary": cap(fields["brief_summary"]),
                "eligibility criteria": cap(study["protocolSection"].get("eligibilityModule", {})
                                            .get("eligibilityCriteria", "")),
            })

    pd.DataFrame(provenance).to_csv(PROVENANCE_OUT, index=False)
    pd.DataFrame(template, columns=["row_id"] + QRELS_COLUMNS).to_csv(TEMPLATE_OUT, index=False)
    write_sheet(sheet_rows, specs, raw_specs, SHEET_OUT)

    summary = {
        "budget": BUDGET,
        "total_to_label": total,
        "initial_total_before_overrun_rule": initial_total,
        "overrun_steps": overrun_steps,
        "pool_depths": POOL_DEPTHS,
        "filter_modes": FILTER_MODES,
        "max_concept_candidates": MAX_CONCEPT_CANDIDATES,
        "final_concept_candidate_cap": candidate_cap,
        "seed": SEED,
        "per_query": details,
        "not_pooled_zero_answer": sorted(q for q, s in specs.items() if not s.has_answer),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nInitial total {initial_total}; overrun steps: {overrun_steps or 'none'}")
    print(pd.DataFrame(details).T.to_string())
    print(f"\nTOTAL TO LABEL: {total} / {BUDGET}")
    for path in (PROVENANCE_OUT, SUMMARY_OUT, TEMPLATE_OUT, SHEET_OUT):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
