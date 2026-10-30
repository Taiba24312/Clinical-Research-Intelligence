# Labeling Rubric v1 — Clinical Trial Relevance Judgments

Applies to: `data/evaluation/query_specs_v1.json`, corpus v1
(`data/corpus_v1_manifest.json`). Status: **DRAFT — items marked PROPOSED
need the labeler's confirmation before labeling starts.**

---

## 1. What you judge, and what you do not

For each (query, trial) pair you record **two concept judgments**:

| Field | Values |
|---|---|
| `disease_match` | `exact` · `broader` · `absent_or_different` · `not_applicable` |
| `intervention_match` | `exact` · `class_level` · `absent_or_different` · `not_applicable` |
| `mention_role` (optional) | `focus` · `prior_therapy` · `concurrent_therapy` · `excluded` · `background` |
| `notes` | free text; write `UNCERTAIN: <reason>` when unsure |
| `labeler`, `label_date` | initials, YYYY-MM-DD |

You do **not** judge status, phase, sex, age or study type. These are checked
automatically from trial metadata. Trials that violate a query's structured
constraints are removed from your labeling pool and scored as grade 0 by code.

`not_applicable` is used only when the query has no concept of that kind
(e.g. Q02 has no intervention). The query spec states which concepts exist.

## 2. How the grade is derived (by code, never typed)

| Grade | Rule |
|---|---|
| **2** | every applicable concept is `exact` |
| **1** | every applicable concept is at least `broader` / `class_level`, none is `absent_or_different` |
| **0** | any applicable concept is `absent_or_different` |

Grade 2 = "this is what I asked for". Grade 1 = "useful, but only partially
matches". Grade 0 = "does not answer this query".

## 3. Definitions

### disease_match
- **exact** — the trial's study population has the query condition, **or a
  narrower form of it** (query "lung cancer", trial NSCLC → exact; query
  "diabetes", trial type 2 diabetes → exact).
- **broader** — the trial's population is broader than the query condition
  **and plausibly includes it** (query NSCLC, trial "advanced solid tumors"
  basket → broader). If a broader trial explicitly excludes the query
  condition, it is `absent_or_different`.
- **absent_or_different** — the population has a different condition, or the
  query condition appears only as background, in exclusion criteria, or as a
  comorbidity the trial does not target.

Prevention trials: the condition being prevented counts as the trial's
condition (HIV PrEP trial → disease HIV = exact).

### intervention_match
- **exact** — the trial administers or studies the query intervention, **or a
  narrower form of it**. If the query names a class (e.g. "SGLT2 inhibitors",
  "immunotherapy"), any named member of that class is exact.
- **class_level** — the query names a **specific drug**, and the trial names
  **only the drug class** without naming a specific drug
  (query pembrolizumab, trial "PD-1 inhibitors" → class_level).
- **absent_or_different** — the trial uses a different named drug, even of the
  same class (query pembrolizumab, trial penpulimab → absent_or_different),
  or the query drug appears only in a non-focus role (see §4).

## 4. Decision rules

**R1 — Narrower is exact.** A trial more specific than the query matches
exactly (query "cancer", trial melanoma; query "immunotherapy", trial
pembrolizumab).

**R2 — Broader must plausibly include the query population.** A solid-tumor
basket trial is `broader` for an NSCLC query unless NSCLC is excluded. A
"cardiovascular disease" trial is `broader` for an HFrEF query only if heart
failure patients are eligible. The broader population must be **defined by a
disease category that contains the query condition** (dementia ⊃ Alzheimer's
disease; solid tumors ⊃ NSCLC). General-population, healthy-volunteer or
aging cohorts that merely list the condition among outcomes are
`absent_or_different`. *(Clarified after the Q03 zero-answer check, see §6b.)*

**R3 — Different named drug of the same class = absent_or_different**, not
class_level. class_level is reserved for trials that name no specific drug.

**R4 — Combination therapy.** If the query drug is one component of the
trial's regimen (pembrolizumab + chemotherapy), it is `exact`. If the query
names a combination ("A plus B"), both components must be present for
`exact`; one component only = `absent_or_different`.

**R5 — Comparator or control arm.** If the query drug is given in any arm,
including as the comparator, it is `exact` (the trial "involves" it). Add a
note `comparator arm`.

**R6 — One of several allowed treatments.** If the study population receives
the query drug as one of several named options (e.g. "pembrolizumab or
cemiplimab"), it is `exact` (consistent with NCT07061379, §6).

**R7 — Non-focus mentions → absent_or_different + mention_role.**
- `prior_therapy`: eligibility requires or allows previous treatment with it.
- `concurrent_therapy`: it is permitted as background/concomitant treatment.
- `excluded`: it appears in exclusion criteria.
- `background`: it appears only in the rationale/description.
A non-focus `mention_role` together with no `absent_or_different` field is
rejected by the loader.

**R8 — Evidence may come from any field.** Title, official title, conditions,
interventions, arm descriptions, inclusion criteria, summary and detailed
description all count. A disease that appears **only in the inclusion
criteria** as the required population is `exact`. A disease that appears
**only in exclusion criteria** is `absent_or_different`.

**R9 — Judge from the record, not from outside guesses.** Use clinical
knowledge for terminology (synonyms, brand names, abbreviations, class
membership, disease hierarchy). Do **not** infer unstated trial facts (e.g.
"this PD-1 trial in 2025 probably used pembrolizumab").

**R10 — Observational studies.** An intervention matches if the observed
cohort receives it as part of what the study is about (e.g. monitoring
patients treated with pembrolizumab → exact).

**R11 — When uncertain**, choose the label you believe is most likely, and
write `UNCERTAIN: <reason>` in `notes`. These pairs are reviewed before
metrics are frozen. Do not leave a pair blank because it is hard — blank
means "not yet looked at".

## 5. Structured constraints (automatic — for reference only)

| Query wording | Constraint | Satisfied when |
|---|---|---|
| recruiting / currently enrolling | status RECRUITING | status = RECRUITING (NOT_YET_RECRUITING does not count yet) |
| not yet recruiting | status NOT_YET_RECRUITING | exact status |
| completed | status COMPLETED | exact status |
| Phase 3 / Phase III | phase PHASE3 | trial lists PHASE3 (PHASE2; PHASE3 counts) |
| Phase 1/2 | phase PHASE1+PHASE2 | trial lists both phases |
| observational / non-interventional / cohort | study_type OBSERVATIONAL | exact type |
| interventional | study_type INTERVENTIONAL | exact type |
| excluding observational | study_type INTERVENTIONAL or EXPANDED_ACCESS | not observational |
| female patients / women | sex FEMALE or ALL | trial accepts women |
| adults | age [18, ∞) | trial age range overlaps; missing bound = no limit |
| pediatric / children | age [0, 18) | as above |
| adolescents | age [12, 18) | as above |
| over 65 / older adults | age [65, ∞) | as above |

## 6. Worked examples — Q01 "Recruiting NSCLC trials involving pembrolizumab"

Evidence from Step 1 (`data/evaluation/q01_ceiling_v1_evidence.csv`).
**These are the Q01 decisions drafted by the user and must be confirmed.**

| Trial | Evidence | disease | intervention | role | Grade |
|---|---|---|---|---|---|
| NCT06195527 | NSCLC in title/conditions; pembrolizumab drug monitoring | exact | exact | focus | **2** |
| NCT07061379 | NSCLC only in title/keywords/eligibility (condition field says "PROMs"); pembrolizumab or cemiplimab in arms | exact (R8) | exact (R6) | focus | **2** |
| NCT05761223 | "Advanced Solid Tumor", lung never mentioned; FB849 + pembrolizumab | broader (R2) | exact (R4) | focus | **1** |
| NCT07337096 | First-line metastatic NSCLC; "PD-1 inhibitors" + chemo, no drug named | exact | class_level | focus | **1** |
| NCT06943820 | NSCLC cohort; PD-1 agent is penpulimab | exact | absent_or_different (R3) | — | **0** |
| NCT06709859 | Unresectable NSCLC; afatinib + chemotherapy | exact | absent_or_different | — | **0** |
| NCT06267729 | Metastatic prostate cancer; "have received ... checkpoint inhibitors (ie, pembrolizumab)" | absent_or_different | absent_or_different (R7) | prior_therapy | **0** |
| NCT04965766 | Metastatic breast cancer; patritumab deruxtecan; no NSCLC, no pembrolizumab | absent_or_different | absent_or_different | — | **0** |
| NCT06545682 | Melanoma / breast cancer; alpelisib + pembrolizumab | absent_or_different | exact | — | **0** |
| NCT07246759 | NSCLC + pembrolizumab, but NOT_YET_RECRUITING | — | — | — | **0, automatic** (status violation; not in your pool) |

Note on NCT06545682: "Solid Tumors" appears only inside "Response Evaluation
Criteria in Solid Tumors" (RECIST). A measurement standard is not a
population, so this is not a `broader` match.

## 6b. Worked examples — Q03 "Recruiting trials for Alzheimer's disease"

From the zero-answer check (`data/evaluation/zero_answer_check_v1_evidence.csv`).
No recruiting trial targets Alzheimer's disease itself. **To be confirmed by
the labeler:**

| Trial | Evidence | disease | Grade |
|---|---|---|---|
| NCT05858996 | Older adults (≥ 60) with dementia confirmed by BIMS/AD8/CDR scores, plus pain | broader (R2: dementia ⊃ AD) | **1** |
| NCT07646782 | General-population aging cohort; "Dementia" is one of 19 listed outcome conditions | absent_or_different (R2) | **0** |
| NCT06914284 | Older adults with bipolar disorder; risk of progression to dementia | absent_or_different | **0** |
| NCT06692296 | Friedreich ataxia ("neurodegenerative disorder") | absent_or_different | **0** |
| any trial | "dementia" or "Alzheimer's" only in exclusion criteria | absent_or_different (R8) | **0** |

If NCT05858996 is confirmed as grade 1, Q03 becomes a query with only
partial answers: strict metrics are undefined, lenient metrics apply.

## 7. Worked examples — other hard cases

The examples below are **hypothetical** (written for illustration, not taken
from the corpus).

| Query | Hypothetical trial | disease | intervention | Grade | Rule |
|---|---|---|---|---|---|
| Q06 immunotherapy for cancer | Phase 2 nivolumab in renal cell carcinoma | exact | exact | 2 | R1 |
| Q06 | Phase 2 carboplatin/paclitaxel in ovarian cancer | exact | absent_or_different | 0 | |
| Q11 Keytruda in lung cancer | Pembrolizumab in small cell lung cancer | exact | exact | 2 | R1 |
| Q11 | Pembrolizumab in "advanced solid tumors", lung eligible | broader | exact | 1 | R2 |
| Q13 SGLT2i in T2DM | Empagliflozin vs placebo in T2DM | exact | exact | 2 | class query |
| Q13 | Metformin trial; "SGLT2 inhibitors not permitted" | exact | absent_or_different (excluded) | 0 | R7 |
| Q14 HFrEF | Heart failure, any ejection fraction | broader | n/a | 1 | R2 |
| Q14 | HFpEF only (EF ≥ 50%) | absent_or_different | n/a | 0 | |
| Q15 anticoagulation in AFib | Apixaban vs warfarin in AF | exact | exact | 2 | class query |
| Q15 | Left atrial appendage occlusion device vs anticoagulation | exact | exact (comparator) | 2 | R5 |
| Q15 | Antiplatelet therapy only in AF | exact | absent_or_different | 0 | |
| Q19 HIV PrEP | Cabotegravir long-acting PrEP | exact | exact | 2 | prevention |
| Q19 | Antiretroviral treatment in people with HIV | exact | absent_or_different | 0 | |
| Q02 breast cancer, Phase 3 | Exercise program for breast-cancer survivors (Phase 3) | exact | n/a | 2 | disease-only query |
| Q05 breast cancer, female | Breast cancer trial, sex = ALL | exact | n/a | 2 | sex checked automatically |

## 8. Query-scope definitions — PROPOSED, please confirm or edit

| Query | Proposed scope |
|---|---|
| Q04 diabetes | Type 1, type 2, gestational and other specific diabetes types = exact. **Prediabetes = absent_or_different.** Diabetic complications (foot ulcer, retinopathy, nephropathy) = exact, because the population has diabetes. |
| Q06 immunotherapy | Checkpoint inhibitors, cancer vaccines, cellular therapies (CAR-T, TIL, NK), bispecific T-cell engagers, cytokines (IL-2, interferon), oncolytic viruses. **Not** tumor-targeted monoclonal antibodies such as trastuzumab (these count as targeted therapy). |
| Q07/Q23 cardiovascular disease | Heart and blood-vessel disease, including stroke, hypertension, peripheral artery disease, venous thromboembolism. **Risk factors alone** (dyslipidemia, obesity, diabetes) = absent_or_different unless the trial targets a cardiovascular disease or CV-event outcome. |
| Q08 metastatic cancer | Trials whose population includes metastatic disease ("advanced or metastatic", "stage IV") = exact. "Advanced" with no metastatic mention = broader. Early-stage or locally advanced only = absent_or_different. |
| Q09 targeted therapy | Small-molecule inhibitors of a specific molecular target (TKIs, PARP, CDK4/6, KRAS inhibitors), tumor-targeted monoclonal antibodies, antibody-drug conjugates, radioligand therapy. **Not** checkpoint inhibitors, cytotoxic chemotherapy or hormone therapy. |
| Q10 infectious disease | Any infection (bacterial, viral, fungal, parasitic), including vaccine and prevention trials and sepsis. Post-infectious conditions (e.g. long COVID) = broader. |
| Q12 weight loss | Population with obesity/overweight, or trials aiming at weight reduction = exact. Semaglutide for glycemic control only in T2DM (weight not a target) = absent_or_different. |
| Q18 epilepsy | Epilepsy and epilepsy syndromes = exact. Febrile seizures or acute symptomatic/neonatal seizures alone = absent_or_different. |
| Q20 Parkinson's disease | Parkinsonism including atypical parkinsonian syndromes = broader. |

## 9. Procedure

1. Labeling files will be generated per query with blank label fields;
   trials violating structured constraints are already removed.
2. Read in this order: title → conditions → interventions/arms → inclusion
   criteria → summary; consult the detailed description and exclusion
   criteria when needed.
3. Paraphrase queries (Q22 ↔ Q01, Q23 ↔ Q07) reuse the original query's
   judgments; you only label trials newly pooled for the paraphrase.
4. Consistency check: about 10% of pairs (random sample, fixed seed) are
   re-labeled one week later without seeing the earlier labels.
5. Expected effort: at roughly 1.5–2 minutes per pair, 500 pairs take about
   13–17 hours.
