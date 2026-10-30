# Labeling Guide v1.1 (one page)

Full rules: `docs/labeling_rubric_v1.1.md`. Sheet: `data/evaluation/labeling_sheet_v1.1.xlsx`
if it exists, otherwise `labeling_sheet_v1.xlsx`. (v1.1: all query scopes decided.)

## What you do
For each row, read the trial and answer two questions about the query:

1. **Does the trial's population have the query's disease?** → `disease_match`
   - `exact` — yes, or a narrower form of it (query "lung cancer", trial NSCLC).
   - `broader` — the trial is for a wider disease group that includes it
     (query NSCLC, trial "advanced solid tumors"). Must be a disease group,
     not a general-population or aging cohort.
   - `absent_or_different` — another disease, or the disease is only mentioned
     in exclusions or background.
2. **Does the trial give or study the query's intervention?** → `intervention_match`
   - `exact` — yes: alone, in a combination, as a comparator, or as one of
     several allowed drugs. If the query names a class, any member counts.
   - `class_level` — the query names one drug, but the trial names only the
     class ("PD-1 inhibitors").
   - `absent_or_different` — a different named drug (even the same class), or
     the drug appears only as prior / concurrent / excluded therapy.
3. **mention_role** (optional): when you chose `absent_or_different` because the
   drug or disease is only a prior therapy, concurrent therapy, exclusion or
   background, pick that role here.
4. **notes**: anything useful; write `UNCERTAIN: <reason>` when unsure.

Grey cells already say `not_applicable` (the query has no such concept).

## What you do NOT judge
Status, phase, sex, age and study type were checked by code; trials that fail
them are not in the sheet. Age uses "admits": a mixed-age trial counts for
"pediatric" or "over 65". Do not lower a label for these reasons.

## How the grade comes out (computed, you never type it)
All `exact` → **2**. Nothing `absent_or_different`, but some `broader` /
`class_level` → **1**. Any `absent_or_different` → **0**.

## Q01 examples — "Recruiting NSCLC trials involving pembrolizumab"

| Trial | What it is | disease | intervention | Grade |
|---|---|---|---|---|
| NCT06195527 | Pembrolizumab drug-level monitoring in NSCLC | exact | exact | 2 |
| NCT07061379 | NSCLC on pembrolizumab or cemiplimab (condition field says "PROMs") | exact | exact | 2 |
| NCT05761223 | FB849 + pembrolizumab, "advanced solid tumors" | broader | exact | 1 |
| NCT07337096 | Metastatic NSCLC, "PD-1 inhibitors" + chemo, no drug named | exact | class_level | 1 |
| NCT06943820 | NSCLC cohort, PD-1 drug is penpulimab | exact | absent_or_different | 0 |
| NCT06709859 | NSCLC, afatinib + chemotherapy | exact | absent_or_different | 0 |
| NCT06267729 | Prostate cancer; pembrolizumab only as required prior therapy | absent_or_different | absent_or_different (role: prior_therapy) | 0 |

## Query scope reminders (rubric §8)
- **Q04 diabetes:** type 1/2, gestational and complications = exact; prediabetes = absent.
- **Q06 immunotherapy:** checkpoint inhibitors, CAR-T, vaccines, BiTEs, oncolytic viruses,
  cytokines, TIL, NK cells. **Q09 targeted:** kinase inhibitors, antigen-targeting
  antibodies, ADCs, PARP/KRAS inhibitors, radioligands.
- **Q07/Q23 cardiovascular:** heart disease, stroke, hypertension, PAD, VTE; risk factors
  alone = absent.
- **Q08 metastatic:** "advanced or metastatic" = exact; "advanced" alone = broader;
  locally advanced only = absent.
- **Q10 infectious:** treatment and prevention/vaccines; long COVID = exact.
- **Q12 weight loss:** obesity/overweight, or diabetes trials whose primary aim is weight
  loss = exact; weight as a secondary outcome only = absent.
- **Q18 epilepsy:** epilepsy and syndromes = exact; single febrile or non-epileptic seizures = absent.
- **Q20 Parkinson's:** PD = exact; mixed parkinsonism = broader; PSP/MSA only = absent.

## Practical tips
- Read: title → conditions → interventions → inclusion criteria → summary.
  The evidence columns show where search terms appear (exclusion criteria and
  prior-therapy wording are tagged). They are **not** suggestions, and no
  snippet does not prove absence — open the ClinicalTrials.gov link if unsure.
- Judge from what the record says; use medical knowledge for terminology
  (synonyms, brands, classes), not to guess unstated facts.
- Q22 and Q23 mean the same as Q01 and Q07; only new trials are listed.
- Leave a row blank if you have not judged it yet. Save often.
- About 10% of rows will be re-labeled a week later to check consistency.
