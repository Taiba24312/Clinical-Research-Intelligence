# RAG manual-check cases (dev) — for the labeler to verify

Source: `rag_outputs.json` (Experiment 34). Flagged by reading each answer
against the context it was given and the trial record. The prompt was NOT
changed after this review.

| # | Query | Trial | Claim in the answer | Why it is flagged | Grade (qrels) |
|---|---|---|---|---|---|
| 1 | Q09 targeted therapy, solid tumors, Phase 1 | NCT06943820 | "involves a targeted therapy (AK129)" | Record: AK129 is an anti-PD-1/LAG-3 bispecific antibody (immunotherapy under rubric §8, not targeted therapy) | 0 |
| 2 | Q09 | — | "None of the trials are currently recruiting" (used as a criterion) | Recruitment status was not part of the question | — |
| 3 | Q01 NSCLC + pembrolizumab, recruiting | NCT07061379 | "not specifically on pembrolizumab as the primary intervention" → only a partial match | Adds a "primary intervention" requirement not in the question; rubric R6 = exact | 2 |
| 4 | Q01 | NCT06195527 | not mentioned at all | Grade-2 trial in the context omitted (intervention field empty, pembrolizumab in the title) | 2 |
| 5 | Q11 Keytruda in lung cancer | NCT06726265 | "does not focus on Keytruda as the sole or primary treatment" → "None match" | Adds a "sole/primary" requirement; the trial gives pembrolizumab in NSCLC | 2 |
| 6 | Q22 pembrolizumab NSCLC, enrolling | NCT07061379 | "not a standalone pembrolizumab study" → "None match" | Same added requirement | 2 |
| 7 | Q20 Parkinson's, not yet recruiting | — | "several trials list Parkinson's disease as a condition" | Only one context trial (NCT06809556) does | — |
| 8 | Q20 | NCT06809556 | rejected: "do not target Parkinson's disease with interventions relevant to its treatment" | The question asks for no intervention | 2 |
| 9 | Q06 immunotherapy for cancer, Phase 2 | NCT02688686 | "uses dendritic cell therapy, not standard immunotherapy" | Cellular immunotherapy (DC-CIK) counts as immunotherapy under rubric §8 | 2 |
| 10 | Q18 pediatric epilepsy | NCT03836924 | "participants aged 12 years or older" | Age is not among the context fields; it matches the record (min age 12) — check whether it came from the title or from outside knowledge | 2 |
| 11 | Q18 | — | Lists two matches, then ends "None of the retrieved trials match this query exactly" | Self-contradictory; citations given without links | — |
| 12 | Q08 metastatic cancer, recruiting, adults | NCT07238283 | "metastatic pancreatic cancer, not broad metastatic cancer" → partial | Narrower = exact (rubric R1) | 2 |

Pattern: most errors come from the model adding requirements the question
did not state ("primary/standalone intervention", "currently recruiting",
"relevant treatment"), which causes the 3 false "no match" answers (Q11,
Q20, Q22) and several demotions to "Partial matches".
