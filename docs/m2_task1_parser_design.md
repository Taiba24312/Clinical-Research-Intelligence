# M2 Task 1 — Structured-constraint parser v2: design (for review)

Status: **proposal, not implemented.** Scope: status, phase, sex, age, study
type and their negations. No disease/intervention extraction.

## 1. Interface and versioning

- New module `src/query_understanding/parser_v2.py`, function
  `parse_query_v2(text) -> ParsedQueryV2` with fields:
  - `semantic_query: str` — text sent to BM25 / dense / reranker, with the
    recognized constraint cues removed;
  - `constraints: dict` — **same schema as the query specs**
    (`status`, `phase` incl. `"PHASE1+PHASE2"`, `sex`, `study_type`,
    `age_range: [lo, hi)`);
  - `unresolved: tuple[{"constraint", "text"}]` — cues recognized but not
    mapped (see §3);
  - `secondary_status: tuple[str]` — `("NOT_YET_RECRUITING",)` when a plain
    "recruiting" request should also surface upcoming trials (see §4).
- The old parser (`parser.py`, `parse_query`) is **not changed**.
  `tests/data/parser_v1_golden_outputs.json` freezes its outputs on all
  phrasings and all 26 evaluation queries; a test fails if they ever change.
- Retrieval must not reuse the evaluation constraint checker (it is the
  reference implementation). `MetadataFilter` gets a **new** method
  `apply_constraints(constraints)` for the spec schema (age ranges, combined
  phases, all six age units incl. hours/minutes); the old `apply(filters)` is
  left unchanged. A test will require `apply_constraints` and the evaluation
  checker to select exactly the same trials for every spec constraint set.

## 2. Grammar decisions encoded in the tests (please confirm)

| Cue | Parsed as |
|---|---|
| recruiting, currently/actively recruiting, enrolling now, currently enrolling, accepting participants/patients, open to enrollment | status RECRUITING **+ secondary NOT_YET_RECRUITING** |
| "only" with a recruiting cue | RECRUITING, no secondary section |
| "open" alone, "upcoming" | RECRUITING or NOT_YET_RECRUITING (your decision 2) |
| recruiting or not yet recruiting | both statuses, no secondary section |
| not yet recruiting, not yet open, have not started enrolling | NOT_YET_RECRUITING |
| completed, finished / terminated / withdrawn / active, not recruiting / enrolling by invitation | the matching status |
| excluding completed | every status except COMPLETED |
| not recruiting, no longer recruiting, ongoing, active (alone) | **unresolved** (not yet? closed? which?) |
| Phase 1–4 in digits, Roman numerals or words; 1b → 1, IIa → 2 | that phase |
| Phase 1/2, I/II, 1-2, 2/3 | combined (`PHASE1+PHASE2`, `PHASE2+PHASE3`) |
| Phase 0, early phase 1 | EARLY_PHASE1 |
| phase 2 or phase 3 | either |
| late-phase, phase 2 and 3, excluding phase 1 | **unresolved** |
| women, female, pregnant women | FEMALE or ALL (your decision 1) |
| women-only, female-only (male likewise) | FEMALE (MALE) only |
| men and women, both sexes | ALL |
| "men" inside words (mental, regimen) | nothing |
| children, kids, pediatric/paediatric | age [0, 18) |
| adolescents, teenagers | [12, 18) |
| infants | [0, 2) |
| adults | [18, ∞) |
| elderly, older adults, seniors, geriatric, aged 65 and above | [65, ∞) |
| N years or older / over N | [N, ∞) |
| under N | [0, N) |
| ages A to B (inclusive) | [A, B+1) |
| children and adults | no age constraint |
| young adults, middle-aged | **unresolved** |
| observational, non-/noninterventional, cohort, registry | OBSERVATIONAL |
| interventional, randomized (controlled) | INTERVENTIONAL |
| expanded access, compassionate use | EXPANDED_ACCESS |
| excluding / not / non-observational | INTERVENTIONAL or EXPANDED_ACCESS |
| excluding interventional | OBSERVATIONAL or EXPANDED_ACCESS |
| conflicting study-type cues | **unresolved** |
| "trial(s)", "study/studies" alone | no study-type constraint |

## 3. Unresolved phrasings

The parser does not guess. An unresolved cue is returned with its matched
text and **no filter is applied for that constraint type**; retrieval runs
as if the cue were absent. Every run records each query's unresolved items
in its config, and the evaluation report lists them.

## 4. NOT_YET_RECRUITING as a separate section

**Retrieval.** For a query with `secondary_status`, the system produces two
lists:
- **main** — strict constraints (status RECRUITING), exactly as today;
- **not_yet_recruiting** — same pipeline and the same other constraints,
  but status NOT_YET_RECRUITING; top 5 trials.
Queries without `secondary_status` produce only the main list.

**Run format.** Add an optional `section` column with values `main` and
`not_yet_recruiting`; ranks restart at 1 in each section. Run files without
the column are treated as all `main`, so every existing run stays valid.
`prepare_run` checks rank and trial uniqueness per (query, system, section).

**Metrics.** `evaluate_run` uses only `section == main`. The secondary
section is therefore never counted as a constraint violation and never
enters nDCG / P / R / MRR. A separate **secondary-section report** gives per
query: trials returned, violations against the secondary constraints
(should be 0), judged share, and grades where judged.

**Limitation.** The current pools removed NOT_YET_RECRUITING trials from
"recruiting" queries as status violators, so almost all secondary-section
trials are unjudged. Judging their quality would need some of the 26 reserve
labels; until then the report shows coverage and correctness of the section,
not its relevance.

## 5. Implementation order (after approval)

1. `parser_v2.py` until all phrasing tests pass (rules written against the
   phrasing list only; evaluation queries are not consulted).
2. `MetadataFilter.apply_constraints` + agreement test with the evaluation
   checker.
3. Run-format `section` support in `judgments.prepare_run` and
   `evaluate_run` + tests; existing results must reproduce exactly.
4. New runs on **dev queries only** (`data/runs/v2_parser_dev/`): bm25_ws,
   dense, rrf, rrf_ce_raw with parser-v2 filters, plus the secondary section.
5. Report: per-query parser output vs spec; dev nDCG@10 (standard and
   unjudged-removed), Judged@10, violation rate — v2 vs old pipeline vs the
   gold-filter ceiling; dev queries that get worse; number of newly
   retrieved unjudged top-10 trials; unresolved items. Nothing on test.
