# Pooling Plan v1 (draft — no pools built yet)

Budget: **500 manually judged query–trial pairs** (the ~10% consistency
re-label is not included in this number).

## Pooling systems (all over corpus v1)

| # | System | Depth | Why |
|---|---|---:|---|
| 1 | RRF (BM25 + bge-small), metadata-filtered — current pipeline | 10 | baseline evaluated at K = 10 |
| 2 | RRF + cross-encoder on `semantic_query` | 10 | likely next reranker variant |
| 3 | BM25, whitespace tokenizer (current), filtered | 5 | component baseline |
| 4 | BM25, punctuation-splitting tokenizer, filtered | 5 | likely M2 variant (pool-only) |
| 5 | Dense bge-small, filtered | 5 | component baseline |
| 6 | RRF + cross-encoder on raw query (Exp 17 setting) | 5 | existing reranker |
| 7 | Concept candidates (Step-1-style multi-field match satisfying constraints) | ≤ 3 | protects the M2 concept-aware system from unjudged-document penalties |

Variants 2 and 4 are pooling-only; core modules are not changed. Trials that
violate a query's structured constraints are removed (grade 0 by code).

## Size estimate (from existing Exp 12/16/17 outputs for Q01–Q10, no new retrieval)

Proxy pool with depth 10 for two lists and depth 5 for four lists, after
removing constraint violators: mean **20.1**, median 20, max 31 trials per
query. Using depth 3 instead of 5 for the secondary lists saves only 1.8 per
query (mean 18.3), so depth 5 is kept.

| Item | Queries | Pairs each | Pairs |
|---|---:|---:|---:|
| Full answerable queries (26 − 3 verified zero-answer − 2 paraphrases; includes Q03 if confirmed answerable) | 21 | 20 + ~2 concept candidates = 22 | 462 |
| Paraphrases Q22, Q23 (reuse Q01/Q07 judgments; only new trials) | 2 | ~6 | 12 |
| Verified zero-answer Q24–Q26 (no pool needed) | 3 | 0 | 0 |
| Already labeled (Q01 positives) | | | −2 |
| **Expected total** | | | **≈ 472** |
| Reserve | | | ≈ 28 (5.6%) |

## Overrun rule (fixed in advance)

If the built pools exceed 500 pairs: reduce systems 3–6 from depth 5 to depth
3, largest pools first, until within budget. Never drop queries and never
reduce the depth-10 lists. If still over: cap concept candidates at 1.

## Labeling order

Dev queries first, then test queries, so a stalled labeling effort still
leaves a usable dev set.
