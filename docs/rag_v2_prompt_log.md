# RAG v2 prompt log (dev only)

All versions: same model (Qwen3-4B-Instruct-2507, greedy, cuda:1), frozen retrieval (`frozen-final`),
top-5 context, the 16 dev queries (15 answerable + Q24), qrels v1.3. Prompt texts: `PROMPTS` in
`src/generation/rag_v2.py` (every version kept). Outputs: `data/evaluation/results/rag_v2_dev/<version>/`.

Selection criteria, fixed before choosing: (1) fewer false "no match" answers, (2) higher grade
agreement with the labeler after the quote check, (3) fewer grade-2 context trials left uncited,
(4) higher relevance of trials cited as matches.

| Version | Change | False no-match | Correct no-match | Grade agreement (kappa) | Disease agr. | Intervention agr. | Grade-2 not cited | Match citations relevant | Quotes not found | Median latency (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RAG v1 | single prompt over the top-5 | 3 | 2 | - | - | - | 5 | 0.931 of 29 | - | 6.22 |
| v2.0 | first two-step version | 0 | 1 | 0.773 (0.59) | 0.843 | 0.625 | 8 | 0.951 of 41 | 1/80 | 9.55 |
| v2.1 | narrower = exact (with examples); class_level only if the drug is not named; answer must mention every verified trial | 0 | 1 | 0.800 (0.63) | 0.886 | 0.625 | 1 | 0.955 of 44 | 2/80 | 8.20 |
| v2.2 | ordered decision procedure for intervention (class questions never class_level); stricter 'broader' | 0 | 1 | 0.827 (0.65) | 0.857 | 0.833 | 3 | 0.936 of 47 | 3/80 | 8.54 |

**Chosen: v2.2** - ties on criterion 1 and is best on criterion 2 (grade agreement 0.827 vs 0.800
for v2.1). The margin is small (2 of 75 judged trials); v2.1 left fewer grade-2 trials uncited (1 vs 3).
Known weakness of all v2 versions: 4-5 trials labeled grade 0 are verified as partial matches
(mostly 'broader'), which also costs the correct "no match" on Q03.

