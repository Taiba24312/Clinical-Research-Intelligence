"""
Project configuration: the frozen final pipeline.

Chosen on the DEV queries only (experiments 28, 31, 33) and frozen before
the single test-set evaluation (experiment 35). Do not change these values
after the test results; add a new configuration instead.
"""

from __future__ import annotations


# Original baseline pipeline (as in data/runs/v1, rrf.parser_filters).
BASELINE_PIPELINE = {
    "parser_version": "v1",
    "doc_version": "v1",
    "tokenizer": "v1",
}

# Final pipeline:
#   parser v1.2  - status/phase/study-type fixes (v1.1) + age groups
#   document v1  - document v2 (MeSH, other names, official title, keywords)
#                  was not adopted: mixed effect on dev (Q09 worse), more
#                  unjudged results, no gain over config C after accounting
#                  for unjudged trials
#   tokenizer v2 - BM25 splits on punctuation/symbols, drops filler words
FINAL_PIPELINE = {
    "parser_version": "v1.2",
    "doc_version": "v1",
    "tokenizer": "v2",
}

# Generation (unchanged since experiment 29; prompt in src/generation/context_builder.py).
FINAL_GENERATION = {
    "model": "Qwen/Qwen3-4B-Instruct-2507",
    "decoding": "greedy, max_new_tokens=350",
    "context_trials": 5,
}
