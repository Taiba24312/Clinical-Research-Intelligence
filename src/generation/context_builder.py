"""
Build the grounded prompt for answer generation and check its citations.

The context lists the top retrieved trials with only structured, short
fields (title, conditions, interventions, status, phase). The system prompt
restricts the model to these trials and requires an explicit statement when
none of them match.
"""

from __future__ import annotations

import re
from typing import Any


TOP_K_CONTEXT = 5
NO_MATCH_SENTENCE = "None of the retrieved trials match this query."

SYSTEM_PROMPT = f"""You are a clinical trial search assistant. You receive a user question and a numbered list of clinical trials retrieved from ClinicalTrials.gov.

Rules:
1. Use ONLY the information in the listed trials. Do not use outside knowledge, and do not invent trials, results or details.
2. A trial is a match only if it fits every part of the question: the condition, the intervention, and any stated recruitment status, phase or population.
3. For each matching trial, write one short line saying what it studies, followed by its citation in the form [NCT ID](https://clinicaltrials.gov/study/NCT ID).
4. A trial that matches only partly (for example a broader population, or a different drug of the same kind) may be listed under "Partial matches:" with a few words on what does not match.
5. If none of the trials match, write exactly: "{NO_MATCH_SENTENCE}" followed by one sentence explaining why.
6. Never cite an NCT ID that is not in the list. Keep the answer under 150 words."""


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def build_context(trials: list[dict[str, Any]], k: int = TOP_K_CONTEXT) -> str:
    blocks = []
    for i, trial in enumerate(trials[:k], start=1):
        blocks.append(
            f"[{i}] {trial['nct_id']}\n"
            f"Title: {_short(trial['title'], 250)}\n"
            f"Conditions: {_short(trial['conditions'], 200)}\n"
            f"Interventions: {_short(trial['interventions'], 250)}\n"
            f"Status: {trial['status']} | Phase: {trial['phase']}"
        )
    return "\n\n".join(blocks) if blocks else "(no trials retrieved)"


def build_messages(query: str, trials: list[dict[str, Any]], k: int = TOP_K_CONTEXT) -> list[dict]:
    user = f"Question: {query}\n\nRetrieved trials:\n\n{build_context(trials, k)}\n\nAnswer:"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def check_citations(answer: str, trials: list[dict[str, Any]], k: int = TOP_K_CONTEXT) -> dict:
    """Which NCT IDs the answer cites, and whether they were in the context."""
    context_ids = [t["nct_id"] for t in trials[:k]]
    cited = list(dict.fromkeys(re.findall(r"NCT\d{8}", answer)))
    return {
        "cited": cited,
        "invalid_citations": [n for n in cited if n not in context_ids],
        "says_no_match": NO_MATCH_SENTENCE.lower() in answer.lower(),
    }
