"""
RAG v2: concept extraction -> per-trial verification with evidence quotes ->
answer generated only from verified trials.

1. Concept extraction (1 LLM call): disease and intervention concepts of the
   question (the rule-based parser only extracts structured constraints).
2. Verification (1 call per trial, batched): disease_match and
   intervention_match labels with a compact version of the labeling rubric,
   plus an evidence quote copied from the trial text. Structured constraints
   are already enforced by retrieval and are not re-judged.
3. Quote check (code): a verification whose quote is not found verbatim in
   the trial text is dropped (treated as not verified).
4. Answer: if no trial is verified as at least a partial match, a fixed "no
   matching trials" message is returned without an LLM call; otherwise the
   LLM writes the answer from the verified trials only.

Every prompt version is kept in PROMPTS; DEFAULT_PROMPT_VERSION is the
frozen one. Uses the same local model as RAG v1 (src/generation/llm.py).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.generation.llm import LocalLLM


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIAL_FIELDS_PATH = PROJECT_ROOT / "data/processed/trials_v2.parquet"
NO_MATCH_TEXT = "No matching trials were found among the retrieved trials."
MIN_QUOTE_CHARS = 8

DISEASE_LABELS = {"exact", "broader", "absent", "not_applicable"}
INTERVENTION_LABELS = {"exact", "class_level", "absent", "not_applicable"}


# =============================================================
# Prompts (every version kept)
# =============================================================

_CONCEPTS_V2_0 = """Extract the core concepts of a clinical-trial search question.
- disease: the condition or disease of the patient population, in the user's words (null if none).
- intervention: the drug, drug class, device or procedure being asked about (null if none).
Ignore recruitment status, phase, age, sex and study type.
If a concept is an abbreviation or a brand name you know, add the full or generic name in brackets, e.g. "NSCLC (non-small cell lung cancer)".
Return only JSON: {"disease": ..., "intervention": ...}"""

_VERIFY_V2_0 = """You check whether ONE clinical trial matches the core concepts of a search question.
Judge ONLY the disease and the intervention. Recruitment status, phase, age, sex and study type were already checked by the search system: do NOT judge them.

disease_match:
- "exact": the trial's study population has the question's condition, or a narrower form of it (question "lung cancer", trial "NSCLC").
- "broader": the trial's population is a wider disease group that includes the condition (question "NSCLC", trial "advanced solid tumors").
- "absent": a different condition, or the condition appears only in exclusion criteria or background text.
- "not_applicable": the question names no disease.
intervention_match:
- "exact": the trial gives or studies the question's intervention, alone, in a combination, as a comparator, or as one of several allowed options. If the question names a class (e.g. "immunotherapy"), any member counts.
- "class_level": the question names one specific drug and the trial names only its class (e.g. "PD-1 inhibitors").
- "absent": a different named drug, or the intervention appears only as prior, concurrent or excluded therapy.
- "not_applicable": the question names no intervention.

Do not add requirements the question does not state (for example "primary intervention", "standalone", "treatment trial").
"evidence" must be a short quote (5-25 words) copied exactly, character for character, from the trial text.
Return only JSON: {"disease_match": "...", "intervention_match": "...", "evidence": "..."}"""

_ANSWER_V2_0 = """Write a short answer to the user's question using ONLY the verified trials listed below.
Each trial is marked MATCH (fits the question) or PARTIAL (broader population or drug class only) and comes with an evidence quote.
- First list the MATCH trials, one line each: what the trial studies, then its citation [NCT ID](https://clinicaltrials.gov/study/NCT ID).
- Then, if there are PARTIAL trials, write "Partial matches:" and list them the same way, saying briefly what is broader.
- If there are no MATCH trials, start with "No trial fully matches the question." and then list the partial matches.
- Do not mention any other trial, do not add details that are not in the list, and do not add requirements the question does not state.
Keep the answer under 150 words."""

# v2.1 (after dev review of v2.0): "broader" was used in the wrong direction
# (specific trials for a broad question) and for general cohorts;
# "class_level" was used when the trial names the drug; the answer step
# dropped verified trials.
_VERIFY_V2_1 = """You check whether ONE clinical trial matches the core concepts of a search question.
Judge ONLY the disease and the intervention. Recruitment status, phase, age, sex and study type were already checked by the search system: do NOT judge them.

disease_match:
- "exact": the trial's population has the question's condition. This includes any type, subtype, stage, complication or specific example of it: narrower is exact. Examples: question "diabetes", trial "gestational diabetes" -> exact; question "cancer", trial "bladder cancer" -> exact; question "infectious disease", trial "hepatitis B" -> exact.
- "broader": ONLY when the question's condition is MORE specific than the trial's population, and the trial's population is defined by a disease group that contains it. Example: question "NSCLC", trial "advanced solid tumors" -> broader.
- "absent": a different condition; or the condition appears only in exclusion criteria, background text or as a risk; or the population is a general, healthy or mixed cohort that is not defined by that disease.
- "not_applicable": the question names no disease.
intervention_match:
- "exact": the trial text names the question's intervention (or its brand or code name) as something participants receive or are studied with: alone, in a combination, as a comparator, or as one of several allowed options. If the question names a class (e.g. "immunotherapy", "SGLT2 inhibitors"), any named member of that class is exact.
- "class_level": ONLY when the question names one specific drug and the trial does NOT name that drug but refers only to its class (e.g. "PD-1 inhibitors").
- "absent": a different named drug, or the intervention appears only as prior, concurrent or excluded therapy.
- "not_applicable": the question names no intervention.

Do not add requirements the question does not state (for example "primary intervention", "standalone", "treatment trial").
"evidence" must be a short quote (5-25 words) copied exactly, character for character, from the trial text.
Return only JSON: {"disease_match": "...", "intervention_match": "...", "evidence": "..."}"""

_ANSWER_V2_1 = """Write a short answer to the user's question using ONLY the verified trials listed below.
Each trial is marked MATCH (fits the question) or PARTIAL (broader population or drug class only) and comes with an evidence quote.
- Mention EVERY listed trial exactly once, with its citation [NCT ID](https://clinicaltrials.gov/study/NCT ID).
- First list the MATCH trials, one short line each saying what the trial studies.
- Then, if there are PARTIAL trials, write "Partial matches:" and list them the same way, saying briefly what is broader.
- If there are no MATCH trials, start with "No trial fully matches the question." and then list the partial matches.
- Do not mention any other trial, do not add details that are not in the list, and do not add requirements the question does not state.
Keep each line short."""

# v2.2 (after dev review of v2.1): "class_level" was still chosen when the
# trial names the question's drug, and for specific members of a class the
# question asks for; "broader" was used for related but different diseases.
_VERIFY_V2_2 = """You check whether ONE clinical trial matches the core concepts of a search question.
Judge ONLY the disease and the intervention. Recruitment status, phase, age, sex and study type were already checked by the search system: do NOT judge them.

disease_match:
- "exact": the trial's population has the question's condition. This includes any type, subtype, stage, complication or specific example of it: narrower is exact. Examples: question "diabetes", trial "gestational diabetes" -> exact; question "cancer", trial "bladder cancer" -> exact.
- "broader": ONLY when the trial's population is a wider disease group that explicitly includes patients with the question's condition. Example: question "NSCLC", trial "advanced solid tumors" -> broader.
- "absent": a different disease (even a related one or a risk factor); the condition appears only in exclusion criteria or background text; or the population is a general, healthy or mixed cohort not defined by that disease.
- "not_applicable": the question names no disease.

intervention_match - decide in this order:
1. The question names no intervention -> "not_applicable".
2. The question names a CLASS or category (e.g. "immunotherapy", "targeted therapy", "SGLT2 inhibitors", "anticoagulants"): "exact" if participants receive any specific member of that category (any named drug, vaccine, cell therapy or device of that kind); otherwise "absent". Never use "class_level" for a class question.
3. The question names a SPECIFIC drug: look for that drug's name (or its brand or code name) in the trial text. If participants receive it - alone, in a combination such as "X + chemotherapy", as a comparator, or as one of several allowed options such as "X/Y" - it is "exact". Only if the drug's name does not appear and the trial refers only to its class (e.g. "PD-1 inhibitors") is it "class_level". Otherwise "absent", including when the drug appears only as prior, concurrent or excluded therapy.

Do not add requirements the question does not state (for example "primary intervention", "standalone", "treatment trial").
"evidence" must be a short quote (5-25 words) copied exactly, character for character, from the trial text.
Return only JSON: {"disease_match": "...", "intervention_match": "...", "evidence": "..."}"""

PROMPTS: dict[str, dict[str, str]] = {
    "v2.0": {"concepts": _CONCEPTS_V2_0, "verify": _VERIFY_V2_0, "answer": _ANSWER_V2_0},
    "v2.1": {"concepts": _CONCEPTS_V2_0, "verify": _VERIFY_V2_1, "answer": _ANSWER_V2_1},
    "v2.2": {"concepts": _CONCEPTS_V2_0, "verify": _VERIFY_V2_2, "answer": _ANSWER_V2_1},
}
DEFAULT_PROMPT_VERSION = "v2.2"   # chosen on dev (docs/rag_v2_prompt_log.md)


# =============================================================
# Helpers
# =============================================================

def _short(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " ..."


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, collapse whitespace (for quote matching)."""
    text = str(text).lower()
    text = re.sub(r"[‘’‚‛`´]", "'", text)
    text = re.sub(r"[“”„‟]", '"', text)
    text = re.sub(r"[‐-―−]", "-", text)
    text = text.replace("...", " ")
    return " ".join(text.split()).strip(" \"'.,;:")


def quote_found(quote: str, trial_text: str) -> bool:
    quote_n = normalize(quote)
    return len(quote_n) >= MIN_QUOTE_CHARS and quote_n in normalize(trial_text)


def extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def derive_grade(disease: str, intervention: str) -> int:
    """Same rule as the labeling rubric: all exact -> 2, any absent -> 0, else 1."""
    applicable = [v for v in (disease, intervention) if v != "not_applicable"]
    if not applicable or "absent" in applicable:
        return 0
    return 2 if all(v == "exact" for v in applicable) else 1


class TrialRecords:
    """Compact trial text for verification (processed v2 fields)."""

    def __init__(self, path: Path = TRIAL_FIELDS_PATH) -> None:
        self.frame = pd.read_parquet(path).set_index("nct_id")

    def text(self, nct_id: str) -> str:
        row = self.frame.loc[nct_id]
        parts = [
            f"Title: {_short(row['title'], 300)}",
            f"Official title: {_short(row['official_title'], 400)}",
            f"Conditions: {_short(row['conditions'], 300)}",
            f"Interventions: {_short(row['interventions'], 400)}",
            f"Intervention other names: {_short(row['intervention_other_names'], 200)}",
            f"Summary: {_short(row['summary'], 1200)}",
            f"Eligibility: {_short(row['eligibility'], 1500)}",
        ]
        return "\n".join(p for p in parts if not p.endswith(": "))


# =============================================================
# Pipeline
# =============================================================

class RAGv2:

    def __init__(self, llm: LocalLLM, prompt_version: str = DEFAULT_PROMPT_VERSION,
                 records: TrialRecords | None = None) -> None:
        self.llm = llm
        self.prompt_version = prompt_version
        self.prompts = PROMPTS[prompt_version]
        self.records = records or TrialRecords()

    @torch.inference_mode()
    def _generate_batch(self, conversations: list[list[dict]], max_new_tokens: int) -> list[str]:
        tokenizer, model = self.llm.tokenizer, self.llm.model
        prompts = [tokenizer.apply_chat_template(c, add_generation_prompt=True, tokenize=False)
                   for c in conversations]
        original_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(self.llm.device)
        finally:
            tokenizer.padding_side = original_side
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                temperature=None, top_p=None, top_k=None,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens]

    def extract_concepts(self, query: str) -> dict[str, str | None]:
        raw = self._generate_batch([[{"role": "system", "content": self.prompts["concepts"]},
                                     {"role": "user", "content": f"Question: {query}"}]], 120)[0]
        parsed = extract_json(raw) or {}
        clean = lambda v: (str(v).strip() or None) if v not in (None, "", "null", "None") else None
        return {"disease": clean(parsed.get("disease")), "intervention": clean(parsed.get("intervention")),
                "raw": raw}

    def verify(self, query: str, concepts: dict, trials: list[dict]) -> list[dict]:
        concept_line = (f"disease = {concepts['disease'] or 'none (not_applicable)'}; "
                        f"intervention = {concepts['intervention'] or 'none (not_applicable)'}")
        texts = [self.records.text(t["nct_id"]) for t in trials]
        conversations = [[{"role": "system", "content": self.prompts["verify"]},
                          {"role": "user", "content": f"Question: {query}\nQuestion concepts: {concept_line}"
                                                      f"\n\nTrial text:\n{text}"}] for text in texts]
        outputs = self._generate_batch(conversations, 160) if conversations else []

        results = []
        for trial, text, raw in zip(trials, texts, outputs):
            parsed = extract_json(raw) or {}
            disease = str(parsed.get("disease_match", "")).strip().lower().replace(" ", "_")
            intervention = str(parsed.get("intervention_match", "")).strip().lower().replace(" ", "_")
            # Concepts the question does not have are not applicable, whatever the model says.
            if concepts["disease"] is None:
                disease = "not_applicable"
            if concepts["intervention"] is None:
                intervention = "not_applicable"
            evidence = str(parsed.get("evidence", "")).strip()
            valid_labels = disease in DISEASE_LABELS and intervention in INTERVENTION_LABELS
            found = quote_found(evidence, text)
            grade = derive_grade(disease, intervention) if valid_labels else 0
            results.append({
                "nct_id": trial["nct_id"], "disease_match": disease, "intervention_match": intervention,
                "evidence": evidence, "quote_found": found, "valid_labels": valid_labels,
                "model_grade": grade,
                # Verified grade: dropped (0) when labels are invalid or the quote is not in the text.
                "verified_grade": grade if (valid_labels and found) else 0,
                "dropped": valid_labels and grade > 0 and not found,
                "raw": raw,
            })
        return results

    def answer(self, query: str, trials: list[dict], verifications: list[dict]) -> str:
        verified = [v for v in verifications if v["verified_grade"] >= 1]
        if not verified:
            return NO_MATCH_TEXT
        by_id = {t["nct_id"]: t for t in trials}
        lines = []
        for v in sorted(verified, key=lambda v: -v["verified_grade"]):
            t = by_id[v["nct_id"]]
            label = "MATCH" if v["verified_grade"] == 2 else "PARTIAL"
            lines.append(f"[{label}] {t['nct_id']}: {_short(t['title'], 200)}\n"
                         f"Conditions: {_short(t['conditions'], 150)} | Interventions: "
                         f"{_short(t['interventions'], 150)} | Status: {t['status']} | Phase: {t['phase']}\n"
                         f"Evidence: \"{v['evidence']}\"")
        user = f"Question: {query}\n\nVerified trials:\n\n" + "\n\n".join(lines) + "\n\nAnswer:"
        return self._generate_batch([[{"role": "system", "content": self.prompts["answer"]},
                                      {"role": "user", "content": user}]], 350)[0]

    def run(self, query: str, trials: list[dict]) -> dict[str, Any]:
        start = time.perf_counter()
        concepts = self.extract_concepts(query)
        verifications = self.verify(query, concepts, trials)
        answer = self.answer(query, trials, verifications)
        cited = list(dict.fromkeys(re.findall(r"NCT\d{8}", answer)))
        verified_ids = [v["nct_id"] for v in verifications if v["verified_grade"] >= 1]
        return {
            "prompt_version": self.prompt_version,
            "concepts": {k: concepts[k] for k in ("disease", "intervention")},
            "verifications": verifications,
            "answer": answer,
            "cited": cited,
            "invalid_citations": [n for n in cited if n not in [t["nct_id"] for t in trials]],
            "cited_unverified": [n for n in cited if n not in verified_ids],
            "says_no_match": not verified_ids,
            "latency_s": round(time.perf_counter() - start, 2),
        }
