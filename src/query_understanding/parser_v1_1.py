"""
Rule-based query parser v1.1 - a small fix of parser.py (v1, kept unchanged).

Changes from v1:
  * status: "not yet recruiting" (also "not-yet-recruiting", "not yet
    enrolling/open") -> NOT_YET_RECRUITING instead of RECRUITING;
    "currently/actively enrolling", "enrolling now" -> RECRUITING;
    "not recruiting" (without "yet") is ambiguous -> no status filter.
  * phase: Roman numerals (I-IV) and phase 4 are recognized.
  * study type: "non-interventional" -> OBSERVATIONAL;
    "excluding observational" / "non-observational" / "not observational"
    -> INTERVENTIONAL or EXPANDED_ACCESS; "interventional" -> INTERVENTIONAL.

Output format is the same as v1 ({"semantic_query", "filters"}), so it
works with MetadataFilter.apply unchanged.
"""

from __future__ import annotations

import re
from typing import Any


PARSER_VERSION = "v1.1"

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}

NOT_YET = r"\bnot[\s-]+yet[\s-]+(?:recruiting|enrolling|open)\b"
NOT_RECRUITING = r"\bnot\s+recruiting\b"
RECRUITING = (r"\b(?:currently|actively)\s+(?:recruiting|enrolling)\b"
              r"|\benrolling\s+now\b|\brecruiting\b")
PHASE = r"\bphase\s*(iv|i{1,3}|[1-4])\b"
NON_OBSERVATIONAL = r"\b(?:non[\s-]?observational|not\s+observational|excluding\s+observational)\b"
NON_INTERVENTIONAL = r"\bnon[\s-]?interventional\b"
OBSERVATIONAL = r"\bobservational\b"
INTERVENTIONAL = r"\binterventional\b"


def parse_query_v1_1(query: str) -> dict[str, Any]:
    if not query or not query.strip():
        raise ValueError("Query must be non-empty.")

    original = query.strip()
    text = original.lower()
    filters: dict[str, Any] = {}

    # Status: the more specific patterns are checked first.
    if re.search(NOT_YET, text):
        filters["status"] = ["NOT_YET_RECRUITING"]
    elif re.search(NOT_RECRUITING, text):
        pass                                   # ambiguous: no status filter
    elif re.search(RECRUITING, text):
        filters["status"] = ["RECRUITING"]

    phase = re.search(PHASE, text)
    if phase:
        number = ROMAN.get(phase.group(1), phase.group(1))
        filters["phase"] = [f"PHASE{number}"]

    if re.search(NON_OBSERVATIONAL, text):
        filters["study_type"] = ["INTERVENTIONAL", "EXPANDED_ACCESS"]
    elif re.search(NON_INTERVENTIONAL, text):
        filters["study_type"] = ["OBSERVATIONAL"]
    elif re.search(OBSERVATIONAL, text):
        filters["study_type"] = ["OBSERVATIONAL"]
    elif re.search(INTERVENTIONAL, text):
        filters["study_type"] = ["INTERVENTIONAL"]

    # Sex and adult age: unchanged from v1.
    if re.search(r"\bfemale\b|\bwomen\b", text):
        filters["sex"] = ["FEMALE"]
    elif re.search(r"\bmale\b|\bmen\b", text):
        filters["sex"] = ["MALE"]
    if re.search(r"\badults?\b", text):
        filters["min_age"] = 18

    # Remove recognized cues from the text used for retrieval.
    semantic = text
    for pattern in [
        NOT_YET, NOT_RECRUITING,
        r"\b(?:currently|actively)\s+(?:recruiting|enrolling)\b", r"\benrolling\s+now\b",
        r"\brecruiting\b", PHASE, NON_OBSERVATIONAL, NON_INTERVENTIONAL,
        OBSERVATIONAL, INTERVENTIONAL,
        r"\bfor\s+adults?\b", r"\bin\s+adults?\b", r"\badults?\b",
        r"\bfemale\s+(?:patients?|participants?)\b", r"\bmale\s+(?:patients?|participants?)\b",
        r"\b(?:trials?|studies)\b",
    ]:
        semantic = re.sub(pattern, " ", semantic)
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,.-?")
    if not semantic:
        semantic = text

    return {"semantic_query": semantic, "filters": filters}
