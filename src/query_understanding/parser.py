from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ParsedQuery:
    semantic_query: str
    filters: dict[str, Any]


class QueryParser:
    """
    Rule-based parser that separates a natural-language trial query into:

    1. semantic_query -> sent to BM25 / dense retrieval
    2. filters -> applied to structured trial metadata
    """

    def parse(self, query: str) -> ParsedQuery:
        if not query or not query.strip():
            raise ValueError("Query must be non-empty.")

        original = query.strip()
        text = original.lower()

        filters: dict[str, Any] = {}

        # -------------------------
        # Recruitment status
        # -------------------------
        if re.search(r"\brecruiting\b", text):
            filters["status"] = ["RECRUITING"]

        # -------------------------
        # Phase
        # -------------------------
        phase_match = re.search(r"\bphase\s*([123])\b", text)

        if phase_match:
            phase_number = phase_match.group(1)
            filters["phase"] = [f"PHASE{phase_number}"]

        # -------------------------
        # Study type
        # -------------------------
        if re.search(r"\bobservational\b", text):
            filters["study_type"] = ["OBSERVATIONAL"]

        # -------------------------
        # Sex
        # -------------------------
        if re.search(r"\bfemale\b|\bwomen\b", text):
            filters["sex"] = ["FEMALE"]

        elif re.search(r"\bmale\b|\bmen\b", text):
            filters["sex"] = ["MALE"]

        # -------------------------
        # Adult population
        # -------------------------
        if re.search(r"\badults?\b", text):
            filters["min_age"] = 18

        # -------------------------
        # Build semantic query
        # -------------------------
        semantic_query = text

        removal_patterns = [
            r"\brecruiting\b",
            r"\bphase\s*[123]\b",
            r"\bobservational\b",
            r"\bfor\s+adults?\b",
            r"\bin\s+adults?\b",
            r"\badults?\b",
            r"\bfemale\s+(?:patients?|participants?)\b",
            r"\bmale\s+(?:patients?|participants?)\b",
        ]

        for pattern in removal_patterns:
            semantic_query = re.sub(pattern, " ", semantic_query)

        # Generic structural words are not useful for semantic retrieval.
        semantic_query = re.sub(
            r"\b(trials?|studies?)\b",
            " ",
            semantic_query,
        )

        # Clean whitespace / punctuation.
        semantic_query = re.sub(r"\s+", " ", semantic_query).strip()
        semantic_query = semantic_query.strip(" ,.-")

        if not semantic_query:
            semantic_query = original.lower()

        return ParsedQuery(
            semantic_query=semantic_query,
            filters=filters,
        )


def parse_query(query: str) -> dict[str, Any]:
    """
    Convenience wrapper around QueryParser.
    """
    parsed = QueryParser().parse(query)

    return {
        "semantic_query": parsed.semantic_query,
        "filters": parsed.filters,
    }