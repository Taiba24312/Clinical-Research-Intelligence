"""
Rule-based query parser v1.2 = parser v1.1 + age groups.

Adds (age semantics "admits": the trial's age range must overlap):
  * pediatric/paediatric, children/child, kids  -> max_age 18 (exclusive)
  * adolescents/adolescent, teenagers/teens     -> min_age 12, max_age 18
  * infants/infant                              -> max_age 2
  * older adults, elderly, seniors, geriatric   -> min_age 65
  * "over N", "older than N", "aged N and above/older", "N years or older"
                                                -> min_age N
  * "under N", "younger than N"                 -> max_age N
  * a child cue together with "adults"          -> no age filter (both groups)

Filter keys: min_age (trial max age >= value, or no maximum) and max_age
(trial min age < value, or no minimum), applied by MetadataFilter.apply.
v1 and v1.1 are unchanged.
"""

from __future__ import annotations

import re
from typing import Any

from src.query_understanding.parser_v1_1 import parse_query_v1_1


PARSER_VERSION = "v1.2"

CHILD = r"\b(?:pa?ediatric|children|child|kids?)\b"
ADOLESCENT = r"\b(?:adolescents?|teenagers?|teens?)\b"
INFANT = r"\binfants?\b"
OLDER = r"\b(?:older\s+adults?|older\s+patients?|elderly|seniors?|geriatric)\b"
ADULT = r"\badults?\b"
AGE_NUMBER = r"(\d{1,3})(?:\s*years?(?:\s*old)?)?"
OVER = [rf"\b(?:over|older\s+than|above)\s+{AGE_NUMBER}",
        rf"\baged\s+{AGE_NUMBER}\s+(?:and|or)\s+(?:above|older|over)\b",
        rf"\b{AGE_NUMBER}\s+(?:and|or)\s+(?:older|above|over)\b"]
UNDER = [rf"\b(?:under|younger\s+than|below)\s+{AGE_NUMBER}"]


AGE_FOLLOWERS = {"years", "year", "yrs", "old", "and", "or", "with", "who", "in", "of"}


def _first_age(patterns: list[str], text: str) -> tuple[int | None, str | None]:
    """First plausible age; a number followed by another noun ("over 100 participants") is not an age."""
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            following = re.match(r"\s*([a-z]+)", text[match.end():])
            if following and following.group(1) not in AGE_FOLLOWERS:
                continue
            if 0 < int(match.group(1)) <= 120:
                return int(match.group(1)), match.group(0)
    return None, None


def parse_query_v1_2(query: str) -> dict[str, Any]:
    parsed = parse_query_v1_1(query)
    text = query.strip().lower()
    filters = dict(parsed["filters"])
    semantic = parsed["semantic_query"]
    cues: list[str] = []

    min_age, max_age = filters.pop("min_age", None), None
    over, over_text = _first_age(OVER, text)
    under, under_text = _first_age(UNDER, text)

    if re.search(INFANT, text):
        min_age, max_age = None, 2
        cues.append(INFANT)
    elif re.search(ADOLESCENT, text):
        min_age, max_age = 12, 18
        cues.append(ADOLESCENT)
    elif re.search(CHILD, text):
        if re.search(ADULT, text):
            min_age, max_age = None, None            # children and adults: both groups
        else:
            min_age, max_age = None, 18
        cues += [CHILD, ADULT]
    elif re.search(OLDER, text):
        min_age = 65
        cues += [OLDER, r"\bolder\b"]          # v1.1 already removed "adults"
    if over is not None:
        min_age = over
        cues.append(re.escape(over_text))
    if under is not None:
        max_age = under
        cues.append(re.escape(under_text))

    if min_age is not None:
        filters["min_age"] = min_age
    if max_age is not None:
        filters["max_age"] = max_age

    for pattern in cues:
        semantic = re.sub(pattern, " ", semantic)
    semantic = re.sub(r"\b(?:with|in|for)\s*$", " ", re.sub(r"\s+", " ", semantic).strip())
    semantic = re.sub(r"\s+", " ", semantic).strip(" ,.-?")
    return {"semantic_query": semantic or parsed["semantic_query"], "filters": filters}
