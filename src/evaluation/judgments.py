"""
Loading and validation of evaluation inputs: query specs, qrels and runs.

File formats are documented in src/evaluation/retrieval_metrics.py.

The relevance grade is never stored by hand. It is derived from the two
manually judged core-concept fields by derive_grade():

    2 = every applicable core concept is "exact" (the trial's own focus)
    1 = every applicable core concept is at least "broader"/"class_level",
        none is "absent_or_different"
    0 = any applicable core concept is "absent_or_different"

Labeling convention for mention_role: if a concept appears in the trial
only as a prior / concurrent / excluded therapy or as background, the
labeler records the match field as "absent_or_different" and uses
mention_role to say why. A non-focus mention_role combined with no
"absent_or_different" field is rejected as inconsistent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


DISEASE_VALUES = ("exact", "broader", "absent_or_different", "not_applicable")
INTERVENTION_VALUES = (
    "exact", "class_level", "absent_or_different", "not_applicable",
)
MENTION_ROLES = (
    "", "focus", "prior_therapy", "concurrent_therapy", "excluded", "background",
)
NON_FOCUS_ROLES = set(MENTION_ROLES) - {"", "focus"}

QRELS_COLUMNS = [
    "query_id", "nct_id", "disease_match", "intervention_match",
    "mention_role", "notes", "labeler", "label_date",
]
RUN_COLUMNS = ["query_id", "system", "nct_id", "rank", "score"]

STRUCTURED_CONSTRAINT_KEYS = ("status", "phase", "sex", "study_type", "age_range")
SPLITS = ("dev", "test")


# =============================================================
# Query specs
# =============================================================

@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    text: str
    core_concepts: dict[str, Any]
    structured_constraints: dict[str, Any]
    category: tuple[str, ...]
    split: str
    has_answer: bool

    @property
    def has_disease(self) -> bool:
        return self.core_concepts.get("disease") is not None

    @property
    def has_intervention(self) -> bool:
        return self.core_concepts.get("intervention") is not None


def parse_query_specs(items: list[dict]) -> dict[str, QuerySpec]:
    """Validate raw query-spec dictionaries and return them by query_id."""

    specs: dict[str, QuerySpec] = {}

    for item in items:
        query_id = item["query_id"]
        if query_id in specs:
            raise ValueError(f"Duplicate query_id in query specs: {query_id}")

        concepts = item.get("core_concepts") or {}
        unknown = set(concepts) - {"disease", "intervention"}
        if unknown:
            raise ValueError(f"{query_id}: unknown core concepts {sorted(unknown)}")
        if concepts.get("disease") is None and concepts.get("intervention") is None:
            raise ValueError(
                f"{query_id}: at least one core concept (disease or "
                "intervention) is required; a grade cannot be derived otherwise."
            )

        constraints = item.get("structured_constraints") or {}
        unknown = set(constraints) - set(STRUCTURED_CONSTRAINT_KEYS)
        if unknown:
            raise ValueError(
                f"{query_id}: unknown structured constraints {sorted(unknown)}"
            )

        split = item["split"]
        if split not in SPLITS:
            raise ValueError(f"{query_id}: split must be one of {SPLITS}")

        category = item.get("category", ())
        if isinstance(category, str):
            category = (category,)

        specs[query_id] = QuerySpec(
            query_id=query_id,
            text=item["text"],
            core_concepts=concepts,
            structured_constraints=constraints,
            category=tuple(category),
            split=split,
            has_answer=bool(item["has_answer"]),
        )

    return specs


def load_query_specs(path: str | Path) -> dict[str, QuerySpec]:
    with Path(path).open(encoding="utf-8") as handle:
        return parse_query_specs(json.load(handle))


# =============================================================
# Qrels
# =============================================================

def derive_grade(
    disease_match: str,
    intervention_match: str,
    mention_role: str = "",
) -> int:
    """Derive the 0/1/2 relevance grade from the core-concept judgments."""

    if disease_match not in DISEASE_VALUES:
        raise ValueError(f"Invalid disease_match: {disease_match!r}")
    if intervention_match not in INTERVENTION_VALUES:
        raise ValueError(f"Invalid intervention_match: {intervention_match!r}")
    if mention_role not in MENTION_ROLES:
        raise ValueError(f"Invalid mention_role: {mention_role!r}")

    applicable = [
        value for value in (disease_match, intervention_match)
        if value != "not_applicable"
    ]
    if not applicable:
        raise ValueError("Both core concepts are not_applicable.")

    if "absent_or_different" in applicable:
        return 0

    if mention_role in NON_FOCUS_ROLES:
        raise ValueError(
            f"mention_role={mention_role!r} but no concept is "
            "absent_or_different; record the non-focus concept as "
            "absent_or_different."
        )

    if all(value == "exact" for value in applicable):
        return 2
    return 1


def prepare_qrels(
    qrels: pd.DataFrame,
    specs: dict[str, QuerySpec],
) -> pd.DataFrame:
    """
    Validate qrels against the query specs and add a derived 'grade' column.

    Rows where disease_match or intervention_match is blank are pending
    (pooled but not yet fully judged; templates pre-fill "not_applicable"
    for a concept the query lacks): they are dropped and counted in the
    returned frame's attrs["pending_rows"].
    """

    missing = set(QRELS_COLUMNS) - set(qrels.columns)
    if missing:
        raise ValueError(f"Qrels missing columns: {sorted(missing)}")

    qrels = qrels[QRELS_COLUMNS].copy()
    for column in QRELS_COLUMNS:
        qrels[column] = qrels[column].fillna("").astype(str).str.strip()

    pending = (qrels["disease_match"] == "") | (qrels["intervention_match"] == "")
    judged = qrels.loc[~pending].copy()

    if judged.duplicated(["query_id", "nct_id"]).any():
        raise ValueError("Duplicate (query_id, nct_id) judgments in qrels.")

    unknown_queries = set(judged["query_id"]) - set(specs)
    if unknown_queries:
        raise ValueError(f"Qrels reference unknown queries: {sorted(unknown_queries)}")

    for row in judged.itertuples(index=False):
        spec = specs[row.query_id]
        for concept, value, present in (
            ("disease", row.disease_match, spec.has_disease),
            ("intervention", row.intervention_match, spec.has_intervention),
        ):
            if present and value == "not_applicable":
                raise ValueError(
                    f"{row.query_id}/{row.nct_id}: query has a {concept} "
                    f"concept but {concept}_match is not_applicable."
                )
            if not present and value != "not_applicable":
                raise ValueError(
                    f"{row.query_id}/{row.nct_id}: query has no {concept} "
                    f"concept, so {concept}_match must be not_applicable."
                )

    judged["grade"] = [
        derive_grade(d, i, m)
        for d, i, m in zip(
            judged["disease_match"],
            judged["intervention_match"],
            judged["mention_role"],
        )
    ]
    judged = judged.reset_index(drop=True)
    judged.attrs["pending_rows"] = int(pending.sum())
    return judged


def load_qrels(path: str | Path, specs: dict[str, QuerySpec]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    return prepare_qrels(frame, specs)


# =============================================================
# Runs
# =============================================================

def prepare_run(run: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a run and sort it by (query_id, system, rank).

    The order is defined by 'rank' alone. Tied scores are fine as long as
    ranks are distinct; duplicate ranks or duplicate trials within one
    (query_id, system) list are errors, because they make metrics ambiguous.
    """

    missing = set(RUN_COLUMNS) - set(run.columns)
    if missing:
        raise ValueError(f"Run missing columns: {sorted(missing)}")

    run = run[RUN_COLUMNS].copy()
    run["query_id"] = run["query_id"].astype(str)
    run["system"] = run["system"].astype(str)
    run["nct_id"] = run["nct_id"].astype(str)
    run["rank"] = pd.to_numeric(run["rank"], errors="raise")
    run["score"] = pd.to_numeric(run["score"], errors="coerce")

    if (run["rank"] < 1).any() or (run["rank"] % 1 != 0).any():
        raise ValueError("Ranks must be positive integers.")
    run["rank"] = run["rank"].astype(int)

    keys = ["query_id", "system"]
    if run.duplicated(keys + ["rank"]).any():
        raise ValueError("Duplicate ranks within a (query_id, system) list.")
    if run.duplicated(keys + ["nct_id"]).any():
        raise ValueError("Duplicate nct_id within a (query_id, system) list.")

    return run.sort_values(keys + ["rank"], kind="stable").reset_index(drop=True)


def load_run(path: str | Path) -> pd.DataFrame:
    return prepare_run(pd.read_csv(path))
