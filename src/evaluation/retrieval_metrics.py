"""
Retrieval evaluation for the Clinical Research Intelligence Assistant.

FILE FORMATS  (loaders and validation: src/evaluation/judgments.py)
------------
Query spec - JSON list, one object per query:
    {
      "query_id": "Q01",
      "text": "Recruiting NSCLC trials involving pembrolizumab",
      "core_concepts": {
        "disease": {"concept": "non-small cell lung cancer",
                    "synonyms": ["NSCLC"]},          # or null
        "intervention": {"concept": "pembrolizumab",
                         "synonyms": ["Keytruda", "MK-3475"]}  # or null
      },
      "structured_constraints": {                     # all keys optional
        "status": ["RECRUITING"],
        "phase": ["PHASE2", "PHASE3"],                # any listed phase
        "sex": ["FEMALE", "ALL"],
        "study_type": ["OBSERVATIONAL"],
        "age_range": [18, null]                       # [lo, hi) in years
      },
      "category": ["abbreviation", "multi_constraint"],
      "split": "dev",                                 # or "test"
      "has_answer": true
    }
    At least one core concept must be non-null.
    age_range: the query population is ages lo <= age < hi (null = open).
    adults = [18, null], pediatric = [0, 18], older adults = [65, null].

Qrels - CSV, one row per judged (query, trial) pair:
    query_id, nct_id, disease_match, intervention_match, mention_role,
    notes, labeler, label_date
    disease_match      : exact | broader | absent_or_different | not_applicable
    intervention_match : exact | class_level | absent_or_different | not_applicable
    mention_role       : "" | focus | prior_therapy | concurrent_therapy |
                         excluded | background
    The 0/1/2 grade is derived in code (judgments.derive_grade), never stored.
    Rows with both match fields blank are pending (pooled, not yet judged).

Run - CSV, one row per retrieved trial:
    query_id, system, nct_id, rank, score
    Order is defined by rank (1 = best). Ranks must be unique per list.
    A query missing from a system's run counts as an empty result list
    (abstention), not as a skipped query.

Trial metadata - trials.parquet (status, phase, sex, study_type, min_age,
max_age), keyed by nct_id.

EFFECTIVE GRADE
---------------
Structured constraints are not labeled by hand. For every trial:
    effective grade = 0                if it violates any structured constraint
                    = derived grade    if judged and no violation
                    = unjudged (None)  otherwise
So a trial counts as relevant only if it matches the core concepts AND the
query's structured constraints. Violators are "resolved" without a label.

The constraint checker here is deliberately independent of
src/retrieval/metadata_filter.py: the evaluation must not share code (and
bugs) with the system under test.

METRICS  (per query; macro = mean over queries where the metric is defined)
-------
Gain: 2**grade - 1  (grade 2 -> 3, grade 1 -> 1, grade 0 -> 0)
nDCG@K = DCG@K / IDCG@K,  DCG@K = sum_{i<=K} gain_i / log2(i + 1)
    IDCG uses ALL judged trials for the query (effective grades), not only
    the retrieved ones. Unjudged retrieved trials contribute gain 0.
P@K (strict: grade = 2, lenient: grade >= 1) = hits in top K / K.
    The denominator is K even when fewer than K trials are returned.
Recall@K = hits in top K / judged relevant trials at that threshold.
MRR = 1 / rank of the first trial meeting the threshold (0 if none).
Judged@K = share of the returned top-K trials with a known effective grade.
cond_* = the same metrics on the condensed list (unjudged trials removed
    before cutting at K), to show how much unjudged trials matter.
violation_rate@K = share of the returned top-K trials violating at least
    one structured constraint (details in EvaluationResult.violations).

Undefined values are NaN and are excluded from macro averages:
  * zero-answer queries (has_answer = false): nDCG, P, Recall, MRR are NaN;
    they are summarized separately (grade-0 share, best score returned).
  * strict P/Recall/MRR are NaN for a query with no judged grade-2 trial;
    lenient ones likewise for no judged grade >= 1 trial; nDCG is NaN when
    IDCG = 0.
  * judged@K and violation_rate@K are NaN for an empty result list.
"""

from __future__ import annotations

import itertools
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.judgments import QuerySpec


THRESHOLDS = {"strict": 2, "lenient": 1}
DEFAULT_KS = (5, 10, 20)
DEFAULT_SEED = 20261030
DEFAULT_N_BOOT = 10_000


def gain(grade: int) -> float:
    """Exponential gain: separates full matches (3) from partial ones (1)."""
    return float(2 ** grade - 1)


# =============================================================
# Structured constraints
# =============================================================

AGE_UNITS_PER_YEAR = {
    "year": 1.0, "month": 12.0, "week": 52.1775,
    "day": 365.25, "hour": 8766.0, "minute": 525960.0,
}
AGE_PATTERN = re.compile(
    r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$",
    re.IGNORECASE,
)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return bool(pd.isna(value))


def _as_list(value: Any) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return [value]
    return list(value)


def parse_age_years(value: Any) -> float | None:
    """'18 Years' -> 18.0, '6 Months' -> 0.5, missing -> None (no limit)."""

    if _is_missing(value) or str(value).strip().upper() in {"N/A", "NA"}:
        return None
    match = AGE_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Unparseable age value: {value!r}")
    return float(match.group(1)) / AGE_UNITS_PER_YEAR[match.group(2).lower()]


def constraint_violations(
    trial: Mapping[str, Any],
    constraints: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return one {constraint, required, actual} dict per violated constraint."""

    violations = []

    def violated(name: str, required: Any, actual: Any) -> None:
        violations.append({
            "constraint": name,
            "required": "|".join(str(v) for v in _as_list(required))
            if name != "age_range" else str(list(required)),
            "actual": "missing" if _is_missing(actual) else str(actual),
        })

    for name, required in constraints.items():
        if name in ("status", "sex", "study_type"):
            actual = trial.get(name)
            if _is_missing(actual) or actual not in _as_list(required):
                violated(name, required, actual)

        elif name == "phase":
            actual = trial.get("phase")
            phases = set() if _is_missing(actual) else {
                part.strip() for part in str(actual).split(";") if part.strip()
            }
            if not phases & set(_as_list(required)):
                violated(name, required, actual)

        elif name == "age_range":
            low, high = required
            low = 0.0 if low is None else float(low)
            high = math.inf if high is None else float(high)
            trial_min = parse_age_years(trial.get("min_age"))
            trial_max = parse_age_years(trial.get("max_age"))
            trial_min = 0.0 if trial_min is None else trial_min
            trial_max = math.inf if trial_max is None else trial_max
            # Query population is [low, high); trial accepts [min, max].
            if trial_min >= high or trial_max < low:
                violated(
                    name, required,
                    f"{trial.get('min_age') or 'none'} to "
                    f"{trial.get('max_age') or 'none'}",
                )
        else:
            raise ValueError(f"Unknown structured constraint: {name!r}")

    return violations


# =============================================================
# Per-list metric primitives (grades: int effective grade or None)
# =============================================================

Grades = Sequence[int | None]


def dcg_at_k(grades: Grades, k: int) -> float:
    return sum(
        gain(grade or 0) / math.log2(position + 2)
        for position, grade in enumerate(grades[:k])
    )


def ndcg_at_k(grades: Grades, ideal_grades: Sequence[int], k: int) -> float:
    ideal = dcg_at_k(sorted(ideal_grades, reverse=True), k)
    return math.nan if ideal == 0 else dcg_at_k(grades, k) / ideal


def _hits(grades: Grades, threshold: int, k: int) -> int:
    return sum(1 for g in grades[:k] if g is not None and g >= threshold)


def precision_at_k(grades: Grades, threshold: int, k: int) -> float:
    return _hits(grades, threshold, k) / k


def recall_at_k(grades: Grades, threshold: int, k: int, n_relevant: int) -> float:
    return math.nan if n_relevant == 0 else _hits(grades, threshold, k) / n_relevant


def reciprocal_rank(grades: Grades, threshold: int, n_relevant: int) -> float:
    if n_relevant == 0:
        return math.nan
    for position, grade in enumerate(grades, start=1):
        if grade is not None and grade >= threshold:
            return 1.0 / position
    return 0.0


def judged_at_k(grades: Grades, k: int) -> float:
    top = grades[:k]
    return math.nan if not top else sum(g is not None for g in top) / len(top)


# =============================================================
# Query- and run-level evaluation
# =============================================================

@dataclass
class EvaluationResult:
    per_query: pd.DataFrame     # one row per (system, query_id)
    violations: pd.DataFrame    # one row per violated constraint in top max(K)
    zero_answer: pd.DataFrame   # one row per (system, zero-answer query)


def _metadata_records(metadata: pd.DataFrame) -> dict[str, dict]:
    if "nct_id" in metadata.columns:
        metadata = metadata.set_index("nct_id")
    if metadata.index.duplicated().any():
        raise ValueError("Trial metadata has duplicate nct_id values.")
    return metadata.to_dict("index")


def _trial(records: Mapping[str, dict], nct_id: str) -> dict:
    try:
        return records[nct_id]
    except KeyError:
        raise KeyError(f"{nct_id} is not in the trial metadata.") from None


def evaluate_query(
    ranked_ids: Sequence[str],
    spec: QuerySpec,
    judged_grades: Mapping[str, int],
    records: Mapping[str, dict],
    ks: Sequence[int] = DEFAULT_KS,
) -> tuple[dict[str, Any], list[list[dict]], list[int | None]]:
    """
    Evaluate one ranked list.

    Returns (metrics, violations per returned trial, effective grades).
    """

    constraints = spec.structured_constraints

    violations = [
        constraint_violations(_trial(records, nct_id), constraints)
        for nct_id in ranked_ids
    ]
    grades = [
        0 if violated else judged_grades.get(nct_id)
        for nct_id, violated in zip(ranked_ids, violations)
    ]

    ideal = [
        0 if constraint_violations(_trial(records, nct_id), constraints) else grade
        for nct_id, grade in judged_grades.items()
    ]
    n_relevant = {
        name: sum(g >= threshold for g in ideal)
        for name, threshold in THRESHOLDS.items()
    }

    if not spec.has_answer and n_relevant["lenient"] > 0:
        raise ValueError(
            f"{spec.query_id} is marked has_answer=false but has judged "
            "trials with effective grade >= 1."
        )

    row: dict[str, Any] = {
        "n_returned": len(ranked_ids),
        "n_judged_pool": len(judged_grades),
        "n_rel_strict": n_relevant["strict"],
        "n_rel_lenient": n_relevant["lenient"],
    }

    for k in ks:
        top = violations[:k]
        row[f"judged@{k}"] = judged_at_k(grades, k)
        row[f"violation_rate@{k}"] = (
            math.nan if not top else sum(bool(v) for v in top) / len(top)
        )
        counts = Counter(v["constraint"] for item in top for v in item)
        row[f"violated@{k}"] = ", ".join(
            f"{name}={count}" for name, count in sorted(counts.items())
        )

    condensed = [g for g in grades if g is not None]

    for prefix, ranked_grades in (("", grades), ("cond_", condensed)):
        for k in ks:
            row[f"{prefix}ndcg@{k}"] = (
                ndcg_at_k(ranked_grades, ideal, k) if spec.has_answer else math.nan
            )
        for name, threshold in THRESHOLDS.items():
            defined = spec.has_answer and n_relevant[name] > 0
            for k in ks:
                row[f"{prefix}p_{name}@{k}"] = (
                    precision_at_k(ranked_grades, threshold, k)
                    if defined else math.nan
                )
                row[f"{prefix}r_{name}@{k}"] = (
                    recall_at_k(ranked_grades, threshold, k, n_relevant[name])
                    if defined else math.nan
                )
            row[f"{prefix}mrr_{name}"] = (
                reciprocal_rank(ranked_grades, threshold, n_relevant[name])
                if defined else math.nan
            )

    return row, violations, grades


def evaluate_run(
    run: pd.DataFrame,
    qrels: pd.DataFrame,
    specs: Mapping[str, QuerySpec],
    metadata: pd.DataFrame,
    ks: Sequence[int] = DEFAULT_KS,
    query_ids: Sequence[str] | None = None,
) -> EvaluationResult:
    """
    Evaluate every system in a prepared run (judgments.prepare_run) against
    prepared qrels (judgments.prepare_qrels).

    Every query in `query_ids` (default: all specs) is evaluated for every
    system; a query with no rows for a system is an empty result list.
    """

    ks = sorted(set(ks))
    records = _metadata_records(metadata)
    query_ids = list(specs) if query_ids is None else list(query_ids)

    judged_by_query = {
        query_id: dict(zip(group["nct_id"], group["grade"]))
        for query_id, group in qrels.groupby("query_id")
    }
    lists = {
        key: group for key, group in run.groupby(["system", "query_id"], sort=False)
    }

    per_query, violation_rows, zero_rows = [], [], []

    for system in run["system"].unique():
        for query_id in query_ids:
            spec = specs[query_id]
            group = lists.get((system, query_id), run.iloc[0:0])
            ranked_ids = group["nct_id"].tolist()
            scores = group["score"].tolist()

            metrics, violations, grades = evaluate_query(
                ranked_ids, spec, judged_by_query.get(query_id, {}), records, ks,
            )
            per_query.append({
                "system": system,
                "query_id": query_id,
                "split": spec.split,
                "category": "|".join(spec.category),
                "has_answer": spec.has_answer,
                **metrics,
            })

            for rank, (nct_id, items) in enumerate(
                zip(ranked_ids[:max(ks)], violations), start=1,
            ):
                for item in items:
                    violation_rows.append({
                        "system": system, "query_id": query_id,
                        "rank": rank, "nct_id": nct_id, **item,
                    })

            if not spec.has_answer:
                zero = {
                    "system": system,
                    "query_id": query_id,
                    "n_returned": len(ranked_ids),
                    "best_score": max(scores) if scores else math.nan,
                    "top1_nct_id": ranked_ids[0] if ranked_ids else "",
                }
                for k in ks:
                    top = grades[:k]
                    zero[f"grade0_share@{k}"] = (
                        math.nan if not top else sum(g == 0 for g in top) / len(top)
                    )
                    zero[f"unjudged_share@{k}"] = (
                        math.nan if not top else sum(g is None for g in top) / len(top)
                    )
                zero_rows.append(zero)

    return EvaluationResult(
        per_query=pd.DataFrame(per_query),
        violations=pd.DataFrame(
            violation_rows,
            columns=["system", "query_id", "rank", "nct_id",
                     "constraint", "required", "actual"],
        ),
        zero_answer=pd.DataFrame(zero_rows),
    )


# =============================================================
# Aggregation, confidence intervals, paired comparison
# =============================================================

def bootstrap_ci(
    values: Iterable[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Mean and percentile-bootstrap CI over queries (NaN values dropped)."""

    data = np.asarray(list(values), dtype=float)
    data = data[~np.isnan(data)]
    if data.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(data.mean())
    if data.size == 1:
        return mean, math.nan, math.nan

    rng = np.random.default_rng(seed)
    samples = data[rng.integers(0, data.size, size=(n_boot, data.size))].mean(axis=1)
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return mean, float(low), float(high)


def metric_columns(per_query: pd.DataFrame) -> list[str]:
    return [
        column for column in per_query.columns
        if ("@" in column or "mrr_" in column) and not column.startswith("violated@")
    ]


def macro_summary(
    per_query: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    split: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Macro-average each metric per system with a bootstrap CI.

    Each row also reports the mean Judged@K over the same queries (for MRR,
    the largest K evaluated) so unjudged-document effects stay visible.
    """

    frame = per_query if split is None else per_query[per_query["split"] == split]
    metrics = list(metrics) if metrics is not None else metric_columns(frame)
    judged_ks = sorted(int(c.split("@")[1]) for c in frame.columns if c.startswith("judged@"))

    rows = []
    for system, group in frame.groupby("system", sort=False):
        for metric in metrics:
            values = group[metric].astype(float)
            defined = values.notna()
            mean, low, high = bootstrap_ci(values[defined], n_boot, seed, alpha)

            k = int(metric.split("@")[1]) if "@" in metric else max(judged_ks)
            judged = group.loc[defined, f"judged@{k}"].astype(float).mean()

            rows.append({
                "system": system,
                "metric": metric,
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_queries": int(defined.sum()),
                "judged_k": k,
                "mean_judged": judged,
            })
    return pd.DataFrame(rows)


def paired_comparison(
    per_query: pd.DataFrame,
    system_a: str,
    system_b: str,
    metric: str,
    split: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = 0.05,
    exact_max_n: int = 16,
) -> dict[str, Any]:
    """
    Compare two systems on the queries where both have a defined value.

    diff = system_a - system_b per query. Reports the mean difference with a
    percentile-bootstrap CI, a two-sided paired sign-flip randomization test
    (exact enumeration for n <= exact_max_n, Monte Carlo otherwise) and
    per-query wins / ties / losses.
    """

    frame = per_query if split is None else per_query[per_query["split"] == split]
    a = frame[frame["system"] == system_a].set_index("query_id")[metric]
    b = frame[frame["system"] == system_b].set_index("query_id")[metric]
    paired = pd.concat([a, b], axis=1, keys=["a", "b"]).astype(float).dropna()
    diff = (paired["a"] - paired["b"]).to_numpy()
    n = diff.size

    result = {
        "metric": metric, "system_a": system_a, "system_b": system_b,
        "n_queries": n,
        "n_excluded": int(len(set(a.index) | set(b.index)) - n),
        "mean_a": float(paired["a"].mean()) if n else math.nan,
        "mean_b": float(paired["b"].mean()) if n else math.nan,
    }
    if n == 0:
        return {**result, "mean_diff": math.nan, "ci_low": math.nan,
                "ci_high": math.nan, "p_value": math.nan, "p_method": "none",
                "wins_a": 0, "ties": 0, "wins_b": 0}

    mean_diff, low, high = bootstrap_ci(diff, n_boot, seed, alpha)
    observed = abs(diff.mean())
    tolerance = 1e-12

    if n <= exact_max_n:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        extreme = np.abs((signs * diff).mean(axis=1)) >= observed - tolerance
        p_value, method = float(extreme.mean()), "exact_sign_flip"
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((1.0, -1.0), size=(n_boot, n))
        extreme = np.abs((signs * diff).mean(axis=1)) >= observed - tolerance
        p_value = float((extreme.sum() + 1) / (n_boot + 1))
        method = "monte_carlo_sign_flip"

    return {
        **result,
        "mean_diff": mean_diff, "ci_low": low, "ci_high": high,
        "p_value": p_value, "p_method": method,
        "wins_a": int((diff > 0).sum()),
        "ties": int((diff == 0).sum()),
        "wins_b": int((diff < 0).sum()),
    }


# =============================================================
# Latency
# =============================================================

def latency_summary(samples_ms: Sequence[float], warmup: int = 1) -> dict[str, float]:
    """Drop the first `warmup` samples, then report p50/p95/mean/max (ms)."""

    if warmup < 0:
        raise ValueError("warmup must be >= 0.")
    data = np.asarray(samples_ms, dtype=float)
    kept = data[warmup:]
    if kept.size == 0:
        raise ValueError("No latency samples left after discarding warm-up runs.")
    return {
        "n": int(kept.size),
        "discarded_warmup": int(min(warmup, data.size)),
        "p50_ms": float(np.percentile(kept, 50)),
        "p95_ms": float(np.percentile(kept, 95)),
        "mean_ms": float(kept.mean()),
        "max_ms": float(kept.max()),
    }


def measure_latency(
    function: Callable[[Any], Any],
    inputs: Sequence[Any],
    warmup_runs: int = 3,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """
    Time function(x) for every input after `warmup_runs` untimed calls on
    inputs[0], so no real query absorbs model loading / CUDA warm-up.

    Pass synchronize=torch.cuda.synchronize when timing GPU code, because
    CUDA kernels run asynchronously.
    """

    if not inputs:
        raise ValueError("inputs must be non-empty.")
    for _ in range(warmup_runs):
        function(inputs[0])
    if synchronize:
        synchronize()

    samples = []
    for item in inputs:
        start = time.perf_counter()
        function(item)
        if synchronize:
            synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    return {**latency_summary(samples, warmup=0), "samples_ms": samples}
