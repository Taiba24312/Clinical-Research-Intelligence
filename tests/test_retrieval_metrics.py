"""
Unit tests for src/evaluation (judgments.py, retrieval_metrics.py).

Run from the project root:
    python -m unittest discover -s tests -v
"""

import math
import unittest
from math import log2
from pathlib import Path

import pandas as pd

from src.evaluation.judgments import (
    derive_grade,
    parse_query_specs,
    prepare_qrels,
    prepare_run,
)
from src.evaluation.retrieval_metrics import (
    bootstrap_ci,
    constraint_violations,
    evaluate_run,
    latency_summary,
    macro_summary,
    measure_latency,
    paired_comparison,
    parse_age_years,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# =============================================================
# Fixtures
# =============================================================

def spec(query_id="QA", has_answer=True, intervention=True, constraints=None, split="dev"):
    return {
        "query_id": query_id,
        "text": f"text of {query_id}",
        "core_concepts": {
            "disease": {"concept": "disease"},
            "intervention": {"concept": "drug"} if intervention else None,
        },
        "structured_constraints": (
            {"status": ["RECRUITING"]} if constraints is None else constraints
        ),
        "category": ["test"],
        "split": split,
        "has_answer": has_answer,
    }


def metadata(**overrides):
    """Trials d1..d9 and dX/dY: RECRUITING adults unless overridden."""
    rows = []
    for nct_id in ["d1", "d2", "d3", "d4", "d5", "d6", "d7", "dX", "dY"]:
        row = {
            "nct_id": nct_id, "status": "RECRUITING", "phase": "PHASE2",
            "sex": "ALL", "study_type": "INTERVENTIONAL",
            "min_age": "18 Years", "max_age": None,
        }
        row.update(overrides.get(nct_id, {}))
        rows.append(row)
    return pd.DataFrame(rows)


GRADE_TO_LABELS = {
    2: ("exact", "exact"),
    1: ("exact", "class_level"),
    0: ("absent_or_different", "exact"),
}


def qrels(rows):
    """rows: (query_id, nct_id, grade) -> qrels frame with derived grades."""
    return pd.DataFrame([
        {
            "query_id": q, "nct_id": n,
            "disease_match": GRADE_TO_LABELS[g][0],
            "intervention_match": GRADE_TO_LABELS[g][1],
            "mention_role": "", "notes": "", "labeler": "t", "label_date": "2026-10-30",
        }
        for q, n, g in rows
    ])


def run(query_id, system, ids, scores=None):
    scores = scores or [float(len(ids) - i) for i in range(len(ids))]
    return pd.DataFrame({
        "query_id": query_id, "system": system, "nct_id": ids,
        "rank": range(1, len(ids) + 1), "score": scores,
    })


def evaluate(run_frame, qrel_rows, specs_list, meta=None, ks=(2, 5, 10)):
    specs = parse_query_specs(specs_list)
    return evaluate_run(
        prepare_run(run_frame),
        prepare_qrels(qrels(qrel_rows), specs),
        specs,
        metadata() if meta is None else meta,
        ks=ks,
    )


# Example A (hand-computed throughout):
#   judged grades: d1=2, d2=1, d3=0, d4=2, d5=1  (d5 never retrieved)
#   run:           d3, d1, dX(unjudged), d2, d4
#   gains (2^g-1): 0,  3,  0,            1,  3
EXAMPLE_A_QRELS = [("QA", "d1", 2), ("QA", "d2", 1), ("QA", "d3", 0),
                   ("QA", "d4", 2), ("QA", "d5", 1)]
EXAMPLE_A_RUN = ["d3", "d1", "dX", "d2", "d4"]
IDCG_5 = 3 / log2(2) + 3 / log2(3) + 1 / log2(4) + 1 / log2(5)   # ideal 2,2,1,1,0
IDCG_2 = 3 / log2(2) + 3 / log2(3)


# =============================================================
# Grade derivation and input validation
# =============================================================

class TestDeriveGrade(unittest.TestCase):

    def test_grade_table(self):
        cases = [
            ("exact", "exact", "", 2),
            ("exact", "exact", "focus", 2),
            ("exact", "not_applicable", "", 2),
            ("not_applicable", "exact", "", 2),
            ("broader", "exact", "", 1),
            ("exact", "class_level", "", 1),
            ("broader", "class_level", "", 1),
            ("broader", "not_applicable", "", 1),
            ("absent_or_different", "exact", "", 0),
            ("exact", "absent_or_different", "prior_therapy", 0),
            ("broader", "absent_or_different", "concurrent_therapy", 0),
        ]
        for disease, intervention, role, expected in cases:
            with self.subTest(disease=disease, intervention=intervention, role=role):
                self.assertEqual(derive_grade(disease, intervention, role), expected)

    def test_non_focus_role_without_absent_concept_is_rejected(self):
        with self.assertRaises(ValueError):
            derive_grade("exact", "exact", "prior_therapy")

    def test_invalid_values_rejected(self):
        for args in [("partial", "exact"), ("exact", "broader"),
                     ("not_applicable", "not_applicable"), ("exact", "exact", "maybe")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                derive_grade(*args)


class TestValidation(unittest.TestCase):

    def test_pending_rows_dropped_and_counted(self):
        specs = parse_query_specs([spec()])
        frame = qrels([("QA", "d1", 2)])
        pending = frame.iloc[0].copy()
        pending["nct_id"], pending["disease_match"], pending["intervention_match"] = "d2", "", ""
        frame = pd.concat([frame, pending.to_frame().T], ignore_index=True)
        prepared = prepare_qrels(frame, specs)
        self.assertEqual(prepared["nct_id"].tolist(), ["d1"])
        self.assertEqual(prepared.attrs["pending_rows"], 1)

    def test_not_applicable_must_match_spec(self):
        no_drug = parse_query_specs([spec(intervention=False)])
        with self.assertRaises(ValueError):   # spec has no drug, label says exact
            prepare_qrels(qrels([("QA", "d1", 2)]), no_drug)
        with_drug = parse_query_specs([spec()])
        frame = qrels([("QA", "d1", 2)])
        frame.loc[0, "intervention_match"] = "not_applicable"
        with self.assertRaises(ValueError):   # spec has a drug, label says n/a
            prepare_qrels(frame, with_drug)

    def test_duplicate_judgments_rejected(self):
        specs = parse_query_specs([spec()])
        with self.assertRaises(ValueError):
            prepare_qrels(qrels([("QA", "d1", 2), ("QA", "d1", 0)]), specs)

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            parse_query_specs([spec(constraints={"stauts": ["RECRUITING"]})])
        bad = spec()
        bad["core_concepts"] = {"disease": None, "intervention": None}
        with self.assertRaises(ValueError):
            parse_query_specs([bad])

    def test_run_ties_and_duplicates(self):
        # Tied scores with distinct ranks: order follows rank, rows may be unsorted.
        frame = pd.DataFrame({
            "query_id": "QA", "system": "s", "nct_id": ["d2", "d1", "d3"],
            "rank": [2, 1, 3], "score": [0.5, 0.5, 0.5],
        })
        self.assertEqual(prepare_run(frame)["nct_id"].tolist(), ["d1", "d2", "d3"])
        with self.assertRaises(ValueError):
            prepare_run(frame.assign(rank=[1, 1, 2]))
        with self.assertRaises(ValueError):
            prepare_run(frame.assign(nct_id=["d1", "d1", "d3"]))


# =============================================================
# Structured constraints
# =============================================================

class TestConstraints(unittest.TestCase):

    def check(self, trial, constraints):
        base = {"status": "RECRUITING", "phase": "PHASE2", "sex": "ALL",
                "study_type": "INTERVENTIONAL", "min_age": None, "max_age": None}
        base.update(trial)
        return [v["constraint"] for v in constraint_violations(base, constraints)]

    def test_age_parsing(self):
        self.assertEqual(parse_age_years("18 Years"), 18.0)
        self.assertEqual(parse_age_years("1 Year"), 1.0)
        self.assertEqual(parse_age_years("6 Months"), 0.5)
        self.assertAlmostEqual(parse_age_years("6 Hours"), 6 / 8766)
        self.assertIsNone(parse_age_years(None))
        self.assertIsNone(parse_age_years(float("nan")))
        with self.assertRaises(ValueError):
            parse_age_years("eighteen")

    def test_status_sex_study_type(self):
        self.assertEqual(self.check({}, {"status": ["RECRUITING"]}), [])
        self.assertEqual(self.check({"status": "NOT_YET_RECRUITING"},
                                    {"status": ["RECRUITING"]}), ["status"])
        self.assertEqual(self.check({"sex": "ALL"}, {"sex": ["FEMALE", "ALL"]}), [])
        self.assertEqual(self.check({"sex": "MALE"}, {"sex": ["FEMALE", "ALL"]}), ["sex"])
        self.assertEqual(self.check({"sex": float("nan")}, {"sex": ["ALL"]}), ["sex"])
        self.assertEqual(self.check({}, {"study_type": ["OBSERVATIONAL"]}), ["study_type"])

    def test_phase(self):
        self.assertEqual(self.check({"phase": "PHASE1; PHASE2"}, {"phase": ["PHASE2"]}), [])
        self.assertEqual(self.check({"phase": "EARLY_PHASE1"}, {"phase": ["PHASE1"]}), ["phase"])
        self.assertEqual(self.check({"phase": "NA"}, {"phase": ["PHASE3"]}), ["phase"])
        self.assertEqual(self.check({"phase": ""}, {"phase": ["PHASE3"]}), ["phase"])

    def test_age_range(self):
        adults, pediatric, older = [18, None], [0, 18], [65, None]
        self.assertEqual(self.check({"max_age": "17 Years"}, {"age_range": adults}), ["age_range"])
        self.assertEqual(self.check({"max_age": None}, {"age_range": adults}), [])
        self.assertEqual(self.check({"max_age": "6 Hours"}, {"age_range": adults}), ["age_range"])
        self.assertEqual(self.check({"min_age": "18 Years"}, {"age_range": pediatric}), ["age_range"])
        self.assertEqual(self.check({"min_age": "17 Years"}, {"age_range": pediatric}), [])
        self.assertEqual(self.check({"max_age": "64 Years"}, {"age_range": older}), ["age_range"])
        self.assertEqual(self.check({"min_age": "18 Years", "max_age": "70 Years"},
                                    {"age_range": older}), [])

    def test_unknown_constraint(self):
        with self.assertRaises(ValueError):
            self.check({}, {"country": ["US"]})


# =============================================================
# Ranking metrics
# =============================================================

class TestRankingMetrics(unittest.TestCase):

    def setUp(self):
        result = evaluate(run("QA", "s", EXAMPLE_A_RUN), EXAMPLE_A_QRELS, [spec()])
        self.row = result.per_query.iloc[0]

    def test_ndcg_uses_all_judged_documents_for_ideal(self):
        dcg_5 = 3 / log2(3) + 1 / log2(5) + 3 / log2(6)
        self.assertAlmostEqual(self.row["ndcg@5"], dcg_5 / IDCG_5)        # 0.598274
        self.assertAlmostEqual(self.row["ndcg@2"], (3 / log2(3)) / IDCG_2)  # 0.386853

    def test_precision_recall_mrr_strict_and_lenient(self):
        r = self.row
        self.assertEqual((r["n_rel_strict"], r["n_rel_lenient"]), (2, 4))
        self.assertAlmostEqual(r["p_strict@5"], 2 / 5)
        self.assertAlmostEqual(r["p_lenient@5"], 3 / 5)
        self.assertAlmostEqual(r["r_strict@5"], 2 / 2)
        self.assertAlmostEqual(r["r_lenient@5"], 3 / 4)
        self.assertAlmostEqual(r["p_strict@2"], 1 / 2)
        self.assertAlmostEqual(r["r_lenient@2"], 1 / 4)
        self.assertAlmostEqual(r["mrr_strict"], 1 / 2)
        self.assertAlmostEqual(r["mrr_lenient"], 1 / 2)

    def test_k_larger_than_list(self):
        self.assertAlmostEqual(self.row["p_strict@10"], 2 / 10)     # denominator K
        self.assertAlmostEqual(self.row["judged@10"], 4 / 5)        # share of returned
        # Only 5 trials returned and 5 judged: DCG@10 = DCG@5 and IDCG@10 = IDCG@5.
        self.assertAlmostEqual(self.row["ndcg@10"], self.row["ndcg@5"])

    def test_judged_and_condensed_list(self):
        self.assertAlmostEqual(self.row["judged@5"], 4 / 5)
        # Condensed list removes dX: d3, d1, d2, d4 -> gains 0, 3, 1, 3.
        cond_dcg_5 = 3 / log2(3) + 1 / log2(4) + 3 / log2(5)
        self.assertAlmostEqual(self.row["cond_ndcg@5"], cond_dcg_5 / IDCG_5)  # 0.632756
        self.assertAlmostEqual(self.row["cond_p_lenient@5"], 3 / 5)
        self.assertAlmostEqual(self.row["cond_mrr_strict"], 1 / 2)

    def test_all_unjudged(self):
        result = evaluate(run("QA", "s", ["dX", "dY"]), EXAMPLE_A_QRELS, [spec()])
        r = result.per_query.iloc[0]
        self.assertEqual(r["judged@5"], 0.0)
        self.assertEqual(r["ndcg@5"], 0.0)
        self.assertEqual(r["p_lenient@5"], 0.0)
        self.assertEqual(r["mrr_strict"], 0.0)
        self.assertEqual(r["cond_ndcg@5"], 0.0)   # empty condensed list

    def test_empty_result_list_is_abstention_not_skip(self):
        frame = pd.concat([run("QA", "s", EXAMPLE_A_RUN)])
        result = evaluate(frame, EXAMPLE_A_QRELS + [("QB", "d1", 2)],
                          [spec(), spec("QB")])
        r = result.per_query.set_index("query_id").loc["QB"]
        self.assertEqual(r["n_returned"], 0)
        self.assertEqual(r["ndcg@5"], 0.0)
        self.assertEqual(r["r_strict@5"], 0.0)
        self.assertTrue(math.isnan(r["judged@5"]))
        self.assertTrue(math.isnan(r["violation_rate@5"]))

    def test_no_strict_relevant_makes_strict_metrics_undefined(self):
        result = evaluate(run("QA", "s", ["d2", "d1"]), [("QA", "d2", 1), ("QA", "d1", 0)], [spec()])
        r = result.per_query.iloc[0]
        for column in ["p_strict@5", "r_strict@5", "mrr_strict"]:
            self.assertTrue(math.isnan(r[column]), column)
        self.assertAlmostEqual(r["p_lenient@5"], 1 / 5)
        self.assertAlmostEqual(r["mrr_lenient"], 1.0)


class TestConstraintAwareGrades(unittest.TestCase):

    def test_violators_get_grade_zero_and_are_reported(self):
        meta = metadata(d6={"status": "NOT_YET_RECRUITING"}, d7={"status": "COMPLETED"})
        # d6 is judged exact/exact (grade 2) but violates status -> effective 0.
        # d7 is unjudged but violates status -> effective 0, counts as judged.
        result = evaluate(
            run("QA", "s", ["d6", "d1", "d7"]),
            [("QA", "d1", 2), ("QA", "d6", 2)], [spec()], meta,
        )
        r = result.per_query.iloc[0]
        self.assertEqual(r["n_rel_strict"], 1)              # d6 not in the ideal set
        self.assertAlmostEqual(r["p_strict@2"], 1 / 2)
        self.assertAlmostEqual(r["mrr_strict"], 1 / 2)
        self.assertAlmostEqual(r["judged@5"], 1.0)
        self.assertAlmostEqual(r["violation_rate@5"], 2 / 3)
        self.assertEqual(r["violated@5"], "status=2")
        self.assertEqual(result.violations["nct_id"].tolist(), ["d6", "d7"])
        self.assertEqual(result.violations["actual"].tolist(),
                         ["NOT_YET_RECRUITING", "COMPLETED"])


class TestZeroAnswerQueries(unittest.TestCase):

    def test_reported_separately_and_excluded_from_averages(self):
        frame = pd.concat([
            run("QA", "s", EXAMPLE_A_RUN),
            run("QZ", "s", ["d3", "dX"], scores=[0.9, 0.4]),
        ])
        result = evaluate(frame, EXAMPLE_A_QRELS + [("QZ", "d3", 0)],
                          [spec(), spec("QZ", has_answer=False)])
        per_query = result.per_query.set_index("query_id")
        for column in ["ndcg@5", "p_strict@5", "r_lenient@5", "mrr_strict"]:
            self.assertTrue(math.isnan(per_query.loc["QZ", column]), column)

        zero = result.zero_answer.iloc[0]
        self.assertEqual(zero["query_id"], "QZ")
        self.assertAlmostEqual(zero["best_score"], 0.9)
        self.assertEqual(zero["top1_nct_id"], "d3")
        self.assertAlmostEqual(zero["grade0_share@5"], 1 / 2)
        self.assertAlmostEqual(zero["unjudged_share@5"], 1 / 2)

        summary = macro_summary(result.per_query, metrics=["ndcg@5"], n_boot=200)
        self.assertEqual(summary.iloc[0]["n_queries"], 1)
        self.assertAlmostEqual(summary.iloc[0]["mean"], per_query.loc["QA", "ndcg@5"])

    def test_inconsistent_has_answer_flag_raises(self):
        with self.assertRaises(ValueError):
            evaluate(run("QZ", "s", ["d1"]), [("QZ", "d1", 2)],
                     [spec("QZ", has_answer=False)])


# =============================================================
# Aggregation, bootstrap, paired comparison, latency
# =============================================================

class TestStatistics(unittest.TestCase):

    def test_bootstrap_ci(self):
        self.assertEqual(bootstrap_ci([0.5, 0.5, 0.5]), (0.5, 0.5, 0.5))
        mean, low, high = bootstrap_ci([0.0, 1.0, float("nan"), 1.0, 0.0], n_boot=2000)
        self.assertAlmostEqual(mean, 0.5)
        self.assertTrue(0.0 <= low <= mean <= high <= 1.0)
        self.assertEqual(bootstrap_ci([0.1, 0.9, 0.4], n_boot=500, seed=7),
                         bootstrap_ci([0.1, 0.9, 0.4], n_boot=500, seed=7))
        mean, low, high = bootstrap_ci([0.3])
        self.assertEqual(mean, 0.3)
        self.assertTrue(math.isnan(low) and math.isnan(high))

    def per_query(self, a_values, b_values):
        rows = []
        for system, values in (("A", a_values), ("B", b_values)):
            for i, value in enumerate(values):
                rows.append({"system": system, "query_id": f"Q{i}", "split": "dev", "m": value})
        return pd.DataFrame(rows)

    def test_paired_identical_systems(self):
        result = paired_comparison(self.per_query([0.2, 0.5, 0.9], [0.2, 0.5, 0.9]), "A", "B", "m")
        self.assertEqual(result["mean_diff"], 0.0)
        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["ties"], 3)

    def test_paired_exact_sign_flip(self):
        # diffs = [1, 1, 1, 1]: only the all-plus and all-minus sign vectors
        # reach |mean| >= 1, so p = 2 / 2^4 = 0.125.
        result = paired_comparison(self.per_query([1, 1, 1, 1], [0, 0, 0, 0]), "A", "B", "m")
        self.assertAlmostEqual(result["p_value"], 0.125)
        self.assertEqual(result["p_method"], "exact_sign_flip")
        self.assertEqual((result["wins_a"], result["wins_b"]), (4, 0))

    def test_paired_monte_carlo_and_nan_exclusion(self):
        a = [1.0] * 20 + [float("nan")]
        b = [0.0] * 20 + [0.5]
        result = paired_comparison(self.per_query(a, b), "A", "B", "m", n_boot=2000)
        self.assertEqual(result["n_queries"], 20)
        self.assertEqual(result["n_excluded"], 1)
        self.assertEqual(result["p_method"], "monte_carlo_sign_flip")
        self.assertLess(result["p_value"], 0.01)

    def test_latency_summary(self):
        summary = latency_summary([100, 10, 12, 11, 13], warmup=1)
        self.assertEqual(summary["n"], 4)
        self.assertAlmostEqual(summary["p50_ms"], 11.5)
        self.assertAlmostEqual(summary["p95_ms"], 12 + 0.85 * (13 - 12))  # linear interp.
        with self.assertRaises(ValueError):
            latency_summary([5.0], warmup=1)

    def test_measure_latency_discards_warmup_calls(self):
        calls = []
        result = measure_latency(calls.append, ["q1", "q2", "q3"], warmup_runs=2)
        self.assertEqual(calls, ["q1", "q1", "q1", "q2", "q3"])
        self.assertEqual(result["n"], 3)


# =============================================================
# Reproduction of the old Q01 numbers (Exp 16/17/18)
# =============================================================

FUSION = PROJECT_ROOT / "data/evaluation/weighted_fusion_candidates.csv"
RERANKED = PROJECT_ROOT / "data/evaluation/reranked_candidates.csv"
TRIALS = PROJECT_ROOT / "data/processed/trials.parquet"


@unittest.skipUnless(FUSION.exists() and RERANKED.exists() and TRIALS.exists(),
                     "experiment outputs not available")
class TestReproduceOldQ01Metrics(unittest.TestCase):
    """
    Old values (reranking_metrics.csv), computed with the two known
    positives and unjudged = non-relevant:
        RRF: R@5 1.0, R@10 1.0, P@5 0.4, P@10 0.2, MRR 1.0
        CE : R@5 1.0, R@10 1.0, P@5 0.4, P@10 0.2, MRR 0.5
    """

    def test_strict_threshold_reproduces_exp17(self):
        fusion = pd.read_csv(FUSION)
        fusion = fusion[(fusion["query_id"] == "Q01")
                        & (fusion["config"] == "rrf_baseline_50_50")]
        reranked = pd.read_csv(RERANKED)
        reranked = reranked[reranked["query_id"] == "Q01"]

        run_frame = pd.concat([
            pd.DataFrame({"query_id": "Q01", "system": "rrf",
                          "nct_id": fusion["nct_id"], "rank": fusion["rank"],
                          "score": fusion["fusion_score"]}),
            pd.DataFrame({"query_id": "Q01", "system": "cross_encoder",
                          "nct_id": reranked["nct_id"], "rank": reranked["reranked_rank"],
                          "score": reranked["cross_encoder_score"]}),
        ])
        q01 = spec("Q01")
        q01["core_concepts"] = {"disease": {"concept": "non-small cell lung cancer"},
                                "intervention": {"concept": "pembrolizumab"}}
        result = evaluate(
            run_frame,
            [("Q01", "NCT06195527", 2), ("Q01", "NCT07061379", 2)],
            [q01], pd.read_parquet(TRIALS), ks=(5, 10),
        )
        rows = result.per_query.set_index("system")

        expected = {
            "rrf": {"r_strict@5": 1.0, "r_strict@10": 1.0, "p_strict@5": 0.4,
                    "p_strict@10": 0.2, "mrr_strict": 1.0},
            "cross_encoder": {"r_strict@5": 1.0, "r_strict@10": 1.0, "p_strict@5": 0.4,
                              "p_strict@10": 0.2, "mrr_strict": 0.5},
        }
        for system, metrics in expected.items():
            for metric, value in metrics.items():
                with self.subTest(system=system, metric=metric):
                    self.assertAlmostEqual(rows.loc[system, metric], value)

        # Context the old numbers did not show: only 2 of the top-5 are judged,
        # and all Exp 16 candidates already satisfy the RECRUITING constraint.
        self.assertAlmostEqual(rows.loc["rrf", "judged@5"], 2 / 5)
        self.assertAlmostEqual(rows.loc["rrf", "violation_rate@10"], 0.0)


if __name__ == "__main__":
    unittest.main()
