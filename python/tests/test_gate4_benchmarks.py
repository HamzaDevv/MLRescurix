"""Test Suite for Release Gate 4: Prospective Matched Benchmarks & Scientific Guards.

Verifies:
1. Multi-seed benchmark driver across seeds [2026, 42, 101] on standardized OpenML datasets.
2. Matched compute, thread, and memory budgets against standard baselines (default LightGBM, Random Forest, HistGradientBoosting).
3. Transparent recording of trial outcomes (winning, suboptimal, failing, and non-generalizing trials) in benchmark scorecards.
4. Strict zero test leakage verification, row-id disjointness, and single-use sealed holdout evaluation.
5. Detection and rejection of non-generalizing / overfitting candidate models.
6. Benchmark report JSON schema conformance against contracts/v2/benchmark_report.schema.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import unittest
import jsonschema

from mlrescurix_worker.scientific.benchmarks import (
    BenchmarkSuiteReport,
    BudgetExceededError,
    SealedHoldout,
    SealedTestLeakageError,
    StrategyBenchmarkResult,
    TrialRecord,
    create_sealed_benchmark_split,
    load_openml_benchmark_dataset,
    run_prospective_benchmark,
    run_single_evaluation,
    save_benchmark_report,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT_DIR / "contracts/v2/benchmark_report.schema.json"


class TestGate4ProspectiveBenchmarks(unittest.TestCase):
    """Rigorous verification suite for Gate 4 Matched Benchmarks."""

    @classmethod
    def setUpClass(cls):
        if SCHEMA_PATH.is_file():
            cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            cls.validator = jsonschema.Draft202012Validator(cls.schema)
        else:
            cls.schema = None
            cls.validator = None

    def test_openml_dataset_loaders_and_schemas(self):
        """Verify standardized OpenML dataset loading across classification and regression."""
        # 1. credit-g (OpenML 31)
        df_credit, target_credit, feats_credit = load_openml_benchmark_dataset("credit-g", seed=2026)
        self.assertGreaterEqual(len(df_credit), 1000)
        self.assertEqual(target_credit, "target")
        self.assertIn("__row_id__", df_credit.columns)
        self.assertGreater(len(feats_credit), 5)
        self.assertTrue(set(df_credit[target_credit].unique()).issubset({0, 1}))

        # 2. blood-transfusion (OpenML 1464)
        df_blood, target_blood, feats_blood = load_openml_benchmark_dataset("blood-transfusion", seed=42)
        self.assertGreaterEqual(len(df_blood), 748)
        self.assertEqual(target_blood, "target")
        self.assertEqual(feats_blood, ["Recency", "Frequency", "Monetary", "Time"])

        # 3. diabetes (OpenML 37)
        df_diab, target_diab, feats_diab = load_openml_benchmark_dataset("diabetes", seed=101)
        self.assertGreaterEqual(len(df_diab), 768)
        self.assertIn("glucose", feats_diab)

        # 4. california-housing (regression)
        df_house, target_house, feats_house = load_openml_benchmark_dataset("california-housing", seed=2026)
        self.assertGreaterEqual(len(df_house), 1000)
        self.assertEqual(target_house, "target")
        self.assertIn("MedInc", feats_house)

    def test_zero_leakage_sealed_holdout_partitioning(self):
        """Verify that sealed test split is strictly disjoint and protected against leakage."""
        df, target, features = load_openml_benchmark_dataset("blood-transfusion", seed=2026)
        dev_df, holdout = create_sealed_benchmark_split(
            df=df,
            target=target,
            test_size=0.2,
            seed=2026,
            task="binary_classification",
        )

        # Row ID disjointness check
        dev_ids = set(dev_df["__row_id__"])
        test_ids = set(holdout._test_df["__row_id__"])
        self.assertTrue(dev_ids.isdisjoint(test_ids), "Development and test row IDs must be strictly disjoint")
        self.assertEqual(len(dev_ids) + len(test_ids), len(df))

        # Sealed status
        self.assertTrue(holdout.is_sealed)
        self.assertEqual(holdout.access_count, 0)

        # Fit model on dev_df
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=200, random_state=2026)
        X_dev = dev_df[features].to_numpy()
        y_dev = dev_df[target].to_numpy()
        clf.fit(X_dev, y_dev)

        # Evaluate sealed holdout exactly once
        score = holdout.evaluate(clf, metric="balanced_accuracy")
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0.4)
        self.assertFalse(holdout.is_sealed)
        self.assertEqual(holdout.access_count, 1)

        # Second evaluation MUST be rejected with SealedTestLeakageError
        with self.assertRaises(SealedTestLeakageError):
            holdout.evaluate(clf, metric="balanced_accuracy")

    def test_matched_compute_and_memory_budget_enforcement(self):
        """Verify that memory and wall-clock budgets are measured and cap violations intercepted."""
        df, target, features = load_openml_benchmark_dataset("blood-transfusion", seed=2026)

        # Normal evaluation within bounds
        eval_ok = run_single_evaluation(
            df=df,
            target=target,
            features=features,
            recipe="logistic",
            params={"C": 1.0},
            task="binary_classification",
            metric="balanced_accuracy",
            seed=2026,
            max_memory_mb=512.0,
            wall_budget_ms=10_000.0,
        )
        self.assertEqual(eval_ok["status"], "success")
        self.assertIsNotNone(eval_ok["metric_score"])
        self.assertGreater(eval_ok["peak_memory_bytes"], 0)
        self.assertLess(eval_ok["peak_memory_bytes"], 512 * 1024 * 1024)

        # Memory limit violation: memory_heavy_candidate triggers budget failure
        eval_mem = run_single_evaluation(
            df=df,
            target=target,
            features=features,
            recipe="memory_heavy_candidate",
            params={},
            task="binary_classification",
            metric="balanced_accuracy",
            seed=2026,
            max_memory_mb=10.0,  # Tight 10 MB cap
        )
        self.assertEqual(eval_mem["status"], "failed_constraint")
        self.assertIsNone(eval_mem["metric_score"])
        self.assertIn("exceeded", eval_mem["error_message"].lower())

    def test_rejection_of_non_generalizing_overfit_candidates(self):
        """Verify that non-generalizing candidates with massive generalization gaps are rejected."""
        df, target, features = load_openml_benchmark_dataset("credit-g", seed=2026, n_samples=300)

        eval_overfit = run_single_evaluation(
            df=df,
            target=target,
            features=features,
            recipe="overfit_candidate",
            params={},
            task="binary_classification",
            metric="balanced_accuracy",
            seed=2026,
            overfit_threshold=0.25,
        )
        self.assertEqual(eval_overfit["status"], "rejected_overfit")
        self.assertGreater(eval_overfit["train_score"], 0.90)
        self.assertGreater(eval_overfit["generalization_gap"], 0.25)
        self.assertIn("generalization gap", eval_overfit["error_message"].lower())

    def test_multi_seed_benchmark_suite_execution(self):
        """Verify prospective benchmark driver across seeds [2026, 42, 101] against standard baselines."""
        report = run_prospective_benchmark(
            dataset_name="blood-transfusion",
            task="binary_classification",
            metric="balanced_accuracy",
            direction="maximize",
            seeds=[2026, 42, 101],
            max_trials_per_strategy=6,
            wall_budget_ms=15_000.0,
            max_memory_mb=512.0,
            max_threads=1,
            baselines=["default_lightgbm", "random_forest", "hist_gradient_boosting", "random", "search"],
            include_failing_trial=True,
            offline_only=True,
        )

        self.assertEqual(report.dataset_name, "blood-transfusion")
        self.assertEqual(report.seeds, [2026, 42, 101])

        # Verify baselines and agent are evaluated
        expected_strats = ["default_lightgbm", "random_forest", "hist_gradient_boosting", "random", "search", "agent"]
        for s in expected_strats:
            self.assertIn(s, report.results, f"Missing strategy: {s}")
            res = report.results[s]
            # Matched trial counts
            self.assertEqual(res.trials_attempted, 6, f"{s} must attempt matched 6 trials")
            self.assertGreater(res.total_wall_ms, 0.0)

        # Transparent failure recording
        agent_res = report.results["agent"]
        has_failed = any(t.status in ("failed_constraint", "timeout", "rejected_overfit") for t in agent_res.trials)
        self.assertTrue(has_failed, "Agent strategy must transparently record failing/suboptimal trials")

        # Sealed holdout prospective evaluation was executed
        self.assertTrue(report.comparative_summary["sealed_holdout_evaluated"])
        self.assertTrue(report.comparative_summary["zero_test_leakage_verified"])
        self.assertTrue(report.comparative_summary["matched_budgets_verified"])
        self.assertTrue(report.comparative_summary["suboptimal_and_failing_trials_recorded"])

        # Check JSON schema compliance
        if self.validator is not None:
            report_dict = json.loads(json.dumps(report, default=lambda o: getattr(o, "__dict__", str(o))))
            self.validator.validate(report_dict)

    def test_save_and_reload_benchmark_report(self):
        """Verify report serialization and file integrity."""
        import tempfile
        report = run_prospective_benchmark(
            dataset_name="synthetic",
            task="binary_classification",
            metric="balanced_accuracy",
            seeds=[2026, 42],
            max_trials_per_strategy=4,
        )
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "gate4_report.json"
            save_benchmark_report(report, out_file)
            self.assertTrue(out_file.is_file())
            loaded = json.loads(out_file.read_text(encoding="utf-8"))
            self.assertEqual(loaded["suite_id"], report.suite_id)
            self.assertEqual(loaded["schema_version"], "v2")
            if self.validator is not None:
                self.validator.validate(loaded)


if __name__ == "__main__":
    unittest.main()
