"""Unit tests for MLRescurix M5 scientific capabilities.

Covers:
- Split-aware feature engineering pipeline and out-of-fold target encoding
- Split-aware probability calibration (isotonic and sigmoid)
- Constrained ensemble recipes (voting and bounded stacking with explicit lineage)
- CPU-only HistGradientBoosting hyperparameter integration and pipeline execution
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict, StratifiedKFold

from mlrescurix_worker.scientific.features import (
    FeatureTransformSpec,
    OutOfFoldTargetEncoder,
    SplitAwareFeaturePipeline,
)
from mlrescurix_worker.scientific.calibration import SplitAwareCalibrator
from mlrescurix_worker.scientific.ensemble import (
    EnsembleSpec,
    VotingEnsemble,
    BoundedStackingEnsemble,
)
from mlrescurix_worker.search.adapter import (
    PARAM_RANGES,
    normalize_parameters,
    OptunaOptimizer,
    DeterministicFallbackOptimizer,
)
from mlrescurix_worker.scientific.workflow import _pipeline


class M5ScientificTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(2026)
        n = 120
        self.df = pd.DataFrame({
            "num1": np.random.randn(n),
            "num2": np.random.uniform(10, 50, n),
            "cat1": np.random.choice(["cat", "dog", "bird"], n),
            "cat2": np.random.choice(["X", "Y"], n),
        })
        self.y_binary = (self.df["num1"] + (self.df["cat1"] == "cat").astype(float) > 0.5).astype(int)
        self.y_multi = np.random.choice([0, 1, 2], n)
        self.y_reg = self.df["num1"] * 2.0 + self.df["num2"] * 0.5 + np.random.randn(n) * 0.1

    def test_oof_target_encoder_no_leakage(self):
        encoder = OutOfFoldTargetEncoder(categorical_columns=["cat1"], n_splits=3, smoothing=5.0)
        transformed = encoder.fit_transform(self.df, self.y_binary)
        self.assertEqual(len(transformed), len(self.df))
        self.assertIn("cat1_te", transformed.columns)
        self.assertTrue(np.issubdtype(transformed["cat1_te"].dtype, np.floating))

        # Test transform on unseen data uses global mapping without recomputing folds
        test_df = pd.DataFrame({"cat1": ["cat", "unknown_animal"]})
        test_out = encoder.transform(test_df)
        self.assertEqual(len(test_out), 2)
        # Unknown category falls back to global mean
        self.assertAlmostEqual(test_out["cat1_te"].iloc[1], encoder.global_mean_, places=5)

    def test_split_aware_feature_pipeline(self):
        spec = FeatureTransformSpec(
            transform_id="tf-all",
            numeric_scaler="robust",
            interactions=True,
            discretize_bins=4,
            missing_indicators=True,
            target_encoding=True,
            max_interaction_features=5,
        )
        pipeline = SplitAwareFeaturePipeline(spec)
        X_train_trans = pipeline.fit_transform(
            self.df, self.y_binary, categorical_cols=["cat1", "cat2"], numeric_cols=["num1", "num2"]
        )
        self.assertEqual(len(X_train_trans), len(self.df))
        self.assertGreater(X_train_trans.shape[1], self.df.shape[1])

        # Test transform on separate validation set
        val_df = self.df.iloc[:20].copy()
        X_val_trans = pipeline.transform(val_df)
        self.assertEqual(X_val_trans.shape[1], X_train_trans.shape[1])
        self.assertEqual(len(X_val_trans), 20)

    def test_split_aware_calibrator_binary(self):
        # Generate simulated uncalibrated probabilities with mild miscalibration
        raw_probs = np.column_stack([1.0 - (self.y_binary * 0.7 + 0.15), self.y_binary * 0.7 + 0.15])
        calibrator = SplitAwareCalibrator(method="isotonic")
        calibrator.fit_from_oof(raw_probs, self.y_binary, classes=[0, 1])

        calibrated = calibrator.transform(raw_probs)
        self.assertEqual(calibrated.shape, raw_probs.shape)
        # Check rows sum to 1.0 within numerical precision
        row_sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        # Check values bounded in [0, 1]
        self.assertTrue(np.all((calibrated >= 0.0) & (calibrated <= 1.0)))

    def test_split_aware_calibrator_sigmoid_multiclass(self):
        classes = [0, 1, 2]
        raw_probs = np.random.dirichlet([1, 1, 1], size=len(self.df))
        calibrator = SplitAwareCalibrator(method="sigmoid")
        calibrator.fit_from_oof(raw_probs, self.y_multi, classes=classes)

        calibrated = calibrator.transform(raw_probs)
        self.assertEqual(calibrated.shape, (len(self.df), 3))
        row_sums = calibrated.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_voting_ensemble_execution_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            m1_path = tmp_path / "m1.joblib"
            m2_path = tmp_path / "m2.joblib"

            # Train two base models
            X = self.df[["num1", "num2"]]
            clf1 = LogisticRegression().fit(X, self.y_binary)
            clf2 = LogisticRegression(C=0.01).fit(X, self.y_binary)
            joblib.dump(clf1, m1_path)
            joblib.dump(clf2, m2_path)

            voting = VotingEnsemble(
                parent_ids=["cand-1", "cand-2"],
                weights=[0.6, 0.4],
                task="binary_classification",
            )
            voting.load_parent_models([m1_path, m2_path])
            result = voting.predict(X)

            self.assertIn("prediction", result)
            self.assertIn("probabilities", result)
            self.assertIn("inference_latency_ms", result)
            self.assertEqual(len(result["prediction"]), len(X))
            self.assertGreaterEqual(result["inference_latency_ms"], 0.0)
            self.assertEqual(voting.parent_ids, ["cand-1", "cand-2"])

    def test_bounded_stacking_ensemble(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            m1_path = tmp_path / "m1.joblib"
            m2_path = tmp_path / "m2.joblib"

            X = self.df[["num1", "num2"]]
            clf1 = LogisticRegression().fit(X, self.y_binary)
            clf2 = LogisticRegression(C=0.1).fit(X, self.y_binary)
            joblib.dump(clf1, m1_path)
            joblib.dump(clf2, m2_path)

            # Real OOF predictions; each row is predicted by a model excluding it.
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=2026)
            oof1 = cross_val_predict(LogisticRegression(), X, self.y_binary, cv=cv, method="predict_proba")[:, 1:2]
            oof2 = cross_val_predict(LogisticRegression(C=0.1), X, self.y_binary, cv=cv, method="predict_proba")[:, 1:2]
            oof_meta = np.column_stack([oof1, oof2])

            stacking = BoundedStackingEnsemble(
                parent_ids=["cand-1", "cand-2"],
                task="binary_classification",
                meta_learner="linear",
            )
            stacking.fit_meta_learner([m1_path, m2_path], oof_meta, self.y_binary)
            result = stacking.predict(X)

            self.assertIn("prediction", result)
            self.assertIn("probabilities", result)
            self.assertIn("inference_latency_ms", result)
            self.assertEqual(len(result["prediction"]), len(X))
            self.assertGreaterEqual(result["inference_latency_ms"], 0.0)

    def test_cpu_hist_boosting_hyperparameter_ranges_and_pipeline(self):
        # 1. Parameter ranges defined
        self.assertIn("hist_boosting", PARAM_RANGES)
        ranges = PARAM_RANGES["hist_boosting"]
        self.assertIn("learning_rate", ranges)
        self.assertIn("max_iter", ranges)
        self.assertIn("max_leaf_nodes", ranges)

        # 2. Parameter normalization
        raw_params = {
            "learning_rate": 0.08,
            "max_iter": 75,
            "max_leaf_nodes": 25,
            "min_samples_leaf": 15,
            "l2_regularization": 0.5,
        }
        norm = normalize_parameters("hist_boosting", "binary_classification", raw_params)
        self.assertEqual(norm["max_iter"], 75)
        self.assertEqual(norm["max_leaf_nodes"], 25)

        # 3. Pipeline creation and fitting on CPU
        clf_pipe = _pipeline(self.df, ["num1", "num2"], "binary_classification", "hist_boosting", 2026, norm)
        X = self.df[["num1", "num2"]]
        clf_pipe.fit(X, self.y_binary)
        preds = clf_pipe.predict(X)
        self.assertEqual(len(preds), len(X))

        # Regressor
        reg_pipe = _pipeline(self.df, ["num1", "num2"], "regression", "hist_boosting", 2026, norm)
        reg_pipe.fit(X, self.y_reg)
        reg_preds = reg_pipe.predict(X)
        self.assertEqual(len(reg_preds), len(X))


if __name__ == "__main__":
    unittest.main()
