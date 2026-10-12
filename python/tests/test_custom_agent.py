import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlrescurix_worker.custom.contract import CustomEstimator, CustomTransformer
from mlrescurix_worker.custom.preflight import run_preflight_checks
from mlrescurix_worker.scientific.workflow import audit_csv, evaluate_custom_candidate, WorkflowError


class SimpleLinearEstimator(CustomEstimator):
    def __init__(self, C: float = 1.0):
        self.C = C
        self._clf = LogisticRegression(C=self.C, max_iter=200)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SimpleLinearEstimator":
        self._clf.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(X)


class CustomAgentWorkerTests(unittest.TestCase):
    def test_preflight_checks_pass_for_compliant_estimator(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "f1": np.random.randn(50),
            "f2": np.random.randn(50),
        })
        y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))

        est = SimpleLinearEstimator(C=0.5)
        res = run_preflight_checks(est, X, y, is_classification=True)
        self.assertTrue(res["passed"], f"Preflight failed: {res.get('error')}")
        self.assertTrue(res["checks"]["fit_succeeded"])
        self.assertTrue(res["checks"]["predict_shape_match"])
        self.assertTrue(res["checks"]["predict_proba_valid"])
        self.assertTrue(res["checks"]["single_row_inference"])
        self.assertTrue(res["checks"]["joblib_roundtrip_valid"])

    def test_isolated_custom_runner_and_evaluator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # Create full dataset and run audit
            n = 100
            np.random.seed(2026)
            df = pd.DataFrame({
                "feat1": np.random.randn(n),
                "feat2": np.random.randn(n),
                "group_slice": np.random.choice(["A", "B"], size=n),
                "target": (np.random.randn(n) > 0).astype(int),
            })
            csv_path = tmp / "raw.csv"
            df.to_csv(csv_path, index=False)

            wf_dir = tmp / "workflow"
            manifest = audit_csv(
                csv_path=csv_path,
                spec={
                    "project_id": "p-custom",
                    "target": "target",
                    "task": "binary_classification",
                    "metric": "balanced_accuracy",
                },
                output_dir=wf_dir,
            )

            # Prepare training CSV (with labels) and validation CSV (features only)
            train_idx = manifest["splits"]["validation"]  # for testing, use train split
            train_idx = manifest["splits"]["train"]
            val_idx = manifest["splits"]["validation"]

            train_df = df.iloc[train_idx][["feat1", "feat2", "target"]]
            val_df = df.iloc[val_idx][["feat1", "feat2"]].copy()
            val_df["row_id"] = val_idx

            train_csv = tmp / "train.csv"
            val_csv = tmp / "val.csv"
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)

            preds_json = tmp / "predictions.json"
            model_joblib = tmp / "custom_model.joblib"

            # Simulate containerized sandbox runner outputs (custom code executes strictly inside sandbox)
            preds = [int(p) for p in (val_df["feat1"] + val_df["feat2"] > 0)]
            probs = [[0.8, 0.2] if p == 0 else [0.2, 0.8] for p in preds]
            preds_payload = {
                "predictions": preds,
                "probabilities": probs,
                "classes": [0, 1],
                "row_ids": val_idx,
                "n_samples": len(val_idx),
            }
            preds_json.write_text(json.dumps(preds_payload))
            model_joblib.write_bytes(b"FROZEN_CANDIDATE_BUNDLE_BYTES")

            # Trusted evaluator evaluates predictions without unpickling or importing custom model
            eval_res = evaluate_custom_candidate(
                workflow_dir=wf_dir,
                candidate_id="custom_linear_v1",
                predictions_path=preds_json,
                bundle_dir=model_joblib,
                slices=["group_slice"],
                constraints={"max_bundle_bytes": {"max": 10_000_000}},
            )

            self.assertEqual(eval_res["candidate_id"], "custom_linear_v1")
            self.assertEqual(eval_res["status"], "evaluated")
            self.assertTrue(eval_res["is_feasible"])
            self.assertIn("balanced_accuracy", eval_res["metrics"])
            self.assertIn("group_slice", eval_res["slice_diagnostics"])
            self.assertIn("A", eval_res["slice_diagnostics"]["group_slice"])
            self.assertIn("B", eval_res["slice_diagnostics"]["group_slice"])

            # Negative test: misaligned row_ids must be rejected
            misaligned_json = tmp / "misaligned_preds.json"
            misaligned_payload = dict(json.loads(preds_json.read_text()))
            misaligned_payload["row_ids"] = list(reversed(misaligned_payload["row_ids"]))
            misaligned_json.write_text(json.dumps(misaligned_payload))

            with self.assertRaises(WorkflowError) as cm:
                evaluate_custom_candidate(
                    workflow_dir=wf_dir,
                    candidate_id="custom_linear_v1",
                    predictions_path=misaligned_json,
                    bundle_dir=model_joblib,
                )
            self.assertIn("row identity misalignment", str(cm.exception))

            # Negative test: constraint violation marks candidate infeasible
            eval_constrained = evaluate_custom_candidate(
                workflow_dir=wf_dir,
                candidate_id="custom_linear_v1",
                predictions_path=preds_json,
                bundle_dir=model_joblib,
                constraints={"max_bundle_bytes": {"max": 1}},  # bundle bytes will exceed 1
            )
            self.assertFalse(eval_constrained["is_feasible"])
            self.assertEqual(eval_constrained["constraints"]["max_bundle_bytes"]["verdict"], "fail")


if __name__ == "__main__":
    unittest.main()
