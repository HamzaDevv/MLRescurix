import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlrescurix_worker.custom.contract import CustomEstimator, CustomTransformer
from mlrescurix_worker.custom.preflight import run_preflight_checks
from mlrescurix_worker.custom.runner import train_and_predict_isolated
from mlrescurix_worker.scientific.workflow import audit_csv, evaluate_custom_candidate


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

            # Create entrypoint script for custom estimator
            script_path = tmp / "my_custom_model.py"
            script_path.write_text("""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from mlrescurix_worker.custom.contract import CustomEstimator

class Estimator(CustomEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Estimator":
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
""")

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
            val_df = df.iloc[val_idx][["feat1", "feat2"]]  # label-free!

            train_csv = tmp / "train.csv"
            val_csv = tmp / "val.csv"
            train_df.to_csv(train_csv, index=False)
            val_df.to_csv(val_csv, index=False)

            preds_json = tmp / "predictions.json"
            model_joblib = tmp / "custom_model.joblib"

            # Run isolated runner
            train_and_predict_isolated(
                entrypoint=str(script_path),
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                target_column="target",
                output_predictions_path=str(preds_json),
                output_model_path=str(model_joblib),
            )
            self.assertTrue(preds_json.is_file())
            self.assertTrue(model_joblib.is_file())

            # Trusted evaluator evaluates predictions without unpickling or importing custom model
            eval_res = evaluate_custom_candidate(
                workflow_dir=wf_dir,
                candidate_id="custom_linear_v1",
                predictions_path=preds_json,
                slices=["group_slice"],
            )

            self.assertEqual(eval_res["candidate_id"], "custom_linear_v1")
            self.assertEqual(eval_res["status"], "evaluated")
            self.assertIn("balanced_accuracy", eval_res["metrics"])
            self.assertIn("group_slice", eval_res["slice_diagnostics"])
            self.assertIn("A", eval_res["slice_diagnostics"]["group_slice"])
            self.assertIn("B", eval_res["slice_diagnostics"]["group_slice"])


if __name__ == "__main__":
    unittest.main()
