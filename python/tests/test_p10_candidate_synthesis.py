"""Comprehensive Unit Tests for Phase P10.1 Autonomous Candidate Synthesis Engine."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.candidate_runner import (
    compute_source_hash,
    execute_candidate_cli,
    validate_candidate_manifest,
    validate_predictions,
)
from mlrescurix_worker.scientific.candidate_synthesis import (
    CandidateSynthesizer,
    ColumnDescriptor,
    InvalidSynthesisSpecError,
    MetricObjective,
    ModelIntent,
    ProblemDefinition,
    SynthesisSpec,
    TargetDescriptor,
    validate_synthesis_spec,
)
from mlrescurix_worker.scientific.workflow import WorkflowError

ROOT = Path(__file__).resolve().parents[2]


class TestP10CandidateSynthesis(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlr_test_p10_"))
        self.schema = json.loads((ROOT / "contracts/v2/candidate_synthesis.schema.json").read_text(encoding="utf-8"))

    def tear_down(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_p10_1_schema_validation(self):
        """P10.1: Validate synthesis spec schema conformance under Draft 2020-12."""
        jsonschema.Draft202012Validator.check_schema(self.schema)

        # 1. Valid binary classification spec
        spec_binary = SynthesisSpec(
            synthesis_id="synth-lgbm-01",
            candidate_id="cand_lgbm_binary",
            name="LightGBM Binary Classifier",
            problem_definition=ProblemDefinition(
                task_type="binary",
                target_column="target",
                dataset_id="ds_churn",
                id_column="row_id",
            ),
            input_descriptors=[
                ColumnDescriptor(name="age", data_type="numerical", imputation_strategy="median"),
                ColumnDescriptor(name="tenure", data_type="numerical", imputation_strategy="median"),
                ColumnDescriptor(name="contract", data_type="categorical", imputation_strategy="mode"),
            ],
            target_descriptor=TargetDescriptor(
                name="target",
                task_type="binary",
                class_order=["no", "yes"],
                positive_class="yes",
            ),
            metric_objective=MetricObjective(
                primary_metric="balanced_accuracy",
                direction="maximize",
                target_threshold=0.85,
            ),
            model_intent=ModelIntent(
                family="boosting",
                framework="lightgbm",
                hyperparameters={"n_estimators": 50, "learning_rate": 0.05},
                feature_pipeline={"scaling": "standard", "categorical_encoding": "onehot"},
            ),
        )
        spec_dict = spec_binary.to_dict()
        jsonschema.validate(instance=spec_dict, schema=self.schema)
        validate_synthesis_spec(spec_dict)

        # 2. Valid regression spec
        spec_reg = SynthesisSpec(
            synthesis_id="synth-rf-02",
            candidate_id="cand_rf_reg",
            name="Random Forest Regressor",
            problem_definition=ProblemDefinition(
                task_type="regression",
                target_column="price",
                dataset_id="ds_housing",
            ),
            metric_objective=MetricObjective(
                primary_metric="rmse",
                direction="minimize",
            ),
            model_intent=ModelIntent(
                family="tree",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 30, "max_depth": 8},
                feature_pipeline={"scaling": "none", "categorical_encoding": "ordinal"},
            ),
        )
        jsonschema.validate(instance=spec_reg.to_dict(), schema=self.schema)

        # 3. Negative validation: invalid task type
        bad_spec = spec_binary.to_dict()
        bad_spec["problem_definition"]["task_type"] = "unsupported_task"
        with self.assertRaises(InvalidSynthesisSpecError):
            validate_synthesis_spec(bad_spec)

        # 4. Negative validation: missing required field
        bad_spec2 = spec_binary.to_dict()
        del bad_spec2["problem_definition"]["target_column"]
        with self.assertRaises(InvalidSynthesisSpecError):
            validate_synthesis_spec(bad_spec2)

    def test_p10_2_lightgbm_candidate_synthesis_and_cli_protocol(self):
        """P10.2: Synthesize decoupled LightGBM candidate, execute fit and predict CLI."""
        cand_dir = self.tmp / "cand_lightgbm"
        spec = SynthesisSpec(
            synthesis_id="synth_lgb_test",
            candidate_id="cand_lgb_v1",
            problem_definition=ProblemDefinition(
                task_type="binary",
                target_column="churn",
                id_column="row_id",
            ),
            target_descriptor=TargetDescriptor(
                name="churn",
                task_type="binary",
                class_order=[0, 1],
            ),
            metric_objective=MetricObjective(
                primary_metric="balanced_accuracy",
                direction="maximize",
            ),
            model_intent=ModelIntent(
                family="boosting",
                framework="lightgbm",
                hyperparameters={"n_estimators": 40, "learning_rate": 0.1},
                feature_pipeline={"scaling": "standard", "categorical_encoding": "onehot"},
            ),
        )

        synthesized = CandidateSynthesizer.synthesize(spec, cand_dir)
        self.assertEqual(synthesized.candidate_id, "cand_lgb_v1")
        self.assertTrue((cand_dir / "entrypoint.py").is_file())
        self.assertTrue((cand_dir / "model.py").is_file())
        self.assertTrue((cand_dir / "features.py").is_file())
        self.assertTrue((cand_dir / "manifest.json").is_file())

        # Verify source hash in manifest matches directory
        computed_hash = compute_source_hash(cand_dir)
        self.assertEqual(synthesized.source_hash, computed_hash)
        self.assertEqual(synthesized.manifest["source_hash"], computed_hash)

        # Create synthetic training dataset
        np.random.seed(2026)
        n = 120
        df = pd.DataFrame({
            "row_id": np.arange(n),
            "num_1": np.random.randn(n),
            "num_2": np.random.uniform(10, 100, n),
            "cat_1": np.random.choice(["type_a", "type_b", "type_c"], n),
            "churn": (np.random.rand(n) > 0.5).astype(int),
        })

        train_csv = self.tmp / "train.csv"
        val_csv = self.tmp / "val_features.csv"
        df.to_csv(train_csv, index=False)

        val_df = df.iloc[:40].drop(columns=["churn"]).copy()
        val_df.to_csv(val_csv, index=False)

        model_dir = self.tmp / "model_out_lgb"
        pred_csv = self.tmp / "pred_lgb.csv"

        # 1. Execute fit CLI
        code, stdout, stderr = execute_candidate_cli(
            cand_dir,
            "entrypoint.py",
            "fit",
            ["--train-path", str(train_csv), "--model-dir", str(model_dir)],
        )
        self.assertEqual(code, 0, f"fit failed: {stderr}\n{stdout}")
        self.assertTrue((model_dir / "feature_pipeline.joblib").is_file() or (model_dir / "model.joblib").is_file())

        # 2. Execute predict CLI
        code, stdout, stderr = execute_candidate_cli(
            cand_dir,
            "entrypoint.py",
            "predict",
            ["--input-path", str(val_csv), "--model-dir", str(model_dir), "--output-path", str(pred_csv)],
        )
        self.assertEqual(code, 0, f"predict failed: {stderr}\n{stdout}")
        self.assertTrue(pred_csv.is_file())

        pred_df = pd.read_csv(pred_csv)
        self.assertEqual(len(pred_df), 40)
        self.assertIn("row_id", pred_df.columns)
        self.assertIn("pred", pred_df.columns)
        self.assertIn("prob_0", pred_df.columns)
        self.assertIn("prob_1", pred_df.columns)

        # Validate with harness prediction validator
        validated = validate_predictions(pred_df, val_df["row_id"].tolist(), synthesized.manifest)
        self.assertEqual(len(validated["prediction"]), 40)
        probs = np.array(validated["probabilities"])
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-3))

    def test_p10_3_pytorch_candidate_synthesis_and_cli_protocol(self):
        """P10.3: Synthesize PyTorch neural network candidate, train and predict."""
        cand_dir = self.tmp / "cand_pytorch"
        spec = SynthesisSpec(
            synthesis_id="synth_pt_test",
            candidate_id="cand_pt_nn",
            problem_definition=ProblemDefinition(
                task_type="multiclass",
                target_column="category",
                id_column="sample_id",
            ),
            target_descriptor=TargetDescriptor(
                name="category",
                task_type="multiclass",
                class_order=["class_a", "class_b", "class_c"],
            ),
            metric_objective=MetricObjective(
                primary_metric="log_loss",
                direction="minimize",
            ),
            model_intent=ModelIntent(
                family="neural",
                framework="pytorch",
                hyperparameters={"hidden_dim": 32, "epochs": 6, "learning_rate": 0.01},
                feature_pipeline={"scaling": "standard", "categorical_encoding": "onehot"},
            ),
        )

        synthesized = CandidateSynthesizer.synthesize(spec, cand_dir)
        self.assertEqual(synthesized.candidate_id, "cand_pt_nn")

        # Generate multiclass dataset
        np.random.seed(42)
        n = 90
        classes = ["class_a", "class_b", "class_c"]
        df = pd.DataFrame({
            "sample_id": np.arange(n),
            "x1": np.random.randn(n),
            "x2": np.random.randn(n) * 2.0,
            "category": np.random.choice(classes, n),
        })

        train_csv = self.tmp / "train_pt.csv"
        val_csv = self.tmp / "val_pt.csv"
        df.to_csv(train_csv, index=False)

        val_df = df.iloc[:30].drop(columns=["category"]).copy()
        val_df.to_csv(val_csv, index=False)

        model_dir = self.tmp / "model_out_pt"
        pred_csv = self.tmp / "pred_pt.csv"

        # Fit
        code, stdout, stderr = execute_candidate_cli(
            cand_dir,
            "entrypoint.py",
            "fit",
            ["--train-path", str(train_csv), "--model-dir", str(model_dir)],
        )
        self.assertEqual(code, 0, f"fit failed: {stderr}\n{stdout}")
        self.assertTrue((model_dir / "pytorch_weights.pt").is_file())
        self.assertTrue((model_dir / "pytorch_meta.json").is_file())

        # Predict
        code, stdout, stderr = execute_candidate_cli(
            cand_dir,
            "entrypoint.py",
            "predict",
            ["--input-path", str(val_csv), "--model-dir", str(model_dir), "--output-path", str(pred_csv)],
        )
        self.assertEqual(code, 0, f"predict failed: {stderr}\n{stdout}")

        pred_df = pd.read_csv(pred_csv)
        self.assertEqual(len(pred_df), 30)
        self.assertIn("sample_id", pred_df.columns)
        self.assertIn("pred", pred_df.columns)
        for c in classes:
            self.assertIn(f"prob_{c}", pred_df.columns)

        validated = validate_predictions(pred_df, val_df["sample_id"].tolist(), synthesized.manifest)
        probs = np.array(validated["probabilities"])
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-3))

    def test_p10_4_regression_candidate_synthesis(self):
        """P10.4: Synthesize Random Forest regression candidate and verify numeric predictions."""
        cand_dir = self.tmp / "cand_reg"
        spec = SynthesisSpec(
            synthesis_id="synth_reg_test",
            candidate_id="cand_rf_reg",
            problem_definition=ProblemDefinition(
                task_type="regression",
                target_column="target_val",
                id_column="row_id",
            ),
            metric_objective=MetricObjective(
                primary_metric="rmse",
                direction="minimize",
            ),
            model_intent=ModelIntent(
                family="tree",
                framework="scikit-learn",
                hyperparameters={"n_estimators": 25, "max_depth": 6},
                feature_pipeline={"scaling": "minmax"},
            ),
        )

        synthesized = CandidateSynthesizer.synthesize(spec, cand_dir)
        self.assertEqual(synthesized.manifest["task_type"], "regression")

        # Synthetic regression dataset
        n = 80
        df = pd.DataFrame({
            "row_id": np.arange(n),
            "f1": np.linspace(0, 10, n),
            "f2": np.random.randn(n),
            "target_val": np.linspace(0, 10, n) * 3.5 + np.random.randn(n) * 0.5,
        })
        train_csv = self.tmp / "train_reg.csv"
        val_csv = self.tmp / "val_reg.csv"
        df.to_csv(train_csv, index=False)
        val_df = df.iloc[50:].drop(columns=["target_val"]).copy()
        val_df.to_csv(val_csv, index=False)

        model_dir = self.tmp / "model_out_reg"
        pred_csv = self.tmp / "pred_reg.csv"

        code, _, stderr = execute_candidate_cli(
            cand_dir, "entrypoint.py", "fit", ["--train-path", str(train_csv), "--model-dir", str(model_dir)]
        )
        self.assertEqual(code, 0, f"fit failed: {stderr}")

        code, _, stderr = execute_candidate_cli(
            cand_dir, "entrypoint.py", "predict", ["--input-path", str(val_csv), "--model-dir", str(model_dir), "--output-path", str(pred_csv)]
        )
        self.assertEqual(code, 0, f"predict failed: {stderr}")

        pred_df = pd.read_csv(pred_csv)
        self.assertEqual(len(pred_df), 30)
        validated = validate_predictions(pred_df, val_df["row_id"].tolist(), synthesized.manifest)
        self.assertIsNone(validated["probabilities"])
        self.assertTrue(np.isfinite(validated["prediction"]).all())

    def test_p10_5_custom_modules_injection(self):
        """P10.5: Verify agent can inject custom modular source files into synthesized candidates."""
        cand_dir = self.tmp / "cand_custom"
        custom_code = '''"""Custom feature engineering module authored by agent."""
def custom_interaction(x1, x2):
    return x1 * x2 + (x1 ** 2)
'''
        spec = SynthesisSpec(
            synthesis_id="synth_custom_test",
            candidate_id="cand_custom_pipe",
            problem_definition=ProblemDefinition(
                task_type="binary",
                target_column="target",
            ),
            target_descriptor=TargetDescriptor(
                name="target",
                task_type="binary",
                class_order=[0, 1],
            ),
            metric_objective=MetricObjective(
                primary_metric="balanced_accuracy",
                direction="maximize",
            ),
            model_intent=ModelIntent(
                family="linear",
                framework="scikit-learn",
                custom_modules={"transforms/custom_math.py": custom_code},
            ),
        )

        synthesized = CandidateSynthesizer.synthesize(spec, cand_dir)
        custom_file = cand_dir / "transforms/custom_math.py"
        self.assertTrue(custom_file.is_file())
        self.assertIn("custom_interaction", custom_file.read_text())

        # Verify source hash includes custom file
        computed_hash = compute_source_hash(cand_dir)
        self.assertEqual(synthesized.source_hash, computed_hash)

    def test_p10_6_label_isolation_enforcement(self):
        """P10.6: Candidate predict step ignores target column if mistakenly present in input."""
        cand_dir = self.tmp / "cand_leakage_guard"
        spec = SynthesisSpec(
            synthesis_id="synth_leak_test",
            candidate_id="cand_leak_test",
            problem_definition=ProblemDefinition(
                task_type="binary",
                target_column="target",
                id_column="row_id",
            ),
            target_descriptor=TargetDescriptor(
                name="target",
                task_type="binary",
                class_order=[0, 1],
            ),
            metric_objective=MetricObjective(
                primary_metric="balanced_accuracy",
                direction="maximize",
            ),
            model_intent=ModelIntent(
                family="tree",
                framework="scikit-learn",
            ),
        )
        CandidateSynthesizer.synthesize(spec, cand_dir)

        # Train
        train_df = pd.DataFrame({
            "row_id": [0, 1, 2, 3],
            "feat": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 0, 1, 1],
        })
        train_csv = self.tmp / "leak_train.csv"
        train_df.to_csv(train_csv, index=False)

        model_dir = self.tmp / "leak_model"
        code, _, err = execute_candidate_cli(cand_dir, "entrypoint.py", "fit", ["--train-path", str(train_csv), "--model-dir", str(model_dir)])
        self.assertEqual(code, 0, err)

        # Input dataframe maliciously contains 'target'
        malicious_input_df = pd.DataFrame({
            "row_id": [10, 11],
            "feat": [1.5, 3.5],
            "target": [999, 999],  # Leaked dummy label
        })
        input_csv = self.tmp / "leak_input.csv"
        malicious_input_df.to_csv(input_csv, index=False)

        output_csv = self.tmp / "leak_out.csv"
        code, _, err = execute_candidate_cli(
            cand_dir, "entrypoint.py", "predict",
            ["--input-path", str(input_csv), "--model-dir", str(model_dir), "--output-path", str(output_csv)]
        )
        self.assertEqual(code, 0, err)

        out_df = pd.read_csv(output_csv)
        self.assertNotIn("target", out_df.columns)
        self.assertIn("pred", out_df.columns)
        self.assertIn("prob_0", out_df.columns)


if __name__ == "__main__":
    unittest.main()
