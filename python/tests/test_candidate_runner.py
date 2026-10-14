"""Tests for Candidate Runner and Phase P4 protocol."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.candidate_runner import (
    compute_source_hash,
    evaluate_candidate_cv,
    evaluate_diagnostic_slices,
    validate_candidate_manifest,
    validate_predictions,
)
from mlrescurix_worker.scientific.workflow import WorkflowError


def test_manifest_validation_and_source_hash():
    with tempfile.TemporaryDirectory() as td:
        cand_dir = Path(td)
        (cand_dir / "entrypoint.py").write_text("print('hello')\n")
        computed_sha = compute_source_hash(cand_dir)

        manifest = {
            "candidate_id": "custom-linear-v1",
            "name": "Custom Linear Classifier",
            "task_type": "binary",
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": "row_id",
            "class_order": [0, 1],
            "source_hash": computed_sha,
        }
        validated = validate_candidate_manifest(manifest, cand_dir)
        assert validated["candidate_id"] == "custom-linear-v1"
        assert validated["source_hash"] == computed_sha

        # Tampering source hash
        bad_manifest = dict(manifest, source_hash="deadbeef" * 8)
        with pytest.raises(WorkflowError, match="source hash mismatch"):
            validate_candidate_manifest(bad_manifest, cand_dir)

        # Missing required field
        bad_manifest2 = dict(manifest)
        del bad_manifest2["task_type"]
        with pytest.raises(WorkflowError, match="missing required field"):
            validate_candidate_manifest(bad_manifest2, cand_dir)

        # Duplicate classes in class_order
        bad_manifest3 = dict(manifest, class_order=[0, 0])
        with pytest.raises(WorkflowError, match="duplicate classes"):
            validate_candidate_manifest(bad_manifest3, cand_dir)


def test_predictions_validation_success_and_failures():
    manifest = {
        "candidate_id": "c1",
        "name": "C1",
        "task_type": "binary",
        "entrypoint": "entrypoint.py",
        "supported_operations": ["fit", "predict"],
        "stable_row_keys": "row_id",
        "class_order": [0, 1],
    }

    # Valid predictions
    df_valid = pd.DataFrame({
        "row_id": [10, 20, 30],
        "pred": [0, 1, 0],
        "prob_0": [0.8, 0.2, 0.9],
        "prob_1": [0.2, 0.8, 0.1],
    })
    res = validate_predictions(df_valid, [10, 20, 30], manifest)
    assert res["prediction"] == [0, 1, 0]
    assert len(res["probabilities"]) == 3

    # Row misalignment: missing row_id column
    df_missing_col = pd.DataFrame({"pred": [0, 1, 0], "prob_0": [0.8, 0.2, 0.9], "prob_1": [0.2, 0.8, 0.1]})
    with pytest.raises(WorkflowError, match="row misalignment: prediction output missing"):
        validate_predictions(df_missing_col, [10, 20, 30], manifest)

    # Row misalignment: wrong count
    df_wrong_count = pd.DataFrame({"row_id": [10, 20], "pred": [0, 1], "prob_0": [0.8, 0.2], "prob_1": [0.2, 0.8]})
    with pytest.raises(WorkflowError, match="row count .* does not match"):
        validate_predictions(df_wrong_count, [10, 20, 30], manifest)

    # Row misalignment: wrong row IDs
    df_wrong_ids = pd.DataFrame({"row_id": [10, 20, 99], "pred": [0, 1, 0], "prob_0": [0.8, 0.2, 0.9], "prob_1": [0.2, 0.8, 0.1]})
    with pytest.raises(WorkflowError, match="prediction row keys do not match input row keys"):
        validate_predictions(df_wrong_ids, [10, 20, 30], manifest)

    # Class order discrepancy: missing prob_1
    df_missing_prob = pd.DataFrame({"row_id": [10, 20, 30], "pred": [0, 1, 0], "prob_0": [0.8, 0.2, 0.9]})
    with pytest.raises(WorkflowError, match="class order discrepancy"):
        validate_predictions(df_missing_prob, [10, 20, 30], manifest)

    # Probabilities don't sum to 1
    df_bad_sum = pd.DataFrame({"row_id": [10, 20, 30], "pred": [0, 1, 0], "prob_0": [0.8, 0.2, 0.9], "prob_1": [0.5, 0.8, 0.1]})
    with pytest.raises(WorkflowError, match="probabilities do not sum to 1.0"):
        validate_predictions(df_bad_sum, [10, 20, 30], manifest)


def test_diagnostic_slices_min_support_suppression():
    y_true = np.array([0] * 30 + [1] * 30 + [0] * 5)
    y_pred = np.array([0] * 30 + [1] * 30 + [1] * 5)
    y_prob = np.column_stack([1 - y_pred * 0.8, y_pred * 0.8])
    frame = pd.DataFrame({
        "category": ["group_A"] * 30 + ["group_B"] * 30 + ["tiny_group"] * 5,
    })

    # min_slice_support = 20
    slices = evaluate_diagnostic_slices(
        frame=frame,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        slice_columns=["category"],
        task_type="binary",
        class_order=[0, 1],
        min_slice_support=20,
    )
    cat_slices = slices["category"]
    assert cat_slices["group_A"]["status"] == "reported"
    assert cat_slices["group_B"]["status"] == "reported"
    assert cat_slices["tiny_group"]["status"] == "suppressed"
    assert "below min_slice_support threshold" in cat_slices["tiny_group"]["reason"]


def test_candidate_cv_evaluation_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cand_dir = root / "candidate"
        cand_dir.mkdir()

        # Create candidate script implementing fit and predict CLI protocol
        entrypoint_code = r"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--train-path", type=Path, required=True)
    fit_p.add_argument("--model-dir", type=Path, required=True)
    
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--input-path", type=Path, required=True)
    pred_p.add_argument("--model-dir", type=Path, required=True)
    pred_p.add_argument("--output-path", type=Path, required=True)
    
    args = parser.parse_args()
    
    if args.action == "fit":
        df = pd.read_csv(args.train_path)
        # Check target is present in train
        assert "target" in df.columns
        mean_val = float(df[df["target"] == 1]["f1"].mean() if (df["target"] == 1).any() else 0.0)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        (args.model_dir / "weights.txt").write_text(str(mean_val))
    else:
        df = pd.read_csv(args.input_path)
        # Verify label isolation: target column MUST NOT be present in prediction input!
        assert "target" not in df.columns, "Security violation: target column leaked into validation features!"
        threshold = float((args.model_dir / "weights.txt").read_text())
        probs_1 = 1.0 / (1.0 + np.exp(-(df["f1"] - threshold)))
        probs_0 = 1.0 - probs_1
        preds = (probs_1 >= 0.5).astype(int)
        
        out_df = pd.DataFrame({
            "row_id": df["row_id"],
            "pred": preds,
            "prob_0": probs_0,
            "prob_1": probs_1,
        })
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
"""
        (cand_dir / "entrypoint.py").write_text(entrypoint_code)
        source_hash = compute_source_hash(cand_dir)

        manifest = {
            "candidate_id": "test-linear-candidate",
            "name": "Test Linear Candidate",
            "task_type": "binary",
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": "row_id",
            "class_order": [0, 1],
            "raw_input_schema": {"features": ["f1", "f2"], "target": "target"},
            "source_hash": source_hash,
        }
        (cand_dir / "manifest.json").write_text(json.dumps(manifest))

        # Create synthetic dataset with 60 rows
        np.random.seed(42)
        n = 60
        f1 = np.random.randn(n)
        f2 = np.random.randn(n)
        target = (f1 + f2 > 0).astype(int)
        cat = ["A"] * 25 + ["B"] * 25 + ["rare"] * 10
        dataset_df = pd.DataFrame({
            "row_id": np.arange(n),
            "f1": f1,
            "f2": f2,
            "target": target,
            "category": cat,
        })
        dataset_csv = root / "dataset.csv"
        dataset_df.to_csv(dataset_csv, index=False)

        # Run evaluate_candidate_cv
        receipt = evaluate_candidate_cv(
            dataset_csv=dataset_csv,
            manifest=manifest,
            candidate_dir=cand_dir,
            folds=3,
            seed=2026,
            diagnostic_slices=["category"],
        )
        assert receipt["status"] == "evaluated"
        assert receipt["candidate_id"] == "test-linear-candidate"
        assert receipt["source_hash"] == source_hash
        assert "balanced_accuracy" in receipt["metrics"]
        assert "log_loss" in receipt["metrics"]
        assert len(receipt["folds_evaluated"]) == 3

        # Slices: group A and B have 25 rows (>= 20), rare has 10 (< 20)
        slices = receipt["diagnostic_slices"]["category"]
        assert slices["A"]["status"] == "reported"
        assert slices["B"]["status"] == "reported"
        assert slices["rare"]["status"] == "suppressed"


def test_cli_candidate_evaluate_invocation():
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cand_dir = root / "candidate"
        cand_dir.mkdir()

        entrypoint_code = r"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--train-path", type=Path, required=True)
    fit_p.add_argument("--model-dir", type=Path, required=True)
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--input-path", type=Path, required=True)
    pred_p.add_argument("--model-dir", type=Path, required=True)
    pred_p.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "fit":
        args.model_dir.mkdir(parents=True, exist_ok=True)
        (args.model_dir / "weights.txt").write_text("0.5")
    else:
        df = pd.read_csv(args.input_path)
        out_df = pd.DataFrame({
            "row_id": df["row_id"],
            "pred": [1] * len(df),
            "prob_0": [0.1] * len(df),
            "prob_1": [0.9] * len(df),
        })
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.output_path, index=False)

if __name__ == "__main__":
    main()
"""
        (cand_dir / "entrypoint.py").write_text(entrypoint_code)
        source_hash = compute_source_hash(cand_dir)

        manifest = {
            "candidate_id": "cli-test-candidate",
            "name": "CLI Test Candidate",
            "task_type": "binary",
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": "row_id",
            "class_order": [0, 1],
            "raw_input_schema": {"features": ["f1"], "target": "target"},
            "source_hash": source_hash,
        }
        (cand_dir / "manifest.json").write_text(json.dumps(manifest))

        dataset_df = pd.DataFrame({
            "row_id": np.arange(30),
            "f1": np.random.randn(30),
            "target": [0] * 15 + [1] * 15,
        })
        dataset_csv = root / "dataset.csv"
        dataset_df.to_csv(dataset_csv, index=False)

        spec = {
            "dataset_path": str(dataset_csv.resolve()),
            "candidate_dir": str(cand_dir.resolve()),
            "manifest": manifest,
            "folds": 2,
            "seed": 2026,
        }
        spec_file = root / "spec.json"
        spec_file.write_text(json.dumps(spec))
        output_file = root / "output.json"

        env = os.environ.copy()
        env["PYTHONPATH"] = "python/src"
        cmd = [
            sys.executable, "-m", "mlrescurix_worker.scientific.cli",
            "candidate-evaluate",
            "--spec", str(spec_file.resolve()),
            "--output", str(output_file.resolve()),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
        assert output_file.exists()
        out = json.loads(output_file.read_text())
        assert out["status"] == "evaluated"
        assert out["candidate_id"] == "cli-test-candidate"

