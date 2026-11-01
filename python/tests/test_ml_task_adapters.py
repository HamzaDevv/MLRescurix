"""Unit and Clean-Environment Tests for ML Task Adapters (Sequential & Tabular)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.custom.contracts import ResourceRequestContract
from mlrescurix_worker.custom.delivery import BundleVerificationError, CustomBundlePackager
from mlrescurix_worker.custom.sequential_candidate import SequentialTorchCandidate
from mlrescurix_worker.custom.tabular_candidate import TabularCandidatePlugin


def test_sequential_candidate_lifecycle(tmp_path: Path):
    # Create 3D sequential synthetic dataset: 100 sequences, 10 time steps, 4 features
    N, seq_len, in_feat = 80, 10, 4
    np.random.seed(42)
    X = np.random.randn(N, seq_len, in_feat).astype(np.float32)
    # Simple binary target based on sum of first feature across time
    y = (X[:, :, 0].sum(axis=1) > 0).astype(int)

    cand = SequentialTorchCandidate(
        hidden_dim=16,
        num_layers=1,
        learning_rate=0.01,
        batch_size=16,
        is_classification=True,
        seq_length=seq_len,
        in_features=in_feat,
    )

    req = ResourceRequestContract(ram_bytes_limit=256 * 1024 * 1024)
    cand.prepare({"in_features": in_feat, "seq_length": seq_len}, req)

    # 1. Pilot
    pilot_rep = cand.pilot((X[:8], y[:8]), req)
    assert pilot_rep.feasible
    assert pilot_rep.microbatch_size > 0

    # 2. Train
    split = 60
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    state = cand.train((X_train, y_train), max_steps=15)
    assert state.step >= 15

    # 3. Evaluate
    metrics = cand.evaluate((X_val, y_val))
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    # 4. Predict
    preds = cand.predict(X_val)
    assert len(preds) == len(X_val)
    assert set(preds).issubset({0, 1})

    probs = cand.predict_proba(X_val)
    assert probs is not None
    assert probs.shape == (len(X_val), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)

    # 5. Checkpoint save & load
    ckpt_dir = tmp_path / "seq_ckpt"
    manifest = cand.save_checkpoint(ckpt_dir, step=15, training_state=state)
    assert (ckpt_dir / "weights.pt").is_file()
    assert "weights_sha256" in manifest

    cand2 = SequentialTorchCandidate(
        hidden_dim=16,
        num_layers=1,
        is_classification=True,
        seq_length=seq_len,
        in_features=in_feat,
    )
    cand2.prepare({"in_features": in_feat, "seq_length": seq_len}, req)
    cand2.load_checkpoint(ckpt_dir)

    preds2 = cand2.predict(X_val)
    np.testing.assert_array_equal(preds, preds2)


def test_tabular_candidate_lifecycle(tmp_path: Path):
    N, n_feat = 100, 5
    np.random.seed(42)
    X = np.random.default_rng(42).standard_normal((N, n_feat)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    cand = TabularCandidatePlugin(max_iter=20, learning_rate=0.1, is_classification=True)
    req = ResourceRequestContract()

    # 1. Pilot
    pilot_rep = cand.pilot((X[:16], y[:16]), req)
    assert pilot_rep.feasible

    # 2. Train
    split = 80
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    state = cand.train((X_train, y_train))
    assert state.step > 0

    # 3. Evaluate
    metrics = cand.evaluate((X_val, y_val))
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.5

    # 4. Predict
    preds = cand.predict(X_val)
    assert len(preds) == len(X_val)

    probs = cand.predict_proba(X_val)
    assert probs is not None
    assert probs.shape == (len(X_val), 2)

    # 5. Checkpoint save & load
    ckpt_dir = tmp_path / "tab_ckpt"
    cand.save_checkpoint(ckpt_dir, step=state.step, training_state=state)
    assert (ckpt_dir / "model.joblib").is_file()

    cand2 = TabularCandidatePlugin(is_classification=True)
    cand2.load_checkpoint(ckpt_dir)
    preds2 = cand2.predict(X_val)
    np.testing.assert_array_equal(preds, preds2)


def test_clean_subprocess_inference_without_pythonpath(tmp_path: Path):
    # Package a trained TabularCandidatePlugin and execute inference.py in clean subprocess
    N, n_feat = 60, 3
    feature_names = ["feat_a", "feat_b", "feat_c"]
    X = np.random.default_rng(42).standard_normal((N, n_feat)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    df_train = pd.DataFrame(X, columns=feature_names)
    cand = TabularCandidatePlugin(max_iter=10, is_classification=True)
    cand.prepare({"features": feature_names}, ResourceRequestContract())
    cand.train((df_train, y))
    cand.decision_threshold = 0.99
    assert np.any(cand.predict(df_train) != cand.model.predict(X))

    packager = CustomBundlePackager()
    bundle_dir = tmp_path / "test_bundle"

    raw_examples = df_train.head(3).to_dict(orient="records")
    weights_path = tmp_path / "weights.joblib"
    cand.save_weights(weights_path)

    with pytest.raises(BundleVerificationError, match="differs from the candidate"):
        packager.package(candidate=cand, output_bundle_dir=tmp_path / "mismatch",
                         target_column="target", task_type="binary_classification",
                         primary_metric="accuracy", primary_metric_score=0.90,
                         metric_direction="maximize", feature_names=feature_names,
                         raw_example_records=raw_examples, source_weights_path=weights_path,
                         decision_threshold=0.5)

    # Note: inference.py generated by CustomBundlePackager loads weights
    bundle = packager.package(
        candidate=cand,
        output_bundle_dir=bundle_dir,
        target_column="target",
        task_type="binary_classification",
        primary_metric="accuracy",
        primary_metric_score=0.90,
        metric_direction="maximize",
        feature_names=feature_names,
        raw_example_records=raw_examples,
        source_weights_path=weights_path,
    )
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert manifest["format"] == "mlrescurix.bundle.v2"
    assert bundle.inference_entrypoint == "inference.py"
    assert (bundle_dir / "manifest.json").is_file()
    assert manifest["model"]["decision_threshold"] == 0.99

    # Prepare input records for inference
    input_records = df_train.head(5).to_dict(orient="records")
    input_json = tmp_path / "input.json"
    output_json = tmp_path / "output.json"
    with open(input_json, "w") as f:
        json.dump(input_records, f)

    # Clean subprocess execution with PYTHONPATH unset!
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)

    cmd = [
        sys.executable,
        str(bundle_dir / "inference.py"),
        "--input",
        str(input_json),
        "--output",
        str(output_json),
    ]
    res = subprocess.run(cmd, env=clean_env, capture_output=True, text=True)
    assert res.returncode == 0, f"Inference script failed: stderr={res.stderr}"
    assert output_json.is_file()

    with open(output_json, "r") as f:
        predictions = json.load(f)
    assert len(predictions["prediction"]) == 5
    np.testing.assert_array_equal(predictions["prediction"], cand.predict(df_train.head(5)))
    np.testing.assert_allclose(predictions["probabilities"], cand.predict_proba(df_train.head(5)))

    # Test malformed input: missing required feature -> fails closed with non-zero exit code
    bad_input_missing = tmp_path / "bad_input_missing.json"
    with open(bad_input_missing, "w") as f:
        json.dump([{"feat_a": 1.0, "feat_b": 2.0}], f) # missing feat_c
    bad_out = tmp_path / "bad_out.json"
    res_bad = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(bad_input_missing), "--output", str(bad_out)],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert res_bad.returncode != 0
    assert "missing required feature 'feat_c'" in res_bad.stderr

    # Test malformed input: uncoercible string value -> fails closed
    bad_input_type = tmp_path / "bad_input_type.json"
    with open(bad_input_type, "w") as f:
        json.dump([{"feat_a": "invalid_not_number", "feat_b": 2.0, "feat_c": 3.0}], f)
    res_bad_type = subprocess.run(
        [sys.executable, str(bundle_dir / "inference.py"), "--input", str(bad_input_type), "--output", str(bad_out)],
        env=clean_env,
        capture_output=True,
        text=True,
    )
    assert res_bad_type.returncode != 0
    assert "cannot be coerced to float" in res_bad_type.stderr
