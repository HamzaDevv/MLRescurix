"""Exploratory evaluation of a packaged bundle on user-supplied labeled data.

This script does not establish holdout provenance, enforce one-time access, or certify a release.

Checks manifest file hashes, requires harness bundle verification, replays
examples in a subprocess, and computes exploratory metrics. These checks do not
verify historical data exposure or enforce a trusted release protocol.
"""
from __future__ import annotations

import sys
# CRITICAL: Prevent Python runtime from writing bytecode (.pyc) into the bundle directory,
# which would corrupt bundle integrity and cause mlrescurix-harness verify-bundle to fail.
sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd


def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    if y.shape != prediction.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("SMAPE requires nonempty, finite, matching targets and predictions")
    denominator = np.abs(y) + np.abs(prediction)
    ratio = np.divide(2 * np.abs(y - prediction), denominator,
                      out=np.zeros_like(denominator), where=denominator != 0)
    return float(100 * np.mean(ratio))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def find_harness_binary() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "target" / "release" / "mlrescurix-harness",
        repo_root / "target" / "debug" / "mlrescurix-harness",
    ]
    for c in candidates:
        if c.is_file() and subprocess.os.access(c, subprocess.os.X_OK):
            return c
    which_path = shutil.which("mlrescurix-harness")
    if which_path:
        return Path(which_path).resolve()
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a packaged candidate bundle on holdout data with release provenance.")
    parser.add_argument("--bundle", required=True, help="Path to packaged bundle directory")
    parser.add_argument("--holdout-csv", required=True, help="Path to holdout CSV")
    parser.add_argument("--target", default="price", help="Target column name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for inference")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle).resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    inference_path = bundle_dir / "inference.py"
    if not inference_path.is_file():
        raise FileNotFoundError(f"Missing inference.py in bundle: {inference_path}")

    # 1. Manifest and Contract Integrity Verification
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in bundle: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate_id = manifest.get("candidate_id")
    project_id = manifest.get("project_id")

    scorecard_path = bundle_dir / "scorecard.json"
    if not scorecard_path.is_file():
        raise FileNotFoundError(f"Missing scorecard.json in bundle: {scorecard_path}")
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))

    dev_cv_score = scorecard.get("primary_metric")
    candidate_identity = scorecard.get("candidate_identity")

    # Verify physical file hashes declared in manifest
    manifest_files = manifest.get("files", {})
    for rel_path, file_meta in manifest_files.items():
        physical_path = bundle_dir / rel_path
        if not physical_path.is_file():
            raise FileNotFoundError(f"File declared in manifest.json missing from bundle: {rel_path}")
        actual_sha = sha256_file(physical_path)
        expected_sha = file_meta.get("sha256")
        if actual_sha != expected_sha:
            raise ValueError(f"File '{rel_path}' hash mismatch: actual {actual_sha} != manifest {expected_sha}")

    # 2. Rust Harness Bundle Verification
    harness_bin = find_harness_binary()
    harness_report = None
    if not harness_bin:
        raise RuntimeError("Harness binary required for bundle integrity verification")
    if harness_bin:
        verify_cmd = [str(harness_bin), "verify-bundle", "--bundle", str(bundle_dir), "--sandbox", "false"]
        v_res = subprocess.run(verify_cmd, capture_output=True, text=True)
        if v_res.returncode != 0:
            raise RuntimeError(f"mlrescurix-harness verify-bundle failed: {v_res.stderr or v_res.stdout}")
        try:
            harness_report = json.loads(v_res.stdout)
        except Exception:
            harness_report = {"raw_output": v_res.stdout.strip()}

    # 3. Independent Subprocess Inference Replay on examples_input.json
    examples_input_path = bundle_dir / "examples_input.json"
    examples_expected_path = bundle_dir / "examples_expected.json"
    if not examples_input_path.is_file() or not examples_expected_path.is_file():
        raise FileNotFoundError("Missing examples_input.json or examples_expected.json in bundle")

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp_out:
        tmp_out_path = Path(tmp_out.name)

    try:
        replay_cmd = ["python3", "-B", "inference.py", "--input", str(examples_input_path), "--output", str(tmp_out_path)]
        replay_res = subprocess.run(replay_cmd, cwd=str(bundle_dir), capture_output=True, text=True)
        if replay_res.returncode != 0:
            raise RuntimeError(f"Subprocess inference replay failed: {replay_res.stderr}")
        replay_output = json.loads(tmp_out_path.read_text(encoding="utf-8"))
        expected_output = json.loads(examples_expected_path.read_text(encoding="utf-8"))

        replay_preds = replay_output.get("prediction", replay_output.get("predictions", []))
        expected_preds = expected_output.get("prediction", expected_output.get("predictions", []))
        if len(replay_preds) != len(expected_preds) or not np.allclose(replay_preds, expected_preds, atol=1e-5):
            raise ValueError(f"Subprocess inference replay output mismatch with examples_expected.json: {replay_preds} vs {expected_preds}")
        subprocess_replay_verified = True
    finally:
        if tmp_out_path.exists():
            tmp_out_path.unlink()

    # 4. Clean in-process holdout inference (with bytecode writing suppressed)
    spec = importlib.util.spec_from_file_location("bundle_inference", str(inference_path))
    bundle_inference = importlib.util.module_from_spec(spec)
    sys.modules["bundle_inference"] = bundle_inference
    spec.loader.exec_module(bundle_inference)

    # Read holdout data
    holdout_path = Path(args.holdout_csv).resolve()
    df = pd.read_csv(holdout_path)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in holdout CSV")

    y_true = df[args.target].to_numpy()
    features_df = df.drop(columns=[args.target])
    model_schema_path = bundle_dir / "model_schema.json"
    if model_schema_path.is_file():
        try:
            schema_meta = json.loads(model_schema_path.read_text(encoding="utf-8"))
            model_features = schema_meta.get("features", [])
            if model_features and all(f in features_df.columns for f in model_features):
                features_df = features_df[model_features]
        except Exception:
            pass

    # Run inference in batches
    n_rows = len(df)
    preds = []
    t0 = time.perf_counter()
    for start_idx in range(0, n_rows, args.batch_size):
        end_idx = min(start_idx + args.batch_size, n_rows)
        batch_records = features_df.iloc[start_idx:end_idx].to_dict(orient="records")
        res = bundle_inference.predict(batch_records)
        batch_preds = res["prediction"]
        preds.extend(batch_preds)

    inference_duration_sec = time.perf_counter() - t0
    y_pred = np.asarray(preds, dtype=float)

    # Compute metrics
    smape_val = compute_smape(y_true, y_pred)
    mae_val = float(np.mean(np.abs(y_true - y_pred)))
    rmse_val = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2_val = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    # Re-check bundle integrity without deleting any files.
    if harness_bin:
        post_verify = subprocess.run(
            [str(harness_bin), "verify-bundle", "--bundle", str(bundle_dir), "--sandbox", "false"],
            capture_output=True, text=True,
        )
        if post_verify.returncode != 0:
            raise RuntimeError(f"Bundle integrity failed post-evaluation: {post_verify.stderr or post_verify.stdout}")

    results = {
        "status": "success",
        "evaluation_scope": "exploratory; holdout provenance and one-time access not verified",
        "production_certified": False,
        "bundle_path": str(bundle_dir),
        "candidate_id": candidate_id,
        "candidate_identity": candidate_identity,
        "project_id": project_id,
        "weights_sha256": manifest_files.get("frozen_model.joblib", {}).get("sha256"),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_integrity_verified": True,
        "harness_bundle_verification": harness_report.get("status") if harness_report else "binary_not_found",
        "subprocess_replay_verified": subprocess_replay_verified,
        "holdout_path": str(holdout_path),
        "holdout_rows": int(n_rows),
        "target": args.target,
        "dev_cv_smape": dev_cv_score,
        "holdout_smape": smape_val,
        "generalization_gap": float(abs(smape_val - dev_cv_score)) if dev_cv_score is not None else None,
        "mae": mae_val,
        "rmse": rmse_val,
        "r2": r2_val,
        "inference_duration_sec": round(inference_duration_sec, 2),
        "throughput_rows_per_sec": round(n_rows / max(inference_duration_sec, 0.001), 1),
        "predictions_summary": {
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
            "mean": float(np.mean(y_pred)),
            "median": float(np.median(y_pred)),
            "std": float(np.std(y_pred)),
        },
        "ground_truth_summary": {
            "min": float(np.min(y_true)),
            "max": float(np.max(y_true)),
            "mean": float(np.mean(y_true)),
            "median": float(np.median(y_true)),
            "std": float(np.std(y_true)),
        },
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
