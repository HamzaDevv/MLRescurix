"""Exploratory Unequal-Compute Comparison: M3 Baseline vs M5 Adaptive Scientific Pipeline.

Predeclaration:
- Dataset: Tabular binary classification with numeric features, categorical columns, and nonlinear interactions (300 rows).
- Splits: 60% Train (180 samples), 20% Protected Validation (60 samples), 20% Sealed Test (60 samples).
- Exploratory budget: five candidate configurations per arm; training work is unequal.
  This standalone script does not exercise authoritative admission or a durable sealed-test gate.
  - Arm A (M3 Baseline): Linear and Random Forest hyperparameter exploration without feature engineering, calibration, or ensembling.
  - Arm B (M5 Adaptive): Split-aware Feature Engineering (robust scaling, out-of-fold target encoding, polynomial interactions)
                         + CPU HistGradientBoosting
                         + Split-aware Probability Calibration (Isotonic on train OOF)
                         + Constrained Voting Ensemble (with explicit parent lineage)
- Fair Accounting:
  - Search, fit, and OOF evaluation wall time tracked.
  - Remote provider expenditure reported separately ($0.00 offline).
  - Feasibility: Warm P95 inference latency (ms) and artifact bundle size (bytes).
  - Final sealed test evaluation performed strictly once per arm's apex candidate.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, brier_score_loss
from sklearn.model_selection import StratifiedKFold

# Ensure python/src is in sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "python" / "src"))

from mlrescurix_worker.scientific.features import (
    FeatureTransformSpec,
    SplitAwareFeaturePipeline,
)
from mlrescurix_worker.scientific.calibration import SplitAwareCalibrator
from mlrescurix_worker.scientific.ensemble import VotingEnsemble
from mlrescurix_worker.scientific.workflow import _pipeline, _score


def generate_benchmark_dataset(seed: int = 2026, n_samples: int = 300) -> pd.DataFrame:
    """Generate synthetic tabular dataset with numeric interactions and high-cardinality categoricals."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0.0, 1.0, n_samples)
    f2 = rng.uniform(-2.0, 2.0, n_samples)
    cat1 = rng.choice(["alpha", "beta", "gamma", "delta"], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    cat2 = rng.choice(["low", "medium", "high"], n_samples)

    # Nonlinear interaction + category signal
    signal = (f1 * f2 > 0.2).astype(float) + (cat1 == "alpha").astype(float) * 0.8 - (cat2 == "high").astype(float) * 0.5
    noise = rng.normal(0.0, 0.3, n_samples)
    prob = 1.0 / (1.0 + np.exp(-(signal + noise)))
    target = (prob > 0.5).astype(int)

    return pd.DataFrame({
        "num1": f1,
        "num2": f2,
        "cat1": cat1,
        "cat2": cat2,
        "target": target,
    })


def measure_p95_latency(predict_fn, sample_df: pd.DataFrame, reps: int = 20) -> float:
    """Measure warm P95 latency in milliseconds."""
    # Warmup
    for _ in range(3):
        predict_fn(sample_df)

    latencies = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        predict_fn(sample_df)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        latencies.append(elapsed_ms)
    return float(np.percentile(latencies, 95))


def run_m3_arm(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], target_col: str, seed: int = 2026) -> dict[str, Any]:
    """Execute M3 baseline search arm (5 matched trials)."""
    t_start = time.perf_counter()
    X_train = train_df[features]
    y_train = train_df[target_col].to_numpy()
    X_val = val_df[features]
    y_val = val_df[target_col].to_numpy()

    trials = [
        {"id": "m3-trial-1", "recipe": "linear", "params": {"C": 1.0}},
        {"id": "m3-trial-2", "recipe": "linear", "params": {"C": 0.1}},
        {"id": "m3-trial-3", "recipe": "tree", "params": {"n_estimators": 100, "max_depth": 6, "min_samples_leaf": 2, "max_features": 1.0}},
        {"id": "m3-trial-4", "recipe": "tree", "params": {"n_estimators": 64, "max_depth": 4, "min_samples_leaf": 4, "max_features": 0.8}},
        {"id": "m3-trial-5", "recipe": "tree", "params": {"n_estimators": 128, "max_depth": 8, "min_samples_leaf": 2, "max_features": 0.8}},
    ]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    trial_results = []
    best_pipe = None
    best_val_score = -1.0
    best_trial_id = None

    for t in trials:
        t_t0 = time.perf_counter()
        oof_scores = []
        for train_idx, test_idx in skf.split(X_train, y_train):
            fold_pipe = _pipeline(train_df.iloc[train_idx], features, "binary_classification", t["recipe"], seed, t["params"])
            fold_pipe.fit(X_train.iloc[train_idx], y_train[train_idx])
            preds = fold_pipe.predict(X_train.iloc[test_idx])
            oof_scores.append(balanced_accuracy_score(y_train[test_idx], preds))

        mean_oof = float(np.mean(oof_scores))
        fit_pipe = _pipeline(train_df, features, "binary_classification", t["recipe"], seed, t["params"])
        fit_pipe.fit(X_train, y_train)

        val_preds = fit_pipe.predict(X_val)
        val_score = float(balanced_accuracy_score(y_val, val_preds))
        val_probs = fit_pipe.predict_proba(X_val)
        val_brier = float(brier_score_loss(y_val, val_probs[:, 1]))

        fit_time_ms = (time.perf_counter() - t_t0) * 1000

        res = {
            "trial_id": t["id"],
            "recipe": t["recipe"],
            "params": t["params"],
            "oof_balanced_accuracy": mean_oof,
            "val_balanced_accuracy": val_score,
            "val_brier_score": val_brier,
            "fit_time_ms": fit_time_ms,
        }
        trial_results.append(res)

        if val_score > best_val_score:
            best_val_score = val_score
            best_pipe = fit_pipe
            best_trial_id = t["id"]

    total_wall_ms = (time.perf_counter() - t_start) * 1000
    p95_ms = measure_p95_latency(lambda df: best_pipe.predict(df[features]), val_df)

    # Serialize to measure artifact size
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        joblib.dump(best_pipe, tmp.name)
        artifact_bytes = os.path.getsize(tmp.name)

    return {
        "arm": "M3_Baseline",
        "trials": trial_results,
        "apex_trial_id": best_trial_id,
        "apex_model": best_pipe,
        "val_balanced_accuracy": best_val_score,
        "p95_inference_latency_ms": p95_ms,
        "artifact_bytes": artifact_bytes,
        "search_wall_ms": total_wall_ms,
        "provider_spend_usd": 0.00,
    }


def run_m5_arm(train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str], target_col: str, seed: int = 2026) -> dict[str, Any]:
    """Execute M5 adaptive scientific arm (5 matched trials with feature eng, boosting, calibration, and ensembling)."""
    t_start = time.perf_counter()
    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()

    # Step 1: Split-Aware Feature Engineering on Training Data Only
    feat_spec = FeatureTransformSpec(
        transform_id="m5-tf-1",
        numeric_scaler="robust",
        interactions=True,
        discretize_bins=4,
        missing_indicators=True,
        target_encoding=True,
        max_interaction_features=5,
    )
    feature_pipeline = SplitAwareFeaturePipeline(feat_spec, task="binary_classification", seed=seed)
    X_train_trans = feature_pipeline.fit_transform_training(train_df[features], y_train)
    X_val_trans = feature_pipeline.transform(val_df[features])

    trans_features = list(X_train_trans.columns)
    combined_train = X_train_trans.copy()
    combined_train[target_col] = y_train

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    trial_results = []
    models: dict[str, Any] = {}

    # Trial 1: Linear on engineered features
    t1_spec = {"recipe": "linear", "params": {"C": 1.0}}
    t1_pipe = _pipeline(combined_train, trans_features, "binary_classification", "linear", seed, t1_spec["params"])
    t1_pipe.fit(X_train_trans, y_train)
    t1_val_preds = t1_pipe.predict(X_val_trans)
    t1_val_score = float(balanced_accuracy_score(y_val, t1_val_preds))
    models["m5-trial-1"] = t1_pipe
    trial_results.append({
        "trial_id": "m5-trial-1",
        "recipe": "linear_engineered",
        "description": "Feature pipeline + Linear model",
        "val_balanced_accuracy": t1_val_score,
    })

    # Trial 2: HistGradientBoosting CPU baseline (learning_rate=0.05, max_iter=50, max_leaf_nodes=15)
    t2_spec = {"recipe": "hist_boosting", "params": {"learning_rate": 0.05, "max_iter": 50, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.1}}
    t2_pipe = _pipeline(combined_train, trans_features, "binary_classification", "hist_boosting", seed, t2_spec["params"])
    t2_pipe.fit(X_train_trans, y_train)
    t2_val_preds = t2_pipe.predict(X_val_trans)
    t2_val_score = float(balanced_accuracy_score(y_val, t2_val_preds))
    models["m5-trial-2"] = t2_pipe
    trial_results.append({
        "trial_id": "m5-trial-2",
        "recipe": "hist_boosting_conservative",
        "description": "Feature pipeline + HistGradientBoosting CPU (conservative)",
        "val_balanced_accuracy": t2_val_score,
    })

    # Trial 3: HistGradientBoosting CPU tuned (learning_rate=0.1, max_iter=75, max_leaf_nodes=31)
    t3_spec = {"recipe": "hist_boosting", "params": {"learning_rate": 0.1, "max_iter": 75, "max_leaf_nodes": 31, "min_samples_leaf": 15, "l2_regularization": 0.5}}
    t3_pipe = _pipeline(combined_train, trans_features, "binary_classification", "hist_boosting", seed, t3_spec["params"])
    t3_pipe.fit(X_train_trans, y_train)
    t3_val_preds = t3_pipe.predict(X_val_trans)
    t3_val_score = float(balanced_accuracy_score(y_val, t3_val_preds))
    t3_val_probs = t3_pipe.predict_proba(X_val_trans)
    models["m5-trial-3"] = t3_pipe
    trial_results.append({
        "trial_id": "m5-trial-3",
        "recipe": "hist_boosting_tuned",
        "description": "Feature pipeline + HistGradientBoosting CPU (tuned)",
        "val_balanced_accuracy": t3_val_score,
        "val_brier_uncalibrated": float(brier_score_loss(y_val, t3_val_probs[:, 1])),
    })

    # Trial 4: Split-Aware Probability Calibration on Trial 3 using Out-of-Fold training probabilities
    oof_probs = np.zeros((len(train_df), 2))
    for train_idx, test_idx in skf.split(train_df[features], y_train):
        # Fit EVERY transform inside the outer fold, including target encoding.
        fold_features = SplitAwareFeaturePipeline(feat_spec, seed=seed)
        fold_train = fold_features.fit_transform_training(train_df[features].iloc[train_idx], y_train[train_idx])
        fold_test = fold_features.transform(train_df[features].iloc[test_idx])
        f_pipe = _pipeline(fold_train, list(fold_train.columns), "binary_classification", "hist_boosting", seed, t3_spec["params"])
        f_pipe.fit(fold_train, y_train[train_idx])
        oof_probs[test_idx] = f_pipe.predict_proba(fold_test)

    calibrator = SplitAwareCalibrator(method="isotonic")
    calibrator.fit_from_oof(oof_probs, y_train, classes=[0, 1])
    cal_val_probs = calibrator.transform(t3_val_probs)
    cal_val_brier = float(brier_score_loss(y_val, cal_val_probs[:, 1]))
    cal_val_preds = (cal_val_probs[:, 1] >= 0.5).astype(int)
    cal_val_score = float(balanced_accuracy_score(y_val, cal_val_preds))

    trial_results.append({
        "trial_id": "m5-trial-4",
        "recipe": "calibrated_hist_boosting",
        "description": "Trial 3 + Isotonic calibration on OOF training probabilities",
        "val_balanced_accuracy": cal_val_score,
        "val_brier_calibrated": cal_val_brier,
        "brier_improvement": float(trial_results[-1]["val_brier_uncalibrated"] - cal_val_brier),
    })

    # Trial 5: Constrained Voting Ensemble (Lineage: parent 1 = m5-trial-1, parent 2 = m5-trial-3)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1_path = tmp_path / "p1.joblib"
        p2_path = tmp_path / "p2.joblib"
        joblib.dump(t1_pipe, p1_path)
        joblib.dump(t3_pipe, p2_path)

        ensemble = VotingEnsemble(
            parent_ids=["m5-trial-1", "m5-trial-3"],
            weights=[0.35, 0.65],
            task="binary_classification",
        )
        ensemble.load_parent_models([p1_path, p2_path])
        ens_pred_dict = ensemble.predict(X_val_trans)
        ens_preds = ens_pred_dict["prediction"]
        ens_val_score = float(balanced_accuracy_score(y_val, ens_preds))

        trial_results.append({
            "trial_id": "m5-trial-5",
            "recipe": "voting_ensemble",
            "description": "Weighted ensemble of Linear and Tuned HistGradientBoosting",
            "val_balanced_accuracy": ens_val_score,
            "parent_lineage": ["m5-trial-1", "m5-trial-3"],
        })

    # Select apex candidate based on validation balanced accuracy
    best_res = max(trial_results, key=lambda r: r["val_balanced_accuracy"])
    best_trial_id = best_res["trial_id"]
    best_val_score = best_res["val_balanced_accuracy"]

    total_wall_ms = (time.perf_counter() - t_start) * 1000

    # Build full inference pipeline function for warm latency measurement
    if best_trial_id == "m5-trial-5":
        def full_pipeline_predict(raw_df: pd.DataFrame):
            trans = feature_pipeline.transform(raw_df[features])
            return ensemble.predict(trans)
    elif best_trial_id == "m5-trial-4":
        def full_pipeline_predict(raw_df: pd.DataFrame):
            trans = feature_pipeline.transform(raw_df[features])
            raw_p = t3_pipe.predict_proba(trans)
            return calibrator.transform(raw_p)
    else:
        chosen_m = models[best_trial_id]
        def full_pipeline_predict(raw_df: pd.DataFrame):
            trans = feature_pipeline.transform(raw_df[features])
            return chosen_m.predict(trans)

    p95_ms = measure_p95_latency(full_pipeline_predict, val_df)

    # Bundle size: feature pipeline + model
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        bundle = {"feature_pipeline": feature_pipeline, "model": models.get(best_trial_id, t3_pipe)}
        if best_trial_id == "m5-trial-4":
            bundle["calibrator"] = calibrator
        elif best_trial_id == "m5-trial-5":
            bundle["model"] = ensemble
        joblib.dump(bundle, tmp.name)
        artifact_bytes = os.path.getsize(tmp.name)

    return {
        "arm": "M5_Adaptive_Scientific",
        "trials": trial_results,
        "apex_trial_id": best_trial_id,
        "apex_predict_fn": full_pipeline_predict,
        "val_balanced_accuracy": best_val_score,
        "p95_inference_latency_ms": p95_ms,
        "artifact_bytes": artifact_bytes,
        "search_wall_ms": total_wall_ms,
        "provider_spend_usd": 0.00,
    }


def evaluate_sealed_test(arm_name: str, predict_fn, test_df: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Exploratory holdout score; this function is NOT a durable sealed-test gate."""
    y_true = test_df[target_col].to_numpy()
    preds = predict_fn(test_df)
    if isinstance(preds, dict):
        pred_labels = preds.get("prediction", preds.get("predictions"))
    elif hasattr(preds, "ndim") and preds.ndim == 2:
        pred_labels = (preds[:, 1] >= 0.5).astype(int)
    else:
        pred_labels = np.asarray(preds)

    bal_acc = float(balanced_accuracy_score(y_true, pred_labels))
    return {
        "sealed_test_balanced_accuracy": bal_acc,
    }


def main():
    print("=== Exploratory Unequal-Compute Comparison: M3 vs M5 ===")
    df = generate_benchmark_dataset(seed=2026, n_samples=300)
    features = ["num1", "num2", "cat1", "cat2"]
    target_col = "target"

    # Splits: 60% Train (180), 20% Val (60), 20% Sealed Test (60)
    n = len(df)
    train_end = int(0.60 * n)
    val_end = int(0.80 * n)

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    print(f"Data partitioning: Train={len(train_df)}, Validation={len(val_df)}, Sealed Test={len(test_df)}")

    # Run M3 Arm
    print("\nExecuting Arm 1 (M3 Baseline Search - 5 matched trials)...")
    m3_res = run_m3_arm(train_df, val_df, features, target_col, seed=2026)
    print(f"  M3 Apex: {m3_res['apex_trial_id']}, Val Balanced Accuracy: {m3_res['val_balanced_accuracy']:.4f}")
    print(f"  M3 P95 Latency: {m3_res['p95_inference_latency_ms']:.2f} ms, Bundle Size: {m3_res['artifact_bytes']} bytes")
    print(f"  M3 Wall Time: {m3_res['search_wall_ms']:.1f} ms, Provider Cost: ${m3_res['provider_spend_usd']:.2f}")

    # Run M5 Arm
    print("\nExecuting Arm 2 (M5 Adaptive Scientific Pipeline - 5 matched trials)...")
    m5_res = run_m5_arm(train_df, val_df, features, target_col, seed=2026)
    print(f"  M5 Apex: {m5_res['apex_trial_id']}, Val Balanced Accuracy: {m5_res['val_balanced_accuracy']:.4f}")
    print(f"  M5 P95 Latency: {m5_res['p95_inference_latency_ms']:.2f} ms, Bundle Size: {m5_res['artifact_bytes']} bytes")
    print(f"  M5 Wall Time: {m5_res['search_wall_ms']:.1f} ms, Provider Cost: ${m5_res['provider_spend_usd']:.2f}")

    # Exploratory holdout evaluation (no durable sealed-test gate)
    print("\nEvaluating exploratory holdout (not acceptance evidence)...")
    m3_test = evaluate_sealed_test("M3", lambda df: m3_res["apex_model"].predict(df[features]), test_df, target_col)
    m5_test = evaluate_sealed_test("M5", m5_res["apex_predict_fn"], test_df, target_col)

    print(f"  M3 Sealed Test Balanced Accuracy: {m3_test['sealed_test_balanced_accuracy']:.4f}")
    print(f"  M5 Sealed Test Balanced Accuracy: {m5_test['sealed_test_balanced_accuracy']:.4f}")

    # Report Summary Comparison
    diff = m5_test["sealed_test_balanced_accuracy"] - m3_test["sealed_test_balanced_accuracy"]
    summary = {
        "status": "exploratory_only",
        "equal_compute": False,
        "durable_sealed_test_gate": False,
        "m3_baseline": {
            "apex_trial": m3_res["apex_trial_id"],
            "val_balanced_accuracy": m3_res["val_balanced_accuracy"],
            "sealed_test_balanced_accuracy": m3_test["sealed_test_balanced_accuracy"],
            "p95_latency_ms": m3_res["p95_inference_latency_ms"],
            "bundle_bytes": m3_res["artifact_bytes"],
            "wall_time_ms": m3_res["search_wall_ms"],
            "provider_cost_usd": m3_res["provider_spend_usd"],
            "trials": m3_res["trials"],
        },
        "m5_scientific": {
            "apex_trial": m5_res["apex_trial_id"],
            "val_balanced_accuracy": m5_res["val_balanced_accuracy"],
            "sealed_test_balanced_accuracy": m5_test["sealed_test_balanced_accuracy"],
            "p95_latency_ms": m5_res["p95_inference_latency_ms"],
            "bundle_bytes": m5_res["artifact_bytes"],
            "wall_time_ms": m5_res["search_wall_ms"],
            "provider_cost_usd": m5_res["provider_spend_usd"],
            "trials": m5_res["trials"],
        },
        "comparison": {
            "test_accuracy_delta": float(diff),
            "outcome": "m5_superior" if diff > 0.005 else ("m3_superior" if diff < -0.005 else "tied_within_tolerance"),
        },
    }

    report_path = repo_root / "docs" / "m5-comparison-results.json"
    report_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary results saved to {report_path}")
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
