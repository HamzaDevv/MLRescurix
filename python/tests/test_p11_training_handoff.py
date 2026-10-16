"""Unit and integration tests for Phase P11: Advanced Model Training & Apex Handoff Engine."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from mlrescurix_worker.scientific.calibration import (
    SplitAwareCalibrator,
    calibrate_probabilities,
    compute_brier_score,
    compute_reliability_curve,
    optimize_decision_threshold,
)
from mlrescurix_worker.scientific.features import (
    CyclicalDatetimeEncoder,
    FrequencyEncoder,
    NumericFeatureCrosses,
    OutOfFoldTargetEncoder,
)
from mlrescurix_worker.scientific.candidate_synthesis import (
    CandidateSynthesizer,
)
from mlrescurix_worker.scientific.pareto_ensemble import (
    ParetoEnsembleSpec,
    ParetoEnsembleSynthesizer,
    optimize_bayesian_blend_weights,
)
from mlrescurix_worker.scientific.autonomous_rescue import bundle_candidate
from mlrescurix_worker.scientific.bundle import verify_bundle


@pytest.fixture
def candidate_bundle_schema():
    schema_path = Path("contracts/v2/candidate_bundle.schema.json")
    assert schema_path.exists(), f"Schema missing: {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


@pytest.fixture
def candidate_synthesis_schema():
    schema_path = Path("contracts/v2/candidate_synthesis.schema.json")
    assert schema_path.exists(), f"Schema missing: {schema_path}"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return schema


# ==============================================================================
# 1. Feature Engineering & Regularized Encoding Primitives (Deliverable 2)
# ==============================================================================

def test_out_of_fold_target_encoder_zero_leakage():
    """Verify OOF target encoder prevents label leakage and handles unseen categories."""
    np.random.seed(2026)
    n = 200
    cats = np.random.choice(["cat_a", "cat_b", "cat_c"], size=n)
    # Target correlated with categories
    y = np.where(cats == "cat_a", 1, np.where(cats == "cat_b", 0, np.random.binomial(1, 0.5, size=n)))
    df = pd.DataFrame({"category": cats, "num": np.random.randn(n)})

    encoder = OutOfFoldTargetEncoder(categorical_columns=["category"], smoothing=10.0, n_splits=5, seed=2026)
    transformed_train = encoder.fit_transform(df, y)

    # 1. Verify transformed column exists
    assert "category_te" in transformed_train.columns
    assert transformed_train["category_te"].dtype == np.float64
    assert not transformed_train["category_te"].isna().any()

    # 2. Verify zero label leakage on test set (test labels are never passed)
    test_df = pd.DataFrame({
        "category": ["cat_a", "cat_b", "cat_c", "unseen_cat"],
        "num": [0.1, 0.2, 0.3, 0.4],
    })
    transformed_test = encoder.transform(test_df)
    assert len(transformed_test) == 4
    # Unseen category must be regularized to prior global mean
    global_mean = float(np.mean(y))
    assert abs(transformed_test["category_te"].iloc[3] - global_mean) < 1e-4

    # Smoothing check: cat_a encoded value should be between category mean and global mean
    cat_a_train_mean = float(np.mean(y[cats == "cat_a"]))
    cat_a_encoded = float(encoder.mapping_["category"]["cat_a"])
    assert min(cat_a_train_mean, global_mean) <= cat_a_encoded <= max(cat_a_train_mean, global_mean)


def test_frequency_encoder():
    """Verify frequency and count encoding on categorical columns."""
    df = pd.DataFrame({"high_card": ["A", "A", "A", "B", "B", "C"]})
    encoder = FrequencyEncoder(categorical_columns=["high_card"])
    encoder.fit(df)
    transformed = encoder.transform(df)

    assert "high_card_freq" in transformed.columns
    assert np.isclose(transformed["high_card_freq"].iloc[0], 3 / 6)
    assert np.isclose(transformed["high_card_freq"].iloc[3], 2 / 6)
    assert np.isclose(transformed["high_card_freq"].iloc[5], 1 / 6)

    # Unseen category defaults to 1/N
    test_df = pd.DataFrame({"high_card": ["Z"]})
    test_transformed = encoder.transform(test_df)
    assert test_transformed["high_card_freq"].iloc[0] == 1.0 / 6


def test_numeric_feature_crosses_numerical_stability():
    """Verify numeric feature crosses with zero-division epsilon protection."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 0.0, -4.0],
        "num2": [2.0, 0.0, 0.0, 2.0],
    })
    crosser = NumericFeatureCrosses(
        numeric_columns=["num1", "num2"],
        include_ratios=True,
        include_differences=True,
        include_polynomials=True,
    )
    crosser.fit(df)
    transformed = crosser.transform(df)

    assert "num1_sq" in transformed.columns
    assert "num1_minus_num2" in transformed.columns
    assert "num1_div_num2" in transformed.columns

    # Check zero division stability (no inf, no nan)
    assert np.all(np.isfinite(transformed["num1_div_num2"]))
    assert np.all(np.isfinite(transformed["num1_minus_num2"]))


def test_cyclical_datetime_encoder():
    """Verify sine/cosine periodic datetime encoding."""
    dates = pd.date_range("2026-01-01", periods=24, freq="h")
    df = pd.DataFrame({"timestamp": dates})

    cyclical = CyclicalDatetimeEncoder(datetime_columns=["timestamp"])
    cyclical.fit(df)
    transformed = cyclical.transform(df)

    assert "timestamp_hour_sin" in transformed.columns
    assert "timestamp_hour_cos" in transformed.columns

    # Hour 0 should have sin(0) = 0 and cos(0) = 1
    assert np.isclose(transformed["timestamp_hour_sin"].iloc[0], 0.0, atol=1e-5)
    assert np.isclose(transformed["timestamp_hour_cos"].iloc[0], 1.0, atol=1e-5)

    # All sin and cos values must be in [-1, 1]
    assert np.all(transformed["timestamp_hour_sin"].between(-1.0, 1.0))
    assert np.all(transformed["timestamp_hour_cos"].between(-1.0, 1.0))


# ==============================================================================
# 2. Probability Calibration & Threshold Optimization (Deliverable 3)
# ==============================================================================

def test_probability_calibration_monotonicity_and_brier_reduction():
    """Verify Platt and Isotonic calibration maintain monotonicity and reduce Brier score."""
    np.random.seed(2026)
    n = 500
    true_logits = np.random.randn(n)
    y_true = (true_logits > 0).astype(int)

    # Intentionally miscalibrated (underconfident) probabilities
    p_uncal = 1.0 / (1.0 + np.exp(-0.3 * true_logits))
    probs = np.column_stack([1.0 - p_uncal, p_uncal])

    # 1. Isotonic calibration
    iso_res = calibrate_probabilities(y_true, probs, method="isotonic", classes=[0, 1])
    assert iso_res["calibration_method"] == "isotonic"
    assert iso_res["brier_delta"] >= 0.0  # Brier score must improve or stay same
    cal_probs = iso_res["calibrated_probabilities"]
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    assert np.allclose(cal_probs.sum(axis=1), 1.0)

    # 2. Platt / Sigmoid calibration
    platt_res = calibrate_probabilities(y_true, probs, method="platt", classes=[0, 1])
    assert platt_res["calibration_method"] == "platt"
    assert platt_res["brier_delta"] >= 0.0

    # 3. Reliability curve structure
    rel_curve = iso_res["reliability_curve_after"]
    assert "bin_edges" in rel_curve
    assert "true_proportions" in rel_curve
    assert "pred_proportions" in rel_curve
    assert len(rel_curve["bin_counts"]) == 10


def test_metric_driven_threshold_optimization():
    """Verify decision threshold search maximizing F1 or minimizing cost."""
    np.random.seed(2026)
    n = 1000
    y_true = np.random.binomial(1, 0.10, size=n)
    probs = np.clip(y_true * 0.35 + np.random.uniform(0.1, 0.6, size=n), 0.01, 0.99)

    f1_opt = optimize_decision_threshold(y_true, probs, metric="f1")
    assert "optimal_threshold" in f1_opt
    assert 0.01 <= f1_opt["optimal_threshold"] <= 0.99
    assert f1_opt["best_score"] >= f1_opt["default_score"]

    cost_matrix = {"c_fp": 1.0, "c_fn": 50.0, "c_tp": 0.0, "c_tn": 0.0}
    cost_opt = optimize_decision_threshold(y_true, probs, metric="cost", cost_matrix=cost_matrix)
    assert cost_opt["optimal_threshold"] < 0.5
    assert cost_opt["best_score"] < cost_opt["default_score"]


# ==============================================================================
# 3. Hybrid Multimodal Column Training & PEFT LoRA (Deliverable 4)
# ==============================================================================

def test_hybrid_peft_lora_synthesis_and_memory_estimation(tmp_path):
    """Verify hybrid model code synthesis with frozen backbone and LoRA adapter."""
    cand_dir = tmp_path / "cand_hybrid_lora"
    spec = {
        "schema_version": "v2",
        "synthesis_id": "synth-p11-hybrid",
        "candidate_id": "cand_hybrid_lora",
        "name": "Candidate Hybrid PEFT LoRA",
        "problem_definition": {
            "task_type": "binary",
            "target_column": "target",
            "dataset_id": "ds_hybrid",
            "id_column": "row_id",
        },
        "target_descriptor": {
            "name": "target",
            "task_type": "binary",
        },
        "metric_objective": {
            "primary_metric": "balanced_accuracy",
            "direction": "maximize",
        },
        "model_intent": {
            "family": "hybrid",
            "framework": "peft_lora",
            "hyperparameters": {
                "lora_rank": 4,
                "lora_alpha": 8.0,
                "num_classes": 2,
            },
            "feature_pipeline": {
                "categorical_encoding": "target_oof",
                "target_encoding_smoothing": 10.0,
                "frequency_encoding": True,
                "numeric_feature_crosses": True,
                "cyclical_datetime": True,
            },
        },
    }

    result = CandidateSynthesizer.synthesize(spec, cand_dir)
    assert result.candidate_id == "cand_hybrid_lora"
    assert (cand_dir / "model.py").exists()
    assert (cand_dir / "features.py").exists()

    model_code = (cand_dir / "model.py").read_text()
    assert "LoRALinear" in model_code
    assert "frozen_ratio" in model_code
    assert "0.95" in model_code  # Asserts >95% frozen backbone ratio

    # Verify features.py contains the P11 regularized encoders
    features_code = (cand_dir / "features.py").read_text()
    assert "target_encoding_map" in features_code
    assert "freq_map" in features_code
    assert "cross_specs" in features_code
    assert "2.0 * np.pi" in features_code

    # Verify analytical memory estimation
    mem_est = CandidateSynthesizer.estimate_peft_memory_mb(spec)
    assert mem_est["frozen_ratio"] >= 0.95
    assert mem_est["estimated_memory_mb"] > 0
    assert mem_est["feasible_under_512mb"] is True


# ==============================================================================
# 4. Multi-Model Pareto Stacking & Blending Engine (Deliverable 1)
# ==============================================================================

def test_bayesian_blend_weight_optimization():
    """Verify SLSQP simplex optimization for Bayesian ensemble weights."""
    np.random.seed(2026)
    n = 200
    y_true = np.random.binomial(1, 0.5, size=n)

    # Sub-model 1: accurate (AUC ~0.9)
    p1 = np.clip(y_true * 0.8 + np.random.uniform(0.0, 0.3, size=n), 0.01, 0.99)
    # Sub-model 2: noisy (AUC ~0.55)
    p2 = np.clip(0.5 + np.random.randn(n) * 0.2, 0.01, 0.99)

    oof_preds = {
        "cand_strong": np.column_stack([1.0 - p1, p1]),
        "cand_weak": np.column_stack([1.0 - p2, p2]),
    }

    weights = optimize_bayesian_blend_weights(oof_preds, y_true, task_type="binary_classification")
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-5)
    assert all(w >= 0.0 for w in weights.values())
    # Strong sub-model should receive higher weight than weak sub-model
    assert weights["cand_strong"] > weights["cand_weak"]


def test_pareto_ensemble_synthesis(tmp_path):
    """Verify ParetoEnsembleSynthesizer generates standalone entrypoint and manifest."""
    np.random.seed(2026)
    n = 100
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    y = np.random.binomial(1, 0.5, size=n)
    df = pd.DataFrame({
        "row_id": [f"r{i}" for i in range(n)],
        "f1": x1,
        "f2": x2,
        "target": y,
    })
    train_csv = tmp_path / "train.csv"
    df.to_csv(train_csv, index=False)

    cand1_dir = tmp_path / "cand1"
    cand2_dir = tmp_path / "cand2"

    spec_cand1 = {
        "schema_version": "v2",
        "synthesis_id": "synth_1",
        "candidate_id": "cand_1",
        "name": "Candidate 1",
        "problem_definition": {"task_type": "binary", "target_column": "target", "id_column": "row_id"},
        "metric_objective": {"primary_metric": "balanced_accuracy", "direction": "maximize"},
        "model_intent": {"family": "linear", "framework": "scikit-learn"},
    }
    spec_cand2 = {
        "schema_version": "v2",
        "synthesis_id": "synth_2",
        "candidate_id": "cand_2",
        "name": "Candidate 2",
        "problem_definition": {"task_type": "binary", "target_column": "target", "id_column": "row_id"},
        "metric_objective": {"primary_metric": "balanced_accuracy", "direction": "maximize"},
        "model_intent": {"family": "boosting", "framework": "lightgbm"},
    }
    CandidateSynthesizer.synthesize(spec_cand1, cand1_dir)
    CandidateSynthesizer.synthesize(spec_cand2, cand2_dir)

    spec = ParetoEnsembleSpec(
        ensemble_id="pareto_stack_v1",
        task_type="binary",
        submodel_dirs=[str(cand1_dir), str(cand2_dir)],
        meta_learner="ridge",
        primary_metric="balanced_accuracy",
        direction="maximize",
        target_column="target",
        id_column="row_id",
        class_order=[0, 1],
        n_splits=3,
        seed=2026,
    )

    ens_dir = tmp_path / "ensemble_out"
    res = ParetoEnsembleSynthesizer.synthesize_ensemble(spec, ens_dir, train_csv)

    assert (ens_dir / "entrypoint.py").exists()
    assert (ens_dir / "manifest.json").exists()
    assert res.meta_learner_type == "ridge"
    assert len(res.submodel_ids) == 2


# ==============================================================================
# 5. Apex Delivery Bundle Packaging & Verification Gate (Deliverables 3 & 5)
# ==============================================================================

def test_calibrated_candidate_bundle_packaging_and_verification(tmp_path, candidate_bundle_schema):
    """Verify candidate bundling packages calibration metadata, threshold, and model schema."""
    np.random.seed(2026)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    prob = 1.0 / (1.0 + np.exp(-(1.5 * x1 - 0.8 * x2)))
    y = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({"f1": x1, "f2": x2, "target": y})
    data_csv = tmp_path / "data.csv"
    df.to_csv(data_csv, index=False)

    cand_dir = tmp_path / "cand_source"
    cand_dir.mkdir()
    bundle_out = tmp_path / "calibrated_bundle"

    result = bundle_candidate(
        candidate_dir=cand_dir,
        output_bundle_dir=bundle_out,
        dataset_path=data_csv,
        test_access_id="gate_p11_auth",
        project_id="proj_p11_apex",
        candidate_id="cand_apex_calibrated",
        task_type="binary_classification",
        target_column="target",
    )

    assert result["status"] == "bundled"
    assert (bundle_out / "manifest.json").exists()
    assert (bundle_out / "model_schema.json").exists()

    manifest = json.loads((bundle_out / "manifest.json").read_text())
    Draft202012Validator(candidate_bundle_schema).validate(manifest)

    # Assert calibration metadata and threshold are serialized in manifest and model_schema
    assert "decision_threshold" in manifest["model"]
    assert "calibration_method" in manifest["model"]
    assert "model_schema" in manifest["contracts"]

    model_schema = json.loads((bundle_out / "model_schema.json").read_text())
    assert model_schema["format"] == "mlrescurix.schema.v2"
    assert "decision_threshold" in model_schema
    assert "calibration_method" in model_schema

    # Run Python bundle verification without host execution sandbox
    summary = verify_bundle(bundle_out, run_sandbox=False)
    assert summary["status"] == "integrity_verified"
    assert summary["manifest_verified"] is True
    assert summary["constraints_verified"] is True
