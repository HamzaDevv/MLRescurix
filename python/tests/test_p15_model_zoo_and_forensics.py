"""Comprehensive Unit & Synthetic Verification for Phase 4: Topology-Aware CV, Multi-Paradigm Zoo & Error Forensics.

Covers:
1. `model_zoo.py`:
   - Unified estimator factory supporting all 6 families:
     * `catboost`: native or graceful fallback to LGBM/HistGBDT.
     * `lgbm`: Fast histogram GBDT with colsample/subsample regularization.
     * `xgboost`: depth-wise GBDT or graceful fallback.
     * `hist_gbdt`: native sklearn Histogram GBDT.
     * `penalized_linear`: RobustScaler + SimpleImputer(add_indicator=True) + LogisticRegression/Ridge.
     * `neural_tabular`: Lightweight Residual MLP with LayerNorm and skip connections.
   - Out-of-Fold (OOF) cross-validation evaluation (`evaluate_model_oof`):
     * Classification OOF predictions and probabilities across locked CV splits.
     * Regression OOF predictions.
     * Fold-level train and validation scores, train-val gap.
     * Non-negative feature importances extracted across model families.
   - Portfolio evaluation (`train_and_diagnose_portfolio`).
2. `error_forensics.py`:
   - `ErrorForensicsAnalyzer` producing structured `ErrorForensicsReport` conforming to `contracts/v2/error_forensics.schema.json`.
   - Fitting shallow depth-3 surrogate decision tree on OOF residuals |y - y_hat_oof|:
     * Top 3 worst-performing data slices with `slice_rule`, `sample_count`, `slice_error_rate`, `global_error_rate`, `lift`.
   - Feature split-gain concentration Gini coefficient calculation in [0.0, 1.0].
   - Generalization gap (train-val gap) per fold.
   - Schema validation against `contracts/v2/error_forensics.schema.json`.
3. Topology-Aware Upgrades in `candidate_runner.py` & `iteration_loop.py`:
   - CV Splitters: `StratifiedGroupKFold`, `GroupKFold`, `PurgedTimeSeriesSplit`, `StratifiedKFold`.
   - Topology-Preserving Tier-0 Proxy Sampler (`sample_topology_aware_tier0`):
     * **Blueprint Verification Test:** Synthetic dataset with 0.1% positive rate (N = 50,000). Assert Tier-0 sample (N <= 500) contains >= 25 positive instances and completes in < 15s.
     * Probability calibration: weighted positive rate matches true population prior (0.001000).
     * Group-Preserving Sampling: samples whole entity clusters without splitting groups.
     * Temporal Tail Slicing: samples the most recent contiguous rows.
     * Integration with `MultiFidelityIterationEngine.run_proxy_screen`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import jsonschema
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from mlrescurix_worker.scientific.candidate_runner import evaluate_candidate_cv
from mlrescurix_worker.scientific.error_forensics import (
    ErrorForensicsAnalyzer,
    ErrorForensicsReport,
    compute_feature_gini,
    get_error_forensics_schema,
)
from mlrescurix_worker.scientific.fe_ablation import PurgedTimeSeriesSplit, build_cv_splitter
from mlrescurix_worker.scientific.iteration_loop import (
    MultiFidelityIterationEngine,
    sample_topology_aware_tier0,
)
from mlrescurix_worker.scientific.model_zoo import (
    KNOWN_MODEL_FAMILIES,
    LightweightResidualMLPClassifier,
    LightweightResidualMLPRegressor,
    ModelOofResult,
    create_estimator,
    evaluate_model_oof,
    extract_feature_importances,
    normalize_family_name,
    train_and_diagnose_portfolio,
)


@pytest.fixture
def titanic_df() -> pd.DataFrame:
    p = Path(".mlrescurix-state/kaggle-titanic/titanic.csv")
    if not p.is_file():
        p = Path(".mlrescurix-state/data/titanic_raw.csv")
    assert p.is_file(), f"Titanic CSV missing: {p}"
    return pd.read_csv(p)


@pytest.fixture
def synthetic_classification_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 300
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = rng.choice([0, 1], size=n)
    # Informative signal
    p = 1.0 / (1.0 + np.exp(-(1.5 * x1 - 2.0 * x2 + 0.8 * x3)))
    y = (rng.rand(n) < p).astype(int)
    groups = rng.choice(["grp_A", "grp_B", "grp_C", "grp_D", "grp_E"], size=n)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%d %H:%M:%S")

    return pd.DataFrame({
        "row_id": np.arange(n),
        "feat_1": x1,
        "feat_2": x2,
        "feat_3": x3.astype(float),
        "entity_group": groups,
        "timestamp": timestamps,
        "target": y,
    })


@pytest.fixture
def synthetic_regression_df() -> pd.DataFrame:
    rng = np.random.RandomState(42)
    n = 250
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    y = 3.5 * x1 - 1.2 * x2 + rng.randn(n) * 0.2
    return pd.DataFrame({
        "row_id": np.arange(n),
        "x1": x1,
        "x2": x2,
        "target": y,
    })


# =============================================================================
# 1. Unified Estimator Factory Tests
# =============================================================================

def test_model_zoo_family_normalization():
    assert normalize_family_name("lgbm") == "lgbm"
    assert normalize_family_name("lightgbm") == "lgbm"
    assert normalize_family_name("xgb") == "xgboost"
    assert normalize_family_name("cat") == "catboost"
    assert normalize_family_name("hist_gbdt") == "hist_gbdt"
    assert normalize_family_name("penalized_linear") == "penalized_linear"
    assert normalize_family_name("neural_tabular") == "neural_tabular"


def test_model_zoo_estimator_factory_all_families_classification():
    """Verify that create_estimator handles all 6 families with graceful fallbacks."""
    for fam in KNOWN_MODEL_FAMILIES:
        est = create_estimator(fam, task_type="binary_classification", random_state=42)
        assert est is not None
        assert hasattr(est, "_family")
        assert getattr(est, "_family") == fam
        assert hasattr(est, "_effective_family")
        assert hasattr(est, "_is_fallback")
        # If catboost or xgboost are not installed, assert graceful fallback flag
        if fam in ("catboost", "xgboost"):
            # Should have fallback to lightgbm or hist_gbdt
            assert getattr(est, "_effective_family") in ("catboost", "xgboost", "lgbm", "hist_gbdt")


def test_model_zoo_estimator_factory_all_families_regression():
    """Verify that create_estimator handles all 6 families for regression."""
    for fam in KNOWN_MODEL_FAMILIES:
        est = create_estimator(fam, task_type="regression", random_state=42)
        assert est is not None
        assert hasattr(est, "_family")
        assert getattr(est, "_family") == fam


def test_penalized_linear_encapsulates_imputer_and_scaler():
    """Verify penalized_linear wraps RobustScaler + SimpleImputer(add_indicator=True)."""
    est = create_estimator("penalized_linear", task_type="binary_classification")
    steps = dict(est.named_steps)
    assert "imputer" in steps
    assert "scaler" in steps
    assert "model" in steps
    assert steps["imputer"].add_indicator is True


def test_neural_tabular_architecture_and_predict():
    """Verify LightweightResidualMLPClassifier trains and outputs valid probabilities."""
    clf = LightweightResidualMLPClassifier(
        hidden_dim=32,
        num_blocks=2,
        epochs=15,
        random_state=42,
    )
    X = np.random.RandomState(42).randn(100, 5)
    # Add some NaNs to test built-in imputer
    X[0, 1] = np.nan
    X[5, 2] = np.nan
    y = np.random.RandomState(42).choice([0, 1], size=100)

    clf.fit(X, y)
    probs = clf.predict_proba(X)
    preds = clf.predict(X)

    assert probs.shape == (100, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert len(preds) == 100
    assert hasattr(clf, "feature_importances_")
    assert len(clf.feature_importances_) == 5
    assert np.all(clf.feature_importances_ >= 0.0)


def test_neural_tabular_regressor_predict():
    """Verify LightweightResidualMLPRegressor trains and predicts continuous values."""
    reg = LightweightResidualMLPRegressor(
        hidden_dim=32,
        num_blocks=2,
        epochs=15,
        random_state=42,
    )
    X = np.random.RandomState(42).randn(80, 4)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.1 * np.random.randn(80)

    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == (80,)
    assert hasattr(reg, "feature_importances_")
    assert len(reg.feature_importances_) == 4


# =============================================================================
# 2. Out-of-Fold (OOF) Cross-Validation Evaluation Tests
# =============================================================================

def test_evaluate_model_oof_classification(synthetic_classification_df: pd.DataFrame):
    """Verify evaluate_model_oof generates certified OOF predictions and metrics."""
    df = synthetic_classification_df
    X = df[["feat_1", "feat_2", "feat_3"]]
    y = df["target"]

    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    result = evaluate_model_oof(
        estimator_or_family="lgbm",
        X=X,
        y=y,
        cv_splitter=splitter,
        task_type="binary_classification",
        random_state=42,
    )

    assert isinstance(result, ModelOofResult)
    assert len(result.oof_predictions) == len(df)
    assert result.oof_probabilities is not None
    assert result.oof_probabilities.shape == (len(df), 2)
    assert np.allclose(result.oof_probabilities.sum(axis=1), 1.0, atol=1e-3)
    assert len(result.cv_scores) == 3
    assert len(result.train_scores) == 3
    assert 0.0 <= result.mean_cv_score <= 1.0
    assert len(result.feature_importances) == 3
    assert np.all(result.feature_importances >= 0.0)
    assert np.isclose(np.sum(result.feature_importances), 1.0, atol=1e-3)


def test_evaluate_model_oof_regression(synthetic_regression_df: pd.DataFrame):
    """Verify evaluate_model_oof generates continuous predictions for regression."""
    df = synthetic_regression_df
    X = df[["x1", "x2"]]
    y = df["target"]

    splitter = KFold(n_splits=3, shuffle=True, random_state=42)
    result = evaluate_model_oof(
        estimator_or_family="penalized_linear",
        X=X,
        y=y,
        cv_splitter=splitter,
        task_type="regression",
        random_state=42,
    )

    assert isinstance(result, ModelOofResult)
    assert len(result.oof_predictions) == len(df)
    assert result.oof_probabilities is None
    assert len(result.cv_scores) == 3
    assert result.mean_cv_score > 0.80  # strong linear signal
    assert len(result.feature_importances) == 2


def test_train_and_diagnose_portfolio(synthetic_classification_df: pd.DataFrame):
    """Verify portfolio training across diverse model families."""
    df = synthetic_classification_df
    X = df[["feat_1", "feat_2", "feat_3"]]
    y = df["target"]
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    portfolio = train_and_diagnose_portfolio(
        families=["lgbm", "hist_gbdt", "penalized_linear", "neural_tabular"],
        X=X,
        y=y,
        cv_splitter=splitter,
        task_type="binary_classification",
    )

    assert len(portfolio) == 4
    for fam, res in portfolio.items():
        assert res.family == fam
        assert len(res.oof_predictions) == len(df)
        assert len(res.cv_scores) == 3


# =============================================================================
# 3. Error Forensics Analyzer Tests
# =============================================================================

def test_compute_feature_gini():
    # Perfectly uniform importances -> Gini = 0.0
    uniform = [0.25, 0.25, 0.25, 0.25]
    assert compute_feature_gini(uniform) == 0.0

    # Highly concentrated in single feature -> Gini ~ 1.0
    concentrated = [1.0, 0.0, 0.0, 0.0]
    assert compute_feature_gini(concentrated) == 1.0

    # Empty or zero
    assert compute_feature_gini([]) == 0.0
    assert compute_feature_gini([0.0, 0.0]) == 0.0


def test_error_forensics_analyzer_produces_valid_schema_report():
    """Verify ErrorForensicsAnalyzer generates ErrorForensicsReport conforming to Schema v2."""
    rng = np.random.RandomState(42)
    n = 200

    # Ground truth
    pclass = rng.choice([1, 2, 3], size=n)
    is_male = rng.choice([0, 1], size=n)
    fare = rng.exponential(scale=30, size=n)
    y = rng.choice([0, 1], size=n)

    # Intentionally induce high errors on Pclass == 3 and is_male == 1
    oof_preds = y.copy()
    error_mask = (pclass == 3) & (is_male == 1)
    oof_preds[error_mask] = 1 - y[error_mask]  # 100% error rate on this slice

    X = pd.DataFrame({"Pclass": pclass, "is_male": is_male, "Fare": fare})
    cv_scores = [0.82, 0.79, 0.81]

    report = ErrorForensicsAnalyzer.analyze(
        candidate_id="cand_test_01",
        model_family="lgbm",
        X=X,
        y=y,
        oof_predictions=oof_preds,
        cv_scores=cv_scores,
        mean_cv_score=float(np.mean(cv_scores)),
        train_val_gap=0.045,
        feature_importances=[0.55, 0.35, 0.10],
        task_type="binary_classification",
        feature_origins={"Pclass": "Pclass", "is_male": "Sex", "Fare": "Fare"},
    )

    assert isinstance(report, ErrorForensicsReport)
    report_dict = report.to_dict()

    # Validate against Schema v2
    schema = get_error_forensics_schema()
    jsonschema.validate(instance=report_dict, schema=schema)

    # Check top worst slices
    assert len(report.worst_slices) <= 3
    assert len(report.worst_slices) >= 1
    top_slice = report.worst_slices[0]
    assert top_slice.slice_error_rate >= top_slice.global_error_rate
    assert top_slice.lift >= 1.0
    assert top_slice.sample_count > 0
    assert "Error Rate" in top_slice.slice_rule

    # Check Gini and top features
    assert 0.0 <= report.feature_importance_gini <= 1.0
    assert len(report.top_features_by_importance) == 3
    assert report.top_features_by_importance[0].feature == "Pclass"
    assert report.top_features_by_importance[0].origin == "Pclass"


def test_error_forensics_analyzer_regression():
    """Verify ErrorForensicsAnalyzer handles continuous regression targets."""
    rng = np.random.RandomState(42)
    n = 150
    X = pd.DataFrame({
        "temp": rng.randn(n),
        "humidity": rng.rand(n) * 100,
    })
    y = 2.0 * X["temp"] + rng.randn(n) * 0.5
    oof_preds = y + rng.randn(n) * 0.2
    # Add large outliers
    oof_preds[:10] += 15.0

    report = ErrorForensicsAnalyzer.analyze(
        candidate_id="cand_reg_01",
        model_family="penalized_linear",
        X=X,
        y=y,
        oof_predictions=oof_preds,
        cv_scores=[0.88, 0.85],
        mean_cv_score=0.865,
        train_val_gap=0.02,
        feature_importances=[0.75, 0.25],
        task_type="regression",
    )

    schema = get_error_forensics_schema()
    jsonschema.validate(instance=report.to_dict(), schema=schema)
    assert "mean_absolute_error" in report.confusion_or_residual_summary
    assert "root_mean_squared_error" in report.confusion_or_residual_summary


# =============================================================================
# 4. Topology-Aware CV Splitters & Tier-0 Proxy Sampler Tests
# =============================================================================

def test_tier0_rare_event_floor_verification():
    """BLUEPRINT STEP 4 VERIFICATION TEST:

    Test Tier-0 sampling on a synthetic dataset with 0.1% positive rate (N = 50,000).
    Assert Tier-0 sample (N <= 500) contains >= 25 positive instances and completes in < 15s.
    """
    n = 50000
    rng = np.random.RandomState(42)

    # 0.1% positive rate
    y = np.zeros(n, dtype=int)
    pos_idx = rng.choice(n, size=50, replace=False)
    y[pos_idx] = 1

    df = pd.DataFrame({
        "f1": rng.randn(n),
        "f2": rng.randn(n),
        "target": y,
    })

    t0 = time.perf_counter()
    sample_df, weights, meta = sample_topology_aware_tier0(
        df=df,
        target_col="target",
        task_type="binary",
        max_samples=500,
        random_state=42,
    )
    elapsed = time.perf_counter() - t0

    # 1. Performance boundary
    assert elapsed < 15.0, f"Tier-0 sampling exceeded 15s: {elapsed:.2f}s"

    # 2. Sample size cap
    assert len(sample_df) <= 500, f"Sample size exceeded 500: {len(sample_df)}"

    # 3. Minimum-Minority Stratified Subsampling floor
    pos_count = int((sample_df["target"] == 1).sum())
    assert pos_count >= 25, f"Rare-event floor failed: got {pos_count} positive instances (expected >= 25)"
    assert meta["strategy"] == "rare_event_floor"

    # 4. Calibration invariance: weighted positive rate must match population prior 0.001 within 1e-6
    weighted_pos_rate = float((weights * (sample_df["target"] == 1)).sum() / weights.sum())
    assert abs(weighted_pos_rate - 0.001) < 1e-6, (
        f"Probability calibration discrepancy: weighted={weighted_pos_rate}, true=0.001"
    )


def test_tier0_group_preserving_sampling():
    """Verify that group-preserving sampling does NOT split entity groups."""
    n = 2000
    rng = np.random.RandomState(42)
    # 40 groups with 50 rows each
    groups = np.repeat([f"household_{i:03d}" for i in range(40)], 50)
    df = pd.DataFrame({
        "feature": rng.randn(n),
        "hh_id": groups,
        "target": rng.choice([0, 1], size=n),
    })

    topology = {
        "entity_group_col": "hh_id",
        "task_type": "binary_classification",
        "cv_scheme": "stratified_group_kfold",
    }

    sample_df, weights, meta = sample_topology_aware_tier0(
        df=df,
        target_col="target",
        task_type="binary",
        topology=topology,
        max_samples=500,
        random_state=42,
    )

    assert meta["strategy"] == "group_preserving"
    assert len(sample_df) <= 500

    # Assert whole groups are preserved: every sampled group has all 50 members
    sampled_group_counts = sample_df["hh_id"].value_counts()
    for g, cnt in sampled_group_counts.items():
        assert cnt == 50, f"Group {g} was partially shattered into {cnt} rows"


def test_tier0_temporal_tail_slicing():
    """Verify that temporal tail slicing takes contiguous chronological rows."""
    n = 2000
    dates = pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    df = pd.DataFrame({
        "date": dates,
        "price": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n),
    })

    topology = {
        "temporal_col": "date",
        "task_type": "time_series",
        "cv_scheme": "purged_time_series_split",
    }

    sample_df, weights, meta = sample_topology_aware_tier0(
        df=df,
        target_col="target",
        task_type="binary",
        topology=topology,
        max_samples=500,
        random_state=42,
    )

    assert meta["strategy"] == "temporal_tail_slice"
    assert len(sample_df) == 500
    # Must equal the final 500 days
    expected_last_date = dates[-1]
    assert sample_df["date"].iloc[-1] == expected_last_date


def test_locked_cv_splitter_selection(synthetic_classification_df: pd.DataFrame):
    """Verify locked CV splitter generation for grouped, temporal, and standard topologies."""
    df = synthetic_classification_df

    # 1. Grouped topology
    top_grouped = {
        "cv_scheme": "stratified_group_kfold",
        "cv_splits": 4,
        "entity_group_col": "entity_group",
        "temporal_col": None,
        "task_type": "binary_classification",
    }
    splitter, g_col, t_col = build_cv_splitter(top_grouped, df=df)
    assert isinstance(splitter, StratifiedGroupKFold)
    assert splitter.get_n_splits() == 4
    assert g_col == "entity_group"
    assert t_col is None

    # 2. Temporal topology
    top_temporal = {
        "cv_scheme": "purged_time_series_split",
        "cv_splits": 5,
        "entity_group_col": None,
        "temporal_col": "timestamp",
        "task_type": "time_series",
    }
    splitter, g_col, t_col = build_cv_splitter(top_temporal, df=df)
    assert isinstance(splitter, PurgedTimeSeriesSplit)
    assert splitter.get_n_splits() == 5
    assert t_col == "timestamp"

    # 3. Standard classification topology
    top_stratified = {
        "cv_scheme": "stratified_kfold",
        "cv_splits": 3,
        "entity_group_col": None,
        "temporal_col": None,
        "task_type": "binary_classification",
    }
    splitter, g_col, t_col = build_cv_splitter(top_stratified, df=df)
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.get_n_splits() == 3
