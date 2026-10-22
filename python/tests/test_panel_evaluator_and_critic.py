"""Tests for Panel Forecasting Evaluator and Measured Critic Feature Dominance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.candidate_pipeline import (
    CandidatePipeline,
    evaluate_candidate,
    extract_measured_importances,
    extract_split_importances,
)


def test_measured_feature_dominance_vs_unsupported():
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "num2": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    folds = [
        ([0, 1, 2, 3, 4, 5, 6, 7], [8, 9]),
        ([2, 3, 4, 5, 6, 7, 8, 9], [0, 1]),
    ]

    # 1. Linear model has coef_ -> measured importances
    spec_linear = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "seed": 42,
    }
    res_linear = evaluate_candidate(spec_linear, df, "target", "binary_classification", folds, "accuracy")
    assert res_linear["feature_dominance_status"] == "measured"
    assert res_linear["feature_importances"] is not None
    assert len(res_linear["feature_importances"]) == 3
    assert res_linear["dominant_feature"] in ["num1", "num2", "cat1"]
    assert 0.0 < res_linear["dominant_gain_ratio"] <= 1.0

    # 2. HistGradientBoosting has no native feature_importances_ -> explicit unsupported gate
    spec_hist = {
        "families": ["hist_gbdt"],
        "weights": [1.0],
        "seed": 42,
    }
    res_hist = evaluate_candidate(spec_hist, df, "target", "binary_classification", folds, "accuracy")
    assert res_hist["feature_dominance_status"] == "unsupported_by_estimator"
    assert res_hist["feature_importances"] is None
    assert res_hist["dominant_feature"] is None
    assert res_hist["dominant_gain_ratio"] is None
    assert any("Feature gain audit unsupported by estimator" in lim for lim in res_hist["limitations"])


def test_panel_forecasting_evaluator_clean_and_leakage():
    dates = pd.date_range("2017-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_nbr": [1] * 15 + [2] * 15,
        "family": ["GROCERY"] * 30,
        "sales": np.abs(np.sin(np.arange(30))) * 100,
        "onpromotion": [0, 1] * 15,
    })

    # Pinned expanding origin folds
    folds = [
        # Fold 0: train days 0..19, val days 20..24
        (list(range(20)), list(range(20, 25))),
        # Fold 1: train days 0..24, val days 25..29
        (list(range(25)), list(range(25, 30))),
    ]

    spec = {
        "families": ["penalized_linear"],
        "weights": [1.0],
        "seed": 42,
        "target_transform": "log1p",
    }
    topology = {
        "benchmark_id": "K5",
        "temporal_col": "date",
        "entity_columns": ["store_nbr", "family"],
        "forecast_horizon_dates": 5,
        "known_at_forecast_columns": ["onpromotion"],
    }

    # Clean run
    res = evaluate_candidate(spec, df, "sales", "panel_forecasting", folds, "rmsle", topology=topology)
    assert res["is_panel_forecasting"] is True
    assert len(res["fold_receipts"]) == 2
    for r in res["fold_receipts"]:
        assert "train_end" in r
        assert "val_start" in r
        assert "val_end" in r
        assert r["train_end"] < r["val_start"]
        assert r["horizon_steps"] == 5

    # Leakage run (future lookahead: train overlaps validation)
    leakage_folds = [
        (list(range(22)), list(range(20, 25))),  # train includes day 20 and 21, while val starts at 20
    ]
    with pytest.raises(ValueError, match="Temporal lookahead leakage detected"):
        evaluate_candidate(spec, df, "sales", "panel_forecasting", leakage_folds, "rmsle", topology=topology)


def test_panel_evaluator_requires_allowlist_and_exact_date_horizon():
    dates = pd.date_range("2025-01-01", periods=8, freq="D")
    df = pd.DataFrame({"date": dates, "store": 1, "sales": np.arange(8.0), "future_proxy": np.arange(8.0)})
    folds = [(list(range(4)), list(range(4, 8)))]
    spec = {"families": ["penalized_linear"], "weights": [1.0]}
    topology = {"temporal_col": "date", "forecast_horizon_dates": 4, "known_at_forecast_columns": ["store"]}
    with pytest.raises(ValueError, match="absent from known-at-forecast allowlist"):
        evaluate_candidate(spec, df, "sales", "panel_forecasting", folds, "rmsle", topology)

    safe = df.drop(columns="future_proxy")
    with pytest.raises(ValueError, match="has 3 validation dates; expected horizon 4"):
        evaluate_candidate(spec, safe, "sales", "panel_forecasting", [(list(range(4)), list(range(4, 7)))], "rmsle", topology)


def test_feature_names_follow_matrix_order_and_split_gains_are_native_only():
    df = pd.DataFrame({
        "cat_first": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        "numeric_middle": np.arange(10.0),
        "cat_last": ["x", "x", "y", "y", "x", "x", "y", "y", "x", "y"],
        "target": [0, 1] * 5,
    })
    folds = [(list(range(6)), list(range(6, 10))), (list(range(4, 10)), list(range(0, 4)))]
    spec = {"families": ["penalized_linear"], "weights": [1.0]}
    result = evaluate_candidate(spec, df, "target", "binary_classification", folds, "accuracy")
    assert result["feature_names"] == ["numeric_middle", "cat_first", "cat_last"]
    assert result["feature_importances"] is not None
    assert result["split_gain_importances"] is None
    assert result["split_gain_dominant_feature"] is None
    fitted = CandidatePipeline(spec, "target", "binary_classification").fit(df)
    assert extract_split_importances(fitted, 3) is None
    from mlrescurix_worker.scientific.model_zoo import LightweightResidualMLPRegressor
    neural = LightweightResidualMLPRegressor()
    neural.feature_importances_ = np.array([0.8, 0.1, 0.1])
    assert extract_split_importances(neural, 3) is None


def test_panel_loop_rejects_random_folds(tmp_path):
    from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop

    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "sales": np.arange(10.0)})
    data = tmp_path / "panel.csv"
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="caller-pinned temporal folds"):
        AutonomousAgentLoop(
            tmp_path / "workspace", "p", "s", object(), data, "sales",
            task_type="panel_forecasting", metric_name="rmsle", metric_direction="minimize",
        )


def test_spec_resolution_never_substitutes_models_or_weights():
    from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop

    loop = object.__new__(AutonomousAgentLoop)
    loop.state = {"active_spec": {"families": ["hist_gbdt"], "weights": [1.0], "transforms": []}}
    exact = loop._spec({"families": ["lgbm", "xgboost"], "weights": [2.0, 3.0]})
    assert exact["families"] == ["lgbm", "xgboost"]
    assert exact["weights"] == [2.0, 3.0]
    for params, reason in [
        ({"families": ["random_forest", "lgbm"], "weights": [1, 1]}, "Unsupported model family"),
        ({"families": ["lgbm", "xgboost"], "weights": [9]}, "Weights must match"),
        ({"families": ["lgbm"], "model_parameters": {"xgboost": {"max_depth": 3}}}, "model_parameters may only name"),
    ]:
        with pytest.raises(ValueError, match=reason):
            loop._spec(params)


def test_log_price_metric_preserves_subunit_predictions():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    df = pd.DataFrame({"date": dates, "x": np.arange(12.0), "price": np.full(12, 0.2)})
    folds = [(list(range(8)), list(range(8, 12)))]
    spec = {"families": ["penalized_linear"], "weights": [1.0], "target_transform": "log"}
    result = evaluate_candidate(spec, df, "price", "regression", folds, "rmse_log_price")
    assert result["fold_receipts"][0]["validation_score"] < 0.01
