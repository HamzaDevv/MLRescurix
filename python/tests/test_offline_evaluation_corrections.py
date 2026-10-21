import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.model_zoo import (
    PredefinedSplitter,
    evaluate_model_oof,
    train_and_diagnose_portfolio,
)
from mlrescurix_worker.scientific.pipeline import (
    FeatureEngineeringGraphTransformer,
    FullInferencePipeline,
    ParetoEnsemblePredictor,
)
from mlrescurix_worker.scientific.autonomous_rescue import AutonomousRescueEngine


def test_locked_fold_consumption_with_index_mapping(tmp_path: Path):
    """Verify that development folds from a split manifest can be mapped from original
    indices to development row indices and consumed by PredefinedSplitter without out-of-bounds error."""
    # Manifest where total rows = 100, dev = 80 (indices 10..89), holdout = 20 (0..9, 90..99)
    total_rows = 100
    dev_indices = list(range(10, 90))  # 80 rows
    holdout_indices = list(range(0, 10)) + list(range(90, 100))
    
    # 2 folds partitioning the original dev indices
    f0_train = dev_indices[:40]
    f0_val = dev_indices[40:]
    f1_train = dev_indices[40:]
    f1_val = dev_indices[:40]
    
    dev_folds = [
        {"fold": 0, "train_indices": f0_train, "val_indices": f0_val},
        {"fold": 1, "train_indices": f1_train, "val_indices": f1_val},
    ]
    
    # Full dataframe
    df_full = pd.DataFrame({
        "feature_1": np.random.randn(total_rows),
        "target": np.random.randn(total_rows),
    })
    
    # Sliced development partition (reset index to 0..79)
    df_dev = df_full.iloc[dev_indices].copy().reset_index(drop=True)
    assert len(df_dev) == 80
    
    # Explicit mapping
    dev_index_map = {orig: dev for dev, orig in enumerate(dev_indices)}
    mapped_folds = [
        (
            [dev_index_map[i] for i in fold["train_indices"]],
            [dev_index_map[i] for i in fold["val_indices"]],
        )
        for fold in dev_folds
    ]
    
    splitter = PredefinedSplitter(mapped_folds)
    assert splitter.get_n_splits() == 2
    
    splits = list(splitter.split(df_dev))
    assert len(splits) == 2
    tr0, val0 = splits[0]
    assert len(tr0) == 40 and len(val0) == 40
    assert set(tr0).isdisjoint(set(val0))
    assert max(tr0) < 80 and max(val0) < 80
    assert min(tr0) >= 0 and min(val0) >= 0


def test_fold_local_learned_preprocessing_no_leakage():
    """Verify that when a preprocessor is passed to evaluate_model_oof, it is fit fold-locally,
    ensuring that validation statistics do not leak into training."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "cat1": ["A" if i % 2 == 0 else "B" for i in range(n)],
        "target": np.random.randn(n),
    })
    
    col_classes = {"num1": "NUMERIC_CONTINUOUS", "cat1": "SAFE_CATEGORICAL", "target": "TARGET"}
    fe = FeatureEngineeringGraphTransformer(column_classifications=col_classes, task_type="regression")
    
    splits = [
        (list(range(0, 50)), list(range(50, 100))),
        (list(range(50, 100)), list(range(0, 50))),
    ]
    
    res = evaluate_model_oof(
        estimator_or_family="penalized_linear",
        X=df[["num1", "cat1"]],
        y=df["target"],
        cv_splitter=splits,
        task_type="regression",
        metric_name="rmse",
        metric_direction="minimize",
        preprocessor=fe,
    )
    
    assert res.mean_cv_score > 0.0
    assert len(res.fitted_estimators) == 2
    assert fe.is_fitted_ is False  # Original preprocessor remains unfitted; clones were fitted in-fold!


def test_metric_direction_and_gap_orientation():
    """Verify that RMSE on log target evaluates with direction 'minimize' and computes train_val_gap as (val - train)."""
    np.random.seed(42)
    n = 80
    X = np.random.randn(n, 5)
    y = np.exp(np.random.randn(n) * 0.5 + 10.0)  # Exponential prices
    
    splits = [(list(range(0, 40)), list(range(40, 80)))]
    
    res = evaluate_model_oof(
        estimator_or_family="penalized_linear",
        X=X,
        y=np.log1p(y),
        cv_splitter=splits,
        task_type="regression",
        metric_name="rmse_log_price",
        metric_direction="minimize",
        log_target=True,
    )
    
    assert res.mean_cv_score > 0.0
    # Overfitting gap in error minimization is (val_error - train_error)
    assert np.isclose(res.train_val_gap, res.mean_cv_score - res.mean_train_score)


def test_full_inference_pipeline_log_target_parity():
    """Verify FullInferencePipeline with log_target=True fits on log scale but predicts on original price scale."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": np.random.uniform(10, 50, n),
        "feature_b": np.random.uniform(1, 5, n),
        "target": np.random.uniform(100000, 300000, n),  # House prices in dollars
    })
    
    pipe = FullInferencePipeline(
        column_classifications={"feature_a": "NUMERIC_CONTINUOUS", "feature_b": "NUMERIC_CONTINUOUS", "target": "TARGET"},
        task_type="regression",
        log_target=True,
    )
    
    pipe.fit(df[["feature_a", "feature_b"]], df["target"])
    assert pipe.is_fitted_ is True
    
    preds_dollar = pipe.predict(df[["feature_a", "feature_b"]])
    assert len(preds_dollar) == n
    # Predictions must be in original dollar scale (> 50,000), not in log scale (~12)
    assert np.all(preds_dollar > 10000.0)
    assert np.median(preds_dollar) > 50000.0
    
    # predict_log returns log scale
    preds_log = pipe.predict_log(df[["feature_a", "feature_b"]])
    assert np.all(preds_log < 20.0)
    assert np.all(preds_log > 8.0)
