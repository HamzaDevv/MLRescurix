"""Engineering Test Fixtures for Custom Architecture & Adapter Independence.

NOTE: These datasets are strictly synthetic test fixtures for validating architecture
decoupling, memory boundaries, and tuning pipelines. They DO NOT represent the user's
five future benchmark problems and make zero domain quality claims.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd


def fixture_multimodal_numeric_categorical(
    n_samples: int = 120, seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:
    """Synthetic engineering fixture with mixed numeric and categorical features."""
    rng = np.random.default_rng(seed)
    categories = rng.choice(["alpha", "beta", "gamma"], size=n_samples)
    regions = rng.choice(["north", "south", "east", "west"], size=n_samples)

    num_1 = rng.normal(0.0, 1.0, size=n_samples)
    num_2 = rng.exponential(1.5, size=n_samples)
    noise = rng.normal(0.0, 0.2, size=n_samples)

    cat_effect = np.where(categories == "alpha", 1.2, np.where(categories == "beta", -0.8, 0.2))
    reg_effect = np.where(regions == "north", 0.5, -0.5)

    logits = 1.5 * num_1 - 0.8 * num_2 + cat_effect + reg_effect + noise
    target = (logits > 0.0).astype(int)

    df = pd.DataFrame({
        "num_feature_1": num_1,
        "num_feature_2": num_2,
        "category_tag": categories,
        "region_tag": regions,
        "target": target,
    })
    return df, "target", "binary_classification", "balanced_accuracy", "maximize"


def fixture_wide_sparse_tabular(
    n_samples: int = 100, n_features: int = 40, seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:
    """Synthetic wide tabular engineering fixture for regression and memory checks."""
    rng = np.random.default_rng(seed)
    # Generate sparse matrix with ~70% zeros
    dense = rng.normal(0.0, 1.0, size=(n_samples, n_features))
    mask = rng.binomial(1, 0.3, size=(n_samples, n_features))
    sparse_data = dense * mask

    weights = rng.normal(0.0, 0.5, size=n_features)
    y = np.dot(sparse_data, weights) + rng.normal(0.0, 0.1, size=n_samples)

    col_names = [f"feat_{i:03d}" for i in range(n_features)]
    df = pd.DataFrame(sparse_data, columns=col_names)
    df["continuous_target"] = y

    return df, "continuous_target", "regression", "rmse", "minimize"


def fixture_sequential_temporal(
    n_steps: int = 150, seed: int = 42
) -> Tuple[pd.DataFrame, str, str, str, str]:
    """Synthetic sequential temporal engineering fixture with step index and lag features."""
    rng = np.random.default_rng(seed)
    time_steps = np.arange(n_steps)
    trend = 0.05 * time_steps
    seasonality = np.sin(2 * np.pi * time_steps / 12) * 2.0
    noise = rng.normal(0.0, 0.5, size=n_steps)
    series = trend + seasonality + noise

    df = pd.DataFrame({
        "step_index": time_steps,
        "lag_1": np.roll(series, 1),
        "lag_2": np.roll(series, 2),
        "lag_3": np.roll(series, 3),
        "trend_feature": trend,
        "temporal_target": series,
    })
    # Drop rolled boundary artifacts
    df = df.iloc[3:].reset_index(drop=True)

    return df, "temporal_target", "regression", "mae", "minimize"
