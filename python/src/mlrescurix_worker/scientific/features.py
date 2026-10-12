"""Split-aware feature engineering transforms for MLRescurix M5.

All transformations must be fitted strictly on the training partition or inside
training cross-validation folds. Protected validation and sealed test sets must
remain unexposed to fitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, RobustScaler, StandardScaler


@dataclass
class FeatureTransformSpec:
    transform_id: str
    numeric_scaler: str = "robust"  # "robust", "standard", "none"
    interactions: bool = False
    discretize_bins: int = 0  # 0 to disable, or n_bins e.g. 5
    missing_indicators: bool = True
    target_encoding: bool = False
    max_interaction_features: int = 10


class OutOfFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder fitted strictly out-of-fold on training data to prevent leakage."""

    def __init__(self, categorical_columns: list[str], n_splits: int = 5, smoothing: float = 10.0, seed: int = 2026):
        self.categorical_columns = categorical_columns
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.global_mean_: float = 0.0
        self.mapping_: dict[str, dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "OutOfFoldTargetEncoder":
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.nanmean(y_arr))
        self.mapping_ = {}

        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            series = X[col].astype(str).reset_index(drop=True)
            col_map: dict[Any, float] = {}
            for val, group in series.groupby(series):
                idx = group.index
                n = len(idx)
                cat_mean = float(y_arr[idx].mean())
                smoothed = (n * cat_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
                col_map[val] = smoothed
            self.mapping_[col] = col_map
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> pd.DataFrame:
        """Fit with out-of-fold target encoding on training folds to prevent target leakage."""
        y_arr = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.nanmean(y_arr))
        self.mapping_ = {}

        X_out = X.copy()
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for col in self.categorical_columns:
            if col not in X.columns:
                continue
            series = X[col].astype(str).reset_index(drop=True)
            oof_encoded = np.empty(len(X), dtype=float)

            # Compute out-of-fold values
            for train_idx, val_idx in kf.split(X):
                fold_series = series.iloc[train_idx]
                fold_mean = float(y_arr[train_idx].mean())
                oof_encoded[val_idx] = fold_mean

                for val, group in fold_series.groupby(fold_series):
                    idx = group.index
                    n = len(idx)
                    cat_mean = float(y_arr[idx].mean())
                    smoothed = (n * cat_mean + self.smoothing * fold_mean) / (n + self.smoothing)
                    val_mask = (series.iloc[val_idx] == val).to_numpy()
                    oof_encoded[val_idx[val_mask]] = smoothed

            # Fit global mapping for subsequent validation/test transforms
            col_map: dict[Any, float] = {}
            for val, group in series.groupby(series):
                idx = group.index
                n = len(idx)
                cat_mean = float(y_arr[idx].mean())
                smoothed = (n * cat_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
                col_map[val] = smoothed
            self.mapping_[col] = col_map
            X_out[f"{col}_te"] = oof_encoded

        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, col_map in self.mapping_.items():
            if col in X.columns:
                series = X[col].astype(str)
                X_out[f"{col}_te"] = series.map(col_map).fillna(self.global_mean_).to_numpy(dtype=float)
        return X_out


class SplitAwareFeaturePipeline:
    """End-to-end feature engineering pipeline fitted exclusively on training rows."""

    def __init__(self, spec: FeatureTransformSpec, task: str = "binary_classification", seed: int = 2026):
        self.spec = spec
        self.task = task
        self.seed = seed
        self.numeric_cols_: list[str] = []
        self.categorical_cols_: list[str] = []
        self.imputer_: SimpleImputer | None = None
        self.scaler_: RobustScaler | StandardScaler | None = None
        self.discretizer_: KBinsDiscretizer | None = None
        self.indicator_: MissingIndicator | None = None
        self.target_encoder_: OutOfFoldTargetEncoder | None = None
        self.interaction_pairs_: list[tuple[str, str]] = []

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
        categorical_cols: list[str] | None = None,
        numeric_cols: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.fit_transform_training(X_train, y_train)

    def fit_transform_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series | np.ndarray,
    ) -> pd.DataFrame:
        """Fit feature transforms strictly on training data and return transformed training frame."""
        self.numeric_cols_ = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c].dtype)]
        self.categorical_cols_ = [c for c in X_train.columns if c not in self.numeric_cols_]

        out = X_train.copy()

        # 1. Missing indicators
        if self.spec.missing_indicators and self.numeric_cols_:
            self.indicator_ = MissingIndicator(features="missing-only", error_on_new=False)
            ind_arr = self.indicator_.fit_transform(X_train[self.numeric_cols_])
            if ind_arr.shape[1] > 0:
                for idx in range(ind_arr.shape[1]):
                    out[f"num_missing_{idx}"] = ind_arr[:, idx].astype(float)

        # 2. Numeric Imputation and Scaling
        if self.numeric_cols_:
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            imputed_num = self.imputer_.fit_transform(X_train[self.numeric_cols_])

            for idx, col in enumerate(self.numeric_cols_):
                out[col] = imputed_num[:, idx]

            if self.spec.numeric_scaler == "robust":
                self.scaler_ = RobustScaler()
                scaled_num = self.scaler_.fit_transform(imputed_num)
                for idx, col in enumerate(self.numeric_cols_):
                    out[col] = scaled_num[:, idx]
            elif self.spec.numeric_scaler == "standard":
                self.scaler_ = StandardScaler()
                scaled_num = self.scaler_.fit_transform(imputed_num)
                for idx, col in enumerate(self.numeric_cols_):
                    out[col] = scaled_num[:, idx]

        # 3. Discretization of numeric features
        if self.spec.discretize_bins > 1 and self.numeric_cols_:
            self.discretizer_ = KBinsDiscretizer(
                n_bins=self.spec.discretize_bins,
                encode="ordinal",
                strategy="quantile",
                random_state=self.seed,
            )
            try:
                binned = self.discretizer_.fit_transform(X_train[self.numeric_cols_].fillna(0))
                for idx, col in enumerate(self.numeric_cols_):
                    out[f"{col}_bin"] = binned[:, idx]
            except Exception:
                self.discretizer_ = None

        # 4. Pairwise interactions (top numeric features)
        if self.spec.interactions and len(self.numeric_cols_) >= 2:
            self.interaction_pairs_ = []
            selected_num = self.numeric_cols_[:5]
            for i in range(len(selected_num)):
                for j in range(i + 1, len(selected_num)):
                    if len(self.interaction_pairs_) >= self.spec.max_interaction_features:
                        break
                    c1, c2 = selected_num[i], selected_num[j]
                    self.interaction_pairs_.append((c1, c2))
                    out[f"{c1}_x_{c2}"] = out[c1] * out[c2]

        # 5. Out-of-fold target encoding for categoricals
        if self.spec.target_encoding and self.categorical_cols_:
            self.target_encoder_ = OutOfFoldTargetEncoder(
                categorical_columns=self.categorical_cols_,
                seed=self.seed,
            )
            out = self.target_encoder_.fit_transform(out, y_train)

        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted transforms to validation or test data without data leakage."""
        out = X.copy()

        # 1. Missing indicators
        if self.indicator_ is not None and self.numeric_cols_:
            ind_arr = self.indicator_.transform(X[self.numeric_cols_])
            if ind_arr.shape[1] > 0:
                for idx in range(ind_arr.shape[1]):
                    out[f"num_missing_{idx}"] = ind_arr[:, idx].astype(float)

        # 2. Imputation and Scaling
        if self.imputer_ is not None and self.numeric_cols_:
            imputed_num = self.imputer_.transform(X[self.numeric_cols_])
            scaled_num = self.scaler_.transform(imputed_num) if self.scaler_ is not None else imputed_num
            for idx, col in enumerate(self.numeric_cols_):
                out[col] = scaled_num[:, idx]

        # 3. Discretization
        if self.discretizer_ is not None and self.numeric_cols_:
            try:
                binned = self.discretizer_.transform(X[self.numeric_cols_].fillna(0))
                for idx, col in enumerate(self.numeric_cols_):
                    out[f"{col}_bin"] = binned[:, idx]
            except Exception:
                pass

        # 4. Interactions
        for c1, c2 in self.interaction_pairs_:
            if c1 in out.columns and c2 in out.columns:
                out[f"{c1}_x_{c2}"] = out[c1] * out[c2]

        # 5. Target encoding
        if self.target_encoder_ is not None:
            out = self.target_encoder_.transform(out)

        return out
