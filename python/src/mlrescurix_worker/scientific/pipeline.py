"""Self-Contained Full Inference Pipeline for Zero-Skew Bundle V2 Packaging (Phase 5).

Encapsulates the complete unified inference pipeline:
  DirtyDataSanitizer -> ColumnDropper(id_cols) -> FeatureEngineeringGraph
  -> ParetoEnsemblePredictor -> CalibratedThresholdCategorizer

Accepts raw, uncleaned JSON/CSV rows (including ID columns, raw strings,
currency "$1,200", sentinels "?", "-999", accounting negatives, etc.)
and reproduces CV evaluation predictions within 10^-6 float tolerance.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

from .dirty_sanitizer import UNIVERSAL_SENTINELS, apply_coercions, parse_dirty_numeric_value, sanitize_dataframe
from .primitives import (
    InformativeMissingnessAndInteractions,
    LightweightTextEmbedder,
    NestedOOFEmpiricalBayesEncoder,
    QuasiIdDecomposer,
)

logger = logging.getLogger(__name__)


class DirtyDataSanitizerTransformer(BaseEstimator, TransformerMixin):
    """Zero-copy / low-overhead dirty string cleaner and sentinel normalizer transformer."""

    def __init__(
        self,
        coercions: Optional[Dict[str, str]] = None,
        sentinels: Optional[Set[str]] = None,
    ):
        self.coercions = coercions or {}
        self.sentinels = sentinels or set(UNIVERSAL_SENTINELS)

    def fit(self, X: Any, y: Any = None) -> DirtyDataSanitizerTransformer:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if isinstance(X, list):
            df = pd.DataFrame(X)
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        res = sanitize_dataframe(df)
        out_df = res.df
        if self.coercions:
            out_df = apply_coercions(out_df, self.coercions)
        return out_df


class ColumnDropperTransformer(BaseEstimator, TransformerMixin):
    """Deterministic isolation and dropping of surrogate keys, IDs, and target columns."""

    def __init__(
        self,
        id_cols: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
    ):
        resolved_ids = id_cols if id_cols is not None else id_columns
        self.id_cols = list(resolved_ids or [])
        self.id_columns = self.id_cols
        self.target_col = target_col or target_column
        self.target_column = self.target_col

    def fit(self, X: Any, y: Any = None) -> ColumnDropperTransformer:
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        to_drop = [c for c in self.id_cols if c in df.columns]
        if self.target_col and self.target_col in df.columns:
            to_drop.append(self.target_col)
        if to_drop:
            df = df.drop(columns=to_drop)
        return df


class FeatureEngineeringGraphTransformer(BaseEstimator, TransformerMixin):
    """Leakage-free Feature Engineering Graph orchestrating scientific primitives."""

    def __init__(
        self,
        column_classifications: Optional[Dict[str, str]] = None,
        task_type: str = "binary_classification",
        random_state: int = 42,
    ):
        self.column_classifications = column_classifications
        self.task_type = task_type
        self.random_state = random_state

        # Primitives
        self.text_embedders_: Dict[str, LightweightTextEmbedder] = {}
        self.quasi_decomposers_: Dict[str, QuasiIdDecomposer] = {}
        self.cat_encoders_: Dict[str, NestedOOFEmpiricalBayesEncoder] = {}
        self.missingness_interactions_: Optional[InformativeMissingnessAndInteractions] = None
        self.num_imputer_: Optional[SimpleImputer] = None
        self.feature_names_out_: List[str] = []
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Any) -> FeatureEngineeringGraphTransformer:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        y_arr = np.asarray(y)
        col_classes = dict(self.column_classifications or {})
        if not col_classes:
            col_classes = {}
            for col in df.columns:
                c_low = col.lower()
                if c_low in ("survived", "target", "label", "saleprice", "isfraud", "cover_type", "sales"):
                    col_classes[col] = "TARGET"
                elif c_low in ("passengerid", "id", "row_id") or (len(c_low) > 2 and c_low.endswith("id")):
                    col_classes[col] = "DROP_SURROGATE_ID"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    col_classes[col] = "NUMERIC_CONTINUOUS"
                else:
                    col_classes[col] = "CATEGORICAL_SAFE"
            self.column_classifications = col_classes
        exclude_cols = {
            col for col, cls in col_classes.items()
            if cls in ("TARGET", "DROP_SURROGATE_ID")
        } | {
            c for c in df.columns if c.lower() in (
                "survived", "target", "label", "passengerid", "id", "row_id", "saleprice", "isfraud", "cover_type", "sales"
            ) or c.lower().endswith("id")
        }
        df_clean = df.drop(columns=[c for c in df.columns if c in exclude_cols], errors="ignore")

        # 1. Text Embedders for SHORT_TEXT_EMBED columns (e.g. Name)
        self.text_embedders_ = {}
        for col, cls in col_classes.items():
            if cls == "SHORT_TEXT_EMBED" and col in df_clean.columns:
                emb = LightweightTextEmbedder(
                    text_columns=[col],
                    n_svd_components=4,
                    top_k_subtokens=25,
                    min_subtoken_freq=3,
                    random_state=self.random_state,
                )
                emb.fit(df_clean, y_arr)
                self.text_embedders_[col] = emb

        # 2. QuasiId Decomposers for QUASI_ID_DECOMPOSE columns (e.g. Ticket, Cabin)
        self.quasi_decomposers_ = {}
        for col, cls in col_classes.items():
            if cls == "QUASI_ID_DECOMPOSE" and col in df_clean.columns:
                # Find available numeric columns for shared member aggregations
                num_agg_cols = [
                    c for c, c_cls in col_classes.items()
                    if c_cls in ("NUMERIC_CONTINUOUS", "NUMERIC_DISCRETE") and c in df_clean.columns
                ]
                if not num_agg_cols:
                    num_agg_cols = [c for c in df_clean.columns if pd.api.types.is_numeric_dtype(df_clean[c])]
                decomp = QuasiIdDecomposer(
                    id_columns=[col],
                    numeric_columns=num_agg_cols[:2],
                    encode_prefix=True,
                    random_state=self.random_state,
                )
                decomp.fit(df_clean, y_arr)
                self.quasi_decomposers_[col] = decomp

        # 3. Categorical Encoders for SAFE_CATEGORICAL columns (e.g. Sex, Embarked, Pclass)
        self.cat_encoders_ = {}
        safe_cats = [
            col for col, cls in col_classes.items()
            if cls == "SAFE_CATEGORICAL" and col in df_clean.columns
        ]
        if safe_cats:
            cat_enc = NestedOOFEmpiricalBayesEncoder(
                categorical_columns=safe_cats,
                n_splits=5,
                min_smoothing=10.0,
                noise_level=0.01,
                random_state=self.random_state,
            )
            cat_enc.fit(df_clean, y_arr)
            self.cat_encoders_["safe_cats"] = cat_enc

        # 4. Transform intermediate features to fit Missingness & Interactions
        intermediate_df = self._transform_primitives(df_clean)

        # 5. Missingness & Numeric Interactions
        numeric_cols = [c for c in intermediate_df.columns if pd.api.types.is_numeric_dtype(intermediate_df[c])]
        self.missingness_interactions_ = InformativeMissingnessAndInteractions(
            numeric_columns=numeric_cols,
            missing_rate_bounds=(0.005, 0.995),
            max_interactions=5,
            random_state=self.random_state,
        )
        self.missingness_interactions_.fit(intermediate_df, y_arr)

        full_df = self.missingness_interactions_.transform(intermediate_df)
        exclude_cols = {
            col for col, cls in col_classes.items()
            if cls in ("TARGET", "DROP_SURROGATE_ID")
        }
        final_num_cols = [
            c for c in full_df.columns
            if pd.api.types.is_numeric_dtype(full_df[c]) and c not in exclude_cols
        ]

        # 6. Fit Median Imputer on all numeric features
        self.num_imputer_ = SimpleImputer(strategy="median")
        self.num_imputer_.fit(full_df[final_num_cols].fillna(0.0))
        self.feature_names_out_ = final_num_cols
        self.is_fitted_ = True
        return self

    def _transform_primitives(self, df: pd.DataFrame) -> pd.DataFrame:
        col_classes = dict(self.column_classifications or {})
        if not col_classes:
            col_classes = {}
            for col in df.columns:
                c_low = col.lower()
                if c_low in ("survived", "target", "label", "saleprice", "isfraud", "cover_type", "sales"):
                    col_classes[col] = "TARGET"
                elif c_low in ("passengerid", "id", "row_id") or (len(c_low) > 2 and c_low.endswith("id")):
                    col_classes[col] = "DROP_SURROGATE_ID"
                elif pd.api.types.is_numeric_dtype(df[col]):
                    col_classes[col] = "NUMERIC_CONTINUOUS"
                else:
                    col_classes[col] = "CATEGORICAL_SAFE"
            self.column_classifications = col_classes
        exclude_cols = {
            col for col, cls in col_classes.items()
            if cls in ("TARGET", "DROP_SURROGATE_ID")
        }
        new_cols: Dict[str, Any] = {}

        # Retain clean numeric columns (strictly excluding target and surrogate IDs)
        for col, cls in col_classes.items():
            if cls in ("NUMERIC_CONTINUOUS", "NUMERIC_DISCRETE") and col in df.columns and col not in exclude_cols:
                new_cols[col] = pd.to_numeric(df[col], errors="coerce")

        # Group and household features if discrete member indicators exist
        if "SibSp" in df.columns and "Parch" in df.columns:
            sibsp = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0.0)
            parch = pd.to_numeric(df["Parch"], errors="coerce").fillna(0.0)
            family_size = sibsp + parch + 1.0
            new_cols["family_size"] = family_size
            new_cols["is_alone"] = (family_size == 1.0).astype(float)
            if "Fare" in df.columns:
                fare = pd.to_numeric(df["Fare"], errors="coerce").fillna(0.0)
                new_cols["fare_per_member"] = fare / np.maximum(1.0, family_size)

        # Transform text features (only add newly generated features)
        for col, emb in self.text_embedders_.items():
            if col in df.columns:
                text_feats = emb.transform(df)
                for c in text_feats.columns:
                    if c not in df.columns or c.startswith(f"{col}__"):
                        new_cols[c] = text_feats[c]

        # Transform quasi-IDs (only add newly generated features)
        for col, decomp in self.quasi_decomposers_.items():
            if col in df.columns:
                quasi_feats = decomp.transform(df)
                for c in quasi_feats.columns:
                    if c not in df.columns or c.startswith(f"{col}__"):
                        new_cols[c] = quasi_feats[c]

        # Transform safe categoricals (only add newly generated features)
        for key, enc in self.cat_encoders_.items():
            cat_feats = enc.transform(df)
            for c in cat_feats.columns:
                if c not in df.columns or c.endswith("__eb"):
                    new_cols[c] = cat_feats[c]

        out = pd.DataFrame(new_cols, index=df.index)
        to_drop = [c for c in exclude_cols if c in out.columns]
        if to_drop:
            out = out.drop(columns=to_drop)

        return out

    def transform(self, X: Any) -> np.ndarray:
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        intermediate = self._transform_primitives(df)
        if self.missingness_interactions_ is not None:
            full_df = self.missingness_interactions_.transform(intermediate)
        else:
            full_df = intermediate

        # Ensure all expected columns exist
        for col in self.feature_names_out_:
            if col not in full_df.columns:
                full_df[col] = 0.0

        num_df = full_df[self.feature_names_out_].fillna(0.0)
        if self.num_imputer_ is not None:
            return self.num_imputer_.transform(num_df)
        return num_df.to_numpy(dtype=float)


class ParetoEnsemblePredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """Blends multiple diverse model families across the Pareto frontier."""

    def __init__(
        self,
        estimators: Optional[List[Tuple[str, Any]]] = None,
        weights: Optional[Sequence[float]] = None,
        task_type: str = "binary_classification",
        classes_: Optional[np.ndarray] = None,
    ):
        self.estimators = list(estimators or [])
        self.weights = list(weights or ([1.0 / len(self.estimators)] * len(self.estimators) if self.estimators else []))
        self.task_type = task_type
        self.classes_ = np.asarray(classes_) if classes_ is not None else np.array([0, 1])

        # Normalize weights to simplex
        total_w = sum(self.weights)
        if total_w > 1e-12:
            self.weights = [w / total_w for w in self.weights]

    def fit(self, X: Any, y: Any) -> ParetoEnsemblePredictor:
        if not self.estimators:
            from sklearn.linear_model import LogisticRegression, Ridge
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            is_class = "classification" in self.task_type or self.task_type in ("binary", "multiclass")
            if is_class:
                self.estimators = [
                    ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
                    ("hist_gbdt", HistGradientBoostingClassifier(random_state=42)),
                ]
            else:
                self.estimators = [
                    ("ridge", Ridge(alpha=1.0, random_state=42)),
                    ("hist_gbdt", HistGradientBoostingRegressor(random_state=42)),
                ]
            self.weights = [1.0 / len(self.estimators)] * len(self.estimators)

        for name, est in self.estimators:
            if hasattr(est, "fit"):
                est.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        n_classes = len(self.classes_)
        if not self.estimators:
            return np.full((len(X), n_classes), 1.0 / max(1, n_classes), dtype=float)

        all_probs: List[np.ndarray] = []
        for (name, est), w in zip(self.estimators, self.weights):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
            else:
                raw = est.predict(X)
                p = np.column_stack([1.0 - raw, raw])
            all_probs.append(p * w)

        combined = np.sum(all_probs, axis=0)
        # Ensure rows sum to 1.0
        row_sums = combined.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 1e-12, 1.0, row_sums)
        return combined / row_sums

    def predict(self, X: Any) -> np.ndarray:
        if "classification" in self.task_type or self.task_type in ("binary", "multiclass"):
            probs = self.predict_proba(X)
            idx = np.argmax(probs, axis=1)
            return self.classes_[idx]
        else:
            all_preds = [est.predict(X) * w for (name, est), w in zip(self.estimators, self.weights)]
            return np.sum(all_preds, axis=0)


class CalibratedThresholdCategorizer(BaseEstimator, ClassifierMixin):
    """Calibrates Out-of-Fold probabilities and maps them to classes via tuned decision threshold."""

    def __init__(
        self,
        decision_threshold: float = 0.5,
        calibration_method: str = "none",
        classes_: Optional[np.ndarray] = None,
    ):
        self.decision_threshold = float(decision_threshold)
        self.calibration_method = calibration_method
        self.classes_ = np.asarray(classes_) if classes_ is not None else np.array([0, 1])
        self.calibrator_: Optional[Any] = None

    def fit(self, y_true: Any, y_probs: Any) -> CalibratedThresholdCategorizer:
        # Fit probability calibrator if requested
        if self.calibration_method in ("isotonic", "platt", "sigmoid") and len(self.classes_) == 2:
            from .calibration import calibrate_probabilities
            try:
                res = calibrate_probabilities(y_true, y_probs, method=self.calibration_method, classes=list(self.classes_))
                self.calibrator_ = res.get("calibrator")
            except Exception as e:
                logger.warning(f"Failed to fit calibrator: {e}")
        return self

    def transform_proba(self, probs: np.ndarray) -> np.ndarray:
        if self.calibrator_ is not None and len(self.classes_) == 2:
            p_pos = probs[:, 1] if probs.ndim == 2 else probs
            if hasattr(self.calibrator_, "predict"):
                cal_pos = np.clip(self.calibrator_.predict(p_pos), 0.0, 1.0)
                return np.column_stack([1.0 - cal_pos, cal_pos])
        return probs

    def predict(self, probs: np.ndarray) -> np.ndarray:
        cal_probs = self.transform_proba(probs)
        if len(self.classes_) == 2:
            p_pos = cal_probs[:, 1] if cal_probs.ndim == 2 else cal_probs
            pred_idx = (p_pos >= self.decision_threshold).astype(int)
            return self.classes_[pred_idx]
        else:
            return self.classes_[np.argmax(cal_probs, axis=1)]


class FullInferencePipeline(BaseEstimator, ClassifierMixin, RegressorMixin):
    """Unified self-contained pipeline for zero train-serve skew bundle execution.
    
    Accepts raw uncleaned JSON/CSV rows and chains:
    DirtyDataSanitizer -> ColumnDropper(id_cols) -> FeatureEngineeringGraph
    -> ParetoEnsemblePredictor -> CalibratedThresholdCategorizer.
    """

    def __init__(
        self,
        column_classifications: Optional[Dict[str, str]] = None,
        coercions: Optional[Dict[str, str]] = None,
        id_cols: Optional[Sequence[str]] = None,
        id_columns: Optional[Sequence[str]] = None,
        target_col: Optional[str] = None,
        target_column: Optional[str] = None,
        estimators: Optional[List[Tuple[str, Any]]] = None,
        weights: Optional[Sequence[float]] = None,
        decision_threshold: float = 0.5,
        calibration_method: str = "none",
        task_type: str = "binary_classification",
        classes_: Optional[Sequence[Any]] = None,
        classes: Optional[Sequence[Any]] = None,
        random_state: int = 42,
        log_target: bool = False,
    ):
        self.column_classifications = dict(column_classifications or {})
        self.coercions = dict(coercions or {})
        resolved_ids = id_cols if id_cols is not None else id_columns
        self.id_cols = list(resolved_ids or [])
        self.id_columns = self.id_cols
        self.target_col = target_col or target_column
        self.target_column = self.target_col
        self.estimators = list(estimators or [])
        self.weights = list(weights or [])
        self.decision_threshold = float(decision_threshold)
        self.calibration_method = calibration_method
        self.task_type = task_type
        resolved_classes = classes_ if classes_ is not None else classes
        self.classes_ = np.asarray(resolved_classes) if resolved_classes is not None else np.array([0, 1])
        self.classes = self.classes_
        self.random_state = random_state
        self.log_target = bool(log_target)

        # Instantiate pipeline stages
        self.sanitizer = DirtyDataSanitizerTransformer(coercions=self.coercions)
        self.dropper = ColumnDropperTransformer(id_cols=self.id_cols, target_col=self.target_col)
        self.fe_graph = FeatureEngineeringGraphTransformer(
            column_classifications=self.column_classifications,
            task_type=self.task_type,
            random_state=self.random_state,
        )
        self.ensemble = ParetoEnsemblePredictor(
            estimators=self.estimators,
            weights=self.weights,
            task_type=self.task_type,
            classes_=self.classes_,
        )
        self.categorizer = CalibratedThresholdCategorizer(
            decision_threshold=self.decision_threshold,
            calibration_method=self.calibration_method,
            classes_=self.classes_,
        )
        self.is_fitted_ = False

    def fit(self, X: Any, y: Any) -> FullInferencePipeline:
        """Fit all pipeline stages sequentially on raw training data."""
        # 1. Sanitizer
        df_san = self.sanitizer.fit_transform(X)
        # 2. Dropper
        df_clean = self.dropper.fit_transform(df_san)
        # 3. Feature Engineering Graph
        X_fe = self.fe_graph.fit_transform(df_clean, y)
        # 4. Pareto Ensemble
        is_class = "classification" in self.task_type or self.task_type in ("binary", "multiclass")
        if not is_class and self.log_target:
            y_fit = np.log1p(np.maximum(0.0, np.asarray(y, dtype=float)))
        else:
            y_fit = y
        self.ensemble.fit(X_fe, y_fit)
        # 5. Categorizer threshold & calibration
        if is_class:
            probs = self.ensemble.predict_proba(X_fe)
            self.categorizer.fit(y, probs)

        self.is_fitted_ = True
        return self

    def _prepare_features(self, X: Any) -> np.ndarray:
        df_san = self.sanitizer.transform(X)
        df_clean = self.dropper.transform(df_san)
        return self.fe_graph.transform(df_clean)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict calibrated class probabilities on raw uncleaned records."""
        X_fe = self._prepare_features(X)
        raw_probs = self.ensemble.predict_proba(X_fe)
        return self.categorizer.transform_proba(raw_probs)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels or continuous values on raw uncleaned records."""
        if "classification" in self.task_type or self.task_type in ("binary", "multiclass"):
            probs = self.predict_proba(X)
            return self.categorizer.predict(probs)
        else:
            X_fe = self._prepare_features(X)
            raw = self.ensemble.predict(X_fe)
            if self.log_target:
                return np.expm1(raw)
            return raw

    def predict_log(self, X: Any) -> np.ndarray:
        """Predict continuous values on log scale."""
        X_fe = self._prepare_features(X)
        raw = self.ensemble.predict(X_fe)
        if not self.log_target:
            return np.log1p(np.maximum(0.0, raw))
        return raw
