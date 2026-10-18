"""Multi-Paradigm Model Zoo & Unified Estimator Factory (Phase 4).

Provides:
- Unified estimator factory supporting diverse model families:
  * `catboost`: Native ordered boosting with categorical features.
  * `lgbm`: Fast histogram GBDT with colsample/subsample regularization.
  * `xgboost`: Depth-wise / loss-guided regularized GBDT.
  * `hist_gbdt`: Scikit-learn native Histogram GBDT.
  * `penalized_linear`: LogisticRegression (L1/L2/ElasticNet) or Ridge/ElasticNet
    wrapped with RobustScaler + SimpleImputer + missingness indicators.
  * `neural_tabular`: Lightweight Residual MLP architecture.
- Graceful fallbacks when optional native binaries (CatBoost / XGBoost) are not installed.
- Out-of-fold (OOF) cross-validation predictions, probabilities, and fold-level metrics.
"""

from __future__ import annotations

import copy
import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# Suppress macOS duplicate OpenMP runtime conflict (EXC_BAD_ACCESS in libomp)
# when both LightGBM and PyTorch native C++ runtimes are loaded in the same process:
if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

# Import torch first on macOS to ensure compatible OpenMP initialization
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    if platform.system() == "Darwin":
        torch.set_num_threads(1)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Detect optional native dependencies
try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# =============================================================================
# PyTorch Lightweight Residual MLP for Tabular Data
# =============================================================================

if HAS_TORCH:
    class _TabularResNetModule(nn.Module):
        """Residual MLP with skip connections and LayerNorm for tabular feature vectors."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            hidden_dim: int = 128,
            num_blocks: int = 2,
            dropout: float = 0.1,
            is_classification: bool = True,
        ):
            super().__init__()
            self.is_classification = is_classification
            self.input_layer = nn.Sequential(
                nn.Linear(in_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_blocks)
            ])
            self.head = nn.Linear(hidden_dim, out_features)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.input_layer(x)
            for block in self.blocks:
                h = self.relu(h + block(h))
            return self.head(h)


class LightweightResidualMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible Lightweight Residual Tabular MLP Classifier."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _prepare_X(self, X: Any, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if fit:
            self._imputer = SimpleImputer(strategy="median")
            self._scaler = RobustScaler()
            X_imp = self._imputer.fit_transform(X_arr)
            return self._scaler.fit_transform(X_imp)
        else:
            X_imp = self._imputer.transform(X_arr)
            return self._scaler.transform(X_imp)

    def fit(self, X: Any, y: Any) -> LightweightResidualMLPClassifier:
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_indexed = np.array([class_to_idx[c] for c in y_arr], dtype=np.int64)

        X_norm = self._prepare_X(X, fit=True)
        in_features = X_norm.shape[1]

        if not HAS_TORCH:
            self._fallback_mlp = MLPClassifier(
                hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
                activation="relu",
                max_iter=self.epochs * 10,
                random_state=self.random_state,
                early_stopping=True,
            )
            self._fallback_mlp.fit(X_norm, y_indexed)
            if hasattr(self._fallback_mlp, "coefs_") and len(self._fallback_mlp.coefs_) > 0:
                self.feature_importances_ = np.linalg.norm(self._fallback_mlp.coefs_[0], axis=1)
                s = self.feature_importances_.sum()
                if s > 0:
                    self.feature_importances_ /= s
            else:
                self.feature_importances_ = np.ones(in_features) / in_features
            return self

        if platform.system() == "Darwin":
            torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        out_features = max(2, n_classes)
        self.model_ = _TabularResNetModule(
            in_features=in_features,
            out_features=out_features,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            is_classification=True,
        )

        dataset = TensorDataset(torch.from_numpy(X_norm).float(), torch.from_numpy(y_indexed).long())
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        # Compute feature importances as L2 norm of first layer weights
        with torch.no_grad():
            first_weight = self.model_.input_layer[0].weight.detach().cpu().numpy()
            raw_imps = np.linalg.norm(first_weight, axis=0)
            total = raw_imps.sum()
            self.feature_importances_ = raw_imps / total if total > 0 else np.ones(in_features) / in_features

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X_norm = self._prepare_X(X, fit=False)
        if not HAS_TORCH:
            return self._fallback_mlp.predict_proba(X_norm)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_norm).float())
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if len(self.classes_) == 2 and probs.shape[1] > 2:
                probs = probs[:, :2]
                probs /= probs.sum(axis=1, keepdims=True)
            elif len(self.classes_) < probs.shape[1]:
                probs = probs[:, :len(self.classes_)]
                probs /= probs.sum(axis=1, keepdims=True)
            return probs

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]


class LightweightResidualMLPRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible Lightweight Residual Tabular MLP Regressor."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 2,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 25,
        batch_size: int = 64,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _prepare_X(self, X: Any, fit: bool = False) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if fit:
            self._imputer = SimpleImputer(strategy="median")
            self._scaler = RobustScaler()
            X_imp = self._imputer.fit_transform(X_arr)
            return self._scaler.fit_transform(X_imp)
        else:
            X_imp = self._imputer.transform(X_arr)
            return self._scaler.transform(X_imp)

    def fit(self, X: Any, y: Any) -> LightweightResidualMLPRegressor:
        y_arr = np.asarray(y, dtype=float)
        self._y_mean = float(np.mean(y_arr))
        self._y_std = float(np.std(y_arr)) if float(np.std(y_arr)) > 1e-6 else 1.0
        y_scaled = (y_arr - self._y_mean) / self._y_std

        X_norm = self._prepare_X(X, fit=True)
        in_features = X_norm.shape[1]

        if not HAS_TORCH:
            self._fallback_mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden_dim, self.hidden_dim // 2),
                activation="relu",
                max_iter=self.epochs * 10,
                random_state=self.random_state,
                early_stopping=True,
            )
            self._fallback_mlp.fit(X_norm, y_scaled)
            if hasattr(self._fallback_mlp, "coefs_") and len(self._fallback_mlp.coefs_) > 0:
                self.feature_importances_ = np.linalg.norm(self._fallback_mlp.coefs_[0], axis=1)
                s = self.feature_importances_.sum()
                if s > 0:
                    self.feature_importances_ /= s
            else:
                self.feature_importances_ = np.ones(in_features) / in_features
            return self

        if platform.system() == "Darwin":
            torch.set_num_threads(1)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model_ = _TabularResNetModule(
            in_features=in_features,
            out_features=1,
            hidden_dim=self.hidden_dim,
            num_blocks=self.num_blocks,
            dropout=self.dropout,
            is_classification=False,
        )

        dataset = TensorDataset(torch.from_numpy(X_norm).float(), torch.from_numpy(y_scaled).float().unsqueeze(1))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model_(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.model_.eval()
        with torch.no_grad():
            first_weight = self.model_.input_layer[0].weight.detach().cpu().numpy()
            raw_imps = np.linalg.norm(first_weight, axis=0)
            total = raw_imps.sum()
            self.feature_importances_ = raw_imps / total if total > 0 else np.ones(in_features) / in_features

        return self

    def predict(self, X: Any) -> np.ndarray:
        X_norm = self._prepare_X(X, fit=False)
        if not HAS_TORCH:
            pred_scaled = self._fallback_mlp.predict(X_norm)
            return pred_scaled * self._y_std + self._y_mean

        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X_norm).float()).cpu().numpy().flatten()
            return out * self._y_std + self._y_mean


# =============================================================================
# Unified Estimator Factory
# =============================================================================

KNOWN_MODEL_FAMILIES = (
    "catboost",
    "lgbm",
    "xgboost",
    "hist_gbdt",
    "penalized_linear",
    "neural_tabular",
)


def normalize_family_name(family: str) -> str:
    """Normalize family string into standard canonical key."""
    norm = family.strip().lower()
    if norm in ("lgb", "lightgbm", "lgbm"):
        return "lgbm"
    if norm in ("xgb", "xgboost"):
        return "xgboost"
    if norm in ("cat", "catboost"):
        return "catboost"
    if norm in ("hgb", "hist_gbdt", "hist_gradient_boosting", "histgradientboosting"):
        return "hist_gbdt"
    if norm in ("linear", "penalized_linear", "logistic_regression", "ridge", "elasticnet"):
        return "penalized_linear"
    if norm in ("neural", "neural_tabular", "mlp", "residual_mlp", "resnet", "tabnet"):
        return "neural_tabular"
    return norm


def create_estimator(
    family: str,
    task_type: str = "binary_classification",
    random_state: int = 42,
    **kwargs: Any,
) -> BaseEstimator:
    """Create a unified scikit-learn compatible estimator for the requested family.

    Enforces graceful fallbacks if optional native binaries (CatBoost / XGBoost / LightGBM)
    are not installed in the environment.

    Args:
        family: One of 'catboost', 'lgbm', 'xgboost', 'hist_gbdt', 'penalized_linear', 'neural_tabular'.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        random_state: Deterministic random seed.
        **kwargs: Optional hyperparameter overrides.

    Returns:
        Fitted scikit-learn compatible BaseEstimator instance with metadata attributes:
        `_family`, `_effective_family`, `_is_fallback`, `_original_family`.
    """
    canonical_family = normalize_family_name(family)
    is_classification = "classification" in task_type or task_type in ("binary", "multiclass")

    is_fallback = False
    effective_family = canonical_family

    if canonical_family == "catboost":
        if HAS_CATBOOST:
            if is_classification:
                est = CatBoostClassifier(
                    iterations=kwargs.get("iterations", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    depth=kwargs.get("depth", 6),
                    random_seed=random_state,
                    verbose=0,
                    thread_count=1,
                )
            else:
                est = CatBoostRegressor(
                    iterations=kwargs.get("iterations", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    depth=kwargs.get("depth", 6),
                    random_seed=random_state,
                    verbose=0,
                    thread_count=1,
                )
        else:
            is_fallback = True
            logger.info("CatBoost binary not installed; falling back to LightGBM / HistGBDT.")
            if HAS_LIGHTGBM:
                effective_family = "lgbm"
                if is_classification:
                    est = LGBMClassifier(
                        n_estimators=kwargs.get("n_estimators", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=random_state,
                        verbose=-1,
                        n_jobs=1,
                    )
                else:
                    est = LGBMRegressor(
                        n_estimators=kwargs.get("n_estimators", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=random_state,
                        verbose=-1,
                        n_jobs=1,
                    )
            else:
                effective_family = "hist_gbdt"
                if is_classification:
                    est = HistGradientBoostingClassifier(
                        max_iter=kwargs.get("max_iter", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        random_state=random_state,
                    )
                else:
                    est = HistGradientBoostingRegressor(
                        max_iter=kwargs.get("max_iter", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        random_state=random_state,
                    )

    elif canonical_family == "lgbm":
        if HAS_LIGHTGBM:
            if is_classification:
                est = LGBMClassifier(
                    n_estimators=kwargs.get("n_estimators", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    subsample=kwargs.get("subsample", 0.8),
                    random_state=random_state,
                    verbose=-1,
                    n_jobs=1,
                )
            else:
                est = LGBMRegressor(
                    n_estimators=kwargs.get("n_estimators", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    subsample=kwargs.get("subsample", 0.8),
                    random_state=random_state,
                    verbose=-1,
                    n_jobs=1,
                )
        else:
            is_fallback = True
            effective_family = "hist_gbdt"
            logger.info("LightGBM not installed; falling back to HistGBDT.")
            if is_classification:
                est = HistGradientBoostingClassifier(
                    max_iter=kwargs.get("max_iter", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    random_state=random_state,
                )
            else:
                est = HistGradientBoostingRegressor(
                    max_iter=kwargs.get("max_iter", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    random_state=random_state,
                )

    elif canonical_family == "xgboost":
        if HAS_XGBOOST:
            if is_classification:
                est = XGBClassifier(
                    n_estimators=kwargs.get("n_estimators", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    max_depth=kwargs.get("max_depth", 6),
                    subsample=kwargs.get("subsample", 0.8),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    reg_alpha=kwargs.get("reg_alpha", 0.1),
                    reg_lambda=kwargs.get("reg_lambda", 1.0),
                    random_state=random_state,
                    n_jobs=1,
                    eval_metric="logloss",
                )
            else:
                est = XGBRegressor(
                    n_estimators=kwargs.get("n_estimators", 300),
                    learning_rate=kwargs.get("learning_rate", 0.05),
                    max_depth=kwargs.get("max_depth", 6),
                    subsample=kwargs.get("subsample", 0.8),
                    colsample_bytree=kwargs.get("colsample_bytree", 0.8),
                    reg_alpha=kwargs.get("reg_alpha", 0.1),
                    reg_lambda=kwargs.get("reg_lambda", 1.0),
                    random_state=random_state,
                    n_jobs=1,
                    eval_metric="rmse",
                )
        else:
            is_fallback = True
            logger.info("XGBoost not installed; falling back to LightGBM / HistGBDT.")
            if HAS_LIGHTGBM:
                effective_family = "lgbm"
                if is_classification:
                    est = LGBMClassifier(
                        n_estimators=kwargs.get("n_estimators", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=random_state,
                        verbose=-1,
                        n_jobs=1,
                    )
                else:
                    est = LGBMRegressor(
                        n_estimators=kwargs.get("n_estimators", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        colsample_bytree=0.8,
                        subsample=0.8,
                        random_state=random_state,
                        verbose=-1,
                        n_jobs=1,
                    )
            else:
                effective_family = "hist_gbdt"
                if is_classification:
                    est = HistGradientBoostingClassifier(
                        max_iter=kwargs.get("max_iter", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        random_state=random_state,
                    )
                else:
                    est = HistGradientBoostingRegressor(
                        max_iter=kwargs.get("max_iter", 300),
                        learning_rate=kwargs.get("learning_rate", 0.05),
                        random_state=random_state,
                    )

    elif canonical_family == "hist_gbdt":
        if is_classification:
            est = HistGradientBoostingClassifier(
                max_iter=kwargs.get("max_iter", 300),
                learning_rate=kwargs.get("learning_rate", 0.05),
                random_state=random_state,
            )
        else:
            est = HistGradientBoostingRegressor(
                max_iter=kwargs.get("max_iter", 300),
                learning_rate=kwargs.get("learning_rate", 0.05),
                random_state=random_state,
            )

    elif canonical_family == "penalized_linear":
        penalty = kwargs.get("penalty", "l2")
        c_param = kwargs.get("C", 1.0)
        alpha = kwargs.get("alpha", 1.0)
        l1_ratio = kwargs.get("l1_ratio", 0.5)

        if is_classification:
            if penalty == "elasticnet":
                linear_model = LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=l1_ratio,
                    C=c_param,
                    max_iter=1000,
                    random_state=random_state,
                )
            elif penalty == "l1":
                linear_model = LogisticRegression(
                    penalty="l1",
                    solver="saga",
                    C=c_param,
                    max_iter=1000,
                    random_state=random_state,
                )
            else:
                linear_model = LogisticRegression(
                    C=c_param,
                    max_iter=1000,
                    random_state=random_state,
                )
        else:
            if penalty == "elasticnet":
                linear_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
            else:
                linear_model = Ridge(alpha=alpha, random_state=random_state)

        # Pipeline: RobustScaler + SimpleImputer with missingness indicators
        est = Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", RobustScaler()),
            ("model", linear_model),
        ])

    elif canonical_family == "neural_tabular":
        if is_classification:
            est = LightweightResidualMLPClassifier(
                hidden_dim=kwargs.get("hidden_dim", 128),
                num_blocks=kwargs.get("num_blocks", 2),
                dropout=kwargs.get("dropout", 0.1),
                lr=kwargs.get("lr", 1e-3),
                epochs=kwargs.get("epochs", 25),
                batch_size=kwargs.get("batch_size", 64),
                random_state=random_state,
            )
        else:
            est = LightweightResidualMLPRegressor(
                hidden_dim=kwargs.get("hidden_dim", 128),
                num_blocks=kwargs.get("num_blocks", 2),
                dropout=kwargs.get("dropout", 0.1),
                lr=kwargs.get("lr", 1e-3),
                epochs=kwargs.get("epochs", 25),
                batch_size=kwargs.get("batch_size", 64),
                random_state=random_state,
            )

    else:
        raise ValueError(
            f"Unknown model family '{family}'. Must be one of: {list(KNOWN_MODEL_FAMILIES)}"
        )

    # Attach diagnostic metadata
    setattr(est, "_family", canonical_family)
    setattr(est, "_effective_family", effective_family)
    setattr(est, "_is_fallback", is_fallback)
    setattr(est, "_original_family", canonical_family)
    return est


# =============================================================================
# Out-of-Fold (OOF) Prediction and Cross-Validation Evaluation
# =============================================================================

@dataclass
class ModelOofResult:
    """Certified Out-of-Fold evaluation result across locked cross-validation splits."""
    family: str
    effective_family: str
    is_fallback: bool
    oof_predictions: np.ndarray
    oof_probabilities: Optional[np.ndarray]
    cv_scores: List[float]
    train_scores: List[float]
    mean_cv_score: float
    mean_train_score: float
    train_val_gap: float
    feature_importances: np.ndarray
    feature_names: List[str]
    fitted_estimators: List[Any]
    task_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "effective_family": self.effective_family,
            "is_fallback": self.is_fallback,
            "cv_scores": [float(s) for s in self.cv_scores],
            "train_scores": [float(s) for s in self.train_scores],
            "mean_cv_score": float(self.mean_cv_score),
            "mean_train_score": float(self.mean_train_score),
            "train_val_gap": float(self.train_val_gap),
            "feature_names": self.feature_names,
            "task_type": self.task_type,
        }


def extract_feature_importances(estimator: Any, n_features: int) -> np.ndarray:
    """Extract or approximate non-negative feature importances from a fitted estimator."""
    # 1. Native feature_importances_ attribute (GBDTs, Random Forests, Residual MLP)
    if hasattr(estimator, "feature_importances_"):
        imps = np.asarray(estimator.feature_importances_, dtype=float)
        if len(imps) == n_features:
            return np.maximum(0.0, imps)
        elif len(imps) > n_features:
            return np.maximum(0.0, imps[:n_features])

    # 2. Pipeline unwrapping
    if isinstance(estimator, Pipeline):
        model_step = estimator.named_steps.get("model")
        if model_step is not None:
            if hasattr(model_step, "coef_"):
                coefs = np.asarray(model_step.coef_, dtype=float)
                if coefs.ndim > 1:
                    raw = np.mean(np.abs(coefs), axis=0)
                else:
                    raw = np.abs(coefs)
                if len(raw) >= n_features:
                    return raw[:n_features]
            sub_imp = extract_feature_importances(model_step, n_features)
            if sub_imp is not None:
                return sub_imp

    # 3. Direct linear model
    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_, dtype=float)
        if coefs.ndim > 1:
            raw = np.mean(np.abs(coefs), axis=0)
        else:
            raw = np.abs(coefs)
        if len(raw) >= n_features:
            return raw[:n_features]

    # Uniform fallback
    return np.ones(n_features, dtype=float) / max(1, n_features)


def evaluate_model_oof(
    estimator_or_family: Union[str, BaseEstimator],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_splitter: Any,
    task_type: str = "binary_classification",
    groups: Optional[Any] = None,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
    **estimator_kwargs: Any,
) -> ModelOofResult:
    """Train estimator across locked CV folds, generating certified OOF predictions and metrics.

    Args:
        estimator_or_family: Estimator instance or family name string ('lgbm', 'catboost', etc.).
        X: Feature matrix (DataFrame or 2D array).
        y: Target series or array.
        cv_splitter: Cross-validation splitter object with `.split(X, y, groups)`.
        task_type: 'binary_classification', 'multiclass_classification', or 'regression'.
        groups: Optional group array for group-aware splitters.
        random_state: Random state for repeatability.
        feature_names: Optional explicit list of feature names.
        **estimator_kwargs: Extra keyword args passed to `create_estimator`.

    Returns:
        ModelOofResult with complete OOF predictions, probabilities, and fold-level metrics.
    """
    is_classification = "classification" in task_type or task_type in ("binary", "multiclass")

    if isinstance(estimator_or_family, str):
        base_estimator = create_estimator(
            family=estimator_or_family,
            task_type=task_type,
            random_state=random_state,
            **estimator_kwargs,
        )
        family_name = normalize_family_name(estimator_or_family)
    else:
        base_estimator = estimator_or_family
        family_name = getattr(base_estimator, "_family", type(base_estimator).__name__)

    effective_family = getattr(base_estimator, "_effective_family", family_name)
    is_fallback = getattr(base_estimator, "_is_fallback", False)

    # Feature names
    if isinstance(X, pd.DataFrame):
        actual_feature_names = list(X.columns)
        X_df = X
    else:
        actual_feature_names = feature_names or [f"f_{i}" for i in range(X.shape[1])]
        X_df = pd.DataFrame(X, columns=actual_feature_names)

    y_arr = np.asarray(y)
    n_samples = len(X_df)
    n_features = len(actual_feature_names)

    # Prepare OOF structures
    classes: Optional[np.ndarray] = None
    if is_classification:
        classes = np.unique(y_arr)
        n_classes = len(classes)
        oof_predictions = np.empty(n_samples, dtype=y_arr.dtype)
        oof_probabilities = np.zeros((n_samples, n_classes), dtype=float)
    else:
        oof_predictions = np.zeros(n_samples, dtype=float)
        oof_probabilities = None

    # Resolve splits
    try:
        splits = list(cv_splitter.split(X_df, y_arr, groups=groups))
    except TypeError:
        try:
            splits = list(cv_splitter.split(X_df, y_arr))
        except TypeError:
            splits = list(cv_splitter.split(X_df))

    if not splits:
        raise ValueError("CV splitter generated 0 splits.")

    train_scores: List[float] = []
    val_scores: List[float] = []
    fitted_estimators: List[Any] = []
    fold_importances: List[np.ndarray] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        est = clone(base_estimator)
        X_tr = X_df.iloc[train_idx]
        y_tr = y_arr[train_idx]
        X_v = X_df.iloc[val_idx]
        y_v = y_arr[val_idx]

        # Fit in-fold
        est.fit(X_tr, y_tr)
        fitted_estimators.append(est)

        # Collect feature importances
        imp = extract_feature_importances(est, n_features)
        fold_importances.append(imp)

        # Evaluate validation
        if is_classification:
            pred_val = est.predict(X_v)
            oof_predictions[val_idx] = pred_val

            if hasattr(est, "predict_proba"):
                prob_val = est.predict_proba(X_v)
                if prob_val.shape[1] == len(classes):
                    oof_probabilities[val_idx] = prob_val
                elif prob_val.shape[1] > len(classes):
                    oof_probabilities[val_idx] = prob_val[:, :len(classes)]
                else:
                    oof_probabilities[val_idx, :prob_val.shape[1]] = prob_val

            pred_tr = est.predict(X_tr)
            train_score = float(balanced_accuracy_score(y_tr, pred_tr))
            val_score = float(balanced_accuracy_score(y_v, pred_val))
        else:
            pred_val = est.predict(X_v).flatten()
            oof_predictions[val_idx] = pred_val

            pred_tr = est.predict(X_tr).flatten()
            train_score = float(r2_score(y_tr, pred_tr))
            val_score = float(r2_score(y_v, pred_val))

        train_scores.append(train_score)
        val_scores.append(val_score)

    mean_train = float(np.mean(train_scores))
    mean_val = float(np.mean(val_scores))
    train_val_gap = float(mean_train - mean_val)

    # Average feature importances across folds
    mean_importances = np.mean(fold_importances, axis=0) if fold_importances else np.ones(n_features) / n_features
    total_imp = np.sum(mean_importances)
    if total_imp > 0:
        mean_importances = mean_importances / total_imp

    return ModelOofResult(
        family=family_name,
        effective_family=effective_family,
        is_fallback=is_fallback,
        oof_predictions=oof_predictions,
        oof_probabilities=oof_probabilities,
        cv_scores=val_scores,
        train_scores=train_scores,
        mean_cv_score=mean_val,
        mean_train_score=mean_train,
        train_val_gap=train_val_gap,
        feature_importances=mean_importances,
        feature_names=actual_feature_names,
        fitted_estimators=fitted_estimators,
        task_type=task_type,
    )


def train_and_diagnose_portfolio(
    families: Sequence[str],
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    cv_splitter: Any,
    task_type: str = "binary_classification",
    groups: Optional[Any] = None,
    random_state: int = 42,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, ModelOofResult]:
    """Train multiple model families across locked CV folds, returning a portfolio of OOF results."""
    results: Dict[str, ModelOofResult] = {}
    for fam in families:
        res = evaluate_model_oof(
            estimator_or_family=fam,
            X=X,
            y=y,
            cv_splitter=cv_splitter,
            task_type=task_type,
            groups=groups,
            random_state=random_state,
            feature_names=feature_names,
        )
        results[fam] = res
    return results
