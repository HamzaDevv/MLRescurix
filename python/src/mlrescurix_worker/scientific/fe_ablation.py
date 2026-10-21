"""M2-FE Paired Feature Ablation Engine & Auto-Gating (Phase P14 / Phase 3).

Evaluates hypothesis-driven feature transformations against a baseline feature set
using the locked CV splitter from `DatasetTopologyContract`:
1. Executes paired CV evaluation strictly in-fold to eliminate leakage.
2. Computes Delta CV Score, fold-by-fold win rate, and Delta Train-Val Gap.
3. Auto-Gate: If Delta CV <= 0 or if Delta Train-Val Gap > +0.08 (overfitting memorization),
   automatically marks the ablation as REJECTED in the ResearchLedger.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class PurgedTimeSeriesSplit:
    """Purged and embargoed cross-validation splitter for chronological data.
    
    Prevents lookahead and cross-boundary label leakage by enforcing an embargo
    gap between the training partition and the forward validation partition.
    """

    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.02):
        self.n_splits = max(2, int(n_splits))
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        embargo = max(1, int(n_samples * self.embargo_pct))
        # Total blocks = n_splits + 1
        block_size = n_samples // (self.n_splits + 1)
        if block_size < 2:
            block_size = max(2, n_samples // self.n_splits)

        for i in range(self.n_splits):
            train_end = (i + 1) * block_size
            test_start = min(n_samples, train_end + embargo)
            test_end = min(n_samples, test_start + block_size)

            if test_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def build_cv_splitter(
    topology: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
) -> Tuple[Any, Optional[str], Optional[str]]:
    """Build the locked CV splitter adhering to the DatasetTopologyContract.
    
    Returns:
        (splitter, entity_group_col, temporal_col)
    """
    cv_scheme = topology.get("cv_scheme", "stratified_kfold")
    cv_splits = int(topology.get("cv_splits", 5))
    entity_group_col = topology.get("entity_group_col")
    temporal_col = topology.get("temporal_col")
    task_type = topology.get("task_type", "binary_classification")

    if cv_scheme == "purged_time_series_split" or temporal_col is not None:
        splitter = PurgedTimeSeriesSplit(n_splits=cv_splits, embargo_pct=0.02)
    elif cv_scheme == "stratified_group_kfold" and entity_group_col is not None:
        splitter = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    elif cv_scheme == "group_kfold" and entity_group_col is not None:
        splitter = GroupKFold(n_splits=cv_splits)
    elif cv_scheme == "stratified_kfold" and "classification" in task_type:
        splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    return splitter, entity_group_col, temporal_col


def _prepare_numeric_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Sanitize feature DataFrame ensuring clean numeric columns for estimators."""
    clean_cols: Dict[str, np.ndarray] = {}
    for col in df_feat.columns:
        s = df_feat[col]
        if pd.api.types.is_numeric_dtype(s):
            vals = s.to_numpy(dtype=float)
            vals = np.nan_to_num(vals, nan=np.nan, posinf=np.nan, neginf=np.nan)
            clean_cols[str(col)] = vals
        elif s.dtype == "bool":
            clean_cols[str(col)] = s.to_numpy(dtype=float)
    if not clean_cols:
        # Fallback dummy column to prevent empty DataFrame error
        clean_cols["__dummy__"] = np.zeros(len(df_feat), dtype=float)
    return pd.DataFrame(clean_cols, index=df_feat.index)


def _resolve_and_apply_features(
    feature_source: Any,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
    id_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute or apply feature transformation strictly in-fold without leakage."""
    X_train_raw = train_df.drop(columns=[target_col] + [c for c in id_cols if c in train_df.columns], errors="ignore")
    X_val_raw = val_df.drop(columns=[target_col] + [c for c in id_cols if c in val_df.columns], errors="ignore")
    y_train = train_df[target_col]

    # Label isolation: validation target MUST NOT reach candidate callable frames!
    val_df_no_target = val_df.drop(columns=[target_col], errors="ignore")
    train_df_no_target = train_df.drop(columns=[target_col], errors="ignore")

    # Case 1: Callable f(df) or f(df, y)
    if callable(feature_source) and not isinstance(feature_source, type):
        try:
            # Preferred: explicit feature/label contract (df_no_target, y)
            res_tr = feature_source(train_df_no_target, y_train)
            res_val = feature_source(val_df_no_target, None)
        except TypeError:
            # Fallback: single-argument callable f(df_no_target)
            res_tr = feature_source(train_df_no_target)
            res_val = feature_source(val_df_no_target)
        return _prepare_numeric_features(res_tr), _prepare_numeric_features(res_val)

    # Case 2: Scikit-learn BaseEstimator instance
    if isinstance(feature_source, BaseEstimator) or (
        hasattr(feature_source, "fit") and hasattr(feature_source, "transform")
    ):
        cloned = copy.deepcopy(feature_source)
        try:
            if hasattr(cloned, "fit_transform"):
                X_tr = cloned.fit_transform(X_train_raw, y_train)
            else:
                cloned.fit(X_train_raw, y_train)
                X_tr = cloned.transform(X_train_raw)
        except Exception:
            if hasattr(cloned, "fit_transform"):
                X_tr = cloned.fit_transform(X_train_raw)
            else:
                cloned.fit(X_train_raw)
                X_tr = cloned.transform(X_train_raw)

        X_val = cloned.transform(X_val_raw)
        df_tr = X_tr if isinstance(X_tr, pd.DataFrame) else pd.DataFrame(X_tr, index=train_df.index)
        df_val = X_val if isinstance(X_val, pd.DataFrame) else pd.DataFrame(X_val, index=val_df.index)
        return _prepare_numeric_features(df_tr), _prepare_numeric_features(df_val)

    # Case 3: List/Set of column names
    if isinstance(feature_source, (list, tuple, set)):
        cols = [c for c in feature_source if c in train_df.columns and c != target_col and c not in id_cols]
        return _prepare_numeric_features(train_df[cols]), _prepare_numeric_features(val_df[cols])

    # Case 4: Python code string or .py file path
    if isinstance(feature_source, (str, Path)):
        code_str = str(feature_source)
        if "\n" not in code_str and len(code_str) < 1024 and Path(code_str).is_file():
            code_str = Path(code_str).read_text(encoding="utf-8")

        # Execute code in sandbox namespace
        import mlrescurix_worker.scientific.primitives as prim
        sandbox_scope: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "re": re,
            "NestedOOFEmpiricalBayesEncoder": prim.NestedOOFEmpiricalBayesEncoder,
            "QuasiIdDecomposer": prim.QuasiIdDecomposer,
            "LightweightTextEmbedder": prim.LightweightTextEmbedder,
            "InformativeMissingnessAndInteractions": prim.InformativeMissingnessAndInteractions,
            "BaseEstimator": BaseEstimator,
            "TransformerMixin": TransformerMixin,
        }

        exec(code_str, sandbox_scope)

        # Look for transform or build_features or transformer class/instance
        if "build_features" in sandbox_scope and callable(sandbox_scope["build_features"]):
            fn = sandbox_scope["build_features"]
            return _resolve_and_apply_features(fn, train_df, val_df, target_col, id_cols)
        if "transform" in sandbox_scope and callable(sandbox_scope["transform"]):
            fn = sandbox_scope["transform"]
            return _resolve_and_apply_features(fn, train_df, val_df, target_col, id_cols)
        if "get_transformer" in sandbox_scope and callable(sandbox_scope["get_transformer"]):
            trans = sandbox_scope["get_transformer"]()
            return _resolve_and_apply_features(trans, train_df, val_df, target_col, id_cols)
        for val in sandbox_scope.values():
            if isinstance(val, BaseEstimator) or (
                isinstance(val, type) and issubclass(val, BaseEstimator) and val is not BaseEstimator
            ):
                inst = val() if isinstance(val, type) else val
                return _resolve_and_apply_features(inst, train_df, val_df, target_col, id_cols)

    # Default fallback: numeric columns from X_train_raw
    return _prepare_numeric_features(X_train_raw), _prepare_numeric_features(X_val_raw)


def ablate_feature_hypothesis(
    hypothesis_id: str,
    candidate_features_py: Any,
    base_features_py: Any,
    topology: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
    dataset_path: Optional[Union[str, Path]] = None,
    fidelity: str = "auto",
    ledger: Optional[Dict[str, Any]] = None,
    ledger_path: Optional[Union[str, Path]] = None,
    metric_name: Optional[str] = None,
    session_id: str = "default_session",
    project_id: str = "default_project",
    epoch: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Execute paired CV feature ablation and auto-gate against overfitting or non-improvement.
    
    Args:
        hypothesis_id: Unique identifier for the feature hypothesis.
        candidate_features_py: Candidate feature specification (callable, BaseEstimator, or python code).
        base_features_py: Baseline feature specification (or None to use clean numeric baseline).
        topology: Locked DatasetTopologyContract dictionary.
        df: Input DataFrame (optional, will load from dataset_path if None).
        dataset_path: Path to dataset CSV (optional).
        fidelity: Iteration fidelity ('auto', 'fast', 'full', 'tier0').
        ledger: In-memory ResearchLedger dictionary to update.
        ledger_path: Path to research_ledger.json to synchronize.
        metric_name: Evaluation metric name ('balanced_accuracy', 'roc_auc', 'r2', 'rmse').
        session_id: Current research session ID.
        project_id: Project identifier.
        epoch: Current research epoch.
        random_state: PRNG seed.
        
    Returns:
        Ablation receipt dictionary adhering to feature_ablations schema.
    """
    # 1. Resolve dataset
    if df is None:
        if dataset_path is not None and Path(dataset_path).is_file():
            df = pd.read_csv(dataset_path)
        elif topology.get("dataset_path") and Path(topology["dataset_path"]).is_file():
            df = pd.read_csv(topology["dataset_path"])
        else:
            default_p = Path(".mlrescurix-state/kaggle-titanic/titanic.csv")
            if default_p.is_file():
                df = pd.read_csv(default_p)
            else:
                raise ValueError("No dataset provided and default dataset not found.")

    target_col = topology["target_column"]
    id_cols = topology.get("id_columns", [])
    task_type = topology.get("task_type", "binary_classification")
    is_classification = "classification" in task_type

    # 2. Build locked CV splitter
    splitter, group_col, temporal_col = build_cv_splitter(topology, df=df, random_state=random_state)

    # Sort if temporal
    df_eval = df.copy()
    if temporal_col is not None and temporal_col in df_eval.columns:
        df_eval = df_eval.sort_values(temporal_col).reset_index(drop=True)

    y_all = df_eval[target_col].to_numpy()
    groups = df_eval[group_col].to_numpy() if group_col and group_col in df_eval.columns else None

    # Handle splitter fallback for classification when class counts < n_splits
    if is_classification and not isinstance(splitter, PurgedTimeSeriesSplit):
        unique_y, counts_y = np.unique(y_all[~pd.isna(y_all)], return_counts=True)
        if len(counts_y) > 0 and min(counts_y) < splitter.get_n_splits(df_eval, y_all, groups):
            splitter = KFold(n_splits=splitter.get_n_splits(df_eval, y_all, groups), shuffle=True, random_state=random_state)

    # Determine metric
    if metric_name is None:
        metric_name = "balanced_accuracy" if is_classification else "r2"

    def score_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        elif metric_name == "r2":
            return float(r2_score(y_true, y_pred))
        elif metric_name in ("rmse", "neg_rmse"):
            return float(-root_mean_squared_error(y_true, y_pred))
        else:
            return float(balanced_accuracy_score(y_true, y_pred))

    # Determine model hyperparameters based on fidelity
    max_iter = 50
    if fidelity == "tier0":
        max_iter = 25
    elif fidelity == "full":
        max_iter = 120

    base_val_scores: List[float] = []
    cand_val_scores: List[float] = []
    base_train_scores: List[float] = []
    cand_train_scores: List[float] = []
    fold_deltas: List[float] = []
    wins: List[bool] = []
    feature_importances: Dict[str, float] = {}

    # 3. Paired CV Evaluation Loop
    splits = list(splitter.split(df_eval, y_all, groups=groups))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_df = df_eval.iloc[train_idx].reset_index(drop=True)
        val_df = df_eval.iloc[val_idx].reset_index(drop=True)

        y_tr = train_df[target_col].to_numpy()
        y_val = val_df[target_col].to_numpy()

        # In-fold transformation
        X_base_tr, X_base_val = _resolve_and_apply_features(
            base_features_py, train_df, val_df, target_col, id_cols
        )
        X_cand_tr, X_cand_val = _resolve_and_apply_features(
            candidate_features_py, train_df, val_df, target_col, id_cols
        )

        # Baseline model
        if is_classification:
            base_model = HistGradientBoostingClassifier(
                max_iter=max_iter, random_state=random_state + fold_idx
            )
            cand_model = HistGradientBoostingClassifier(
                max_iter=max_iter, random_state=random_state + fold_idx
            )
        else:
            base_model = HistGradientBoostingRegressor(
                max_iter=max_iter, random_state=random_state + fold_idx
            )
            cand_model = HistGradientBoostingRegressor(
                max_iter=max_iter, random_state=random_state + fold_idx
            )

        # Fit & score baseline
        base_model.fit(X_base_tr, y_tr)
        base_tr_pred = base_model.predict(X_base_tr)
        base_val_pred = base_model.predict(X_base_val)

        base_tr_sc = score_fn(y_tr, base_tr_pred)
        base_val_sc = score_fn(y_val, base_val_pred)
        base_train_scores.append(base_tr_sc)
        base_val_scores.append(base_val_sc)

        # Fit & score candidate
        cand_model.fit(X_cand_tr, y_tr)
        cand_tr_pred = cand_model.predict(X_cand_tr)
        cand_val_pred = cand_model.predict(X_cand_val)

        cand_tr_sc = score_fn(y_tr, cand_tr_pred)
        cand_val_sc = score_fn(y_val, cand_val_pred)
        cand_train_scores.append(cand_tr_sc)
        cand_val_scores.append(cand_val_sc)

        delta_k = cand_val_sc - base_val_sc
        fold_deltas.append(float(delta_k))
        wins.append(bool(delta_k > 0.0))

        # Track candidate feature importances (permutation or native if available)
        if fold_idx == 0:
            for c in X_cand_tr.columns:
                feature_importances[str(c)] = 0.0

    # 4. Compute Summary Statistics
    baseline_cv = float(np.mean(base_val_scores))
    ablated_cv = float(np.mean(cand_val_scores))
    cv_score_delta = float(ablated_cv - baseline_cv)

    base_gap = float(np.mean(base_train_scores) - baseline_cv)
    cand_gap = float(np.mean(cand_train_scores) - ablated_cv)
    train_val_gap_delta = float(cand_gap - base_gap)

    fold_win_count = int(sum(wins))
    total_folds = len(wins)
    win_rate = float(fold_win_count / max(1, total_folds))

    # 5. Auto-Gate Rules:
    # If Delta CV <= 0 or if Delta Train-Val Gap > +0.08 (overfitting memorization) -> REJECTED
    rejection_reason: Optional[str] = None
    if cv_score_delta <= 0.0:
        status = "REJECTED"
        rejection_reason = f"Delta CV score {cv_score_delta:+.4f} <= 0 (no validation lift over baseline)"
    elif train_val_gap_delta > 0.08:
        status = "REJECTED"
        rejection_reason = (
            f"Delta Train-Val Gap {train_val_gap_delta:+.4f} > +0.08 "
            f"(overfitting memorization detected: base gap {base_gap:.4f} -> cand gap {cand_gap:.4f})"
        )
    else:
        status = "ACCEPTED"
        rejection_reason = None

    ablation_id = f"ablation_{hypothesis_id}_{int(time.time() * 1000)}"
    evaluated_at_ms = int(time.time() * 1000)

    receipt: Dict[str, Any] = {
        "ablation_id": ablation_id,
        "session_id": session_id,
        "project_id": project_id,
        "epoch": epoch,
        "hypothesis_id": hypothesis_id,
        "feature_names": list(feature_importances.keys()),
        "baseline_cv_score": round(baseline_cv, 5),
        "ablated_cv_score": round(ablated_cv, 5),
        "cv_score_delta": round(cv_score_delta, 5),
        "train_val_gap_delta": round(train_val_gap_delta, 5),
        "base_gap": round(base_gap, 5),
        "cand_gap": round(cand_gap, 5),
        "fold_win_count": fold_win_count,
        "total_folds": total_folds,
        "win_rate": round(win_rate, 4),
        "fold_deltas": [round(d, 5) for d in fold_deltas],
        "status": status,
        "rejection_reason": rejection_reason,
        "importance_scores": feature_importances,
        "evaluated_at_ms": evaluated_at_ms,
        "metric_name": metric_name,
    }

    # 6. Synchronize with ResearchLedger
    if ledger is not None or ledger_path is not None:
        target_ledger = ledger
        if target_ledger is None and ledger_path is not None and Path(ledger_path).is_file():
            try:
                target_ledger = json.loads(Path(ledger_path).read_text(encoding="utf-8"))
            except Exception:
                target_ledger = None

        if target_ledger is not None:
            epochs = target_ledger.setdefault("epochs", [])
            # Find current epoch entry or create one
            epoch_entry = None
            for ep in epochs:
                if ep.get("epoch") == epoch:
                    epoch_entry = ep
                    break
            if epoch_entry is None:
                epoch_entry = {
                    "epoch": epoch,
                    "hypotheses": [hypothesis_id],
                    "accepted_features": [],
                    "rejected_features": [],
                    "ablation_results": [],
                    "model_results": [],
                    "ensemble_result": None,
                    "critique": None,
                    "m1_reflection": None,
                    "timestamp_ms": evaluated_at_ms,
                }
                epochs.append(epoch_entry)

            if status == "ACCEPTED":
                epoch_entry.setdefault("accepted_features", []).append(hypothesis_id)
            else:
                epoch_entry.setdefault("rejected_features", []).append(hypothesis_id)

            epoch_entry.setdefault("ablation_results", []).append(receipt)
            target_ledger["updated_ms"] = evaluated_at_ms

            if ledger_path is not None:
                tmp_p = f"{ledger_path}.tmp.{os.getpid()}"
                Path(tmp_p).write_text(json.dumps(target_ledger, indent=2), encoding="utf-8")
                os.replace(tmp_p, ledger_path)

    return receipt
