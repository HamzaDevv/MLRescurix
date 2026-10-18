"""Error Forensics Analyzer & Worst Data Slice Extractor (Phase 4).

Conforms strictly to `contracts/v2/error_forensics.schema.json`.
Provides:
1. Extraction of the Top 3 Worst-Performing Data Slices by fitting a shallow depth-3
   surrogate decision tree on out-of-fold (OOF) absolute prediction errors.
2. Feature split-gain inequality concentration (Gini / Herfindahl index) across features.
3. Train-vs-validation generalization gap reporting.
4. Structured error diagnostics conforming to Schema v2.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, mean_absolute_error, root_mean_squared_error
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path("contracts/v2/error_forensics.schema.json")
_CACHED_SCHEMA: Optional[Dict[str, Any]] = None


def get_error_forensics_schema() -> Dict[str, Any]:
    """Load and cache contracts/v2/error_forensics.schema.json."""
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        p = SCHEMA_PATH
        if not p.is_file():
            # Search parent directories
            for parent in Path(__file__).resolve().parents:
                cand = parent / "contracts" / "v2" / "error_forensics.schema.json"
                if cand.is_file():
                    p = cand
                    break
        if not p.is_file():
            raise FileNotFoundError(f"error_forensics.schema.json not found at {SCHEMA_PATH}")
        with open(p, "r", encoding="utf-8") as f:
            _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA


def compute_feature_gini(importances: Union[Sequence[float], np.ndarray]) -> float:
    """Calculate the Gini concentration coefficient of feature importances.

    Measures inequality of feature split-gains across the feature portfolio:
    - 0.0: Perfectly uniform distribution across all features.
    - 1.0: Complete concentration in a single dominant feature.

    Bounded in [0.0, 1.0].
    """
    x = np.asarray(importances, dtype=float)
    x = np.maximum(0.0, x)
    total = float(np.sum(x))
    n = len(x)
    if total <= 1e-12 or n <= 1:
        return 0.0

    sorted_x = np.sort(x)
    index = np.arange(1, n + 1)
    gini = float(np.sum((2 * index - n - 1) * sorted_x) / (n * total))

    # Small-sample bias correction factor n / (n - 1) so single-feature concentration equals 1.0
    if n > 1:
        gini = gini * (n / (n - 1))

    return float(np.clip(gini, 0.0, 1.0))


@dataclass
class WorstSliceRecord:
    slice_rule: str
    sample_count: int
    slice_error_rate: float
    global_error_rate: float
    lift: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_rule": self.slice_rule,
            "sample_count": int(self.sample_count),
            "slice_error_rate": float(np.clip(self.slice_error_rate, 0.0, 1.0)),
            "global_error_rate": float(np.clip(self.global_error_rate, 0.0, 1.0)),
            "lift": round(float(self.lift), 4),
        }


@dataclass
class FeatureImportanceRecord:
    feature: str
    importance: float
    origin: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "feature": str(self.feature),
            "importance": round(float(self.importance), 6),
        }
        if self.origin is not None:
            res["origin"] = str(self.origin)
        else:
            res["origin"] = None
        return res


@dataclass
class ErrorForensicsReport:
    schema_version: str
    forensics_id: str
    candidate_id: str
    model_family: str
    cv_scores: List[float]
    mean_cv_score: float
    train_val_gap: float
    worst_slices: List[WorstSliceRecord]
    feature_importance_gini: float
    top_features_by_importance: List[FeatureImportanceRecord]
    timestamp_ms: int
    confusion_or_residual_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "forensics_id": self.forensics_id,
            "candidate_id": self.candidate_id,
            "model_family": self.model_family,
            "cv_scores": [round(float(s), 6) for s in self.cv_scores],
            "mean_cv_score": round(float(self.mean_cv_score), 6),
            "train_val_gap": round(float(self.train_val_gap), 6),
            "worst_slices": [s.to_dict() for s in self.worst_slices],
            "feature_importance_gini": round(float(np.clip(self.feature_importance_gini, 0.0, 1.0)), 6),
            "top_features_by_importance": [f.to_dict() for f in self.top_features_by_importance],
            "timestamp_ms": int(self.timestamp_ms),
        }
        if self.confusion_or_residual_summary is not None:
            res["confusion_or_residual_summary"] = self.confusion_or_residual_summary
        return res

    def validate_schema(self) -> None:
        """Validate this report dictionary against contracts/v2/error_forensics.schema.json."""
        try:
            import jsonschema
            schema = get_error_forensics_schema()
            jsonschema.validate(instance=self.to_dict(), schema=schema)
        except ImportError:
            logger.warning("jsonschema not installed; skipping strict validation.")


class ErrorForensicsAnalyzer:
    """Surrogate decision-tree failure forensics analyzer."""

    @classmethod
    def analyze(
        cls,
        candidate_id: str,
        model_family: str,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        oof_predictions: np.ndarray,
        cv_scores: Sequence[float],
        mean_cv_score: float,
        train_val_gap: float,
        oof_probabilities: Optional[np.ndarray] = None,
        feature_importances: Optional[Union[Sequence[float], np.ndarray]] = None,
        feature_names: Optional[Sequence[str]] = None,
        task_type: str = "binary_classification",
        feature_origins: Optional[Dict[str, str]] = None,
        max_depth: int = 3,
        min_slice_support: Optional[int] = None,
    ) -> ErrorForensicsReport:
        """Extract top worst data slices and split-gain concentration diagnostics."""
        timestamp_ms = int(time.time() * 1000)
        forensics_id = f"forensics_{uuid.uuid4().hex[:12]}"
        is_classification = "classification" in task_type or task_type in ("binary", "multiclass")

        y_arr = np.asarray(y)
        oof_pred_arr = np.asarray(oof_predictions)
        n_samples = len(y_arr)

        # 1. Compute per-instance binary error indicator
        if is_classification:
            errors = (y_arr != oof_pred_arr).astype(int)
        else:
            residuals = np.abs(y_arr.astype(float) - oof_pred_arr.astype(float))
            # Define error indicator as exceeding the 75th percentile / median residual
            thresh = float(np.percentile(residuals, 75)) if len(residuals) > 0 else 0.0
            errors = (residuals > thresh).astype(int)

        global_error_rate = float(np.mean(errors)) if n_samples > 0 else 0.0

        # 2. Prepare feature matrix for surrogate decision tree
        if isinstance(X, pd.DataFrame):
            col_names = list(X.columns)
            # Convert non-numeric columns
            clean_cols: Dict[str, np.ndarray] = {}
            for col in col_names:
                s = X[col]
                if pd.api.types.is_numeric_dtype(s):
                    clean_cols[col] = s.to_numpy(dtype=float)
                elif s.dtype == "bool":
                    clean_cols[col] = s.to_numpy(dtype=float)
                else:
                    # Factorize categorical strings
                    codes, _ = pd.factorize(s)
                    clean_cols[col] = codes.astype(float)
            X_clean_df = pd.DataFrame(clean_cols, index=X.index)
            X_mat = X_clean_df.to_numpy(dtype=float)
        else:
            col_names = list(feature_names) if feature_names else [f"f_{i}" for i in range(X.shape[1])]
            X_mat = np.asarray(X, dtype=float)

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_mat)

        # 3. Fit shallow depth-3 surrogate decision tree on prediction errors
        min_support = min_slice_support if min_slice_support is not None else max(5, int(0.02 * n_samples))
        min_support = max(1, min(min_support, max(1, n_samples // 4)))

        worst_slices: List[WorstSliceRecord] = []
        if n_samples >= 5 and len(np.unique(errors)) > 1:
            surrogate_tree = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=min_support,
                random_state=42,
            )
            surrogate_tree.fit(X_imputed, errors)

            # Traverse tree leaves to extract failure rules
            tree_ = surrogate_tree.tree_
            feature = tree_.feature
            threshold = tree_.threshold
            children_left = tree_.children_left
            children_right = tree_.children_right
            value = tree_.value
            n_node_samples = tree_.n_node_samples

            discovered_slices: List[WorstSliceRecord] = []

            def _dfs(node_id: int, conditions: List[str]) -> None:
                left = children_left[node_id]
                right = children_right[node_id]
                if left == right:  # Leaf node
                    cnt = int(n_node_samples[node_id])
                    class_vals = value[node_id][0]
                    val_sum = float(np.sum(class_vals))
                    if val_sum > 0:
                        slice_err = float(class_vals[1] / val_sum) if len(class_vals) > 1 else 0.0
                    else:
                        slice_err = 0.0
                    lift = float(slice_err / max(global_error_rate, 1e-6))
                    cond_str = " AND ".join(conditions) if conditions else "GLOBAL"
                    discovered_slices.append(WorstSliceRecord(
                        slice_rule=f"{cond_str} -> Error Rate {slice_err * 100:.1f}%",
                        sample_count=cnt,
                        slice_error_rate=float(np.clip(slice_err, 0.0, 1.0)),
                        global_error_rate=float(np.clip(global_error_rate, 0.0, 1.0)),
                        lift=lift,
                    ))
                    return

                feat_idx = feature[node_id]
                fname = col_names[feat_idx]
                thresh_val = threshold[node_id]

                # Distinguish boolean/binary indicator conditions
                col_vals = X_imputed[:, feat_idx]
                is_binary = np.all(np.isin(col_vals, [0.0, 1.0]))
                if is_binary and 0.0 < thresh_val < 1.0:
                    left_cond = f"{fname} == 0"
                    right_cond = f"{fname} == 1"
                else:
                    left_cond = f"{fname} <= {thresh_val:.2f}"
                    right_cond = f"{fname} > {thresh_val:.2f}"

                _dfs(left, conditions + [left_cond])
                _dfs(right, conditions + [right_cond])

            _dfs(0, [])

            # Filter slices by support, sort by error rate descending then lift descending
            viable = [s for s in discovered_slices if s.sample_count >= min_support]
            if not viable:
                viable = discovered_slices
            viable.sort(key=lambda s: (s.slice_error_rate, s.lift, s.sample_count), reverse=True)
            worst_slices = viable[:3]

        if not worst_slices:
            worst_slices = [WorstSliceRecord(
                slice_rule=f"GLOBAL -> Error Rate {global_error_rate * 100:.1f}%",
                sample_count=n_samples,
                slice_error_rate=global_error_rate,
                global_error_rate=global_error_rate,
                lift=1.0,
            )]

        # 4. Feature importance Gini coefficient & Top features
        if feature_importances is not None and len(feature_importances) > 0:
            raw_imps = np.asarray(feature_importances, dtype=float)
            gini = compute_feature_gini(raw_imps)
            sorted_indices = np.argsort(raw_imps)[::-1]
            top_feats: List[FeatureImportanceRecord] = []
            for idx in sorted_indices[:20]:
                fname = col_names[idx] if idx < len(col_names) else f"f_{idx}"
                origin = feature_origins.get(fname) if feature_origins else None
                top_feats.append(FeatureImportanceRecord(
                    feature=fname,
                    importance=float(raw_imps[idx]),
                    origin=origin,
                ))
        else:
            gini = 0.0
            top_feats = [
                FeatureImportanceRecord(
                    feature=col_names[i],
                    importance=1.0 / len(col_names),
                    origin=feature_origins.get(col_names[i]) if feature_origins else None,
                )
                for i in range(min(20, len(col_names)))
            ]

        # 5. Confusion matrix or residual summary
        confusion_or_residual: Dict[str, Any] = {}
        if is_classification:
            try:
                unique_classes = np.unique(y_arr)
                if len(unique_classes) == 2:
                    tn, fp, fn, tp = confusion_matrix(y_arr, oof_pred_arr, labels=unique_classes).ravel()
                    confusion_or_residual = {
                        "true_negatives": int(tn),
                        "false_positives": int(fp),
                        "false_negatives": int(fn),
                        "true_positives": int(tp),
                        "global_error_rate": float(global_error_rate),
                    }
                else:
                    cm = confusion_matrix(y_arr, oof_pred_arr)
                    confusion_or_residual = {
                        "confusion_matrix": cm.tolist(),
                        "global_error_rate": float(global_error_rate),
                    }
            except Exception as e:
                confusion_or_residual = {"error": str(e), "global_error_rate": float(global_error_rate)}
        else:
            res_arr = np.abs(y_arr.astype(float) - oof_pred_arr.astype(float))
            confusion_or_residual = {
                "mean_absolute_error": float(mean_absolute_error(y_arr, oof_pred_arr)),
                "root_mean_squared_error": float(root_mean_squared_error(y_arr, oof_pred_arr)),
                "p95_residual": float(np.percentile(res_arr, 95)) if len(res_arr) > 0 else 0.0,
                "max_residual": float(np.max(res_arr)) if len(res_arr) > 0 else 0.0,
            }

        report = ErrorForensicsReport(
            schema_version="v2",
            forensics_id=forensics_id,
            candidate_id=candidate_id,
            model_family=model_family,
            cv_scores=[float(s) for s in cv_scores],
            mean_cv_score=float(mean_cv_score),
            train_val_gap=float(train_val_gap),
            worst_slices=worst_slices,
            feature_importance_gini=float(gini),
            top_features_by_importance=top_feats,
            timestamp_ms=timestamp_ms,
            confusion_or_residual_summary=confusion_or_residual,
        )

        report.validate_schema()
        return report
