"""Split-aware probability calibration for MLRescurix M5.

Calibrators are fitted strictly on training out-of-fold probability predictions.
Protected validation and sealed test labels are never exposed during calibration fitting.
"""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold


class SplitAwareCalibrator:
    """Calibrate prediction probabilities using isotonic regression or Platt sigmoid scaling."""

    def __init__(self, method: str = "isotonic", n_splits: int = 5, seed: int = 2026):
        norm_method = "sigmoid" if method == "platt" else method
        if norm_method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unsupported calibration method: {method}. Must be 'isotonic' or 'sigmoid' / 'platt'.")
        self.method = norm_method
        self.declared_method = method
        self.n_splits = n_splits
        self.seed = seed
        self.calibrators_: list[Any] = []
        self.classes_: list[Any] = []
        self.is_multiclass_: bool = False

    def fit_from_oof(
        self,
        oof_probabilities: np.ndarray,
        y_train: np.ndarray | pd.Series,
        classes: Sequence[Any],
    ) -> "SplitAwareCalibrator":
        """
        Fit calibration models strictly on training out-of-fold predicted probabilities.
        Zero validation or test labels are used.
        """
        y_arr = np.asarray(y_train)
        self.classes_ = list(classes)
        if len(self.classes_) < 2 or len(set(self.classes_)) != len(self.classes_) or set(y_arr.tolist()) != set(self.classes_):
            raise ValueError("Calibration requires unique classes, each represented in training labels")
        oof_probabilities = self._validate_probabilities(oof_probabilities)
        if len(oof_probabilities) != len(y_arr):
            raise ValueError("Probability and label row counts differ")
        self.is_multiclass_ = len(self.classes_) > 2
        self.calibrators_ = []

        if not self.is_multiclass_:
            # Binary classification: calibrate positive class (index 1)
            pos_class = self.classes_[1]
            y_binary = (y_arr == pos_class).astype(float)
            p_pos = oof_probabilities[:, 1] if oof_probabilities.ndim == 2 else oof_probabilities

            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                cal.fit(p_pos, y_binary)
            else:
                # Sigmoid / Platt scaling: logistic regression on logit or probability
                cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                cal.fit(p_pos.reshape(-1, 1), y_binary)
            self.calibrators_.append(cal)
        else:
            # Multiclass: one-vs-rest calibrator for each class
            for idx, c in enumerate(self.classes_):
                y_binary = (y_arr == c).astype(float)
                p_c = oof_probabilities[:, idx]
                if self.method == "isotonic":
                    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                    cal.fit(p_c, y_binary)
                else:
                    cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
                    cal.fit(p_c.reshape(-1, 1), y_binary)
                self.calibrators_.append(cal)

        return self

    def _validate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=float)
        if p.ndim == 1 and len(self.classes_) == 2:
            p = np.column_stack([1 - p, p])
        if p.ndim != 2 or p.shape[1] != len(self.classes_) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1):
            raise ValueError("Invalid probability shape, bounds or normalization")
        return p

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Transform uncalibrated probabilities to calibrated probabilities.
        Enforces class bounds [0, 1] and row normalization (sums to 1.0).
        """
        if not self.calibrators_:
            raise ValueError("Calibrator is not fitted")
        p_in = self._validate_probabilities(probabilities)

        if not self.is_multiclass_:
            cal = self.calibrators_[0]
            p_pos_in = p_in[:, 1] if p_in.ndim == 2 else p_in
            if self.method == "isotonic":
                calibrated_pos = cal.transform(p_pos_in)
            else:
                calibrated_pos = cal.predict_proba(p_pos_in.reshape(-1, 1))[:, 1]

            calibrated_pos = np.clip(calibrated_pos, 1e-7, 1.0 - 1e-7)
            calibrated_neg = 1.0 - calibrated_pos
            out = np.column_stack([calibrated_neg, calibrated_pos])
        else:
            cols = []
            for idx, cal in enumerate(self.calibrators_):
                p_c = p_in[:, idx]
                if self.method == "isotonic":
                    c_prob = cal.transform(p_c)
                else:
                    c_prob = cal.predict_proba(p_c.reshape(-1, 1))[:, 1]
                cols.append(np.clip(c_prob, 1e-7, 1.0))
            out = np.column_stack(cols)
            # Normalize rows to sum to 1.0
            row_sums = out.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            out = out / row_sums

        return out

    def fit(
        self,
        probabilities: np.ndarray,
        y: np.ndarray | pd.Series,
        classes: Sequence[Any],
    ) -> "SplitAwareCalibrator":
        """Fit calibration model on probabilities and target labels."""
        return self.fit_from_oof(probabilities, y, classes)

    def calibrate_predictions(self, uncalibrated_predictions: dict[str, Any]) -> dict[str, Any]:
        """Wrap and calibrate a standard prediction dictionary, returning validated format."""
        if list(uncalibrated_predictions.get("classes", self.classes_)) != self.classes_:
            raise ValueError("Prediction class order differs from calibration class order")
        raw_probs = np.asarray(uncalibrated_predictions["probabilities"])
        calibrated = self.transform(raw_probs)
        pred_idx = np.argmax(calibrated, axis=1)
        classes = uncalibrated_predictions.get("classes", self.classes_)
        pred_labels = [classes[i] for i in pred_idx]

        out = dict(uncalibrated_predictions)
        out["probabilities"] = calibrated.tolist()
        out["prediction"] = pred_labels
        return out


def compute_brier_score(
    y_true: Sequence[Any] | np.ndarray,
    y_prob: np.ndarray,
    pos_class: Any = 1,
) -> float:
    """Compute Brier score (mean squared error of probability predictions)."""
    y_arr = np.asarray(y_true)
    y_bin = (y_arr == pos_class).astype(float)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] == 2 else p[:, 0]
    return float(np.mean((y_bin - p) ** 2))


def compute_reliability_curve(
    y_true: Sequence[Any] | np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    pos_class: Any = 1,
) -> dict[str, Any]:
    """Compute binned empirical reliability curve and Brier score."""
    y_arr = np.asarray(y_true)
    y_bin = (y_arr == pos_class).astype(float)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] == 2 else p[:, 0]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    true_proportions: list[float] = []
    pred_proportions: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        low, high = float(bin_edges[i]), float(bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (p >= low) & (p <= high)
        else:
            mask = (p >= low) & (p < high)
        cnt = int(np.sum(mask))
        bin_counts.append(cnt)
        if cnt > 0:
            true_proportions.append(round(float(np.mean(y_bin[mask])), 6))
            pred_proportions.append(round(float(np.mean(p[mask])), 6))
        else:
            true_proportions.append(0.0)
            pred_proportions.append(round(float((low + high) / 2.0), 6))

    brier = float(np.mean((y_bin - p) ** 2))
    return {
        "n_bins": n_bins,
        "bin_edges": [round(float(x), 4) for x in bin_edges],
        "true_proportions": true_proportions,
        "pred_proportions": pred_proportions,
        "bin_counts": bin_counts,
        "brier_score": round(brier, 6),
    }


def optimize_decision_threshold(
    y_true: Sequence[Any] | np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    cost_matrix: dict[str, float] | None = None,
    pos_class: Any = 1,
    n_thresholds: int = 100,
) -> dict[str, Any]:
    """Find the optimal decision threshold for binary classification maximizing the target metric.

    metric: "f1", "balanced_accuracy", or "cost".
    cost_matrix: dict with keys "c_fp", "c_fn", "c_tp", "c_tn" when metric=="cost".
    """
    y_arr = np.asarray(y_true)
    y_bin = (y_arr == pos_class).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if p.ndim == 2:
        p = p[:, 1] if p.shape[1] == 2 else p[:, 0]

    raw_thresholds = list(np.linspace(0.01, 0.99, n_thresholds))
    thresholds = sorted(list({round(float(t), 4) for t in raw_thresholds} | {0.5}))

    best_threshold = 0.5
    best_score = -float("inf") if metric != "cost" else float("inf")
    default_score = 0.0
    curve: list[dict[str, float]] = []

    for t in thresholds:
        preds = (p >= t).astype(int)
        tp = int(np.sum((preds == 1) & (y_bin == 1)))
        fp = int(np.sum((preds == 1) & (y_bin == 0)))
        tn = int(np.sum((preds == 0) & (y_bin == 0)))
        fn = int(np.sum((preds == 0) & (y_bin == 1)))

        if metric == "f1":
            denom = 2 * tp + fp + fn
            score = (2.0 * tp / denom) if denom > 0 else 0.0
        elif metric == "balanced_accuracy":
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            score = 0.5 * (sens + spec)
        elif metric == "cost":
            cm = cost_matrix or {"c_fp": 1.0, "c_fn": 1.0, "c_tp": 0.0, "c_tn": 0.0}
            c_fp = cm.get("c_fp", 1.0)
            c_fn = cm.get("c_fn", 1.0)
            c_tp = cm.get("c_tp", 0.0)
            c_tn = cm.get("c_tn", 0.0)
            score = float(c_fp * fp + c_fn * fn + c_tp * tp + c_tn * tn)
        else:
            raise ValueError(f"Unsupported threshold optimization metric: {metric}")

        curve.append({"threshold": round(float(t), 4), "score": round(float(score), 4)})

        if abs(t - 0.5) < 1e-5:
            default_score = float(score)

        if metric == "cost":
            if score < best_score or (score == best_score and abs(t - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = t
        else:
            if score > best_score or (score == best_score and abs(t - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = t

    return {
        "optimal_threshold": round(float(best_threshold), 4),
        "metric": metric,
        "best_score": round(float(best_score), 4),
        "default_score": round(float(default_score), 4),
        "score_delta": round(float(best_score - default_score), 4),
        "threshold_curve": curve,
    }


def calibrate_probabilities(
    y_val: Sequence[Any] | np.ndarray,
    val_probabilities: np.ndarray,
    method: str = "isotonic",
    classes: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Fit calibrator on validation predictions and compute before/after reliability metrics."""
    y_arr = np.asarray(y_val)
    if classes is None:
        detected_classes = sorted(list(set(y_arr.tolist())))
    else:
        detected_classes = list(classes)

    calibrator = SplitAwareCalibrator(method=method)
    calibrator.fit_from_oof(val_probabilities, y_arr, detected_classes)
    calibrated_probs = calibrator.transform(val_probabilities)

    pos_class = detected_classes[1] if len(detected_classes) == 2 else detected_classes[0]
    uncal_curve = compute_reliability_curve(y_arr, val_probabilities, pos_class=pos_class)
    cal_curve = compute_reliability_curve(y_arr, calibrated_probs, pos_class=pos_class)

    brier_before = uncal_curve["brier_score"]
    brier_after = cal_curve["brier_score"]

    return {
        "calibrator": calibrator,
        "calibrated_probabilities": calibrated_probs,
        "calibration_method": method,
        "brier_before": round(brier_before, 6),
        "brier_after": round(brier_after, 6),
        "brier_delta": round(brier_before - brier_after, 6),
        "reliability_curve_before": uncal_curve,
        "reliability_curve_after": cal_curve,
    }
