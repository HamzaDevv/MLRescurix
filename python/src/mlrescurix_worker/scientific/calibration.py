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
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unsupported calibration method: {method}. Must be 'isotonic' or 'sigmoid'.")
        self.method = method
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
