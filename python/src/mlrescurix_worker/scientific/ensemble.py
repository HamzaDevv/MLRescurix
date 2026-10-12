"""Constrained ensemble recipes (Voting/Averaging and Bounded Stacking) for MLRescurix M5.

Tracks explicit parent lineage and measures complete delivered pipeline inference cost.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge


@dataclass
class EnsembleSpec:
    ensemble_id: str
    ensemble_type: str  # "voting" or "stacking"
    parent_candidate_ids: list[str]
    weights: list[float] | None = None
    meta_learner: str = "linear"  # for stacking: "linear"
    task: str = "binary_classification"
    seed: int = 2026


class VotingEnsemble:
    """Weighted probability or prediction averaging with explicit parent lineage."""

    def __init__(self, parent_ids: list[str], weights: list[float] | None = None, task: str = "binary_classification"):
        self.parent_ids = list(parent_ids)
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("Parent candidate IDs must be unique")
        if not self.parent_ids:
            raise ValueError("Voting ensemble requires at least one parent candidate")
        if weights is None:
            self.weights = [1.0 / len(self.parent_ids)] * len(self.parent_ids)
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (len(self.parent_ids),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
                raise ValueError("Weights must match parents and be finite, nonnegative with positive sum")
            self.weights = (w / w.sum()).tolist()
        self.task = task
        self.models_: list[Any] = []
        self.classes_: list[Any] = []

    def load_parent_models(self, model_paths: list[Path | str]) -> "VotingEnsemble":
        if len(model_paths) != len(self.parent_ids):
            raise ValueError("Model count must match parent lineage")
        # Only trusted, verified artifacts may be deserialized; untrusted artifacts require sandboxing.
        self.models_ = [joblib.load(p) for p in model_paths]
        if self.task != "regression" and self.models_:
            first_model = self.models_[0]
            if hasattr(first_model, "classes_"):
                self.classes_ = first_model.classes_.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        """Inference across all parent models with measured total latency."""
        if len(self.models_) != len(self.parent_ids):
            raise ValueError("Parent models are not loaded")
        start = time.perf_counter_ns()
        if self.task == "regression":
            preds = np.zeros(len(X), dtype=float)
            for model, w in zip(self.models_, self.weights):
                p = np.asarray(model.predict(X), dtype=float)
                preds += w * p
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            return {"prediction": preds.tolist(), "inference_latency_ms": elapsed_ms}
        else:
            prob_sum = None
            for model, w in zip(self.models_, self.weights):
                if hasattr(model, "predict_proba"):
                    probs = np.asarray(model.predict_proba(X), dtype=float)
                    model_classes = list(model.classes_)
                    if len(model_classes) != len(self.classes_) or set(model_classes) != set(self.classes_):
                        raise ValueError("Parent class sets differ")
                    probs = probs[:, [model_classes.index(c) for c in self.classes_]]
                    if probs.shape != (len(X), len(self.classes_)) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(axis=1), 1):
                        raise ValueError("Invalid parent probabilities")
                else:
                    preds = model.predict(X)
                    # Convert hard predictions to one-hot probabilities
                    probs = np.zeros((len(X), len(self.classes_)), dtype=float)
                    for r_idx, label in enumerate(preds):
                        if label not in self.classes_:
                            raise ValueError("Unknown parent prediction label")
                        probs[r_idx, self.classes_.index(label)] = 1.0
                if prob_sum is None:
                    prob_sum = w * probs
                else:
                    prob_sum += w * probs

            # Row normalize probabilities
            row_sums = prob_sum.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            prob_norm = prob_sum / row_sums

            best_idx = np.argmax(prob_norm, axis=1)
            pred_labels = [self.classes_[i] for i in best_idx]
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

            return {
                "prediction": pred_labels,
                "probabilities": prob_norm.tolist(),
                "classes": self.classes_,
                "inference_latency_ms": elapsed_ms,
            }


class BoundedStackingEnsemble:
    """Two-level stacking ensemble with out-of-fold level-0 meta-features."""

    def __init__(
        self,
        parent_ids: list[str],
        task: str = "binary_classification",
        meta_learner: str = "linear",
        seed: int = 2026,
    ):
        self.parent_ids = list(parent_ids)
        self.task = task
        self.meta_learner = meta_learner
        self.seed = seed
        self.base_models_: list[Any] = []
        self.meta_model_: Any = None
        self.classes_: list[Any] = []

    def fit_meta_learner(
        self,
        base_model_paths: list[Path | str],
        oof_meta_features: np.ndarray,
        y_train: np.ndarray | pd.Series,
    ) -> "BoundedStackingEnsemble":
        """Fit level-1 meta model on training out-of-fold predictions."""
        if not self.parent_ids or len(set(self.parent_ids)) != len(self.parent_ids) or len(base_model_paths) != len(self.parent_ids):
            raise ValueError("Stacking requires unique parents and matching model count")
        if self.meta_learner != "linear":
            raise ValueError("Unsupported meta learner")
        self.base_models_ = [joblib.load(p) for p in base_model_paths]
        y_arr = np.asarray(y_train)
        oof_meta_features = np.asarray(oof_meta_features, dtype=float)
        expected_width = sum(1 if self.task == "regression" or len(m.classes_) == 2 else len(m.classes_) for m in self.base_models_)
        if oof_meta_features.shape != (len(y_arr), expected_width) or not np.all(np.isfinite(oof_meta_features)):
            raise ValueError("Invalid OOF meta-feature shape or values")
        if self.task != "regression":
            self.classes_ = self.base_models_[0].classes_.tolist()
            if any(list(m.classes_) != self.classes_ for m in self.base_models_):
                raise ValueError("Stacking parent class order must match")

        if self.task == "regression":
            self.meta_model_ = Ridge(alpha=1.0, random_state=self.seed)
            self.meta_model_.fit(oof_meta_features, y_arr)
        else:
            self.meta_model_ = LogisticRegression(C=1.0, max_iter=200, random_state=self.seed)
            self.meta_model_.fit(oof_meta_features, y_arr)
            if hasattr(self.meta_model_, "classes_"):
                self.classes_ = self.meta_model_.classes_.tolist()
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, Any]:
        """Full inference pipeline: Level-0 base models -> Level-1 meta model."""
        start = time.perf_counter_ns()
        level0_outputs = []
        for model in self.base_models_:
            if self.task != "regression" and hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                # For binary classification, one probability column suffices
                if p.shape[1] == 2:
                    level0_outputs.append(p[:, 1:2])
                else:
                    level0_outputs.append(p)
            else:
                p = np.asarray(model.predict(X)).reshape(-1, 1)
                level0_outputs.append(p)

        meta_input = np.column_stack(level0_outputs)
        pred = self.meta_model_.predict(meta_input)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6

        out: dict[str, Any] = {
            "prediction": pred.tolist(),
            "inference_latency_ms": elapsed_ms,
        }
        if self.task != "regression" and hasattr(self.meta_model_, "predict_proba"):
            out["probabilities"] = self.meta_model_.predict_proba(meta_input).tolist()
            out["classes"] = self.classes_

        return out
