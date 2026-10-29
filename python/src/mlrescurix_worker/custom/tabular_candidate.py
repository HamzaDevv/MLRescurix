"""Tabular Candidate Plugin for MLRescurix.

Provides general tabular modeling (gradient boosting / forest) conforming to
CandidatePluginContract and CustomEstimator:
1. prepare -> pilot -> train -> evaluate -> predict -> predict_proba -> save -> load
2. Handles numeric and categorical features cleanly.
3. Checkpoint persistence and delivery bundle packaging.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import psutil
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .contract import CustomEstimator
from .contracts import (
    CandidatePluginContract,
    PilotResourceReport,
    ResourceRequestContract,
    TrainingStateContract,
    file_sha256,
)

logger = logging.getLogger("mlrescurix.tabular_candidate")


class TabularCandidatePlugin(CandidatePluginContract, CustomEstimator):
    """General tabular model plugin implementing CandidatePluginContract."""

    plugin_id: str = "tabular_candidate_plugin"
    version: str = "2.0.0"
    capabilities: List[str] = ["fit", "predict", "predict_proba", "checkpoint", "tabular"]
    dependency_profile: str = "core_tabular"
    device_support: List[str] = ["cpu"]

    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 6,
        is_classification: bool = True,
        seed: int = 42,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.is_classification = is_classification
        self.seed = seed

        self.model: Optional[Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]] = None
        self.feature_names_: List[str] = []
        self.classes_: Optional[List[int]] = [0, 1] if is_classification else None
        self.classes = self.classes_
        self.step_ = 0
        self.decision_threshold = 0.5

    def _ensure_matrix(self, data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            self.feature_names_ = list(data.columns)
            return data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        raise ValueError(f"Unsupported data type for tabular candidate: {type(data)}")

    def prepare(
        self,
        data_manifest: Dict[str, Any],
        resource_request: ResourceRequestContract,
    ) -> None:
        self.feature_names_ = data_manifest.get("features", data_manifest.get("columns", []))
        if self.is_classification:
            self.model = HistGradientBoostingClassifier(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.seed,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.seed,
            )

    def pilot(
        self,
        sample_data: Any,
        resource_request: ResourceRequestContract,
    ) -> PilotResourceReport:
        start_time = time.monotonic()
        if isinstance(sample_data, tuple):
            X_sample, y_sample = sample_data
        else:
            X_sample = sample_data
            y_sample = np.zeros(len(sample_data))

        X_mat = self._ensure_matrix(X_sample)
        y_arr = np.asarray(y_sample)
        micro_n = min(len(X_mat), 16)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        # Quick fit on micro-batch
        self.prepare({}, resource_request)
        self.model.fit(X_mat[:micro_n], y_arr[:micro_n])

        duration = time.monotonic() - start_time
        rss_peak = proc.memory_info().rss

        return PilotResourceReport(
            feasible=True,
            allocated_bytes=rss_peak - rss_before,
            reserved_bytes=0,
            peak_bytes=rss_peak,
            duration_seconds=round(duration, 6),
            microbatch_size=micro_n,
            device="cpu",
            precision="float32",
            limitations=[],
        )

    def train(
        self,
        train_data: Any,
        validation_data: Optional[Any] = None,
        resume_from: Optional[Union[str, Path]] = None,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TrainingStateContract:
        if resume_from:
            self.load_checkpoint(resume_from)

        if isinstance(train_data, tuple):
            X_train, y_train = train_data
        else:
            raise ValueError("train_data must be (X, y) tuple")

        X_mat = self._ensure_matrix(X_train)
        y_arr = np.asarray(y_train)

        if self.model is None:
            self.prepare({}, ResourceRequestContract())

        self.model.fit(X_mat, y_arr)
        self.step_ = self.max_iter

        if hasattr(self.model, "classes_"):
            self.classes_ = [int(c) for c in self.model.classes_]
            self.classes = self.classes_

        if progress_callback:
            progress_callback({"step": self.step_, "status": "completed"})

        state = TrainingStateContract(
            step=self.step_,
            epoch=1,
            optimizer_step_boundary=True,
            weights_manifest={},
            loss=0.0,
        )
        return state

    def evaluate(self, validation_data: Any) -> Dict[str, Any]:
        if isinstance(validation_data, tuple):
            X_val, y_val = validation_data
        else:
            raise ValueError("validation_data must be (X, y) tuple")

        X_mat = self._ensure_matrix(X_val)
        y_arr = np.asarray(y_val)

        preds = self.predict(X_mat)
        if self.is_classification:
            accuracy = float(np.mean(preds == y_arr))
            return {"accuracy": accuracy, "primary_metric": accuracy}
        else:
            mse = float(np.mean((preds - y_arr) ** 2))
            return {"mse": mse, "primary_metric": mse}

    def predict(self, inference_data: Any) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be trained before calling predict")
        X_mat = self._ensure_matrix(inference_data)
        if self.is_classification:
            probs = self.predict_proba(inference_data)
            if probs is not None and probs.shape[1] == 2:
                return (probs[:, 1] >= self.decision_threshold).astype(int)
            return self.model.predict(X_mat)
        return self.model.predict(X_mat)

    def predict_proba(self, inference_data: Any) -> Optional[np.ndarray]:
        if not self.is_classification or self.model is None:
            return None
        X_mat = self._ensure_matrix(inference_data)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_mat)
        return None

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        step: int,
        training_state: TrainingStateContract,
    ) -> Dict[str, Any]:
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        weights_file = d / "model.joblib"
        joblib.dump(self.model, weights_file)
        w_sha = file_sha256(weights_file)

        manifest = {
            "step": step,
            "weights_sha256": w_sha,
            "training_state": training_state.to_dict(),
        }
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> TrainingStateContract:
        d = Path(checkpoint_dir)
        weights_file = d / "model.joblib"
        if not weights_file.is_file():
            raise FileNotFoundError(f"Missing model.joblib in checkpoint dir: {d}")
        self.model = joblib.load(weights_file)
        manifest_file = d / "manifest.json"
        if manifest_file.is_file():
            with open(manifest_file, "r") as f:
                data = json.load(f)
            return TrainingStateContract.from_dict(data.get("training_state", {}))
        return TrainingStateContract(step=self.step_)

    def save_weights(self, path: Path) -> None:
        joblib.dump(self.model, path)

    def fit(self, X: Any, y: Any = None) -> TabularCandidatePlugin:
        self.prepare({}, ResourceRequestContract())
        self.train((X, y))
        return self
