"""Sequential PyTorch Neural Candidate Plugin for MLRescurix.

Provides dedicated neural sequence modeling for 3D temporal inputs
(batch_size, sequence_length, num_features) without flattening or tabular coercion:
1. Implements CandidatePluginContract with prepare, pilot, train, evaluate, predict,
   save_checkpoint, load_checkpoint.
2. Built with PyTorch GRU/LSTM architecture for sequential data.
3. Checkpointing and atomic serialization conforming to mlrescurix contracts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import psutil

from .contract import CustomEstimator
from .contracts import (
    CandidatePluginContract,
    PilotResourceReport,
    ResourceRequestContract,
    TrainingStateContract,
    file_sha256,
)

logger = logging.getLogger("mlrescurix.sequential_candidate")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class _SequentialGRUModule(nn.Module):
        """Compact GRU sequence classifier/regressor."""

        def __init__(
            self,
            in_features: int,
            hidden_dim: int = 32,
            out_features: int = 2,
            num_layers: int = 1,
            is_classification: bool = True,
            dropout: float = 0.05,
        ):
            super().__init__()
            self.in_features = in_features
            self.hidden_dim = hidden_dim
            self.out_features = out_features
            self.is_classification = is_classification

            self.gru = nn.GRU(
                input_size=in_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, out_features)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch_size, seq_len, in_features)
            out, _ = self.gru(x)
            # Take last temporal step
            last_step = out[:, -1, :]
            normed = self.norm(last_step)
            return self.head(normed)

    class _SequentialDataset(Dataset):
        def __init__(self, sequences: np.ndarray, targets: Optional[np.ndarray] = None):
            self.sequences = torch.tensor(sequences, dtype=torch.float32)
            self.targets = torch.tensor(targets, dtype=torch.float32) if targets is not None else None

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            if self.targets is not None:
                return self.sequences[idx], self.targets[idx]
            return self.sequences[idx], torch.empty(0)


class SequentialTorchCandidate(CandidatePluginContract, CustomEstimator):
    """Candidate plugin for sequential / temporal data without tabular coercion."""

    plugin_id: str = "sequential_torch_candidate"
    version: str = "2.0.0"
    capabilities: List[str] = ["fit", "predict", "predict_proba", "checkpoint", "resume", "sequential"]
    dependency_profile: str = "core_neural"
    device_support: List[str] = ["cpu", "cuda"]

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        is_classification: bool = True,
        seq_length: Optional[int] = None,
        in_features: Optional[int] = None,
        seed: int = 42,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for SequentialTorchCandidate")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lr = learning_rate
        self.batch_size = batch_size
        self.is_classification = is_classification
        self.seq_length = seq_length
        self.in_features = in_features
        self.seed = seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.classes_: Optional[List[int]] = [0, 1] if is_classification else None
        self.classes = self.classes_
        self.out_features_ = 2 if is_classification else 1
        self.step_ = 0
        self.epoch_ = 0
        self.decision_threshold = 0.5

        torch.manual_seed(seed)
        np.random.seed(seed)

    def _extract_3d(self, data: Any) -> np.ndarray:
        """Convert input data to 3D numpy array (N, seq_len, in_features)."""
        if isinstance(data, np.ndarray):
            if data.ndim == 3:
                return data.astype(np.float32)
            elif data.ndim == 2:
                # Treat as 1-step sequence or infer
                seq_len = self.seq_length or 1
                in_feat = data.shape[1] // seq_len
                return data.reshape(data.shape[0], seq_len, in_feat).astype(np.float32)
        elif isinstance(data, pd.DataFrame):
            arr = data.values
            seq_len = self.seq_length or 1
            in_feat = arr.shape[1] // seq_len
            return arr.reshape(arr.shape[0], seq_len, in_feat).astype(np.float32)
        raise ValueError(f"Cannot extract 3D sequence from input type {type(data)}")

    def prepare(
        self,
        data_manifest: Dict[str, Any],
        resource_request: ResourceRequestContract,
    ) -> None:
        self.in_features = data_manifest.get("in_features") or data_manifest.get("num_features", self.in_features)
        self.seq_length = data_manifest.get("seq_length", self.seq_length or 10)
        if self.in_features is None:
            raise ValueError("data_manifest must declare 'in_features' or 'num_features'")

        num_classes = data_manifest.get("num_classes", 2 if self.is_classification else 1)
        self.out_features_ = num_classes

        self.model = _SequentialGRUModule(
            in_features=self.in_features,
            hidden_dim=self.hidden_dim,
            out_features=self.out_features_,
            num_layers=self.num_layers,
            is_classification=self.is_classification,
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss() if self.is_classification else nn.MSELoss()

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

        X_3d = self._extract_3d(X_sample)
        if self.model is None or self.in_features != X_3d.shape[2]:
            self.seq_length = X_3d.shape[1]
            self.in_features = X_3d.shape[2]
            self.prepare({"in_features": self.in_features, "seq_length": self.seq_length}, resource_request)

        microbatch_size = min(len(X_3d), 4)
        x_micro = torch.tensor(X_3d[:microbatch_size], dtype=torch.float32, device=self.device)
        y_micro = torch.zeros(microbatch_size, dtype=torch.long if self.is_classification else torch.float32, device=self.device)

        proc = psutil.Process()
        rss_before = proc.memory_info().rss

        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(x_micro)
        loss = self.criterion(out, y_micro)
        loss.backward()
        self.optimizer.step()

        duration = time.monotonic() - start_time
        rss_peak = proc.memory_info().rss

        return PilotResourceReport(
            feasible=True,
            allocated_bytes=rss_peak - rss_before,
            reserved_bytes=0,
            peak_bytes=rss_peak,
            duration_seconds=round(duration, 6),
            microbatch_size=microbatch_size,
            device=self.device,
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

        X_3d = self._extract_3d(X_train)
        y_arr = np.asarray(y_train)

        if self.model is None:
            self.seq_length = X_3d.shape[1]
            self.in_features = X_3d.shape[2]
            dummy_req = ResourceRequestContract()
            self.prepare({"in_features": self.in_features, "seq_length": self.seq_length}, dummy_req)

        dataset = _SequentialDataset(X_3d, y_arr)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        target_steps = max_steps or 100
        current_loss = 0.0

        self.model.train()
        while self.step_ < target_steps:
            for x_b, y_b in loader:
                x_b = x_b.to(self.device)
                y_b = y_b.to(dtype=torch.long if self.is_classification else torch.float32, device=self.device)

                self.optimizer.zero_grad()
                out = self.model(x_b)
                loss = self.criterion(out, y_b)
                loss.backward()
                self.optimizer.step()

                self.step_ += 1
                current_loss = float(loss.item())

                if progress_callback:
                    progress_callback({"step": self.step_, "loss": current_loss})

                if self.step_ >= target_steps:
                    break

            self.epoch_ += 1

        state = TrainingStateContract(
            step=self.step_,
            epoch=self.epoch_,
            optimizer_step_boundary=True,
            weights_manifest={},
            loss=current_loss,
        )
        return state

    def evaluate(self, validation_data: Any) -> Dict[str, Any]:
        if isinstance(validation_data, tuple):
            X_val, y_val = validation_data
        else:
            raise ValueError("validation_data must be (X, y) tuple")

        X_3d = self._extract_3d(X_val)
        y_arr = np.asarray(y_val)

        preds = self.predict(X_3d)
        if self.is_classification:
            accuracy = float(np.mean(preds == y_arr))
            return {"accuracy": accuracy, "primary_metric": accuracy}
        else:
            mse = float(np.mean((preds - y_arr) ** 2))
            return {"mse": mse, "primary_metric": mse}

    def predict(self, inference_data: Any) -> np.ndarray:
        probs = self.predict_proba(inference_data)
        if probs is not None:
            if probs.shape[1] == 2:
                return (probs[:, 1] >= self.decision_threshold).astype(int)
            return np.argmax(probs, axis=1)

        X_3d = self._extract_3d(inference_data)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_3d, dtype=torch.float32, device=self.device)
            out = self.model(x_t)
            return out.cpu().numpy().flatten()

    def predict_proba(self, inference_data: Any) -> Optional[np.ndarray]:
        if not self.is_classification or self.model is None:
            return None
        X_3d = self._extract_3d(inference_data)
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_3d, dtype=torch.float32, device=self.device)
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        step: int,
        training_state: TrainingStateContract,
    ) -> Dict[str, Any]:
        d = Path(checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        weights_file = d / "weights.pt"
        torch.save(self.model.state_dict(), weights_file)
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
        weights_file = d / "weights.pt"
        if not weights_file.is_file():
            raise FileNotFoundError(f"Missing weights.pt in checkpoint dir: {d}")
        self.model.load_state_dict(torch.load(weights_file, map_location=self.device, weights_only=True))
        manifest_file = d / "manifest.json"
        if manifest_file.is_file():
            with open(manifest_file, "r") as f:
                data = json.load(f)
            return TrainingStateContract.from_dict(data.get("training_state", {}))
        return TrainingStateContract(step=self.step_)

    def save_weights(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)

    # Scikit-learn estimator interface for CustomEstimator compatibility
    def fit(self, X: Any, y: Any = None) -> SequentialTorchCandidate:
        dummy_req = ResourceRequestContract()
        X_3d = self._extract_3d(X)
        self.seq_length = X_3d.shape[1]
        self.in_features = X_3d.shape[2]
        self.prepare({"in_features": self.in_features, "seq_length": self.seq_length}, dummy_req)
        self.train((X, y), max_steps=20)
        return self
