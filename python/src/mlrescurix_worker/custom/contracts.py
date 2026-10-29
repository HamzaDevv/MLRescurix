"""Component Contracts for MLRescurix Custom Candidate Lifecycle.

Defines the six fundamental contracts specified in ARCHITECTURE_BUILD_HANDOFF.md:
1. ProblemValidationContract: Modality, target/prediction semantics, inference context, allowed data, split identities.
2. CandidatePluginContract: Versioned prepare, pilot, train, evaluate, predict, save, load lifecycle.
3. ResourceRequestContract: RAM, VRAM, CPU, disk, accelerator, duration, privacy/network, budget.
4. PilotResourceReport: Preflight measurement of allocation, peak memory, duration, and feasibility.
5. TrainingStateContract: Optimizer-step boundary, weights manifest, optimizer/scheduler/scaler, RNG, cursor.
6. ExecutionBackendContract: Preflight, stage, submit, status, cancel, collect, resume lifecycle.
7. ResultBundleContract: Primary metric, aligned predictions, resources, artifacts, locks, entrypoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def file_sha256(path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute SHA-256 hash across a file in bounded chunks (avoids eager full-file reads)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


# =============================================================================
# 1. Problem / Validation Contract
# =============================================================================

@dataclass
class ProblemValidationContract:
    problem_id: str
    modality: str  # "tabular", "neural_sequential", "custom_neural", "text", "vision"
    task_type: str  # "binary_classification", "multiclass_classification", "regression", "custom"
    metric_name: str
    metric_direction: str  # "maximize" or "minimize"
    target_column: Optional[str] = None
    prediction_semantics: Dict[str, Any] = field(default_factory=dict)
    inference_context: Dict[str, Any] = field(default_factory=dict)
    allowed_data_sources: List[str] = field(default_factory=list)
    immutable_split_identities: List[str] = field(default_factory=list)
    data_hash: str = ""

    def validate(self) -> None:
        if not self.problem_id or not isinstance(self.problem_id, str):
            raise ValueError("problem_id must be a non-empty string")
        if self.metric_direction not in ("maximize", "minimize"):
            raise ValueError(f"metric_direction must be 'maximize' or 'minimize', got {self.metric_direction!r}")
        if self.modality not in ("tabular", "neural_sequential", "custom_neural", "text", "vision", "multimodal"):
            raise ValueError(f"Unsupported modality: {self.modality!r}")
        if self.task_type not in (
            "binary_classification", "multiclass_classification", "regression",
            "binary", "multiclass", "custom"
        ):
            raise ValueError(f"Unsupported task_type: {self.task_type!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProblemValidationContract:
        inst = cls(**data)
        inst.validate()
        return inst


# =============================================================================
# 2. Resource Request Contract & Pilot Report
# =============================================================================

@dataclass
class ResourceRequestContract:
    ram_bytes_limit: int = 2 * 1024 * 1024 * 1024  # 2 GB host RAM
    vram_bytes_limit: Optional[int] = None  # None for CPU, or bytes for GPU
    cpu_count: int = 1
    disk_bytes_limit: int = 10 * 1024 * 1024 * 1024  # 10 GB
    accelerator_type: str = "cpu"  # "cpu", "gpu", "tpu", "any"
    accelerator_count: int = 0
    max_duration_seconds: float = 300.0
    network_allowed: bool = False
    privacy_level: str = "offline_isolated"
    budget_usd_limit: Optional[float] = None
    quota_minutes_limit: Optional[float] = None

    # Measured fields populated post-execution
    measured_peak_ram_bytes: Optional[int] = None
    measured_peak_vram_bytes: Optional[int] = None
    measured_cpu_seconds: Optional[float] = None
    measured_wall_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResourceRequestContract:
        return cls(**data)


@dataclass
class PilotResourceReport:
    """Bounded forward/backward/optimizer pilot measurement report."""
    feasible: bool
    allocated_bytes: int
    reserved_bytes: int
    peak_bytes: int
    duration_seconds: float
    microbatch_size: int
    device: str
    precision: str
    limitations: List[str] = field(default_factory=list)

    @property
    def peak_memory_bytes(self) -> int:
        return self.peak_bytes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PilotResourceReport:
        return cls(**data)


# =============================================================================
# 3. Training State Contract
# =============================================================================

@dataclass
class TrainingStateContract:
    """Exact training state snapshot at an optimizer-step boundary."""
    step: int
    epoch: int
    optimizer_step_boundary: bool
    weights_manifest: Dict[str, str]  # relative filename -> sha256
    optimizer_state_hash: Optional[str] = None
    scheduler_state_hash: Optional[str] = None
    scaler_state_hash: Optional[str] = None
    rng_state: Dict[str, Any] = field(default_factory=dict)
    data_cursor: Dict[str, Any] = field(default_factory=dict)
    accumulation_step: int = 0
    code_hash: str = ""
    environment_hash: str = ""
    data_hash: str = ""
    fold_identity: Optional[str] = None
    loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingStateContract:
        return cls(**data)


# =============================================================================
# 4. Candidate Plugin Contract (Lifecycle)
# =============================================================================

class CandidatePluginContract(ABC):
    """
    Abstract contract for agent-authored or library-provided model plugins.
    Enforces the complete lifecycle:
      prepare -> pilot -> train -> evaluate -> predict -> save -> load
    """

    plugin_id: str = "generic_candidate_plugin"
    version: str = "1.0.0"
    capabilities: List[str] = ["fit", "predict"]
    dependency_profile: str = "core_tabular"
    device_support: List[str] = ["cpu"]

    @abstractmethod
    def prepare(
        self,
        data_manifest: Dict[str, Any],
        resource_request: ResourceRequestContract,
    ) -> None:
        """Inspect data metadata, allocate internal buffers, validate dimensions."""
        raise NotImplementedError

    @abstractmethod
    def pilot(
        self,
        sample_data: Any,
        resource_request: ResourceRequestContract,
    ) -> PilotResourceReport:
        """Run single microbatch forward/backward pilot step to measure real memory requirements."""
        raise NotImplementedError

    @abstractmethod
    def train(
        self,
        train_data: Any,
        validation_data: Optional[Any] = None,
        resume_from: Optional[Union[str, Path]] = None,
        max_steps: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TrainingStateContract:
        """Execute training with support for checkpoint resume and step-boundary interruption."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, validation_data: Any) -> Dict[str, Any]:
        """Compute evaluation metrics, validation loss, and aligned predictions."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, inference_data: Any) -> np.ndarray:
        """Generate point predictions for inference rows."""
        raise NotImplementedError

    def predict_proba(self, inference_data: Any) -> Optional[np.ndarray]:
        """Generate class probabilities summing to 1.0 along axis 1 (if classification)."""
        return None

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_dir: Union[str, Path],
        step: int,
        training_state: TrainingStateContract,
    ) -> Dict[str, Any]:
        """
        Atomically persist model weights, optimizer, scheduler, and RNG state.
        Flushes all files and writes manifest.json with SHA-256 hashes.
        """
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, checkpoint_dir: Union[str, Path]) -> TrainingStateContract:
        """Load and verify checkpoint from directory with hash verification."""
        raise NotImplementedError


# =============================================================================
# 5. Execution Backend Contract
# =============================================================================

class ExecutionBackendContract(ABC):
    """
    Contract for execution backends (local sandbox, Docker, Kaggle compute pool).
    Coordinates preflight, staging, asynchronous dispatch, polling, and collection.
    """

    backend_id: str = "base_backend"

    @abstractmethod
    def preflight(
        self,
        request: ResourceRequestContract,
        plugin_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Probe environment, accelerator capabilities, quota, and check feasibility."""
        raise NotImplementedError

    @abstractmethod
    def stage(
        self,
        job_id: str,
        code_dir: Path,
        data_path: Path,
        manifest: Dict[str, Any],
    ) -> Path:
        """Stage worker code, isolated environment config, and input data."""
        raise NotImplementedError

    @abstractmethod
    def submit(self, staged_dir: Path, job_params: Dict[str, Any]) -> str:
        """Submit staged workload and return durable job handle."""
        raise NotImplementedError

    @abstractmethod
    def status(self, job_handle: str) -> Dict[str, Any]:
        """
        Poll lifecycle status:
        PLANNED, STAGED, SUBMITTED, RUNNING, COLLECTING, VERIFIED, FAILED, NEEDS_RESOURCES, CANCELLED.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job_handle: str) -> bool:
        """Request remote cancellation; return True if successfully dispatched."""
        raise NotImplementedError

    @abstractmethod
    def collect(self, job_handle: str, target_dir: Path) -> Dict[str, Any]:
        """Collect remote outputs, verify hashes, and return result payload."""
        raise NotImplementedError

    @abstractmethod
    def resume(
        self,
        job_handle: str,
        checkpoint_manifest: Dict[str, Any],
    ) -> str:
        """Resume a failed or interrupted job from a verified durable checkpoint."""
        raise NotImplementedError


# =============================================================================
# 6. Result Bundle Contract
# =============================================================================

@dataclass
class ResultBundleContract:
    candidate_id: str
    primary_metric_name: str
    primary_metric_score: float
    primary_metric_direction: str
    aligned_predictions: List[Any]
    row_ids: List[Any]
    probabilities: Optional[List[List[float]]] = None
    classes: Optional[List[Any]] = None
    measured_resources: Dict[str, Any] = field(default_factory=dict)
    model_artifacts: Dict[str, str] = field(default_factory=dict)
    transforms: List[Dict[str, Any]] = field(default_factory=list)
    dependency_lock: Dict[str, str] = field(default_factory=dict)
    inference_entrypoint: str = "inference.py"
    failure_limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ResultBundleContract:
        return cls(**data)


# =============================================================================
# 7. Versioned Worker, Job, Checkpoint, and Result Contracts (v2)
# =============================================================================

@dataclass
class CustomJobSpecContract:
    job_id: str
    project_id: str
    task_id: str
    generation: int
    intent_revision: int
    operation: str  # "custom_train", "custom_resume", "custom_evaluate", "custom_export"
    model_plugin: str
    dataset_id: str
    metric_name: str
    metric_direction: str  # "maximize" or "minimize"
    max_steps: int
    batch_size: int
    schema_version: str = "v2"
    candidate_id: Optional[str] = None
    target_column: Optional[str] = None
    resume_checkpoint: Optional[str] = None
    learning_rate: Optional[float] = None
    hidden_dim: Optional[int] = None
    seed: Optional[int] = 42
    resources: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.job_id or not self.project_id or not self.task_id:
            raise ValueError("job_id, project_id, and task_id must be non-empty")
        if self.operation not in ("custom_train", "custom_resume", "custom_evaluate", "custom_export"):
            raise ValueError(f"Unsupported custom operation: {self.operation}")
        if self.metric_direction not in ("maximize", "minimize"):
            raise ValueError(f"Invalid metric direction: {self.metric_direction}")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.resume_checkpoint is not None:
            if (
                not self.resume_checkpoint
                or len(self.resume_checkpoint) > 256
                or "\0" in self.resume_checkpoint
                or ".." in self.resume_checkpoint
                or ":" in self.resume_checkpoint
                or self.resume_checkpoint.startswith("/")
                or self.resume_checkpoint.startswith("\\")
                or Path(self.resume_checkpoint).is_absolute()
            ):
                raise ValueError(
                    "invalid resume_checkpoint reference; must be a valid relative checkpoint identity without directory traversal"
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CustomJobSpecContract:
        inst = cls(**data)
        inst.validate()
        return inst


@dataclass
class TrainingCheckpointContract:
    checkpoint_id: str
    job_id: str
    step: int
    epoch: int
    batch_offset: int
    sampler_position: int
    files: Dict[str, str]  # filename -> sha256
    timestamp_unix: float
    schema_version: str = "v2"
    project_id: Optional[str] = None
    candidate_id: Optional[str] = None
    weights_sha256: Optional[str] = None
    dataset_sha256: Optional[str] = None
    split_sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    environment_id: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    replay_guarantee: str = "unverified"
    loss: Optional[float] = None
    training_state: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.checkpoint_id or not self.job_id:
            raise ValueError("checkpoint_id and job_id must be non-empty")
        if not self.files:
            raise ValueError("checkpoint files cannot be empty")
        for fname, sha in self.files.items():
            if len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha.lower()):
                raise ValueError(f"Invalid file sha256 for {fname}: {sha}")
        if self.weights_sha256 and (len(self.weights_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.weights_sha256.lower())):
            raise ValueError(f"Invalid weights_sha256 in checkpoint: {self.weights_sha256}")
        if self.dataset_sha256 and (len(self.dataset_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.dataset_sha256.lower())):
            raise ValueError(f"Invalid dataset_sha256 in checkpoint: {self.dataset_sha256}")
        if self.split_sha256 and (len(self.split_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.split_sha256.lower())):
            raise ValueError(f"Invalid split_sha256 in checkpoint: {self.split_sha256}")
        if self.source_sha256 and (len(self.source_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.source_sha256.lower())):
            raise ValueError(f"Invalid source_sha256 in checkpoint: {self.source_sha256}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingCheckpointContract:
        inst = cls(**data)
        inst.validate()
        return inst


@dataclass
class TrainingResultContract:
    job_id: str
    status: str  # "succeeded", "failed", "timed_out", "needs_resources", "interrupted"
    step_reached: int
    epoch_reached: int
    peak_memory_bytes: int
    duration_seconds: float
    artifacts: List[str]
    schema_version: str = "v2"
    project_id: Optional[str] = None
    candidate_id: Optional[str] = None
    weights_sha256: Optional[str] = None
    dataset_sha256: Optional[str] = None
    split_sha256: Optional[str] = None
    source_sha256: Optional[str] = None
    environment_id: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    replay_guarantee: str = "unverified"
    primary_metric_name: Optional[str] = None
    primary_metric_score: Optional[float] = None
    validation_loss: Optional[float] = None
    limitations: List[str] = field(default_factory=list)
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.job_id:
            raise ValueError("job_id must be non-empty")
        if self.status not in ("succeeded", "failed", "timed_out", "needs_resources", "interrupted"):
            raise ValueError(f"Invalid training result status: {self.status}")
        if self.weights_sha256 and (len(self.weights_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.weights_sha256.lower())):
            raise ValueError(f"Invalid weights_sha256 in result: {self.weights_sha256}")
        if self.dataset_sha256 and (len(self.dataset_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.dataset_sha256.lower())):
            raise ValueError(f"Invalid dataset_sha256 in result: {self.dataset_sha256}")
        if self.split_sha256 and (len(self.split_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.split_sha256.lower())):
            raise ValueError(f"Invalid split_sha256 in result: {self.split_sha256}")
        if self.source_sha256 and (len(self.source_sha256) != 64 or not all(c in "0123456789abcdef" for c in self.source_sha256.lower())):
            raise ValueError(f"Invalid source_sha256 in result: {self.source_sha256}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrainingResultContract:
        inst = cls(**data)
        inst.validate()
        return inst


@dataclass
class WorkerHandshakeContract:
    protocol: str
    worker_id: str
    capabilities: List[str]
    dependency_profile: str
    device: str
    platform: str
    schema_version: str = "v2"

    def validate(self) -> None:
        if not self.worker_id or not self.protocol:
            raise ValueError("worker_id and protocol must be non-empty")
        if not self.capabilities:
            raise ValueError("capabilities cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> WorkerHandshakeContract:
        inst = cls(**data)
        inst.validate()
        return inst
