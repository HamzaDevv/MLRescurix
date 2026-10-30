"""Local Scientific Custom Model Worker CLI.

Executes custom model training, resumption, evaluation, and delivery packaging
at the supervised Python process boundary:
- Enforces thread caps and process resource limits.
- Emits experimental checkpoint and result artifacts; exact replay is not established.
- Worker-reported scores and artifacts require independent controller verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import hashlib
from .contracts import (
    CustomJobSpecContract,
    TrainingCheckpointContract,
    TrainingResultContract,
    file_sha256,
)
from .delivery import CustomBundlePackager
from .torch_candidate import TinyTorchNeuralCandidate

logger = logging.getLogger("mlrescurix.custom_worker")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


def configure_process_limits(spec: CustomJobSpecContract) -> None:
    """Enforce CPU/memory limits and thread caps on child processes."""
    threads = None
    if spec.resources:
        threads = spec.resources.get("num_threads") or spec.resources.get("cpu_cores")
    if threads is None:
        threads = os.environ.get("MLRESCURIX_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")

    if threads is not None:
        try:
            t_int = int(threads)
            import torch
            torch.set_num_threads(t_int)
            torch.set_num_interop_threads(t_int)
        except Exception as exc:
            logger.debug("Failed to set PyTorch thread caps: %s", exc)

    if spec.resources:
        cpu_seconds = spec.resources.get("cpu_seconds")
        if cpu_seconds:
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 5))
            except Exception as exc:
                logger.debug("Failed to set RLIMIT_CPU: %s", exc)

        if sys.platform.startswith("linux") and hasattr(resource, "RLIMIT_AS"):
            mem_bytes = spec.resources.get("memory_bytes")
            if mem_bytes:
                try:
                    resource.setrlimit(resource.RLIMIT_AS, (int(mem_bytes), int(mem_bytes)))
                except Exception as exc:
                    logger.debug("Failed to set RLIMIT_AS: %s", exc)


def load_dataset(dataset_path: Union[str, Path], target_column: str) -> pd.DataFrame:
    """Provisional / test helper for validating dataset target column presence."""
    p = Path(dataset_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found at {p}")
    df = pd.read_csv(p)
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: '{target_column}' in columns {list(df.columns)}")
    return df


def load_dataset_with_protocol(
    dataset_path: Path,
    workflow_dir: Path,
    target_column: Optional[str] = None,
    expected_metric: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str, str, str, str, str]:
    """Loads CSV data using the experimental worker manifest.

    Rejects invented splits and silent metric substitutions.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    manifest_path = workflow_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset_manifest.json not found in workflow {workflow_dir}; "
            "authorized development execution requires audited dataset splits"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Format check: strictly "mlrescurix.dataset.v1"
    manifest_format = manifest.get("format")
    if manifest_format != "mlrescurix.dataset.v1":
        raise ValueError(f"Invalid dataset manifest format: {manifest_format!r}; expected 'mlrescurix.dataset.v1'")

    # 2. SHA-256 and integrity checks
    file_hash = file_sha256(dataset_path)
    manifest_sha = manifest.get("sha256") or manifest.get("dataset_sha256")
    if not manifest_sha or not isinstance(manifest_sha, str) or len(manifest_sha) != 64 or not all(c in "0123456789abcdef" for c in manifest_sha.lower()):
        raise ValueError(
            f"Invalid or missing dataset sha256 in manifest: {manifest_sha!r}"
        )

    dev_sha = os.environ.get("MLRESCURIX_DEV_DATASET_SHA256") or manifest.get("dev_sha256") or manifest.get("dev_dataset_sha256")
    if dev_sha:
        if file_hash.lower() != dev_sha.lower() and file_hash.lower() != manifest_sha.lower():
            raise ValueError(
                f"Dataset SHA-256 mismatch: file hash is {file_hash}, expected {dev_sha} (dev) or {manifest_sha} (full)"
            )
    else:
        if file_hash.lower() != manifest_sha.lower():
            raise ValueError(
                f"Dataset SHA-256 mismatch: manifest has {manifest_sha}, actual is {file_hash}"
            )
    dataset_sha256 = manifest_sha

    # 3. Splits check: authoritative protocol mandates train, validation, and test splits
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or "train" not in splits or "validation" not in splits or "test" not in splits:
        raise ValueError("Audited dataset_manifest.json missing immutable 'train', 'validation', or 'test' splits")

    train_idx = splits["train"]
    val_idx = splits["validation"]
    test_idx = splits["test"]
    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Train, validation, or test split is empty in dataset_manifest.json")

    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        if not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError(f"Invalid row indices in {name} split; must be non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate row indices in {name} split")

    # Strict partition disjointness check
    overlap_val = set(train_idx) & set(val_idx)
    if overlap_val:
        raise ValueError(f"Data leakage detected: train and validation splits overlap on rows {overlap_val}")
    leak_test = (set(train_idx) | set(val_idx)) & set(test_idx)
    if leak_test:
        raise ValueError(f"Data leakage detected: train/validation splits overlap with sealed test split on rows {leak_test}")

    target_col = manifest.get("target") or manifest.get("target_column") or target_column
    if not target_col:
        raise ValueError("Target column not specified in dataset_manifest.json or job spec")
    if target_column and target_column != target_col:
        raise ValueError(
            f"Target column mismatch: spec requested {target_column!r}, manifest specifies {target_col!r}"
        )

    features = manifest.get("features") or manifest.get("feature_columns")
    df = pd.read_csv(dataset_path)

    # 4. Strict sealed holdout protection:
    # Development workers must only be given access to approved training and validation data.
    # Reading the full dataset and slicing out test rows is NOT isolation.
    if "__mlr_row_id" in df.columns:
        present_row_ids = set(df["__mlr_row_id"])
        leak = set(test_idx) & present_row_ids
        if leak:
            raise ValueError(
                f"Data leakage detected: sealed test partition row(s) present in development dataset: {sorted(leak)}"
            )
        train_rows = df[df["__mlr_row_id"].isin(train_idx)]
        val_rows = df[df["__mlr_row_id"].isin(val_idx)]
        if len(train_rows) == 0:
            raise ValueError("Development dataset contains zero training rows matching manifest train split")
        if len(val_rows) == 0:
            raise ValueError("Development dataset contains zero validation rows matching manifest validation split")
    else:
        expected_dev_len = len(train_idx) + len(val_idx)
        total_len = expected_dev_len + len(test_idx)
        if len(df) >= total_len:
            raise ValueError(
                "Data leakage detected: development worker given access to full dataset containing sealed test partition; "
                "reading full dataset and slicing out test rows is prohibited"
            )
        raise ValueError(
            "Development dataset missing required '__mlr_row_id' column to index partition rows"
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col!r} not found in dataset columns: {list(df.columns)}")

    if not features:
        features = [c for c in df.columns if c != target_col and c != "__mlr_row_id"]
    else:
        features = [c for c in features if c != "__mlr_row_id"]
        for f in features:
            if f not in df.columns:
                raise ValueError(f"Declared feature {f!r} not found in dataset columns")

    if target_col in features:
        raise ValueError("Target column cannot be included in features")

    metric_name = manifest.get("metric") or manifest.get("primary_metric")
    if not metric_name:
        raise ValueError("Validation metric missing from dataset manifest")
    metric_direction = "minimize" if metric_name in ("log_loss", "mae", "rmse", "mse") else "maximize"
    if expected_metric and expected_metric != metric_name:
        raise ValueError(
            f"Metric mismatch: spec requested {expected_metric!r}, but immutable validation protocol mandates {metric_name!r}"
        )

    split_sha256 = file_sha256(manifest_path)

    X_train = train_rows[features].reset_index(drop=True)
    y_train = train_rows[target_col].reset_index(drop=True)
    X_val = val_rows[features].reset_index(drop=True)
    y_val = val_rows[target_col].reset_index(drop=True)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        features,
        target_col,
        dataset_sha256,
        split_sha256,
        metric_name,
        metric_direction,
    )


def get_peak_rss() -> int:
    """Read current process RSS in bytes."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            # ru_maxrss is in KiB on Linux, bytes on macOS
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return int(usage)
            return int(usage * 1024)
        except Exception:
            return 0


def resolve_safe_checkpoint_dir(workflow_dir: Path, ckpt_ref: Optional[str], default_job_id: str) -> Path:
    """Resolves and bounds checkpoint directory within project checkpoints directory."""
    checkpoints_base = (workflow_dir / "checkpoints").resolve()
    checkpoints_base.mkdir(parents=True, exist_ok=True)
    if ckpt_ref is not None:
        if (
            not ckpt_ref
            or len(ckpt_ref) > 256
            or "\0" in ckpt_ref
            or ".." in ckpt_ref
            or ":" in ckpt_ref
            or ckpt_ref.startswith("/")
            or ckpt_ref.startswith("\\")
            or Path(ckpt_ref).is_absolute()
        ):
            raise ValueError(
                f"invalid resume_checkpoint reference {ckpt_ref!r}; directory traversal and arbitrary paths prohibited"
            )
        resolved = (checkpoints_base / ckpt_ref).resolve()
    else:
        resolved = (checkpoints_base / default_job_id).resolve()

    try:
        resolved.relative_to(checkpoints_base)
    except ValueError:
        raise ValueError(f"Checkpoint path escapes project sandbox: {resolved}")
    if resolved == checkpoints_base:
        raise ValueError(f"Checkpoint path cannot be the root checkpoints directory: {resolved}")
    return resolved


def run_worker(
    operation: str,
    spec_path: Path,
    output_dir: Path,
    dataset_path: Optional[Path],
    workflow_dir: Path,
    candidate_id: Optional[str] = None,
    test_access_id: Optional[str] = None,
) -> int:
    """Executes the requested custom worker operation conforming to MLRescurix v2 contracts."""
    t_start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_dir.mkdir(parents=True, exist_ok=True)

    job_id = "unknown"
    cand_id = candidate_id
    op = operation
    model: Optional[TinyTorchNeuralCandidate] = None

    try:
        # 1. Parse and validate job spec
        with open(spec_path, "r", encoding="utf-8") as f:
            spec_dict = json.load(f)
        job_id = spec_dict.get("job_id", "unknown")
        spec = CustomJobSpecContract.from_dict(spec_dict)
        job_id = spec.job_id
        op = operation or spec.operation
        if op != spec.operation:
            raise ValueError("Requested operation does not match job specification")
        if spec.model_plugin != "tiny_torch":
            raise ValueError(f"Unsupported model plugin: {spec.model_plugin}")

        # 2. Configure limits
        configure_process_limits(spec)

        # Reject holdout access attempts: custom development operations cannot receive or consume a test access gate
        if test_access_id is not None:
            raise ValueError(
                f"Holdout access is sealed; custom scientific development operation {op!r} cannot receive or consume a test access gate"
            )

        source_sha256 = os.environ.get("MLRESCURIX_SOURCE_SHA256") or ""
        env_id = os.environ.get("MLRESCURIX_ENVIRONMENT_IDENTITY") or getattr(spec, "environment_id", None)
        project_id = spec.project_id
        training_config = {
            "learning_rate": spec.learning_rate,
            "batch_size": spec.batch_size,
            "hidden_dim": spec.hidden_dim,
            "seed": spec.seed,
            "metric_name": spec.metric_name,
            "metric_direction": spec.metric_direction,
        }

        cand_id = candidate_id or spec.candidate_id
        if not cand_id:
            raise ValueError("Candidate identity is required for custom scientific operations; model_plugin substitution prohibited")
        if candidate_id and spec.candidate_id and candidate_id != spec.candidate_id:
            raise ValueError(f"Candidate identity mismatch: CLI passed {candidate_id!r} but spec specifies {spec.candidate_id!r}")
        for name, value in (("job_id", spec.job_id), ("candidate_id", cand_id)):
            if not isinstance(value, str) or value in (".", "..") or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
                raise ValueError(f"Invalid {name}; must be a safe identity")
        model = TinyTorchNeuralCandidate(
            hidden_dim=spec.hidden_dim or 32,
            lr=spec.learning_rate or 0.001,
            batch_size=spec.batch_size or 32,
            random_seed=spec.seed or 42,
        )
        replay_guarantee = "unverified"

        if op == "custom_train":
            if not dataset_path:
                raise ValueError("custom_train requires --dataset")
            (
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                target_col,
                dataset_sha256,
                split_sha256,
                metric_name,
                metric_direction,
            ) = load_dataset_with_protocol(
                dataset_path=dataset_path,
                workflow_dir=workflow_dir,
                target_column=spec.target_column,
                expected_metric=spec.metric_name,
            )

            train_state = model.train(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val),
                max_steps=spec.max_steps,
            )

            # Persist checkpoint in workflow directory with full identity binding
            ckpt_dir = workflow_dir / "checkpoints" / spec.job_id
            ckpt_manifest = model.save_checkpoint(
                ckpt_dir,
                model.current_step,
                train_state,
                candidate_id=cand_id,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                replay_guarantee=replay_guarantee,
                project_id=project_id,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
            )

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]
            weights_sha256 = ckpt_manifest["files"]["weights.pt"]

            ckpt_contract = TrainingCheckpointContract(
                checkpoint_id=f"ckpt_{spec.job_id}_{model.current_step}",
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=weights_sha256,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                step=model.current_step,
                epoch=model.current_epoch,
                batch_offset=model.batch_offset,
                sampler_position=model.sampler_position,
                files=ckpt_manifest["files"],
                timestamp_unix=time.time(),
                loss=eval_res.get("loss"),
                training_state=train_state.to_dict(),
            )
            (output_dir / "checkpoint.json").write_text(json.dumps(ckpt_contract.to_dict(), indent=2), encoding="utf-8")
            final_ckpt_dir = Path(ckpt_manifest.get("checkpoint_dir", ckpt_dir))
            if (final_ckpt_dir / "weights.pt").is_file():
                shutil.copy2(final_ckpt_dir / "weights.pt", output_dir / "weights.pt")

            artifacts = ["result.json", "checkpoint.json"]
            if (output_dir / "weights.pt").is_file():
                artifacts.append("weights.pt")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=weights_sha256,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                status="succeeded",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=artifacts,
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric),
                validation_loss=eval_res.get("loss"),
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.info("custom_train succeeded at step %d: %s = %.4f", model.current_step, spec.metric_name, float(val_metric))
            return 0

        elif op == "custom_resume":
            if not dataset_path:
                raise ValueError("custom_resume requires --dataset")
            (
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                target_col,
                dataset_sha256,
                split_sha256,
                metric_name,
                metric_direction,
            ) = load_dataset_with_protocol(
                dataset_path=dataset_path,
                workflow_dir=workflow_dir,
                target_column=spec.target_column,
                expected_metric=spec.metric_name,
            )

            resume_dir = resolve_safe_checkpoint_dir(workflow_dir, spec.resume_checkpoint, spec.job_id)
            if not resume_dir.exists():
                raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_dir}")

            train_state = model.train(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val),
                resume_from=resume_dir,
                max_steps=spec.max_steps,
                expected_dataset_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                expected_candidate_id=cand_id,
                expected_project_id=project_id,
                expected_source_sha256=source_sha256 if source_sha256 else None,
                expected_environment_id=env_id if env_id else None,
            )

            resumed_ckpt_dir = workflow_dir / "checkpoints" / f"{spec.job_id}_resumed"
            ckpt_manifest = model.save_checkpoint(
                resumed_ckpt_dir,
                model.current_step,
                train_state,
                candidate_id=cand_id,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                replay_guarantee=replay_guarantee,
                project_id=project_id,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
            )

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]
            weights_sha256 = ckpt_manifest["files"]["weights.pt"]

            ckpt_contract = TrainingCheckpointContract(
                checkpoint_id=f"ckpt_{spec.job_id}_{model.current_step}",
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=weights_sha256,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                step=model.current_step,
                epoch=model.current_epoch,
                batch_offset=model.batch_offset,
                sampler_position=model.sampler_position,
                files=ckpt_manifest["files"],
                timestamp_unix=time.time(),
                loss=eval_res.get("loss"),
                training_state=train_state.to_dict(),
            )
            (output_dir / "checkpoint.json").write_text(json.dumps(ckpt_contract.to_dict(), indent=2), encoding="utf-8")
            final_resumed_ckpt_dir = Path(ckpt_manifest.get("checkpoint_dir", resumed_ckpt_dir))
            if (final_resumed_ckpt_dir / "weights.pt").is_file():
                shutil.copy2(final_resumed_ckpt_dir / "weights.pt", output_dir / "weights.pt")

            artifacts = ["result.json", "checkpoint.json"]
            if (output_dir / "weights.pt").is_file():
                artifacts.append("weights.pt")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=weights_sha256,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                status="succeeded",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=artifacts,
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric),
                validation_loss=eval_res.get("loss"),
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.info("custom_resume succeeded at step %d: %s = %.4f", model.current_step, spec.metric_name, float(val_metric))
            return 0

        elif op == "custom_evaluate":
            if not dataset_path:
                raise ValueError("custom_evaluate requires --dataset")
            (
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                target_col,
                dataset_sha256,
                split_sha256,
                metric_name,
                metric_direction,
            ) = load_dataset_with_protocol(
                dataset_path=dataset_path,
                workflow_dir=workflow_dir,
                target_column=spec.target_column,
                expected_metric=spec.metric_name,
            )

            if not spec.resume_checkpoint:
                raise ValueError("Evaluation and export require an explicit checkpoint")
            resume_dir = resolve_safe_checkpoint_dir(workflow_dir, spec.resume_checkpoint, spec.job_id)
            model.load_checkpoint(
                resume_dir,
                expected_dataset_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                expected_candidate_id=cand_id,
                expected_project_id=project_id,
                expected_source_sha256=source_sha256 if source_sha256 else None,
                expected_environment_id=env_id if env_id else None,
            )

            eval_weights_file = resume_dir / "weights.pt"
            if not eval_weights_file.is_file():
                step_dirs = [d for d in resume_dir.glob("step_*") if d.is_dir() and (d / "weights.pt").is_file()]
                if step_dirs:
                    step_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
                    eval_weights_file = step_dirs[-1] / "weights.pt"
                else:
                    raise FileNotFoundError(f"Model weights.pt not found in checkpoint {resume_dir}")

            weights_sha256 = file_sha256(eval_weights_file)

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            # Quality gate: Reject candidate if score is NaN/inf or negative under maximize
            is_valid = np.isfinite(val_metric)
            if spec.metric_direction == "maximize" and val_metric < 0.0:
                is_valid = False

            cand_status = "evaluated_development" if is_valid else "rejected"
            rejection_reason = None if is_valid else f"Unacceptable validation score: {val_metric}"

            trials_dir = workflow_dir / "trials" / cand_id
            trials_dir.mkdir(parents=True, exist_ok=True)

            scorecard = {
                "bundle_version": "2.0.0",
                "candidate_id": cand_id,
                "project_id": project_id,
                "evaluation_scope": "development_validation",
                "primary_metric": spec.metric_name,
                "primary_metric_score": float(val_metric) if np.isfinite(val_metric) else None,
                "primary_metric_direction": spec.metric_direction,
                "dataset_sha256": dataset_sha256,
                "split_sha256": split_sha256,
                "weights_sha256": weights_sha256,
                "source_sha256": source_sha256,
                "environment_id": env_id,
                "seed": training_config.get("seed") if training_config else spec.seed,
                "audit_status": "UNVERIFIED_DEVELOPMENT",
            }
            scorecard_bytes = json.dumps(scorecard, indent=2).encode("utf-8")
            (trials_dir / "scorecard.json").write_bytes(scorecard_bytes)
            (output_dir / "scorecard.json").write_bytes(scorecard_bytes)
            scorecard_sha = hashlib.sha256(scorecard_bytes).hexdigest()

            trial_record = {
                "project_id": project_id,
                "candidate_id": cand_id,
                "status": cand_status,
                "metric_name": spec.metric_name,
                "metric_score": float(val_metric) if np.isfinite(val_metric) else None,
                "validation_loss": eval_res.get("loss"),
                "dataset_sha256": dataset_sha256,
                "split_sha256": split_sha256,
                "weights_sha256": weights_sha256,
                "source_sha256": source_sha256,
                "environment_id": env_id,
                "training_config": training_config,
                "seed": training_config.get("seed") if training_config else spec.seed,
                "scorecard_hash": scorecard_sha,
                "replay_guarantee": replay_guarantee,
                "checkpoint_ref": spec.resume_checkpoint,
                "rejection_reason": rejection_reason,
                "evaluated_at": time.time(),
            }
            (trials_dir / "trial.json").write_text(json.dumps(trial_record, indent=2), encoding="utf-8")

            # Update workflow results.json while strictly preserving user pins
            results_path = workflow_dir / "results.json"
            results_data = {}
            if results_path.is_file():
                try:
                    results_data = json.loads(results_path.read_text(encoding="utf-8"))
                except Exception:
                    results_data = {}

            candidates_list = results_data.get("candidates", [])
            existing_idx = next((i for i, c in enumerate(candidates_list) if c.get("candidate_id") == cand_id), None)
            if existing_idx is not None:
                candidates_list[existing_idx] = trial_record
            else:
                candidates_list.append(trial_record)
            results_data["candidates"] = candidates_list

            pinned = results_data.get("pinned_candidate")
            if pinned:
                results_data["selected_candidate"] = pinned
            else:
                eligible = [
                    c for c in candidates_list
                    if c.get("status") == "evaluated_development" and c.get("metric_score") is not None
                ]
                if eligible:
                    if spec.metric_direction == "maximize":
                        best = max(eligible, key=lambda c: float(c["metric_score"]))
                    else:
                        best = min(eligible, key=lambda c: float(c["metric_score"]))
                    results_data["selected_candidate"] = best["candidate_id"]
                    results_data["best_cv_score"] = float(best["metric_score"])

            results_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

            eval_res["project_id"] = project_id
            eval_res["weights_sha256"] = weights_sha256
            eval_res["scorecard_hash"] = scorecard_sha
            eval_res["checkpoint_ref"] = spec.resume_checkpoint
            eval_res["candidate_id"] = cand_id
            eval_res["dataset_sha256"] = dataset_sha256
            eval_res["split_sha256"] = split_sha256
            eval_res["source_sha256"] = source_sha256
            eval_res["environment_id"] = env_id
            eval_res["seed"] = training_config.get("seed") if training_config else spec.seed
            eval_res["training_config"] = training_config
            eval_res["replay_guarantee"] = replay_guarantee
            (output_dir / "evaluation.json").write_text(json.dumps(eval_res, indent=2), encoding="utf-8")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=weights_sha256,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                status="succeeded" if is_valid else "failed",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=["result.json", "evaluation.json", "scorecard.json"],
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric) if np.isfinite(val_metric) else None,
                validation_loss=eval_res.get("loss"),
                reason=rejection_reason,
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return 0 if is_valid else 1

        elif op == "custom_export":
            if not dataset_path:
                raise ValueError("custom_export requires --dataset")
            (
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                target_col,
                dataset_sha256,
                split_sha256,
                metric_name,
                metric_direction,
            ) = load_dataset_with_protocol(
                dataset_path=dataset_path,
                workflow_dir=workflow_dir,
                target_column=spec.target_column,
                expected_metric=spec.metric_name,
            )

            # 1. Preserve explicit user pins first
            results_path = workflow_dir / "results.json"
            if results_path.is_file():
                results_data = json.loads(results_path.read_text(encoding="utf-8"))
                pinned = results_data.get("pinned_candidate")
                if pinned and pinned != cand_id:
                    raise ValueError(
                        f"Explicit user pin is set to {pinned!r}; cannot export candidate {cand_id!r} without operator override"
                    )

            # 2. Gate: Candidate must be evaluated and approved in workflow
            trial_file = workflow_dir / "trials" / cand_id / "trial.json"
            if not trial_file.is_file():
                raise ValueError(f"Candidate {cand_id!r} has not been evaluated; export refused")

            trial_record = json.loads(trial_file.read_text(encoding="utf-8"))
            if trial_record.get("status") not in ("evaluated", "evaluated_development", "verified_development_only"):
                raise ValueError(
                    f"Candidate {cand_id!r} has status {trial_record.get('status')!r}; "
                    "rejected or unaudited candidate cannot be exported as approved or deployable"
                )

            evaluated_weights_sha = trial_record.get("weights_sha256")
            if not evaluated_weights_sha:
                raise ValueError(f"Candidate {cand_id!r} trial record missing weights_sha256")

            if not spec.resume_checkpoint:
                raise ValueError("Evaluation and export require an explicit checkpoint")

            resume_dir = resolve_safe_checkpoint_dir(workflow_dir, spec.resume_checkpoint, spec.job_id)
            model.load_checkpoint(
                resume_dir,
                expected_dataset_sha256=dataset_sha256,
                expected_split_sha256=split_sha256,
                expected_candidate_id=cand_id,
                expected_project_id=project_id,
                expected_source_sha256=source_sha256 if source_sha256 else None,
                expected_environment_id=env_id if env_id else None,
                expected_weights_sha256=evaluated_weights_sha,
            )

            export_weights_file = resume_dir / "weights.pt"
            if not export_weights_file.is_file():
                step_dirs = [d for d in resume_dir.glob("step_*") if d.is_dir() and (d / "weights.pt").is_file()]
                if step_dirs:
                    step_dirs.sort(key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0)
                    export_weights_file = step_dirs[-1] / "weights.pt"
                else:
                    raise FileNotFoundError(f"Model weights.pt not found in checkpoint {resume_dir}")

            export_weights_sha = file_sha256(export_weights_file)
            if export_weights_sha != evaluated_weights_sha:
                raise ValueError(
                    f"Export weights hash {export_weights_sha} does not match evaluated weights hash {evaluated_weights_sha}; "
                    "candidate was altered after evaluation"
                )

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            # Quality gate: Reject candidate if score is NaN/inf
            if not np.isfinite(val_metric):
                raise ValueError(f"Export validation score is non-finite: {val_metric}")

            # Safe destination check
            bundle_target = workflow_dir / "bundle"
            if bundle_target.exists():
                raise FileExistsError(f"Bundle destination already exists: {bundle_target}")

            packager = CustomBundlePackager()
            task_type = "classification" if (y_val.nunique() <= 10 or isinstance(y_val.dtype, pd.CategoricalDtype)) else "regression"
            raw_examples = X_val.head(10).to_dict(orient="records")

            receipt = packager.package(
                candidate=model,
                output_bundle_dir=bundle_target,
                target_column=target_col,
                task_type=task_type,
                primary_metric=spec.metric_name,
                primary_metric_score=float(val_metric),
                metric_direction=spec.metric_direction,
                feature_names=feature_cols,
                raw_example_records=raw_examples,
                candidate_id=cand_id,
                accepted_evaluation=trial_record,
                source_weights_path=export_weights_file,
                project_id=project_id,
            )

            # Copy top-level bundle descriptor files to output directory for supervisor verification
            shutil.copy2(bundle_target / "manifest.json", output_dir / "bundle_manifest.json")
            shutil.copy2(bundle_target / "scorecard.json", output_dir / "scorecard.json")
            shutil.copy2(bundle_target / "final.json", output_dir / "final.json")
            shutil.copy2(bundle_target / "verification.json", output_dir / "verification.json")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                project_id=project_id,
                candidate_id=cand_id,
                weights_sha256=export_weights_sha,
                dataset_sha256=dataset_sha256,
                split_sha256=split_sha256,
                source_sha256=source_sha256,
                environment_id=env_id,
                training_config=training_config,
                replay_guarantee=replay_guarantee,
                status="succeeded",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=["result.json", "manifest.json", "scorecard.json", "verification.json", "final.json"],
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric),
                validation_loss=eval_res.get("loss"),
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.info("custom_export succeeded for %s: %s = %.4f", cand_id, spec.metric_name, float(val_metric))
            return 0

        else:
            raise ValueError(f"Unsupported operation: {op}")

    except Exception as exc:
        duration = time.time() - t_start
        is_oom = isinstance(exc, MemoryError) or "out of memory" in str(exc).lower()
        err_msg = str(exc)
        logger.error("Operation %s failed: %s", op, err_msg, exc_info=True)

        status = "needs_resources" if is_oom else "failed"
        result = TrainingResultContract(
            job_id=job_id,
            status=status,
            candidate_id=cand_id,
            step_reached=getattr(model, "current_step", 0) if model else 0,
            epoch_reached=getattr(model, "current_epoch", 0) if model else 0,
            peak_memory_bytes=get_peak_rss(),
            duration_seconds=duration,
            artifacts=[],
            limitations=[err_msg],
            reason=err_msg,
        )
        try:
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        except Exception:
            pass
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="MLRescurix Local Custom Scientific Worker")
    parser.add_argument("--operation", required=False, help="Custom operation to run")
    parser.add_argument("--spec", required=True, help="Path to job spec JSON file")
    parser.add_argument("--output", required=True, help="Path to staging output directory")
    parser.add_argument("--dataset", required=False, help="Path to input dataset CSV")
    parser.add_argument("--workflow", required=False, default="/tmp/workflow", help="Path to workflow dir")
    parser.add_argument("--candidate", required=False, help="Candidate identifier")
    parser.add_argument("--test-access-id", required=False, help="Test access gate identity")

    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    output_dir = Path(args.output).resolve()
    dataset_path = Path(args.dataset).resolve() if args.dataset else None
    workflow_dir = Path(args.workflow).resolve()

    code = run_worker(
        operation=args.operation,
        spec_path=spec_path,
        output_dir=output_dir,
        dataset_path=dataset_path,
        workflow_dir=workflow_dir,
        candidate_id=args.candidate,
        test_access_id=args.test_access_id,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
