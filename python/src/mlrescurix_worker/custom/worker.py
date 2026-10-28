"""Local Scientific Custom Model Worker CLI.

Executes custom model training, resumption, evaluation, and delivery packaging
at the supervised Python process boundary:
- Enforces thread caps and process resource limits.
- Supports deterministic step-boundary checkpointing and exact replay resumption.
- Produces verified result contracts and bundle artifacts conforming to MLRescurix v2.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


def load_dataset(dataset_path: Path, target_column: Optional[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], str]:
    """Loads CSV dataset and produces deterministic train/val splits."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    if target_column:
        if target_column not in df.columns:
            raise ValueError(f"Target column not found: {target_column}")
        target_col = target_column
    elif "target" in df.columns:
        target_col = "target"
    else:
        target_col = df.columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * 0.8)
    if split_idx == 0 or split_idx == len(df):
        X_train, y_train = X, y
        X_val, y_val = X, y
    else:
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]

    return X, y, X_train, y_train, X_val, y_val, feature_cols, target_col


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

    # 1. Parse and validate job spec
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_dict = json.load(f)
    spec = CustomJobSpecContract.from_dict(spec_dict)
    op = operation or spec.operation
    if op != spec.operation:
        raise ValueError("Requested operation does not match job specification")
    if spec.model_plugin != "tiny_torch":
        raise ValueError(f"Unsupported model plugin: {spec.model_plugin}")

    # 2. Configure limits
    configure_process_limits(spec)

    cand_id = candidate_id or getattr(spec, "candidate_id", None) or spec.model_plugin or "tiny_torch"
    model = TinyTorchNeuralCandidate(
        hidden_dim=spec.hidden_dim or 32,
        lr=spec.learning_rate or 0.001,
        batch_size=spec.batch_size or 32,
        random_seed=spec.seed or 42,
    )

    try:
        if op == "custom_train":
            if not dataset_path:
                raise ValueError("custom_train requires --dataset")
            X, y, X_train, y_train, X_val, y_val, feature_cols, target_col = load_dataset(dataset_path, spec.target_column)

            train_state = model.train(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val),
                max_steps=spec.max_steps,
            )

            # Persist checkpoint in workflow directory
            ckpt_dir = workflow_dir / "checkpoints" / spec.job_id
            ckpt_manifest = model.save_checkpoint(ckpt_dir, model.current_step, train_state)

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            ckpt_contract = TrainingCheckpointContract(
                checkpoint_id=f"ckpt_{spec.job_id}_{model.current_step}",
                job_id=spec.job_id,
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
            X, y, X_train, y_train, X_val, y_val, feature_cols, target_col = load_dataset(dataset_path, spec.target_column)

            resume_dir = None
            if spec.resume_checkpoint:
                resume_dir = Path(spec.resume_checkpoint)
            else:
                resume_dir = workflow_dir / "checkpoints" / spec.job_id

            if not resume_dir.exists():
                raise FileNotFoundError(f"Resume checkpoint directory not found: {resume_dir}")

            train_state = model.train(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val),
                resume_from=resume_dir,
                max_steps=spec.max_steps,
            )

            resumed_ckpt_dir = workflow_dir / "checkpoints" / f"{spec.job_id}_resumed"
            ckpt_manifest = model.save_checkpoint(resumed_ckpt_dir, model.current_step, train_state)

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            ckpt_contract = TrainingCheckpointContract(
                checkpoint_id=f"ckpt_{spec.job_id}_{model.current_step}",
                job_id=spec.job_id,
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
            X, y, X_train, y_train, X_val, y_val, feature_cols, target_col = load_dataset(dataset_path, spec.target_column)

            if spec.resume_checkpoint:
                model.load_checkpoint(Path(spec.resume_checkpoint))

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            (output_dir / "evaluation.json").write_text(json.dumps(eval_res, indent=2), encoding="utf-8")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                status="succeeded",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=["result.json", "evaluation.json"],
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric),
                validation_loss=eval_res.get("loss"),
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return 0

        elif op == "custom_export":
            if not dataset_path:
                raise ValueError("custom_export requires --dataset")
            X, y, X_train, y_train, X_val, y_val, feature_cols, target_col = load_dataset(dataset_path, spec.target_column)

            if spec.resume_checkpoint:
                model.load_checkpoint(Path(spec.resume_checkpoint))

            eval_res = model.evaluate((X_val, y_val))
            val_metric = eval_res[spec.metric_name]

            # Package delivery bundle inside workflow directory
            bundle_target = workflow_dir / "bundle"
            if bundle_target.exists():
                raise FileExistsError(f"Bundle destination already exists: {bundle_target}")

            packager = CustomBundlePackager()
            task_type = "classification" if (y.nunique() <= 10 or pd.api.types.is_categorical_dtype(y)) else "regression"
            raw_examples = X.head(10).to_dict(orient="records")

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
            )

            # Copy top-level bundle descriptor files to output directory for supervisor verification
            shutil.copy2(bundle_target / "manifest.json", output_dir / "bundle_manifest.json")
            shutil.copy2(bundle_target / "scorecard.json", output_dir / "scorecard.json")
            shutil.copy2(bundle_target / "final.json", output_dir / "final.json")
            shutil.copy2(bundle_target / "verification.json", output_dir / "verification.json")

            duration = time.time() - t_start
            result = TrainingResultContract(
                job_id=spec.job_id,
                status="succeeded",
                step_reached=model.current_step,
                epoch_reached=model.current_epoch,
                peak_memory_bytes=get_peak_rss(),
                duration_seconds=duration,
                artifacts=["result.json", "bundle_manifest.json", "scorecard.json", "final.json", "verification.json"],
                primary_metric_name=spec.metric_name,
                primary_metric_score=float(val_metric),
                validation_loss=eval_res.get("loss"),
            )
            (output_dir / "result.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            logger.info("custom_export created and verified bundle at %s", bundle_target)
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
            job_id=spec.job_id,
            status=status,
            step_reached=getattr(model, "current_step", 0),
            epoch_reached=getattr(model, "current_epoch", 0),
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
