"""Executable Candidate Runner for Phase P4.

Implements the CLI fit/predict protocol, manifest validation, stable row key
and class order verification, fold-isolated cross-validation, and diagnostic
slice evaluation with min_slice_support suppression.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold, TimeSeriesSplit

from .workflow import WorkflowError
from .diagnostic_bridge import DiagnosticBridge, DiagnosticReceipt
try:
    from .fe_ablation import PurgedTimeSeriesSplit, build_cv_splitter
except ImportError:
    PurgedTimeSeriesSplit = None  # type: ignore
    build_cv_splitter = None  # type: ignore


class CandidateExecutionError(WorkflowError):
    """Execution error during candidate fit/predict/validation with attached diagnostic receipt."""

    def __init__(self, message: str, diagnostic_receipt: DiagnosticReceipt | None = None):
        super().__init__(message)
        self.diagnostic_receipt = diagnostic_receipt


def compute_source_hash(directory: Path | str) -> str:
    """Compute deterministic SHA-256 hash across all candidate source files."""
    d = Path(directory)
    if not d.exists() or not d.is_dir():
        raise WorkflowError(f"candidate directory '{d}' does not exist or is not a directory")

    files: list[tuple[str, bytes]] = []
    for root, _, filenames in os.walk(d):
        rel_dir = Path(root).relative_to(d)
        if any(part.startswith(".") or part == "__pycache__" for part in rel_dir.parts):
            continue
        for fn in filenames:
            if fn.startswith(".") or fn.endswith(".pyc") or fn.endswith(".staging") or fn == "manifest.json":
                continue
            fpath = Path(root) / fn
            rel_file = (rel_dir / fn).as_posix()
            try:
                content = fpath.read_bytes()
                files.append((rel_file, content))
            except OSError as e:
                raise WorkflowError(f"cannot read candidate file '{rel_file}': {e}") from e

    files.sort(key=lambda x: x[0])
    hasher = hashlib.sha256()
    for rel_path, content in files:
        file_sha = hashlib.sha256(content).hexdigest()
        hasher.update(f"{rel_path}:{len(content)}:{file_sha}\n".encode("utf-8"))
    return hasher.hexdigest()


def validate_candidate_manifest(manifest: dict[str, Any], candidate_dir: Path | str) -> dict[str, Any]:
    """Validate candidate manifest against Phase P4 contract."""
    d = Path(candidate_dir)
    if not manifest:
        raise WorkflowError("candidate manifest cannot be empty")

    required = ["candidate_id", "name", "task_type", "entrypoint", "supported_operations"]
    for field in required:
        if field not in manifest:
            raise WorkflowError(f"candidate manifest missing required field: '{field}'")

    cid = manifest["candidate_id"]
    if not isinstance(cid, str) or not cid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-:" for c in cid):
        raise WorkflowError(f"invalid candidate_id: '{cid}'")

    task_type = manifest["task_type"]
    if task_type not in ("binary", "multiclass", "regression"):
        raise WorkflowError(f"unsupported task_type in candidate manifest: '{task_type}'")

    ops = manifest["supported_operations"]
    if not isinstance(ops, list) or "fit" not in ops or "predict" not in ops:
        raise WorkflowError("candidate supported_operations must contain 'fit' and 'predict'")

    entrypoint = manifest["entrypoint"]
    if not isinstance(entrypoint, str) or not entrypoint:
        raise WorkflowError("candidate entrypoint must be a non-empty string")

    stable_row_keys = manifest.get("stable_row_keys", "row_id")
    if not isinstance(stable_row_keys, str) or not stable_row_keys:
        raise WorkflowError("stable_row_keys must be a non-empty string")

    class_order = manifest.get("class_order")
    if task_type in ("binary", "multiclass"):
        if not isinstance(class_order, list) or len(class_order) < 2:
            raise WorkflowError(f"classification candidate manifest requires non-empty class_order list, got: {class_order}")
        if len(set(class_order)) != len(class_order):
            raise WorkflowError(f"duplicate classes in manifest class_order: {class_order}")
    else:
        if class_order is not None and not isinstance(class_order, list):
            raise WorkflowError("regression candidate class_order must be null or list")

    computed_hash = compute_source_hash(d)
    declared_hash = manifest.get("source_hash")
    if declared_hash and declared_hash != computed_hash:
        raise WorkflowError(
            f"candidate source hash mismatch: declared '{declared_hash}', computed '{computed_hash}'. "
            "Source files have been modified without updating the manifest hash."
        )

    res = dict(manifest)
    res["stable_row_keys"] = stable_row_keys
    res["source_hash"] = computed_hash
    return res


def validate_predictions(
    predictions_df: pd.DataFrame,
    expected_row_ids: list[Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate prediction dataframe against stable row keys, class order, and bounds."""
    row_key = manifest.get("stable_row_keys", "row_id")
    task_type = manifest.get("task_type", "binary")
    class_order = manifest.get("class_order")

    if row_key not in predictions_df.columns:
        raise WorkflowError(f"row misalignment: prediction output missing stable row key column '{row_key}'")

    pred_ids = predictions_df[row_key].tolist()
    if len(pred_ids) != len(expected_row_ids):
        raise WorkflowError(
            f"row misalignment: prediction row count ({len(pred_ids)}) does not match expected row count ({len(expected_row_ids)})"
        )

    if pred_ids != expected_row_ids:
        missing = set(expected_row_ids) - set(pred_ids)
        extra = set(pred_ids) - set(expected_row_ids)
        if missing or extra:
            raise WorkflowError(
                f"row misalignment: prediction row keys do not match input row keys (missing {len(missing)}, extra {len(extra)})"
            )
        # Slices or reordering: ensure exact aligned mapping
        sorter = {k: i for i, k in enumerate(expected_row_ids)}
        predictions_df = predictions_df.iloc[predictions_df[row_key].map(sorter).argsort()].reset_index(drop=True)

    if "pred" not in predictions_df.columns:
        raise WorkflowError("prediction output missing required 'pred' column")

    preds = predictions_df["pred"].to_numpy()

    if task_type == "regression":
        if not np.issubdtype(preds.dtype, np.number) or not np.isfinite(preds).all():
            raise WorkflowError("regression predictions must be finite numeric values")
        return {
            "row_ids": expected_row_ids,
            "prediction": preds.tolist(),
            "probabilities": None,
            "classes": None,
        }

    # Classification checks
    expected_classes = class_order if class_order is not None else []
    for val in preds:
        if val not in expected_classes:
            try:
                # Try integer/string coercion if numeric type mismatch
                if int(val) in expected_classes:
                    continue
            except (ValueError, TypeError):
                pass
            raise WorkflowError(f"prediction value '{val}' is not in declared class_order: {expected_classes}")

    prob_cols = [f"prob_{c}" for c in expected_classes]
    missing_prob_cols = [c for c in prob_cols if c not in predictions_df.columns]
    if missing_prob_cols:
        raise WorkflowError(
            f"class order discrepancy: prediction output missing probability columns {missing_prob_cols}. "
            f"Columns must match manifest class order {expected_classes}"
        )

    prob_matrix = predictions_df[prob_cols].to_numpy(dtype=float)
    if not np.isfinite(prob_matrix).all():
        raise WorkflowError("prediction probabilities contain NaN or non-finite values")

    if (prob_matrix < -1e-4).any() or (prob_matrix > 1.0 + 1e-4).any():
        raise WorkflowError("prediction probabilities out of bounded range [0, 1]")

    prob_matrix = np.clip(prob_matrix, 0.0, 1.0)
    row_sums = prob_matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise WorkflowError("prediction probabilities do not sum to 1.0 across classes")

    # Renormalize to exact 1.0
    prob_matrix /= prob_matrix.sum(axis=1, keepdims=True)

    return {
        "row_ids": expected_row_ids,
        "prediction": preds.tolist(),
        "probabilities": prob_matrix.tolist(),
        "classes": expected_classes,
    }


def execute_candidate_cli(
    candidate_dir: Path,
    entrypoint: str,
    action: str,
    cli_args: list[str],
    timeout_s: int = 45,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Execute candidate fit or predict command using the candidate CLI protocol."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(candidate_dir.parent.resolve()) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)

    # Entrypoint can be a module name or script filename
    cmd = [sys.executable]
    if entrypoint.endswith(".py"):
        entry_path = candidate_dir / entrypoint if not Path(entrypoint).is_absolute() else Path(entrypoint)
        cmd.extend([str(entry_path), action])
    else:
        cmd.extend(["-m", entrypoint, action])
    cmd.extend(cli_args)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(candidate_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", f"candidate execution exceeded deadline of {timeout_s}s"
    except Exception as exc:
        return 1, "", str(exc)


def evaluate_diagnostic_slices(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    slice_columns: list[str],
    task_type: str,
    class_order: list[Any] | None,
    min_slice_support: int,
) -> dict[str, Any]:
    """Compute performance metrics across diagnostic slices, enforcing min_slice_support."""
    slice_results: dict[str, Any] = {}

    for col in slice_columns:
        if col not in frame.columns:
            continue

        col_slices: dict[str, Any] = {}
        unique_vals = frame[col].dropna().unique()
        # Bound number of reported slice values to prevent unbounded enumeration
        for val in sorted(unique_vals, key=lambda x: str(x))[:16]:
            val_str = str(val)
            mask = (frame[col] == val).to_numpy()
            n_support = int(mask.sum())

            if n_support < min_slice_support:
                col_slices[val_str] = {
                    "count": n_support,
                    "status": "suppressed",
                    "reason": f"below min_slice_support threshold ({min_slice_support})",
                }
            else:
                slice_y = y_true[mask]
                slice_pred = y_pred[mask]
                metrics: dict[str, float] = {}

                if task_type == "regression":
                    metrics["mae"] = float(mean_absolute_error(slice_y, slice_pred))
                    metrics["rmse"] = float(math.sqrt(mean_squared_error(slice_y, slice_pred)))
                else:
                    try:
                        metrics["balanced_accuracy"] = float(balanced_accuracy_score(slice_y, slice_pred))
                    except Exception:
                        metrics["balanced_accuracy"] = 0.0

                    if y_prob is not None and class_order is not None:
                        slice_prob = y_prob[mask]
                        present_classes = np.unique(slice_y)
                        if len(present_classes) > 1:
                            try:
                                metrics["log_loss"] = float(log_loss(slice_y, slice_prob, labels=class_order))
                            except Exception:
                                pass

                col_slices[val_str] = {
                    "count": n_support,
                    "status": "reported",
                    "metrics": metrics,
                }

        slice_results[col] = col_slices

    return slice_results


def evaluate_candidate_cv(
    dataset_csv: Path | str,
    manifest: dict[str, Any],
    candidate_dir: Path | str,
    folds: int = 3,
    seed: int = 2026,
    diagnostic_slices: list[str] | None = None,
    split_policy: str = "stratified",
    group_column: str | None = None,
    time_column: str | None = None,
    topology: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute harness-managed fold-isolated cross validation on an executable candidate.

    Invariants:
    1. Validation fold feature rows NEVER contain the target column.
    2. Candidate code runs in separate subprocess invocations per fold.
    3. Row keys, predictions, probabilities, and class order are strictly validated.
    4. Diagnostic slices below min_slice_support are suppressed.
    """
    manifest = validate_candidate_manifest(manifest, candidate_dir)
    cid = manifest["candidate_id"]
    entrypoint = manifest["entrypoint"]
    task_type = manifest["task_type"]
    class_order = manifest.get("class_order")
    row_key = manifest.get("stable_row_keys", "row_id")
    target_col = manifest.get("raw_input_schema", {}).get("target", "target")

    df = pd.read_csv(dataset_csv, low_memory=False)
    if target_col not in df.columns:
        raise WorkflowError(f"dataset missing target column '{target_col}'")

    # Ensure stable row_id is present in dataset
    if row_key not in df.columns:
        df[row_key] = np.arange(len(df))

    all_row_ids = df[row_key].tolist()
    y = df[target_col].to_numpy()

    # Determine validation query minimum slice support: max(20, 5% of dataset rows)
    min_slice_support = max(20, int(math.floor(0.05 * len(df))))

    # Select splitter based on topology or split policy
    if topology is not None and build_cv_splitter is not None:
        splitter, top_group_col, top_time_col = build_cv_splitter(topology, df=df, random_state=seed)
        if top_time_col and top_time_col in df.columns:
            df = df.sort_values(by=top_time_col).reset_index(drop=True)
            all_row_ids = df[row_key].tolist()
            y = df[target_col].to_numpy()
            splits = list(splitter.split(df))
        elif top_group_col and top_group_col in df.columns:
            if isinstance(splitter, (StratifiedGroupKFold, GroupKFold)):
                try:
                    splits = list(splitter.split(df, y, groups=df[top_group_col]))
                except Exception:
                    splits = list(GroupKFold(n_splits=folds).split(df, y, groups=df[top_group_col]))
            else:
                splits = list(splitter.split(df, y))
        else:
            if isinstance(splitter, (StratifiedKFold, StratifiedGroupKFold)):
                splits = list(splitter.split(df, y))
            else:
                splits = list(splitter.split(df))
    elif split_policy == "grouped" and group_column and group_column in df.columns:
        if task_type in ("binary", "multiclass"):
            splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
            try:
                splits = list(splitter.split(df, y, groups=df[group_column]))
            except Exception:
                splitter = GroupKFold(n_splits=folds)
                splits = list(splitter.split(df, y, groups=df[group_column]))
        else:
            splitter = GroupKFold(n_splits=folds)
            splits = list(splitter.split(df, y, groups=df[group_column]))
    elif split_policy == "chronological" and time_column and time_column in df.columns:
        df = df.sort_values(by=time_column).reset_index(drop=True)
        all_row_ids = df[row_key].tolist()
        y = df[target_col].to_numpy()
        splitter = PurgedTimeSeriesSplit(n_splits=folds, embargo_pct=0.02)
        splits = list(splitter.split(df))
    elif task_type in ("binary", "multiclass"):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, y))
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(df))

    oof_predictions = [None] * len(df)
    n_classes = len(class_order) if class_order is not None else 0
    oof_probabilities = np.full((len(df), n_classes), np.nan) if n_classes > 0 else None
    fold_details = []

    with tempfile.TemporaryDirectory(prefix="mlr-eval-cv-") as scratch_dir:
        scratch = Path(scratch_dir)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_dir = scratch / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            model_dir = fold_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            # Split / label isolation: validation features file MUST NOT contain the target!
            val_features_df = val_df.drop(columns=[target_col]).copy()

            train_csv = fold_dir / "train.csv"
            val_features_csv = fold_dir / "val_features.csv"
            predictions_csv = fold_dir / "predictions.csv"

            train_df.to_csv(train_csv, index=False)
            val_features_df.to_csv(val_features_csv, index=False)

            # 1. Fit fold model
            fit_args = [
                "--train-path", str(train_csv.resolve()),
                "--model-dir", str(model_dir.resolve()),
            ]
            code, stdout, stderr = execute_candidate_cli(
                candidate_dir, entrypoint, "fit", fit_args, timeout_s=45
            )
            if code != 0:
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=code,
                    stderr=stderr,
                    stdout=stdout,
                    message=f"Candidate '{cid}' fit failed on fold {fold_idx}",
                    candidate_id=cid,
                    stage="fit",
                    candidate_dir=str(candidate_dir),
                    context_metadata={"fold": fold_idx},
                )
                raise CandidateExecutionError(
                    f"Candidate '{cid}' fit failed on fold {fold_idx} (exit code {code}):\n{stderr}\n{stdout}",
                    diagnostic_receipt=diag,
                )

            # 2. Predict on validation features (without labels)
            predict_args = [
                "--input-path", str(val_features_csv.resolve()),
                "--model-dir", str(model_dir.resolve()),
                "--output-path", str(predictions_csv.resolve()),
            ]
            code, stdout, stderr = execute_candidate_cli(
                candidate_dir, entrypoint, "predict", predict_args, timeout_s=45
            )
            if code != 0:
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=code,
                    stderr=stderr,
                    stdout=stdout,
                    message=f"Candidate '{cid}' predict failed on fold {fold_idx}",
                    candidate_id=cid,
                    stage="predict",
                    candidate_dir=str(candidate_dir),
                    context_metadata={"fold": fold_idx},
                )
                raise CandidateExecutionError(
                    f"Candidate '{cid}' predict failed on fold {fold_idx} (exit code {code}):\n{stderr}\n{stdout}",
                    diagnostic_receipt=diag,
                )

            if not predictions_csv.exists():
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=1,
                    stderr=f"Missing predictions file at '{predictions_csv}'",
                    message=f"Candidate '{cid}' did not produce predictions file at '{predictions_csv}'",
                    candidate_id=cid,
                    stage="predict",
                    candidate_dir=str(candidate_dir),
                    context_metadata={"fold": fold_idx},
                )
                raise CandidateExecutionError(
                    f"Candidate '{cid}' did not produce predictions file at '{predictions_csv}'",
                    diagnostic_receipt=diag,
                )

            pred_df = pd.read_csv(predictions_csv, low_memory=False)
            val_row_ids = val_features_df[row_key].tolist()
            try:
                validated = validate_predictions(pred_df, val_row_ids, manifest)
            except WorkflowError as exc:
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=1,
                    stderr=str(exc),
                    message=str(exc),
                    candidate_id=cid,
                    stage="predict",
                    candidate_dir=str(candidate_dir),
                    context_metadata={"fold": fold_idx},
                )
                raise CandidateExecutionError(str(exc), diagnostic_receipt=diag) from exc

            # Record fold OOF outputs
            for i, pos in enumerate(val_idx):
                oof_predictions[pos] = validated["prediction"][i]
                if oof_probabilities is not None and validated["probabilities"] is not None:
                    oof_probabilities[pos] = validated["probabilities"][i]

            fold_details.append({
                "fold": fold_idx,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "status": "succeeded",
            })

    # Validate OOF coverage
    covered_mask = [p is not None for p in oof_predictions]
    if not any(covered_mask):
        raise WorkflowError(f"Candidate '{cid}' produced no valid out-of-fold predictions")

    covered_indices = [i for i, c in enumerate(covered_mask) if c]
    covered_y_true = y[covered_indices]
    covered_y_pred = np.array([oof_predictions[i] for i in covered_indices])
    covered_y_prob = oof_probabilities[covered_indices] if oof_probabilities is not None else None

    # Compute overall official metrics
    overall_metrics: dict[str, float] = {}
    if task_type == "regression":
        overall_metrics["mae"] = float(mean_absolute_error(covered_y_true, covered_y_pred))
        overall_metrics["rmse"] = float(math.sqrt(mean_squared_error(covered_y_true, covered_y_pred)))
        primary_metric = overall_metrics["rmse"]
    else:
        overall_metrics["balanced_accuracy"] = float(balanced_accuracy_score(covered_y_true, covered_y_pred))
        primary_metric = overall_metrics["balanced_accuracy"]
        if covered_y_prob is not None and class_order is not None:
            overall_metrics["log_loss"] = float(log_loss(covered_y_true, covered_y_prob, labels=class_order))

    # Diagnostic slice evaluation with min_slice_support suppression
    slice_cols = diagnostic_slices or []
    slices_evaluated = evaluate_diagnostic_slices(
        df.iloc[covered_indices].reset_index(drop=True),
        covered_y_true,
        covered_y_pred,
        covered_y_prob,
        slice_cols,
        task_type,
        class_order,
        min_slice_support,
    )

    scorecard_payload = {
        "candidate_id": cid,
        "source_hash": manifest["source_hash"],
        "task_type": task_type,
        "primary_metric": primary_metric,
        "metrics": overall_metrics,
        "timestamp_ms": int(time.time() * 1000),
    }
    scorecard_hash = hashlib.sha256(json.dumps(scorecard_payload, sort_keys=True).encode("utf-8")).hexdigest()

    return {
        "status": "evaluated",
        "candidate_id": cid,
        "source_hash": manifest["source_hash"],
        "task_type": task_type,
        "primary_metric": primary_metric,
        "metrics": overall_metrics,
        "diagnostic_slices": slices_evaluated,
        "min_slice_support": min_slice_support,
        "folds": folds,
        "folds_evaluated": fold_details,
        "scorecard_hash": scorecard_hash,
    }
