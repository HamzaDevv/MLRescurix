"""Integration tests for local scientific custom worker subprocess lifecycle."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.custom.contracts import (
    CustomJobSpecContract,
    TrainingCheckpointContract,
    TrainingResultContract,
)
from mlrescurix_worker.custom.worker import resolve_safe_checkpoint_dir


@pytest.fixture
def test_dataset(tmp_path: Path) -> Path:
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "feat1": rng.randn(n),
        "feat2": rng.randn(n),
        "feat3": rng.randn(n),
        "target": rng.randint(0, 2, size=n),
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def setup_dataset_manifest(workflow_dir: Path, dataset_path: Path) -> dict:
    workflow_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    train_idx = list(range(70))
    val_idx = list(range(70, 90))
    test_idx = list(range(90, 100))

    # Generate isolated dev_dataset.csv containing only train and validation rows with __mlr_row_id
    df_full = pd.read_csv(dataset_path)
    df_dev = df_full.iloc[train_idx + val_idx].copy()
    df_dev["__mlr_row_id"] = train_idx + val_idx
    dev_path = workflow_dir / "dev_dataset.csv"
    df_dev.to_csv(dev_path, index=False)
    dev_h = hashlib.sha256(dev_path.read_bytes()).hexdigest()

    manifest = {
        "format": "mlrescurix.dataset.v1",
        "sha256": h,
        "dataset_sha256": h,
        "dev_sha256": dev_h,
        "target_column": "target",
        "feature_columns": ["feat1", "feat2", "feat3"],
        "primary_metric": "accuracy",
        "metric_direction": "maximize",
        "splits": {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        },
    }
    (workflow_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def test_custom_worker_cli_train_resume_evaluate_export(tmp_path: Path, test_dataset: Path) -> None:
    output_dir = tmp_path / "staging"
    workflow_dir = tmp_path / "workflow"
    manifest = setup_dataset_manifest(workflow_dir, test_dataset)
    dev_dataset = workflow_dir / "dev_dataset.csv"
    spec_path = tmp_path / "job_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_train_001",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
        target_column="target",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    # 1. Run custom_train
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(dev_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_train",
            "--candidate",
            "cand_torch_001",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"Worker stdout: {proc.stdout}\nWorker stderr: {proc.stderr}"

    result_json_path = output_dir / "result.json"
    ckpt_json_path = output_dir / "checkpoint.json"
    assert result_json_path.exists()
    assert ckpt_json_path.exists()

    result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
    ckpt_data = json.loads(ckpt_json_path.read_text(encoding="utf-8"))

    res = TrainingResultContract.from_dict(result_data)
    ckpt = TrainingCheckpointContract.from_dict(ckpt_data)

    assert res.status == "succeeded"
    assert res.step_reached == 10
    assert res.candidate_id == "cand_torch_001"
    assert res.project_id == "proj_001"
    assert res.weights_sha256 is not None and len(res.weights_sha256) == 64
    assert res.dataset_sha256 == manifest["dataset_sha256"]
    assert res.replay_guarantee == "unverified"

    assert ckpt.step == 10
    assert ckpt.candidate_id == "cand_torch_001"
    assert ckpt.project_id == "proj_001"
    assert ckpt.weights_sha256 == res.weights_sha256
    assert ckpt.dataset_sha256 == manifest["dataset_sha256"]
    assert ckpt.replay_guarantee == "unverified"
    assert "weights.pt" in ckpt.files

    # 2. Run custom_resume (relative checkpoint name)
    resume_output_dir = tmp_path / "staging_resume"
    resume_spec = CustomJobSpecContract(
        job_id="job_resume_002",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_resume",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=20,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_001",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    resume_spec_path = tmp_path / "resume_spec.json"
    resume_spec_path.write_text(json.dumps(resume_spec.to_dict(), indent=2), encoding="utf-8")

    proc_resume = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(resume_spec_path),
            "--output",
            str(resume_output_dir),
            "--dataset",
            str(dev_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_resume",
            "--candidate",
            "cand_torch_001",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_resume.returncode == 0, f"Worker resume stderr: {proc_resume.stderr}"

    resume_res = TrainingResultContract.from_dict(json.loads((resume_output_dir / "result.json").read_text(encoding="utf-8")))
    assert resume_res.status == "succeeded"
    assert resume_res.step_reached == 20
    assert resume_res.project_id == "proj_001"
    assert resume_res.replay_guarantee == "unverified"

    # 3. Run custom_evaluate to evaluate on immutable validation split
    eval_output_dir = tmp_path / "staging_eval"
    eval_spec = CustomJobSpecContract(
        job_id="job_eval_003",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_evaluate",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=20,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_resume_002_resumed",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    eval_spec_path = tmp_path / "eval_spec.json"
    eval_spec_path.write_text(json.dumps(eval_spec.to_dict(), indent=2), encoding="utf-8")

    proc_eval = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(eval_spec_path),
            "--output",
            str(eval_output_dir),
            "--dataset",
            str(dev_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_evaluate",
            "--candidate",
            "cand_torch_001",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_eval.returncode == 0, f"Worker evaluate stderr: {proc_eval.stderr}"

    trial_path = workflow_dir / "trials" / "cand_torch_001" / "trial.json"
    assert trial_path.is_file()
    trial_data = json.loads(trial_path.read_text(encoding="utf-8"))
    assert trial_data["status"] == "evaluated_development"
    assert trial_data["metric_score"] >= 0.0
    assert "weights_sha256" in trial_data
    assert len(trial_data["weights_sha256"]) == 64

    results_path = workflow_dir / "results.json"
    assert results_path.is_file()

    # 4. Run custom_export to generate clean-room delivery bundle
    export_output_dir = tmp_path / "staging_export"
    export_spec = CustomJobSpecContract(
        job_id="job_export_004",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_export",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=20,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_resume_002_resumed",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    export_spec_path = tmp_path / "export_spec.json"
    export_spec_path.write_text(json.dumps(export_spec.to_dict(), indent=2), encoding="utf-8")

    proc_export = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(export_spec_path),
            "--output",
            str(export_output_dir),
            "--dataset",
            str(dev_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_export",
            "--candidate",
            "cand_torch_001",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc_export.returncode == 0, f"Worker export stderr: {proc_export.stderr}"

    export_res = TrainingResultContract.from_dict(json.loads((export_output_dir / "result.json").read_text(encoding="utf-8")))
    assert export_res.status == "succeeded"
    assert (export_output_dir / "bundle_manifest.json").exists()
    assert (export_output_dir / "scorecard.json").exists()
    assert (export_output_dir / "verification.json").exists()
    assert (export_output_dir / "final.json").exists()

    # Verify bundle directory contents in workflow/bundle
    bundle_dir = workflow_dir / "bundle"
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "frozen_model.joblib").exists()
    assert (bundle_dir / "weights.pt").exists()
    assert (bundle_dir / "inference.py").exists()

    verif_data = json.loads((bundle_dir / "verification.json").read_text(encoding="utf-8"))
    assert verif_data["weights_sha256"] == trial_data["weights_sha256"]
    assert verif_data["all_passed"] is False

    scorecard_data = json.loads((bundle_dir / "scorecard.json").read_text(encoding="utf-8"))
    assert scorecard_data["audit_status"] == "UNVERIFIED_DEVELOPMENT"


def test_custom_worker_missing_manifest_fails_closed(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that custom worker strictly requires dataset_manifest.json and refuses to invent splits."""
    output_dir = tmp_path / "staging_no_manifest"
    workflow_dir = tmp_path / "workflow_empty"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    spec_path = tmp_path / "job_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_train_nomanifest",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_1",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(test_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_train",
            "--candidate",
            "cand_1",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert (output_dir / "result.json").exists()
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("dataset_manifest.json not found" in lim for lim in res.limitations)


def test_custom_worker_export_unverified_candidate_rejected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that custom_export rejects candidates that were never evaluated or were rejected."""
    output_dir = tmp_path / "staging_unverified"
    workflow_dir = tmp_path / "workflow_uneval"
    setup_dataset_manifest(workflow_dir, test_dataset)
    spec_path = tmp_path / "export_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_export_unverified",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_never_evaluated",
        generation=1,
        intent_revision=1,
        operation="custom_export",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
        target_column="target",
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(workflow_dir / "dev_dataset.csv"),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_export",
            "--candidate",
            "cand_never_evaluated",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert (output_dir / "result.json").exists()
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("has not been evaluated" in lim for lim in res.limitations)


def test_custom_worker_failure_reporting(tmp_path: Path) -> None:
    output_dir = tmp_path / "staging_err"
    spec_path = tmp_path / "invalid_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_fail_001",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_fail",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="non_existent",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(tmp_path / "does_not_exist.csv"),
            "--workflow",
            str(tmp_path / "workflow"),
            "--operation",
            "custom_train",
            "--candidate",
            "cand_fail",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert (output_dir / "result.json").exists()
    res_data = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    res = TrainingResultContract.from_dict(res_data)
    assert res.status == "failed"
    assert len(res.limitations) > 0


def test_custom_worker_resume_checkpoint_traversal_rejected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that directory traversal and arbitrary paths in resume_checkpoint are strictly rejected."""
    workflow_dir = tmp_path / "workflow_trav"
    setup_dataset_manifest(workflow_dir, test_dataset)
    output_dir = tmp_path / "staging_trav"

    # Contract validation checks
    for bad_ckpt in ("../escaped", "/etc/passwd", "\\windows\\system32", "", "a" * 300, "foo/../../bar", "C:test", "null\0byte"):
        with pytest.raises(ValueError, match="invalid resume_checkpoint reference"):
            CustomJobSpecContract(
                job_id="job_trav",
                project_id="p1",
                task_id="t1",
                generation=1,
                intent_revision=1,
                operation="custom_resume",
                model_plugin="tiny_torch",
                dataset_id="ds1",
                metric_name="accuracy",
                metric_direction="maximize",
                max_steps=10,
                batch_size=16,
                resume_checkpoint=bad_ckpt,
            )

    # CLI process invocation with raw JSON containing traversal path
    spec_path = tmp_path / "spec_trav.json"
    raw_spec = {
        "schema_version": "v2",
        "job_id": "job_trav_cli",
        "project_id": "p1",
        "task_id": "t1",
        "generation": 1,
        "intent_revision": 1,
        "operation": "custom_resume",
        "model_plugin": "tiny_torch",
        "dataset_id": "ds1",
        "metric_name": "accuracy",
        "metric_direction": "maximize",
        "max_steps": 10,
        "batch_size": 16,
        "resume_checkpoint": "../../../etc/passwd",
    }
    spec_path.write_text(json.dumps(raw_spec, indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(test_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_resume",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    assert (output_dir / "result.json").exists()
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("invalid resume_checkpoint reference" in lim for lim in res.limitations)


def test_custom_worker_data_leakage_detected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that data leakage into validation or sealed test partitions fails closed."""
    workflow_dir = tmp_path / "workflow_leak"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(test_dataset.read_bytes()).hexdigest()

    # Leaking train and val splits
    leaky_manifest = {
        "format": "mlrescurix.dataset.v1",
        "sha256": h,
        "dataset_sha256": h,
        "target_column": "target",
        "feature_columns": ["feat1", "feat2", "feat3"],
        "primary_metric": "accuracy",
        "metric_direction": "maximize",
        "splits": {
            "train": [0, 1, 2, 3, 4],
            "validation": [4, 5, 6],  # Row 4 is in both!
            "test": [7, 8, 9],
        },
    }

    # Generate dev_dataset with __mlr_row_id matching train and validation rows
    df_full = pd.read_csv(test_dataset)
    df_dev = df_full.iloc[:7].copy()
    df_dev["__mlr_row_id"] = list(range(7))
    dev_path = workflow_dir / "dev_dataset.csv"
    df_dev.to_csv(dev_path, index=False)
    leaky_manifest["dev_sha256"] = hashlib.sha256(dev_path.read_bytes()).hexdigest()
    (workflow_dir / "dataset_manifest.json").write_text(json.dumps(leaky_manifest, indent=2), encoding="utf-8")

    output_dir = tmp_path / "staging_leak"
    spec_path = tmp_path / "spec_leak.json"
    spec = CustomJobSpecContract(
        job_id="job_leak",
        project_id="p1",
        task_id="t1",
        candidate_id="cand_leak",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds1",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(dev_path),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_train",
            "--candidate",
            "cand_leak",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("Data leakage detected" in lim for lim in res.limitations)


def test_custom_worker_metric_mismatch_detected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that silent metric substitution is rejected."""
    workflow_dir = tmp_path / "workflow_metric"
    setup_dataset_manifest(workflow_dir, test_dataset)  # Manifest specifies accuracy
    output_dir = tmp_path / "staging_metric"

    spec_path = tmp_path / "spec_metric.json"
    spec = CustomJobSpecContract(
        job_id="job_metric",
        project_id="p1",
        task_id="t1",
        candidate_id="cand_metric",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds1",
        metric_name="log_loss",  # Mismatched metric!
        metric_direction="minimize",
        max_steps=10,
        batch_size=16,
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(workflow_dir / "dev_dataset.csv"),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_train",
            "--candidate",
            "cand_metric",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("Metric mismatch" in lim for lim in res.limitations)


def test_custom_worker_pinned_candidate_preserved(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that explicit user pins in results.json prevent unauthorized candidate export."""
    workflow_dir = tmp_path / "workflow_pin"
    setup_dataset_manifest(workflow_dir, test_dataset)

    # Pretend cand_torch_001 was evaluated
    trials_dir = workflow_dir / "trials" / "cand_torch_001"
    trials_dir.mkdir(parents=True, exist_ok=True)
    trial_data = {
        "candidate_id": "cand_torch_001",
        "status": "evaluated_development",
        "metric_score": 0.88,
        "dataset_sha256": hashlib.sha256(test_dataset.read_bytes()).hexdigest(),
    }
    (trials_dir / "trial.json").write_text(json.dumps(trial_data, indent=2), encoding="utf-8")

    # Set explicit user pin to a DIFFERENT candidate
    results_data = {
        "pinned_candidate": "cand_user_pinned_apex",
        "selected_candidate": "cand_user_pinned_apex",
    }
    (workflow_dir / "results.json").write_text(json.dumps(results_data, indent=2), encoding="utf-8")

    output_dir = tmp_path / "staging_pin"
    spec_path = tmp_path / "spec_pin.json"
    spec = CustomJobSpecContract(
        job_id="job_pin",
        project_id="p1",
        task_id="t1",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_export",
        model_plugin="tiny_torch",
        dataset_id="ds1",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
        target_column="target",
    )
    spec_path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlrescurix_worker.custom.worker",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir),
            "--dataset",
            str(workflow_dir / "dev_dataset.csv"),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_export",
            "--candidate",
            "cand_torch_001",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert proc.returncode == 1
    res = TrainingResultContract.from_dict(json.loads((output_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert any("Explicit user pin is set" in lim for lim in res.limitations)


def test_resolve_safe_checkpoint_dir_boundaries(tmp_path: Path) -> None:
    """Verifies that resolve_safe_checkpoint_dir rejects sandbox escapes and root directory references."""
    workflow = tmp_path / "wf"

    # Valid relative checkpoint
    resolved = resolve_safe_checkpoint_dir(workflow, "ckpt_1", "default_job")
    assert resolved == (workflow / "checkpoints" / "ckpt_1").resolve()

    # None defaults to default_job_id
    default_resolved = resolve_safe_checkpoint_dir(workflow, None, "default_job")
    assert default_resolved == (workflow / "checkpoints" / "default_job").resolve()

    # Root checkpoints directory itself rejected (must be a specific checkpoint subpath)
    with pytest.raises(ValueError, match="cannot be the root checkpoints directory"):
        resolve_safe_checkpoint_dir(workflow, ".", "default_job")

    # Escaping attempts
    for escape in ("..", "../checkpoints", "/tmp", "\\evil", "a/../../b"):
        with pytest.raises(ValueError):
            resolve_safe_checkpoint_dir(workflow, escape, "default_job")


@pytest.mark.parametrize("mutation", [
    "negative",
    "duplicate",
    "missing_hash",
    "invalid_format",
    "hash_mismatch",
    "target_feature",
    "overlap_train_val",
    "leak_test",
])
def test_protocol_rejects_unsafe_manifest(tmp_path: Path, test_dataset: Path, mutation: str) -> None:
    from mlrescurix_worker.custom.worker import load_dataset_with_protocol

    workflow = tmp_path / "workflow"
    manifest = setup_dataset_manifest(workflow, test_dataset)
    if mutation == "negative":
        manifest["splits"]["train"][0] = -1
    elif mutation == "duplicate":
        manifest["splits"]["train"].append(0)
    elif mutation == "missing_hash":
        manifest.pop("sha256", None)
        manifest.pop("dataset_sha256", None)
    elif mutation == "invalid_format":
        manifest["format"] = "bogus.dataset.v0"
    elif mutation == "hash_mismatch":
        manifest["sha256"] = "a" * 64
        manifest["dataset_sha256"] = "a" * 64
    elif mutation == "target_feature":
        manifest["feature_columns"].append("target")
    elif mutation == "overlap_train_val":
        manifest["splits"]["validation"].append(manifest["splits"]["train"][0])
    elif mutation == "leak_test":
        manifest["splits"]["train"].append(manifest["splits"]["test"][0])

    (workflow / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset_with_protocol(test_dataset, workflow)


def test_checkpoint_candidate_identity_requires_exact_match(tmp_path: Path) -> None:
    from mlrescurix_worker.custom.torch_candidate import TinyTorchNeuralCandidate

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "COMMIT").write_text("committed", encoding="utf-8")
    (checkpoint / "manifest.json").write_text(json.dumps({"candidate_id": "candidate-extended"}), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate mismatch"):
        TinyTorchNeuralCandidate().load_checkpoint(checkpoint, expected_candidate_id="candidate")


def test_custom_worker_export_different_checkpoint_rejected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that exporting a checkpoint with different weights than evaluated is rejected."""
    from mlrescurix_worker.custom.worker import run_worker

    workflow = tmp_path / "workflow"
    setup_dataset_manifest(workflow, test_dataset)
    dev_dataset = workflow / "dev_dataset.csv"

    # 1. Train checkpoint A (10 steps)
    spec_a = CustomJobSpecContract(
        job_id="job_train_a",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_test_001",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
        target_column="target",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_a_path = tmp_path / "spec_a.json"
    spec_a_path.write_text(json.dumps(spec_a.to_dict()), encoding="utf-8")
    ret_train_a = run_worker("custom_train", spec_a_path, tmp_path / "out_train_a", dev_dataset, workflow, candidate_id="cand_test_001")
    assert ret_train_a == 0

    # 2. Train checkpoint B (resume to 20 steps)
    spec_b = CustomJobSpecContract(
        job_id="job_train_b",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_test_001",
        generation=1,
        intent_revision=1,
        operation="custom_resume",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=20,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_a",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_b_path = tmp_path / "spec_b.json"
    spec_b_path.write_text(json.dumps(spec_b.to_dict()), encoding="utf-8")
    ret_train_b = run_worker("custom_resume", spec_b_path, tmp_path / "out_train_b", dev_dataset, workflow, candidate_id="cand_test_001")
    assert ret_train_b == 0

    # 3. Evaluate checkpoint A
    spec_eval_a = CustomJobSpecContract(
        job_id="job_eval_a",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_test_001",
        generation=1,
        intent_revision=1,
        operation="custom_evaluate",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=10,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_a",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_eval_a_path = tmp_path / "spec_eval_a.json"
    spec_eval_a_path.write_text(json.dumps(spec_eval_a.to_dict()), encoding="utf-8")
    ret_eval = run_worker("custom_evaluate", spec_eval_a_path, tmp_path / "out_eval_a", dev_dataset, workflow, candidate_id="cand_test_001")
    assert ret_eval == 0

    # 4. Attempt export with checkpoint B (different weights than evaluated checkpoint A)
    spec_export_b = CustomJobSpecContract(
        job_id="job_export_b",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_test_001",
        generation=1,
        intent_revision=1,
        operation="custom_export",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=20,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_b_resumed",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_export_b_path = tmp_path / "spec_export_b.json"
    spec_export_b_path.write_text(json.dumps(spec_export_b.to_dict()), encoding="utf-8")
    ret_export = run_worker("custom_export", spec_export_b_path, tmp_path / "out_export_b", dev_dataset, workflow, candidate_id="cand_test_001")
    assert ret_export == 1

    # Verify result.json records the rejection
    res_data = json.loads((tmp_path / "out_export_b" / "result.json").read_text(encoding="utf-8"))
    assert res_data["status"] == "failed"
    assert "does not match evaluated weights hash" in (res_data.get("reason") or "")


def test_custom_worker_development_data_isolated_from_sealed_test_partition(tmp_path: Path, test_dataset: Path) -> None:
    """Verify returned development frames use isolated dev_dataset.csv excluding test rows."""
    from mlrescurix_worker.custom.worker import load_dataset_with_protocol

    workflow = tmp_path / "workflow"
    setup_dataset_manifest(workflow, test_dataset)
    dev_dataset = workflow / "dev_dataset.csv"

    ret = load_dataset_with_protocol(dev_dataset, workflow)
    assert len(ret) == 10
    X_train, y_train, X_val, y_val, features, target_col, dataset_sha256, split_sha256, metric_name, metric_direction = ret

    # Test rows 90-99 must NEVER appear in train or validation sets
    sealed_rows = set(range(90, 100))
    assert not (set(X_train.index) & sealed_rows)
    assert not (set(X_val.index) & sealed_rows)
    assert len(X_train) == 70
    assert len(X_val) == 20


def test_custom_worker_full_dataset_with_test_rows_fails_closed(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that giving development workers access to full dataset containing test rows fails closed."""
    from mlrescurix_worker.custom.worker import load_dataset_with_protocol

    workflow = tmp_path / "workflow"
    setup_dataset_manifest(workflow, test_dataset)

    with pytest.raises(ValueError, match="Data leakage detected: development worker given access to full dataset"):
        load_dataset_with_protocol(test_dataset, workflow)


def test_custom_worker_bound_identity_mismatch_rejected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that load_checkpoint rejects mutations to dataset_sha256, split_sha256, or candidate_id."""
    from mlrescurix_worker.custom.torch_candidate import TinyTorchNeuralCandidate

    ckpt_dir = tmp_path / "checkpoints" / "job_01"
    model = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, random_seed=42)
    df = pd.read_csv(test_dataset)
    model.train((df[["feat1", "feat2", "feat3"]].iloc[:20], df["target"].iloc[:20]), max_steps=5)

    model.save_checkpoint(
        ckpt_dir,
        step=5,
        candidate_id="cand_orig",
        dataset_sha256="d" * 64,
        split_sha256="s" * 64,
        project_id="proj_orig",
        source_sha256="1" * 64,
        environment_id="env_orig",
        replay_guarantee="unverified",
    )

    verifier = TinyTorchNeuralCandidate(hidden_dim=16, is_classification=True, random_seed=42)

    # Dataset hash mismatch
    with pytest.raises(ValueError, match="Checkpoint dataset mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_dataset_sha256="e" * 64)

    # Split hash mismatch
    with pytest.raises(ValueError, match="Checkpoint split mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_split_sha256="e" * 64)

    # Candidate ID mismatch
    with pytest.raises(ValueError, match="Checkpoint candidate mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_candidate_id="cand_impostor")

    # Project ID mismatch
    with pytest.raises(ValueError, match="Checkpoint project mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_project_id="proj_different")

    # Source hash mismatch
    with pytest.raises(ValueError, match="Checkpoint source mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_source_sha256="2" * 64)

    # Environment ID mismatch
    with pytest.raises(ValueError, match="Checkpoint environment mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_environment_id="env_different")

    # Weights hash mismatch
    with pytest.raises(ValueError, match="Checkpoint weights mismatch"):
        verifier.load_checkpoint(ckpt_dir, expected_weights_sha256="3" * 64)


def test_custom_worker_rejects_test_access_id(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that attempting to pass test_access_id to custom worker fails closed."""
    from mlrescurix_worker.custom.worker import run_worker

    workflow = tmp_path / "workflow_test_access"
    setup_dataset_manifest(workflow, test_dataset)

    spec = CustomJobSpecContract(
        job_id="job_holdout_access",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
    )
    spec_path = tmp_path / "spec_holdout.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    out_dir = tmp_path / "out_holdout"

    ret = run_worker(
        operation="custom_train",
        spec_path=spec_path,
        output_dir=out_dir,
        dataset_path=test_dataset,
        workflow_dir=workflow,
        candidate_id="cand_torch_001",
        test_access_id="unauthorized_test_gate_123",
    )
    assert ret == 1
    res = TrainingResultContract.from_dict(json.loads((out_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert "Holdout access is sealed" in (res.reason or "")


def test_custom_worker_rejects_manifest_without_test_split(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that an audited manifest lacking a sealed 'test' partition is rejected."""
    from mlrescurix_worker.custom.worker import run_worker

    workflow = tmp_path / "workflow_no_test_split"
    workflow.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(test_dataset.read_bytes()).hexdigest()

    manifest_no_test = {
        "format": "mlrescurix.dataset.v1",
        "sha256": h,
        "dataset_sha256": h,
        "target_column": "target",
        "feature_columns": ["feat1", "feat2", "feat3"],
        "primary_metric": "accuracy",
        "metric_direction": "maximize",
        "splits": {
            "train": list(range(70)),
            "validation": list(range(70, 90)),
            # missing "test" split!
        },
    }
    (workflow / "dataset_manifest.json").write_text(json.dumps(manifest_no_test, indent=2), encoding="utf-8")

    spec = CustomJobSpecContract(
        job_id="job_no_test_split",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_torch_001",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
    )
    spec_path = tmp_path / "spec_no_test.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    out_dir = tmp_path / "out_no_test"

    ret = run_worker(
        operation="custom_train",
        spec_path=spec_path,
        output_dir=out_dir,
        dataset_path=test_dataset,
        workflow_dir=workflow,
        candidate_id="cand_torch_001",
    )
    assert ret == 1
    res = TrainingResultContract.from_dict(json.loads((out_dir / "result.json").read_text(encoding="utf-8")))
    assert res.status == "failed"
    assert "missing immutable 'train', 'validation', or 'test' splits" in (res.reason or "")


def test_custom_worker_overwrite_bundle_rejected(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that custom_export fails closed if destination bundle already exists."""
    from mlrescurix_worker.custom.worker import run_worker

    workflow = tmp_path / "workflow"
    setup_dataset_manifest(workflow, test_dataset)
    dev_dataset = workflow / "dev_dataset.csv"

    # Train and evaluate candidate
    spec_train = CustomJobSpecContract(
        job_id="job_train_ovw",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_ovw",
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_train_path = tmp_path / "spec_train_ovw.json"
    spec_train_path.write_text(json.dumps(spec_train.to_dict()), encoding="utf-8")
    run_worker("custom_train", spec_train_path, tmp_path / "out_train", dev_dataset, workflow, candidate_id="cand_ovw")

    spec_eval = CustomJobSpecContract(
        job_id="job_eval_ovw",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_ovw",
        generation=1,
        intent_revision=1,
        operation="custom_evaluate",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_ovw",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_eval_path = tmp_path / "spec_eval_ovw.json"
    spec_eval_path.write_text(json.dumps(spec_eval.to_dict()), encoding="utf-8")
    run_worker("custom_evaluate", spec_eval_path, tmp_path / "out_eval", dev_dataset, workflow, candidate_id="cand_ovw")

    # Pre-create bundle directory
    bundle_target = workflow / "bundle"
    bundle_target.mkdir(parents=True, exist_ok=True)
    (bundle_target / "existing.txt").write_text("pre-existing bundle data")

    spec_export = CustomJobSpecContract(
        job_id="job_export_ovw",
        project_id="proj_001",
        task_id="task_001",
        candidate_id="cand_ovw",
        generation=1,
        intent_revision=1,
        operation="custom_export",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_ovw",
        hidden_dim=16,
        learning_rate=0.01,
        seed=123,
    )
    spec_export_path = tmp_path / "spec_export_ovw.json"
    spec_export_path.write_text(json.dumps(spec_export.to_dict()), encoding="utf-8")
    ret = run_worker("custom_export", spec_export_path, tmp_path / "out_export", dev_dataset, workflow, candidate_id="cand_ovw")
    assert ret == 1

    res_data = json.loads((tmp_path / "out_export" / "result.json").read_text(encoding="utf-8"))
    assert res_data["status"] == "failed"
    assert "already exists" in (res_data.get("reason") or "")


def test_custom_worker_scorecard_generated_on_evaluation(tmp_path: Path, test_dataset: Path) -> None:
    """Verifies that custom_evaluate writes scorecard.json to trials/<cand>/scorecard.json with real metrics."""
    from mlrescurix_worker.custom.worker import run_worker

    workflow = tmp_path / "workflow"
    setup_dataset_manifest(workflow, test_dataset)
    dev_dataset = workflow / "dev_dataset.csv"

    candidate = "cand_eval_sc"
    spec_train = CustomJobSpecContract(
        job_id="job_train_sc",
        project_id="proj_001",
        task_id="task_001",
        candidate_id=candidate,
        generation=1,
        intent_revision=1,
        operation="custom_train",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
        hidden_dim=16,
        learning_rate=0.01,
        seed=456,
    )
    spec_train_path = tmp_path / "spec_train_sc.json"
    spec_train_path.write_text(json.dumps(spec_train.to_dict()), encoding="utf-8")
    assert run_worker("custom_train", spec_train_path, tmp_path / "out_tr_sc", dev_dataset, workflow, candidate_id=candidate) == 0

    spec_eval = CustomJobSpecContract(
        job_id="job_eval_sc",
        project_id="proj_001",
        task_id="task_001",
        candidate_id=candidate,
        generation=1,
        intent_revision=1,
        operation="custom_evaluate",
        model_plugin="tiny_torch",
        dataset_id="ds_001",
        metric_name="accuracy",
        metric_direction="maximize",
        max_steps=5,
        batch_size=16,
        target_column="target",
        resume_checkpoint="job_train_sc",
        hidden_dim=16,
        learning_rate=0.01,
        seed=456,
    )
    spec_eval_path = tmp_path / "spec_eval_sc.json"
    spec_eval_path.write_text(json.dumps(spec_eval.to_dict()), encoding="utf-8")
    assert run_worker("custom_evaluate", spec_eval_path, tmp_path / "out_ev_sc", dev_dataset, workflow, candidate_id=candidate) == 0

    sc_path = workflow / "trials" / candidate / "scorecard.json"
    assert sc_path.is_file()
    sc = json.loads(sc_path.read_text(encoding="utf-8"))
    assert sc["candidate_id"] == candidate
    assert sc["audit_status"] == "UNVERIFIED_DEVELOPMENT"
    assert "weights_sha256" in sc
    assert len(sc["weights_sha256"]) == 64
    assert sc["primary_metric"] == "accuracy"
    assert sc["seed"] == 456

@pytest.mark.parametrize("job_id,candidate_arg,candidate_spec", [
    ("../escape", "safe", "safe"),
    ("safe", "../escape", None),
    ("safe", "other", "expected"),
])
def test_worker_rejects_unsafe_or_mismatched_identities(tmp_path: Path, job_id: str, candidate_arg: str, candidate_spec: str | None) -> None:
    from mlrescurix_worker.custom.worker import run_worker

    spec = CustomJobSpecContract(
        job_id=job_id, project_id="project", task_id="task", candidate_id=candidate_spec,
        generation=1, intent_revision=1, operation="custom_train", model_plugin="tiny_torch",
        dataset_id="dataset", metric_name="accuracy", metric_direction="maximize",
        max_steps=1, batch_size=1,
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    assert run_worker("custom_train", spec_path, tmp_path / "output", None, tmp_path / "workflow", candidate_id=candidate_arg) == 1
    result = json.loads((tmp_path / "output" / "result.json").read_text(encoding="utf-8"))
    assert "identity" in result["reason"] or "Invalid job_id" in result["reason"]
    assert not (tmp_path / "escape").exists()
