"""Integration tests for local scientific custom worker subprocess lifecycle."""

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


def test_custom_worker_cli_train_resume_export(tmp_path: Path, test_dataset: Path) -> None:
    output_dir = tmp_path / "staging"
    workflow_dir = tmp_path / "workflow"
    spec_path = tmp_path / "job_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_train_001",
        project_id="proj_001",
        task_id="task_001",
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
            str(test_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_train",
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
    assert ckpt.step == 10
    assert "weights.pt" in ckpt.files

    # 2. Run custom_resume
    resume_output_dir = tmp_path / "staging_resume"
    resume_spec = CustomJobSpecContract(
        job_id="job_resume_002",
        project_id="proj_001",
        task_id="task_001",
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
        resume_checkpoint=str(workflow_dir / "checkpoints" / "job_train_001"),
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
            str(test_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_resume",
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

    # 3. Run custom_export
    export_output_dir = tmp_path / "staging_export"
    export_spec = CustomJobSpecContract(
        job_id="job_export_003",
        project_id="proj_001",
        task_id="task_001",
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
        resume_checkpoint=str(workflow_dir / "checkpoints" / "job_resume_002_resumed"),
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
            str(test_dataset),
            "--workflow",
            str(workflow_dir),
            "--operation",
            "custom_export",
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

    # Verify that the bundle directory itself in workflow/bundle is valid and clean-room ready
    bundle_dir = workflow_dir / "bundle"
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "frozen_model.joblib").exists()
    assert (bundle_dir / "inference.py").exists()


def test_custom_worker_failure_reporting(tmp_path: Path) -> None:
    output_dir = tmp_path / "staging_err"
    spec_path = tmp_path / "invalid_spec.json"

    spec = CustomJobSpecContract(
        job_id="job_fail_001",
        project_id="proj_001",
        task_id="task_001",
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
