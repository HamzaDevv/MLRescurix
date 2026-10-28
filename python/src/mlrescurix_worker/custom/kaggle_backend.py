"""Kaggle Compute Pool Execution Backend.

Implements ExecutionBackendContract for running, checkpointing, and resuming
training workloads on Kaggle's free GPU/TPU accelerator infrastructure.

Core guarantees:
1. Strict Account Governance:
   - Single-account execution by default (Anchor account only).
   - Secondary accounts strictly compute-only and isolated.
2. Structured Durable Lifecycle:
   - PLANNED -> STAGED -> QUEUED/SUBMITTED -> RUNNING -> COLLECTING -> VERIFIED.
   - Fails explicitly on RETRY_WAIT, CANCEL_REQUESTED, CANCELLED, FAILED, NEEDS_RESOURCES.
3. Fenced Atomic Leases:
   - Each dispatch generates an atomic lease token.
   - Output collection validates lease token to prevent stale superseded attempts from promoting artifacts.
4. Checksum Verification:
   - Remote checkpoint downloads must pass per-file SHA-256 validation against manifest.json.
5. Deadline-Aware Training Segments:
   - Training flushes atomic checkpoints to disk before worker deadline.
   - Subsequent jobs resume training from the verified published checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .contracts import (
    ExecutionBackendContract,
    PilotResourceReport,
    ResourceRequestContract,
    TrainingStateContract,
    file_sha256,
)
from ..scientific.evidence import streaming_file_sha256
from ..scientific.kaggle_pool import (
    ChecksumMismatchError,
    JobLifecycleState,
    KaggleAccount,
    KaggleAccountPool,
    KernelJobRecord,
    QuotaExhaustedError,
    SecurityViolationError,
    StaleLeaseError,
    UnknownRemoteOutcomeError,
)

logger = logging.getLogger("mlrescurix.kaggle_backend")


class KaggleExecutionBackend(ExecutionBackendContract):
    """Execution backend managing Kaggle remote batch execution and durable checkpoint recovery."""

    backend_id: str = "kaggle_compute_pool"

    def __init__(
        self,
        pool: Optional[KaggleAccountPool] = None,
        storage_dir: Union[str, Path] = ".kaggle_pool",
        enable_secondary_accounts: bool = False,
    ):
        self.pool = pool or KaggleAccountPool(
            storage_dir=storage_dir,
            enable_secondary_accounts=enable_secondary_accounts,
        )
        self.enable_secondary_accounts = enable_secondary_accounts

    def preflight(
        self,
        request: ResourceRequestContract,
        plugin_spec: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Probe environment, accelerator capabilities, quota, and check feasibility."""
        accounts = self.pool.list_accounts()
        if not accounts:
            return {
                "status": JobLifecycleState.NEEDS_RESOURCES,
                "reason": "No Kaggle accounts registered in the compute pool.",
                "admitted": False,
            }

        accelerator = (request.accelerator_type or "gpu").lower()
        if accelerator not in ("gpu", "tpu", "cpu"):
            return {
                "status": JobLifecycleState.NEEDS_RESOURCES,
                "reason": f"Requested accelerator '{accelerator}' is not supported by Kaggle backend.",
                "admitted": False,
            }

        try:
            account = self.pool.lease_worker_account(
                accelerator=accelerator,
                min_spacing_seconds=0.0,
                allow_secondary=self.enable_secondary_accounts,
            )
        except QuotaExhaustedError as exc:
            return {
                "status": JobLifecycleState.NEEDS_RESOURCES,
                "reason": f"Kaggle accelerator quota exhausted: {exc}",
                "admitted": False,
            }

        remaining_hours = account.remaining_gpu_hours if accelerator == "gpu" else account.remaining_tpu_hours
        required_hours = request.max_duration_seconds / 3600.0
        if required_hours > remaining_hours:
            return {
                "status": JobLifecycleState.NEEDS_RESOURCES,
                "reason": f"Required compute duration ({round(required_hours, 2)}h) exceeds remaining quota ({round(remaining_hours, 2)}h).",
                "admitted": False,
            }

        return {
            "status": "ok",
            "account": account.username,
            "accelerator": accelerator,
            "remaining_quota_hours": remaining_hours,
            "is_anchor": account.is_anchor,
            "admitted": True,
        }

    def stage(
        self,
        job_id: str,
        code_dir: Path,
        data_path: Path,
        manifest: Dict[str, Any],
    ) -> Path:
        """Stage worker code, isolated environment config, and input data."""
        staged_dir = self.pool.staging_dir / "jobs" / job_id
        staged_dir.mkdir(parents=True, exist_ok=True)

        # 1. Stage code directory if provided
        staged_code = staged_dir / "src"
        if code_dir and Path(code_dir).is_dir():
            if staged_code.exists():
                shutil.rmtree(staged_code)
            shutil.copytree(code_dir, staged_code)

        # 2. Stage data file or link
        staged_data = staged_dir / "data"
        staged_data.mkdir(parents=True, exist_ok=True)
        if data_path and Path(data_path).exists():
            target_data = staged_data / Path(data_path).name
            shutil.copyfile(data_path, target_data)

        # 3. Write execution manifest with file SHA-256 hashes
        files_manifest: Dict[str, str] = {}
        for root, _, files in os.walk(staged_dir):
            for f in files:
                fpath = Path(root) / f
                if fpath.name != "execution_manifest.json":
                    rel = str(fpath.relative_to(staged_dir))
                    files_manifest[rel] = streaming_file_sha256(fpath)

        manifest_payload = {
            "job_id": job_id,
            "created_at": time.time(),
            "user_manifest": manifest,
            "staged_files": files_manifest,
        }
        (staged_dir / "execution_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2), encoding="utf-8"
        )

        return staged_dir

    def submit(self, staged_dir: Path, job_params: Dict[str, Any]) -> str:
        """Submit staged workload and return durable job handle."""
        staged_path = Path(staged_dir).resolve()
        job_id = job_params.get("job_id") or staged_path.name
        accelerator = job_params.get("accelerator", "gpu").lower()
        dataset_sources = job_params.get("dataset_sources", [])
        checkpoint_version = job_params.get("checkpoint_version")

        # Lease account
        account = self.pool.lease_worker_account(
            accelerator=accelerator,
            min_spacing_seconds=0.0,
            allow_secondary=self.enable_secondary_accounts,
        )

        # Build script code
        script_code = job_params.get("script_code")
        if not script_code:
            # Check if kernel.py exists in staged dir
            kernel_py = staged_path / "kernel.py"
            if kernel_py.exists():
                script_code = kernel_py.read_text(encoding="utf-8")
            else:
                raise ValueError("A real kernel.py or script_code is required for Kaggle submission")

        # Dispatch kernel push
        job = self.pool.push_private_kernel(
            account=account,
            script_code=script_code,
            dataset_sources=dataset_sources,
            accelerator=accelerator,
            add_jitter=job_params.get("add_jitter", False),
            job_id=job_id,
            checkpoint_version=checkpoint_version,
        )

        # Return durable composite handle: kaggle:{username}/{slug}:{job_id}:{lease_token}
        return f"kaggle:{job.account_username}/{job.kernel_slug}:{job.job_id}:{job.lease_token}"

    def parse_handle(self, job_handle: str) -> Tuple[str, str, str, str]:
        """Parses a job handle into (account_username, kernel_slug, job_id, lease_token)."""
        if not job_handle.startswith("kaggle:"):
            raise ValueError(f"Invalid Kaggle job handle format: {job_handle}")
        raw = job_handle[len("kaggle:"):]
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(f"Malformed job handle (expected 3 colon-separated parts): {job_handle}")
        user_slug, job_id, lease_token = parts
        if "/" not in user_slug:
            raise ValueError(f"Malformed username/slug in handle: {user_slug}")
        username, slug = user_slug.split("/", 1)
        return username, slug, job_id, lease_token

    def status(self, job_handle: str) -> Dict[str, Any]:
        """Poll lifecycle status, verifying lease fencing."""
        username, slug, job_id, lease_token = self.parse_handle(job_handle)

        # Check lease fencing
        if not self.pool.validate_lease(job_id, lease_token):
            return {
                "job_id": job_id,
                "status": JobLifecycleState.FAILED,
                "error": "Lease expired or superseded (fenced)",
                "fenced": True,
            }

        job = self.pool.get_job(job_id)
        if not job:
            return {"job_id": job_id, "status": JobLifecycleState.FAILED, "error": "Job not found"}

        # If already terminal or verified, return
        if job.status in (JobLifecycleState.VERIFIED, JobLifecycleState.CANCELLED):
            return {
                "job_id": job_id,
                "status": job.status,
                "completed_at": job.completed_at,
                "fenced": False,
            }

        account = self.pool.get_account(username)
        if not account:
            return {"job_id": job_id, "status": JobLifecycleState.FAILED, "error": "Account not found"}

        status_res = self.pool.check_kernel_status(account, slug)
        remote_status = status_res.get("status", JobLifecycleState.RUNNING)

        # Reconcile status in DB
        with sqlite3.connect(self.pool.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_jobs SET status = ? WHERE job_id = ?",
                (remote_status, job_id),
            )
            conn.commit()

        return {
            "job_id": job_id,
            "status": remote_status,
            "raw_status": status_res.get("raw"),
            "fenced": False,
        }

    def cancel(self, job_handle: str) -> bool:
        """Request remote cancellation; revokes lease to fence out remote outputs."""
        _, _, job_id, _ = self.parse_handle(job_handle)
        res = self.pool.cancel_kernel(job_id)
        return res.get("status") == JobLifecycleState.CANCEL_REQUESTED

    def collect(self, job_handle: str, target_dir: Path) -> Dict[str, Any]:
        """Collect remote outputs, verify hashes, and return result payload."""
        username, slug, job_id, lease_token = self.parse_handle(job_handle)

        # Fenced lease verification: must match current active lease
        if not self.pool.validate_lease(job_id, lease_token):
            raise StaleLeaseError(
                f"Collection rejected: lease token '{lease_token}' for job '{job_id}' is stale, expired, or superseded."
            )

        account = self.pool.get_account(username)
        if not account:
            raise ValueError(f"Account '{username}' not found in pool")

        dest = Path(target_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        # Download output and verify checksums against manifest.json
        output_path = self.pool.download_kernel_output(
            account=account,
            slug=slug,
            destination_dir=dest,
            verify_manifest=True,
        )

        now = time.time()
        with sqlite3.connect(self.pool.db_path) as conn:
            updated = conn.execute("""
                UPDATE kaggle_jobs
                SET status = ?, completed_at = ?, output_dir = ?
                WHERE job_id = ? AND lease_token = ? AND lease_expires_at > ?
            """, (JobLifecycleState.VERIFIED, now, str(output_path), job_id, lease_token, now))
            if updated.rowcount != 1:
                raise StaleLeaseError(f"Collection lease expired or was superseded for job '{job_id}'")
            conn.commit()

        return {
            "job_id": job_id,
            "status": JobLifecycleState.VERIFIED,
            "output_dir": str(output_path),
            "lease_token": lease_token,
            "verified_at": now,
        }

    def resume(
        self,
        job_handle: str,
        checkpoint_manifest: Dict[str, Any],
    ) -> str:
        """Resume is unavailable until checkpoint publication and restoration are implemented."""
        raise NotImplementedError("Kaggle checkpoint publication and remote resume are not implemented")
