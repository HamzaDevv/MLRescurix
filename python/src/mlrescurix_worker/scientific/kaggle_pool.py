"""Multi-Account Kaggle Compute Pool & Anti-Ban Isolation Manager.

Allows MLRescurix to safely pool free GPU (30h/wk) and TPU (20h/wk) compute
across multiple separate phone-verified Kaggle accounts without risking bans or disqualifications:
1. Strict Account Segregation:
   - Exactly ONE designated Anchor account submits to competitions.
   - Secondary accounts are strictly compute-only workers executing private background kernels.
   - Zero-Tolerance Guard: If a compute-only account attempts to submit to a competition,
     an AntiBanViolationError is raised immediately.
2. Private Polymorphic Kernels:
   - All kernels pushed are strictly marked `is_private: true`.
   - Polymorphic slugs (`opt-job-{uuid}`) and randomized generic titles.
   - Isolated staging folders per account (`.kaggle_staging/{username}/{slug}/`).
   - Credentials passed via subprocess environment variables, never contaminating global state or job DB.
3. Burst Decoupling & Rate Limiting:
   - Randomized jitter (2-5s) before push dispatch.
   - Minimum polling interval (>= 15s) for status checks.
4. Quota-Aware Scheduling:
   - Tracks weekly GPU and TPU quota consumption.
   - Leases the account with the highest available compute budget.
5. Durable Intent, Lease Fencing & Recovery:
   - Structured lifecycle: planned -> staged -> queued/submitted -> running -> collecting -> verified.
   - Fenced atomic leases prevent stale collectors from finalizing superseded attempts.
   - Checksum verification of downloaded artifacts against manifest.json before promotion.
   - Unknown-outcome reconciliation after timeouts.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .evidence import streaming_file_sha256

logger = logging.getLogger("mlrescurix.kaggle_pool")


class AntiBanViolationError(RuntimeError):
    """Raised when an operation violates Kaggle multi-account anti-abuse rules."""


class QuotaExhaustedError(RuntimeError):
    """Raised when all accounts in the pool have depleted their weekly accelerator quota."""


class StaleLeaseError(RuntimeError):
    """Raised when a collection attempt is made with an expired or superseded lease token."""


class ChecksumMismatchError(RuntimeError):
    """Raised when downloaded output files fail per-file hash verification against manifest."""


class SecurityViolationError(RuntimeError):
    """Raised when a staged payload or configuration contains potential credential leaks."""


class UnknownRemoteOutcomeError(RuntimeError):
    """Raised when a remote dispatch command times out with an unknown remote execution outcome."""


class JobLifecycleState:
    PLANNED = "planned"
    STAGED = "staged"
    SUBMITTED = "queued"  # Maps to Kaggle CLI 'queued' / submitted lifecycle state
    RUNNING = "running"
    COLLECTING = "collecting"
    VERIFIED = "verified"
    RETRY_WAIT = "retry_wait"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    FAILED = "failed"
    NEEDS_RESOURCES = "needs_resources"


def resolve_credential(credential_ref: str, storage_dir: Optional[Path] = None) -> str:
    """Resolves a credential reference into an actual API key without persisting it into databases."""
    if not credential_ref:
        return ""
    if credential_ref.startswith("env:"):
        var_name = credential_ref[4:]
        return os.environ.get(var_name, "").strip()
    if credential_ref.startswith("file:"):
        path_str = credential_ref[5:]
        p = Path(path_str).expanduser()
        if not p.is_absolute() and storage_dir:
            p = (storage_dir / p).resolve()
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
        return ""
    # Raw key fallback for legacy/in-memory test fixtures
    return credential_ref


@dataclass
class KaggleAccount:
    username: str
    credential_ref: str = ""
    is_anchor: bool = False
    phone_verified: bool = True
    gpu_hours_limit: float = 30.0
    tpu_hours_limit: float = 20.0
    gpu_hours_used: float = 0.0
    tpu_hours_used: float = 0.0
    last_dispatched_at: float = 0.0
    active_kernel_slug: Optional[str] = None
    created_at: float = dataclasses.field(default_factory=time.time)
    _storage_dir: Optional[Path] = dataclasses.field(default=None, repr=False)

    def __init__(
        self,
        username: str,
        credential_ref: str = "",
        api_key: Optional[str] = None,
        is_anchor: bool = False,
        phone_verified: bool = True,
        gpu_hours_limit: float = 30.0,
        tpu_hours_limit: float = 20.0,
        gpu_hours_used: float = 0.0,
        tpu_hours_used: float = 0.0,
        last_dispatched_at: float = 0.0,
        active_kernel_slug: Optional[str] = None,
        created_at: Optional[float] = None,
        storage_dir: Optional[Path] = None,
    ):
        self.username = username
        self.credential_ref = credential_ref or api_key or ""
        self.is_anchor = is_anchor
        self.phone_verified = phone_verified
        self.gpu_hours_limit = gpu_hours_limit
        self.tpu_hours_limit = tpu_hours_limit
        self.gpu_hours_used = gpu_hours_used
        self.tpu_hours_used = tpu_hours_used
        self.last_dispatched_at = last_dispatched_at
        self.active_kernel_slug = active_kernel_slug
        self.created_at = created_at if created_at is not None else time.time()
        self._storage_dir = storage_dir

    @property
    def api_key(self) -> str:
        """Resolve API key on-the-fly from credential reference."""
        return resolve_credential(self.credential_ref, self._storage_dir)

    @property
    def remaining_gpu_hours(self) -> float:
        return max(0.0, self.gpu_hours_limit - self.gpu_hours_used)

    @property
    def remaining_tpu_hours(self) -> float:
        return max(0.0, self.tpu_hours_limit - self.tpu_hours_used)


@dataclass
class KernelJobRecord:
    job_id: str
    account_username: str
    kernel_slug: str
    kernel_title: str
    is_private: bool
    accelerator: str
    status: str
    pushed_at: float
    completed_at: Optional[float]
    gpu_hours_billed: float
    local_staging_dir: str
    attempt: int = 1
    lease_token: str = ""
    lease_expires_at: float = 0.0
    payload_hash: str = ""
    checkpoint_version: Optional[str] = None
    output_dir: Optional[str] = None
    failure_reason: Optional[str] = None
    reconciliation_notes: Optional[str] = None


class KaggleAccountPool:
    """Manages multi-account Kaggle pooling, isolation, and anti-ban governance."""

    def __init__(
        self,
        storage_dir: Union[str, Path] = ".kaggle_pool",
        enable_secondary_accounts: bool = True,
    ):
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_dir = self.storage_dir / "credentials"
        self.credentials_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.credentials_dir, 0o700)
        except OSError:
            pass
        self.staging_dir = self.storage_dir / "staging"
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_dir / "pool_state.sqlite"
        self.enable_secondary_accounts = enable_secondary_accounts
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kaggle_accounts (
                    username TEXT PRIMARY KEY,
                    credential_ref TEXT NOT NULL,
                    is_anchor INTEGER NOT NULL DEFAULT 0,
                    phone_verified INTEGER NOT NULL DEFAULT 1,
                    gpu_hours_limit REAL NOT NULL DEFAULT 30.0,
                    tpu_hours_limit REAL NOT NULL DEFAULT 20.0,
                    gpu_hours_used REAL NOT NULL DEFAULT 0.0,
                    tpu_hours_used REAL NOT NULL DEFAULT 0.0,
                    last_dispatched_at REAL DEFAULT 0.0,
                    active_kernel_slug TEXT,
                    created_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kaggle_jobs (
                    job_id TEXT PRIMARY KEY,
                    account_username TEXT NOT NULL,
                    kernel_slug TEXT NOT NULL,
                    kernel_title TEXT NOT NULL,
                    is_private INTEGER NOT NULL DEFAULT 1,
                    accelerator TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 1,
                    lease_token TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0.0,
                    payload_hash TEXT NOT NULL DEFAULT '',
                    checkpoint_version TEXT,
                    pushed_at REAL NOT NULL,
                    completed_at REAL,
                    gpu_hours_billed REAL DEFAULT 0.0,
                    local_staging_dir TEXT NOT NULL,
                    output_dir TEXT,
                    failure_reason TEXT,
                    reconciliation_notes TEXT,
                    FOREIGN KEY(account_username) REFERENCES kaggle_accounts(username)
                )
            """)
            conn.commit()

            # Handle migrations for existing DBs
            cur = conn.execute("PRAGMA table_info(kaggle_accounts)")
            acc_cols = [r[1] for r in cur.fetchall()]
            if "api_key" in acc_cols and "credential_ref" not in acc_cols:
                conn.execute("ALTER TABLE kaggle_accounts RENAME COLUMN api_key TO credential_ref")
                conn.commit()

            cur = conn.execute("PRAGMA table_info(kaggle_jobs)")
            job_cols = [r[1] for r in cur.fetchall()]
            for col_name, col_type in [
                ("attempt", "INTEGER NOT NULL DEFAULT 1"),
                ("lease_token", "TEXT NOT NULL DEFAULT ''"),
                ("lease_expires_at", "REAL NOT NULL DEFAULT 0.0"),
                ("payload_hash", "TEXT NOT NULL DEFAULT ''"),
                ("checkpoint_version", "TEXT"),
                ("output_dir", "TEXT"),
                ("failure_reason", "TEXT"),
                ("reconciliation_notes", "TEXT"),
            ]:
                if col_name not in job_cols:
                    conn.execute(f"ALTER TABLE kaggle_jobs ADD COLUMN {col_name} {col_type}")
            conn.commit()

    def register_account(
        self,
        username: str,
        api_key: Optional[str] = None,
        credential_ref: Optional[str] = None,
        is_anchor: bool = False,
        phone_verified: bool = True,
        gpu_hours_limit: float = 30.0,
        tpu_hours_limit: float = 20.0,
    ) -> KaggleAccount:
        """Register or update a Kaggle account. Enforces that only ONE anchor account exists.

        Persists credentials as external references (files/env vars), never raw in SQLite.
        """
        username = username.strip()
        if not username:
            raise ValueError("Username must not be empty")

        final_ref = ""
        if credential_ref:
            final_ref = credential_ref.strip()
        elif api_key:
            api_key = api_key.strip()
            if api_key.startswith("env:") or api_key.startswith("file:"):
                final_ref = api_key
            else:
                # Store secret in restricted credentials file outside the DB
                key_file = self.credentials_dir / f"{username}.key"
                key_file.write_text(api_key, encoding="utf-8")
                try:
                    os.chmod(key_file, 0o600)
                except OSError:
                    pass
                final_ref = f"file:{key_file}"

        if not final_ref:
            raise ValueError("Username and API key or credential_ref must not be empty")

        with sqlite3.connect(self.db_path) as conn:
            if is_anchor:
                # Check if another account is already designated anchor
                cur = conn.execute(
                    "SELECT username FROM kaggle_accounts WHERE is_anchor = 1 AND username != ?",
                    (username,),
                )
                existing_anchor = cur.fetchone()
                if existing_anchor:
                    raise AntiBanViolationError(
                        f"Account '{existing_anchor[0]}' is already the designated official anchor. "
                        "Having multiple submitter accounts in the pool violates anti-ban team rules."
                    )

            conn.execute("""
                INSERT INTO kaggle_accounts (
                    username, credential_ref, is_anchor, phone_verified, gpu_hours_limit, tpu_hours_limit,
                    gpu_hours_used, tpu_hours_used, last_dispatched_at, active_kernel_slug, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0.0, 0.0, 0.0, NULL, ?)
                ON CONFLICT(username) DO UPDATE SET
                    credential_ref = excluded.credential_ref,
                    is_anchor = excluded.is_anchor,
                    phone_verified = excluded.phone_verified,
                    gpu_hours_limit = excluded.gpu_hours_limit,
                    tpu_hours_limit = excluded.tpu_hours_limit
            """, (
                username, final_ref, int(is_anchor), int(phone_verified),
                gpu_hours_limit, tpu_hours_limit, time.time()
            ))
            conn.commit()

        return self.get_account(username)  # type: ignore

    def get_account(self, username: str) -> Optional[KaggleAccount]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT username, credential_ref, is_anchor, phone_verified, gpu_hours_limit, tpu_hours_limit,
                       gpu_hours_used, tpu_hours_used, last_dispatched_at, active_kernel_slug, created_at
                FROM kaggle_accounts WHERE username = ?
            """, (username,))
            row = cur.fetchone()
            if not row:
                return None
            return KaggleAccount(
                username=row[0],
                credential_ref=row[1],
                is_anchor=bool(row[2]),
                phone_verified=bool(row[3]),
                gpu_hours_limit=float(row[4]),
                tpu_hours_limit=float(row[5]),
                gpu_hours_used=float(row[6]),
                tpu_hours_used=float(row[7]),
                last_dispatched_at=float(row[8]),
                active_kernel_slug=row[9],
                created_at=float(row[10]),
                storage_dir=self.storage_dir,
            )

    def get_anchor_account(self) -> Optional[KaggleAccount]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT username FROM kaggle_accounts WHERE is_anchor = 1")
            row = cur.fetchone()
            if not row:
                return None
            return self.get_account(row[0])

    def list_accounts(self) -> List[KaggleAccount]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT username FROM kaggle_accounts ORDER BY is_anchor DESC, username ASC")
            return [self.get_account(r[0]) for r in cur.fetchall() if r[0]]  # type: ignore

    def lease_worker_account(
        self,
        accelerator: str = "gpu",
        min_spacing_seconds: float = 15.0,
        allow_secondary: Optional[bool] = None,
    ) -> KaggleAccount:
        """Lease a compute account with the highest available quota.

        If allow_secondary is False (or self.enable_secondary_accounts is False),
        strictly leases the designated anchor account to enforce single-account execution first.
        """
        accelerator = accelerator.lower()
        accounts = self.list_accounts()
        if not accounts:
            raise QuotaExhaustedError("No Kaggle accounts registered in the pool")

        use_secondary = self.enable_secondary_accounts if allow_secondary is None else allow_secondary

        if not use_secondary:
            anchor = self.get_anchor_account()
            if not anchor:
                raise QuotaExhaustedError("No designated anchor account registered in pool for single-account execution")
            remaining = anchor.remaining_gpu_hours if accelerator == "gpu" else anchor.remaining_tpu_hours
            if remaining <= 0.2:
                raise QuotaExhaustedError(
                    f"Anchor account '{anchor.username}' has depleted its weekly {accelerator.upper()} quota."
                )
            chosen = anchor
        else:
            candidates = [a for a in accounts if not a.is_anchor]
            if not candidates:
                candidates = accounts

            now = time.time()
            viable: List[Tuple[float, float, KaggleAccount]] = []
            for acc in candidates:
                remaining = acc.remaining_gpu_hours if accelerator == "gpu" else acc.remaining_tpu_hours
                if remaining <= 0.2:
                    continue
                idle_time = now - acc.last_dispatched_at
                viable.append((remaining, idle_time, acc))

            if not viable:
                raise QuotaExhaustedError(
                    f"All accounts in the pool have depleted their weekly {accelerator.upper()} quota."
                )

            viable.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = viable[0][2]

        now = time.time()
        time_since_dispatch = now - chosen.last_dispatched_at
        if time_since_dispatch < min_spacing_seconds:
            wait_time = min_spacing_seconds - time_since_dispatch
            time.sleep(wait_time)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_accounts SET last_dispatched_at = ? WHERE username = ?",
                (time.time(), chosen.username),
            )
            conn.commit()

        return self.get_account(chosen.username)  # type: ignore

    def acquire_lease(
        self, job_id: str, ttl_seconds: float = 300.0
    ) -> Tuple[int, str, float]:
        """Acquires a new lease attempt for job_id, incrementing attempt and returning (attempt, lease_token, expires_at)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT attempt FROM kaggle_jobs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            attempt = (row[0] + 1) if row else 1
            lease_token = f"lease-{uuid.uuid4().hex}"
            expires_at = time.time() + ttl_seconds
            conn.execute("""
                UPDATE kaggle_jobs
                SET attempt = ?, lease_token = ?, lease_expires_at = ?
                WHERE job_id = ?
            """, (attempt, lease_token, expires_at, job_id))
            conn.commit()
        return attempt, lease_token, expires_at

    def validate_lease(self, job_id: str, lease_token: str) -> bool:
        """Validates that lease_token matches the current active lease and has not expired."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT lease_token, lease_expires_at FROM kaggle_jobs WHERE job_id = ?",
                (job_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            cur_token, expires_at = row[0], float(row[1])
            if cur_token != lease_token:
                return False
            if time.time() > expires_at:
                return False
            return True

    def revoke_lease(self, job_id: str) -> None:
        """Revokes any active lease for job_id to fence out in-flight collectors."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_jobs SET lease_expires_at = 0.0, lease_token = 'REVOKED' WHERE job_id = ?",
                (job_id,),
            )
            conn.commit()

    def get_job(self, job_id: str) -> Optional[KernelJobRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT job_id, account_username, kernel_slug, kernel_title, is_private,
                       accelerator, status, pushed_at, completed_at, gpu_hours_billed,
                       local_staging_dir, attempt, lease_token, lease_expires_at,
                       payload_hash, checkpoint_version, output_dir, failure_reason, reconciliation_notes
                FROM kaggle_jobs WHERE job_id = ?
            """, (job_id,))
            row = cur.fetchone()
            if not row:
                return None
            return KernelJobRecord(
                job_id=row[0],
                account_username=row[1],
                kernel_slug=row[2],
                kernel_title=row[3],
                is_private=bool(row[4]),
                accelerator=row[5],
                status=row[6],
                pushed_at=float(row[7]),
                completed_at=float(row[8]) if row[8] is not None else None,
                gpu_hours_billed=float(row[9]),
                local_staging_dir=row[10],
                attempt=int(row[11]),
                lease_token=row[12],
                lease_expires_at=float(row[13]),
                payload_hash=row[14],
                checkpoint_version=row[15],
                output_dir=row[16],
                failure_reason=row[17],
                reconciliation_notes=row[18],
            )

    def list_jobs(self) -> List[KernelJobRecord]:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT job_id FROM kaggle_jobs ORDER BY pushed_at DESC")
            return [self.get_job(r[0]) for r in cur.fetchall() if r[0]]  # type: ignore

    def push_private_kernel(
        self,
        account: KaggleAccount,
        script_code: str,
        dataset_sources: Optional[List[str]] = None,
        accelerator: str = "gpu",
        add_jitter: bool = True,
        job_id: Optional[str] = None,
        lease_ttl_seconds: float = 300.0,
        checkpoint_version: Optional[str] = None,
    ) -> KernelJobRecord:
        """Stage and push a private polymorphic kernel to Kaggle.

        Ensures strict anti-ban and security compliance:
        - Kernel is ALWAYS private (`is_private: true`).
        - Slug is randomly generated (`opt-job-[uuid]`) to prevent collision/heuristics.
        - Staging directory is isolated.
        - Subprocess environment isolates credentials per call.
        - Credential scrubber blocks raw secret embedding in staged scripts.
        - Duplicate submissions with identical payload/job_id are idempotent.
        - Intent is logged to durable SQLite before submission.
        - Timeouts are captured for reconciliation.
        """
        # 1. Credential Scrubbing Guard
        if re.search(r"sk-[A-Za-z0-9_\-]{16,}", script_code):
            raise SecurityViolationError(
                "Refusing to push kernel: credential leakage detected (raw sk- token) in kernel source."
            )
        leak_pattern = re.compile(
            r"""(?:DEEPSEEK_API_KEY|OPENROUTER_API_KEY|KAGGLE_KEY|AWS_SECRET_ACCESS_KEY)\s*[:=]\s*["']?([^\s"']+)["']?""",
            re.IGNORECASE,
        )
        if leak_pattern.search(script_code):
            raise SecurityViolationError(
                "Refusing to push kernel: credential leakage detected (sensitive key assignment) in kernel source."
            )
        if "Bearer sk-" in script_code:
            raise SecurityViolationError(
                "Refusing to push kernel: credential leakage detected (Bearer sk-) in kernel source."
            )

        payload_hash = hashlib.sha256(script_code.encode("utf-8")).hexdigest()

        # 2. Duplicate Submission Idempotency Check
        if job_id:
            existing_job = self.get_job(job_id)
            if existing_job and existing_job.status in (JobLifecycleState.SUBMITTED, JobLifecycleState.RUNNING):
                logger.info("Found existing submitted/running job with job_id=%s; returning existing record", job_id)
                return existing_job

        if add_jitter:
            time.sleep(random.uniform(2.0, 5.0))

        actual_job_id = job_id or f"job-{uuid.uuid4().hex[:12]}"
        slug = f"opt-job-{uuid.uuid4().hex[:10]}"
        title = f"Optimization Job {slug}"

        # 3. Setup isolated staging directory
        stage_dir = self.staging_dir / account.username / slug
        stage_dir.mkdir(parents=True, exist_ok=True)

        script_path = stage_dir / "kernel.py"
        script_path.write_text(script_code, encoding="utf-8")

        metadata = {
            "id": f"{account.username}/{slug}",
            "title": title,
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true" if accelerator.lower() == "gpu" else "false",
            "enable_tpu": "true" if accelerator.lower() == "tpu" else "false",
            "enable_internet": "true",
            "dataset_sources": dataset_sources or [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (stage_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        lease_token = f"lease-{uuid.uuid4().hex}"
        lease_expires_at = time.time() + lease_ttl_seconds
        now = time.time()

        # 4. Persist durable intent to SQLite BEFORE external command
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO kaggle_jobs (
                    job_id, account_username, kernel_slug, kernel_title, is_private,
                    accelerator, status, attempt, lease_token, lease_expires_at, payload_hash,
                    checkpoint_version, pushed_at, completed_at, gpu_hours_billed, local_staging_dir
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, NULL, 0.0, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    account_username = excluded.account_username,
                    kernel_slug = excluded.kernel_slug,
                    kernel_title = excluded.kernel_title,
                    is_private = excluded.is_private,
                    accelerator = excluded.accelerator,
                    status = excluded.status,
                    lease_token = excluded.lease_token,
                    lease_expires_at = excluded.lease_expires_at,
                    payload_hash = excluded.payload_hash,
                    checkpoint_version = excluded.checkpoint_version,
                    pushed_at = excluded.pushed_at,
                    local_staging_dir = excluded.local_staging_dir
            """, (
                actual_job_id, account.username, slug, title, 1, accelerator,
                JobLifecycleState.STAGED, lease_token, lease_expires_at, payload_hash,
                checkpoint_version, now, str(stage_dir),
            ))
            conn.execute(
                "UPDATE kaggle_accounts SET active_kernel_slug = ? WHERE username = ?",
                (slug, account.username),
            )
            conn.commit()

        # 5. Allowlist subprocess environment
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR", "HOME", "USER")}
        env["KAGGLE_USERNAME"] = account.username
        env["KAGGLE_KEY"] = account.api_key

        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "push", "-p", str(stage_dir)],
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            if proc.returncode == 0:
                final_status = JobLifecycleState.SUBMITTED
                failure_msg = None
            else:
                final_status = JobLifecycleState.FAILED
                failure_msg = proc.stderr.strip() or proc.stdout.strip()
        except subprocess.TimeoutExpired as exc:
            final_status = JobLifecycleState.RETRY_WAIT
            failure_msg = "Push command timed out; outcome unknown on Kaggle, reconciliation required"
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE kaggle_jobs SET status = ?, failure_reason = ?, reconciliation_notes = ? WHERE job_id = ?",
                    (final_status, failure_msg, "TIMEOUT_UNKNOWN_OUTCOME", actual_job_id),
                )
                conn.commit()
            raise UnknownRemoteOutcomeError(f"Push timed out for {actual_job_id}; reconciliation required") from exc

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_jobs SET status = ?, failure_reason = ? WHERE job_id = ?",
                (final_status, failure_msg, actual_job_id),
            )
            conn.commit()

        if final_status == JobLifecycleState.FAILED:
            logger.error("Failed to push kernel: %s", failure_msg)
            raise RuntimeError(f"Kaggle kernel push failed: {failure_msg}")

        return self.get_job(actual_job_id)  # type: ignore

    def check_kernel_status(self, account: KaggleAccount, slug: str) -> dict[str, Any]:
        """Poll kernel status with isolated environment and return parsed status."""
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR", "HOME", "USER")}
        env["KAGGLE_USERNAME"] = account.username
        env["KAGGLE_KEY"] = account.api_key

        proc = subprocess.run(
            ["kaggle", "kernels", "status", f"{account.username}/{slug}"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        output = (proc.stdout.strip() + " " + proc.stderr.strip()).strip()
        output_lower = output.lower()
        if "404" in output_lower or "not found" in output_lower:
            status = JobLifecycleState.RETRY_WAIT
        elif "cancelpending" in output_lower or "canceled" in output_lower or "cancelled" in output_lower:
            status = JobLifecycleState.CANCELLED
        elif "ready" in output_lower or "complete" in output_lower:
            status = JobLifecycleState.COLLECTING
        elif "running" in output_lower:
            status = JobLifecycleState.RUNNING
        elif "queued" in output_lower:
            status = JobLifecycleState.SUBMITTED
        elif "error" in output_lower or proc.returncode != 0:
            status = JobLifecycleState.FAILED
        else:
            status = JobLifecycleState.RUNNING if proc.returncode == 0 else JobLifecycleState.FAILED

        return {"raw": output, "status": status, "exit_code": proc.returncode}

    def reconcile_job(self, job_id: str) -> KernelJobRecord:
        """Reconciles an unknown or timed-out job with remote Kaggle status."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        account = self.get_account(job.account_username)
        if not account:
            raise ValueError(f"Account {job.account_username} not found")

        status_res = self.check_kernel_status(account, job.kernel_slug)
        raw = status_res.get("raw", "").lower()
        if "404" in raw or "not found" in raw:
            new_status = JobLifecycleState.RETRY_WAIT
            notes = "Remote kernel not found; marked RETRY_WAIT"
        elif status_res["status"] == JobLifecycleState.COLLECTING:
            new_status = JobLifecycleState.COLLECTING
            notes = "Remote kernel completed; ready for collection"
        elif status_res["status"] == JobLifecycleState.RUNNING:
            new_status = JobLifecycleState.RUNNING
            notes = "Remote kernel running"
        elif status_res["status"] == JobLifecycleState.SUBMITTED:
            new_status = JobLifecycleState.SUBMITTED
            notes = "Remote kernel queued"
        else:
            new_status = JobLifecycleState.FAILED
            notes = f"Remote kernel status: {status_res['status']}"

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_jobs SET status = ?, reconciliation_notes = ? WHERE job_id = ?",
                (new_status, notes, job_id),
            )
            conn.commit()
        return self.get_job(job_id)  # type: ignore

    def download_kernel_output(
        self,
        account: KaggleAccount,
        slug: str,
        destination_dir: Union[str, Path],
        verify_manifest: bool = True,
    ) -> Path:
        """Download output files from a completed kernel and verify manifest SHA-256 hashes."""
        dest = Path(destination_dir).resolve()
        dest.mkdir(parents=True, exist_ok=True)

        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR", "HOME", "USER")}
        env["KAGGLE_USERNAME"] = account.username
        env["KAGGLE_KEY"] = account.api_key

        proc = subprocess.run(
            ["kaggle", "kernels", "output", f"{account.username}/{slug}", "-p", str(dest)],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to download kernel output: {proc.stderr.strip() or proc.stdout.strip()}")

        if verify_manifest:
            manifest_path = dest / "manifest.json"
            if not manifest_path.exists():
                found = list(dest.rglob("manifest.json"))
                if found:
                    manifest_path = found[0]

            if not manifest_path.exists():
                raise ChecksumMismatchError(
                    f"Incomplete download: manifest.json is missing in output directory '{dest}'"
                )

            manifest_dir = manifest_path.parent
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                files_manifest = manifest_data.get("files")
                if not isinstance(files_manifest, dict) or not files_manifest:
                    raise ChecksumMismatchError("Output manifest must contain a non-empty files mapping")
                for rel_name, expected_sha in files_manifest.items():
                    if not isinstance(rel_name, str) or not isinstance(expected_sha, str):
                        raise ChecksumMismatchError("Invalid output manifest entry")
                    target_file = (manifest_dir / rel_name).resolve()
                    if not target_file.is_relative_to(manifest_dir.resolve()) or target_file == manifest_dir.resolve():
                        raise ChecksumMismatchError(f"Output manifest path escapes output directory: {rel_name}")
                    if not target_file.is_file():
                        raise ChecksumMismatchError(
                            f"Incomplete download: manifest file '{rel_name}' is missing in output"
                        )
                    actual_sha = streaming_file_sha256(target_file)
                    if actual_sha != expected_sha:
                        raise ChecksumMismatchError(
                            f"Checksum mismatch for '{rel_name}': expected {expected_sha}, got {actual_sha}"
                        )
            except json.JSONDecodeError as exc:
                raise ChecksumMismatchError("Corrupt manifest.json in downloaded output") from exc

        return dest

    def cancel_kernel(self, job_id: str) -> Dict[str, Any]:
        """Cancel kernel intent.

        Note on Kaggle API limitation:
        Official Kaggle CLI does not provide a remote kernel force-kill API.
        This method marks CANCEL_REQUESTED and revokes the lease token so that
        any output produced by the remote process is fenced and discarded.
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        self.revoke_lease(job_id)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE kaggle_jobs SET status = ?, reconciliation_notes = ? WHERE job_id = ?",
                (
                    JobLifecycleState.CANCEL_REQUESTED,
                    "Cancel requested. Note: Kaggle API lacks remote force-kill; lease revoked to fence outputs.",
                    job_id,
                ),
            )
            conn.commit()

        return {
            "status": JobLifecycleState.CANCEL_REQUESTED,
            "job_id": job_id,
            "lease_revoked": True,
            "limitation": "Kaggle public API does not support remote kernel force-kill. Lease revoked to fence remote artifacts.",
        }

    def submit_competition_prediction(
        self,
        competition_id: str,
        submission_file: Union[str, Path],
        message: str,
        account: Optional[KaggleAccount] = None,
    ) -> dict[str, Any]:
        """Submit prediction to an official Kaggle competition.

        CRITICAL ANTI-BAN ENFORCEMENT:
        Only the designated Anchor account is allowed to submit to Kaggle competitions.
        Any attempt by secondary/worker accounts to submit is blocked.
        """
        anchor = self.get_anchor_account()
        if not anchor:
            raise AntiBanViolationError(
                "No official Anchor account configured in the pool. "
                "You must designate exactly one account as anchor using register_account(..., is_anchor=True)."
            )

        if account is not None and account.username != anchor.username:
            raise AntiBanViolationError(
                f"Anti-Ban Violation: Worker account '{account.username}' attempted to submit to competition '{competition_id}'. "
                f"Only the designated Anchor account '{anchor.username}' is permitted to make competition submissions."
            )

        sub_path = Path(submission_file).resolve()
        if not sub_path.exists():
            raise FileNotFoundError(f"Submission file '{sub_path}' not found")

        env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "LANG", "TMPDIR", "HOME", "USER")}
        env["KAGGLE_USERNAME"] = anchor.username
        env["KAGGLE_KEY"] = anchor.api_key

        proc = subprocess.run(
            ["kaggle", "competitions", "submit", "-c", competition_id, "-f", str(sub_path), "-m", message],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

        return {
            "status": "success" if proc.returncode == 0 else "error",
            "account_used": anchor.username,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "exit_code": proc.returncode,
        }
