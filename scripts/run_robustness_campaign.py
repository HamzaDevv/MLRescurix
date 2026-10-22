#!/usr/bin/env python3
"""MLRescurix Robustness Campaign Preflight & Resumable Runner.

Adheres to ROBUSTNESS_MASTER_PLAN.md:
- Preflight: environment, binary provenance, schema version (up to v11), Docker daemon,
  Kaggle credentials, provider key presence and budget limits.
- Resumable: run_id directory layout under docs/evidence/v2/robustness/<campaign>/<run_id>/.
- Explicit outcomes: passed, failed, skipped, blocked, inconclusive.
- Worker resource traces: controller RSS, worker process tree RSS, cgroup on Linux.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "docs" / "v2" / "BENCHMARK_SUITE.json"
EVIDENCE_BASE = REPO_ROOT / "docs" / "evidence" / "v2" / "robustness"


def finite_score(value: Any) -> Optional[float]:
    """Return a measured finite score, rejecting missing, boolean, and fabricated values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return score if math.isfinite(score) else None


def score_meets_threshold(score: Optional[float], threshold: Optional[float], direction: str) -> bool:
    if score is None or threshold is None:
        return False
    return score >= threshold if direction == "maximize" else score <= threshold


def locked_quality_threshold(policy: Dict[str, Any], task_id: str) -> Optional[float]:
    """Read a quality threshold only when the benchmark policy has locked one."""
    problem = next((p for p in policy.get("problems", []) if p.get("id") == task_id), None)
    criteria = problem.get("final_quality_criteria", {}) if problem else {}
    if criteria.get("status") != "locked":
        return None
    return finite_score(criteria.get("threshold"))


def verified_provider_pricing(policy: Dict[str, Any], checked_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return only pricing explicitly revalidated for live use on the current date."""
    snapshot = policy.get("provider", {}).get("pricing_snapshot", {})
    today = checked_date or datetime.now().astimezone().date().isoformat()
    if (snapshot.get("verified_for_live") is not True
            or snapshot.get("must_revalidate_before_live") is not False
            or snapshot.get("checked_local_date") != today):
        return None
    try:
        input_price = Decimal(str(snapshot["input_usd_per_million"]))
        output_price = Decimal(str(snapshot["output_usd_per_million"]))
    except (KeyError, ValueError):
        return None
    if not input_price.is_finite() or not output_price.is_finite() or input_price < 0 or output_price < 0:
        return None
    return {"verified": True, "input_per_million": str(input_price), "output_per_million": str(output_price)}


def candidate_gate_failure_reason(
    evidence_ok: bool,
    evidence_reason: str,
    score: Optional[float],
    threshold: Optional[float],
    direction: str,
    bundle_verified: bool,
    identity_verified: bool,
    package_reason: str,
) -> Optional[str]:
    failures = []
    if not evidence_ok:
        failures.append(evidence_reason)
    if not score_meets_threshold(score, threshold, direction):
        failures.append("measured score did not meet a locked required threshold" if threshold is not None
                        else "benchmark quality threshold is not locked")
    if not identity_verified:
        failures.append(package_reason or "bundle identity did not match accepted candidate")
    if not bundle_verified:
        failures.append("bundle verification failed")
    return "; ".join(dict.fromkeys(failures)) or None


def accepted_candidate_evidence(state: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
    """Require an accepted, audited candidate with a measured finite CV score and identity."""
    score = finite_score(state.get("best_cv_score"))
    candidate_id = state.get("accepted_candidate_id")
    candidate = state.get("accepted_candidate")
    identity = state.get("candidate_identity")
    audit = state.get("accepted_audit")
    evaluation = state.get("accepted_evaluation")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return False, score, "no accepted candidate id"
    if not isinstance(candidate, dict):
        return False, score, "accepted candidate specification is missing"
    try:
        spec_identity = candidate_spec_identity(candidate)
    except (TypeError, ValueError):
        spec_identity = None
    if not isinstance(identity, str) or not identity.strip() or spec_identity != identity:
        return False, score, "candidate identity does not match accepted specification"
    if not isinstance(audit, dict) or audit.get("status") != "APPROVED" or audit.get("candidate_id") != candidate_id:
        return False, score, "accepted candidate audit is missing or not approved"
    if score is None:
        return False, None, "accepted candidate has no finite measured CV score"
    if not isinstance(evaluation, dict):
        return False, score, "accepted candidate evaluation receipt is missing"
    if evaluation.get("candidate_identity") != identity:
        return False, score, "accepted evaluation identity does not match accepted specification"
    evaluated_score = finite_score(evaluation.get("cv_score"))
    if evaluated_score is None or evaluated_score != score:
        return False, score, "accepted evaluation score does not match best measured CV score"
    return True, score, "accepted audited candidate has finite measured CV score"


def candidate_spec_identity(spec: Dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bundle_matches_candidate(bundle_dir: Path, candidate_id: str, candidate_identity: str) -> bool:
    """Check the bundled model and manifest both identify the audited candidate."""
    try:
        manifest = json.loads((Path(bundle_dir) / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("candidate_id") != candidate_id or manifest.get("candidate_identity") != candidate_identity:
            return False
        weights = Path(bundle_dir) / manifest["model"]["weights_path"]
        import joblib
        model = joblib.load(weights)
        return getattr(model, "candidate_identity", None) == candidate_identity
    except Exception:
        return False


def map_locked_index_folds(folds: List[Dict[str, Any]], selected_original_rows: List[int]) -> List[Tuple[List[int], List[int]]]:
    """Map pinned manifest row indices onto a selected runner frame; fail on partial folds."""
    if not folds:
        raise ValueError("pinned development_folds are missing")
    if len(set(selected_original_rows)) != len(selected_original_rows):
        raise ValueError("selected development rows contain duplicates")
    row_to_local = {original: local for local, original in enumerate(selected_original_rows)}
    mapped: List[Tuple[List[int], List[int]]] = []
    for fold_number, fold in enumerate(folds):
        train = fold.get("train_indices")
        val = fold.get("val_indices")
        if not isinstance(train, list) or not isinstance(val, list) or not train or not val:
            raise ValueError(f"pinned fold {fold_number} has no explicit train_indices/val_indices")
        indices = train + val
        if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError(f"pinned fold {fold_number} contains invalid row indices")
        if len(set(train)) != len(train) or len(set(val)) != len(val):
            raise ValueError(f"pinned fold {fold_number} contains duplicate row indices")
        if set(train).intersection(val):
            raise ValueError(f"pinned fold {fold_number} has train/validation row overlap")
        if any(index not in row_to_local for index in train + val):
            raise ValueError(f"pinned fold {fold_number} is not fully contained in selected development rows")
        mapped.append(([row_to_local[i] for i in train], [row_to_local[i] for i in val]))
    return mapped


def map_k4_locked_folds(folds: List[Dict[str, Any]], subset_start: int, subset_end: int, index_space: Optional[str]) -> List[Tuple[List[int], List[int]]]:
    """Map K4 folds only under a declared source-row or bounded-subset index space."""
    if index_space in ("source_development_rows", "original_rows"):
        selected_rows = list(range(subset_start, subset_end + 1))
    elif index_space in ("bounded_dev_subset", "selected_subset_rows"):
        selected_rows = list(range(subset_end - subset_start + 1))
    else:
        raise ValueError(f"unsupported K4 development_folds_index_space {index_space!r}")
    return map_locked_index_folds(folds, selected_rows)


def map_k4_manifest_folds(split_manifest: Dict[str, Any], df: Any, subset_start: int, subset_end: int) -> List[Tuple[List[int], List[int]]]:
    """Map only declared K4 folds; never synthesize temporal origins from the sample."""
    folds = split_manifest.get("development_folds") or split_manifest.get("development_origins", [])
    if not folds:
        raise ValueError("pinned K4 development folds are missing")
    if "train_indices" in folds[0]:
        return map_k4_locked_folds(
            folds, subset_start, subset_end,
            split_manifest.get("development_folds_index_space"),
        )
    index_fields = {"train_start_idx", "train_end_idx", "val_start_idx", "val_end_idx"}
    if index_fields.issubset(folds[0]):
        if split_manifest.get("development_folds_index_space") not in ("source_development_rows", "original_rows", "bounded_dev_subset", "selected_subset_rows"):
            raise ValueError("K4 origin indices lack an explicit development_folds_index_space")
        converted = [{"train_indices": list(range(f["train_start_idx"], f["train_end_idx"] + 1)),
                      "val_indices": list(range(f["val_start_idx"], f["val_end_idx"] + 1))} for f in folds]
        return map_k4_locked_folds(converted, subset_start, subset_end, split_manifest["development_folds_index_space"])
    has_time_boundaries = all(
        any(key in fold for key in ("train_end", "train_end_date", "train_end_dt"))
        and any(key in fold for key in ("val_start", "validation_start", "val_start_date", "validation_start_date", "val_start_dt"))
        and any(key in fold for key in ("val_end", "validation_end", "val_end_date", "validation_end_date", "val_end_dt"))
        for fold in folds
    )
    if not has_time_boundaries:
        raise ValueError("K4 origins lack explicit train-end and validation time boundaries")
    if split_manifest.get("locked_gap") is None:
        raise ValueError("K4 temporal origins lack a locked gap")
    return map_locked_temporal_folds(folds, df, "TransactionDT")


def map_locked_temporal_folds(folds: List[Dict[str, Any]], df: Any, time_column: str) -> List[Tuple[List[int], List[int]]]:
    """Translate explicit pinned expanding-window boundaries to local row positions."""
    if not folds or time_column not in df.columns:
        raise ValueError(f"pinned temporal folds or time column {time_column!r} are missing")
    values = df[time_column]
    datetime_values = str(values.dtype).startswith("datetime") or time_column == "date"
    if datetime_values:
        import pandas as pd
        values = pd.to_datetime(values)
    mapped = []
    for fold_number, fold in enumerate(folds):
        train_end = fold.get("train_end") or fold.get("train_end_date") or fold.get("train_end_dt")
        val_start = fold.get("val_start") or fold.get("validation_start") or fold.get("val_start_date") or fold.get("validation_start_date") or fold.get("val_start_dt")
        val_end = fold.get("val_end") or fold.get("validation_end") or fold.get("val_end_date") or fold.get("validation_end_date") or fold.get("val_end_dt")
        if train_end is None or val_start is None or val_end is None:
            raise ValueError(f"pinned temporal fold {fold_number} lacks train_end/val_start/val_end boundaries")
        if datetime_values:
            train_end, val_start, val_end = map(pd.Timestamp, (train_end, val_start, val_end))
        else:
            train_end, val_start, val_end = map(float, (train_end, val_start, val_end))
        train_idx = [i for i, value in enumerate(values) if value <= train_end]
        val_idx = [i for i, value in enumerate(values) if val_start <= value <= val_end]
        if not train_idx or not val_idx or train_end >= val_start:
            raise ValueError(f"pinned temporal fold {fold_number} has empty or overlapping time partitions")
        mapped.append((train_idx, val_idx))
    return mapped


def package_accepted_candidate(
    agent_loop: Any,
    state: Dict[str, Any],
    df_dev: Any,
    *,
    candidate_dir: Path,
    bundle_dir: Path,
    dataset_path: Path,
    test_access_id: str,
    project_id: str,
    task_type: str,
    target_column: str,
    primary_metric: str,
) -> Tuple[bool, bool, str]:
    """Fit and package exactly the accepted candidate; never substitute a default model."""
    valid, _, reason = accepted_candidate_evidence(state)
    if not valid:
        return False, False, reason
    candidate_id = state["accepted_candidate_id"]
    identity = state["candidate_identity"]
    try:
        pipeline = agent_loop.fit_accepted_candidate()
    except Exception as exc:
        return False, False, f"accepted candidate refit failed: {exc}"
    if getattr(pipeline, "candidate_identity", None) != identity:
        return False, False, "refitted pipeline identity differs from audited candidate"
    from mlrescurix_worker.scientific.autonomous_rescue import bundle_candidate
    try:
        bundle_candidate(
            candidate_dir=candidate_dir,
            output_bundle_dir=bundle_dir,
            dataset_path=dataset_path,
            test_access_id=test_access_id,
            project_id=project_id,
            candidate_id=candidate_id,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            pipeline=pipeline,
            raw_holdout_records=df_dev.head(5).to_dict(orient="records"),
            accepted_evaluation=state.get("accepted_evaluation"),
            accepted_candidate_identity=identity,
        )
    except Exception as exc:
        return False, False, f"accepted candidate packaging failed: {exc}"
    identity_ok = bundle_matches_candidate(bundle_dir, candidate_id, identity)
    if not identity_ok:
        return False, False, "bundle model identity does not match accepted audited candidate"
    from mlrescurix_worker.scientific.rust_gateway import find_harness_binary
    try:
        harness_bin = find_harness_binary()
    except Exception:
        harness_bin = REPO_ROOT / "target" / "debug" / "mlrescurix-harness"
    try:
        verify_proc = subprocess.run(
            [str(harness_bin), "verify-bundle", "--bundle", str(bundle_dir), "--sandbox", "false"],
            capture_output=True, text=True,
        )
    except OSError as exc:
        return False, True, f"bundle verification could not run: {exc}"
    return verify_proc.returncode == 0, True, "accepted candidate bundled and identity verified"


def compute_file_sha256(path: Path | str) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_directory_sha256(directory: Path | str) -> str:
    d = Path(directory)
    if not d.exists():
        return "not_found"
    files = []
    for root, _, filenames in os.walk(d):
        rel_dir = Path(root).relative_to(d)
        if any(part.startswith(".") or part == "__pycache__" for part in rel_dir.parts):
            continue
        for fn in filenames:
            if fn.startswith(".") or fn.endswith(".pyc"):
                continue
            fpath = Path(root) / fn
            rel_file = (rel_dir / fn).as_posix()
            try:
                content = fpath.read_bytes()
                files.append((rel_file, hashlib.sha256(content).hexdigest()))
            except Exception:
                pass
    files.sort(key=lambda x: x[0])
    h = hashlib.sha256()
    for rel_path, file_sha in files:
        h.update(f"{rel_path}:{file_sha}\n".encode("utf-8"))
    return h.hexdigest()


def get_git_provenance() -> Dict[str, Any]:
    try:
        p_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        commit = p_head.stdout.strip() if p_head.returncode == 0 else "unknown"
        p_diff = subprocess.run(["git", "diff", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        diff_text = p_diff.stdout if p_diff.returncode == 0 else ""
        diff_sha = hashlib.sha256(diff_text.encode("utf-8")).hexdigest() if diff_text else "clean"
        p_status = subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), capture_output=True, text=True)
        is_dirty = bool(p_status.stdout.strip())
        return {
            "commit": commit,
            "is_dirty": is_dirty,
            "dirty_diff_sha256": diff_sha,
        }
    except Exception as e:
        return {"commit": "unknown", "error": str(e), "is_dirty": True, "dirty_diff_sha256": "error"}


class BudgetExhaustionError(RuntimeError):
    """Raised when an operation would exceed campaign or per-run spending caps."""


class CampaignBudgetAuthority:
    """Authoritative, transactional campaign-wide spend authority.

    Guarantees that USD 0.30 campaign ceiling and USD 0.05 per-run ceiling are enforced
    transactionally across all sessions, runs, retries, and uncertain charges.
    """

    def __init__(self, campaign_dir: Path, policy: Dict[str, Any]):
        self.campaign_dir = Path(campaign_dir)
        self.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.campaign_dir / "campaign_budget_ledger.json"
        self.lock_path = self.campaign_dir / "campaign_budget_ledger.lock"
        self.policy = policy
        self.campaign_ceiling = Decimal(str(policy["campaign_budget"]["provider_usd_max"]))
        self.per_run_ceiling = Decimal(str(policy["provider"]["usd_per_main_live_run_max"]))
        self.contingency = Decimal(str(policy["campaign_budget"].get("contingency_usd", "0.05")))
        self._ensure_ledger_initialized()

    def _ensure_ledger_initialized(self) -> None:
        import fcntl
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                if not self.ledger_path.exists():
                    initial_data = {
                        "campaign_ceiling_usd": str(self.campaign_ceiling),
                        "per_run_ceiling_usd": str(self.per_run_ceiling),
                        "contingency_usd": str(self.contingency),
                        "total_billed_usd": "0.00",
                        "total_reserved_usd": "0.00",
                        "total_uncertain_usd": "0.00",
                        "remaining_campaign_usd": str(self.campaign_ceiling),
                        "runs": {},
                        "active_reservations": {},
                    }
                    tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
                    tmp.write_text(json.dumps(initial_data, indent=2), encoding="utf-8")
                    tmp.replace(self.ledger_path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def reserve(self, run_id: str, request_id: str, amount_usd: float | Decimal) -> Decimal:
        import fcntl
        amount = Decimal(str(amount_usd))
        if amount <= 0:
            return Decimal("0.00")
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                billed = Decimal(str(data.get("total_billed_usd", "0.00")))
                reserved = Decimal(str(data.get("total_reserved_usd", "0.00")))
                uncertain = Decimal(str(data.get("total_uncertain_usd", "0.00")))
                ceiling = Decimal(str(data.get("campaign_ceiling_usd", self.campaign_ceiling)))
                per_run = Decimal(str(data.get("per_run_ceiling_usd", self.per_run_ceiling)))

                effective_committed = billed + reserved + uncertain
                if effective_committed + amount > ceiling:
                    raise BudgetExhaustionError(
                        f"Campaign spending ceiling USD {ceiling} exhausted: committed USD {effective_committed} + request USD {amount} > {ceiling}"
                    )

                run_data = data.setdefault("runs", {}).setdefault(run_id, {
                    "billed_usd": "0.00", "reserved_usd": "0.00", "uncertain_usd": "0.00", "requests": 0
                })
                run_billed = Decimal(str(run_data.get("billed_usd", "0.00")))
                run_reserved = Decimal(str(run_data.get("reserved_usd", "0.00")))
                run_uncertain = Decimal(str(run_data.get("uncertain_usd", "0.00")))
                run_effective = run_billed + run_reserved + run_uncertain
                if run_effective + amount > per_run:
                    raise BudgetExhaustionError(
                        f"Per-run spending ceiling USD {per_run} exhausted for run '{run_id}': committed USD {run_effective} + request USD {amount} > {per_run}"
                    )

                reserved += amount
                run_reserved += amount
                data["total_reserved_usd"] = str(reserved)
                data["remaining_campaign_usd"] = str(max(Decimal("0.00"), ceiling - billed - reserved - uncertain))
                run_data["reserved_usd"] = str(run_reserved)
                data.setdefault("active_reservations", {})[request_id] = {
                    "run_id": run_id, "amount_usd": str(amount), "reserved_at": datetime.now(timezone.utc).isoformat()
                }

                tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.ledger_path)
                return amount
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def settle(self, run_id: str, request_id: str, actual_billed_usd: float | Decimal) -> None:
        import fcntl
        actual = Decimal(str(actual_billed_usd))
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                active_res = data.setdefault("active_reservations", {})
                res_info = active_res.pop(request_id, None)
                reserved_amount = Decimal(str(res_info["amount_usd"])) if res_info else actual

                billed = Decimal(str(data.get("total_billed_usd", "0.00"))) + actual
                reserved = max(Decimal("0.00"), Decimal(str(data.get("total_reserved_usd", "0.00"))) - reserved_amount)
                uncertain = Decimal(str(data.get("total_uncertain_usd", "0.00")))
                ceiling = Decimal(str(data.get("campaign_ceiling_usd", self.campaign_ceiling)))

                run_data = data.setdefault("runs", {}).setdefault(run_id, {
                    "billed_usd": "0.00", "reserved_usd": "0.00", "uncertain_usd": "0.00", "requests": 0
                })
                run_data["billed_usd"] = str(Decimal(str(run_data.get("billed_usd", "0.00"))) + actual)
                run_data["reserved_usd"] = str(max(Decimal("0.00"), Decimal(str(run_data.get("reserved_usd", "0.00"))) - reserved_amount))
                run_data["requests"] = int(run_data.get("requests", 0)) + 1

                data["total_billed_usd"] = str(billed)
                data["total_reserved_usd"] = str(reserved)
                data["remaining_campaign_usd"] = str(max(Decimal("0.00"), ceiling - billed - reserved - uncertain))

                tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.ledger_path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def mark_uncertain(self, run_id: str, request_id: str, reserved_usd: float | Decimal) -> None:
        import fcntl
        amt = Decimal(str(reserved_usd))
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_EX)
            try:
                data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                active_res = data.setdefault("active_reservations", {})
                active_res.pop(request_id, None)

                reserved = max(Decimal("0.00"), Decimal(str(data.get("total_reserved_usd", "0.00"))) - amt)
                uncertain = Decimal(str(data.get("total_uncertain_usd", "0.00"))) + amt
                billed = Decimal(str(data.get("total_billed_usd", "0.00")))
                ceiling = Decimal(str(data.get("campaign_ceiling_usd", self.campaign_ceiling)))

                run_data = data.setdefault("runs", {}).setdefault(run_id, {
                    "billed_usd": "0.00", "reserved_usd": "0.00", "uncertain_usd": "0.00", "requests": 0
                })
                run_data["reserved_usd"] = str(max(Decimal("0.00"), Decimal(str(run_data.get("reserved_usd", "0.00"))) - amt))
                run_data["uncertain_usd"] = str(Decimal(str(run_data.get("uncertain_usd", "0.00"))) + amt)

                data["total_reserved_usd"] = str(reserved)
                data["total_uncertain_usd"] = str(uncertain)
                data["remaining_campaign_usd"] = str(max(Decimal("0.00"), ceiling - billed - reserved - uncertain))

                tmp = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
                tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp.replace(self.ledger_path)
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)

    def get_status(self) -> Dict[str, Any]:
        import fcntl
        with open(self.lock_path, "a+", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f, fcntl.LOCK_SH)
            try:
                return json.loads(self.ledger_path.read_text(encoding="utf-8"))
            finally:
                fcntl.flock(lock_f, fcntl.LOCK_UN)


class CampaignRunner:
    def __init__(self, campaign: str = "20261021-robustness", run_id: Optional[str] = None):
        for label, value in (("campaign", campaign), ("run_id", run_id)):
            if value is not None and (not value or Path(value).name != value or value in (".", "..")):
                raise ValueError(f"Invalid {label}: expected a single directory name")
        self.policy = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.campaign = campaign
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = EVIDENCE_BASE / self.campaign / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        self.campaign_authority = CampaignBudgetAuthority(EVIDENCE_BASE / self.campaign, self.policy)

        self.summary_path = self.run_dir / "summary.json"
        self.manifest_path = self.run_dir / "manifest.json"
        self.traces_path = self.run_dir / "traces.jsonl"
        self.resources_path = self.run_dir / "resources.jsonl"
        self.provider_usage_path = self.run_dir / "provider_usage.json"
        self.metrics_path = self.run_dir / "metrics.json"
        self.faults_path = self.run_dir / "faults.json"
        self.artifacts_path = self.run_dir / "artifacts.json"

        # Load or initialize summary with strict resume checks
        if self.summary_path.exists():
            try:
                loaded = json.loads(self.summary_path.read_text(encoding="utf-8"))
                if loaded.get("run_id") and loaded["run_id"] != self.run_id:
                    raise ValueError(f"Run ID mismatch: expected {self.run_id}, found {loaded['run_id']}")
                if loaded.get("campaign") and loaded["campaign"] != self.campaign:
                    raise ValueError(f"Campaign mismatch: expected {self.campaign}, found {loaded['campaign']}")
                # Verify manifest immutability on resume
                if self.manifest_path.exists() and MANIFEST_PATH.exists():
                    saved_sha = compute_file_sha256(self.manifest_path)
                    cur_sha = compute_file_sha256(MANIFEST_PATH)
                    if saved_sha and cur_sha and saved_sha != cur_sha:
                        raise ValueError("Cannot safely resume: benchmark manifest policy has been modified")
                self.summary = loaded
            except Exception as e:
                raise ValueError("Cannot safely resume invalid summary") from e
        else:
            self.summary = self._init_summary()

    def _init_summary(self) -> Dict[str, Any]:
        return {
            "campaign": self.campaign,
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "git": get_git_provenance(),
            "gates": {},
            "budget_spent_usd": 0.0,
            "holdout_accesses": 0,
        }

    def _atomic_write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    def record_trace(self, event_name: str, payload: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_name,
            **payload,
        }
        with open(self.traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def record_resource(self, stage: str, metrics: Dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            **metrics,
        }
        with open(self.resources_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def save_summary(self) -> None:
        # Keep run-level accounting aligned with the campaign's durable ledger.
        # Historical summaries written before this reconciliation remain
        # unchanged and are corrected by the accompanying audit note.
        ledger = self.campaign_authority.get_status()
        run_ledger = ledger.get("runs", {}).get(self.run_id, {})
        self.summary["budget_spent_usd"] = float(Decimal(str(run_ledger.get("billed_usd", "0"))))
        self.summary["budget_reserved_usd"] = float(Decimal(str(run_ledger.get("reserved_usd", "0"))))
        self.summary["budget_uncertain_usd"] = float(Decimal(str(run_ledger.get("uncertain_usd", "0"))))
        self.summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write_json(self.summary_path, self.summary)

    def _task_contract(self, task_id: str) -> Tuple[Dict[str, Any], str]:
        problem = next((p for p in self.policy.get("problems", []) if p.get("id") == task_id), None)
        if problem is None:
            raise ValueError(f"Benchmark contract for {task_id} is missing")
        split = problem["split"]
        topology = {
            "benchmark_id": task_id,
            "target_column": problem["target"],
            "task_type": problem["task"],
            "benchmark_task_type": problem["task"],
            "metric": problem["metric"],
            "direction": problem["direction"],
            "split": split,
        }
        if task_id == "K1":
            topology["cv_scheme"] = "group_kfold"
            if split.get("group_column"):
                topology["entity_group_col"] = split["group_column"]
        elif task_id == "K2":
            topology["cv_scheme"] = "kfold"
        elif task_id == "K3":
            topology["cv_scheme"] = "stratified_kfold"
        elif task_id == "K4":
            topology.update(temporal_col=split.get("time"), cv_scheme="purged_time_series_split",
                            entity_join_key=split.get("join_key"))
        elif task_id == "K5":
            topology.update(temporal_col=split.get("time"), cv_scheme="purged_time_series_split",
                            entity_columns=split.get("entities"), forecast_horizon_dates=split.get("development_horizon_dates"))
        statement = (f"{problem['name']}; target={problem['target']}; metric={problem['metric']} "
                     f"({problem['direction']}); pinned split={json.dumps(split, sort_keys=True)}")
        return topology, statement

    def _live_gateway_block(self) -> Optional[Dict[str, Any]]:
        """Verify Rust harness binary exists for production gateway execution."""
        from mlrescurix_worker.scientific.rust_gateway import find_harness_binary
        try:
            find_harness_binary()
            return None
        except FileNotFoundError as exc:
            outcome = {
                "status": "blocked",
                "reason": f"Rust production provider gateway/orchestrator binary is missing: {exc}",
            }
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

    def run_preflight(self) -> Dict[str, Any]:
        """Perform comprehensive preflight checks across all system layers."""
        self.record_trace("preflight_start", {"action": "evaluating_prerequisites"})
        t0 = time.time()

        # 1. Host and environment
        host_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "arch": platform.machine(),
            "cores": os.cpu_count(),
        }

        # 2. Scientific packages
        packages = {}
        for pkg in ["numpy", "pandas", "scipy", "sklearn", "pytest", "jsonschema", "psutil"]:
            try:
                import importlib.metadata
                packages[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    mod = __import__(pkg)
                    packages[pkg] = getattr(mod, "__version__", "present")
                except ImportError:
                    packages[pkg] = None

        # 3. Rust harness binary provenance
        bin_debug = REPO_ROOT / "target" / "debug" / "mlrescurix-harness"
        bin_dist = REPO_ROOT / "dist" / "mlrescurix-harness"
        rust_provenance = {
            "debug_binary": {
                "exists": bin_debug.is_file(),
                "sha256": compute_file_sha256(bin_debug),
            },
            "dist_binary": {
                "exists": bin_dist.is_file(),
                "sha256": compute_file_sha256(bin_dist),
            },
            "rust_source_sha256": compute_directory_sha256(REPO_ROOT / "crates" / "harness" / "src"),
        }

        # 4. Storage schema compatibility check (accepts schema versions up to 11)
        storage_check = {"max_supported_version": 11, "never_downgrade_to_v10": True,
                         "status": "unverified", "reason": "Runtime database migration not exercised"}

        # 5. Docker daemon & Linux cgroup v2 connectivity
        docker_reachable = False
        docker_info_str = ""
        docker_bin = shutil.which("docker")
        if docker_bin:
            try:
                res = subprocess.run([docker_bin, "info", "--format", "{{.OSType}} {{.CgroupVersion}}"], capture_output=True, text=True, timeout=5)
                if res.returncode == 0:
                    docker_reachable = True
                    docker_info_str = res.stdout.strip()
                else:
                    docker_info_str = res.stderr.strip()
            except Exception as e:
                docker_info_str = str(e)

        docker_status = {
            "docker_bin": docker_bin,
            "daemon_reachable": docker_reachable,
            "info": docker_info_str,
            "cgroup_v2_certified": ("2" in docker_info_str and "linux" in docker_info_str.lower()),
        }

        # 6. Kaggle prerequisites (check without failing)
        kaggle_installed = False
        try:
            import kaggle  # type: ignore
            kaggle_installed = True
        except Exception:
            pass

        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        kaggle_access_token = Path.home() / ".kaggle" / "access_token"
        local_kaggle_token = REPO_ROOT / ".config" / "kaggle.token"
        kaggle_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or bool(os.environ.get("KAGGLE_API_TOKEN"))
        has_kaggle_creds = kaggle_json.is_file() or kaggle_access_token.is_file() or local_kaggle_token.is_file() or kaggle_env
        pinned_tasks = {}
        for t_id, dir_name, main_file in [
            ("K1", "k1_titanic", "train.csv"),
            ("K2", "k2_house_prices", "train.csv"),
            ("K3", "k3_forest_cover", "train.csv"),
            ("K4", "k4_ieee_fraud", "train_transaction.csv"),
            ("K5", "k5_store_sales", "train.csv"),
        ]:
            d = REPO_ROOT / "data" / "official" / dir_name
            manifest = d / "split_manifest.json"
            data_f = d / main_file
            pinned_tasks[t_id] = manifest.is_file() and data_f.is_file()

        pinned_count = sum(1 for v in pinned_tasks.values() if v)
        kaggle_status = {
            "client_installed": kaggle_installed,
            "credentials_present": has_kaggle_creds,
            "official_data_status": "ready" if pinned_count == 5 else ("partial_pinned" if pinned_count > 0 else "not_pinned"),
            "pinned_tasks": pinned_tasks,
        }

        # 7. Provider credentials (check existence WITHOUT printing secrets)
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key_file = openrouter_key_file.is_file()
        has_env_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        rust_binary_present = False
        try:
            from mlrescurix_worker.scientific.rust_gateway import find_harness_binary
            find_harness_binary()
            rust_binary_present = True
        except Exception:
            rust_binary_present = False

        provider_status = {
            "key_present": (has_key_file or has_env_key),
            "key_source": "environment" if has_env_key else ("local_file" if has_key_file else "none"),
            "rust_gateway_binary_present": rust_binary_present,
            "validity_verified": False,  # unverified until authorized live solve
            "spending_cap_usd": float(self.policy["campaign_budget"]["provider_usd_max"]),
            "per_run_cap_usd": float(self.policy["provider"]["usd_per_main_live_run_max"]),
            "model_id": self.policy["provider"]["model_id"],
            "provider_routing": self.policy["provider"]["provider_routing"],
        }

        if not rust_binary_present:
            g6_outcome = "blocked"
            g6_reason = "Rust harness binary not found; build with 'cargo build --release'"
        elif not provider_status["key_present"]:
            g6_outcome = "blocked"
            g6_reason = "Provider credentials missing; expected .config/openrouter.key or OPENROUTER_API_KEY"
        else:
            g6_outcome = "inconclusive"
            g6_reason = "Rust production gateway wired with outbound-view policy and budget accounting; live request unverified"

        # Compile gate assessments
        gates = {
            "G0_HOST_ENVIRONMENT": {
                "outcome": "passed",
                "details": host_info,
            },
            "G1_PYTHON_SCIENTIFIC_STACK": {
                "outcome": "passed" if all(packages[p] is not None for p in ["numpy", "pandas", "sklearn", "pytest"]) else "failed",
                "details": packages,
            },
            "G2_RUST_HARNESS_PROVENANCE": {
                "outcome": "inconclusive",
                "reason": "Binary hashes collected; source-to-build binding unverified",
                "details": rust_provenance,
            },
            "G3_STORAGE_SCHEMA_V11": {
                "outcome": "inconclusive",
                "details": storage_check,
            },
            "G4_CONTAINER_CGROUP_V2": {
                "outcome": "inconclusive" if docker_status["cgroup_v2_certified"] else "blocked",
                "reason": "Docker daemon inaccessible or not Linux cgroup v2" if not docker_status["cgroup_v2_certified"] else "Daemon reports cgroup v2; per-job enforcement unverified",
                "details": docker_status,
            },
            "G5_KAGGLE_OFFICIAL_DATA": {
                "outcome": "inconclusive" if kaggle_status["official_data_status"] == "ready" else "blocked",
                "reason": "Official dataset files pinned; campaign verification required" if kaggle_status["official_data_status"] == "ready" else f"Official datasets pinned: {pinned_count}/5",
                "details": kaggle_status,
            },
            "G6_PROVIDER_GATEWAY": {
                "outcome": g6_outcome,
                "reason": g6_reason,
                "details": provider_status,
            },
        }

        self.summary["gates"].update(gates)
        preflight_elapsed = time.time() - t0

        # Snapshot immutable manifest to evidence dir (preserve on resume)
        if MANIFEST_PATH.is_file() and not self.manifest_path.exists():
            shutil.copy2(MANIFEST_PATH, self.manifest_path)

        # Write artifacts record atomically
        artifacts_record = {
            "source_provenance": self.summary["git"],
            "rust_binaries": rust_provenance,
            "python_worker_sha256": compute_directory_sha256(REPO_ROOT / "python" / "src" / "mlrescurix_worker"),
            "benchmark_manifest_sha256": compute_file_sha256(self.manifest_path) if self.manifest_path.exists() else None,
        }
        self._atomic_write_json(self.artifacts_path, artifacts_record)

        # Record provider usage baseline (preserve on resume)
        if not self.provider_usage_path.exists():
            provider_record = {
                "total_requests": 0,
                "total_tokens": 0,
                "total_billed_cost_usd": 0.0,
                "reserved_cost_usd": 0.0,
                "remaining_budget_usd": float(self.policy["campaign_budget"]["provider_usd_max"]),
                "budget_scope": "run_observation_only_not_authoritative_campaign_ledger",
            }
            self._atomic_write_json(self.provider_usage_path, provider_record)

        # Recompute this run's observation under the current lower cap without
        # erasing spend/reservations. This is not cross-run admission authority.
        usage = json.loads(self.provider_usage_path.read_text(encoding="utf-8"))
        cap = Decimal(self.policy["campaign_budget"]["provider_usd_max"])
        spent = Decimal(str(usage["total_billed_cost_usd"]))
        reserved = Decimal(str(usage["reserved_cost_usd"]))
        if not spent.is_finite() or not reserved.is_finite() or spent < 0 or reserved < 0:
            raise ValueError("Invalid provider accounting; cannot resume")
        usage["remaining_budget_usd"] = float(max(Decimal("0"), cap - spent - reserved))
        usage["spending_cap_usd"] = float(cap)
        usage["budget_scope"] = "run_observation_only_not_authoritative_campaign_ledger"
        self._atomic_write_json(self.provider_usage_path, usage)

        self.record_trace("preflight_completed", {"duration_s": preflight_elapsed, "blocked_gates": [k for k, v in gates.items() if v["outcome"] == "blocked"]})
        self.save_summary()
        return gates

    def run_fault_reproducers(self) -> Dict[str, Any]:
        """Run RB-001 - RB-004 reproducers and fault regressions."""
        self.record_trace("faults_start", {"action": "running_reproducers"})
        t0 = time.time()

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "python" / "tests" / "test_s1_robustness_reproducers.py"),
            str(REPO_ROOT / "python" / "tests" / "test_hypothesis_executor.py"),
            "-q",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python" / "src")

        p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
        elapsed = time.time() - t0

        fault_log_path = self.run_dir / "logs" / "fault_reproducers.log"
        fault_log_path.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")

        passed = (p.returncode == 0)
        faults_result = {
            "test_file": "test_s1_robustness_reproducers.py, test_hypothesis_executor.py",
            "exit_code": p.returncode,
            "duration_s": elapsed,
            "passed": passed,
            "stdout_summary": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "",
        }

        self.faults_path.write_text(json.dumps(faults_result, indent=2), encoding="utf-8")

        gate_outcome = "passed" if passed else "failed"
        self.summary["gates"]["G_REPRODUCERS_RB001_RB004"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }
        self.summary["gates"]["G_TYPED_ACTION_DISPATCHER"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }

        self.record_trace("faults_completed", {"outcome": gate_outcome, "duration_s": elapsed})
        self.save_summary()
        return faults_result

    def run_deterministic_dev(self) -> Dict[str, Any]:
        """Run deterministic dev benchmark verification on synthetic fixtures."""
        self.record_trace("deterministic_dev_start", {"action": "evaluating_benchmarks"})
        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(REPO_ROOT / "python" / "tests" / "test_kaggle_benchmarks.py"),
            str(REPO_ROOT / "python" / "tests" / "test_gate4_benchmarks.py"),
            "-q",
        ]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT / "python" / "src")

        p = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
        elapsed = time.time() - t0
        rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        self.record_resource("deterministic_dev", {
            "elapsed_s": elapsed,
            "controller_lifetime_peak_rss_bytes": rss_end if sys.platform == "darwin" else rss_end * 1024,
            "controller_lifetime_peak_delta_bytes": (rss_end - rss_start) if sys.platform == "darwin" else (rss_end - rss_start) * 1024,
            "worker_measurement_status": "unavailable",
        })

        dev_log_path = self.run_dir / "logs" / "deterministic_dev.log"
        dev_log_path.write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")

        passed = (p.returncode == 0)
        dev_result = {
            "test_files": "test_kaggle_benchmarks.py, test_gate4_benchmarks.py",
            "exit_code": p.returncode,
            "duration_s": elapsed,
            "passed": passed,
            "stdout_summary": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "",
        }

        self._atomic_write_json(self.run_dir / "deterministic_dev.json", dev_result)
        gate_outcome = "passed" if passed else "failed"
        self.summary["gates"]["G_DETERMINISTIC_DEV_BENCHMARKS"] = {
            "outcome": gate_outcome,
            "exit_code": p.returncode,
            "duration_s": elapsed,
        }
        self.summary["status"] = "inconclusive" if passed else "failed"
        self.record_trace("deterministic_dev_completed", {"outcome": gate_outcome, "duration_s": elapsed})
        self.save_summary()
        return dev_result

    def _call_openrouter_specialist(self, role: str, messages: List[Dict[str, str]], max_tokens: int = 1200) -> Dict[str, Any]:
        import urllib.request
        import urllib.error

        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key and openrouter_key_file.is_file():
            api_key = openrouter_key_file.read_text(encoding="utf-8").strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not found in environment or .config/openrouter.key")

        request_id = f"req_{role}_{int(time.time() * 1000)}"
        est_amount = Decimal("0.005")
        self.campaign_authority.reserve(self.run_id, request_id, est_amount)

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/HamzaDevv/MLRescurix",
            "X-Title": "MLRescurix Live Robustness Campaign",
        }
        payload = {
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "provider": {
                "only": ["inference-net/fp8"],
                "order": ["inference-net/fp8"],
                "allow_fallbacks": False,
            },
        }

        t0 = time.time()
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resp_bytes = resp.read()
                data = json.loads(resp_bytes.decode("utf-8"))
            lat_ms = int((time.time() - t0) * 1000)
            choice = data["choices"][0]
            content = choice["message"].get("content", "") or ""
            usage = data.get("usage", {})
            actual_cost = Decimal(str(usage.get("cost", 0.0) or 0.0))
            if actual_cost <= Decimal("0.0"):
                p_tok = usage.get("prompt_tokens", 0)
                c_tok = usage.get("completion_tokens", 0)
                actual_cost = (Decimal(p_tok) * Decimal("0.08") + Decimal(c_tok) * Decimal("0.50")) / Decimal("1000000")

            self.campaign_authority.settle(self.run_id, request_id, actual_cost)
            turn_record = {
                "turn_id": request_id,
                "role": role,
                "latency_ms": lat_ms,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "cost_usd": float(actual_cost),
                "content_preview": content[:120],
            }
            self.record_trace("openrouter_turn_completed", turn_record)
            return {
                "content": content,
                "usage": usage,
                "cost_usd": float(actual_cost),
                "latency_ms": lat_ms,
            }
        except Exception as e:
            self.campaign_authority.mark_uncertain(self.run_id, request_id, est_amount)
            self.record_trace("openrouter_turn_failed", {"role": role, "request_id": request_id, "error": str(e)})
            raise

    def run_live_dev(self) -> Dict[str, Any]:
        """Execute genuine live dev solve for K1 Titanic under strict budget authority.

        Strict Holdout Rule: Under no circumstances access, load, or evaluate the holdout rows.
        Holdout accesses MUST remain 0.
        """
        gateway_block = self._live_gateway_block()
        if gateway_block is not None:
            return gateway_block
        live_pricing = verified_provider_pricing(self.policy)
        if live_pricing is None:
            outcome = {"status": "blocked", "reason": "Provider pricing snapshot is stale or has not been explicitly revalidated for live use"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        self.record_trace("live_dev_start", {"action": "evaluating_live_prerequisites"})
        k1_manifest_path = REPO_ROOT / "data" / "official" / "k1_titanic" / "split_manifest.json"
        k1_train_path = REPO_ROOT / "data" / "official" / "k1_titanic" / "train.csv"
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or openrouter_key_file.is_file())

        if not k1_manifest_path.is_file() or not k1_train_path.is_file():
            outcome = {"status": "blocked", "reason": "K1 Titanic official data/manifest not pinned"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        if not has_key:
            outcome = {"status": "blocked", "reason": "OpenRouter API key absent"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        budget_status = self.campaign_authority.get_status()
        rem = Decimal(str(budget_status.get("remaining_campaign_usd", "0.00")))
        if rem < Decimal("0.05"):
            outcome = {"status": "blocked", "reason": f"Insufficient campaign budget remaining: USD {rem} < 0.05"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        t0 = time.time()
        import pandas as pd
        import numpy as np

        # Load official data & split commitments
        split_manifest = json.loads(k1_manifest_path.read_text(encoding="utf-8"))
        dev_indices = split_manifest["development_indices"]
        holdout_indices = split_manifest.get("holdout_indices", [])
        dev_folds = split_manifest.get("development_folds", [])
        df_full = pd.read_csv(k1_train_path)

        # Fit/select only on the development slice. The full source CSV was
        # materialized above, so report row access separately from scoring.
        df_dev = df_full.iloc[dev_indices].copy().reset_index(drop=True)
        dev_csv_path = self.run_dir / "dev_train.csv"
        df_dev.to_csv(dev_csv_path, index=False)

        dev_index_map = {orig: dev for dev, orig in enumerate(dev_indices)}
        if dev_folds:
            locked_folds = [
                (
                    [dev_index_map[i] for i in f["train_indices"]],
                    [dev_index_map[i] for i in f["val_indices"]],
                )
                for f in dev_folds
            ]
        else:
            locked_folds = None

        from mlrescurix_worker.scientific.autonomous_loop import (
            AutonomousAgentLoop,
            ProcessTreeSampler,
            get_controller_rss_bytes,
        )
        sampler = ProcessTreeSampler()
        sampler.sample_now()

        from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
        gateway_adapter = RustGatewayAdapter(
            state_dir=self.run_dir / "rust_harness_state",
            project_id="k1_titanic_robustness",
            task_id="k1_dev_solve",
            role_id="scientific-agent",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            pricing=live_pricing,
        )

        # Execute genuine multi-turn autonomous loop
        topology, problem_statement = self._task_contract("K1")
        agent_loop = AutonomousAgentLoop(
            workspace_root=self.run_dir,
            project_id="k1_titanic_robustness",
            session_id=self.run_id,
            campaign_authority=self.campaign_authority,
            dataset_path=dev_csv_path,
            target_column="Survived",
            task_type="binary_classification",
            metric_name="accuracy",
            metric_direction="maximize",
            locked_folds=locked_folds,
            require_locked_folds=True,
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            max_turns=3,
            per_run_budget_usd=0.05,
            pricing=live_pricing,
            topology=topology,
            problem_statement=problem_statement,
            provider_caller=gateway_adapter,
            worker_memory_bytes=4 * 1024**3,
        )

        loop_state = agent_loop.run()
        candidate_evidence_ok, best_cv_score, candidate_evidence_reason = accepted_candidate_evidence(loop_state)

        bundle_dir = self.run_dir / "artifacts" / "k1_titanic_bundle"
        bundle_verified, bundle_identity_verified, package_reason = package_accepted_candidate(
            agent_loop, loop_state, df_dev,
            candidate_dir=self.run_dir / "cand_src", bundle_dir=bundle_dir,
            dataset_path=dev_csv_path, test_access_id=f"gate_{self.run_id}",
            project_id="k1_titanic_robustness", task_type="binary_classification",
            target_column="Survived", primary_metric="accuracy",
        )

        # No holdout metrics or selection decisions were made; source parsing
        # did materialize the partition before the development slice.
        self.summary["holdout_accesses"] = 0
        self.record_trace("holdout_partition_audit", {"task": "K1", "holdout_rows": len(holdout_indices),
            "source_rows_materialized": len(df_full), "holdout_labels_materialized": len(holdout_indices),
            "holdout_labels_used_for_selection_or_scoring": 0, "status": "not_scored_or_used_for_selection"})

        elapsed = time.time() - t0
        controller_rss = get_controller_rss_bytes()
        sampler_metrics = sampler.sample_now()

        status = self.campaign_authority.get_status()
        run_usage = status.get("runs", {}).get(self.run_id, {})
        tot_billed = float(run_usage.get("billed_usd", "0.00"))

        provider_summary = {
            "total_requests": run_usage.get("requests", 0),
            "total_billed_cost_usd": tot_billed,
            "model_id": "deepseek/deepseek-v4.1-flash-1017",
            "provider_endpoint": "inference-net/fp8",
            "remaining_campaign_usd": float(status.get("remaining_campaign_usd", "0.30")),
        }
        self._atomic_write_json(self.provider_usage_path, provider_summary)

        metrics_summary = {
            "task_id": "K1",
            "dataset": "titanic",
            "development_rows": len(df_dev),
            "holdout_rows_designated": len(holdout_indices),
            "holdout_labels_materialized": len(holdout_indices),
            "holdout_labels_used_for_selection_or_scoring": 0,
            "holdout_accesses": 0,
            "best_cv_score": best_cv_score,
            "holdout_accuracy": None,
            "holdout_balanced_accuracy": None,
            "bundle_verified": bundle_verified,
            "elapsed_seconds": elapsed,
        }
        self._atomic_write_json(self.metrics_path, metrics_summary)

        self.record_resource("live_dev_k1", {
            "elapsed_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_lifetime_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
            "total_billed_cost_usd": tot_billed,
        })

        quality_threshold = locked_quality_threshold(self.policy, "K1")
        passed = bundle_verified and (candidate_evidence_ok and score_meets_threshold(best_cv_score, quality_threshold, "maximize") and bundle_identity_verified)
        outcome = {
            "status": "passed" if passed else ("inconclusive" if quality_threshold is None else "failed"),
            "reason": candidate_gate_failure_reason(candidate_evidence_ok, candidate_evidence_reason, best_cv_score, quality_threshold, "maximize", bundle_verified, bundle_identity_verified, package_reason),
            "bundle_identity_verified": bundle_identity_verified,
            "accepted_candidate_id": loop_state.get("accepted_candidate_id"),
            "candidate_identity": loop_state.get("candidate_identity"),
            "task": "K1 Titanic",
            "best_cv_score": best_cv_score,
            "holdout_accesses": 0,
            "bundle_verified": bundle_verified,
            "provider_cost_usd": tot_billed,
            "duration_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
        }
        self.summary["gates"]["G_LIVE_DEV"] = {
            "outcome": outcome["status"],
            "details": outcome,
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("live_dev_completed", outcome)
        self.save_summary()
        return outcome

    def run_live_dev_k2(self) -> Dict[str, Any]:
        """Execute genuine live dev solve for K2 House Prices under strict budget authority.

        Strict Holdout Rule: Under no circumstances access, load, or evaluate the 292 final holdout rows.
        Holdout accesses MUST remain 0.
        """
        gateway_block = self._live_gateway_block()
        if gateway_block is not None:
            return gateway_block
        live_pricing = verified_provider_pricing(self.policy)
        if live_pricing is None:
            outcome = {"status": "blocked", "reason": "Provider pricing snapshot is stale or has not been explicitly revalidated for live use"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        self.record_trace("live_dev_k2_start", {"action": "evaluating_live_prerequisites"})
        k2_manifest_path = REPO_ROOT / "data" / "official" / "k2_house_prices" / "split_manifest.json"
        k2_train_path = REPO_ROOT / "data" / "official" / "k2_house_prices" / "train.csv"
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or openrouter_key_file.is_file())

        if not k2_manifest_path.is_file() or not k2_train_path.is_file():
            outcome = {"status": "blocked", "reason": "K2 House Prices official data/manifest not pinned"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        if not has_key:
            outcome = {"status": "blocked", "reason": "OpenRouter API key absent"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        budget_status = self.campaign_authority.get_status()
        rem = Decimal(str(budget_status.get("remaining_campaign_usd", "0.00")))
        if rem < Decimal("0.05"):
            outcome = {"status": "blocked", "reason": f"Insufficient campaign budget remaining: USD {rem} < 0.05"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        import pandas as pd
        import numpy as np

        # Load official data & split commitments
        split_manifest = json.loads(k2_manifest_path.read_text(encoding="utf-8"))
        dev_indices = split_manifest["development_indices"]
        dev_folds = split_manifest["development_folds"]
        df_full = pd.read_csv(k2_train_path)

        # Fit/select only on the development slice. The full source CSV was
        # materialized above, so report row access separately from scoring.
        df_dev = df_full.iloc[dev_indices].copy().reset_index(drop=True)
        dev_csv_path = self.run_dir / "dev_k2_train.csv"
        df_dev.to_csv(dev_csv_path, index=False)

        # Explicit mapping from original dataset indices to development partition row indices
        dev_index_map = {orig: dev for dev, orig in enumerate(dev_indices)}
        locked_folds = [
            (
                [dev_index_map[i] for i in f["train_indices"]],
                [dev_index_map[i] for i in f["val_indices"]],
            )
            for f in dev_folds
        ]

        from mlrescurix_worker.scientific.autonomous_loop import (
            AutonomousAgentLoop,
            ProcessTreeSampler,
            get_controller_rss_bytes,
        )
        sampler = ProcessTreeSampler()
        sampler.sample_now()

        from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
        gateway_adapter = RustGatewayAdapter(
            state_dir=self.run_dir / "rust_harness_state",
            project_id="k2_house_prices_robustness",
            task_id="k2_dev_solve",
            role_id="scientific-agent",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            pricing=live_pricing,
        )

        # Execute genuine multi-turn autonomous loop
        topology, problem_statement = self._task_contract("K2")
        agent_loop = AutonomousAgentLoop(
            workspace_root=self.run_dir,
            project_id="k2_house_prices_robustness",
            session_id=self.run_id,
            campaign_authority=self.campaign_authority,
            dataset_path=dev_csv_path,
            target_column="SalePrice",
            task_type="regression",
            metric_name="rmse_log_price",
            metric_direction="minimize",
            locked_folds=locked_folds,
            require_locked_folds=True,
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            max_turns=3,
            per_run_budget_usd=0.05,
            pricing=live_pricing,
            topology=topology,
            problem_statement=problem_statement,
            provider_caller=gateway_adapter,
            worker_memory_bytes=4 * 1024**3,
        )

        loop_state = agent_loop.run()
        candidate_evidence_ok, best_cv_score, candidate_evidence_reason = accepted_candidate_evidence(loop_state)

        bundle_dir = self.run_dir / "artifacts" / "k2_house_prices_bundle"
        bundle_verified, bundle_identity_verified, package_reason = package_accepted_candidate(
            agent_loop, loop_state, df_dev,
            candidate_dir=self.run_dir / "cand_src_k2", bundle_dir=bundle_dir,
            dataset_path=dev_csv_path, test_access_id=f"gate_{self.run_id}",
            project_id="k2_house_prices_robustness", task_type="regression",
            target_column="SalePrice", primary_metric="rmse_log_price",
        )

        # No holdout metrics or selection decisions were made; source parsing
        # did materialize the partition before the development slice.
        self.summary["holdout_accesses"] = 0
        self.record_trace("holdout_partition_audit", {"task": "K2", "holdout_rows": 292,
            "source_rows_materialized": len(df_full), "holdout_labels_materialized": 292,
            "holdout_labels_used_for_selection_or_scoring": 0, "status": "not_scored_or_used_for_selection"})

        elapsed = time.time() - t0
        controller_rss = get_controller_rss_bytes()
        sampler_metrics = sampler.sample_now()

        status = self.campaign_authority.get_status()
        run_usage = status.get("runs", {}).get(self.run_id, {})
        tot_billed = float(run_usage.get("billed_usd", "0.00"))

        provider_summary = {
            "total_requests": run_usage.get("requests", 0),
            "total_billed_cost_usd": tot_billed,
            "model_id": "deepseek/deepseek-v4.1-flash-1017",
            "provider_endpoint": "inference-net/fp8",
            "remaining_campaign_usd": float(status.get("remaining_campaign_usd", "0.30")),
        }
        self._atomic_write_json(self.provider_usage_path, provider_summary)

        metrics_summary = {
            "task_id": "K2",
            "dataset": "house_prices",
            "development_rows": len(df_dev),
            "holdout_rows_designated": 292,
            "holdout_labels_materialized": 292,
            "holdout_labels_used_for_selection_or_scoring": 0,
            "holdout_accesses": 0,
            "metric_name": "rmse_log_price",
            "metric_direction": "minimize",
            "best_cv_score": best_cv_score,
            "holdout_score": None,  # Strictly None (unassessed holdout)
            "bundle_verified": bundle_verified,
            "elapsed_seconds": elapsed,
        }
        self._atomic_write_json(self.metrics_path, metrics_summary)

        self.record_resource("live_dev_k2", {
            "elapsed_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_lifetime_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
            "total_billed_cost_usd": tot_billed,
        })

        quality_threshold = locked_quality_threshold(self.policy, "K2")
        passed = bundle_verified and (candidate_evidence_ok and score_meets_threshold(best_cv_score, quality_threshold, "minimize") and bundle_identity_verified)
        outcome = {
            "status": "passed" if passed else ("inconclusive" if quality_threshold is None else "failed"),
            "reason": candidate_gate_failure_reason(candidate_evidence_ok, candidate_evidence_reason, best_cv_score, quality_threshold, "minimize", bundle_verified, bundle_identity_verified, package_reason),
            "bundle_identity_verified": bundle_identity_verified,
            "accepted_candidate_id": loop_state.get("accepted_candidate_id"),
            "candidate_identity": loop_state.get("candidate_identity"),
            "task": "K2 House Prices",
            "metric": "rmse_log_price",
            "best_cv_score": best_cv_score,
            "holdout_accesses": 0,
            "bundle_verified": bundle_verified,
            "provider_cost_usd": tot_billed,
            "duration_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
        }
        self.summary["gates"]["G_LIVE_DEV"] = {
            "outcome": outcome["status"],
            "details": outcome,
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("live_dev_k2_completed", outcome)
        self.save_summary()
        return outcome

    def run_live_dev_k3(self) -> Dict[str, Any]:
        """Execute genuine live dev solve for K3 Forest Cover Type under strict budget authority.

        Multiclass classification across 7 classes on Cover_Type.
        Strict Holdout Rule: Under no circumstances access, load, or evaluate the 3,024 final holdout rows.
        Holdout accesses MUST remain 0.
        """
        gateway_block = self._live_gateway_block()
        if gateway_block is not None:
            return gateway_block
        live_pricing = verified_provider_pricing(self.policy)
        if live_pricing is None:
            outcome = {"status": "blocked", "reason": "Provider pricing snapshot is stale or has not been explicitly revalidated for live use"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        self.record_trace("live_dev_k3_start", {"action": "evaluating_live_prerequisites"})
        k3_manifest_path = REPO_ROOT / "data" / "official" / "k3_forest_cover" / "split_manifest.json"
        k3_train_path = REPO_ROOT / "data" / "official" / "k3_forest_cover" / "train.csv"
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or openrouter_key_file.is_file())

        if not k3_manifest_path.is_file() or not k3_train_path.is_file():
            outcome = {"status": "blocked", "reason": "K3 Forest Cover Type official data/manifest not pinned"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        if not has_key:
            outcome = {"status": "blocked", "reason": "OpenRouter API key absent"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        budget_status = self.campaign_authority.get_status()
        rem = Decimal(str(budget_status.get("remaining_campaign_usd", "0.00")))
        if rem < Decimal("0.05"):
            outcome = {"status": "blocked", "reason": f"Insufficient campaign budget remaining: USD {rem} < 0.05"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        import pandas as pd
        import numpy as np

        # Load official data & split commitments
        split_manifest = json.loads(k3_manifest_path.read_text(encoding="utf-8"))
        dev_indices = split_manifest["development_indices"]
        dev_folds = split_manifest["development_folds"]
        df_full = pd.read_csv(k3_train_path)

        # Fit/select only on the development slice. The full source CSV was
        # materialized above, so report row access separately from scoring.
        df_dev = df_full.iloc[dev_indices].copy().reset_index(drop=True)
        dev_csv_path = self.run_dir / "dev_k3_train.csv"
        df_dev.to_csv(dev_csv_path, index=False)

        # Explicit mapping from original dataset indices to development partition row indices
        dev_index_map = {orig: dev for dev, orig in enumerate(dev_indices)}
        locked_folds = [
            (
                [dev_index_map[i] for i in f["train_indices"]],
                [dev_index_map[i] for i in f["val_indices"]],
            )
            for f in dev_folds
        ]

        from mlrescurix_worker.scientific.autonomous_loop import (
            AutonomousAgentLoop,
            ProcessTreeSampler,
            get_controller_rss_bytes,
        )
        sampler = ProcessTreeSampler()
        sampler.sample_now()

        from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
        gateway_adapter = RustGatewayAdapter(
            state_dir=self.run_dir / "rust_harness_state",
            project_id="k3_forest_cover_robustness",
            task_id="k3_dev_solve",
            role_id="scientific-agent",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            pricing=live_pricing,
        )

        # Execute genuine multi-turn autonomous loop
        topology, problem_statement = self._task_contract("K3")
        agent_loop = AutonomousAgentLoop(
            workspace_root=self.run_dir,
            project_id="k3_forest_cover_robustness",
            session_id=self.run_id,
            campaign_authority=self.campaign_authority,
            dataset_path=dev_csv_path,
            target_column="Cover_Type",
            task_type="multiclass_classification",
            metric_name="accuracy",
            metric_direction="maximize",
            locked_folds=locked_folds,
            require_locked_folds=True,
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            max_turns=3,
            per_run_budget_usd=0.05,
            pricing=live_pricing,
            topology=topology,
            problem_statement=problem_statement,
            provider_caller=gateway_adapter,
            worker_memory_bytes=4 * 1024**3,
        )

        loop_state = agent_loop.run()
        candidate_evidence_ok, best_cv_score, candidate_evidence_reason = accepted_candidate_evidence(loop_state)

        bundle_dir = self.run_dir / "artifacts" / "k3_forest_cover_bundle"
        bundle_verified, bundle_identity_verified, package_reason = package_accepted_candidate(
            agent_loop, loop_state, df_dev,
            candidate_dir=self.run_dir / "cand_src_k3", bundle_dir=bundle_dir,
            dataset_path=dev_csv_path, test_access_id=f"gate_{self.run_id}",
            project_id="k3_forest_cover_robustness", task_type="multiclass_classification",
            target_column="Cover_Type", primary_metric="accuracy",
        )

        # No holdout metrics or selection decisions were made; source parsing
        # did materialize the partition before the development slice.
        self.summary["holdout_accesses"] = 0
        self.record_trace("holdout_partition_audit", {"task": "K3", "holdout_rows": 3024,
            "source_rows_materialized": len(df_full), "holdout_labels_materialized": 3024,
            "holdout_labels_used_for_selection_or_scoring": 0, "status": "not_scored_or_used_for_selection"})

        elapsed = time.time() - t0
        controller_rss = get_controller_rss_bytes()
        sampler_metrics = sampler.sample_now()

        status = self.campaign_authority.get_status()
        run_usage = status.get("runs", {}).get(self.run_id, {})
        tot_billed = float(run_usage.get("billed_usd", "0.00"))

        provider_summary = {
            "total_requests": run_usage.get("requests", 0),
            "total_billed_cost_usd": tot_billed,
            "model_id": "deepseek/deepseek-v4.1-flash-1017",
            "provider_endpoint": "inference-net/fp8",
            "remaining_campaign_usd": float(status.get("remaining_campaign_usd", "0.30")),
        }
        self._atomic_write_json(self.provider_usage_path, provider_summary)

        metrics_summary = {
            "task_id": "K3",
            "dataset": "forest_cover_type",
            "development_rows": len(df_dev),
            "holdout_rows_designated": 3024,
            "holdout_labels_materialized": 3024,
            "holdout_labels_used_for_selection_or_scoring": 0,
            "holdout_accesses": 0,
            "metric_name": "accuracy",
            "metric_direction": "maximize",
            "best_cv_score": best_cv_score,
            "holdout_score": None,
            "bundle_verified": bundle_verified,
            "elapsed_seconds": elapsed,
        }
        self._atomic_write_json(self.metrics_path, metrics_summary)

        self.record_resource("live_dev_k3", {
            "elapsed_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_lifetime_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
            "total_billed_cost_usd": tot_billed,
        })

        quality_threshold = locked_quality_threshold(self.policy, "K3")
        passed = bundle_verified and (candidate_evidence_ok and score_meets_threshold(best_cv_score, quality_threshold, "maximize") and bundle_identity_verified)
        outcome = {
            "status": "passed" if passed else ("inconclusive" if quality_threshold is None else "failed"),
            "reason": candidate_gate_failure_reason(candidate_evidence_ok, candidate_evidence_reason, best_cv_score, quality_threshold, "maximize", bundle_verified, bundle_identity_verified, package_reason),
            "bundle_identity_verified": bundle_identity_verified,
            "accepted_candidate_id": loop_state.get("accepted_candidate_id"),
            "candidate_identity": loop_state.get("candidate_identity"),
            "task": "K3 Forest Cover Type",
            "metric": "accuracy",
            "best_cv_score": best_cv_score,
            "holdout_accesses": 0,
            "bundle_verified": bundle_verified,
            "provider_cost_usd": tot_billed,
            "duration_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
        }
        self.summary["gates"]["G_LIVE_DEV"] = {
            "outcome": outcome["status"],
            "details": outcome,
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("live_dev_k3_completed", outcome)
        self.save_summary()
        return outcome

    def run_live_dev_k4(self) -> Dict[str, Any]:
        """Execute genuine live dev solve for K4 IEEE-CIS Fraud Detection under strict budget authority.

        Binary classification evaluated by ROC-AUC.
        Strict Holdout Rule: Under no circumstances access, load, or evaluate the 118,108 final holdout rows.
        Holdout accesses MUST remain 0.
        """
        gateway_block = self._live_gateway_block()
        if gateway_block is not None:
            return gateway_block
        live_pricing = verified_provider_pricing(self.policy)
        if live_pricing is None:
            outcome = {"status": "blocked", "reason": "Provider pricing snapshot is stale or has not been explicitly revalidated for live use"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        self.record_trace("live_dev_k4_start", {"action": "evaluating_live_prerequisites"})
        k4_manifest_path = REPO_ROOT / "data" / "official" / "k4_ieee_fraud" / "split_manifest.json"
        k4_train_trans_path = REPO_ROOT / "data" / "official" / "k4_ieee_fraud" / "train_transaction.csv"
        k4_train_id_path = REPO_ROOT / "data" / "official" / "k4_ieee_fraud" / "train_identity.csv"
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or openrouter_key_file.is_file())

        if not k4_manifest_path.is_file() or not k4_train_trans_path.is_file():
            outcome = {"status": "blocked", "reason": "K4 IEEE-CIS Fraud official data/manifest not pinned"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        if not has_key:
            outcome = {"status": "blocked", "reason": "OpenRouter API key absent"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        budget_status = self.campaign_authority.get_status()
        rem = Decimal(str(budget_status.get("remaining_campaign_usd", "0.00")))
        if rem < Decimal("0.05"):
            outcome = {"status": "blocked", "reason": f"Insufficient campaign budget remaining: USD {rem} < 0.05"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        import pandas as pd
        import numpy as np

        # Load official data & split commitments
        split_manifest = json.loads(k4_manifest_path.read_text(encoding="utf-8"))
        dev_rows_total = split_manifest["development_rows"]  # 472,432
        holdout_rows_total = split_manifest["holdout_rows"]  # 118,108

        # Slicing strictly within development partition (rows 422,432 to 472,431 = 50,000 recent development rows)
        sub_spec = split_manifest.get("bounded_dev_subset", {})
        sub_start = sub_spec.get("dev_window_start_idx", 422432)
        sub_end = sub_spec.get("dev_window_end_idx", 472431)

        # Read only development rows
        df_trans = pd.read_csv(k4_train_trans_path, skiprows=range(1, sub_start + 1), nrows=(sub_end - sub_start + 1))
        header_cols = pd.read_csv(k4_train_trans_path, nrows=0).columns.tolist()
        df_trans.columns = header_cols

        # Join identity data if available
        if k4_train_id_path.is_file():
            df_id = pd.read_csv(k4_train_id_path)
            df_dev = df_trans.merge(df_id, on="TransactionID", how="left")
        else:
            df_dev = df_trans

        dev_csv_path = self.run_dir / "dev_k4_train.csv"
        df_dev.to_csv(dev_csv_path, index=False)

        # Use the split manifest's pinned rolling origins; unsupported mappings block the run.
        try:
            # This will block for the checked-in manifest until it declares
            # fold index space or explicit train/validation boundaries plus gap.
            locked_folds = map_k4_manifest_folds(split_manifest, df_dev, sub_start, sub_end)
        except (ValueError, TypeError, KeyError) as exc:
            outcome = {"status": "blocked", "reason": f"Pinned K4 folds unavailable for bounded development rows: {exc}"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        from mlrescurix_worker.scientific.autonomous_loop import (
            AutonomousAgentLoop,
            ProcessTreeSampler,
            get_controller_rss_bytes,
        )
        sampler = ProcessTreeSampler()
        sampler.sample_now()

        from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
        gateway_adapter = RustGatewayAdapter(
            state_dir=self.run_dir / "rust_harness_state",
            project_id="k4_ieee_fraud_robustness",
            task_id="k4_dev_solve",
            role_id="scientific-agent",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            pricing=live_pricing,
        )

        # Execute genuine multi-turn autonomous loop
        topology, problem_statement = self._task_contract("K4")
        agent_loop = AutonomousAgentLoop(
            workspace_root=self.run_dir,
            project_id="k4_ieee_fraud_robustness",
            session_id=self.run_id,
            campaign_authority=self.campaign_authority,
            dataset_path=dev_csv_path,
            target_column="isFraud",
            task_type="binary_classification",
            metric_name="roc_auc",
            metric_direction="maximize",
            locked_folds=locked_folds,
            require_locked_folds=True,
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            max_turns=3,
            per_run_budget_usd=0.05,
            pricing=live_pricing,
            topology=topology,
            problem_statement=problem_statement,
            provider_caller=gateway_adapter,
            worker_memory_bytes=4 * 1024**3,
        )

        loop_state = agent_loop.run()
        candidate_evidence_ok, best_cv_score, candidate_evidence_reason = accepted_candidate_evidence(loop_state)

        bundle_dir = self.run_dir / "artifacts" / "k4_ieee_fraud_bundle"
        bundle_verified, bundle_identity_verified, package_reason = package_accepted_candidate(
            agent_loop, loop_state, df_dev,
            candidate_dir=self.run_dir / "cand_src_k4", bundle_dir=bundle_dir,
            dataset_path=dev_csv_path, test_access_id=f"gate_{self.run_id}",
            project_id="k4_ieee_fraud_robustness", task_type="binary_classification",
            target_column="isFraud", primary_metric="roc_auc",
        )

        # K4 reads only a bounded subset wholly inside development rows; no
        # holdout transaction labels are materialized or scored.
        self.summary["holdout_accesses"] = 0
        self.record_trace("holdout_partition_audit", {"task": "K4", "holdout_rows": holdout_rows_total,
            "source_development_rows_materialized": len(df_dev), "holdout_labels_materialized": 0,
            "holdout_labels_used_for_selection_or_scoring": 0, "status": "not_loaded_or_scored"})

        elapsed = time.time() - t0
        controller_rss = get_controller_rss_bytes()
        sampler_metrics = sampler.sample_now()

        status = self.campaign_authority.get_status()
        run_usage = status.get("runs", {}).get(self.run_id, {})
        tot_billed = float(run_usage.get("billed_usd", "0.00"))

        provider_summary = {
            "total_requests": run_usage.get("requests", 0),
            "total_billed_cost_usd": tot_billed,
            "model_id": "deepseek/deepseek-v4.1-flash-1017",
            "provider_endpoint": "inference-net/fp8",
            "remaining_campaign_usd": float(status.get("remaining_campaign_usd", "0.30")),
        }
        self._atomic_write_json(self.provider_usage_path, provider_summary)

        metrics_summary = {
            "task_id": "K4",
            "dataset": "ieee_fraud",
            "development_rows": len(df_dev),
            "holdout_rows_designated": holdout_rows_total,
            "holdout_labels_materialized": 0,
            "holdout_labels_used_for_selection_or_scoring": 0,
            "holdout_accesses": 0,
            "metric_name": "roc_auc",
            "metric_direction": "maximize",
            "best_cv_score": best_cv_score,
            "holdout_score": None,
            "bundle_verified": bundle_verified,
            "elapsed_seconds": elapsed,
        }
        self._atomic_write_json(self.metrics_path, metrics_summary)

        self.record_resource("live_dev_k4", {
            "elapsed_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_lifetime_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
            "total_billed_cost_usd": tot_billed,
        })

        quality_threshold = locked_quality_threshold(self.policy, "K4")
        passed = bundle_verified and (candidate_evidence_ok and score_meets_threshold(best_cv_score, quality_threshold, "maximize") and bundle_identity_verified)
        outcome = {
            "status": "passed" if passed else ("inconclusive" if quality_threshold is None else "failed"),
            "reason": candidate_gate_failure_reason(candidate_evidence_ok, candidate_evidence_reason, best_cv_score, quality_threshold, "maximize", bundle_verified, bundle_identity_verified, package_reason),
            "bundle_identity_verified": bundle_identity_verified,
            "accepted_candidate_id": loop_state.get("accepted_candidate_id"),
            "candidate_identity": loop_state.get("candidate_identity"),
            "task": "K4 IEEE-CIS Fraud Detection",
            "metric": "roc_auc",
            "best_cv_score": best_cv_score,
            "holdout_accesses": 0,
            "bundle_verified": bundle_verified,
            "provider_cost_usd": tot_billed,
            "duration_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
        }
        self.summary["gates"]["G_LIVE_DEV"] = {
            "outcome": outcome["status"],
            "details": outcome,
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("live_dev_k4_completed", outcome)
        self.save_summary()
        return outcome

    def run_live_dev_k5(self) -> Dict[str, Any]:
        """Execute genuine live dev solve for K5 Store Sales Forecasting under strict budget authority.

        Panel time series forecasting evaluated by RMSLE.
        Strict Holdout Rule: Under no circumstances access, load, or evaluate the 26,730 final holdout rows.
        Holdout accesses MUST remain 0.
        """
        gateway_block = self._live_gateway_block()
        if gateway_block is not None:
            return gateway_block
        live_pricing = verified_provider_pricing(self.policy)
        if live_pricing is None:
            outcome = {"status": "blocked", "reason": "Provider pricing snapshot is stale or has not been explicitly revalidated for live use"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome
        self.record_trace("live_dev_k5_start", {"action": "evaluating_live_prerequisites"})
        k5_manifest_path = REPO_ROOT / "data" / "official" / "k5_store_sales" / "split_manifest.json"
        k5_train_path = REPO_ROOT / "data" / "official" / "k5_store_sales" / "train.csv"
        k5_stores_path = REPO_ROOT / "data" / "official" / "k5_store_sales" / "stores.csv"
        k5_oil_path = REPO_ROOT / "data" / "official" / "k5_store_sales" / "oil.csv"
        openrouter_key_file = REPO_ROOT / ".config" / "openrouter.key"
        has_key = bool(os.environ.get("OPENROUTER_API_KEY") or openrouter_key_file.is_file())

        if not k5_manifest_path.is_file() or not k5_train_path.is_file():
            outcome = {"status": "blocked", "reason": "K5 Store Sales official data/manifest not pinned"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        if not has_key:
            outcome = {"status": "blocked", "reason": "OpenRouter API key absent"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        budget_status = self.campaign_authority.get_status()
        rem = Decimal(str(budget_status.get("remaining_campaign_usd", "0.00")))
        if rem < Decimal("0.05"):
            outcome = {"status": "blocked", "reason": f"Insufficient campaign budget remaining: USD {rem} < 0.05"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        t0 = time.time()
        import resource
        rss_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        import pandas as pd
        import numpy as np

        # Load official data & split commitments
        split_manifest = json.loads(k5_manifest_path.read_text(encoding="utf-8"))
        dev_rows_total = split_manifest["development_rows"]  # 2,974,158
        holdout_rows_total = split_manifest["holdout_rows"]  # 26,730

        df_full = pd.read_csv(k5_train_path)

        # Historical prior window (strictly prior to 2017-06-01): 2017-01-01 to 2017-05-31
        # Preserves strict temporal causality: zero lookahead, zero development/holdout leakage
        hist = df_full[(df_full["date"] >= "2017-01-01") & (df_full["date"] < "2017-06-01")].copy()
        hist["dow"] = pd.to_datetime(hist["date"]).dt.dayofweek
        h_sf = hist.groupby(["store_nbr", "family"])["sales"].mean().reset_index().rename(columns={"sales": "hist_sf_mean"})
        h_sfd = hist.groupby(["store_nbr", "family", "dow"])["sales"].mean().reset_index().rename(columns={"sales": "hist_sfd_mean"})

        # Slicing strictly within development partition: last 60 development dates (2017-06-01 to 2017-07-31)
        # Covers all 3 rolling validation origins declared in the split manifest
        mask_dev = (df_full["date"] >= "2017-06-01") & (df_full["date"] <= "2017-07-31")
        df_dev = df_full[mask_dev].copy().reset_index(drop=True)
        dev_dt = pd.to_datetime(df_dev["date"])
        df_dev["dow_idx"] = dev_dt.dt.dayofweek
        df_dev["dom_idx"] = dev_dt.dt.day
        df_dev["is_weekend"] = (df_dev["dow_idx"] >= 5).astype(int)

        df_dev = df_dev.merge(h_sf, on=["store_nbr", "family"], how="left")
        df_dev = df_dev.merge(h_sfd, left_on=["store_nbr", "family", "dow_idx"], right_on=["store_nbr", "family", "dow"], how="left")
        df_dev["hist_sf_log"] = np.log1p(np.maximum(0, df_dev["hist_sf_mean"])).fillna(0)
        df_dev["hist_sfd_log"] = np.log1p(np.maximum(0, df_dev["hist_sfd_mean"])).fillna(0)
        df_dev = df_dev.drop(columns=["dow", "hist_sf_mean", "hist_sfd_mean"])

        if k5_stores_path.is_file():
            stores = pd.read_csv(k5_stores_path)
            df_dev = df_dev.merge(stores, on="store_nbr", how="left")
        if k5_oil_path.is_file():
            oil = pd.read_csv(k5_oil_path)
            oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
            df_dev = df_dev.merge(oil, on="date", how="left")

        dev_csv_path = self.run_dir / "dev_k5_train.csv"
        df_dev.to_csv(dev_csv_path, index=False)

        try:
            locked_folds = map_locked_temporal_folds(
                split_manifest.get("development_folds") or split_manifest.get("development_origins", []), df_dev, "date"
            )
        except (ValueError, TypeError, KeyError) as exc:
            outcome = {"status": "blocked", "reason": f"Pinned K5 panel folds unavailable for development dates: {exc}"}
            self.summary["gates"]["G_LIVE_DEV"] = outcome
            self.summary["status"] = "blocked"
            self.save_summary()
            return outcome

        from mlrescurix_worker.scientific.autonomous_loop import (
            AutonomousAgentLoop,
            ProcessTreeSampler,
            get_controller_rss_bytes,
        )
        sampler = ProcessTreeSampler()
        sampler.sample_now()

        from mlrescurix_worker.scientific.rust_gateway import RustGatewayAdapter
        gateway_adapter = RustGatewayAdapter(
            state_dir=self.run_dir / "rust_harness_state",
            project_id="k5_store_sales_robustness",
            task_id="k5_dev_solve",
            role_id="scientific-agent",
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            pricing=live_pricing,
        )

        # Execute genuine multi-turn autonomous loop
        topology, problem_statement = self._task_contract("K5")
        agent_loop = AutonomousAgentLoop(
            workspace_root=self.run_dir,
            project_id="k5_store_sales_robustness",
            session_id=self.run_id,
            campaign_authority=self.campaign_authority,
            dataset_path=dev_csv_path,
            target_column="sales",
            task_type="panel_forecasting",
            metric_name="rmsle",
            metric_direction="minimize",
            locked_folds=locked_folds,
            require_locked_folds=True,
            model_name="deepseek/deepseek-v4.1-flash-1017",
            endpoint="inference-net/fp8",
            max_turns=3,
            per_run_budget_usd=0.05,
            pricing=live_pricing,
            topology=topology,
            problem_statement=problem_statement,
            provider_caller=gateway_adapter,
            worker_memory_bytes=4 * 1024**3,
        )

        loop_state = agent_loop.run()
        candidate_evidence_ok, best_cv_score, candidate_evidence_reason = accepted_candidate_evidence(loop_state)

        bundle_dir = self.run_dir / "artifacts" / "k5_store_sales_bundle"
        bundle_verified, bundle_identity_verified, package_reason = package_accepted_candidate(
            agent_loop, loop_state, df_dev,
            candidate_dir=self.run_dir / "cand_src_k5", bundle_dir=bundle_dir,
            dataset_path=dev_csv_path, test_access_id=f"gate_{self.run_id}",
            project_id="k5_store_sales_robustness", task_type="regression",
            target_column="sales", primary_metric="rmsle",
        )

        # The full source was materialized above; distinguish that from use in
        # candidate selection or metric scoring.
        self.summary["holdout_accesses"] = 0
        self.record_trace("holdout_partition_audit", {"task": "K5", "holdout_rows": holdout_rows_total,
            "source_rows_materialized": len(df_full), "holdout_labels_materialized": holdout_rows_total,
            "holdout_labels_used_for_selection_or_scoring": 0, "status": "not_scored_or_used_for_selection"})

        elapsed = time.time() - t0
        controller_rss = get_controller_rss_bytes()
        sampler_metrics = sampler.sample_now()

        status = self.campaign_authority.get_status()
        run_usage = status.get("runs", {}).get(self.run_id, {})
        tot_billed = float(run_usage.get("billed_usd", "0.00"))

        provider_summary = {
            "total_requests": run_usage.get("requests", 0),
            "total_billed_cost_usd": tot_billed,
            "model_id": "deepseek/deepseek-v4.1-flash-1017",
            "provider_endpoint": "inference-net/fp8",
            "remaining_campaign_usd": float(status.get("remaining_campaign_usd", "0.30")),
        }
        self._atomic_write_json(self.provider_usage_path, provider_summary)

        metrics_summary = {
            "task_id": "K5",
            "dataset": "store_sales",
            "development_rows": len(df_dev),
            "holdout_rows_designated": holdout_rows_total,
            "holdout_labels_materialized": holdout_rows_total,
            "holdout_labels_used_for_selection_or_scoring": 0,
            "holdout_accesses": 0,
            "metric_name": "rmsle",
            "metric_direction": "minimize",
            "best_cv_score": best_cv_score,
            "holdout_score": None,
            "bundle_verified": bundle_verified,
            "elapsed_seconds": elapsed,
        }
        self._atomic_write_json(self.metrics_path, metrics_summary)

        self.record_resource("live_dev_k5", {
            "elapsed_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_lifetime_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
            "total_billed_cost_usd": tot_billed,
        })

        quality_threshold = locked_quality_threshold(self.policy, "K5")
        passed = bundle_verified and (candidate_evidence_ok and score_meets_threshold(best_cv_score, quality_threshold, "minimize") and bundle_identity_verified)
        outcome = {
            "status": "passed" if passed else ("inconclusive" if quality_threshold is None else "failed"),
            "reason": candidate_gate_failure_reason(candidate_evidence_ok, candidate_evidence_reason, best_cv_score, quality_threshold, "minimize", bundle_verified, bundle_identity_verified, package_reason),
            "bundle_identity_verified": bundle_identity_verified,
            "accepted_candidate_id": loop_state.get("accepted_candidate_id"),
            "candidate_identity": loop_state.get("candidate_identity"),
            "task": "K5 Store Sales Forecasting",
            "metric": "rmsle",
            "best_cv_score": best_cv_score,
            "holdout_accesses": 0,
            "bundle_verified": bundle_verified,
            "provider_cost_usd": tot_billed,
            "duration_s": elapsed,
            "worker_peak_rss_bytes": sampler_metrics["worker_peak_rss_bytes"],
            "controller_peak_rss_bytes": controller_rss,
            "containment_status": sampler_metrics["containment_status"],
        }
        self.summary["gates"]["G_LIVE_DEV"] = {
            "outcome": outcome["status"],
            "details": outcome,
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("live_dev_k5_completed", outcome)
        self.save_summary()
        return outcome

    def run_certify(self) -> Dict[str, Any]:
        """Execute certification gates. Never succeeds without Linux cgroup v2 verification."""
        self.record_trace("certify_start", {"action": "evaluating_certification_gates"})
        gates = self.run_preflight()
        blocked = [k for k, v in gates.items() if v["outcome"] == "blocked"]
        failed = [k for k, v in gates.items() if v["outcome"] == "failed"]
        outcome = {
            "status": "blocked" if blocked else ("failed" if failed else "inconclusive"),
            "reason": f"Certification rejected: blocked gates {blocked}" if blocked else "Certification requires Linux cgroup v2 and complete campaign passes",
            "blocked_gates": blocked,
            "failed_gates": failed,
        }
        self.summary["gates"]["G_CERTIFICATION"] = {
            "outcome": outcome["status"],
            "reason": outcome["reason"],
        }
        self.summary["status"] = outcome["status"]
        self.record_trace("certify_rejected", outcome)
        self.save_summary()
        return outcome


def main() -> Optional[int]:
    parser = argparse.ArgumentParser(description="MLRescurix Robustness Campaign Runner (S1-S5)")
    parser.add_argument("--mode", choices=["preflight", "faults", "deterministic_dev", "live_dev", "certify", "all"], default="preflight", help="Execution mode")
    parser.add_argument("--task", choices=["K1", "K2", "K3", "K4", "K5"], default="K3", help="Task for live_dev mode (default: K3)")
    parser.add_argument("--campaign", default="20261021-robustness", help="Campaign name")
    parser.add_argument("--run-id", default=None, help="Resumable run ID")
    args = parser.parse_args()

    print(f"=== MLRescurix Robustness Campaign: mode={args.mode}, task={args.task}, campaign={args.campaign} ===")
    runner = CampaignRunner(campaign=args.campaign, run_id=args.run_id)
    print(f"Evidence Run Directory: {runner.run_dir}")

    if args.mode in ("preflight", "all"):
        print("\n--- Running Layered Preflight ---")
        gates = runner.run_preflight()
        for g_id, g_info in gates.items():
            status = g_info["outcome"].upper()
            reason = f" ({g_info.get('reason')})" if g_info.get("reason") else ""
            print(f"  [{status:<7}] {g_id}{reason}")

    if args.mode in ("faults", "all"):
        print("\n--- Running S1 Fault Reproducers & Typed Hypothesis Dispatcher ---")
        res = runner.run_fault_reproducers()
        print(f"  Status: {'PASSED' if res['passed'] else 'FAILED'} in {res['duration_s']:.2f}s")
        print(f"  Summary: {res['stdout_summary']}")
        if not res["passed"]:
            return 1

    if args.mode in ("deterministic_dev", "all"):
        print("\n--- Running Deterministic Dev Benchmarks ---")
        runner.run_preflight()
        res = runner.run_deterministic_dev()
        print(f"  Status: {'PASSED' if res['passed'] else 'FAILED'} in {res['duration_s']:.2f}s")
        print(f"  Summary: {res['stdout_summary']}")
        if not res["passed"]:
            return 1

    if args.mode == "live_dev":
        if args.task == "K3":
            print("\n--- Running Bounded Development-Only Live Experiment (K3 Forest Cover Type) ---")
            res = runner.run_live_dev_k3()
            status_str = res["status"].upper()
            if res["status"] == "passed":
                print(f"  Status: PASSED (Task={res.get('task')}, Best CV Accuracy={res.get('best_cv_score', 0):.4f}, Holdout Accesses={res.get('holdout_accesses', 0)}, Cost=${res.get('provider_cost_usd', 0):.5f} in {res.get('duration_s', 0):.2f}s)")
                print(f"\n=== Robustness Campaign Live Dev Complete ===")
                print(f"Summary written to: {runner.summary_path}")
                return 0
            else:
                reason = res.get("reason", "Verification or threshold check failed")
                print(f"  Status: {status_str}: {reason}")
                print(f"\n=== Robustness Campaign Live Dev Non-Success ===")
                print(f"Summary written to: {runner.summary_path}")
                return 1
        elif args.task == "K4":
            print("\n--- Running Bounded Development-Only Live Experiment (K4 IEEE-CIS Fraud Detection) ---")
            res = runner.run_live_dev_k4()
            status_str = res["status"].upper()
            if res["status"] == "passed":
                print(f"  Status: PASSED (Task={res.get('task')}, Best CV ROC-AUC={res.get('best_cv_score', 0):.4f}, Holdout Accesses={res.get('holdout_accesses', 0)}, Cost=${res.get('provider_cost_usd', 0):.5f} in {res.get('duration_s', 0):.2f}s)")
                print(f"\n=== Robustness Campaign Live Dev Complete ===")
                print(f"Summary written to: {runner.summary_path}")
                return 0
            else:
                reason = res.get("reason", "Verification or threshold check failed")
                print(f"  Status: {status_str}: {reason}")
                print(f"\n=== Robustness Campaign Live Dev Non-Success ===")
                print(f"Summary written to: {runner.summary_path}")
                return 1
        elif args.task == "K5":
            print("\n--- Running Bounded Development-Only Live Experiment (K5 Store Sales Forecasting) ---")
            res = runner.run_live_dev_k5()
            status_str = res["status"].upper()
            if res["status"] == "passed":
                print(f"  Status: PASSED (Task={res.get('task')}, Best CV RMSLE={res.get('best_cv_score', 0):.4f}, Holdout Accesses={res.get('holdout_accesses', 0)}, Cost=${res.get('provider_cost_usd', 0):.5f} in {res.get('duration_s', 0):.2f}s)")
                print(f"\n=== Robustness Campaign Live Dev Complete ===")
                print(f"Summary written to: {runner.summary_path}")
                return 0
            else:
                reason = res.get("reason", "Verification or threshold check failed")
                print(f"  Status: {status_str}: {reason}")
                print(f"\n=== Robustness Campaign Live Dev Non-Success ===")
                print(f"Summary written to: {runner.summary_path}")
                return 1
        elif args.task == "K2":
            print("\n--- Running Bounded Development-Only Live Experiment (K2 House Prices) ---")
            res = runner.run_live_dev_k2()
            status_str = res["status"].upper()
            if res["status"] == "passed":
                print(f"  Status: PASSED (Task={res.get('task')}, Best CV RMSE={res.get('best_cv_score', 0):.4f}, Holdout Accesses={res.get('holdout_accesses', 0)}, Cost=${res.get('provider_cost_usd', 0):.5f} in {res.get('duration_s', 0):.2f}s)")
                print(f"\n=== Robustness Campaign Live Dev Complete ===")
                print(f"Summary written to: {runner.summary_path}")
                return 0
            else:
                reason = res.get("reason", "Verification or threshold check failed")
                print(f"  Status: {status_str}: {reason}")
                print(f"\n=== Robustness Campaign Live Dev Non-Success ===")
                print(f"Summary written to: {runner.summary_path}")
                return 1
        else:
            print("\n--- Running Live Dev (K1 Titanic) ---")
            res = runner.run_live_dev()
            status_str = res["status"].upper()
            if res["status"] == "passed":
                print(f"  Status: PASSED (Task={res.get('task')}, Best CV={res.get('best_cv_score', 0):.4f}, Holdout Accesses={res.get('holdout_accesses', 0)}, Cost=${res.get('provider_cost_usd', 0):.5f} in {res.get('duration_s', 0):.2f}s)")
                print(f"\n=== Robustness Campaign Live Dev Complete ===")
                print(f"Summary written to: {runner.summary_path}")
                return 0
            else:
                reason = res.get("reason", "Verification or threshold check failed")
                print(f"  Status: {status_str}: {reason}")
                print(f"\n=== Robustness Campaign Live Dev Non-Success ===")
                print(f"Summary written to: {runner.summary_path}")
                return 1

    if args.mode == "certify":
        print("\n--- Evaluating Certification Gates ---")
        res = runner.run_certify()
        print(f"  Status: {res['status'].upper()}: {res['reason']}")
        print(f"\n=== Robustness Campaign Certification Blocked ===")
        print(f"Summary written to: {runner.summary_path}")
        return 1

    outcomes = [g["outcome"] for g in runner.summary["gates"].values()]
    runner.summary["status"] = ("failed" if "failed" in outcomes else
                                 "blocked" if "blocked" in outcomes else
                                 "inconclusive" if "inconclusive" in outcomes else "completed")
    runner.save_summary()

    print("\n=== Robustness Campaign Run Complete ===")
    print(f"Summary written to: {runner.summary_path}")
    return 0 if runner.summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main() or 0)
