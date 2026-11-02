"""Frozen fold mappings and trusted candidate/delivery gates shared by all tasks."""
from __future__ import annotations
import hashlib
import json
import math
from decimal import Decimal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
REPO_ROOT = Path(__file__).resolve().parents[4]

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
        failures.append(package_reason or "bundle verification failed")
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
    import sys
    orig_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
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
    finally:
        sys.dont_write_bytecode = orig_dont_write_bytecode


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
            raw_holdout_records=df_dev.drop(columns=[target_column]).head(5).to_dict(orient="records"),
            accepted_evaluation=state.get("accepted_evaluation"),
            accepted_candidate_identity=identity,
            delivery_evidence={"binding": agent_loop.binding, "candidate_id": candidate_id,
                "candidate_identity": identity, "evaluation": state['accepted_evaluation'],
                "approved_audit": state['accepted_audit'], "trace_session_id": agent_loop.session_id,
                "trace_sha256": hashlib.sha256((agent_loop.workspace_root / 'autonomous_trace.jsonl').read_bytes()).hexdigest(),
                "provider_receipts": [{"request_id": turn['request_id'],
                    "request_sha256": turn.get('provider_receipt', {}).get('request_sha256'),
                    "provider_response_id": turn.get('provider_receipt', {}).get('provider_response_id'),
                    "receipt_sha256": candidate_spec_identity(turn.get('provider_receipt', {}))}
                    for turn in state.get('turns', [])],
                "human_commands_applied": state.get('human_commands_applied', [])},
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
    if verify_proc.returncode != 0:
        err_msg = (verify_proc.stderr or verify_proc.stdout or "").strip()
        return False, True, f"bundle verification failed (exit {verify_proc.returncode}): {err_msg}"
    agent_loop.state["bundle_verified"] = True
    agent_loop.state["bundle_dir"] = str(bundle_dir)
    agent_loop._save_state(agent_loop.state)
    return True, True, "accepted candidate bundled and identity verified"
