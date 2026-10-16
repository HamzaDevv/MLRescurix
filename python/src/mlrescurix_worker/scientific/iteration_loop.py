"""Multi-Fidelity Iteration & Hypothesis Loop for Phase P10.3.

Provides:
- Progressive evaluation tiers (Tier 0 fast proxy screen N <= 500, Tier 1 full k-fold CV).
- Early discard of non-viable mutations to conserve compute budget.
- Hypothesis verification & settlement (confirmed, refuted, inconclusive).
- Multi-objective Pareto frontier tracking (validation score vs latency ms vs peak memory bytes).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from .workflow import WorkflowError
from .candidate_runner import (
    CandidateExecutionError,
    evaluate_candidate_cv,
    execute_candidate_cli,
    validate_candidate_manifest,
    validate_predictions,
)
from .diagnostic_bridge import DiagnosticBridge, DiagnosticReceipt


def get_peak_rss_bytes() -> int:
    """Return peak resident memory in bytes portably across macOS and Linux."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(usage)
    return int(usage * 1024)


@dataclass
class ProxyScreenReceipt:
    candidate_id: str
    status: str  # "admitted", "rejected", "failed"
    proxy_metric_name: str
    proxy_metric_value: float | None
    latency_ms: float
    peak_memory_bytes: int
    screening_threshold: float | None
    reason: str
    samples_used: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MultiFidelityIterationEngine:
    """Multi-fidelity iteration, progressive screening, hypothesis settlement, and Pareto tracking."""

    @staticmethod
    def run_proxy_screen(
        dataset_csv: Path | str,
        manifest: dict[str, Any],
        candidate_dir: Path | str,
        max_samples: int = 500,
        screening_threshold: float | None = None,
        direction: str = "maximize",
        seed: int = 2026,
        timeout_s: int = 30,
    ) -> ProxyScreenReceipt:
        """Execute Tier 0 fast subsampled proxy screening (N <= max_samples).

        Discards non-viable candidates early before spending full k-fold CV budget.
        """
        c_dir = Path(candidate_dir)
        validate_candidate_manifest(manifest, c_dir)
        cid = manifest["candidate_id"]
        task_type = manifest["task_type"]
        entrypoint = manifest["entrypoint"]

        df = pd.read_csv(dataset_csv)
        if len(df) == 0:
            raise WorkflowError("dataset is empty")

        target_col = "target"
        if target_col not in df.columns:
            target_candidates = [c for c in df.columns if c not in ("row_id", "id", "timestamp")]
            target_col = target_candidates[-1] if target_candidates else df.columns[-1]

        # Subsample if N > max_samples
        n_samples = min(len(df), max(10, max_samples))
        if len(df) > n_samples:
            if task_type in ("binary", "multiclass") and df[target_col].nunique() > 1:
                try:
                    df_sample, _ = train_test_split(
                        df, train_size=n_samples, stratify=df[target_col], random_state=seed
                    )
                except Exception:
                    df_sample = df.sample(n=n_samples, random_state=seed)
            else:
                df_sample = df.sample(n=n_samples, random_state=seed)
        else:
            df_sample = df.copy()

        # Split 80/20 train/val for proxy screen
        if task_type in ("binary", "multiclass") and df_sample[target_col].nunique() > 1:
            try:
                train_df, val_df = train_test_split(
                    df_sample, test_size=0.25, stratify=df_sample[target_col], random_state=seed
                )
            except Exception:
                train_df, val_df = train_test_split(df_sample, test_size=0.25, random_state=seed)
        else:
            train_df, val_df = train_test_split(df_sample, test_size=0.25, random_state=seed)

        scratch = Path(tempfile.mkdtemp(prefix=f"m2_proxy_{cid}_"))
        start_time = time.perf_counter()
        mem_start = get_peak_rss_bytes()

        try:
            model_dir = scratch / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            train_csv = scratch / "train.csv"
            val_features_csv = scratch / "val_features.csv"
            predictions_csv = scratch / "predictions.csv"

            train_df.to_csv(train_csv, index=False)
            # Label isolation: validation features MUST NOT contain the target!
            val_features_df = val_df.drop(columns=[target_col]).copy()
            val_features_df.to_csv(val_features_csv, index=False)

            # 1. Fit proxy model
            fit_args = [
                "--train-path", str(train_csv.resolve()),
                "--model-dir", str(model_dir.resolve()),
            ]
            code, stdout, stderr = execute_candidate_cli(
                c_dir, entrypoint, "fit", fit_args, timeout_s=timeout_s
            )
            if code != 0:
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=code,
                    stderr=stderr,
                    stdout=stdout,
                    message=f"Candidate '{cid}' proxy fit failed",
                    candidate_id=cid,
                    stage="fit",
                    candidate_dir=str(c_dir),
                )
                return ProxyScreenReceipt(
                    candidate_id=cid,
                    status="failed",
                    proxy_metric_name="rmse" if task_type == "regression" else "balanced_accuracy",
                    proxy_metric_value=None,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    peak_memory_bytes=get_peak_rss_bytes(),
                    screening_threshold=screening_threshold,
                    reason=f"Proxy fit failed (exit code {code}): {stderr.strip()}",
                    samples_used=len(df_sample),
                )

            # 2. Predict on validation features
            predict_args = [
                "--input-path", str(val_features_csv.resolve()),
                "--model-dir", str(model_dir.resolve()),
                "--output-path", str(predictions_csv.resolve()),
            ]
            code, stdout, stderr = execute_candidate_cli(
                c_dir, entrypoint, "predict", predict_args, timeout_s=timeout_s
            )
            if code != 0 or not predictions_csv.exists():
                diag = DiagnosticBridge.diagnose_failure(
                    exit_code=code if code != 0 else 1,
                    stderr=stderr,
                    stdout=stdout,
                    message=f"Candidate '{cid}' proxy predict failed",
                    candidate_id=cid,
                    stage="predict",
                    candidate_dir=str(c_dir),
                )
                return ProxyScreenReceipt(
                    candidate_id=cid,
                    status="failed",
                    proxy_metric_name="rmse" if task_type == "regression" else "balanced_accuracy",
                    proxy_metric_value=None,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    peak_memory_bytes=get_peak_rss_bytes(),
                    screening_threshold=screening_threshold,
                    reason=f"Proxy predict failed (exit code {code}): {stderr.strip()}",
                    samples_used=len(df_sample),
                )

            # 3. Compute metric score
            row_key = manifest.get("stable_row_keys", "row_id")
            pred_df = pd.read_csv(predictions_csv, low_memory=False)
            val_row_ids = val_df[row_key].tolist()
            try:
                preds_payload = validate_predictions(pred_df, val_row_ids, manifest)
            except WorkflowError as exc:
                return ProxyScreenReceipt(
                    candidate_id=cid,
                    status="failed",
                    proxy_metric_name="rmse" if task_type == "regression" else "balanced_accuracy",
                    proxy_metric_value=None,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                    peak_memory_bytes=get_peak_rss_bytes(),
                    screening_threshold=screening_threshold,
                    reason=f"Proxy validation failed: {exc}",
                    samples_used=len(df_sample),
                )

            y_true = val_df[target_col].to_numpy()
            y_pred = np.array(preds_payload["prediction"])

            if task_type == "regression":
                metric_name = "rmse"
                metric_val = float(math.sqrt(mean_squared_error(y_true, y_pred)))
            else:
                metric_name = "balanced_accuracy"
                metric_val = float(balanced_accuracy_score(y_true, y_pred))

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            peak_mem = get_peak_rss_bytes()

            # 4. Threshold check
            if screening_threshold is not None:
                if direction == "maximize" and metric_val < screening_threshold:
                    return ProxyScreenReceipt(
                        candidate_id=cid,
                        status="rejected",
                        proxy_metric_name=metric_name,
                        proxy_metric_value=metric_val,
                        latency_ms=latency_ms,
                        peak_memory_bytes=peak_mem,
                        screening_threshold=screening_threshold,
                        reason=f"Proxy score {metric_val:.4f} failed screening threshold {screening_threshold:.4f} (direction: {direction})",
                        samples_used=len(df_sample),
                    )
                elif direction == "minimize" and metric_val > screening_threshold:
                    return ProxyScreenReceipt(
                        candidate_id=cid,
                        status="rejected",
                        proxy_metric_name=metric_name,
                        proxy_metric_value=metric_val,
                        latency_ms=latency_ms,
                        peak_memory_bytes=peak_mem,
                        screening_threshold=screening_threshold,
                        reason=f"Proxy score {metric_val:.4f} exceeded screening threshold {screening_threshold:.4f} (direction: {direction})",
                        samples_used=len(df_sample),
                    )

            return ProxyScreenReceipt(
                candidate_id=cid,
                status="admitted",
                proxy_metric_name=metric_name,
                proxy_metric_value=metric_val,
                latency_ms=latency_ms,
                peak_memory_bytes=peak_mem,
                screening_threshold=screening_threshold,
                reason=f"Admitted via proxy screening: score {metric_val:.4f}",
                samples_used=len(df_sample),
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    @staticmethod
    def settle_hypothesis(
        hypothesis_spec: dict[str, Any],
        observed_metric: float | None,
        baseline_score: float | None,
        screening_rejected: bool = False,
        rejection_reason: str | None = None,
    ) -> dict[str, Any]:
        """Settle an iteration hypothesis as confirmed, refuted, or inconclusive."""
        now_ms = int(time.time() * 1000)
        exp_delta = hypothesis_spec.get("expected_metric_delta", {})
        metric_name = exp_delta.get("metric", "primary_metric")
        direction = exp_delta.get("direction", "maximize")
        target_threshold = exp_delta.get("target_threshold")
        min_improvement = float(exp_delta.get("minimum_improvement", 0.005))

        if screening_rejected:
            outcome = {
                "status": "refuted",
                "baseline_metric": baseline_score,
                "proxy_metric_observed": observed_metric,
                "full_metric_observed": None,
                "metric_delta": (observed_metric - baseline_score) if (observed_metric is not None and baseline_score is not None) else None,
                "reason": rejection_reason or "Rejected during proxy screening; mutation not viable.",
                "settled_at_ms": now_ms,
            }
            res = dict(hypothesis_spec)
            res["outcome"] = outcome
            return res

        if observed_metric is None:
            outcome = {
                "status": "refuted",
                "baseline_metric": baseline_score,
                "proxy_metric_observed": None,
                "full_metric_observed": None,
                "metric_delta": None,
                "reason": rejection_reason or "Candidate evaluation produced no valid metric score.",
                "settled_at_ms": now_ms,
            }
            res = dict(hypothesis_spec)
            res["outcome"] = outcome
            return res

        delta = (observed_metric - baseline_score) if baseline_score is not None else 0.0

        if direction == "maximize":
            improvement = delta
            is_better = improvement >= min_improvement
            is_worse = improvement < 0.0
            meets_target = (target_threshold is None) or (observed_metric >= target_threshold)

            if baseline_score is None:
                status = "confirmed" if meets_target else "refuted"
                reason = f"Observed score {observed_metric:.4f} without baseline (target threshold: {target_threshold})"
            elif is_better and meets_target:
                status = "confirmed"
                reason = f"Confirmed: improvement +{improvement:.4f} >= {min_improvement:.4f} and observed {observed_metric:.4f} meets target."
            elif is_worse:
                status = "refuted"
                reason = f"Refuted: performance degraded ({delta:.4f}) relative to baseline {baseline_score:.4f}."
            elif not meets_target:
                status = "inconclusive"
                reason = f"Inconclusive: improved by +{improvement:.4f} but did not reach target threshold {target_threshold}."
            else:
                status = "inconclusive"
                reason = f"Inconclusive: improvement +{improvement:.4f} was below required threshold {min_improvement:.4f}."
        else:  # minimize
            improvement = -delta  # reduction is positive improvement
            is_better = improvement >= min_improvement
            is_worse = improvement < 0.0
            meets_target = (target_threshold is None) or (observed_metric <= target_threshold)

            if baseline_score is None:
                status = "confirmed" if meets_target else "refuted"
                reason = f"Observed metric {observed_metric:.4f} without baseline (target threshold: {target_threshold})"
            elif is_better and meets_target:
                status = "confirmed"
                reason = f"Confirmed: metric reduced by {improvement:.4f} >= {min_improvement:.4f} and reached target."
            elif is_worse:
                status = "refuted"
                reason = f"Refuted: metric increased by {delta:.4f} relative to baseline {baseline_score:.4f}."
            elif not meets_target:
                status = "inconclusive"
                reason = f"Inconclusive: reduced by {improvement:.4f} but did not achieve target threshold {target_threshold}."
            else:
                status = "inconclusive"
                reason = f"Inconclusive: improvement {improvement:.4f} was below required threshold {min_improvement:.4f}."

        outcome = {
            "status": status,
            "baseline_metric": baseline_score,
            "proxy_metric_observed": None,
            "full_metric_observed": observed_metric,
            "metric_delta": delta,
            "reason": reason,
            "settled_at_ms": now_ms,
        }
        res = dict(hypothesis_spec)
        res["outcome"] = outcome
        return res

    @staticmethod
    def run_multi_fidelity_iteration(
        dataset_csv: Path | str,
        manifest: dict[str, Any],
        candidate_dir: Path | str,
        hypothesis_spec: dict[str, Any] | None = None,
        baseline_score: float | None = None,
        skip_proxy: bool = False,
        folds: int = 3,
        seed: int = 2026,
    ) -> dict[str, Any]:
        """Progressive multi-fidelity evaluation: Tier 0 fast screen, Tier 1 full CV, and settlement."""
        cid = manifest["candidate_id"]
        exp_delta = hypothesis_spec.get("expected_metric_delta", {}) if hypothesis_spec else {}
        direction = exp_delta.get("direction", "maximize")

        budget = hypothesis_spec.get("multi_fidelity_budget", {}) if hypothesis_spec else {}
        proxy_budget = budget.get("proxy_tier", {})
        full_budget = budget.get("full_tier", {})

        max_samples = proxy_budget.get("max_samples", 500)
        screening_threshold = proxy_budget.get("screening_threshold")
        cv_folds = full_budget.get("folds", folds)
        cv_seed = full_budget.get("seed", seed)

        proxy_receipt: ProxyScreenReceipt | None = None
        if not skip_proxy:
            proxy_receipt = MultiFidelityIterationEngine.run_proxy_screen(
                dataset_csv=dataset_csv,
                manifest=manifest,
                candidate_dir=candidate_dir,
                max_samples=max_samples,
                screening_threshold=screening_threshold,
                direction=direction,
                seed=seed,
            )

            # If rejected or failed, early terminate without running full CV
            if proxy_receipt.status != "admitted":
                settled = None
                if hypothesis_spec:
                    settled = MultiFidelityIterationEngine.settle_hypothesis(
                        hypothesis_spec=hypothesis_spec,
                        observed_metric=proxy_receipt.proxy_metric_value,
                        baseline_score=baseline_score,
                        screening_rejected=True,
                        rejection_reason=proxy_receipt.reason,
                    )
                return {
                    "status": "rejected_early",
                    "candidate_id": cid,
                    "tier_reached": 0,
                    "proxy_screening": proxy_receipt.to_dict(),
                    "full_evaluation": None,
                    "hypothesis": settled,
                }

        # Admitted or skipped proxy -> Proceed to Tier 1: Full CV
        start_full = time.perf_counter()
        cv_result = evaluate_candidate_cv(
            dataset_csv=dataset_csv,
            manifest=manifest,
            candidate_dir=candidate_dir,
            folds=cv_folds,
            seed=cv_seed,
        )
        full_latency_ms = (time.perf_counter() - start_full) * 1000.0
        full_mem_bytes = get_peak_rss_bytes()

        cv_result["latency_ms"] = full_latency_ms
        cv_result["peak_memory_bytes"] = full_mem_bytes

        primary_metric_val = cv_result.get("primary_metric")
        settled = None
        if hypothesis_spec:
            settled = MultiFidelityIterationEngine.settle_hypothesis(
                hypothesis_spec=hypothesis_spec,
                observed_metric=primary_metric_val,
                baseline_score=baseline_score,
                screening_rejected=False,
            )

        return {
            "status": "completed",
            "candidate_id": cid,
            "tier_reached": 1,
            "proxy_screening": proxy_receipt.to_dict() if proxy_receipt else None,
            "full_evaluation": cv_result,
            "hypothesis": settled,
        }

    @staticmethod
    def compute_pareto_frontier(
        records: list[dict[str, Any]],
        primary_metric_name: str,
        direction: str = "maximize",
        project_id: str = "default",
        frontier_id: str | None = None,
    ) -> dict[str, Any]:
        """Compute non-dominated Pareto frontier across (score, latency, memory)."""
        now_ms = int(time.time() * 1000)
        f_id = frontier_id or f"frontier-{project_id}-{now_ms}"

        processed_records: list[dict[str, Any]] = []
        for r in records:
            processed_records.append({
                "candidate_id": str(r["candidate_id"]),
                "parent_candidate_id": r.get("parent_candidate_id"),
                "validation_score": float(r["validation_score"]),
                "latency_ms": float(r.get("latency_ms", 0.0)),
                "peak_memory_bytes": int(r.get("peak_memory_bytes", 0)),
                "domination_status": "non_dominated",
                "dominated_by": [],
                "metadata": r.get("metadata", {}),
            })

        n = len(processed_records)
        for i in range(n):
            cand_a = processed_records[i]
            for j in range(n):
                if i == j:
                    continue
                cand_b = processed_records[j]
                # Check if cand_b dominates cand_a
                # B dominates A iff:
                # 1. B is no worse in all 3 objectives
                # 2. B is strictly better in at least 1 objective
                if direction == "maximize":
                    score_better_or_equal = cand_b["validation_score"] >= cand_a["validation_score"]
                    score_strictly_better = cand_b["validation_score"] > cand_a["validation_score"]
                else:
                    score_better_or_equal = cand_b["validation_score"] <= cand_a["validation_score"]
                    score_strictly_better = cand_b["validation_score"] < cand_a["validation_score"]

                lat_better_or_equal = cand_b["latency_ms"] <= cand_a["latency_ms"]
                lat_strictly_better = cand_b["latency_ms"] < cand_a["latency_ms"]

                mem_better_or_equal = cand_b["peak_memory_bytes"] <= cand_a["peak_memory_bytes"]
                mem_strictly_better = cand_b["peak_memory_bytes"] < cand_a["peak_memory_bytes"]

                if (
                    score_better_or_equal
                    and lat_better_or_equal
                    and mem_better_or_equal
                    and (score_strictly_better or lat_strictly_better or mem_strictly_better)
                ):
                    cand_a["dominated_by"].append(cand_b["candidate_id"])
                    cand_a["domination_status"] = "dominated"

        frontier_ids = [r["candidate_id"] for r in processed_records if r["domination_status"] == "non_dominated"]
        non_dom_count = len(frontier_ids)
        dom_count = n - non_dom_count

        return {
            "schema_version": "v2",
            "frontier_id": f_id,
            "project_id": project_id,
            "primary_metric_name": primary_metric_name,
            "optimization_direction": direction,
            "records": processed_records,
            "frontier_candidate_ids": frontier_ids,
            "non_dominated_count": non_dom_count,
            "dominated_count": dom_count,
            "evaluated_at_ms": now_ms,
        }
