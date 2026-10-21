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


def sample_topology_aware_tier0(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "binary",
    topology: Optional[dict[str, Any]] = None,
    max_samples: int = 500,
    random_state: int = 2026,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Topology-Preserving Tier-0 Proxy Sampler (N <= 500, <= 15s).

    Adheres to blueprint section 6.2:
    1. Temporal Tail Slicing: For time-series datasets, take the most recent contiguous rows.
    2. Group-Preserving Sampling: Sample whole `entity_group_col` clusters rather than individual rows.
    3. Rare-Event Floor (Minimum-Minority Stratified Subsampling):
       If minority class prevalence p_min yields < 30 minority samples in 500 rows,
       retain up to min(N_minority, 100) minority samples, sample 400 majority samples,
       and compute sample_weight / prior-adjustment offset so probabilities remain calibrated.
    """
    n_total = len(df)
    if n_total <= max_samples:
        weights = np.ones(n_total, dtype=float)
        df_out = df.copy()
        df_out["__sample_weight__"] = weights
        return df_out, weights, {"strategy": "full_dataset", "samples": n_total}

    # 1. Temporal Tail Slicing
    temporal_col = None
    if topology is not None:
        temporal_col = topology.get("temporal_col")
        cv_scheme = topology.get("cv_scheme")
        if not temporal_col and cv_scheme == "purged_time_series_split":
            classifs = topology.get("column_classifications", {})
            for c, role in classifs.items():
                if role == "TEMPORAL_AXIS" and c in df.columns:
                    temporal_col = c
                    break

    if temporal_col and temporal_col in df.columns:
        df_sorted = df.sort_values(by=temporal_col).reset_index(drop=True)
        n_take = min(n_total, max_samples)
        df_sample = df_sorted.tail(n_take).copy().reset_index(drop=True)
        weights = np.ones(len(df_sample), dtype=float)
        df_sample["__sample_weight__"] = weights
        return df_sample, weights, {
            "strategy": "temporal_tail_slice",
            "temporal_col": temporal_col,
            "samples": len(df_sample),
        }

    # 2. Group-Preserving Sampling
    group_col = None
    if topology is not None:
        group_col = topology.get("entity_group_col")

    if group_col and group_col in df.columns and df[group_col].nunique() > 1:
        rng = np.random.RandomState(random_state)
        unique_groups = df[group_col].dropna().unique()
        rng.shuffle(unique_groups)
        group_counts = df[group_col].value_counts().to_dict()

        selected_groups = []
        total_rows = 0
        for g in unique_groups:
            cnt = group_counts[g]
            if total_rows + cnt <= max_samples or len(selected_groups) == 0:
                selected_groups.append(g)
                total_rows += cnt
            if total_rows >= max_samples:
                break

        df_sample = df[df[group_col].isin(selected_groups)].copy().reset_index(drop=True)
        weights = np.ones(len(df_sample), dtype=float)
        df_sample["__sample_weight__"] = weights
        return df_sample, weights, {
            "strategy": "group_preserving",
            "group_col": group_col,
            "groups_count": len(selected_groups),
            "samples": len(df_sample),
        }

    # 3. Rare-Event Floor (Minimum-Minority Stratified Subsampling)
    is_classification = "classification" in task_type or task_type in ("binary", "multiclass")
    if is_classification and target_col in df.columns and df[target_col].nunique() > 1:
        class_counts = df[target_col].value_counts().to_dict()
        min_class = min(class_counts, key=class_counts.get)
        n_minority = class_counts[min_class]
        p_min = n_minority / n_total
        expected_minority = max_samples * p_min

        if expected_minority < 30:
            # Rare-Event Floor Activated!
            # Retain up to min(N_minority, 100) minority samples
            k_minority = min(n_minority, 100)
            minority_indices = df[df[target_col] == min_class].index.to_numpy()
            majority_indices = df[df[target_col] != min_class].index.to_numpy()

            k_majority = min(len(majority_indices), 400)

            rng = np.random.RandomState(random_state)
            sampled_minority_idx = rng.choice(minority_indices, size=k_minority, replace=False)
            sampled_majority_idx = rng.choice(majority_indices, size=k_majority, replace=False)

            combined_idx = np.concatenate([sampled_minority_idx, sampled_majority_idx])
            rng.shuffle(combined_idx)

            df_sample = df.loc[combined_idx].copy().reset_index(drop=True)
            n_sampled = len(df_sample)

            # Prior-adjusted sample weights for calibrated probabilities
            p_pos = n_minority / n_total
            p_neg = (n_total - n_minority) / n_total
            q_pos = k_minority / n_sampled
            q_neg = k_majority / n_sampled

            w_pos = p_pos / q_pos if q_pos > 0 else 1.0
            w_neg = p_neg / q_neg if q_neg > 0 else 1.0

            weights = np.where(df_sample[target_col] == min_class, w_pos, w_neg).astype(float)
            df_sample["__sample_weight__"] = weights

            return df_sample, weights, {
                "strategy": "rare_event_floor",
                "minority_class": min_class,
                "minority_sampled": k_minority,
                "majority_sampled": k_majority,
                "expected_minority": expected_minority,
                "p_min": p_min,
                "samples": n_sampled,
            }
        else:
            # Standard stratified subsampling
            try:
                df_sample, _ = train_test_split(
                    df, train_size=max_samples, stratify=df[target_col], random_state=random_state
                )
                df_sample = df_sample.reset_index(drop=True)
                weights = np.ones(len(df_sample), dtype=float)
                df_sample["__sample_weight__"] = weights
                return df_sample, weights, {
                    "strategy": "standard_stratified",
                    "samples": len(df_sample),
                }
            except Exception:
                pass

    # 4. Standard uniform random subsampling fallback
    df_sample = df.sample(n=min(n_total, max_samples), random_state=random_state).reset_index(drop=True)
    weights = np.ones(len(df_sample), dtype=float)
    df_sample["__sample_weight__"] = weights
    return df_sample, weights, {
        "strategy": "random_subsample",
        "samples": len(df_sample),
    }


class MultiFidelityIterationEngine:
    """Multi-fidelity iteration, progressive screening, hypothesis settlement, and Pareto tracking."""

    sample_tier0 = staticmethod(sample_topology_aware_tier0)

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
        topology: dict[str, Any] | None = None,
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

        target_col = manifest.get("raw_input_schema", {}).get("target") or manifest.get("target_column") or "target"
        if target_col not in df.columns:
            from .semantic_sieve import resolve_target_column
            try:
                target_col = resolve_target_column(df)
            except Exception:
                target_candidates = [c for c in df.columns if c not in ("row_id", "id", "timestamp")]
                target_col = target_candidates[-1] if target_candidates else df.columns[-1]

        # Topology-aware Tier-0 subsampling
        df_sample, sample_weights, sample_meta = sample_topology_aware_tier0(
            df=df,
            target_col=target_col,
            task_type=task_type,
            topology=topology,
            max_samples=max_samples,
            random_state=seed,
        )

        # Topology-aware splitting for proxy screen (RB-003 fix)
        temporal_col = None
        group_col = None
        if topology is not None:
            temporal_col = topology.get("temporal_col")
            group_col = topology.get("entity_group_col")
            cv_scheme = topology.get("cv_scheme")
            if not temporal_col and cv_scheme == "purged_time_series_split":
                for c, role in topology.get("column_classifications", {}).items():
                    if role == "TEMPORAL_AXIS" and c in df_sample.columns:
                        temporal_col = c
                        break

        if not temporal_col and sample_meta.get("temporal_col"):
            temporal_col = sample_meta.get("temporal_col")
        if not group_col and sample_meta.get("group_col"):
            group_col = sample_meta.get("group_col")

        if temporal_col and temporal_col in df_sample.columns:
            # Strictly chronological split: train on past, validate on future by discrete timestamp cutoffs
            df_sorted = df_sample.sort_values(by=temporal_col).reset_index(drop=True)
            unique_times = np.sort(df_sorted[temporal_col].unique())
            if len(unique_times) < 2:
                raise WorkflowError(f"Temporal proxy split impossible: temporal column '{temporal_col}' has fewer than 2 distinct timestamps in sample")
            n_val_times = max(1, int(len(unique_times) * 0.25))
            cutoff_time = unique_times[-n_val_times]
            train_df = df_sorted[df_sorted[temporal_col] < cutoff_time].copy().reset_index(drop=True)
            val_df = df_sorted[df_sorted[temporal_col] >= cutoff_time].copy().reset_index(drop=True)
            if len(train_df) == 0:
                train_df = df_sorted[df_sorted[temporal_col] == unique_times[0]].copy().reset_index(drop=True)
                val_df = df_sorted[df_sorted[temporal_col] > unique_times[0]].copy().reset_index(drop=True)
            if len(train_df) == 0 or len(val_df) == 0:
                raise WorkflowError("Failed to form non-empty chronological train/val partitions for temporal proxy screen")
            assert train_df[temporal_col].max() < val_df[temporal_col].min(), "Temporal lookahead violation in proxy screen"

            if group_col and group_col in df_sample.columns:
                # Also check group coverage on panel
                tr_grps = set(train_df[group_col])
                val_grps = set(val_df[group_col])
                # Report / verify panel entities
        elif group_col and group_col in df_sample.columns:
            # Strictly group-disjoint split: train and val share zero groups
            unique_groups = list(df_sample[group_col].dropna().unique())
            if len(unique_groups) < 2:
                raise WorkflowError(f"Group proxy split impossible: group column '{group_col}' has fewer than 2 distinct groups in sample")
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            try:
                tr_idx, val_idx = next(gss.split(df_sample, groups=df_sample[group_col]))
                train_df = df_sample.iloc[tr_idx].copy().reset_index(drop=True)
                val_df = df_sample.iloc[val_idx].copy().reset_index(drop=True)
                assert set(train_df[group_col]).isdisjoint(set(val_df[group_col])), "Group leakage detected in proxy screen"
            except Exception:
                n_val_groups = max(1, int(len(unique_groups) * 0.25))
                val_groups = set(unique_groups[:n_val_groups])
                train_df = df_sample[~df_sample[group_col].isin(val_groups)].copy().reset_index(drop=True)
                val_df = df_sample[df_sample[group_col].isin(val_groups)].copy().reset_index(drop=True)
                assert set(train_df[group_col]).isdisjoint(set(val_df[group_col])), "Group leakage detected in fallback proxy split"
            if len(train_df) == 0 or len(val_df) == 0:
                raise WorkflowError("Failed to form non-empty group-disjoint partitions for proxy screen")
        elif task_type in ("binary", "multiclass") and df_sample[target_col].nunique() > 1:
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
            val_features_df = val_df.drop(columns=[target_col, "__sample_weight__"], errors="ignore").copy()
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
            if row_key not in val_df.columns:
                if "id" in val_df.columns:
                    row_key = "id"
                else:
                    val_df = val_df.copy()
                    row_key = "row_id"
                    val_df["row_id"] = [f"row_{idx}" for idx in val_df.index]
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
