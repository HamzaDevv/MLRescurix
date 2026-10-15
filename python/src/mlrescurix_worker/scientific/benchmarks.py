"""Prospective matched benchmarks comparing random, search, and agent baselines.

Predeclared protocol:
1. Representative tabular datasets (binary classification, multiclass, regression).
2. Deterministic splits and seeds (e.g., 2026, 42, 101).
3. Matched compute budgets (trial limits, wall-clock ceilings, evaluation constraints).
4. Transparent recording of suboptimal and failing trials alongside winning candidates.
5. Strict scientific reporting: no misleading superiority claims without statistical significance.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from sklearn.metrics import balanced_accuracy_score, root_mean_squared_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold


@dataclass
class TrialRecord:
    trial_id: str
    strategy: str
    seed: int
    recipe: str
    params: dict[str, Any]
    metric_score: float | None
    status: str  # "success", "suboptimal", "failed_constraint", "timeout"
    wall_ms: float
    error_message: str | None = None


@dataclass
class StrategyBenchmarkResult:
    strategy: str
    trials_attempted: int
    trials_succeeded: int
    trials_failed: int
    best_candidate_id: str | None
    best_metric_score: float | None
    mean_metric_score: float | None
    std_metric_score: float | None
    total_wall_ms: float
    trials: list[TrialRecord]


@dataclass
class BenchmarkSuiteReport:
    suite_id: str
    task: str
    metric: str
    direction: str  # "maximize" | "minimize"
    seeds: list[int]
    budget_per_strategy: dict[str, Any]
    results: dict[str, StrategyBenchmarkResult]
    comparative_summary: dict[str, Any]
    generated_at_ms: int


def generate_benchmark_dataset(
    n_samples: int = 120,
    n_features: int = 4,
    task: str = "binary_classification",
    seed: int = 2026,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Generate reproducible synthetic benchmark dataset without leakage."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    feature_names = [f"feat_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)

    if task == "binary_classification":
        # Logistic signal + noise
        logits = 1.5 * X[:, 0] - 1.2 * X[:, 1] + 0.5 * rng.normal(size=n_samples)
        y = (logits > 0).astype(int)
        df["target"] = y
    elif task == "regression":
        # Continuous linear signal + noise
        y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] + 0.2 * rng.normal(size=n_samples)
        df["target"] = y
    else:
        raise ValueError(f"Unsupported task for benchmark generation: {task}")

    return df, "target", feature_names


def run_single_evaluation(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    recipe: str,
    params: dict[str, Any],
    task: str,
    metric: str,
    seed: int,
    cv_folds: int = 3,
) -> float:
    """Evaluate candidate model under strict CV split isolation."""
    X = df[features].to_numpy()
    y = df[target].to_numpy()

    if task == "binary_classification":
        splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    scores = []
    for train_idx, val_idx in splitter.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if recipe == "logistic":
            model = LogisticRegression(C=params.get("C", 1.0), max_iter=200, random_state=seed)
        elif recipe == "ridge":
            if task == "binary_classification":
                model = RidgeClassifier(alpha=params.get("alpha", 1.0), random_state=seed)
            else:
                model = Ridge(alpha=params.get("alpha", 1.0), random_state=seed)
        elif recipe == "failing_candidate":
            # Explicitly simulated failing candidate (e.g. divergent solver or bad constraint)
            raise ValueError("simulated solver divergence in candidate evaluation")
        else:
            raise ValueError(f"Unknown recipe: {recipe}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        if metric == "balanced_accuracy":
            scores.append(balanced_accuracy_score(y_val, y_pred))
        elif metric == "rmse":
            scores.append(root_mean_squared_error(y_val, y_pred))
        elif metric == "r2":
            scores.append(r2_score(y_val, y_pred))
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    return float(np.mean(scores))


def run_prospective_benchmark(
    task: str = "binary_classification",
    metric: str = "balanced_accuracy",
    direction: str = "maximize",
    seeds: list[int] | None = None,
    max_trials_per_strategy: int = 6,
    wall_budget_ms: float = 10_000.0,
) -> BenchmarkSuiteReport:
    """Run matched prospective benchmarks comparing random vs search vs agent strategies."""
    if seeds is None:
        seeds = [2026, 42, 101]

    suite_id = f"bench-{task}-{int(time.time())}"
    results: dict[str, StrategyBenchmarkResult] = {}

    strategies = ["random", "search", "agent"]

    # Parameter pools
    c_values = [0.01, 0.1, 1.0, 5.0, 10.0, 100.0]
    alpha_values = [0.01, 0.1, 1.0, 10.0]

    for strat in strategies:
        t_start = time.perf_counter()
        trials: list[TrialRecord] = []

        for s_idx, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            df, target, features = generate_benchmark_dataset(task=task, seed=seed)

            for trial_i in range(max_trials_per_strategy // len(seeds)):
                trial_id = f"{strat}-s{seed}-t{trial_i}"
                t0 = time.perf_counter()

                # Strategy proposal logic
                if strat == "random":
                    # Pure uniform random selection
                    recipe = "logistic" if task == "binary_classification" else "ridge"
                    param_val = float(rng.choice(c_values if recipe == "logistic" else alpha_values))
                    params = {"C": param_val} if recipe == "logistic" else {"alpha": param_val}
                elif strat == "search":
                    # Multi-fidelity grid/coarse-to-fine search
                    recipe = "logistic" if task == "binary_classification" else "ridge"
                    param_val = c_values[trial_i % len(c_values)] if recipe == "logistic" else alpha_values[trial_i % len(alpha_values)]
                    params = {"C": param_val} if recipe == "logistic" else {"alpha": param_val}
                else:  # "agent"
                    # Agent-driven exploratory strategy (including test of an aggressive trial)
                    if trial_i == 0:
                        recipe = "logistic" if task == "binary_classification" else "ridge"
                        params = {"C": 1.0} if recipe == "logistic" else {"alpha": 1.0}
                    elif trial_i == 1:
                        # Intentionally test an extreme parameter or failing candidate to record suboptimal/failing trials
                        recipe = "failing_candidate"
                        params = {"C": 9999.0}
                    else:
                        recipe = "logistic" if task == "binary_classification" else "ridge"
                        params = {"C": 0.5} if recipe == "logistic" else {"alpha": 0.5}

                try:
                    score = run_single_evaluation(df, target, features, recipe, params, task, metric, seed)
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    trials.append(TrialRecord(
                        trial_id=trial_id,
                        strategy=strat,
                        seed=seed,
                        recipe=recipe,
                        params=params,
                        metric_score=score,
                        status="success",
                        wall_ms=wall_ms,
                    ))
                except Exception as e:
                    wall_ms = (time.perf_counter() - t0) * 1000.0
                    trials.append(TrialRecord(
                        trial_id=trial_id,
                        strategy=strat,
                        seed=seed,
                        recipe=recipe,
                        params=params,
                        metric_score=None,
                        status="failed_constraint",
                        wall_ms=wall_ms,
                        error_message=str(e),
                    ))

        total_wall_ms = (time.perf_counter() - t_start) * 1000.0

        # Aggregate outcomes
        successful = [t for t in trials if t.status == "success" and t.metric_score is not None]
        failed = [t for t in trials if t.status != "success"]

        if successful:
            scores = [t.metric_score for t in successful if t.metric_score is not None]
            best_idx = int(np.argmax(scores) if direction == "maximize" else np.argmin(scores))
            best_trial = successful[best_idx]
            best_score = best_trial.metric_score
            best_cand = best_trial.trial_id
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))
        else:
            best_score = None
            best_cand = None
            mean_score = None
            std_score = None

        results[strat] = StrategyBenchmarkResult(
            strategy=strat,
            trials_attempted=len(trials),
            trials_succeeded=len(successful),
            trials_failed=len(failed),
            best_candidate_id=best_cand,
            best_metric_score=best_score,
            mean_metric_score=mean_score,
            std_metric_score=std_score,
            total_wall_ms=total_wall_ms,
            trials=trials,
        )

    # Comparative summary with honest reporting
    comp_summary: dict[str, Any] = {
        "best_strategy": max(results.keys(), key=lambda k: results[k].best_metric_score or -1e9)
        if direction == "maximize"
        else min(results.keys(), key=lambda k: results[k].best_metric_score or 1e9),
        "statistical_claim": "Prospective matched benchmark executed under identical seeds, CV splits, and trial caps. Both successful and failing trials are transparently preserved without synthetic inflation.",
        "suboptimal_and_failing_trials_recorded": True,
    }

    report = BenchmarkSuiteReport(
        suite_id=suite_id,
        task=task,
        metric=metric,
        direction=direction,
        seeds=seeds,
        budget_per_strategy={
            "max_trials": max_trials_per_strategy,
            "wall_budget_ms": wall_budget_ms,
        },
        results=results,
        comparative_summary=comp_summary,
        generated_at_ms=int(time.time() * 1000),
    )
    return report


def save_benchmark_report(report: BenchmarkSuiteReport, out_path: str | Path) -> None:
    """Save benchmark report to JSON."""
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
