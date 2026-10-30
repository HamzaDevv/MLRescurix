"""Experimental, sequential SQLite-backed random search.

Cost and elapsed-time checks occur between trials; they do not reserve budget,
interrupt running work, or enforce a hard ceiling. This is not an Optuna adapter
and must not be used as the authority for paid or remote dispatch.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("mlrescurix.tuning")

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    from optuna.trial import TrialState as OptunaTrialState, create_trial as create_optuna_trial
    from optuna.distributions import CategoricalDistribution, FloatDistribution, IntDistribution
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class TrialState:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    PRUNED = "PRUNED"
    FAIL = "FAIL"


@dataclass
class BudgetedTrial:
    """Represents a single budgeted trial in an optimization study."""
    trial_id: str
    trial_number: int
    params: Dict[str, Any]
    state: str = TrialState.PENDING
    final_value: Optional[float] = None
    intermediate_values: Dict[int, float] = field(default_factory=dict)
    cost_usd: float = 0.0
    duration_seconds: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    _pruner: Optional[Any] = field(default=None, repr=False)
    _direction: str = field(default="maximize", repr=False)
    _optuna_trial: Optional[Any] = field(default=None, repr=False)

    def report(self, value: float, step: int) -> None:
        """Report an intermediate evaluation score at a specific epoch/step."""
        self.intermediate_values[step] = float(value)
        if self._optuna_trial is not None:
            try:
                self._optuna_trial.report(float(value), step)
            except Exception:
                pass

    def should_prune(self, step: Optional[int] = None) -> bool:
        """Check whether the trial should be pruned based on intermediate results."""
        opt_prune = False
        if self._optuna_trial is not None:
            try:
                opt_prune = self._optuna_trial.should_prune()
            except Exception:
                pass
        if opt_prune:
            return True
        if self._pruner is None:
            return False
        last_step = step if step is not None else (max(self.intermediate_values.keys()) if self.intermediate_values else 0)
        return self._pruner.should_prune(self, last_step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "params": self.params,
            "state": self.state,
            "final_value": self.final_value,
            "intermediate_values": self.intermediate_values,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TrialPruningHook:
    """Median pruner comparing trial intermediate scores against historical benchmarks."""

    def __init__(
        self,
        n_startup_trials: int = 2,
        n_warmup_steps: int = 1,
        direction: str = "maximize",
    ):
        self.n_startup_trials = n_startup_trials
        self.n_warmup_steps = n_warmup_steps
        self.direction = direction.lower()
        self._history: Dict[int, List[float]] = {}  # step -> list of values

    def record_step(self, step: int, value: float) -> None:
        if math.isfinite(value):
            self._history.setdefault(step, []).append(float(value))

    def should_prune(self, trial: BudgetedTrial, step: int) -> bool:
        if step < self.n_warmup_steps:
            return False

        hist = self._history.get(step, [])
        if len(hist) < self.n_startup_trials:
            return False

        current_val = trial.intermediate_values.get(step)
        if current_val is None or not math.isfinite(current_val):
            return True

        sorted_hist = sorted(hist)
        mid = len(sorted_hist) // 2
        median_val = sorted_hist[mid] if len(sorted_hist) % 2 != 0 else (sorted_hist[mid - 1] + sorted_hist[mid]) / 2.0

        if self.direction == "maximize":
            # If current score is worse than median, prune
            return current_val < median_val
        else:
            # For minimize, higher than median is worse -> prune
            return current_val > median_val


class BudgetedOptunaTuner:
    """Resource-bounded hyperparameter optimization coordinator with durable persistence."""

    def __init__(
        self,
        study_name: str,
        param_space: Dict[str, Dict[str, Any]],
        direction: str = "maximize",
        max_trials: int = 10,
        max_duration_seconds: float = 3600.0,
        max_cost_usd: float = 10.0,
        storage_path: Optional[Union[str, Path]] = None,
        pruner: Optional[TrialPruningHook] = None,
        campaign_authority: Optional[Any] = None,
        seed: int = 42,
        max_concurrent_trials: int = 1,
        trial_timeout_seconds: Optional[float] = None,
    ):
        self.study_name = study_name
        self.param_space = param_space
        self.direction = direction.lower()
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"Direction must be 'maximize' or 'minimize', got '{direction}'")
        self.max_trials = max_trials
        self.max_duration_seconds = max_duration_seconds
        self.max_cost_usd = max_cost_usd
        self.campaign_authority = campaign_authority
        self.seed = seed
        self.rng = random.Random(seed)
        self.max_concurrent_trials = max_concurrent_trials
        self.trial_timeout_seconds = trial_timeout_seconds

        self.storage_path = Path(storage_path).resolve() if storage_path else None
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()

        self.pruner = pruner or TrialPruningHook(direction=self.direction)
        self.study: Optional[Any] = None
        if HAS_OPTUNA:
            optuna_pruner = MedianPruner(
                n_startup_trials=self.pruner.n_startup_trials,
                n_warmup_steps=self.pruner.n_warmup_steps,
            )
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=TPESampler(seed=self.seed),
                pruner=optuna_pruner,
            )

        self.trials: List[BudgetedTrial] = []
        self._load_persisted_trials()

        self.total_cost_usd: float = sum(t.cost_usd for t in self.trials)
        self.total_duration_seconds: float = sum(t.duration_seconds for t in self.trials)
        self.start_wallclock: float = time.time()

    def _init_db(self) -> None:
        assert self.storage_path is not None
        with sqlite3.connect(self.storage_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tuning_trials (
                    trial_id TEXT PRIMARY KEY,
                    study_name TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    params TEXT NOT NULL,
                    state TEXT NOT NULL,
                    final_value REAL,
                    intermediate_values TEXT NOT NULL,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    start_time REAL NOT NULL,
                    end_time REAL
                )
            """)
            conn.commit()

    def _load_persisted_trials(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with sqlite3.connect(self.storage_path) as conn:
            cur = conn.execute("""
                SELECT trial_id, trial_number, params, state, final_value,
                       intermediate_values, cost_usd, duration_seconds, start_time, end_time
                FROM tuning_trials WHERE study_name = ? ORDER BY trial_number ASC
            """, (self.study_name,))
            for row in cur.fetchall():
                t = BudgetedTrial(
                    trial_id=row[0],
                    trial_number=int(row[1]),
                    params=json.loads(row[2]),
                    state=row[3],
                    final_value=float(row[4]) if row[4] is not None else None,
                    intermediate_values={int(k): float(v) for k, v in json.loads(row[5]).items()},
                    cost_usd=float(row[6]),
                    duration_seconds=float(row[7]),
                    start_time=float(row[8]),
                    end_time=float(row[9]) if row[9] is not None else None,
                    _pruner=self.pruner,
                    _direction=self.direction,
                )
                self.trials.append(t)
                # Seed pruner history with completed trials
                if t.state == TrialState.COMPLETE and t.intermediate_values:
                    for s, v in t.intermediate_values.items():
                        self.pruner.record_step(s, v)

                # Recover into Optuna study if initialized
                if HAS_OPTUNA and self.study is not None and t.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
                    try:
                        dists = {}
                        for pname, spec in self.param_space.items():
                            if pname in t.params:
                                ptype = spec.get("type", "float")
                                if ptype == "categorical":
                                    dists[pname] = CategoricalDistribution(choices=tuple(spec.get("choices", [])))
                                elif ptype == "int":
                                    dists[pname] = IntDistribution(
                                        low=spec.get("low", 1),
                                        high=spec.get("high", 100),
                                        step=spec.get("step", 1),
                                    )
                                elif ptype == "float":
                                    dists[pname] = FloatDistribution(
                                        low=float(spec.get("low", 1e-4)),
                                        high=float(spec.get("high", 1.0)),
                                        log=bool(spec.get("log", False)),
                                    )
                        opt_state = (
                            OptunaTrialState.COMPLETE if t.state == TrialState.COMPLETE
                            else (OptunaTrialState.PRUNED if t.state == TrialState.PRUNED else OptunaTrialState.FAIL)
                        )
                        f_trial = create_optuna_trial(
                            state=opt_state,
                            value=t.final_value,
                            params=t.params,
                            distributions=dists,
                            intermediate_values=t.intermediate_values,
                        )
                        self.study.add_trial(f_trial)
                    except Exception as e:
                        logger.debug("Failed recovering trial %s into Optuna study: %s", t.trial_id, e)

    def is_concurrency_exhausted(self) -> Tuple[bool, str]:
        """Checks if active running trials have reached the concurrency bound."""
        running = len([t for t in self.trials if t.state == TrialState.RUNNING])
        if running >= self.max_concurrent_trials:
            return True, f"Max concurrent trials reached ({running}/{self.max_concurrent_trials})"
        return False, ""

    def is_budget_exhausted(self) -> Tuple[bool, str]:
        """Checks trial count, duration, cost, and campaign ledger limits."""
        if len(self.trials) >= self.max_trials:
            return True, f"Max trials budget reached ({len(self.trials)}/{self.max_trials})"

        elapsed = time.time() - self.start_wallclock
        if elapsed >= self.max_duration_seconds:
            return True, f"Max duration budget exhausted ({round(elapsed, 1)}s/{self.max_duration_seconds}s)"

        if self.total_cost_usd >= self.max_cost_usd:
            return True, f"Max cost budget exhausted (${round(self.total_cost_usd, 4)}/${self.max_cost_usd})"

        if self.campaign_authority:
            status = self.campaign_authority.get_status() if hasattr(self.campaign_authority, "get_status") else {}
            remaining = status.get("remaining_campaign_usd", status.get("remaining_usd"))
            if remaining is None:
                return True, "Campaign budget status unavailable"
            remaining = float(remaining)
            if not math.isfinite(remaining):
                return True, "Campaign budget status invalid"
            if remaining <= 0.001:
                return True, "Campaign budget ledger exhausted"

        return False, ""

    def _sample_params(self) -> Dict[str, Any]:
        """Sample hyperparameter configuration from parameter space specification."""
        sampled = {}
        for name, spec in self.param_space.items():
            ptype = spec.get("type", "float")
            if ptype == "categorical":
                choices = spec.get("choices", [])
                sampled[name] = self.rng.choice(choices)
            elif ptype == "int":
                low = spec.get("low", 1)
                high = spec.get("high", 100)
                step = spec.get("step", 1)
                sampled[name] = self.rng.randrange(low, high + 1, step)
            elif ptype == "float":
                low = float(spec.get("low", 1e-4))
                high = float(spec.get("high", 1.0))
                if spec.get("log", False):
                    log_val = self.rng.uniform(math.log(low), math.log(high))
                    sampled[name] = float(math.exp(log_val))
                else:
                    sampled[name] = float(self.rng.uniform(low, high))
            else:
                sampled[name] = spec.get("default")
        return sampled

    def ask(self) -> Optional[BudgetedTrial]:
        """Generate and lease the next budgeted trial."""
        exhausted, reason = self.is_budget_exhausted()
        if exhausted:
            logger.info("Cannot lease new trial: %s", reason)
            return None

        concur_exhausted, concur_reason = self.is_concurrency_exhausted()
        if concur_exhausted:
            logger.info("Cannot lease new trial: %s", concur_reason)
            return None

        trial_num = len(self.trials)
        trial_id = f"trial_{self.study_name}_{trial_num}"

        optuna_trial = None
        if HAS_OPTUNA and self.study is not None:
            try:
                optuna_trial = self.study.ask()
                params = {}
                for name, spec in self.param_space.items():
                    ptype = spec.get("type", "float")
                    if ptype == "categorical":
                        choices = spec.get("choices", [])
                        params[name] = optuna_trial.suggest_categorical(name, choices)
                    elif ptype == "int":
                        low = spec.get("low", 1)
                        high = spec.get("high", 100)
                        step = spec.get("step", 1)
                        params[name] = optuna_trial.suggest_int(name, low, high, step=step)
                    elif ptype == "float":
                        low = float(spec.get("low", 1e-4))
                        high = float(spec.get("high", 1.0))
                        log = bool(spec.get("log", False))
                        params[name] = optuna_trial.suggest_float(name, low, high, log=log)
                    else:
                        params[name] = spec.get("default")
            except Exception as e:
                logger.debug("Optuna ask fallback: %s", e)
                params = self._sample_params()
        else:
            params = self._sample_params()

        # Apply campaign reservation if governed authority is bound
        if self.campaign_authority and hasattr(self.campaign_authority, "reserve"):
            est_cost = 0.01
            try:
                self.campaign_authority.reserve("tuning", trial_id, est_cost)
            except Exception as exc:
                logger.warning("Campaign reservation failed for trial %s: %s", trial_id, exc)
                return None

        trial = BudgetedTrial(
            trial_id=trial_id,
            trial_number=trial_num,
            params=params,
            state=TrialState.RUNNING,
            start_time=time.time(),
            _pruner=self.pruner,
            _direction=self.direction,
            _optuna_trial=optuna_trial,
        )

        self.trials.append(trial)

        # Persist intent to DB
        if self.storage_path:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    INSERT INTO tuning_trials (
                        trial_id, study_name, trial_number, params, state,
                        final_value, intermediate_values, cost_usd, duration_seconds, start_time, end_time
                    ) VALUES (?, ?, ?, ?, ?, NULL, ?, 0.0, 0.0, ?, NULL)
                """, (
                    trial.trial_id, self.study_name, trial.trial_number,
                    json.dumps(trial.params), trial.state,
                    json.dumps(trial.intermediate_values), trial.start_time
                ))
                conn.commit()

        return trial

    def tell(
        self,
        trial: BudgetedTrial,
        state: str,
        final_value: Optional[float] = None,
        duration_seconds: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        """Report completion, pruning, or failure of a trial, reconciling study and budget state."""
        if not any(t is trial for t in self.trials) or trial.state != TrialState.RUNNING:
            raise ValueError("Only a running trial belonging to this tuner can be completed")
        if state not in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL):
            raise ValueError("Trial completion requires a terminal state")
        if not math.isfinite(cost_usd) or cost_usd < 0 or not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise ValueError("Trial cost and duration must be finite and nonnegative")
        if final_value is not None and not math.isfinite(final_value):
            raise ValueError("Trial result must be finite")
        # Do not free the running slot or report success if accounting fails.
        # The reservation remains held for operator reconciliation.
        if self.campaign_authority:
            settle = getattr(self.campaign_authority, "settle", None)
            if not callable(settle):
                raise ValueError("Campaign authority cannot settle trial costs")
            settle("tuning", trial.trial_id, cost_usd, cost_kind="trial_measured")
        now = time.time()
        trial.state = state
        trial.final_value = final_value
        trial.duration_seconds = duration_seconds or (now - trial.start_time)
        trial.cost_usd = cost_usd
        trial.end_time = now

        self.total_cost_usd += trial.cost_usd
        self.total_duration_seconds += trial.duration_seconds

        # Update Optuna study if initialized
        if HAS_OPTUNA and self.study is not None and getattr(trial, "_optuna_trial", None) is not None:
            try:
                if trial.state == TrialState.COMPLETE:
                    self.study.tell(trial._optuna_trial, state=OptunaTrialState.COMPLETE, values=final_value)
                elif trial.state == TrialState.PRUNED:
                    self.study.tell(trial._optuna_trial, state=OptunaTrialState.PRUNED)
                else:
                    self.study.tell(trial._optuna_trial, state=OptunaTrialState.FAIL)
            except Exception as e:
                logger.debug("Optuna study.tell exception: %s", e)

        # Update pruner step observations for completed runs
        if trial.state == TrialState.COMPLETE and trial.intermediate_values:
            for s, v in trial.intermediate_values.items():
                self.pruner.record_step(s, v)

        # Persist update
        if self.storage_path:
            with sqlite3.connect(self.storage_path) as conn:
                conn.execute("""
                    UPDATE tuning_trials
                    SET state = ?, final_value = ?, intermediate_values = ?,
                        cost_usd = ?, duration_seconds = ?, end_time = ?
                    WHERE trial_id = ?
                """, (
                    trial.state, trial.final_value,
                    json.dumps(trial.intermediate_values),
                    trial.cost_usd, trial.duration_seconds, trial.end_time,
                    trial.trial_id,
                ))
                conn.commit()

    def optimize(
        self,
        objective_fn: Callable[[BudgetedTrial], Optional[float]],
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes trials sequentially up to budget boundaries."""
        target_trials = n_trials or self.max_trials

        while len([t for t in self.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]) < target_trials:
            trial = self.ask()
            if trial is None:
                break

            t0 = time.time()
            try:
                score = objective_fn(trial)
            except Exception as exc:
                duration = time.time() - t0
                logger.error("Trial %s raised exception: %s", trial.trial_id, exc)
                self.tell(trial, TrialState.FAIL, final_value=None, duration_seconds=duration)
                continue
            duration = time.time() - t0
            if self.trial_timeout_seconds and duration > self.trial_timeout_seconds:
                logger.warning("Trial %s exceeded duration timeout (%fs > %fs)", trial.trial_id, duration, self.trial_timeout_seconds)
                self.tell(trial, TrialState.FAIL, final_value=None, duration_seconds=duration)
            elif score is None:
                self.tell(trial, TrialState.PRUNED, final_value=None, duration_seconds=duration)
            else:
                self.tell(trial, TrialState.COMPLETE, final_value=score, duration_seconds=duration)

        return self.summary()

    def best_trial(self) -> Optional[BudgetedTrial]:
        """Returns the best completed trial according to optimization direction."""
        completed = [t for t in self.trials if t.state == TrialState.COMPLETE and t.final_value is not None]
        if not completed:
            return None
        if self.direction == "maximize":
            return max(completed, key=lambda t: t.final_value)  # type: ignore
        return min(completed, key=lambda t: t.final_value)  # type: ignore

    def summary(self) -> Dict[str, Any]:
        """Generates structured summary of study metrics, best parameters, and pruning stats."""
        best = self.best_trial()
        completed_count = len([t for t in self.trials if t.state == TrialState.COMPLETE])
        pruned_count = len([t for t in self.trials if t.state == TrialState.PRUNED])
        failed_count = len([t for t in self.trials if t.state == TrialState.FAIL])

        exhausted, reason = self.is_budget_exhausted()

        return {
            "study_name": self.study_name,
            "direction": self.direction,
            "total_trials": len(self.trials),
            "completed_trials": completed_count,
            "pruned_trials": pruned_count,
            "failed_trials": failed_count,
            "best_trial_id": best.trial_id if best else None,
            "best_value": best.final_value if best else None,
            "best_params": best.params if best else None,
            "total_duration_seconds": self.total_duration_seconds,
            "total_cost_usd": self.total_cost_usd,
            "budget_exhausted": exhausted,
            "budget_exhaustion_reason": reason if exhausted else None,
            "pruning_rate": (pruned_count / max(1, len(self.trials))),
        }
