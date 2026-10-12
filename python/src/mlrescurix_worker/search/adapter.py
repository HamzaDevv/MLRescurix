"""Search sidecar adapter for MLRescurix M3 constrained search.

Wraps Optuna ask/tell behind a replaceable adapter with separate optimizer
state per fidelity and recipe, and a deterministic fallback when Optuna is unavailable.
"""
from __future__ import annotations

import json
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

PARAM_RANGES = {
    "linear": {
        "classification": {"C": (0.001, 1000.0, True)},
        "regression": {"alpha": (0.001, 1000.0, True)},
    },
    "tree": {
        "n_estimators": (16, 128),
        "max_depth": (2, 20),
        "min_samples_leaf": (1, 20),
        "max_features": (0.2, 1.0),
    },
    "dummy": {},
}


def normalize_parameters(recipe: str, task: str, raw_params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize parameters into a deterministic, canonical representation."""
    if recipe == "dummy":
        if raw_params:
            raise ValueError("dummy recipe does not accept tunable parameters")
        return {}
    if recipe == "linear":
        is_regression = (task == "regression")
        param_name = "alpha" if is_regression else "C"
        allowed = {param_name}
        unknown = set(raw_params) - allowed
        if unknown:
            raise ValueError(f"linear recipe received unknown parameters: {sorted(unknown)}")
        if param_name not in raw_params:
            raise ValueError(f"linear recipe missing required parameter: {param_name}")
        val = raw_params[param_name]
        if not isinstance(val, (int, float)) or not math.isfinite(val) or isinstance(val, bool):
            raise ValueError(f"parameter {param_name} must be a finite number")
        val = float(val)
        low, high, _ = PARAM_RANGES["linear"]["regression" if is_regression else "classification"][param_name]
        if not (low <= val <= high):
            raise ValueError(f"parameter {param_name}={val} out of bounds [{low}, {high}]")
        return {param_name: round(val, 6)}
    if recipe == "tree":
        allowed = {"n_estimators", "max_depth", "min_samples_leaf", "max_features"}
        unknown = set(raw_params) - allowed
        if unknown:
            raise ValueError(f"tree recipe received unknown parameters: {sorted(unknown)}")
        out: dict[str, Any] = {}
        for k in ("n_estimators", "max_depth", "min_samples_leaf"):
            if k in raw_params:
                v = raw_params[k]
                if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
                    raise ValueError(f"parameter {k} must be an integer")
                iv = int(v)
                low, high = PARAM_RANGES["tree"][k]
                if not (low <= iv <= high):
                    raise ValueError(f"parameter {k}={iv} out of bounds [{low}, {high}]")
                out[k] = iv
            else:
                raise ValueError(f"tree recipe missing required parameter: {k}")
        if "max_features" in raw_params:
            mf = raw_params["max_features"]
            if not isinstance(mf, (int, float)) or isinstance(mf, bool) or not math.isfinite(mf):
                raise ValueError("parameter max_features must be a float")
            mf = float(mf)
            low, high = PARAM_RANGES["tree"]["max_features"]
            if not (low <= mf <= high):
                raise ValueError(f"parameter max_features={mf} out of bounds [{low}, {high}]")
            out["max_features"] = round(mf, 6)
        else:
            raise ValueError("tree recipe missing required parameter: max_features")
        return dict(sorted(out.items()))
    raise ValueError(f"unsupported recipe: {recipe}")


class SearchOptimizer(ABC):
    """Abstract ask/tell optimizer interface."""

    @abstractmethod
    def ask(self, recipe: str, fidelity: float, task: str) -> tuple[int, dict[str, Any]]:
        """Propose parameters for a trial. Returns (trial_id, normalized_params)."""

    @abstractmethod
    def tell(self, recipe: str, fidelity: float, trial_id: int, value: float | None, state: str) -> None:
        """Report completed or failed outcome for a proposed trial."""


class OptunaOptimizer(SearchOptimizer):
    """Optuna ask/tell adapter with separate studies per recipe and fidelity."""

    def __init__(self, storage_dir: Path, session_id: str, direction: str, seed: int = 2026):
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.optuna = optuna
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.direction = direction
        self.seed = seed
        self.db_path = self.storage_dir / f"optuna_{session_id}.db"
        self.storage = f"sqlite:///{self.db_path.resolve()}"

    def _study_name(self, recipe: str, fidelity: float) -> str:
        f_int = int(round(fidelity * 100))
        return f"{self.session_id}_{recipe}_f{f_int}"

    def _get_study(self, recipe: str, fidelity: float):
        name = self._study_name(recipe, fidelity)
        sampler = self.optuna.samplers.TPESampler(seed=self.seed)
        return self.optuna.create_study(
            study_name=name,
            storage=self.storage,
            direction=self.direction,
            load_if_exists=True,
            sampler=sampler,
        )

    def ask(self, recipe: str, fidelity: float, task: str) -> tuple[int, dict[str, Any]]:
        study = self._get_study(recipe, fidelity)
        trial = study.ask()
        raw: dict[str, Any] = {}
        if recipe == "linear":
            is_regression = (task == "regression")
            param_name = "alpha" if is_regression else "C"
            low, high, log = PARAM_RANGES["linear"]["regression" if is_regression else "classification"][param_name]
            raw[param_name] = trial.suggest_float(param_name, low, high, log=log)
        elif recipe == "tree":
            raw["n_estimators"] = trial.suggest_int("n_estimators", 16, 128)
            raw["max_depth"] = trial.suggest_int("max_depth", 2, 20)
            raw["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 20)
            raw["max_features"] = trial.suggest_float("max_features", 0.2, 1.0)
        norm = normalize_parameters(recipe, task, raw)
        return trial.number, norm

    def tell(self, recipe: str, fidelity: float, trial_id: int, value: float | None, state: str) -> None:
        study = self._get_study(recipe, fidelity)
        if state == "FAIL" or value is None:
            study.tell(trial_id, state=self.optuna.trial.TrialState.FAIL)
        elif state == "PRUNED":
            study.tell(trial_id, state=self.optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial_id, values=value, state=self.optuna.trial.TrialState.COMPLETE)


class DeterministicFallbackOptimizer(SearchOptimizer):
    """Deterministic fallback optimizer used when Optuna is unavailable or for testing."""

    def __init__(self, storage_dir: Path, session_id: str, direction: str, seed: int = 2026):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self.direction = direction
        self.seed = seed
        self.state_file = self.storage_dir / f"fallback_{session_id}.json"
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                self.state = json.loads(self.state_file.read_text())
            except Exception:
                self.state = {}
        else:
            self.state = {}

    def _save(self) -> None:
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _study_key(self, recipe: str, fidelity: float) -> str:
        f_int = int(round(fidelity * 100))
        return f"{recipe}_f{f_int}"

    def ask(self, recipe: str, fidelity: float, task: str) -> tuple[int, dict[str, Any]]:
        self._load()
        key = self._study_key(recipe, fidelity)
        study = self.state.setdefault(key, {"trials": [], "next_id": 0})
        trial_id = study["next_id"]
        study["next_id"] += 1

        # Deterministic sequence based on seed and trial_id
        step = trial_id
        raw: dict[str, Any] = {}
        if recipe == "linear":
            is_regression = (task == "regression")
            param_name = "alpha" if is_regression else "C"
            # Sequence covering log scale [0.001, 1000.0]
            values = [0.01, 0.1, 1.0, 10.0, 100.0, 0.001, 0.05, 0.5, 5.0, 50.0]
            raw[param_name] = values[step % len(values)]
        elif recipe == "tree":
            estimators = [16, 32, 48, 64, 80, 96, 112, 128]
            depths = [3, 5, 8, 10, 12, 15, 18, 20]
            leafs = [1, 2, 4, 6, 8, 10, 15, 20]
            features = [0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.5, 0.7]
            raw["n_estimators"] = estimators[step % len(estimators)]
            raw["max_depth"] = depths[(step * 2) % len(depths)]
            raw["min_samples_leaf"] = leafs[(step * 3) % len(leafs)]
            raw["max_features"] = features[(step * 4) % len(features)]
        norm = normalize_parameters(recipe, task, raw)
        study["trials"].append({"trial_id": trial_id, "params": norm, "status": "asked", "value": None})
        self._save()
        return trial_id, norm

    def tell(self, recipe: str, fidelity: float, trial_id: int, value: float | None, state: str) -> None:
        self._load()
        key = self._study_key(recipe, fidelity)
        study = self.state.setdefault(key, {"trials": [], "next_id": trial_id + 1})
        for t in study["trials"]:
            if t["trial_id"] == trial_id:
                t["status"] = state
                t["value"] = value
                break
        else:
            study["trials"].append({"trial_id": trial_id, "params": {}, "status": state, "value": value})
        self._save()


def get_optimizer(storage_dir: str | Path, session_id: str, direction: str, seed: int = 2026) -> SearchOptimizer:
    """Factory creating an OptunaOptimizer if available, otherwise DeterministicFallbackOptimizer."""
    try:
        import optuna  # noqa: F401
        return OptunaOptimizer(Path(storage_dir), session_id, direction, seed)
    except ImportError:
        return DeterministicFallbackOptimizer(Path(storage_dir), session_id, direction, seed)
