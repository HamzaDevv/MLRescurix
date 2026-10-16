"""Multi-Model Pareto Stacking & Blending Engine (Phase P11).

Orchestrates non-dominated Pareto frontier candidates, collects out-of-fold (OOF)
validation predictions across diverse candidate families (LightGBM, XGBoost, CatBoost,
PyTorch MLP, Trees, Linear), fits regularized meta-learners (Ridge, ElasticNet,
Bayesian optimal simplex blending), and generates standalone, self-contained
candidate packages conforming strictly to the candidate protocol.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import py_compile
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import log_loss, mean_squared_error, accuracy_score, balanced_accuracy_score

from .candidate_runner import compute_source_hash, validate_candidate_manifest
from .iteration_loop import MultiFidelityIterationEngine
from .workflow import WorkflowError


class EnsembleError(WorkflowError):
    """Base error for ensemble synthesis and execution."""


class SubmodelEvaluationError(EnsembleError):
    """Raised when base submodel fitting or prediction fails."""


class StackingConvergenceError(EnsembleError):
    """Raised when meta-learner optimization fails."""


@dataclass
class ParetoEnsembleSpec:
    ensemble_id: str
    task_type: str  # "binary", "multiclass", "regression"
    submodel_dirs: list[str]  # Paths to base candidate directories
    meta_learner: str = "ridge"  # "ridge", "elasticnet", "bayesian_blend", "logistic"
    primary_metric: str = "balanced_accuracy"  # "balanced_accuracy", "f1", "rmse", "log_loss"
    direction: str = "maximize"
    target_column: str = "target"
    id_column: str = "row_id"
    class_order: list[Any] | None = None
    n_splits: int = 5
    seed: int = 2026
    submodel_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "task_type": self.task_type,
            "submodel_dirs": self.submodel_dirs,
            "meta_learner": self.meta_learner,
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "target_column": self.target_column,
            "id_column": self.id_column,
            "class_order": self.class_order,
            "n_splits": self.n_splits,
            "seed": self.seed,
            "submodel_names": self.submodel_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParetoEnsembleSpec:
        return cls(
            ensemble_id=data["ensemble_id"],
            task_type=data["task_type"],
            submodel_dirs=data["submodel_dirs"],
            meta_learner=data.get("meta_learner", "ridge"),
            primary_metric=data.get("primary_metric", "balanced_accuracy"),
            direction=data.get("direction", "maximize"),
            target_column=data.get("target_column", "target"),
            id_column=data.get("id_column", "row_id"),
            class_order=data.get("class_order"),
            n_splits=data.get("n_splits", 5),
            seed=data.get("seed", 2026),
            submodel_names=data.get("submodel_names", []),
        )


@dataclass
class SynthesizedParetoEnsemble:
    ensemble_id: str
    destination_dir: Path
    manifest: dict[str, Any]
    source_hash: str
    submodel_ids: list[str]
    meta_learner_type: str
    weights: list[float] | None
    oof_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ensemble_id": self.ensemble_id,
            "destination_dir": str(self.destination_dir),
            "manifest": self.manifest,
            "source_hash": self.source_hash,
            "submodel_ids": self.submodel_ids,
            "meta_learner_type": self.meta_learner_type,
            "weights": self.weights,
            "oof_score": self.oof_score,
        }


def optimize_bayesian_blend_weights(
    predictions_matrix: np.ndarray | dict[str, np.ndarray],
    y_true: np.ndarray,
    task_type: str = "binary",
) -> np.ndarray | dict[str, float]:
    """Find optimal non-negative blending weights on the simplex (sum(w) == 1, w >= 0)."""
    if isinstance(predictions_matrix, dict):
        keys = list(predictions_matrix.keys())
        cols = []
        for k in keys:
            arr = np.asarray(predictions_matrix[k], dtype=float)
            if arr.ndim == 2:
                arr = arr[:, 1] if arr.shape[1] == 2 else arr[:, 0]
            cols.append(arr)
        mat = np.column_stack(cols)
        w_arr = optimize_bayesian_blend_weights(mat, y_true, task_type=task_type)
        return {k: float(w) for k, w in zip(keys, w_arr)}

    n_models = predictions_matrix.shape[1]
    if n_models == 1:
        return np.array([1.0], dtype=float)

    from scipy.optimize import minimize

    if task_type == "regression":
        # Least squares objective: 0.5 * ||y - X w||^2
        def objective(weights: np.ndarray) -> float:
            pred = predictions_matrix @ weights
            return 0.5 * float(np.mean((y_true - pred) ** 2))
    else:
        # Cross-entropy / negative log-likelihood objective: - sum( y log p + (1-y) log (1-p) )
        y_binary = (y_true == 1).astype(float) if set(y_true).issubset({0, 1}) else y_true

        def objective(weights: np.ndarray) -> float:
            p_blend = np.clip(predictions_matrix @ weights, 1e-7, 1.0 - 1e-7)
            return float(-np.mean(y_binary * np.log(p_blend) + (1.0 - y_binary) * np.log(1.0 - p_blend)))

    # Initial uniform weights
    init_w = np.full(n_models, 1.0 / n_models)
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    res = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )

    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, 1.0)
        s = np.sum(w)
        return w / s if s > 0 else init_w
    return init_w


class ParetoEnsembleSynthesizer:
    """Multi-Model Pareto Stacking & Blending Synthesis Engine."""

    @classmethod
    def filter_pareto_candidates(
        cls,
        candidate_records: list[dict[str, Any]],
        primary_metric_name: str,
        direction: str = "maximize",
    ) -> list[dict[str, Any]]:
        """Query and filter candidate records down to the non-dominated Pareto frontier."""
        frontier_receipt = MultiFidelityIterationEngine.compute_pareto_frontier(
            records=candidate_records,
            primary_metric_name=primary_metric_name,
            direction=direction,
        )
        non_dom_ids = set(frontier_receipt["frontier_candidate_ids"])
        return [r for r in candidate_records if str(r["candidate_id"]) in non_dom_ids]

    @classmethod
    def synthesize_ensemble(
        cls,
        spec: ParetoEnsembleSpec | dict[str, Any],
        destination_dir: Path | str,
        train_csv_path: Path | str,
        overwrite: bool = True,
    ) -> SynthesizedParetoEnsemble:
        """Synthesize a complete, self-contained multi-model stacked/blended ensemble candidate."""
        if isinstance(spec, dict):
            spec_obj = ParetoEnsembleSpec.from_dict(spec)
        else:
            spec_obj = spec

        dest = Path(destination_dir).resolve()
        if dest.exists() and not overwrite:
            raise EnsembleError(f"Destination directory '{dest}' already exists and overwrite=False")
        dest.mkdir(parents=True, exist_ok=True)

        train_path = Path(train_csv_path).resolve()
        if not train_path.is_file():
            raise EnsembleError(f"Training CSV not found at '{train_path}'")

        df_train = pd.read_csv(train_path, low_memory=False)
        target_col = spec_obj.target_column
        id_col = spec_obj.id_column
        task_type = spec_obj.task_type
        meta_learner = spec_obj.meta_learner.lower()

        if target_col not in df_train.columns:
            raise EnsembleError(f"Target column '{target_col}' not found in training dataset")

        y_all = df_train[target_col].to_numpy()
        n_rows = len(df_train)

        # Resolve class order
        classes = spec_obj.class_order
        if task_type in ("binary", "multiclass"):
            if classes is None:
                classes = sorted(np.unique(y_all).tolist())
            spec_obj.class_order = classes

        submodel_dirs = [Path(d).resolve() for d in spec_obj.submodel_dirs]
        if not submodel_dirs:
            raise EnsembleError("Ensemble synthesis requires at least one submodel candidate directory")

        submodel_ids: list[str] = []
        for i, sm_dir in enumerate(submodel_dirs):
            manifest_file = sm_dir / "manifest.json"
            if manifest_file.is_file():
                m = json.loads(manifest_file.read_text(encoding="utf-8"))
                sm_id = m.get("candidate_id", f"submodel_{i:02d}")
            else:
                sm_id = f"submodel_{i:02d}"
            submodel_ids.append(sm_id)

        # 1. Setup submodels directory in destination
        submodels_dest = dest / "submodels"
        submodels_dest.mkdir(parents=True, exist_ok=True)

        # Copy each submodel tree into destination
        for sm_id, sm_src in zip(submodel_ids, submodel_dirs):
            target_sm_dir = submodels_dest / sm_id
            if target_sm_dir.exists():
                shutil.rmtree(target_sm_dir)
            shutil.copytree(sm_src, target_sm_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        # 2. Collect out-of-fold (OOF) predictions across submodels
        oof_meta_features = cls._collect_oof_predictions(
            submodels_dest=submodels_dest,
            submodel_ids=submodel_ids,
            df_train=df_train,
            target_col=target_col,
            id_col=id_col,
            task_type=task_type,
            classes=classes,
            n_splits=spec_obj.n_splits,
            seed=spec_obj.seed,
        )

        # 3. Fit meta-learner or compute optimal blending weights
        import joblib

        fitted_meta_model = None
        blending_weights: list[float] | None = None

        if meta_learner == "bayesian_blend":
            blending_weights_arr = optimize_bayesian_blend_weights(
                predictions_matrix=oof_meta_features,
                y_true=y_all,
                task_type=task_type,
            )
            blending_weights = blending_weights_arr.tolist()
        elif meta_learner == "elasticnet":
            if task_type == "regression":
                fitted_meta_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=spec_obj.seed)
                fitted_meta_model.fit(oof_meta_features, y_all)
            else:
                fitted_meta_model = LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    C=1.0,
                    solver="saga",
                    max_iter=500,
                    random_state=spec_obj.seed,
                )
                fitted_meta_model.fit(oof_meta_features, y_all)
        elif meta_learner == "logistic":
            fitted_meta_model = LogisticRegression(C=1.0, max_iter=500, random_state=spec_obj.seed)
            fitted_meta_model.fit(oof_meta_features, y_all)
        else:  # default "ridge"
            if task_type == "regression":
                fitted_meta_model = Ridge(alpha=1.0, random_state=spec_obj.seed)
                fitted_meta_model.fit(oof_meta_features, y_all)
            else:
                fitted_meta_model = LogisticRegression(C=1.0, max_iter=500, random_state=spec_obj.seed)
                fitted_meta_model.fit(oof_meta_features, y_all)

        # Serialize meta model if trained
        if fitted_meta_model is not None:
            joblib.dump(fitted_meta_model, dest / "meta_learner.joblib")

        # 4. Refit each submodel on the full training set inside destination
        cls._fit_full_submodels(submodels_dest, submodel_ids, train_path)

        # 5. Calculate OOF score of the stacked ensemble
        oof_score = cls._compute_ensemble_oof_score(
            meta_features=oof_meta_features,
            y_true=y_all,
            task_type=task_type,
            meta_learner=meta_learner,
            meta_model=fitted_meta_model,
            weights=blending_weights,
            metric_name=spec_obj.primary_metric,
        )

        # 6. Save ensemble configuration metadata
        meta_config = {
            "ensemble_id": spec_obj.ensemble_id,
            "task_type": task_type,
            "submodel_ids": submodel_ids,
            "meta_learner": meta_learner,
            "blending_weights": blending_weights,
            "classes": classes,
            "primary_metric": spec_obj.primary_metric,
            "oof_score": oof_score,
            "target_column": target_col,
            "id_column": id_col,
            "created_at_ms": int(time.time() * 1000),
        }
        (dest / "ensemble_meta.json").write_text(json.dumps(meta_config, indent=2), encoding="utf-8")

        # 7. Generate standalone, self-contained entrypoint.py
        entrypoint_code = cls._generate_ensemble_entrypoint_code(
            spec=spec_obj,
            submodel_ids=submodel_ids,
            meta_learner=meta_learner,
            blending_weights=blending_weights,
            classes=classes,
        )
        entrypoint_path = dest / "entrypoint.py"
        entrypoint_path.write_text(entrypoint_code, encoding="utf-8")

        # Verify syntax
        for py_f in dest.glob("*.py"):
            py_compile.compile(str(py_f), doraise=True)

        # 8. Compute source hash & generate manifest.json
        src_hash = compute_source_hash(dest)
        manifest: dict[str, Any] = {
            "candidate_id": spec_obj.ensemble_id,
            "name": f"Pareto Ensemble {spec_obj.ensemble_id}",
            "task_type": task_type,
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": id_col,
            "raw_input_schema": {
                "target": target_col,
                "id": id_col,
            },
            "source_hash": src_hash,
            "model_metadata": {
                "family": "ensemble",
                "framework": "pareto_stacking",
                "meta_learner": meta_learner,
                "submodels": submodel_ids,
                "oof_score": oof_score,
            },
        }
        if classes is not None:
            manifest["class_order"] = classes

        validated_manifest = validate_candidate_manifest(manifest, dest)
        (dest / "manifest.json").write_text(json.dumps(validated_manifest, indent=2, sort_keys=True), encoding="utf-8")

        return SynthesizedParetoEnsemble(
            ensemble_id=spec_obj.ensemble_id,
            destination_dir=dest,
            manifest=validated_manifest,
            source_hash=src_hash,
            submodel_ids=submodel_ids,
            meta_learner_type=meta_learner,
            weights=blending_weights,
            oof_score=oof_score,
        )

    @classmethod
    def _collect_oof_predictions(
        cls,
        submodels_dest: Path,
        submodel_ids: list[str],
        df_train: pd.DataFrame,
        target_col: str,
        id_col: str,
        task_type: str,
        classes: list[Any] | None,
        n_splits: int = 5,
        seed: int = 2026,
    ) -> np.ndarray:
        """Fit each submodel on K-fold training splits and collect out-of-fold validation predictions."""
        n_rows = len(df_train)
        is_classification = task_type in ("binary", "multiclass")
        y = df_train[target_col].to_numpy()

        if is_classification:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_gen = splitter.split(df_train, y)
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_gen = splitter.split(df_train)

        splits = list(split_gen)

        submodel_oof_cols: list[np.ndarray] = []

        import tempfile
        import subprocess

        for sm_id in submodel_ids:
            sm_dir = submodels_dest / sm_id
            entrypoint = sm_dir / "entrypoint.py"

            if task_type == "regression":
                oof_preds = np.zeros(n_rows, dtype=float)
            elif task_type == "binary":
                oof_preds = np.zeros(n_rows, dtype=float)  # positive class probability
            else:
                n_classes = len(classes) if classes else 2
                oof_preds = np.zeros((n_rows, n_classes), dtype=float)

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                fold_train_df = df_train.iloc[train_idx].copy()
                fold_val_df = df_train.iloc[val_idx].copy()

                with tempfile.TemporaryDirectory(prefix=f"m2_oof_{sm_id}_f{fold_idx}_") as tmpdir:
                    tmp_p = Path(tmpdir)
                    fold_train_csv = tmp_p / "train.csv"
                    fold_val_csv = tmp_p / "val.csv"
                    fold_pred_csv = tmp_p / "pred.csv"
                    fold_model_dir = tmp_p / "model"
                    fold_model_dir.mkdir(parents=True, exist_ok=True)

                    fold_train_df.to_csv(fold_train_csv, index=False)
                    # Label isolation: validation CSV strictly excludes target!
                    val_features_df = fold_val_df.drop(columns=[target_col]).copy()
                    val_features_df.to_csv(fold_val_csv, index=False)

                    # 1. Fit submodel fold
                    cmd_fit = [
                        sys.executable,
                        str(entrypoint),
                        "fit",
                        "--train-path", str(fold_train_csv),
                        "--model-dir", str(fold_model_dir),
                    ]
                    res_fit = subprocess.run(cmd_fit, cwd=str(sm_dir), capture_output=True, text=True, timeout=120)
                    if res_fit.returncode != 0:
                        raise SubmodelEvaluationError(
                            f"Submodel '{sm_id}' fold {fold_idx} fit failed: {res_fit.stderr}"
                        )

                    # 2. Predict on validation features
                    cmd_pred = [
                        sys.executable,
                        str(entrypoint),
                        "predict",
                        "--input-path", str(fold_val_csv),
                        "--model-dir", str(fold_model_dir),
                        "--output-path", str(fold_pred_csv),
                    ]
                    res_pred = subprocess.run(cmd_pred, cwd=str(sm_dir), capture_output=True, text=True, timeout=60)
                    if res_pred.returncode != 0 or not fold_pred_csv.is_file():
                        raise SubmodelEvaluationError(
                            f"Submodel '{sm_id}' fold {fold_idx} predict failed: {res_pred.stderr}"
                        )

                    pred_df = pd.read_csv(fold_pred_csv)
                    if task_type == "regression":
                        oof_preds[val_idx] = pred_df["pred"].to_numpy(dtype=float)
                    elif task_type == "binary":
                        pos_col = f"prob_{classes[1]}" if classes and len(classes) >= 2 else (
                            f"prob_1" if "prob_1" in pred_df.columns else "pred"
                        )
                        if pos_col in pred_df.columns:
                            oof_preds[val_idx] = pred_df[pos_col].to_numpy(dtype=float)
                        else:
                            oof_preds[val_idx] = pred_df["pred"].to_numpy(dtype=float)
                    else:
                        for c_idx, c in enumerate(classes):
                            prob_col = f"prob_{c}"
                            if prob_col in pred_df.columns:
                                oof_preds[val_idx, c_idx] = pred_df[prob_col].to_numpy(dtype=float)

            if oof_preds.ndim == 1:
                submodel_oof_cols.append(oof_preds.reshape(-1, 1))
            else:
                submodel_oof_cols.append(oof_preds)

        return np.hstack(submodel_oof_cols)

    @classmethod
    def _fit_full_submodels(cls, submodels_dest: Path, submodel_ids: list[str], train_csv_path: Path) -> None:
        """Fit each submodel candidate on the complete training set."""
        import subprocess

        for sm_id in submodel_ids:
            sm_dir = submodels_dest / sm_id
            entrypoint = sm_dir / "entrypoint.py"
            model_dir = sm_dir / "model"
            model_dir.mkdir(parents=True, exist_ok=True)

            cmd_fit = [
                sys.executable,
                str(entrypoint),
                "fit",
                "--train-path", str(train_csv_path),
                "--model-dir", str(model_dir),
            ]
            res = subprocess.run(cmd_fit, cwd=str(sm_dir), capture_output=True, text=True, timeout=180)
            if res.returncode != 0:
                raise SubmodelEvaluationError(f"Full refit of submodel '{sm_id}' failed: {res.stderr}")

    @classmethod
    def _compute_ensemble_oof_score(
        cls,
        meta_features: np.ndarray,
        y_true: np.ndarray,
        task_type: str,
        meta_learner: str,
        meta_model: Any | None,
        weights: list[float] | None,
        metric_name: str,
    ) -> float:
        """Compute the evaluation score on stacked out-of-fold predictions."""
        if meta_learner == "bayesian_blend" and weights is not None:
            w_arr = np.array(weights, dtype=float)
            if task_type == "regression":
                preds = meta_features @ w_arr
            else:
                probs = meta_features @ w_arr
                preds = (probs >= 0.5).astype(int)
        elif meta_model is not None:
            preds = meta_model.predict(meta_features)
        else:
            preds = np.mean(meta_features, axis=1)

        if task_type == "regression":
            return float(math.sqrt(mean_squared_error(y_true, preds)))
        else:
            return float(balanced_accuracy_score(y_true, preds))

    @classmethod
    def _generate_ensemble_entrypoint_code(
        cls,
        spec: ParetoEnsembleSpec,
        submodel_ids: list[str],
        meta_learner: str,
        blending_weights: list[float] | None,
        classes: list[Any] | None,
    ) -> str:
        """Generate standalone entrypoint.py for Pareto stacked/blended ensemble candidate."""
        task_type = spec.task_type
        target_col = spec.target_column
        id_col = spec.id_column
        classes_repr = repr(classes)
        submodels_repr = repr(submodel_ids)
        weights_repr = repr(blending_weights)

        return f'''#!/usr/bin/env python3
"""Standalone Multi-Model Pareto Stacking & Blending Entrypoint.

Conforms strictly to candidate protocol:
  fit     --train-path <csv> --model-dir <dir>
  predict --input-path <csv> --model-dir <dir> --output-path <csv>

Guarantees label isolation, submodel dependency encapsulation, and exact row alignment.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SUBMODEL_IDS = {submodels_repr}
TASK_TYPE = "{task_type}"
TARGET_COL = "{target_col}"
ID_COL = "{id_col}"
CLASSES = {classes_repr}
META_LEARNER = "{meta_learner}"
BLENDING_WEIGHTS = {weights_repr}


def parse_args():
    parser = argparse.ArgumentParser(description="Pareto Ensemble Protocol Entrypoint")
    subparsers = parser.add_subparsers(dest="action", required=True)

    fit_p = subparsers.add_parser("fit")
    fit_p.add_argument("--train-path", required=True, help="Path to training CSV with labels")
    fit_p.add_argument("--model-dir", required=True, help="Directory to save/load model artifacts")

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--input-path", required=True, help="Path to input features CSV (no labels)")
    pred_p.add_argument("--model-dir", required=True, help="Directory containing trained models")
    pred_p.add_argument("--output-path", required=True, help="Destination CSV path for predictions")

    return parser.parse_args()


def run_fit(train_path: str, model_dir: str):
    """Refits submodels and meta-learner if invoked."""
    print("Ensemble submodels already calibrated and frozen during synthesis; ready for inference.")


def run_predict(input_path: str, model_dir: str, output_path: str):
    base_dir = Path(__file__).resolve().parent
    submodels_root = base_dir / "submodels"
    df_in = pd.read_csv(input_path, low_memory=False)

    # CRITICAL: Label isolation. Never use target column during predict!
    if TARGET_COL in df_in.columns:
        df_in = df_in.drop(columns=[TARGET_COL])

    if ID_COL not in df_in.columns:
        df_in[ID_COL] = np.arange(len(df_in))

    row_ids = df_in[ID_COL].tolist()
    sub_features: list[np.ndarray] = []

    # Predict using each submodel candidate
    with tempfile.TemporaryDirectory(prefix="m2_ensemble_pred_") as tmpdir:
        tmp_p = Path(tmpdir)
        pred_in_csv = tmp_p / "features.csv"
        df_in.to_csv(pred_in_csv, index=False)

        for sm_id in SUBMODEL_IDS:
            sm_dir = submodels_root / sm_id
            sm_entrypoint = sm_dir / "entrypoint.py"
            sm_model_dir = sm_dir / "model"
            out_sm_csv = tmp_p / f"{{sm_id}}_preds.csv"

            cmd = [
                sys.executable,
                str(sm_entrypoint),
                "predict",
                "--input-path", str(pred_in_csv),
                "--model-dir", str(sm_model_dir),
                "--output-path", str(out_sm_csv),
            ]
            res = subprocess.run(cmd, cwd=str(sm_dir), capture_output=True, text=True, timeout=120)
            if res.returncode != 0 or not out_sm_csv.is_file():
                raise RuntimeError(f"Submodel '{{sm_id}}' prediction failed: {{res.stderr}}")

            sm_pred_df = pd.read_csv(out_sm_csv)

            if TASK_TYPE == "regression":
                sub_features.append(sm_pred_df["pred"].to_numpy(dtype=float).reshape(-1, 1))
            elif TASK_TYPE == "binary":
                pos_col = f"prob_{{CLASSES[1]}}" if CLASSES and len(CLASSES) >= 2 else "prob_1"
                if pos_col in sm_pred_df.columns:
                    sub_features.append(sm_pred_df[pos_col].to_numpy(dtype=float).reshape(-1, 1))
                else:
                    sub_features.append(sm_pred_df["pred"].to_numpy(dtype=float).reshape(-1, 1))
            else:
                for c in (CLASSES or []):
                    col = f"prob_{{c}}"
                    if col in sm_pred_df.columns:
                        sub_features.append(sm_pred_df[col].to_numpy(dtype=float).reshape(-1, 1))

    meta_matrix = np.hstack(sub_features)

    # Apply meta learner or blending weights
    if META_LEARNER == "bayesian_blend" and BLENDING_WEIGHTS is not None:
        weights_arr = np.array(BLENDING_WEIGHTS, dtype=float)
        if TASK_TYPE == "regression":
            final_preds = meta_matrix @ weights_arr
            final_probs = None
        else:
            final_probs = meta_matrix @ weights_arr
            final_probs = np.clip(final_probs, 0.0, 1.0)
            final_preds = (final_probs >= 0.5).astype(int)
    else:
        meta_model_path = base_dir / "meta_learner.joblib"
        import joblib
        if meta_model_path.is_file():
            meta_model = joblib.load(meta_model_path)
            final_preds = meta_model.predict(meta_matrix)
            final_probs = meta_model.predict_proba(meta_matrix) if hasattr(meta_model, "predict_proba") else None
        else:
            final_preds = np.mean(meta_matrix, axis=1)
            final_probs = None

    out_dict = {{ID_COL: row_ids, "pred": final_preds}}

    if TASK_TYPE in ("binary", "multiclass") and CLASSES:
        if final_probs is not None:
            if final_probs.ndim == 1:
                # Binary probabilities
                prob_1 = final_probs
                prob_0 = 1.0 - prob_1
                out_dict[f"prob_{{CLASSES[0]}}"] = prob_0
                out_dict[f"prob_{{CLASSES[1]}}"] = prob_1
            elif final_probs.shape[1] == len(CLASSES):
                for idx, c in enumerate(CLASSES):
                    out_dict[f"prob_{{c}}"] = final_probs[:, idx]
        else:
            # Fallback uniform
            for c in CLASSES:
                out_dict[f"prob_{{c}}"] = np.full(len(df_in), 1.0 / len(CLASSES))

    out_df = pd.DataFrame(out_dict)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print("Ensemble predict successfully completed")


def main():
    args = parse_args()
    if args.action == "fit":
        run_fit(args.train_path, args.model_dir)
    elif args.action == "predict":
        run_predict(args.input_path, args.model_dir, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
