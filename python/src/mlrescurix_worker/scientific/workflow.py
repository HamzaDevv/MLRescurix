"""Small, deterministic CSV workflow for the M2 trusted-recipe boundary.

This module only loads joblib artifacts created by this trusted workflow. It is
not a safe loader for artifacts from an untrusted source.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mlrescurix_worker.search.adapter import normalize_parameters

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_CSV_BYTES = 32 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 128
MAX_CATEGORIES = 64
MAX_CSV_FIELD = 1_000_000
SEED = 2026


class WorkflowError(ValueError):
    """The input or requested workflow is outside the supported contract."""


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    target: str
    task: str
    metric: str
    split: str = "stratified"
    group_column: str | None = None
    positive_label: str | None = None
    inference_batch_size: int = 128
    inference_repetitions: int = 20
    max_latency_ms: float | None = None
    max_bundle_bytes: int | None = None

    def validate(self) -> None:
        if not self.project_id or len(self.project_id) > 128:
            raise WorkflowError("project_id must be 1..128 characters")
        if self.task not in {"binary_classification", "multiclass_classification", "regression"}:
            raise WorkflowError("unsupported task")
        metrics = {
            "binary_classification": {"balanced_accuracy", "log_loss"},
            "multiclass_classification": {"balanced_accuracy", "log_loss"},
            "regression": {"mae", "rmse"},
        }
        if self.metric not in metrics[self.task]:
            raise WorkflowError("metric is incompatible with task")
        if self.split not in {"stratified", "random", "grouped", "chronological"}:
            raise WorkflowError("unsupported split policy")
        if self.task == "regression" and self.split == "stratified":
            raise WorkflowError("regression requires random, grouped, or chronological split")
        if self.task != "regression" and self.split == "random":
            # Explicit random classification remains possible, but is never selected implicitly.
            pass
        if self.split == "grouped" and not self.group_column:
            raise WorkflowError("grouped split requires group_column")
        if not 1 <= self.inference_batch_size <= 4096 or not 1 <= self.inference_repetitions <= 100:
            raise WorkflowError("inference workload bounds exceeded")
        if self.max_latency_ms is not None and (not math.isfinite(self.max_latency_ms) or self.max_latency_ms <= 0):
            raise WorkflowError("max_latency_ms must be a finite positive value")
        if self.max_bundle_bytes is not None and self.max_bundle_bytes <= 0:
            raise WorkflowError("max_bundle_bytes must be positive")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()
    fd, tmp = tempfile.mkstemp(prefix=".m2-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _read_csv(path: Path) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise WorkflowError("CSV must be a regular, non-symlink file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_CSV_BYTES:
        raise WorkflowError(f"CSV size must be between 1 and {MAX_CSV_BYTES} bytes")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    try:
        frame = pd.read_csv(path, nrows=MAX_ROWS + 1, low_memory=False)
    except Exception as exc:
        raise WorkflowError(f"CSV parse failed: {exc}") from exc
    if len(frame) > MAX_ROWS:
        raise WorkflowError(f"CSV exceeds {MAX_ROWS} row limit")
    if not 2 <= len(frame.columns) <= MAX_COLUMNS:
        raise WorkflowError(f"CSV must have 2..{MAX_COLUMNS} columns")
    if frame.columns.has_duplicates or any(not isinstance(c, str) or not c.strip() for c in frame.columns):
        raise WorkflowError("column names must be nonempty and unique")
    if frame.empty or any(frame[c].dtype == object and frame[c].map(lambda x: isinstance(x, (list, dict))).any() for c in frame):
        raise WorkflowError("CSV must contain bounded scalar rows")
    return frame, digest.hexdigest()


def _profile(frame: pd.DataFrame) -> dict[str, Any]:
    columns = {}
    for name in frame.columns:
        col = frame[name]
        columns[name] = {
            "dtype": str(col.dtype), "missing": int(col.isna().sum()),
            "unique": int(col.nunique(dropna=True)),
            "categorical": bool(not pd.api.types.is_numeric_dtype(col.dtype)),
        }
    return {"rows": len(frame), "columns": columns}


def _validate_target(frame: pd.DataFrame, spec: ProjectSpec) -> np.ndarray:
    if spec.target not in frame:
        raise WorkflowError("target column is absent")
    target = frame[spec.target]
    if target.isna().any():
        raise WorkflowError("target contains missing values")
    if spec.group_column and spec.group_column == spec.target:
        raise WorkflowError("target cannot also be the group column")
    if spec.group_column and spec.group_column not in frame:
        raise WorkflowError("group column is absent")
    y = target.to_numpy()
    labels = np.unique(y)
    if spec.task == "regression":
        if not pd.api.types.is_numeric_dtype(target.dtype) or not np.isfinite(y.astype(float)).all():
            raise WorkflowError("regression target must be finite numeric values")
        if len(y) < 10:
            raise WorkflowError("at least 10 rows are required")
    else:
        if len(labels) < 2 or len(labels) > 100:
            raise WorkflowError("classification needs 2..100 observed classes")
        if spec.task == "binary_classification" and len(labels) != 2:
            raise WorkflowError("binary classification requires exactly two labels")
        if spec.task == "multiclass_classification" and len(labels) < 3:
            raise WorkflowError("multiclass classification requires at least three labels")
        if min(int((y == label).sum()) for label in labels) < 5:
            raise WorkflowError("each class needs at least five rows for the fixed split")
        if spec.positive_label is not None and str(spec.positive_label) not in {str(x) for x in labels}:
            raise WorkflowError("positive_label is not an observed class")
    return y


def _split(frame: pd.DataFrame, y: np.ndarray, spec: ProjectSpec) -> dict[str, list[int]]:
    indices = np.arange(len(frame))
    stratify = y if spec.split == "stratified" else None
    if spec.split == "grouped":
        groups = frame[spec.group_column].to_numpy()
        if pd.isna(groups).any() or len(np.unique(groups)) < 5:
            raise WorkflowError("group split requires at least five complete groups")
        outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        dev, test = next(outer.split(indices, y, groups))
        inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
        train_rel, val_rel = next(inner.split(dev, y[dev], groups[dev]))
        train, val = dev[train_rel], dev[val_rel]
    elif spec.split == "chronological":
        # Input row order is the declared time order; user must order it before snapshot.
        n_test = max(1, int(round(len(indices) * 0.2)))
        n_val = max(1, int(round(len(indices) * 0.2)))
        train, val, test = indices[: len(indices)-n_val-n_test], indices[len(indices)-n_val-n_test:len(indices)-n_test], indices[-n_test:]
    else:
        try:
            dev, test = train_test_split(indices, test_size=0.2, random_state=SEED, stratify=stratify)
            train, val = train_test_split(dev, test_size=0.25, random_state=SEED + 1,
                                          stratify=y[dev] if spec.split == "stratified" else None)
        except ValueError as exc:
            raise WorkflowError(f"fixed split is infeasible: {exc}") from exc
    if min(len(train), len(val), len(test)) < 2:
        raise WorkflowError("split produced a partition with fewer than two rows")
    if spec.task != "regression":
        labels = set(y)
        for name, part in (("train", train), ("validation", val), ("test", test)):
            if set(y[part]) != labels:
                raise WorkflowError(f"{name} partition is missing a class")
    return {"train": train.tolist(), "validation": val.tolist(), "test": test.tolist()}


def _normal_spec(value: ProjectSpec | dict[str, Any]) -> ProjectSpec:
    spec = value if isinstance(value, ProjectSpec) else ProjectSpec(**value)
    spec.validate()
    return spec


def audit_csv(csv_path: str | Path, spec: ProjectSpec | dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    spec = _normal_spec(spec)
    frame, fingerprint = _read_csv(Path(csv_path))
    y = _validate_target(frame, spec)
    features = [c for c in frame.columns if c != spec.target and c != spec.group_column]
    if not features:
        raise WorkflowError("at least one feature column is required")
    splits = _split(frame, y, spec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "dataset.csv"
    if snapshot.exists():
        old, old_hash = _read_csv(snapshot)
        if old_hash != fingerprint:
            raise WorkflowError("dataset snapshot already exists with another fingerprint")
    else:
        shutil.copyfile(csv_path, snapshot)
        os.chmod(snapshot, 0o444)
        _, snapshot_hash = _read_csv(snapshot)
        if snapshot_hash != fingerprint:
            raise WorkflowError("source CSV changed while creating the owned snapshot")
    manifest = {
        "format": "mlrescurix.dataset.v1", "project_id": spec.project_id,
        "sha256": fingerprint, "bytes": Path(csv_path).stat().st_size,
        "row_ids": "zero-based input row ordinal", "profile": _profile(frame),
        "target": spec.target, "task": spec.task, "metric": spec.metric,
        "split_policy": spec.split, "seed": SEED, "features": features,
        "splits": splits, "labels": [x.item() if hasattr(x, "item") else x for x in np.unique(y)] if spec.task != "regression" else None,
        "spec": asdict(spec),
    }
    _atomic_json(output / "dataset_manifest.json", manifest)
    return manifest


def _pipeline(frame: pd.DataFrame, features: list[str], task: str, recipe: str, seed: int, params: dict[str, Any] | None = None) -> Pipeline:
    p = normalize_parameters(recipe, task, params or {}) if params else {}
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(frame[c].dtype)]
    categorical = [c for c in features if c not in numeric]
    transformers = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if recipe == "linear":
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=MAX_CATEGORIES,
                                     sparse_output=True)),
        ]), categorical))
    preprocess = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.3,
                                   verbose_feature_names_out=False)
    if task == "regression":
        if recipe == "dummy":
            estimator = DummyRegressor(strategy="median")
        elif recipe == "linear":
            estimator = Ridge(alpha=p.get("alpha", 1.0), random_state=seed)
        else:
            estimator = RandomForestRegressor(
                n_estimators=p.get("n_estimators", 64),
                max_depth=p.get("max_depth", 12),
                min_samples_leaf=p.get("min_samples_leaf", 2),
                max_features=p.get("max_features", 1.0),
                n_jobs=1,
                random_state=seed,
            )
    else:
        if recipe == "dummy":
            estimator = DummyClassifier(strategy="prior")
        elif recipe == "linear":
            estimator = LogisticRegression(max_iter=500, C=p.get("C", 1.0), random_state=seed, n_jobs=1)
        else:
            estimator = RandomForestClassifier(
                n_estimators=p.get("n_estimators", 64),
                max_depth=p.get("max_depth", 12),
                min_samples_leaf=p.get("min_samples_leaf", 2),
                max_features=p.get("max_features", 1.0),
                n_jobs=1,
                random_state=seed,
            )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _predict(pipeline: Pipeline, frame: pd.DataFrame, features: list[str], task: str) -> dict[str, Any]:
    x = frame.loc[:, features]
    pred = pipeline.predict(x)
    result: dict[str, Any] = {"prediction": pred.tolist()}
    if task != "regression" and hasattr(pipeline, "predict_proba"):
        result["probabilities"] = pipeline.predict_proba(x).tolist()
        result["classes"] = pipeline.classes_.tolist()
    return result


def _score(y: np.ndarray, prediction: dict[str, Any], task: str) -> dict[str, float]:
    p = np.asarray(prediction["prediction"])
    if task == "regression":
        return {"mae": float(mean_absolute_error(y, p)), "rmse": float(math.sqrt(mean_squared_error(y, p)))}
    scores = {"balanced_accuracy": float(balanced_accuracy_score(y, p))}
    if "probabilities" in prediction:
        scores["log_loss"] = float(log_loss(y, prediction["probabilities"], labels=prediction["classes"]))
    return scores


def fit_trial_worker(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Fixed recipe worker: reads training labels and feature-only prediction rows."""
    spec = json.loads(Path(spec_path).read_text())
    root = Path(output_dir)
    train = pd.read_csv(root / "train.csv", low_memory=False)
    predict_frame = pd.read_csv(root / "predict.csv", low_memory=False)
    features = spec["features"]
    ids = spec["row_ids"]
    if "__mlr_row_id" not in predict_frame or predict_frame["__mlr_row_id"].tolist() != ids:
        raise WorkflowError("worker prediction row identity mismatch")
    x_predict = predict_frame.drop(columns=["__mlr_row_id"])
    if list(x_predict.columns) != features:
        raise WorkflowError("worker feature schema mismatch")
    model = _pipeline(train, features, spec["task"], spec["recipe"], spec["seed"], params=spec.get("parameters"))
    start = time.perf_counter()
    model.fit(train.loc[:, features], train[spec["target"]].to_numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000
    prediction = _predict(model, x_predict, features, spec["task"])
    model_path = root / "candidate.joblib"
    joblib.dump(model, model_path, compress=3)
    result = {"row_ids": ids, **prediction, "train_ms": elapsed_ms,
              "model_sha256": _sha256(model_path)}
    _atomic_json(root / "predictions.json", result)
    return result


def predict_trial_worker(spec_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(spec_path).read_text())
    root = Path(output_dir)
    frame = pd.read_csv(root / "predict.csv", low_memory=False)
    ids = spec["row_ids"]
    if "__mlr_row_id" not in frame or frame["__mlr_row_id"].tolist() != ids:
        raise WorkflowError("worker prediction row identity mismatch")
    features = spec["features"]
    x = frame.drop(columns=["__mlr_row_id"])
    if list(x.columns) != features:
        raise WorkflowError("worker feature schema mismatch")
    model = joblib.load(root / "model.joblib")
    task = spec["task"]
    result = {"row_ids": ids, **_predict(model, x, features, task)}
    _atomic_json(root / "predictions.json", result)
    return result


def _run_isolated_fit(root: Path, frame: pd.DataFrame, features: list[str], target: str,
                      task: str, recipe: str, seed: int, train_ids: list[int],
                      predict_ids: list[int], artifact_name: str | None = None,
                      params: dict[str, Any] | None = None) -> tuple[dict[str, Any], Path]:
    with tempfile.TemporaryDirectory(prefix="m2-fit-", dir=root) as scratch_name:
        scratch = Path(scratch_name)
        train = frame.iloc[train_ids].loc[:, features + [target]]
        predict = frame.iloc[predict_ids].loc[:, features].copy()
        predict.insert(0, "__mlr_row_id", predict_ids)
        train.to_csv(scratch / "train.csv", index=False)
        predict.to_csv(scratch / "predict.csv", index=False)
        _atomic_json(scratch / "worker-spec.json", {"features": features, "target": target,
                      "task": task, "recipe": recipe, "seed": seed, "row_ids": predict_ids,
                      "parameters": params})
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "PYTHONPATH", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}}
        env["OMP_NUM_THREADS"] = env["OPENBLAS_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
        proc = subprocess.run([sys.executable, "-m", "mlrescurix_worker.scientific.cli", "fit-trial",
                              "--spec", str(scratch / "worker-spec.json"), "--output", str(scratch)],
                             env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=3600, check=False)
        if proc.returncode != 0:
            raise WorkflowError(f"fixed recipe worker failed: {proc.stderr.decode(errors='replace')[:2000]}")
        result = json.loads((scratch / "predictions.json").read_text())
        _validate_worker_prediction(result, predict_ids, task)
        candidate = root / (artifact_name or f"{recipe}.joblib")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scratch / "candidate.joblib", candidate)
        result["model_sha256"] = _sha256(candidate)
        return result, candidate


def _validate_worker_prediction(result: dict[str, Any], row_ids: list[int], task: str) -> None:
    if result.get("row_ids") != row_ids or len(result.get("prediction", [])) != len(row_ids):
        raise WorkflowError("evaluator rejected worker prediction identities or length")
    pred = np.asarray(result["prediction"])
    if pred.ndim != 1:
        raise WorkflowError("evaluator rejected invalid prediction shape")
    if task == "regression":
        if not np.isfinite(pred.astype(float)).all():
            raise WorkflowError("evaluator rejected nonfinite regression values")
    else:
        classes = np.asarray(result.get("classes", []))
        if classes.ndim != 1 or len(classes) < 2 or not set(pred).issubset(set(classes)):
            raise WorkflowError("evaluator rejected classification labels or class order")
        probabilities = np.asarray(result.get("probabilities"), dtype=float)
        if probabilities.shape != (len(row_ids), len(classes)) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
            raise WorkflowError("evaluator rejected invalid probability output")


def _run_isolated_predict(root: Path, features_frame: pd.DataFrame, features: list[str], task: str,
                          model_path: Path, row_ids: list[int]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m2-predict-", dir=root) as scratch_name:
        scratch = Path(scratch_name)
        predict = features_frame.loc[:, features].copy()
        predict.insert(0, "__mlr_row_id", row_ids)
        predict.to_csv(scratch / "predict.csv", index=False)
        shutil.copy2(model_path, scratch / "model.joblib")
        _atomic_json(scratch / "worker-spec.json", {"features": features, "task": task, "row_ids": row_ids})
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "PYTHONPATH", "TMPDIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"}}
        proc = subprocess.run([sys.executable, "-m", "mlrescurix_worker.scientific.cli", "predict-trial",
                              "--spec", str(scratch / "worker-spec.json"), "--output", str(scratch)],
                             env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=600, check=False)
        if proc.returncode != 0:
            raise WorkflowError(f"inference worker failed: {proc.stderr.decode(errors='replace')[:2000]}")
        result = json.loads((scratch / "predictions.json").read_text())
        _validate_worker_prediction(result, row_ids, task)
        return result


def run_baselines(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    spec = _normal_spec(manifest["spec"])
    snapshot, digest = _read_csv(root / "dataset.csv")
    if digest != manifest["sha256"]:
        raise WorkflowError("dataset snapshot fingerprint changed")
    features, target = manifest["features"], manifest["target"]
    y = snapshot[target].to_numpy()
    rows = []
    for recipe in ("dummy", "linear", "tree"):
        train_ids, val_ids = manifest["splits"]["train"], manifest["splits"]["validation"]
        predicted, candidate_path = _run_isolated_fit(root, snapshot, features, target, spec.task, recipe, SEED, train_ids, val_ids)
        train_ms = predicted.pop("train_ms")
        scores = _score(y[val_ids], predicted, spec.task)
        prediction_path = root / f"{recipe}.validation.json"
        _atomic_json(prediction_path, {"row_ids": val_ids, "class_order": predicted.get("classes"), **predicted})
        rows.append({"candidate_id": recipe, "recipe": recipe, "status": "evaluated",
                     "metrics": scores, "train_ms": train_ms,
                     "model_sha256": _sha256(candidate_path),
                     "validation_predictions_sha256": _sha256(prediction_path)})
    results = {"format": "mlrescurix.results.v1", "project_id": spec.project_id,
               "dataset_sha256": digest, "metric": spec.metric,
               "direction": "minimize" if spec.metric in {"log_loss", "mae", "rmse"} else "maximize",
               "split_policy": spec.split, "candidates": rows}
    _atomic_json(root / "results.json", results)
    return results


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _get_fidelity_train_ids(manifest: dict[str, Any], frame: pd.DataFrame, fidelity: float, seed: int = SEED) -> list[int]:
    full_ids = manifest["splits"]["train"]
    if fidelity >= 1.0:
        return full_ids
    if fidelity <= 0.0 or fidelity > 1.0:
        raise WorkflowError("fidelity must be in (0, 1]")
    policy = manifest.get("split_policy", "random")
    target = manifest["target"]
    y = frame[target].to_numpy()
    n = max(1, int(round(len(full_ids) * fidelity)))
    if policy == "stratified":
        y_train = y[full_ids]
        class_counts = {c: int((y_train == c).sum()) for c in np.unique(y_train)}
        if any(cnt < 2 for cnt in class_counts.values()):
            raise WorkflowError("infeasible screening protocol: insufficient class counts for stratification")
        try:
            subset, _ = train_test_split(full_ids, train_size=fidelity, random_state=seed, stratify=y_train)
        except ValueError as exc:
            raise WorkflowError(f"infeasible screening protocol: {exc}") from exc
        y_subset = y[subset]
        if len(np.unique(y_subset)) < len(class_counts):
            raise WorkflowError("infeasible screening protocol: subset missing observed class")
        return sorted(subset)
    elif policy == "chronological":
        subset = full_ids[:n]
        if manifest.get("task") != "regression":
            observed_classes = set(y[full_ids])
            if set(y[subset]) != observed_classes:
                raise WorkflowError("infeasible screening protocol: chronological screening missing class")
        return subset
    elif policy == "grouped":
        group_col = manifest["spec"].get("group_column")
        if not group_col or group_col not in frame:
            raise WorkflowError("grouped split requires group_column in dataset")
        groups = frame.iloc[full_ids][group_col].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, train_size=fidelity, random_state=seed)
        try:
            sub_idx, _ = next(gss.split(full_ids, y[full_ids], groups))
            subset = [full_ids[i] for i in sub_idx]
        except Exception as exc:
            raise WorkflowError(f"infeasible screening protocol: {exc}") from exc
        if len(subset) < 2:
            raise WorkflowError("infeasible screening protocol: grouped subset has fewer than 2 rows")
        return subset
    else:
        subset, _ = train_test_split(full_ids, train_size=fidelity, random_state=seed)
        if manifest.get("task") != "regression":
            observed_classes = set(y[full_ids])
            if set(y[subset]) != observed_classes:
                raise WorkflowError("infeasible screening protocol: random screening missing class")
        return sorted(subset)


def run_trial(output_dir: str | Path, experiment_spec: dict[str, Any]) -> dict[str, Any]:
    """Execute a single bounded trial under the declared multi-fidelity search protocol."""
    root = Path(output_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    spec = _normal_spec(manifest["spec"])
    snapshot, digest = _read_csv(root / "dataset.csv")
    if digest != manifest["sha256"]:
        raise WorkflowError("dataset snapshot fingerprint changed")
    features, target = manifest["features"], manifest["target"]
    y = snapshot[target].to_numpy()

    exp_id = str(experiment_spec["experiment_id"])
    recipe = str(experiment_spec["recipe"])
    seed = int(experiment_spec.get("seed", SEED))
    fidelity = float(experiment_spec.get("fidelity", 1.0))
    params = normalize_parameters(recipe, spec.task, experiment_spec.get("parameters", {}))
    session_id = str(experiment_spec.get("session_id", "default"))
    node_type = str(experiment_spec.get("node_type", "screening" if fidelity < 1.0 else "full_fidelity"))
    parent_id = experiment_spec.get("parent_experiment_id")

    train_ids = _get_fidelity_train_ids(manifest, snapshot, fidelity, seed)
    val_ids = manifest["splits"]["validation"]

    trial_dir = root / "trials" / exp_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifact_rel = f"trials/{exp_id}/candidate.joblib"

    predicted, candidate_path = _run_isolated_fit(
        root, snapshot, features, target, spec.task, recipe, seed, train_ids, val_ids,
        params=params, artifact_name=artifact_rel,
    )
    train_ms = predicted.pop("train_ms")
    scores = _score(y[val_ids], predicted, spec.task)
    pred_path = trial_dir / "predictions.json"
    _atomic_json(pred_path, {"row_ids": val_ids, "class_order": predicted.get("classes"), **predicted})

    # Measure development inference latency on validation sample
    model = joblib.load(candidate_path)
    development = snapshot.iloc[val_ids].loc[:, features]
    sample = development.iloc[: min(spec.inference_batch_size, len(development))]
    timings = []
    for _ in range(spec.inference_repetitions):
        start = time.perf_counter_ns()
        model.predict(sample)
        timings.append((time.perf_counter_ns() - start) / 1e6)
    warm_p95_ms = float(np.percentile(timings, 95))
    model_size = candidate_path.stat().st_size

    latency_pass = (spec.max_latency_ms is None or warm_p95_ms <= spec.max_latency_ms)
    size_pass = (spec.max_bundle_bytes is None or model_size <= spec.max_bundle_bytes)
    is_feasible = bool(latency_pass and size_pass)

    constraints = {
        "inference_latency_ms": {
            "threshold": spec.max_latency_ms,
            "value": warm_p95_ms,
            "verdict": "pass" if latency_pass else "fail",
            "enforcement": "measured",
        },
        "model_bytes": {
            "threshold": None,
            "value": model_size,
            "verdict": "pass",
            "enforcement": "measured",
        },
        "memory": {"threshold": None, "value": None, "verdict": "unknown", "enforcement": "unsupported"},
    }

    trial_record = {
        "experiment_id": exp_id,
        "session_id": session_id,
        "recipe": recipe,
        "parameters": params,
        "fidelity": fidelity,
        "seed": seed,
        "node_type": node_type,
        "parent_experiment_id": parent_id,
        "status": "evaluated" if is_feasible else "infeasible",
        "primary_metric": scores[spec.metric],
        "metrics": scores,
        "constraints": constraints,
        "is_feasible": is_feasible,
        "train_ms": train_ms,
        "model_sha256": _sha256(candidate_path),
        "predictions_sha256": _sha256(pred_path),
        "artifact_path": f"trials/{exp_id}/trial.json",
    }
    _atomic_json(trial_dir / "trial.json", trial_record)
    return trial_record


def finalize_candidate(output_dir: str | Path, candidate_id: str, *, test_access_id: str) -> dict[str, Any]:
    """Refit one frozen recipe and consume the single test gate before scoring."""
    root = Path(output_dir)
    if not test_access_id or len(test_access_id) > 128:
        raise WorkflowError("test_access_id is required")
    gate_path = root / "test_access.json"
    results = json.loads((root / "results.json").read_text())
    candidate = next((x for x in results.get("candidates", []) if x["candidate_id"] == candidate_id), None)
    if candidate is None:
        trial_file = root / "trials" / candidate_id / "trial.json"
        if trial_file.exists():
            candidate = json.loads(trial_file.read_text())
            candidate["candidate_id"] = candidate_id
    if candidate is None or candidate.get("status") not in {"evaluated", "verified_development_only"}:
        raise WorkflowError("candidate is not eligible for finalization")
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    spec = _normal_spec(manifest["spec"])
    snapshot, digest = _read_csv(root / "dataset.csv")
    if digest != manifest["sha256"]:
        raise WorkflowError("dataset fingerprint changed")
    frozen_path = root / "frozen_model.joblib"
    if gate_path.exists():
        old = json.loads(gate_path.read_text())
        if old.get("test_access_id") != test_access_id:
            raise WorkflowError("test assessment is reserved for another access key")
        if old.get("candidate_id") != candidate_id or old.get("dataset_sha256") != digest:
            raise WorkflowError("test assessment identity changed")
        if (root / "final.json").exists():
            raise WorkflowError("test assessment already has a stored result; inspect final.json")
        if not frozen_path.is_file() or old.get("frozen_model_sha256") != _sha256(frozen_path):
            raise WorkflowError("reserved frozen artifact is missing or has changed")
        model = joblib.load(frozen_path)
    else:
        freeze_path = root / "freeze.json"
        freeze = {"candidate_id": candidate_id, "dataset_sha256": digest,
                  "split_sha256": _sha256(root / "dataset_manifest.json"), "status": "fitting"}
        if freeze_path.exists():
            previous = json.loads(freeze_path.read_text())
            if previous.get("dataset_sha256") != digest:
                raise WorkflowError("a different candidate is already frozen")
            if previous.get("candidate_id") != candidate_id and previous.get("status") != "rejected":
                raise WorkflowError("a different candidate is already frozen")
            if previous.get("candidate_id") == candidate_id and previous.get("status") == "rejected":
                raise WorkflowError("this candidate already failed its pre-test feasibility check")
        else:
            _atomic_json(freeze_path, freeze)
        trainval = manifest["splits"]["train"] + manifest["splits"]["validation"]
        candidate_params = candidate.get("parameters")
        _, refit_path = _run_isolated_fit(root, snapshot, manifest["features"], spec.target, spec.task,
                                          candidate["recipe"], SEED, trainval, manifest["splits"]["validation"],
                                          params=candidate_params,
                                          artifact_name="frozen-refit.joblib")
        shutil.copy2(refit_path, frozen_path)
        model = joblib.load(frozen_path)
        frozen_hash = _sha256(frozen_path)
        freeze.update({"status": "frozen", "frozen_model_sha256": frozen_hash})
        _atomic_json(freeze_path, freeze)
        # Constraints are checked on development features before test access.
        development = snapshot.iloc[manifest["splits"]["validation"]].loc[:, manifest["features"]]
        sample = development.iloc[: min(spec.inference_batch_size, len(development))]
        timings = []
        for _ in range(spec.inference_repetitions):
            start = time.perf_counter_ns()
            model.predict(sample)
            timings.append((time.perf_counter_ns() - start) / 1e6)
        latency = {"cold_load_ms": None, "batch_size": len(sample), "repetitions": len(timings),
                   "warm_p50_ms": float(np.percentile(timings, 50)), "warm_p95_ms": float(np.percentile(timings, 95)),
                   "warm_p99_ms": float(np.percentile(timings, 99)),
                   "rows_per_second": float(len(sample) / (np.median(timings) / 1000)) if np.median(timings) > 0 else None,
                   "measurement": "development raw-input features; perf_counter_ns; no competing harness work"}

        # Measure complete bundle bytes via staging before test gate access
        stage_dir = root / ".bundle_stage"
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(frozen_path, stage_dir / "frozen_model.joblib")
        (stage_dir / "inference.py").write_text(_inference_source())
        (stage_dir / "inference.lock").write_text(f"numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={__import__('scipy').__version__}\nscikit-learn=={sklearn.__version__}\njoblib=={joblib.__version__}\n")
        _atomic_json(stage_dir / "input_schema.json", {"format": "mlrescurix.input-schema.v1", "features": manifest["features"], "target": spec.target, "task": spec.task, "label_order": manifest["labels"]})
        (stage_dir / "README.md").write_text("MLRescurix verified development bundle\n")
        staged_training = stage_dir / "training"
        (staged_training / "source").mkdir(parents=True, exist_ok=True)
        (staged_training / "source" / "workflow.py").write_text(Path(__file__).read_text())
        _atomic_json(staged_training / "recipe.json", {"candidate_id": candidate_id, "recipe": candidate["recipe"], "parameters": candidate_params, "task": spec.task, "metric": spec.metric, "seed": SEED, "features": manifest["features"], "dependency_lock": "../inference.lock", "dataset_sha256": digest})
        ex_rows = snapshot.iloc[manifest["splits"]["validation"]].loc[:, manifest["features"]].head(5)
        _atomic_json(stage_dir / "examples_input.json", ex_rows.to_dict(orient="records"))
        # Stage projected scorecard, verification, final, and manifest
        proj_scorecard = {
            "project_id": spec.project_id,
            "candidate_id": candidate_id,
            "dataset_sha256": digest,
            "validation_metrics": candidate.get("metrics"),
            "final_test_metrics": {"projected": True, spec.metric: 0.0},
            "primary_metric": spec.metric,
            "direction": results.get("direction", "maximize"),
            "inference": latency,
            "constraints": {
                "inference_latency_ms": {"threshold": spec.max_latency_ms, "value": latency["warm_p95_ms"], "verdict": "pass", "enforcement": "measured"},
                "model_bytes": {"threshold": None, "value": frozen_path.stat().st_size, "verdict": "pass", "enforcement": "measured"},
                "bundle_bytes": {"threshold": spec.max_bundle_bytes, "value": 0, "verdict": "pass", "enforcement": "measured"},
                "memory": {"threshold": None, "value": None, "verdict": "unknown", "enforcement": "unsupported"},
            },
            "resource_limits": {"memory_quota": "unsupported"},
            "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        }
        _atomic_json(stage_dir / "scorecard.json", proj_scorecard)
        _atomic_json(stage_dir / "verification.json", {"frozen_model_sha256": frozen_hash, "bundle_candidate": candidate_id, "dataset_sha256": digest, "test_access_id": test_access_id, "raw_input_api": "predict(records)"})
        _atomic_json(stage_dir / "final.json", {"status": "verified_development_only", "candidate_id": candidate_id, "test_metrics": {spec.metric: 0.0}, "constraints": proj_scorecard["constraints"], "test_access_id": test_access_id, "bundle": "bundle"})
        if (root / "lineage.json").exists():
            shutil.copy2(root / "lineage.json", stage_dir / "lineage.json")
        staged_files_pre = [p for p in stage_dir.rglob("*") if p.is_file()]
        _atomic_json(stage_dir / "manifest.json", {
            "format": "mlrescurix.bundle.v1", "project_id": spec.project_id, "candidate_id": candidate_id,
            "files": {p.relative_to(stage_dir).as_posix(): {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in staged_files_pre}
        })

        staged_files = [p for p in stage_dir.rglob("*") if p.is_file()]
        bundle_bytes = sum(p.stat().st_size for p in staged_files)
        shutil.rmtree(stage_dir)

        latency_pass = (spec.max_latency_ms is None or latency["warm_p95_ms"] <= spec.max_latency_ms)
        bundle_pass = (spec.max_bundle_bytes is None or bundle_bytes <= spec.max_bundle_bytes)

        constraints = {
            "inference_latency_ms": {"threshold": spec.max_latency_ms, "value": latency["warm_p95_ms"],
                                     "verdict": "pass" if latency_pass else "fail",
                                     "enforcement": "measured"},
            "model_bytes": {"threshold": None, "value": frozen_path.stat().st_size,
                            "verdict": "pass",
                            "enforcement": "measured"},
            "bundle_bytes": {"threshold": spec.max_bundle_bytes, "value": bundle_bytes,
                             "verdict": "pass" if bundle_pass else "fail",
                             "enforcement": "measured"},
            "memory": {"threshold": None, "value": None, "verdict": "unknown", "enforcement": "unsupported"},
        }
        if any(v["verdict"] == "fail" for v in constraints.values()):
            final = {"status": "provisional_infeasible", "candidate_id": candidate_id,
                     "constraints": constraints, "test_access": "not_consumed"}
            freeze.update({"status": "rejected", "constraints": constraints})
            _atomic_json(freeze_path, freeze)
            _atomic_json(root / "final.json", final)
            return final
        frozen = {"project_id": spec.project_id, "candidate_id": candidate_id,
                  "dataset_sha256": digest, "split_sha256": _sha256(root / "dataset_manifest.json"),
                  "frozen_model_sha256": frozen_hash, "test_access_id": test_access_id,
                  "status": "reserved"}
        # Atomic exclusive creation records the frozen assessment identity before scoring.
        fd = os.open(gate_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(frozen, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
    test_ids = manifest["splits"]["test"]
    test = snapshot.iloc[test_ids]
    measured = _run_isolated_predict(root, test.loc[:, manifest["features"]], manifest["features"],
                                     spec.task, frozen_path, test_ids)
    test_scores = _score(test[spec.target].to_numpy(), measured, spec.task)
    metrics = {"mae": test_scores["mae"], "rmse": test_scores["rmse"]} if spec.task == "regression" else test_scores
    inference = root / "inference.py"
    inference.write_text(_inference_source())
    inference_lock = root / "inference.lock"
    inference_lock.write_text(f"numpy=={np.__version__}\npandas=={pd.__version__}\nscipy=={__import__('scipy').__version__}\nscikit-learn=={sklearn.__version__}\njoblib=={joblib.__version__}\n")
    schema = {"format": "mlrescurix.input-schema.v1", "features": manifest["features"], "target": spec.target,
              "task": spec.task, "label_order": manifest["labels"], "unknown_categories": "ignored by fitted one-hot encoder"}
    _atomic_json(root / "input_schema.json", schema)

    search_lineage = None
    selection_file = root / "selection.json"
    if selection_file.exists():
        try:
            search_lineage = json.loads(selection_file.read_text())
        except Exception:
            pass

    scorecard_data = {
        "project_id": spec.project_id,
        "candidate_id": candidate_id,
        "dataset_sha256": digest,
        "validation_metrics": candidate.get("metrics"),
        "final_test_metrics": metrics,
        "primary_metric": spec.metric,
        "direction": results.get("direction", "maximize"),
        "inference": latency,
        "constraints": constraints,
        "resource_limits": {"memory_quota": "unsupported"},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    if search_lineage:
        scorecard_data["search_lineage"] = search_lineage.get("lineage", search_lineage)
        scorecard_data["repeats"] = search_lineage.get("repeats")
        scorecard_data["selection_rationale"] = search_lineage.get("rationale")

    _atomic_json(root / "scorecard.json", scorecard_data)
    _atomic_json(root / "verification.json", {"frozen_model_sha256": frozen_hash, "bundle_candidate": candidate_id,
                    "dataset_sha256": digest, "test_access_id": test_access_id, "raw_input_api": "predict(records)"})
    final = {"status": "verified_development_only", "candidate_id": candidate_id, "test_metrics": metrics,
             "constraints": constraints, "test_access_id": test_access_id, "bundle": "bundle"}
    _atomic_json(root / "final.json", final)
    gate = json.loads(gate_path.read_text())
    gate["status"] = "consumed"
    gate["test_result_sha256"] = _sha256(root / "final.json")
    _atomic_json(gate_path, gate)
    bundle = root / "bundle"
    bundle.mkdir(exist_ok=True)
    for name in ("frozen_model.joblib", "inference.py", "inference.lock", "input_schema.json", "scorecard.json", "verification.json", "final.json"):
        shutil.copy2(root / name, bundle / name)
    training = bundle / "training"
    (training / "source").mkdir(parents=True, exist_ok=True)
    (training / "source" / "workflow.py").write_text(Path(__file__).read_text())
    _atomic_json(training / "recipe.json", {"candidate_id": candidate_id, "recipe": candidate["recipe"],
                  "parameters": candidate.get("parameters"),
                  "task": spec.task, "metric": spec.metric, "seed": SEED, "features": manifest["features"],
                  "dependency_lock": "../inference.lock", "dataset_sha256": digest})
    example_rows = snapshot.iloc[manifest["splits"]["validation"]].loc[:, manifest["features"]].head(5)
    example_prediction = _predict(model, example_rows, manifest["features"], spec.task)
    _atomic_json(bundle / "examples_input.json", example_rows.to_dict(orient="records"))
    _atomic_json(bundle / "examples_expected.json", example_prediction)
    if (root / "lineage.json").exists():
        shutil.copy2(root / "lineage.json", bundle / "lineage.json")
    files = sorted(p for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.json")
    _atomic_json(bundle / "manifest.json", {"format": "mlrescurix.bundle.v1", "project_id": spec.project_id,
                  "candidate_id": candidate_id, "files": {p.relative_to(bundle).as_posix():
                  {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files}})
    (bundle / "README.md").write_text("MLRescurix verified development bundle\n\nInstall dependencies from inference.lock, then import inference.predict(records). Only load this trusted bundle. The scorecard records the validation protocol, final test, measured constraints, and limitations.\n")
    manifest_data = json.loads((bundle / "manifest.json").read_text())
    manifest_data["files"]["README.md"] = {"bytes": (bundle / "README.md").stat().st_size,
                                             "sha256": _sha256(bundle / "README.md")}
    _atomic_json(bundle / "manifest.json", manifest_data)
    actual_bundle_files = [p for p in bundle.rglob("*") if p.is_file()]
    actual_bundle_bytes = sum(p.stat().st_size for p in actual_bundle_files)
    if spec.max_bundle_bytes is not None and actual_bundle_bytes > spec.max_bundle_bytes:
        final["status"] = "provisional_infeasible"
        final["constraints"]["bundle_bytes"]["verdict"] = "fail"
        final["constraints"]["bundle_bytes"]["value"] = actual_bundle_bytes
        _atomic_json(root / "final.json", final)
        return final
    return final


def _inference_source() -> str:
    return '''"""Standalone inference for a trusted MLRescurix bundle."""
from pathlib import Path
import joblib
import pandas as pd
_ROOT = Path(__file__).resolve().parent
_MODEL = joblib.load(_ROOT / "frozen_model.joblib")
_SCHEMA = __import__("json").loads((_ROOT / "input_schema.json").read_text())
def predict(records):
    frame = pd.DataFrame(records)
    required = _SCHEMA["features"]
    missing = set(required) - set(frame.columns)
    extra = set(frame.columns) - set(required)
    if missing or extra:
        raise ValueError(f"input schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    frame = frame.loc[:, required]
    labels = _MODEL.predict(frame).tolist()
    result = {"prediction": labels}
    if hasattr(_MODEL, "predict_proba"):
        result["probabilities"] = _MODEL.predict_proba(frame).tolist()
        result["classes"] = _MODEL.classes_.tolist()
    return result
'''


def evaluate_custom_candidate(
    workflow_dir: str | Path,
    candidate_id: str,
    predictions_path: str | Path,
    slices: list[str] | None = None,
) -> dict[str, Any]:
    """
    Independent trusted evaluator evaluation of custom candidate predictions.
    Reads row-keyed predictions from disk without importing or unpickling untrusted model code.
    Computes official development split metrics and requested slice diagnostics.
    """
    root = Path(workflow_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text())
    dataset = pd.read_csv(root / "dataset.csv")
    val_ids = manifest["splits"]["validation"]
    target = manifest["target"]
    task = manifest["task"]
    y_val = dataset.iloc[val_ids][target].to_numpy()

    with open(predictions_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    preds = payload["predictions"]
    if len(preds) != len(val_ids):
        raise WorkflowError(f"Prediction count {len(preds)} does not match validation set size {len(val_ids)}")

    pred_dict: dict[str, Any] = {
        "prediction": preds,
    }
    if payload.get("probabilities"):
        pred_dict["probabilities"] = payload["probabilities"]
        pred_dict["classes"] = manifest.get("labels", [0, 1])

    scores = _score(y_val, pred_dict, task)

    # Compute authorized aggregate slice diagnostics
    slice_diagnostics = {}
    if slices:
        for slice_col in slices:
            if slice_col in dataset.columns:
                slice_series = dataset.iloc[val_ids][slice_col].reset_index(drop=True)
                slice_results = {}
                for category, group_idx in slice_series.groupby(slice_series).groups.items():
                    idx_list = list(group_idx)
                    if len(idx_list) >= 2:
                        y_slice = y_val[idx_list]
                        slice_pred = {
                            "prediction": [preds[i] for i in idx_list],
                        }
                        if payload.get("probabilities"):
                            slice_pred["probabilities"] = [payload["probabilities"][i] for i in idx_list]
                            slice_pred["classes"] = pred_dict["classes"]
                        slice_results[str(category)] = {
                            "count": len(group_idx),
                            "metrics": _score(y_slice, slice_pred, task),
                        }
                slice_diagnostics[slice_col] = slice_results

    result = {
        "candidate_id": candidate_id,
        "status": "evaluated",
        "metrics": scores,
        "slice_diagnostics": slice_diagnostics,
        "n_samples": len(val_ids),
    }

    results_path = root / "results.json"
    if results_path.is_file():
        res_data = json.loads(results_path.read_text())
        candidates = res_data.get("candidates", [])
        candidates.append({
            "candidate_id": candidate_id,
            "recipe": "custom",
            "parameters": {},
            "metrics": scores,
            "status": "evaluated",
            "is_feasible": True,
            "slice_diagnostics": slice_diagnostics,
        })
        res_data["candidates"] = candidates
        _atomic_json(results_path, res_data)

    return result
