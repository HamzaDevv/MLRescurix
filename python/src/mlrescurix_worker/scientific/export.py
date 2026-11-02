"""Native, SafeTensors, and ONNX model export adapters with documented tolerances.

Enforces:
1. Strict sandbox containment during candidate serialization (no path escapes, no symlinks).
2. Explicit documented numerical tolerances per modality (classification vs regression vs tree).
3. Zero-pickle standalone SafeTensors prediction adapter without joblib dependency.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Explicit documented numerical tolerances per modality and export format
DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "binary_classification": {
        "probabilities_rtol": 1e-5,
        "probabilities_atol": 1e-7,
        "classes_atol": 0.0,
    },
    "multiclass_classification": {
        "probabilities_rtol": 1e-5,
        "probabilities_atol": 1e-7,
        "classes_atol": 0.0,
    },
    "regression": {
        "prediction_rtol": 1e-5,
        "prediction_atol": 1e-6,
    },
    "tree_discrete": {
        "prediction_rtol": 1e-4,
        "prediction_atol": 1e-5,
    },
}


class SafeTensorsPredictor:
    """Zero-pickle inference engine executing predictions directly from SafeTensors arrays."""

    def __init__(self, export_dir: str | Path):
        from safetensors.numpy import load_file

        self.root = Path(export_dir).resolve()
        weights_path = self.root / "weights.safetensors"
        meta_path = self.root / "safetensors_meta.json"
        if not weights_path.is_file():
            raise FileNotFoundError(f"SafeTensors weights missing at {weights_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"SafeTensors metadata missing at {meta_path}")

        self.weights = load_file(str(weights_path))
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.model_type = self.meta.get("model_type", "linear")
        self.task = self.meta.get("task", "binary_classification")
        self.features = self.meta.get("features", [])
        self.classes = np.array(self.meta.get("classes", [0, 1]))
        self.decision_threshold = float(self.meta.get("decision_threshold", 0.5))

    def predict(self, records: list[dict[str, Any]] | np.ndarray) -> dict[str, Any]:
        """Compute predictions using extracted tensors and NumPy operations."""
        if isinstance(records, list):
            # Extract features in order
            matrix = np.array([[row[f] for f in self.features] for row in records], dtype=np.float32)
        else:
            matrix = np.asarray(records, dtype=np.float32)

        # Apply standard scaling if present
        if "scaler_mean" in self.weights and "scaler_scale" in self.weights:
            mean = self.weights["scaler_mean"]
            scale = self.weights["scaler_scale"]
            matrix = (matrix - mean) / np.where(scale == 0, 1.0, scale)

        coef = self.weights.get("coef")
        intercept = self.weights.get("intercept", np.zeros((1,), dtype=np.float32))

        if coef is None:
            raise ValueError("weights.safetensors missing 'coef' tensor")

        # Linear activation: z = X @ coef.T + intercept
        z = matrix @ coef.T + intercept

        if self.task in ("binary_classification", "multiclass_classification"):
            if z.ndim == 1 or (z.ndim == 2 and z.shape[1] == 1):
                # Binary logistic sigmoid
                z_flat = z.ravel()
                prob_1 = 1.0 / (1.0 + np.exp(-np.clip(z_flat, -50.0, 50.0)))
                prob_0 = 1.0 - prob_1
                probs = np.column_stack([prob_0, prob_1])
                pred_indices = (prob_1 >= self.decision_threshold).astype(int)
            else:
                # Multiclass softmax
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                pred_indices = np.argmax(probs, axis=1)

            preds = self.classes[pred_indices].tolist()
            return {
                "prediction": preds,
                "probabilities": probs.tolist(),
                "classes": self.classes.tolist(),
            }
        else:
            # Regression continuous output
            preds = z.ravel().tolist()
            return {
                "prediction": preds,
            }


def export_model(
    model: Any,
    export_dir: str | Path,
    formats: list[str] | None = None,
    task: str = "binary_classification",
    features: list[str] | None = None,
    classes: list[Any] | None = None,
    tolerances: dict[str, float] | None = None,
    decision_threshold: float | None = None,
    calibration_method: str | None = None,
    calibration_metrics: dict[str, Any] | None = None,
    reliability_curves: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export fitted model into specified formats enforcing strict sandbox boundaries.

    Returns summary manifest of exported formats and hashes.
    """
    import joblib
    from safetensors.numpy import save_file

    if formats is None:
        formats = ["joblib", "safetensors"]

    raw_path = Path(export_dir)
    # Enforce sandbox bounds: export_dir must not contain parent escapes
    if ".." in raw_path.parts:
        raise ValueError(f"Export directory '{export_dir}' escapes sandbox boundaries")
    out_dir = raw_path.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    declared_tolerances = DEFAULT_TOLERANCES.get(task, DEFAULT_TOLERANCES["binary_classification"]).copy()
    if tolerances:
        declared_tolerances.update(tolerances)

    manifest: dict[str, Any] = {
        "task": task,
        "features": features or [],
        "classes": [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in (classes or [0, 1])],
        "formats": {},
        "tolerances": declared_tolerances,
    }
    if decision_threshold is not None:
        manifest["decision_threshold"] = float(decision_threshold)
    if calibration_method is not None:
        manifest["calibration_method"] = str(calibration_method)
    if calibration_metrics is not None:
        manifest["calibration_metrics"] = calibration_metrics
    if reliability_curves is not None:
        manifest["reliability_curves"] = reliability_curves

    # 1. Native Joblib Export
    if "joblib" in formats:
        joblib_path = out_dir / "frozen_model.joblib"
        joblib.dump(model, joblib_path)
        manifest["formats"]["joblib"] = {
            "file": "frozen_model.joblib",
            "bytes": joblib_path.stat().st_size,
            "sha256": _file_sha256(joblib_path),
        }

    # 2. SafeTensors Zero-Pickle Tensor Export
    if "safetensors" in formats:
        tensors: dict[str, np.ndarray] = {}
        model_type = type(model).__name__

        # Extract underlying estimator if pipeline
        est = model
        scaler = None
        if hasattr(model, "named_steps"):
            scaler = model.named_steps.get("scaler") or model.named_steps.get("standardscaler")
            est = model.named_steps.get("model") or model.named_steps.get("classifier") or model.named_steps.get("regressor") or est

        if scaler is not None and hasattr(scaler, "mean_") and scaler.mean_ is not None:
            tensors["scaler_mean"] = np.asarray(scaler.mean_, dtype=np.float32)
            tensors["scaler_scale"] = np.asarray(scaler.scale_, dtype=np.float32)

        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_, dtype=np.float32)
            tensors["coef"] = coef
        if hasattr(est, "intercept_"):
            intercept = np.asarray(est.intercept_, dtype=np.float32)
            tensors["intercept"] = intercept

        if "coef" in tensors:
            safetensors_path = out_dir / "weights.safetensors"
            save_file(tensors, str(safetensors_path))

            meta_data = {
                "model_type": model_type,
                "task": task,
                "features": features or [],
                "classes": manifest["classes"],
                "has_scaler": scaler is not None,
                "tensors": list(tensors.keys()),
                "decision_threshold": decision_threshold if decision_threshold is not None else 0.5,
            }
            (out_dir / "safetensors_meta.json").write_text(json.dumps(meta_data, indent=2), encoding="utf-8")

            manifest["formats"]["safetensors"] = {
                "file": "weights.safetensors",
                "bytes": safetensors_path.stat().st_size,
                "sha256": _file_sha256(safetensors_path),
                "meta": "safetensors_meta.json",
                "meta_sha256": _file_sha256(out_dir / "safetensors_meta.json"),
                "zero_pickle": True,
            }

    # 3. Model schema contract serialization
    model_schema = {
        "format": "mlrescurix.schema.v2",
        "task": task,
        "model_type": type(model).__name__,
        "features": features or [],
        "classes": manifest["classes"] if task != "regression" else None,
        "decision_threshold": decision_threshold if decision_threshold is not None else 0.5,
        "calibration_method": calibration_method or "none",
        "calibration_metrics": calibration_metrics or {},
        "reliability_curves": reliability_curves or {},
    }
    (out_dir / "model_schema.json").write_text(json.dumps(model_schema, indent=2), encoding="utf-8")

    # 4. Save export manifest and documented tolerances
    (out_dir / "tolerances.json").write_text(json.dumps(declared_tolerances, indent=2), encoding="utf-8")
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_export_parity(
    model: Any,
    export_dir: str | Path,
    test_inputs: list[dict[str, Any]],
    task: str = "binary_classification",
) -> dict[str, Any]:
    """Verify exported formats against native outputs; only load trusted joblib files."""
    root = Path(export_dir).resolve()
    meta_path = root / "export_manifest.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Export manifest missing at {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not test_inputs:
        raise ValueError("Parity verification requires at least one input")
    if meta.get("task") != task:
        raise ValueError("Parity task differs from export manifest")
    formats = meta.get("formats")
    if not isinstance(formats, dict) or not formats or set(formats) - {"joblib", "safetensors"}:
        raise ValueError("No verifiable exported formats or unsupported format")
    # These hashes detect corruption relative to this manifest. A caller must
    # separately bind the manifest itself to a trusted controller receipt.
    for kind, entry in formats.items():
        if not isinstance(entry, dict):
            raise ValueError("Invalid export format entry")
        expected_name = "frozen_model.joblib" if kind == "joblib" else "weights.safetensors"
        if entry.get("file") != expected_name:
            raise ValueError("Invalid export artifact path")
        paths = [(expected_name, entry.get("sha256"))]
        if kind == "safetensors":
            if entry.get("meta") != "safetensors_meta.json":
                raise ValueError("Invalid SafeTensors metadata path")
            paths.append(("safetensors_meta.json", entry.get("meta_sha256")))
        for name, expected_hash in paths:
            path = root / name
            if (path.is_symlink() or not path.is_file() or not isinstance(expected_hash, str)
                    or _file_sha256(path) != expected_hash):
                raise ValueError(f"Export artifact integrity mismatch: {name}")
    tolerances = meta.get("tolerances", DEFAULT_TOLERANCES.get(task, {}))
    results: dict[str, Any] = {"verified": True, "formats": {}}

    # Compute baseline predictions from native model
    import pandas as pd
    features = meta.get("features", [])
    if features:
        expected_features = set(features)
        if len(expected_features) != len(features) or any(
            not isinstance(row, dict) or set(row) != expected_features for row in test_inputs
        ):
            raise ValueError("Parity inputs must match the declared features exactly")
    df = pd.DataFrame(test_inputs)
    if features:
        df = df.reindex(columns=features)

    is_classif = task in ("binary_classification", "multiclass_classification")
    native_pred = model.predict(df).tolist()
    native_prob = model.predict_proba(df).tolist() if is_classif and hasattr(model, "predict_proba") else None
    if not is_classif and not np.isfinite(np.asarray(native_pred, dtype=float)).all():
        raise ValueError("Native predictions must be finite for parity verification")
    if native_prob is not None and not np.isfinite(np.asarray(native_prob, dtype=float)).all():
        raise ValueError("Native probabilities must be finite for parity verification")

    # Joblib uses pickle and may execute code while loading. This function is
    # only for artifacts from a trusted export, not arbitrary worker uploads.
    if "joblib" in meta["formats"]:
        import joblib
        joblib_file = meta["formats"]["joblib"].get("file")
        if joblib_file != "frozen_model.joblib" or (root / joblib_file).is_symlink():
            raise ValueError("Invalid joblib artifact path")
        restored = joblib.load(root / joblib_file)
        restored_pred = np.asarray(restored.predict(df))
        native_pred_arr = np.asarray(native_pred)
        if restored_pred.shape != native_pred_arr.shape:
            raise ValueError("Joblib prediction shape mismatch")
        if is_classif:
            if not np.array_equal(restored_pred, native_pred_arr):
                raise ValueError("Joblib class parity violation")
        elif not np.allclose(restored_pred, native_pred_arr,
                             rtol=tolerances.get("prediction_rtol", 1e-5),
                             atol=tolerances.get("prediction_atol", 1e-6)):
            raise ValueError("Joblib prediction parity violation")
        detail = {"verified": True}
        if native_prob is not None:
            if not hasattr(restored, "predict_proba"):
                raise ValueError("Joblib probabilities missing")
            restored_prob = np.asarray(restored.predict_proba(df), dtype=float)
            native_prob_arr = np.asarray(native_prob, dtype=float)
            if restored_prob.shape != native_prob_arr.shape or not np.allclose(
                restored_prob, native_prob_arr,
                rtol=tolerances.get("probabilities_rtol", 1e-5),
                atol=tolerances.get("probabilities_atol", 1e-7),
            ):
                raise ValueError("Joblib probability parity violation")
            detail["max_prob_diff"] = float(np.max(np.abs(restored_prob - native_prob_arr)))
        results["formats"]["joblib"] = detail

    # Verify SafeTensors if exported
    if "safetensors" in meta["formats"]:
        predictor = SafeTensorsPredictor(root)
        st_res = predictor.predict(test_inputs)
        st_pred = st_res["prediction"]

        if is_classif:
            # Classes must match exactly
            if st_pred != native_pred:
                raise ValueError(f"SafeTensors class mismatch: {st_pred} != {native_pred}")

            if native_prob is not None and "probabilities" in st_res:
                st_prob_arr = np.asarray(st_res["probabilities"], dtype=float)
                nat_prob_arr = np.asarray(native_prob, dtype=float)
                diff = np.max(np.abs(st_prob_arr - nat_prob_arr))
                rtol = tolerances.get("probabilities_rtol", 1e-5)
                atol = tolerances.get("probabilities_atol", 1e-7)
                if not np.allclose(st_prob_arr, nat_prob_arr, rtol=rtol, atol=atol):
                    raise ValueError(
                        f"SafeTensors probability parity violation: max diff {diff:.2e} exceeds rtol={rtol}, atol={atol}"
                    )
                results["formats"]["safetensors"] = {"verified": True, "max_prob_diff": float(diff)}
            elif native_prob is not None:
                raise ValueError("SafeTensors probabilities missing")
            else:
                results["formats"]["safetensors"] = {"verified": True}
        else:
            # Regression continuous comparison
            st_pred_arr = np.asarray(st_pred, dtype=float)
            nat_pred_arr = np.asarray(native_pred, dtype=float)
            diff = np.max(np.abs(st_pred_arr - nat_pred_arr))
            rtol = tolerances.get("prediction_rtol", 1e-5)
            atol = tolerances.get("prediction_atol", 1e-6)
            if not np.allclose(st_pred_arr, nat_pred_arr, rtol=rtol, atol=atol):
                raise ValueError(
                    f"SafeTensors prediction parity violation: max diff {diff:.2e} exceeds rtol={rtol}, atol={atol}"
                )
            results["formats"]["safetensors"] = {"verified": True, "max_pred_diff": float(diff)}

    return results
