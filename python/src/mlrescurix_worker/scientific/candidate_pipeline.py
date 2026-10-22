"""Versioned, raw-frame candidate pipelines with fold-local fitted preprocessing.

Generated scripts produce advisory evidence; only this evaluated specification can
be promoted to a model bundle. No optional estimator is silently substituted.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.metaestimators import available_if
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, log_loss, r2_score, mean_absolute_error)

from .model_zoo import create_estimator, normalize_family_name


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def score_metric(name, y, prediction, probability=None, classes=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if name == "accuracy": return float(accuracy_score(y, prediction))
    if name == "balanced_accuracy": return float(balanced_accuracy_score(y, prediction))
    if name in ("roc_auc", "average_precision"):
        if probability is None or probability.shape[1] != 2:
            raise ValueError(f"{name} requires binary probabilities")
        fn = roc_auc_score if name == "roc_auc" else average_precision_score
        return float(fn((y == classes[1]).astype(int), probability[:, 1]))
    if name in ("log_loss", "multiclass_log_loss"):
        return float(log_loss(y, probability, labels=classes))
    if name == "r2": return float(r2_score(y, prediction))
    if name == "mae": return float(mean_absolute_error(y, prediction))
    if name == "rmse": return float(np.sqrt(np.mean((y - prediction) ** 2)))
    if name == "rmsle":
        if np.any(y < 0): raise ValueError("RMSLE requires nonnegative targets")
        return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(np.maximum(0, prediction))) ** 2)))
    if name == "rmse_log_price":
        if np.any(y <= 0) or np.any(prediction <= 0):
            raise ValueError("log-price RMSE requires strictly positive targets and predictions")
        return float(np.sqrt(np.mean((np.log(y) - np.log(prediction)) ** 2)))
    raise ValueError(f"Unsupported metric: {name}")


class CandidatePipeline:
    """Serializable ensemble; every fold owns a separate instance and encoders."""
    def __init__(self, spec, target_column, task_type):
        self.spec = copy.deepcopy(spec)
        self.candidate_identity = identity(spec)
        self.target_column = target_column
        self.task_type = task_type
        self.is_classification = "classification" in task_type or task_type in ("binary", "multiclass")

    def _frame(self, frame, fit=False):
        if not isinstance(frame, pd.DataFrame): frame = pd.DataFrame(frame)
        x = frame.drop(columns=[self.target_column], errors="ignore").copy()
        if fit: self.input_columns_ = list(x.columns)
        absent = set(self.input_columns_) - set(x.columns)
        if absent: raise ValueError(f"Missing raw columns: {sorted(absent)}")
        x = x[self.input_columns_]
        for tr in self.spec.get("transforms", []):
            action, p = tr["action"], tr["parameters"]
            if action == "ADD_MNAR_INDICATOR":
                for c in p["columns"]: x[f"{c}__missing"] = x[c].isna().astype(float)
            elif action == "DELETE_FEATURE": x = x.drop(columns=p["columns"])
            elif action == "APPLY_SKEW_TRANSFORM":
                for c in p["columns"]:
                    a = pd.to_numeric(x[c], errors="raise")
                    if p.get("method", "log1p") == "log1p":
                        if (a.dropna() <= -1).any(): raise ValueError("log1p domain requires values > -1; select signed_log1p explicitly")
                        x[c] = np.log1p(a)
                    elif p["method"] == "signed_log1p": x[c] = np.sign(a) * np.log1p(np.abs(a))
                    elif p["method"] == "sqrt":
                        if (a.dropna() < 0).any(): raise ValueError("sqrt requires nonnegative values")
                        x[c] = np.sqrt(a)
                    else: raise ValueError("Unsupported skew method")
            elif action == "ADD_INTERACTION":
                for a, b in p["feature_pairs"]:
                    for op in p.get("operations", ["multiply"]):
                        aa, bb = pd.to_numeric(x[a]), pd.to_numeric(x[b])
                        if op == "multiply": value = aa * bb
                        elif op == "add": value = aa + bb
                        elif op == "subtract": value = aa - bb
                        elif op == "divide": value = aa / bb.replace(0, np.nan)
                        else: raise ValueError("Unsupported interaction")
                        x[f"{a}__{op}__{b}"] = value
            elif action == "ENCODE_HIGH_CARD_CAT":
                if p.get("strategy", "frequency") not in ("frequency", "target_or_frequency_safe_cv"):
                    raise ValueError("Only fold-local frequency encoding supported")
                for c in p["columns"]:
                    key = c + "__frequency"
                    if fit: self.frequency_maps_[key] = x[c].astype("string").value_counts(normalize=True).to_dict()
                    x[c] = x[c].astype("string").map(self.frequency_maps_[key]).fillna(0).astype(float)
            else: raise ValueError(f"Unsupported transform {action}")
        return x.replace([np.inf, -np.inf], np.nan)

    def _matrix(self, frame, fit=False):
        x = self._frame(frame, fit)
        if fit:
            self.numeric_ = list(x.select_dtypes(include=[np.number, "bool"]).columns)
            self.categorical_ = [c for c in x if c not in self.numeric_]
            # _matrix concatenates numeric then categorical blocks; audit labels
            # must follow the same order as estimator coefficients/importances.
            self.feature_names_ = self.numeric_ + self.categorical_
            if not self.feature_names_: raise ValueError("Candidate has no features")
            self.imputer_ = SimpleImputer(strategy="median", keep_empty_features=True)
            self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        arrays = []
        if self.numeric_:
            n = x[self.numeric_].astype(float)
            arrays.append(self.imputer_.fit_transform(n) if fit else self.imputer_.transform(n))
        if self.categorical_:
            c = x[self.categorical_].fillna("__MISSING__").astype(str)
            arrays.append(self.encoder_.fit_transform(c) if fit else self.encoder_.transform(c))
        return np.hstack(arrays)

    def fit(self, frame, y=None):
        self.frequency_maps_ = {}
        y = np.asarray(frame[self.target_column] if y is None else y)
        self.target_transform_ = self.spec.get("target_transform", "none")
        if self.is_classification and self.target_transform_ != "none": raise ValueError("Target transform only supports regression")
        if self.target_transform_ == "log":
            if np.any(y <= 0): raise ValueError("Log target requires positive labels")
            y = np.log(y)
        elif self.target_transform_ == "log1p":
            if np.any(y < 0): raise ValueError("Log1p target requires nonnegative labels")
            y = np.log1p(y)
        x = self._matrix(frame, fit=True)
        self.classes_ = np.unique(y) if self.is_classification else None
        self.estimators_ = []
        for family in self.spec["families"]:
            est = create_estimator(family, task_type=self.task_type, random_state=self.spec.get("seed", 42),
                **self.spec.get("model_parameters", {}).get(family, {}))
            if getattr(est, "_is_fallback", False): raise ValueError(f"Unavailable model family: {family}; fallback refused")
            options = self.spec.get("model_parameters", {}).get(family, {})
            available = est.get_params(deep=True)
            for name, value in options.items():
                key = f"model__{name}" if family == "penalized_linear" else name
                if key not in available: raise ValueError(f"Hyperparameter {name} is unsupported for this {family} task")
                # The factory selects solver/type; apply every requested supported value explicitly.
                est.set_params(**{key: value})
            est.fit(x, y)
            self.estimators_.append(est)
        self.is_fitted_ = True
        return self

    @available_if(lambda self: self.is_classification)
    def predict_proba(self, frame):
        if not self.is_classification: raise ValueError("Regression has no probabilities")
        x = self._matrix(frame)
        probs = []
        for est in self.estimators_:
            raw = est.predict_proba(x)
            aligned = np.zeros((len(x), len(self.classes_)))
            for i, label in enumerate(est.classes_): aligned[:, np.flatnonzero(self.classes_ == label)[0]] = raw[:, i]
            probs.append(aligned)
        return np.average(probs, axis=0, weights=self.spec["weights"])

    def predict(self, frame):
        if self.is_classification: return self.classes_[np.argmax(self.predict_proba(frame), axis=1)]
        x = self._matrix(frame)
        prediction = np.average([est.predict(x) for est in self.estimators_], axis=0, weights=self.spec["weights"])
        if self.target_transform_ == "log": prediction = np.exp(prediction)
        if self.target_transform_ == "log1p": prediction = np.expm1(prediction)
        return prediction


def validate_spec(spec):
    if set(spec) - {"families", "weights", "transforms", "seed", "model_parameters", "target_transform"}:
        raise ValueError("Unknown candidate specification fields")
    if spec.get("target_transform", "none") not in ("none", "log", "log1p"):
        raise ValueError("Unsupported target transform")
    if not isinstance(spec.get("transforms", []), list) or len(spec.get("transforms", [])) > 12:
        raise ValueError("Transform limit is 12")
    families = spec.get("families")
    if not isinstance(families, list) or not 1 <= len(families) <= 6 or len(set(families)) != len(families):
        raise ValueError("Require 1–6 distinct model families")
    supported = {"hist_gbdt", "penalized_linear", "lgbm", "xgboost", "catboost", "neural_tabular"}
    if any(f not in supported or normalize_family_name(f) != f for f in families): raise ValueError("Unsupported model family")
    allowed_params = {
        "penalized_linear": {"alpha", "C", "penalty", "l1_ratio", "max_iter"},
        "hist_gbdt": {"max_iter", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization"},
        "lgbm": {"n_estimators", "learning_rate", "max_depth", "min_child_samples", "reg_alpha", "reg_lambda"},
        "xgboost": {"n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"},
        "catboost": {"iterations", "learning_rate", "depth", "l2_leaf_reg"},
        "neural_tabular": {"hidden_dim", "num_blocks", "dropout", "lr", "epochs", "batch_size"}}
    options = spec.get("model_parameters", {})
    if not isinstance(options, dict) or set(options) - set(families): raise ValueError("model_parameters must match selected families")
    for family, params in options.items():
        if not isinstance(params, dict) or set(params) - allowed_params[family]: raise ValueError(f"Unsupported hyperparameter for {family}")
        for name, value in params.items():
            if name != "penalty" and (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)):
                raise ValueError("Hyperparameters must be finite numeric scalars")
    weights = spec.get("weights", [])
    if len(weights) != len(families) or not all(not isinstance(w, bool) and isinstance(w, (float, int)) and np.isfinite(w) and w > 0 for w in weights):
        raise ValueError("Weights must be finite positive values matching families")


def extract_measured_importances(estimator: Any, n_features: int) -> Optional[np.ndarray]:
    """Extract measured non-negative feature importances from a fitted estimator, or None."""
    # Unwrap sklearn Pipeline if present
    if hasattr(estimator, "named_steps"):
        for step_name in ("model", "estimator", "clf", "regressor"):
            if step_name in estimator.named_steps:
                sub = extract_measured_importances(estimator.named_steps[step_name], n_features)
                if sub is not None:
                    return sub
        if hasattr(estimator, "steps") and estimator.steps:
            sub = extract_measured_importances(estimator.steps[-1][1], n_features)
            if sub is not None:
                return sub

    if hasattr(estimator, "feature_importances_"):
        raw_attr = estimator.feature_importances_
        if raw_attr is not None:
            imps = np.asarray(raw_attr, dtype=float)
            if len(imps) == n_features:
                return np.maximum(0.0, imps)
            elif len(imps) > n_features:
                return np.maximum(0.0, imps[:n_features])
    if hasattr(estimator, "coef_"):
        coefs = np.asarray(estimator.coef_, dtype=float)
        if coefs.ndim > 1:
            raw = np.mean(np.abs(coefs), axis=0)
        else:
            raw = np.abs(coefs)
        if len(raw) >= n_features:
            return np.maximum(0.0, raw[:n_features])
    return None


def extract_split_importances(estimator: Any, n_features: int) -> Optional[np.ndarray]:
    """Return native tree split importances only, suitable for the critic gain gate."""
    if hasattr(estimator, "steps"):
        for _, step in reversed(estimator.steps):
            result = extract_split_importances(step, n_features)
            if result is not None:
                return result
    # Neural wrappers also expose feature_importances_ derived from input
    # weight norms. That attribute alone is not evidence of native tree gain.
    if not type(estimator).__module__.startswith((
        "lightgbm.", "xgboost.", "catboost.", "sklearn.tree.", "sklearn.ensemble."
    )):
        return None
    raw = getattr(estimator, "feature_importances_", None)
    if raw is None:
        return None
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or len(values) != n_features or not np.isfinite(values).all() or np.any(values < 0):
        return None
    total = float(values.sum())
    return values / total if total > 1e-12 else None


def evaluate_candidate(spec, frame, target, task_type, folds, metric_name, topology=None):
    """Evaluate exact spec on caller-owned folds; never train on validation rows."""
    validate_spec(spec)
    y = frame[target].to_numpy()
    fold_receipts = []
    predictions, observed, probabilities = [], [], []
    validation_frames = []
    fold_imps = []
    fold_split_imps = []
    is_panel = (
        task_type in ("panel_forecasting", "forecasting")
        or (isinstance(topology, dict) and (
            topology.get("forecast_horizon")
            or topology.get("forecast_horizon_dates")
            or topology.get("benchmark_id") == "K5"
            or topology.get("panel_columns")
        ))
    )
    time_col = None
    if is_panel and isinstance(topology, dict):
        time_col = topology.get("temporal_col") or topology.get("time_column") or topology.get("time")
    panel_entities = []
    expected_horizon = None
    temporal_values = None
    if is_panel:
        if not time_col or time_col not in frame.columns:
            raise ValueError("Panel forecasting requires an authoritative temporal column")
        known = topology.get("known_at_forecast_columns") if isinstance(topology, dict) else None
        if not isinstance(known, list) or not known or any(not isinstance(c, str) for c in known):
            raise ValueError("Panel forecasting requires an explicit known_at_forecast_columns allowlist")
        if any(c not in frame.columns for c in known):
            raise ValueError("known_at_forecast_columns contains columns absent from development data")
        panel_entities = topology.get("entity_columns", []) if isinstance(topology, dict) else []
        if not isinstance(panel_entities, list) or any(c not in frame.columns for c in panel_entities):
            raise ValueError("Panel entity_columns must name existing development columns")
        permitted = set(known) | {time_col} | set(panel_entities)
        unknown_features = set(frame.columns) - {target} - permitted
        if unknown_features:
            raise ValueError(f"Panel predictors absent from known-at-forecast allowlist: {sorted(unknown_features)}")
        expected_horizon = (topology.get("forecast_horizon_dates") or topology.get("forecast_horizon"))
        if isinstance(expected_horizon, bool) or not isinstance(expected_horizon, int) or expected_horizon <= 0:
            raise ValueError("Panel forecasting requires a positive forecast_horizon_dates/horizon")
        if panel_entities:
            if frame.duplicated(panel_entities + [time_col]).any():
                raise ValueError("Panel data must have at most one row per entity and timestamp")
        temporal_values = frame[time_col]
        if not pd.api.types.is_numeric_dtype(temporal_values):
            temporal_values = pd.to_datetime(temporal_values, errors="raise")
        if temporal_values.isna().any():
            raise ValueError("Panel temporal column contains missing values")

    secondary = metric_name if metric_name in ("roc_auc", "average_precision", "rmsle", "rmse_log_price") else (
        "balanced_accuracy" if "classification" in task_type or task_type in ("binary", "multiclass")
        else "r2"
    )
    for i, (train, valid) in enumerate(folds):
        train_df, valid_df = frame.iloc[train], frame.iloc[valid]
        if is_panel and time_col and time_col in frame.columns:
            tr_times = temporal_values.iloc[train]
            va_times = temporal_values.iloc[valid]
            if max(tr_times) >= min(va_times):
                raise ValueError(
                    f"Temporal lookahead leakage detected in fold {i}: "
                    f"max(train)={max(tr_times)} >= min(valid)={min(va_times)}"
                )
            validation_dates = len(pd.unique(va_times))
            if validation_dates != expected_horizon:
                raise ValueError(
                    f"Panel fold {i} has {validation_dates} validation dates; expected horizon {expected_horizon}"
                )

        pipeline = CandidatePipeline(spec, target, task_type).fit(train_df, y[train])
        tr_pred, va_pred = pipeline.predict(train_df), pipeline.predict(valid_df)
        if metric_name in ("rmsle", "rmse_log_price"):
            # Log-price RMSE is defined on ln(price), so valid sub-unit prices
            # must remain sub-unit. Only protect against zero/negative outputs.
            floor = np.finfo(float).tiny if metric_name == "rmse_log_price" else 0.0
            tr_pred = np.maximum(floor, tr_pred)
            va_pred = np.maximum(floor, va_pred)
        tr_prob = pipeline.predict_proba(train_df) if pipeline.is_classification else None
        va_prob = pipeline.predict_proba(valid_df) if pipeline.is_classification else None
        train_score = score_metric(metric_name, y[train], tr_pred, tr_prob, pipeline.classes_)
        valid_score = score_metric(metric_name, y[valid], va_pred, va_prob, pipeline.classes_)

        # Extract fold-level measured feature importances
        fold_est_imps = []
        for est in pipeline.estimators_:
            imp = extract_measured_importances(est, len(pipeline.feature_names_))
            if imp is not None:
                fold_est_imps.append(imp)
        if fold_est_imps:
            # Normalize before combining because estimator scales differ.
            normalized = [v / v.sum() for v in fold_est_imps if float(v.sum()) > 1e-12]
            if normalized:
                fold_imps.append(np.mean(normalized, axis=0))
        split_imps = [extract_split_importances(est, len(pipeline.feature_names_)) for est in pipeline.estimators_]
        split_imps = [v for v in split_imps if v is not None]
        if split_imps:
            fold_split_imps.append(np.mean(split_imps, axis=0))

        receipt = {
            "fold": i,
            "train_rows": len(train),
            "validation_rows": len(valid),
            "train_score": train_score,
            "validation_score": valid_score,
            "critic_train_score": score_metric(secondary, y[train], tr_pred, tr_prob, getattr(pipeline, "classes_", None)),
            "critic_validation_score": score_metric(secondary, y[valid], va_pred, va_prob, getattr(pipeline, "classes_", None)),
            "features": pipeline.feature_names_,
            "split_hash": identity({"train": list(map(int, train)), "valid": list(map(int, valid))}),
        }
        if is_panel and time_col and time_col in frame.columns:
            receipt["train_end"] = str(max(tr_times))
            receipt["val_start"] = str(min(va_times))
            receipt["val_end"] = str(max(va_times))
            receipt["horizon_steps"] = len(set(va_times))
        fold_receipts.append(receipt)
        predictions.extend(va_pred.tolist()); observed.extend(y[valid].tolist())
        validation_frames.append(valid_df.drop(columns=[target]).copy())
        if va_prob is not None: probabilities.extend(va_prob.tolist())

    scores = np.array([f["validation_score"] for f in fold_receipts])
    if not np.isfinite(scores).all(): raise ValueError("Nonfinite fold metric")

    # Feature dominance diagnostics
    measured_importances = None
    dominant_feature = None
    dominant_gain_ratio = None
    feature_dominance_status = "unsupported_by_estimator"
    if fold_imps and len(fold_imps) == len(folds):
        mean_imps = np.mean(fold_imps, axis=0)
        tot = float(np.sum(mean_imps))
        if tot > 1e-12:
            norm_imps = mean_imps / tot
            measured_importances = norm_imps.tolist()
            max_idx = int(np.argmax(norm_imps))
            dominant_feature = fold_receipts[0]["features"][max_idx]
            dominant_gain_ratio = float(norm_imps[max_idx])
            feature_dominance_status = "measured"

    split_gain_importances = None
    split_gain_dominant_feature = None
    split_gain_dominant_ratio = None
    if fold_split_imps and len(fold_split_imps) == len(folds):
        mean_gains = np.mean(fold_split_imps, axis=0)
        total_gain = float(mean_gains.sum())
        if total_gain > 1e-12:
            split_gain_importances = (mean_gains / total_gain).tolist()
            split_idx = int(np.argmax(mean_gains))
            split_gain_dominant_feature = fold_receipts[0]["features"][split_idx]
            split_gain_dominant_ratio = float(mean_gains[split_idx] / total_gain)

    limitations = ["Ordinal categorical encoding; specialized text/image/audio models unsupported"]
    if measured_importances is None:
        limitations.insert(0, "Feature gain audit unsupported by estimator; no fabricated importances")

    # Simple measured slices expose missingness errors without fitting on validation labels.
    validation_frame = pd.concat(validation_frames, ignore_index=True)
    is_classification = "classification" in task_type or task_type in ("binary", "multiclass")
    error = (np.asarray(predictions) != np.asarray(observed)).astype(float) if is_classification else np.abs(np.asarray(predictions) - np.asarray(observed))
    slices = []
    for col in validation_frame.columns:
        missing = validation_frame[col].isna().to_numpy()
        for label, mask in (("missing", missing), ("present", ~missing)):
            if int(mask.sum()) >= 5:
                slices.append({"column": col, "slice": label, "support": int(mask.sum()), "mean_error": float(error[mask].mean())})
    slices.sort(key=lambda item: item["mean_error"], reverse=True)

    # Mean of pinned fold metrics, not pooled rows (rolling folds may overlap).
    return {
        "cv_score": float(scores.mean()),
        "cv_scores": scores.tolist(),
        "fold_receipts": fold_receipts,
        "train_score": float(np.mean([f["train_score"] for f in fold_receipts])),
        "critic_train_score": float(np.mean([f["critic_train_score"] for f in fold_receipts])),
        "critic_validation_score": float(np.mean([f["critic_validation_score"] for f in fold_receipts])),
        "feature_names": fold_receipts[0]["features"],
        "candidate_identity": identity(spec),
        "metric_name": metric_name,
        "feature_importances": measured_importances,
        "dominant_feature": dominant_feature,
        "dominant_gain_ratio": dominant_gain_ratio,
        "feature_dominance_status": feature_dominance_status,
        "feature_importance_source": "normalized_estimator_importance" if measured_importances is not None else "unsupported",
        "split_gain_importances": split_gain_importances,
        "split_gain_dominant_feature": split_gain_dominant_feature,
        "split_gain_dominant_ratio": split_gain_dominant_ratio,
        "is_panel_forecasting": is_panel,
        "error_slices": slices[:10],
        "limitations": limitations,
    }
