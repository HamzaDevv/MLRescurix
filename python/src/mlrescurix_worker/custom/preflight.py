import io
import tempfile
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

from .contract import CustomEstimator


def run_preflight_checks(
    estimator: CustomEstimator,
    sample_X: pd.DataFrame,
    sample_y: pd.Series,
    is_classification: bool = True,
) -> Dict[str, Any]:
    """
    Run comprehensive contract and stability preflight checks on a custom estimator:
    1. Shape and type contract on training fit.
    2. Prediction array shape and length equality.
    3. Probability sums to 1.0 (if classification with predict_proba).
    4. Single-row inference shape.
    5. Joblib serialization round-trip fidelity.
    6. NaN stability check.
    """
    checks = {}
    try:
        # 1. Fit check
        estimator.fit(sample_X, sample_y)
        checks["fit_succeeded"] = True

        # 2. Predict shape check
        preds = estimator.predict(sample_X)
        if not isinstance(preds, np.ndarray):
            return {"passed": False, "checks": checks, "error": f"predict returned {type(preds)}, expected np.ndarray"}
        if len(preds) != len(sample_X):
            return {"passed": False, "checks": checks, "error": f"predict length {len(preds)} != X length {len(sample_X)}"}
        checks["predict_shape_match"] = True

        # 3. Probabilities check
        if is_classification:
            probs = estimator.predict_proba(sample_X)
            if probs is not None:
                if not isinstance(probs, np.ndarray):
                    return {"passed": False, "checks": checks, "error": "predict_proba returned non-numpy array"}
                if probs.ndim != 2 or probs.shape[0] != len(sample_X):
                    return {"passed": False, "checks": checks, "error": f"predict_proba invalid shape: {probs.shape}"}
                row_sums = probs.sum(axis=1)
                if not np.allclose(row_sums, 1.0, atol=1e-3):
                    return {"passed": False, "checks": checks, "error": "predict_proba rows do not sum to 1.0"}
                checks["predict_proba_valid"] = True
            else:
                checks["predict_proba_valid"] = "not_provided"

        # 4. Single-row inference check
        single_X = sample_X.iloc[0:1]
        single_pred = estimator.predict(single_X)
        if len(single_pred) != 1:
            return {"passed": False, "checks": checks, "error": f"single-row predict returned length {len(single_pred)}"}
        checks["single_row_inference"] = True

        # 5. Joblib round-trip serialization
        buf = io.BytesIO()
        joblib.dump(estimator, buf)
        buf.seek(0)
        reloaded: CustomEstimator = joblib.load(buf)
        reloaded_preds = reloaded.predict(sample_X)
        if not np.array_equal(preds, reloaded_preds):
            return {"passed": False, "checks": checks, "error": "joblib roundtrip prediction mismatch"}
        checks["joblib_roundtrip_valid"] = True

        # 6. NaN resilience check
        nan_X = sample_X.copy()
        first_col = nan_X.columns[0]
        nan_X.loc[nan_X.index[0], first_col] = np.nan
        try:
            nan_pred = estimator.predict(nan_X)
            checks["nan_handling"] = "supported" if len(nan_pred) == len(sample_X) else "failed"
        except Exception:
            checks["nan_handling"] = "raises_exception"

        return {
            "passed": True,
            "checks": checks,
            "error": None,
        }
    except Exception as exc:
        return {
            "passed": False,
            "checks": checks,
            "error": f"preflight exception: {str(exc)}",
        }
