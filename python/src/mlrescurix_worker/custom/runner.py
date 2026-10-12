import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from .contract import CustomEstimator


def load_custom_estimator(entrypoint_path: str, class_name: str = "Estimator") -> CustomEstimator:
    """Dynamically load custom estimator class from an entrypoint script."""
    path = Path(entrypoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Custom estimator entrypoint not found: {entrypoint_path}")

    spec = importlib.util.spec_from_file_location("custom_candidate_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {entrypoint_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_candidate_module"] = module
    spec.loader.exec_module(module)

    if not hasattr(module, class_name):
        # Look for any CustomEstimator subclass if default name not found
        for attr_name in dir(module):
            val = getattr(module, attr_name)
            if isinstance(val, type) and issubclass(val, CustomEstimator) and val is not CustomEstimator:
                return val()
        raise AttributeError(f"Module {entrypoint_path} does not define {class_name} or a CustomEstimator subclass")

    cls = getattr(module, class_name)
    return cls()


def train_and_predict_isolated(
    entrypoint: str,
    train_csv: str,
    val_csv: str,
    target_column: str,
    output_predictions_path: str,
    output_model_path: Optional[str] = None,
) -> dict:
    """
    Train custom candidate estimator on authorized training data and generate
    predictions for validation features. The validation split features are strictly label-free.
    """
    df_train = pd.read_csv(train_csv)
    if target_column not in df_train.columns:
        raise ValueError(f"Target column {target_column} missing from training CSV")

    X_train = df_train.drop(columns=[target_column])
    y_train = df_train[target_column]

    df_val = pd.read_csv(val_csv)
    # Validation container is label-free
    if target_column in df_val.columns:
        X_val = df_val.drop(columns=[target_column])
    else:
        X_val = df_val.copy()

    row_ids = None
    for id_col in ["row_id", "row_ids", "id", "_id"]:
        if id_col in X_val.columns:
            row_ids = X_val[id_col].tolist()
            X_val = X_val.drop(columns=[id_col])
            break
    if row_ids is None:
        if df_val.index is not None and not isinstance(df_val.index, pd.RangeIndex):
            row_ids = df_val.index.tolist()
        else:
            row_ids = list(range(len(df_val)))

    estimator = load_custom_estimator(entrypoint)
    estimator.fit(X_train, y_train)

    preds = estimator.predict(X_val)
    if not isinstance(preds, np.ndarray):
        preds = np.asarray(preds)

    if np.issubdtype(preds.dtype, np.number):
        if not np.isfinite(preds).all():
            raise ValueError("Predictions contain non-finite values (NaN or Inf)")

    probs = None
    classes = None
    try:
        p = estimator.predict_proba(X_val)
        if p is not None:
            p_arr = np.asarray(p)
            if not np.isfinite(p_arr).all():
                raise ValueError("Probabilities contain non-finite values (NaN or Inf)")
            if (p_arr < -1e-6).any() or (p_arr > 1.0 + 1e-6).any():
                raise ValueError("Probabilities out of bounds [0, 1]")
            probs = p_arr.tolist()
            classes_attr = getattr(estimator, "classes_", None)
            if classes_attr is None and hasattr(estimator, "model"):
                classes_attr = getattr(estimator.model, "classes_", None)
            if classes_attr is None:
                unique_y = sorted(y_train.dropna().unique().tolist())
                if len(unique_y) == p_arr.shape[1]:
                    classes_attr = unique_y
            if classes_attr is not None:
                classes = [
                    int(c) if isinstance(c, (int, np.integer))
                    else (str(c) if isinstance(c, (str, np.str_)) else float(c))
                    for c in classes_attr
                ]
    except Exception as e:
        if isinstance(e, ValueError):
            raise

    out_payload = {
        "predictions": preds.tolist() if isinstance(preds, np.ndarray) else list(preds),
        "probabilities": probs,
        "classes": classes,
        "row_ids": row_ids,
        "n_samples": len(preds),
    }

    out_path = Path(output_predictions_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f)

    if output_model_path:
        import joblib
        model_path = Path(output_model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(estimator, model_path)

    return out_payload


def main():
    parser = argparse.ArgumentParser(description="Isolated custom candidate trainer & predictor")
    parser.add_argument("--entrypoint", required=True, help="Path to Python script with CustomEstimator")
    parser.add_argument("--train-csv", required=True, help="Path to training CSV with labels")
    parser.add_argument("--val-csv", required=True, help="Path to validation CSV (features only)")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--output-predictions", required=True, help="Path to output predictions JSON")
    parser.add_argument("--output-model", default=None, help="Path to save frozen model artifact")
    args = parser.parse_args()

    train_and_predict_isolated(
        entrypoint=args.entrypoint,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        target_column=args.target,
        output_predictions_path=args.output_predictions,
        output_model_path=args.output_model,
    )
    print(f"Predictions written to {args.output_predictions}")


if __name__ == "__main__":
    main()
