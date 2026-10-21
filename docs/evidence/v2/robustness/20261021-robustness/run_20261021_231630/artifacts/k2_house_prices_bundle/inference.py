"""Standalone certified inference module for MLRescurix candidate cand_apex_k2."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

_DIR = Path(__file__).resolve().parent
_MODEL = None
_SCHEMA = None
_THRESHOLD = 0.5

def _load_model():
    global _MODEL, _SCHEMA, _THRESHOLD
    if _MODEL is None:
        _MODEL = joblib.load(_DIR / "frozen_model.joblib")
        _SCHEMA = json.loads((_DIR / "input_schema.json").read_text(encoding="utf-8"))
        ms_path = _DIR / "model_schema.json"
        if ms_path.is_file():
            try:
                ms = json.loads(ms_path.read_text(encoding="utf-8"))
                _THRESHOLD = float(ms.get("decision_threshold", 0.5))
            except Exception:
                _THRESHOLD = 0.5
    return _MODEL, _SCHEMA, _THRESHOLD

def predict(records):
    """Generate predictions for a list of input records."""
    model, schema, threshold = _load_model()
    features = schema.get("features", [])
    if not isinstance(records, list):
        raise ValueError("records must be a list of dictionaries")
    if not records:
        return {"prediction": []}

    allowed_ids = {"row_id", "id", "index", "ID", "Id", "PassengerId"}.union({'Id'})
    for r in records:
        for f in features:
            if f not in r:
                raise ValueError(f"Missing required feature: '{f}'")
        for k in r:
            if k not in features and k not in allowed_ids:
                raise ValueError(f"Unexpected extra feature: '{k}'")

    res = {}
    try:
        # Check if model handles raw records directly (e.g. FullInferencePipeline)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(records)
            res["probabilities"] = probs.tolist() if hasattr(probs, "tolist") else probs
            res["classes"] = [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in getattr(model, "classes_", [0, 1])]
            preds = model.predict(records)
            res["prediction"] = [int(p) if isinstance(p, (int, np.integer)) else p for p in preds]
        else:
            preds = model.predict(records)
            res["prediction"] = preds.tolist() if hasattr(preds, "tolist") else preds
    except Exception:
        # Fallback for simple estimators expecting numeric matrix only
        df = pd.DataFrame(records)
        num_cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        X = df[num_cols].fillna(0.0) if num_cols else df.fillna(0.0)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            res["probabilities"] = probs.tolist() if hasattr(probs, "tolist") else probs
            res["classes"] = [int(c) if isinstance(c, (int, np.integer)) else str(c) for c in getattr(model, "classes_", [0, 1])]
            if len(res["classes"]) == 2 and threshold is not None:
                p_pos = probs[:, 1] if getattr(probs, "ndim", 1) == 2 else probs
                preds_idx = (np.asarray(p_pos) >= threshold).astype(int)
                res["prediction"] = [res["classes"][i] for i in preds_idx]
            else:
                preds = model.predict(X)
                res["prediction"] = preds.tolist() if hasattr(preds, "tolist") else preds
        else:
            preds = model.predict(X)
            res["prediction"] = preds.tolist() if hasattr(preds, "tolist") else preds
    return res

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inp = json.loads(Path(args.input).read_text(encoding="utf-8"))
    out = predict(inp)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
