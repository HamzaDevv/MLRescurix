import sys
import os
import json
import math
import argparse
from pathlib import Path

REQUIRED_FEATURES = ["x1","x2"]
IS_REGRESSION = True

def load_weights(bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    for fname in ["model_weights.json", "weights.json", "frozen_model.joblib"]:
        p = bundle_dir / fname
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No weights file found in {bundle_dir}")

def get_solution_module(bundle_dir):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    bundle_str = str(bundle_dir)
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)
    try:
        import solution
        return solution
    except Exception:
        return None

def predict_single(row, weights_data, bundle_dir=None):
    if not isinstance(row, dict):
        raise ValueError("Each input row must be a dictionary")

    # Strict schema check: reject unexpected extra columns
    for k in row.keys():
        if k not in REQUIRED_FEATURES:
            raise ValueError(f"Unexpected column '{k}' not in schema")

    # Strict schema check: reject missing columns
    for f in REQUIRED_FEATURES:
        if f not in row:
            raise ValueError(f"Missing required feature '{f}'")

    vec = [float(row[f]) for f in REQUIRED_FEATURES]
    sol = get_solution_module(bundle_dir)
    if sol and hasattr(sol, "basis"):
        try:
            phi = sol.basis(*vec)
        except TypeError:
            phi = sol.basis(vec)
    elif sol and hasattr(sol, "basis_expansion"):
        phi = sol.basis_expansion(vec)
    else:
        phi = [1.0] + vec + [v * v for v in vec] + [math.sin(v) for v in vec]

    w = weights_data["weights"]
    if len(phi) != len(w):
        raise ValueError(f"Basis dimension mismatch: phi has {len(phi)}, weights has {len(w)}")

    val = sum(w[j] * phi[j] for j in range(len(w)))
    if not IS_REGRESSION:
        p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
        pred_val = 1.0 if p >= 0.5 else 0.0
    else:
        pred_val = val
    return round(pred_val, 6)

def predict(input_data, bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    weights_data = load_weights(bundle_dir)
    if isinstance(input_data, dict):
        return {"prediction": predict_single(input_data, weights_data, bundle_dir=bundle_dir)}
    elif isinstance(input_data, list):
        preds = [predict_single(r, weights_data, bundle_dir=bundle_dir) for r in input_data]
        return preds
    else:
        raise ValueError("input_data must be a dict or a list of dicts")

def main():
    parser = argparse.ArgumentParser(description="Standalone Bundle CLI Inference")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        in_data = json.load(f)

    res = predict(in_data, bundle_dir=args.bundle_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
