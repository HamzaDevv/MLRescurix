import sys
import os
import json
import math
import csv
import argparse

FEATURES = ["x1","x2"]
TARGET = "target"
IS_REGRESSION = True

def extract_features(row):
    return [float(row[f]) for f in FEATURES]

def basis_expansion(x_vec):
    phi = [1.0]
    for v in x_vec:
        phi.append(v)
    for v in x_vec:
        phi.append(v * v)
    for v in x_vec:
        phi.append(math.sin(v))
    return phi

def fit(train_csv_path, weights_out_path):
    rows = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            y = float(r[TARGET])
            rows.append((phi, y))

    if not rows:
        raise ValueError("no training rows found")

    dim = len(rows[0][0])
    w = [0.0] * dim
    lr = 0.04
    n = len(rows)

    if IS_REGRESSION:
        gamma = 2.0
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                y_pred = sum(w[j] * phi[j] for j in range(dim))
                r = y_pred - y
                factor = r / (1.0 + (r / gamma) ** 2)
                for j in range(dim):
                    grad[j] += factor * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)
    else:
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                z = sum(w[j] * phi[j] for j in range(dim))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
                r = p - y
                for j in range(dim):
                    grad[j] += r * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)

    model_payload = {
        "candidate_id": "cand_proj_s1_evidence_demo_309ff456",
        "task_type": "regression",
        "features": FEATURES,
        "target": TARGET,
        "weights": w,
        "basis_dim": dim,
        "gamma": 2.0 if IS_REGRESSION else None
    }
    with open(weights_out_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

def infer(features_csv_path, weights_in_path, predictions_out_path):
    with open(weights_in_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    w = model["weights"]
    dim = len(w)

    preds = []
    with open(features_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row_id = int(r["__mlr_row_id"])
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            if len(phi) != dim:
                raise ValueError("basis dimension mismatch")
            val = sum(w[j] * phi[j] for j in range(dim))
            if not IS_REGRESSION:
                p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
                pred_val = 1.0 if p >= 0.5 else 0.0
            else:
                pred_val = val
            preds.append({"row_id": row_id, "prediction": round(pred_val, 6)})

    with open(predictions_out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    fit_p = subparsers.add_parser("fit")
    fit_p.add_argument("--train-data", required=True)
    fit_p.add_argument("--weights-out", required=True)

    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("--features-in", required=True)
    infer_p.add_argument("--weights-in", required=True)
    infer_p.add_argument("--predictions-out", required=True)

    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
