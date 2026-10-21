# MLRescurix Model Bundle v2: cand_apex

- Project: k1_titanic_robustness
- Task: binary_classification
- Primary Metric (balanced_accuracy): 0.8865
- Sealed Test Gate: gate_run_20261021_232923
- Decision Threshold: 0.5000
- Calibration Method: none

## Standalone CLI Prediction
```sh
python3 inference.py --input examples_input.json --output results.json
```

## Standalone HTTP Serving
```sh
python3 serve.py --port 8080
```
