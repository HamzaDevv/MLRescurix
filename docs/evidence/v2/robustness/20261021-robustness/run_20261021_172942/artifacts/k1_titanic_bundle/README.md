# MLRescurix Model Bundle v2: cand_ensemble_ep2

- Project: k1_titanic_campaign
- Task: binary_classification
- Primary Metric (balanced_accuracy): 1.0000
- Sealed Test Gate: test-gate-run_20261021_172942
- Decision Threshold: 0.3000
- Calibration Method: isotonic

## Standalone CLI Prediction
```sh
python3 inference.py --input examples_input.json --output results.json
```

## Standalone HTTP Serving
```sh
python3 serve.py --port 8080
```
