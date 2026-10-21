# MLRescurix Model Bundle v2: base_lgbm

- Project: k4_ieee_fraud_campaign
- Task: binary_classification
- Primary Metric (balanced_accuracy): 0.5027
- Sealed Test Gate: test-gate-run_20261021_193634
- Decision Threshold: 0.5000
- Calibration Method: isotonic

## Standalone CLI Prediction
```sh
python3 inference.py --input examples_input.json --output results.json
```

## Standalone HTTP Serving
```sh
python3 serve.py --port 8080
```
