# MLRescurix Model Bundle v2: base_lgbm

- Project: k3_forest_cover_campaign
- Task: multiclass_classification
- Primary Metric (balanced_accuracy): 0.5772
- Sealed Test Gate: test-gate-run_20261021_191510
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
