# MLRescurix Model Bundle v2: base_hist_gbdt

- Project: k5_store_sales_campaign
- Task: regression
- Primary Metric (rmse): 1049.2775
- Sealed Test Gate: test-gate-run_20261021_200725
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
