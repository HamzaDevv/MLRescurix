# MLRescurix Model Bundle v2: cand_apex_k2

- Project: k2_house_prices_robustness
- Task: regression
- Primary Metric (rmse): 16127.7707
- Sealed Test Gate: gate_run_20261021_231630
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
