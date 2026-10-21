# MLRescurix Model Bundle v2: base_hist_gbdt

- Project: k2_house_prices_campaign
- Task: regression
- Primary Metric (rmse): 52327.5866
- Sealed Test Gate: test-gate-run_20261021_183624
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
