# Amazon Fresh Autonomous Test Execution & Results Report (2026-11-02)

## 1. Overview & Verification

On 2026-11-02, a complete, live autonomous run was executed on the Amazon ML Challenge dataset using DeepSeek V4.1 Flash via OpenRouter route `wafer/us`.

The run satisfied all governance, privacy, budget, and verification requirements:
- **Run Identifier**: `amazon-live-20261102-run2` (preceded by recovery run `run1`)
- **Project**: `amazon-live-20261102`
- **Model**: `deepseek/deepseek-v4.1-flash-1017` via `wafer/us` (`allow_fallbacks: false`)
- **Policy**: `docs/v2/POLICY_AMAZON_LIVE_20261102.json`
- **Status**: Completed at terminal state (`token_budget_exhaustion` with audited apex candidate)
- **Delivered Model**: `.mlrescurix-state/amazon-2025/live-20261102-run2/exported_model/frozen_model.joblib` (884,937 bytes, SHA-256 `e6ef87104c290c39044453b7b985a48847cef941eaca6a7b9644ce62d5293259`)
- **Prediction Parity**: 100% verified across sample inputs via `verify_export_parity`
- **Total Provider Billed Spend**: $0.00158 across all live runs today (authorized ceiling: $0.20)
- **Preserved Campaign Remaining Balance**: $0.183220000

---

## 2. Dataset Lineage & Validation

- **Dataset**: Amazon ML Challenge catalog dataset, 75,000 development rows (split indices 0–74,999) staged strictly in development-only space (`.mlrescurix-state/amazon-2025/staged/development.csv`).
- **Stage SHA-256**: `e41cc5d4ec37593b9a106f00f4d681af9456cd9bdf2317042e66cfea3d491214`.
- **Validation**: 2 caller-pinned folds (`.mlrescurix-state/amazon-2025/development_folds.json`, hash `6bcb576f918a7ac51048225683faf74fee9ad4544791b24117572bb2ce2b19ea`), exactly 37,500 train and 37,500 validation rows per fold.
- **Target & Metric**: `price` (regression, metric: `smape`, direction: `minimize`).
- **Holdout Status**: NO untouched final holdout was evaluated during this development run. Final holdout data is strictly gated under one-time assessment lockbox policy.
- **Kaggle Status**: Remote Kaggle execution was NOT used because remote Kaggle credentials were not provisioned. Verified isolated local fallback execution was used.

---

## 3. Multi-Turn Trace & Independent Evaluation

| Iteration | Action Type | Family / Architecture | Transforms & Target | Fold 0 SMAPE | Fold 1 SMAPE | Mean CV SMAPE | Fold Std | Overfit Gap | Critic Audit | Apex Status |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | `HYPOTHESIS_TRANSFORM` | Penalized Linear | `TEXT_TFIDF_SVD`, identity target | 74.8947 | 74.9335 | **74.9141** | 0.0194 | 0.0099 | `NOT_AUDITED` | Baseline |
| **2** | `EVALUATE_PORTFOLIO` | `HistGradientBoostingRegressor` | `TEXT_TFIDF_SVD`, `log1p` target | 64.6451 | 65.0949 | **64.8700** | 0.2249 | 1.2786 | **`APPROVED`** | 👑 **Apex #1 (Selected)** |

- **Turn 0 (Digest `56b18bb6...`)**: Proposed `TEXT_TFIDF_SVD` on `catalog_content`. Evaluated baseline penalized linear model. CV SMAPE: 74.9141.
- **Turn 1 (Digest `847acbe6...`)**: Proposed `EVALUATE_PORTFOLIO` with `hist_gbdt`, `penalized_linear`, `lgbm`, and `log1p` target transform. Evaluated on active pipeline. Independent CV SMAPE: 64.8700 (substantial improvement from 74.9141). Critic audit status: `APPROVED`. Promoted to Apex #1 and selected as accepted candidate `cand_eac378c681458b946531af7f`.
- **Turn 2 (Digest `39901d62...`)**: Proposed `EVALUATE_PORTFOLIO` with `xgboost`, `catboost`, `neural_tabular`. Since `xgboost` was unavailable in the local environment, fallback was refused and status `disposition: unsupported` was safely recorded into typed history without failing gateway disclosure.
- **Terminal Stop**: `token_budget_exhaustion` (reached the bounded turn token budget limit). Preserved accepted candidate `cand_eac378c681458b946531af7f`.

---

## 4. Operator Privacy Approval Summary

Every outbound view was staged in SQLite, inspected by the operator supervisor (`scripts/operator_supervisor.py`), verified for content privacy (no raw dataset rows, no credentials, single pinned route, bounded tokens), and approved:
- `56b18bb66910763d040e7503aaae1bac072443217c7590e578def62fdca79010`: Approved (Turn 0)
- `847acbe622f6f321313290cb6079ba52b8754c6707bf94532884e42bdca3df7b`: Approved (Turn 1)
- `39901d62127e7705947453fefe020facc3883a7e9040cdc4ea3f303a900b00e2`: Approved (Turn 2)

---

## 5. Artifacts and Inference

- Export Directory: `.mlrescurix-state/amazon-2025/live-20261102-run2/exported_model/`
- Frozen Model: `frozen_model.joblib` (884,937 bytes, SHA-256 `e6ef87104c290c39044453b7b985a48847cef941eaca6a7b9644ce62d5293259`)
- Verification: `verify_export_parity` verified exact output alignment across sample records with zero divergence.
- Inference Code:
  ```python
  import joblib
  import pandas as pd

  pipeline = joblib.load(".mlrescurix-state/amazon-2025/live-20261102-run2/exported_model/frozen_model.joblib")
  preds = pipeline.predict(pd.DataFrame([{"catalog_content": "Description text..."}]))
  ```

---

## 6. Limitations

1. **Development Scope**: Evaluations were conducted on the staged 75,000 development rows. Final holdout test sets were not evaluated; production certification is False pending holdout gate execution.
2. **Feature Scope**: The agent utilized fold-fitted TF-IDF and SVD representations of product catalog text; large language model fine-tuning was not executed locally.
3. **Kaggle Execution**: Local subprocess fallback was used; remote Kaggle transport was verified offline via test suites rather than live remote submissions.
