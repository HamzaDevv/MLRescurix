# MLRescurix — Capability Status & Benchmark Intake Template

Updated: 2026-10-29. The gates below remain open; this matrix supersedes earlier Session 5 completion claims. See `CORE_ARCHITECTURE_GROUND_TRUTH.md` and `REMAINING_ENGINEERING_PLAN.md` for boundaries and next steps.

## 1. Capability Status Matrix

| Capability | Status and evidence | Remaining limitation |
|---|---|---|
| Custom local execution | Admission disabled. Component and historical container fixtures exist. | Shared writable workflow, trusted evaluation, identity binding, reproducible image and independent review remain open. |
| Dataset splitting | Bounded CSV parser; hash covers the parsed stream; exclusive publication, frozen-protocol and private-snapshot primitives have focused offline tests. | The new snapshot is not wired into execution; no sealed-data isolation across the container lifecycle. Snapshot parent directory must be trusted. |
| Kaggle | Pool and lifecycle prototypes exercised with mock API; on 2026-10-30 the installed CLI successfully listed account notebooks for each configured token. The Rust prototype now fences concurrent dispatch and retains ambiguous intent. | No integrated production transport or live job verification; quotas and checkpoint/cancellation capabilities are unverified. |
| Tuning | Experimental Optuna/random search and SQLite fixtures. | No hard process deadline, defensible reservation or distributed lease authority; do not use for paid or remote dispatch. |
| Provider routing | Three live OpenRouter calls through the campaign-bound Rust gateway drove an Iris development agent on 2026-10-30; see `REMAINING_ENGINEERING_PLAN.md` for costs and scope. Legacy DeepSeek search/fetch fail closed. | Provider bills are not yet reconciled into the ledger; fallback, independently verified spending controls across all routes and a search provider remain open. Verify current route and pricing before each exercise. |
| Task adapters and delivery | Small tabular and 3D sequence prototypes have synthetic tests. | Not integrated as a verified end-to-end custom path; clean independent environment and authoritative promotion remain open. |
| Five real benchmarks | Not selected. | User will choose later; finite resources and quotas can prevent a job from fitting. |

---

## 2. Setup and Verification Commands

### Environment Setup
```bash
# Core requirements (Python 3.13)
pip install -e python

# Optional deep learning and tuning profiles
pip install -e "python[deep-learning,tuning]"
```

### Running Test Suites
```bash
# Session 1: Contracts and local PyTorch training
pytest python/tests/test_session1_contracts_and_torch.py

# Session 2: Kaggle compute pool, lease fencing, and OOM recovery
pytest python/tests/test_session2_kaggle_execution.py

# Session 3: Budgeted tuning, clean delivery, and adapter independence
pytest python/tests/test_session3_tuning_and_delivery.py

# Full regression & consolidation suites
pytest python/tests/test_architecture_consolidation.py python/tests/test_autonomy_regressions.py python/tests/test_autonomous_agent_loop.py python/tests/test_kaggle_pool.py
```

---

## 5. Reusable Benchmark Problem Intake Template

When the user selects the five benchmark problems, each problem should be ingested using this standardized specification.

```json
{
  "problem_id": "BENCHMARK_PROMPT_ID",
  "title": "Descriptive Problem Title",
  "modality": "tabular" | "sequence" | "multimodal" | "image" | "graph",
  "task_type": "binary_classification" | "multiclass_classification" | "regression" | "time_series_forecasting",
  "target_column": "target_name",
  "id_column": "row_id_name",
  "primary_metric": "log_loss" | "rmse" | "balanced_accuracy" | "auc" | "mae",
  "metric_direction": "minimize" | "maximize",
  "validation_protocol": {
    "cv_scheme": "stratified_kfold" | "group_kfold" | "time_series_split",
    "n_splits": 5,
    "group_column": null,
    "time_column": null,
    "split_manifest_path": "data/split_manifest.json",
    "split_manifest_sha256": "EXPECTED_SHA256"
  },
  "data_files": {
    "train_path": "data/train.csv",
    "train_sha256": "EXPECTED_SHA256",
    "test_path": "data/test.csv",
    "test_sha256": "EXPECTED_SHA256"
  },
  "resource_bounds": {
    "max_duration_seconds": 3600,
    "max_local_memory_bytes": 4294967296,
    "accelerator_type": "cpu" | "gpu" | "tpu",
    "max_cost_usd": 5.0
  },
  "quality_gate": {
    "status": "locked",
    "threshold": 0.85,
    "critic_flaw_tolerance": 0
  }
}
```

### Benchmark Intake Protocol
1. **Data Verification**: Calculate SHA-256 of raw data files and split manifests; record hashes in the problem intake specification.
2. **Modality & Contract Binding**: Verify adapter compatibility without coercing non-tabular modalities into tabular feature tables.
3. **Resource Preflight**: Run bounded `PilotResourceReport` on local host; evaluate admission to local execution or Kaggle offload.
4. **Search & Tuning**: Run `BudgetedOptunaTuner` within authorized spend and duration bounds.
5. **Clean Delivery**: Export final candidate via `CustomBundlePackager` and verify portability in clean subprocess without `PYTHONPATH`.
