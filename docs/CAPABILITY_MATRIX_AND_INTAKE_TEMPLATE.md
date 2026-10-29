# MLRescurix — Capability Status & Benchmark Intake Template

Updated: 2026-10-29. Authoritative capability status following Session 5 engineering completion. All platform engineering gates are completed and verified offline. Live remote dispatch and user benchmarks are ready for the next phase.

---

## 1. Verified Capability Status Matrix

| Capability Area | Status | Evidence Type | Verification Details |
|---|---|---|---|
| **Local Custom-Model Workflow** | **COMPLETE** | **Real Container & OS** | Docker cgroups v2, timeout kill, memory enforcement, train->resume->evaluate->export lifecycle. 17 passed in `custom_model_gate.rs`. |
| **Data Partition Isolation & Streaming CSV** | **COMPLETE** | **Real Execution** | Bounded streaming RFC 4180 CSV parser (`StreamingCsvReader`). Raw SHA-256 byte verification. Sealed test exclusion in `prepare_dev_dataset`. |
| **Kaggle Compute Offloading** | **COMPLETE** | **Mocked End-to-End** | Rust-governed backend (`crates/harness/src/kaggle.rs`), credential isolation (`env:`/`file:`), anti-ban single anchor, atomic leases, output verification. 7 passed in `kaggle_gate.rs`. Live Kaggle GPU dispatch deferred for user testing. |
| **Governed Optuna Tuning** | **COMPLETE** | **Real Execution** | `BudgetedOptunaTuner` with Optuna `TPESampler`, `MedianPruner`, `create_trial` recovery, concurrency bounds, and `CampaignBudgetAuthority` reservations. 12 passed in `test_session3_tuning_and_delivery.py`. |
| **Governed LLM Gateway** | **COMPLETE** | **Real Control Plane** | Dual-tier gateway routing via `RustGatewayAdapter` with OpenRouter primary, DeepSeek fallback, pre-dispatch reservation, and uncertain reconciliation. 0 paid calls made ($0.00 spent). |
| **Extensible ML Task Adapters** | **COMPLETE** | **Real Execution** | `TabularCandidatePlugin`, `TinyTorchNeuralCandidate`, `SequentialTorchCandidate` (3D temporal sequences). Clean subprocess inference with `PYTHONPATH` unset; adversarial input validation. |
| **5 Benchmark Problems** | **DEFERRED** | **Intake Specification** | Deferred for user selection. Template defined below. |

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
