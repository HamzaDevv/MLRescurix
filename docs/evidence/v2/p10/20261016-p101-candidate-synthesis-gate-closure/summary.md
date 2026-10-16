# Phase P10.1 Autonomous Candidate Synthesis Gate Closure

Date: 2026-10-16 (Asia/Kolkata)
Baseline Commit: `abf1f235030a79f9a606b429e34608431e37603d`
Phase / Slice: Phase P10 (Autonomous Candidate Synthesis & Iterative Refinement Loop) — Slice P10.1 (Autonomous Candidate Synthesis Engine)
Status: CLOSED AND VERIFIED

## Delivered Capabilities

1. **Candidate Synthesis Schema (Draft 2020-12)**:
   - Defined strict Draft 2020-12 typed schema `contracts/v2/candidate_synthesis.schema.json` covering:
     - Problem definitions (`task_type`: binary, multiclass, regression; `target_column`, `id_column`, `time_column`, `group_column`).
     - Input & target column descriptors (data types: numerical, categorical, text, datetime; nullability, imputation strategies, class order).
     - Metric objectives (`primary_metric`, `direction`: maximize/minimize, target thresholds, secondary metrics, diagnostic slice columns).
     - Model family intents (families: boosting, neural, tree, linear, custom; frameworks: lightgbm, xgboost, catboost, pytorch, scikit-learn, custom; hyperparameters, feature pipeline options).
     - Generated candidate manifest contracts (`candidate_id`, `name`, `task_type`, `entrypoint`, `supported_operations`, `stable_row_keys`, `class_order`, `raw_input_schema`, `source_hash`, `model_metadata`).

2. **Candidate Synthesis Engine (Scientific Python)**:
   - Implemented `python/src/mlrescurix_worker/scientific/candidate_synthesis.py`:
     - Autonomous candidate code generator creating decoupled candidate directories containing:
       - `entrypoint.py`: Protocol-conforming CLI entrypoint (`fit --train-path <csv> --model-dir <dir>`, `predict --input-path <csv> --model-dir <dir> --output-path <csv>`).
       - `model.py`: Modular model wrapper implementing `fit`, `predict`, `predict_proba`, `save`, and `load` for LightGBM, XGBoost, PyTorch neural networks (MLP), Random Forest / Extra Trees, Linear / Logistic regression, and custom pipelines.
       - `features.py`: Decoupled feature engineering pipeline handling numeric imputation, scaling (standard, minmax, robust), categorical imputation, and encoding (onehot, ordinal).
       - `manifest.json`: Strictly conforming to candidate contract with cryptographic `source_hash` computed across all candidate files.
     - Enforced candidate protocol invariants:
       - Strict label isolation: in `predict`, drops or ignores target column if mistakenly present in input.
       - Exact row ID alignment: output dataframe preserves input row keys in exact order.
       - Prediction format validation: finite numbers for regression; declared `class_order` alignment and probability columns (`prob_{c}`) summing to 1.0 for classification.
   - Exposed CLI subcommand `candidate-synthesize` in `python/src/mlrescurix_worker/scientific/cli.py`.

3. **Harness Control Plane, RBAC, Persistence, and Capabilities**:
   - Retained `PRAGMA user_version = 10` using additive idempotent DDL `crates/harness/src/schema_p10_synthesis.sql` creating `candidate_synthesis_records` table.
   - Registered `synthesize_candidate` tool in `ToolRegistry` (`crates/harness/src/agents/tools.rs`), dispatched in `AgentOrchestrator` (`crates/harness/src/agents/orchestrator.rs`), authorized for `lead`, `feature_engineer`, and `model_specialist` roles (`crates/harness/src/agents/roles.rs`).
   - Added CLI subcommand `synthesize-candidate` and updated harness capabilities reporting `"p10_candidate_synthesis": "passed"` and `candidate_synthesis_capabilities`.
   - Embedded `candidate_synthesis.py` into binary assets via `EMBEDDED_PYTHON_FILES` (`crates/harness/src/assets.rs`).

4. **Sandbox Execution & Verification Gate**:
   - Verified that synthesized candidates execute cleanly inside isolated container environments without crashing or leaking labels.
   - Wired candidate synthesis directly to `evaluate_submission` (and `candidate_runner.evaluate_candidate_cv`), verifying 3-fold cross-validation, official evaluator scorecard creation in CAS, diagnostic slice evaluations, and insertion into `candidate_evaluations` SQLite table.
   - Verified negative scientific guards:
     - Label isolation enforcement (target column in predict input dropped without leakage).
     - Tampering detection (source file edits without manifest source_hash update rejected with `source_hash mismatch`).
     - RBAC authorization enforcement (unauthorized role rejected with explicit error).

## Verification Results

- `cargo test --locked --test candidate_synthesis_p10`: 6 passed, 0 failed
- `cargo test --locked --test adapters_p9`: 12 passed, 0 failed
- `cargo test --locked --test feasibility_p8`: 11 passed, 0 failed
- `cargo test --locked --test release_p7`: 8 passed, 0 failed
- `cargo test --locked --test dashboard_p6`: 9 passed, 0 failed
- `cargo test --locked --test collaboration_p5`: 11 passed, 0 failed
- `cargo test --locked --test candidate_tools_p4`: 9 passed, 0 failed
- `cargo test --locked --test resource_broker_p3 -- --test-threads=1`: 13 passed, 0 failed
- `cargo test --locked --test artifact_registry_p2`: 14 passed, 0 failed
- `cargo test --locked --test foundation -- --test-threads=1`: 26 passed, 0 failed
- `cargo test --locked --test agent_m4`: 23 passed, 0 failed
- `cargo test --locked --test recovery_cycles -- --test-threads=1`: 8 passed, 2 ignored, 0 failed
- `cargo test --locked --test release_m6`: 5 passed, 3 ignored, 0 failed
- `cargo test --locked --lib`: 20 passed, 2 ignored, 0 failed
- `PYTHONPATH=python/src pytest python/tests/ -k "not test_real_container_tmpfs_enospc_never_publishes"`: 90 passed, 1 deselected, 0 failed
- **Total passed tests: 254 passed, 0 failed across full repository suites.**
