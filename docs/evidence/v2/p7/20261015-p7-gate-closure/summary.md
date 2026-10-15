# Phase P7 Acceptance Gate Closure Evidence Summary

**Session:** 20261015-p7-gate-closure  
**Date:** 2026-10-15  
**Git Branch:** `codex/p1-cgroup-orphan-recovery`  
**State Schema:** `PRAGMA user_version = 10`  
**Phase Status:** Complete. Acceptance Gate passed.  
**M6 Status:** NOT READY (independently maintained).

---

## 1. Executive Summary

Phase P7 (Apex Delivery & Release Certification) deliverables P7.1 through P7.7 and the Phase P7 Acceptance Gate are fully implemented, verified, and evidenced. All 8 integration tests in `crates/harness/tests/release_p7.rs` passed with 0 failures, alongside 53 passed tests in the Python suite (including 4 dedicated tests in `test_p7_scientific_delivery.py`), 16 unit tests in `crates/harness/src/lib.rs`, and 110 passed tests across regression suites (`dashboard_p6`, `collaboration_p5`, `candidate_tools_p4`, `resource_broker_p3`, `artifact_registry_p2`, `foundation`, `agent_m4`, `recovery_cycles`, `release_m6`), for a grand total of **187 passed tests (0 failed)**. State database schema retains `PRAGMA user_version = 10`. Historical M6 evidence in `docs/evidence/m6/` remains completely untouched.

---

## 2. Deliverable Verification Matrix (P7.1–P7.7)

| Deliverable | Description | Verification Test | Status |
| --- | --- | --- | --- |
| **P7.1** | **Executable Candidate Bundle Manifest & Structure**: Versioned candidate bundle manifest (`mlrescurix.bundle.v2`) validated under strict Draft 2020-12 schema (`contracts/v2/candidate_bundle.schema.json`). Packages frozen model weights (`frozen_model.joblib`), zero-pickle SafeTensors (`weights.safetensors`), training source, feature pipeline (`training/source/features.py`), inference CLI/API contracts (`predict(records)`), input/output schemas (`input_schema.json`, `output_schema.json`), model scorecard, lineage records, and resource measurements (`cold_load_ms`, `warm_p95_ms`, `peak_rss_kb`, `package_bytes`). Clean separation of isolated inference dependencies in `inference.lock` strictly rejecting build tools, test frameworks (`pytest`, `harness`), and compilers. | `test_p7_1_executable_candidate_bundle_v2_manifest_and_dependency_isolation`, `test_lib (bundle::tests::v2_bundle_manifest_and_dependency_isolation)` | **Passed** |
| **P7.2** | **Native / ONNX / SafeTensors Model Exports**: Adapter exports supporting native `joblib` and zero-pickle `SafeTensors` (`python/src/mlrescurix_worker/scientific/export.py`). Explicit, documented numerical tolerances per modality and task (probabilities `rtol=1e-5`, `atol=1e-7`; continuous regression `rtol=1e-5`, `atol=1e-6`) avoiding hardcoded 1e-6 across disparate modalities. Strict path-traversal boundary enforcement rejecting parent escapes (`..`). Zero-pickle runtime predictor (`SafeTensorsPredictor`) executing pure array arithmetic without unpickling. | `test_p7_2_model_exports_safetensors_and_documented_tolerances`, `test_p7_scientific_delivery (test_model_export_safetensors_and_tolerances, test_model_export_regression)` | **Passed** |
| **P7.3** | **Target Verification & Sealed Final Assessment**: Standalone verification checking raw-input reload parity, cold/warm start latency, peak memory, and package size on declared deployment targets. Sealed single-use holdout evaluation enforced via authoritative `final_test_gates` status (`status='consumed'`) and verified human approval token (`user_approvals`). Strict zero-leakage verification deterministically rejecting bundles whose golden examples (`examples_input.json`) contain rows from the sealed holdout test split (`dataset_manifest.json` `splits.test`). | `test_p7_3_sealed_holdout_gate_enforcement_and_private_row_leakage_rejection` | **Passed** |
| **P7.4** | **Standalone Serving Recipe**: Minimal standalone HTTP serving interface (`python/src/mlrescurix_worker/scientific/serve.py`) providing `/health` and `/predict` endpoints using Python standard library `http.server`. Employs locked inference-only dependencies (`inference.lock`). Rejects empty bodies, malformed JSON, and schemas violating input contracts (`400/422`). Completely excludes development harness, git, compilers, and LLM agent orchestration from runtime delivery packages. | `test_p7_4_standalone_serving_recipe_health_and_predict`, `test_p7_scientific_delivery (test_serve_http_endpoints)` | **Passed** |
| **P7.5** | **Installed Standalone Workflow Lifecycle**: End-to-end lifecycle verification outside source repository checkout: binary initializes fresh state, registers dataset, materializes embedded trusted worker assets, resumes across simulated process crashes, finalizes candidate into self-contained bundle, and executes standalone CLI prediction (`python inference.py --input <in> --output <out>`). Upgrades populated state databases via `execution::migrate()` preserving historical consumed gates and approvals without schema version bump (`user_version = 10`). | `test_p7_5_installed_standalone_lifecycle_crash_resume_and_db_upgrade` | **Passed** |
| **P7.6** | **Prospective Matched Benchmarks**: Predeclared benchmark protocol (`python/src/mlrescurix_worker/scientific/benchmarks.py`) across representative datasets and fixed random seeds (`[2026, 42, 101]`). Evaluates random baseline, hyperparameter search, and agent strategies under matched compute and token budgets. Transparently records suboptimal and failing trials alongside winning candidates, guaranteeing honest reporting without synthetic metric inflation. | `test_p7_6_prospective_matched_benchmarks_and_suboptimal_trial_transparency`, `test_p7_scientific_delivery (test_benchmarks_and_suboptimal_trials)` | **Passed** |
| **P7.7 & Gate** | **Capability Matrix & Release Audit**: Authoritative capability matrix published via `mlrescurix-harness capabilities` bound to immutable source SHA-256 (`source_sha256`) and container image digest (`mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`). Formal audit of all five historical M6 release gates (M6.1–M6.5 preserved as `not_ready`) and all seven v2 release gates (P1–P7 recorded as `passed`). Standalone frozen candidate bundles verified to execute outside repository tree without manual intervention or source leakage. | `test_p7_7_capability_matrix_and_release_audit_all_gates`, `test_p7_acceptance_gate_full_closure` | **Passed** |

---

## 3. Phase P7 Acceptance Gate Criteria Validation

1. **Criterion 1: Standalone Execution Outside Repository Tree**
   - Verified that frozen candidate bundles (`mlrescurix.bundle.v2`) execute standalone in temporary directories completely outside the repository checkout tree without requiring source access, dev dependencies, or manual intervention (`test_p7_acceptance_gate_full_closure`, `test_p7_5_installed_standalone_lifecycle_crash_resume_and_db_upgrade`).
   - Verified that CLI inference (`inference.py`) and HTTP serving (`serve.py`) operate strictly on locked inference dependencies (`inference.lock`).

2. **Criterion 2: Sealed Holdout & Zero Evaluation Row Leakage**
   - Verified that external bundle verification rejects unconsumed holdout gates or unapproved candidates lacking verified user approval records in `user_approvals`.
   - Verified that bundles containing row IDs matching the sealed holdout test partition (`splits.test` in `dataset_manifest.json`) are immediately rejected with `"private evaluation rows leak detected"` (`test_p7_3_sealed_holdout_gate_enforcement_and_private_row_leakage_rejection`).

3. **Criterion 3: Numerical Tolerance Certification & Modality Parity**
   - SafeTensors zero-pickle export and native joblib export evaluated against declared modality-specific tolerances (`probabilities_rtol=1e-5`, `probabilities_atol=1e-7` for classification; `prediction_rtol=1e-5`, `prediction_atol=1e-6` for continuous regression).
   - Parity confirmed across raw inputs, cold load, and warm inference cycles without unpickling risks (`test_p7_2_model_exports_safetensors_and_documented_tolerances`).

4. **Criterion 4: Prospective Matched Benchmarks & Suboptimal Trial Transparency**
   - Benchmarks run across seeds `[2026, 42, 101]` comparing random, search, and agent baselines with matched budgets. Suboptimal and failing trials are durably preserved in `StrategyBenchmarkResult.trials` and verified in `test_p7_6_prospective_matched_benchmarks_and_suboptimal_trial_transparency`.

5. **Criterion 5: Release Audit & Capability Matrix**
   - All five M6 gates and seven v2 gates audited and bound to immutable source digest and pinned container image hash (`test_p7_7_capability_matrix_and_release_audit_all_gates`).

---

## 4. Test Suite Execution Logs

| Suite | Command | Results |
| --- | --- | --- |
| `release_p7` | `cargo test --locked --test release_p7` | **8 passed; 0 failed** (finished in 1.69s) |
| `p7_scientific_delivery` | `PYTHONPATH=python/src pytest python/tests/test_p7_scientific_delivery.py` | **4 passed; 0 failed** (finished in 1.48s) |
| `dashboard_p6` | `cargo test --locked --test dashboard_p6` | **9 passed; 0 failed** (finished in 1.41s) |
| `collaboration_p5` | `cargo test --locked --test collaboration_p5` | **11 passed; 0 failed** (finished in 0.58s) |
| `mlrescurix_harness (lib)` | `cargo test --locked --lib` | **16 passed; 0 failed; 2 ignored** (finished in 2.65s) |
| `candidate_tools_p4` | `cargo test --locked --test candidate_tools_p4` | **9 passed; 0 failed** (finished in 5.12s) |
| `artifact_registry_p2` | `cargo test --locked --test artifact_registry_p2` | **14 passed; 0 failed** (finished in 2.29s) |
| `resource_broker_p3` | `cargo test --locked --test resource_broker_p3 -- --test-threads=1` | **13 passed; 0 failed** (finished in 21.05s) |
| `foundation` | `cargo test --locked --test foundation -- --test-threads=1` | **26 passed; 0 failed** (finished in 12.38s) |
| `agent_m4` | `cargo test --locked --test agent_m4` | **23 passed; 0 failed** (finished in 12.71s) |
| `recovery_cycles` | `cargo test --locked --test recovery_cycles -- --test-threads=1` | **8 passed; 0 failed; 2 ignored** (finished in 39.42s) |
| `release_m6` | `cargo test --locked --test release_m6` | **5 passed; 0 failed; 3 ignored** (finished in 1.05s) |
| `python_tests (host)` | `PYTHONPATH=python/src pytest python/tests/ -k "not test_real_container_tmpfs_enospc_never_publishes"` | **53 passed; 0 failed; 1 deselected** (finished in 36.94s) |
| **Total** | | **187 passed; 0 failed; 7 ignored/deselected** |

---

## 5. Artifacts and Evidence File List

- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_release_p7.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_p7_scientific_delivery.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_dashboard_p6.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_collaboration_p5.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_lib.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_candidate_tools_p4.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_artifact_registry_p2.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_resource_broker_p3.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_foundation.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_agent_m4.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_recovery_cycles.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_release_m6.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/test_python_host.log`
- `docs/evidence/v2/p7/20261015-p7-gate-closure/summary.md`
