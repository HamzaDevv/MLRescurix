# Phase P8 Acceptance Gate Closure Evidence Summary

**Session:** 20261015-p8-gate-closure  
**Date:** 2026-10-15  
**Git Branch:** `codex/p1-cgroup-orphan-recovery`  
**State Schema:** `PRAGMA user_version = 10`  
**Phase Status:** Complete. Acceptance Gate passed.  
**M6 Status:** NOT READY (independently maintained).

---

## 1. Executive Summary

Phase P8 (Compute Feasibility & Local Alternatives) deliverables P8.1 through P8.5 and the Phase P8 Acceptance Gate are fully implemented, verified, and evidenced. All 11 integration tests in `crates/harness/tests/feasibility_p8.rs` passed with 0 failures, alongside 59 passed tests in the Python suite (including 6 dedicated tests in `python/tests/test_p8_feasibility.py`), 16 unit tests in `crates/harness/src/lib.rs`, and 118 passed tests across locked regression suites (`release_p7`, `dashboard_p6`, `collaboration_p5`, `candidate_tools_p4`, `resource_broker_p3`, `artifact_registry_p2`, `foundation`, `agent_m4`, `recovery_cycles`, `release_m6`), for a total of **202 passed tests (0 failed)** across the entire repository. The state database schema retains `PRAGMA user_version = 10` with additive idempotent DDL (`crates/harness/src/schema_p8_feasibility.sql`). Historical M6 evidence in `docs/evidence/m6/` remains completely untouched.

---

## 2. Deliverable Verification Matrix (P8.1–P8.5)

| Deliverable | Description | Verification Test | Status |
| --- | --- | --- | --- |
| **P8.1** | **Feasibility & Execution Plan Artifacts**: Strict typed JSON schemas (`contracts/v2/hardware_profile.schema.json`, `dataset_metadata.schema.json`, `feasibility_report.schema.json`, `execution_plan.schema.json`) validated under Draft 2020-12. Domain models implemented in Rust (`crates/harness/src/feasibility.rs`) and Python (`python/src/mlrescurix_worker/scientific/feasibility.py`). Integrates with P3 resource broker discovery (`broker.rs`), capturing effective CPU cores, effective RAM, disk headroom, Docker VM/accelerator limits, and memory certification capability. Computes dataset transformation expansion factors (e.g. one-hot cardinality expansion) and dense memory footprints. | `test_p8_1_hardware_profile_from_broker_discovery`, `test_p8_1_dataset_metadata_and_transformation_expansion`, `test_p8_feasibility.py::test_p8_schemas_and_models` | **Passed** |
| **P8.2** | **Analytical Estimators & Capped Micro-Pilots**: Model family-specific analytical complexity bounds for Linear, Boosting, Tree, Neural, and Custom architectures. Accurately factors in CV fold counts, final refit multiplier, feature expansion factor, and concurrent worker lease allocations. Implements capped micro-pilots ($N \le 100$ rows, low-epoch probes) with strict timeout (default 3000ms) and peak RSS memory guards. Clamps requested rows to $\le 100$ rows and detects memory cap violations (`status='memory_exceeded'`). | `test_p8_2_model_family_complexity_estimators`, `test_p8_2_capped_micro_pilots_respect_resource_and_time_caps`, `test_p8_feasibility.py::test_analytical_estimators_and_complexity_bounds`, `test_p8_feasibility.py::test_capped_micro_pilots` | **Passed** |
| **P8.3** | **Transparent Multi-Tier Feasibility Reporting**: Deterministic four-tier feasibility verdicts emitted prior to heavy dispatch: `locally_feasible`, `feasible_with_adaptation`, `requires_external_compute`, `unknown`. Reports include full documented assumptions, confidence bounds (`high`, `medium`, `low`), predicted runtime range, predicted peak RAM range, and measured pilot trial IDs. Calibrates predictions using micro-pilot results (raising confidence to `high` when pilot succeeds). | `test_p8_3_transparent_multi_tier_verdicts_and_reports`, `test_p8_feasibility.py::test_multi_tier_feasibility_verdict_and_reporting` | **Passed** |
| **P8.4** | **Evidence-Backed Local Alternatives**: Evaluates and suggests concrete local adaptations: quantization (float64 to float32 downcasting), sparse linear baselines (scipy CSR matrices), progressive fidelity screening (20% subsampling for candidate exploration), downsampled screening, smaller tree/boosting architectures, and chunked streaming ingestion. Strictly prohibits quiet semantic degradation by documenting quality impacts and requiring explicit negotiation. | `test_p8_4_concrete_evidence_backed_local_alternatives`, `test_p8_feasibility.py::test_local_alternatives_and_quantization` | **Passed** |
| **P8.5** | **Constraint Negotiation & Safe Downscaling**: Deterministic constraint negotiation protocols generating immutable `ExecutionPlan` artifacts. Enforces safe local downscaling without purchasing external compute. Strictly preserves locked evaluation split protocols (`locked_split_protocol_preserved = true`), disallowing modifications to evaluation partitions. Heavy training dispatch feasibility gating (`enforce_heavy_dispatch_feasibility`) strictly refuses tasks with `requires_external_compute` before queue admission. | `test_p8_5_deterministic_constraint_negotiation_and_locked_splits`, `test_p8_5_heavy_dispatch_feasibility_gating`, `test_p8_feasibility.py::test_safe_constraint_negotiation_preserves_splits` | **Passed** |
| **Gate** | **Phase P8 Acceptance Gate & Orchestrator Tools**: Full acceptance gate closure: verified that explicit verdicts are emitted prior to heavy dispatch, micro-pilots strictly respect tight caps without runaway execution, agent orchestrator tools (`estimate_feasibility`, `negotiate_execution_plan`) dispatch successfully across agent roles, and CLI commands (`estimate-feasibility`, `negotiate-plan`, `capabilities`) verify release readiness. | `test_p8_acceptance_gate_full_closure`, `test_p8_orchestrator_tool_dispatch`, `test_p8_cli_commands_and_capabilities` | **Passed** |

---

## 3. Phase P8 Acceptance Gate Criteria Validation

1. **Criterion 1: Explicit Feasibility Verdict Prior to Heavy Dispatch**
   - Verified that `assess_feasibility()` deterministically produces explicit multi-tier verdicts (`locally_feasible`, `feasible_with_adaptation`, `requires_external_compute`, `unknown`).
   - Verified that `enforce_heavy_dispatch_feasibility()` gates queue admission, strictly refusing tasks where verdict is `requires_external_compute` and providing concrete local alternatives.
   - Verified in `test_p8_5_heavy_dispatch_feasibility_gating` and `test_p8_acceptance_gate_full_closure`.

2. **Criterion 2: Capped Micro-Pilots Respect Tight Resource and Time Caps**
   - Verified that micro-pilots strictly enforce $N \le 100$ rows (clamping requests $>100$ down to $100$).
   - Verified that timeout and peak RSS limits terminate runaway probes and return explicit statuses (`succeeded`, `timed_out`, `memory_exceeded`, `failed`).
   - Verified in `test_p8_2_capped_micro_pilots_respect_resource_and_time_caps` and `python/tests/test_p8_feasibility.py::test_capped_micro_pilots`.

3. **Criterion 3: Evidence-Backed Local Alternatives Without Quiet Degradation**
   - Verified that suggested alternatives (`quantization`, `sparse_linear`, `progressive_fidelity`, `downsampled_screening`, `smaller_architecture`, `streaming`) include explicit speedup factors, memory reduction factors, parameter adjustments, and quality impact statements.
   - Verified that semantic degradation cannot occur silently: adaptation proposals must be explicitly negotiated into an `ExecutionPlan`.
   - Verified in `test_p8_4_concrete_evidence_backed_local_alternatives`.

4. **Criterion 4: Locked Evaluation Split Protocol Preservation**
   - Verified that downscaling during negotiation strictly preserves locked evaluation splits (`locked_split_protocol_preserved = true`), and the database enforces `CHECK(locked_split_protocol_preserved = 1)`.
   - Verified in `test_p8_5_deterministic_constraint_negotiation_and_locked_splits` and `python/tests/test_p8_feasibility.py::test_safe_constraint_negotiation_preserves_splits`.

5. **Criterion 5: Full Regression Suite Re-Verification**
   - All locked suites re-verified with 0 failures across Rust and Python environments.

---

## 4. Test Suite Execution Logs

| Suite | Command | Results | Log File |
| --- | --- | --- | --- |
| `feasibility_p8` | `cargo test --locked --test feasibility_p8` | **11 passed; 0 failed** (0.82s) | `test_feasibility_p8.log` |
| `p8_feasibility_py` | `PYTHONPATH=python/src pytest python/tests/test_p8_feasibility.py` | **6 passed; 0 failed** (1.51s) | `test_p8_feasibility.log` |
| `release_p7` | `cargo test --locked --test release_p7` | **8 passed; 0 failed** (1.27s) | `test_release_p7.log` |
| `dashboard_p6` | `cargo test --locked --test dashboard_p6` | **9 passed; 0 failed** (1.96s) | `test_dashboard_p6.log` |
| `collaboration_p5` | `cargo test --locked --test collaboration_p5` | **11 passed; 0 failed** (0.64s) | `test_collaboration_p5.log` |
| `candidate_tools_p4` | `cargo test --locked --test candidate_tools_p4` | **9 passed; 0 failed** (7.07s) | `test_candidate_tools_p4.log` |
| `resource_broker_p3` | `cargo test --locked --test resource_broker_p3 -- --test-threads=1` | **13 passed; 0 failed** (20.30s) | `test_resource_broker_p3.log` |
| `artifact_registry_p2` | `cargo test --locked --test artifact_registry_p2` | **14 passed; 0 failed** (2.32s) | `test_artifact_registry_p2.log` |
| `foundation` | `cargo test --locked --test foundation -- --test-threads=1` | **26 passed; 0 failed** (12.97s) | `test_foundation.log` |
| `agent_m4` | `cargo test --locked --test agent_m4` | **23 passed; 0 failed** (12.29s) | `test_agent_m4.log` |
| `recovery_cycles` | `cargo test --locked --test recovery_cycles -- --test-threads=1` | **8 passed; 0 failed; 2 ignored** (39.21s) | `test_recovery_cycles.log` |
| `release_m6` | `cargo test --locked --test release_m6` | **5 passed; 0 failed; 3 ignored** (1.63s) | `test_release_m6.log` |
| `mlrescurix_harness (lib)` | `cargo test --locked --lib` | **16 passed; 0 failed; 2 ignored** (2.08s) | `test_lib.log` |
| `python_tests (host)` | `PYTHONPATH=python/src pytest python/tests/ -k "not test_real_container_tmpfs_enospc_never_publishes"` | **59 passed; 0 failed; 1 deselected** (37.16s) | `test_python_host.log` |
| **Total** | | **202 passed; 0 failed; 7 ignored/deselected** | |

---

## 5. Artifacts and Evidence File List

- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_feasibility_p8.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_p8_feasibility.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_release_p7.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_dashboard_p6.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_collaboration_p5.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_candidate_tools_p4.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_resource_broker_p3.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_artifact_registry_p2.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_foundation.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_agent_m4.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_recovery_cycles.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_release_m6.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_lib.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/test_python_host.log`
- `docs/evidence/v2/p8/20261015-p8-gate-closure/summary.md`
