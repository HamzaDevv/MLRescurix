# Phase P4 Acceptance Gate Closure Evidence Summary

**Session:** 20261014-p4-gate-closure  
**Date:** 2026-10-14  
**Git Branch:** `codex/p1-cgroup-orphan-recovery`  
**State Schema:** `PRAGMA user_version = 10`  
**Phase Status:** Complete. Acceptance Gate passed.  
**M6 Status:** NOT READY (independently maintained).

---

## 1. Executive Summary

Phase P4 (Agent-First Tools and Executable Candidates) deliverables P4.1 through P4.3 and the Phase P4 Acceptance Gate are fully implemented, verified, and evidenced. All 9 integration tests in `crates/harness/tests/candidate_tools_p4.rs` passed with 0 failures, alongside 5 passed unit tests in `python/tests/test_candidate_runner.py` and 89 passed tests across all repository regression suites. Historical M6 evidence in `docs/evidence/m6/` remains untouched.

---

## 2. Deliverable Verification Matrix

| Deliverable | Description | Verification Test | Status |
| --- | --- | --- | --- |
| **P4.1** | Versioned generic tool adapters: `workspace_fs` (read, write, patch, list, glob, search with optimistic concurrency hash fencing), `sandbox_exec` (isolated execution with `/workspace:ro`, `/scratch:rw` tmpfs, `/outputs:rw` tmpfs, thread caps, and honest Docker fallback refusal), `artifact_crud` (publish, get, list, lineage, pin, unpin, tombstone), `discover_capabilities` (contract boundaries, limits, environment profiles), `dataset_query` (schema inspection without label/holdout leakage), and `experiment_tree` (DAG visibility). Legacy `file_*` operations retained for backward compatibility. | `p4_1_workspace_fs_operations_and_concurrency_fencing`, `p4_1_sandbox_exec_isolation_and_fallback`, `p4_1_discover_capabilities_reports_contract_boundaries` | **Passed** |
| **P4.2** | Executable candidate protocol & manifest: `manifest.json` schema (`task_type`, `entrypoint`, `supported_operations` (`fit`, `predict`), `stable_row_keys` (`row_id`), `class_order`, `raw_input_schema`, `source_hash`), CLI `fit`/`predict` invocation contract, prediction validation (probabilities in `[0, 1]`, summing to $1.0 \pm 10^{-3}$, matching class order, finite numbers), and diagnostic slice evaluation with minimum support threshold (`max(20, 0.05 * N)`). | `p4_2_p4_3_executable_candidate_protocol_and_evaluation`, `p4_2_prediction_rejections_and_tamper_fencing`, `p4_4_validation_query_budget_and_slice_suppression`, `python/tests/test_candidate_runner.py` | **Passed** |
| **P4.3** | Custom runner implementation & decoupling: candidate evaluation executed without modifying any recipe registry, harness-driven cross-validation (`StratifiedKFold`, `KFold`, `GroupKFold`, `TimeSeriesSplit`), fold-level label isolation (validation features provided strictly without target column), and baseline rule/model validation. Arbitrary recipe parameters and DAG nodes supported. | `p4_2_p4_3_executable_candidate_protocol_and_evaluation`, `p4_3_split_label_isolation_and_evaluator_authority`, `p4_3_decoupled_recipes_and_experiment_tree` | **Passed** |
| **P4 Gate** | Agents can create and fix arbitrary candidate code without modifying a recipe registry; evaluation, scoring, and budgets remain under trusted authority; split/label isolation, single-use final test gate, and generation/intent fencing are strictly enforced. Official scorecards cannot be published or forged by untrusted agent roles. Validation query budget (max 50 queries per project) prevents adaptive holdout overfitting. | `p4_acceptance_gate_full_closure` | **Passed** |

---

## 3. Phase P4 Acceptance Gate Criteria Validation

1. **Criterion 1: Arbitrary Candidate Creation & Repair Without Recipe Modification**
   - Verified that agents can author arbitrary novel candidate code (`candidate.py` and `manifest.json`) in their branch workspace using `workspace_fs`.
   - Verified that evaluation runs successfully via `evaluate_submission` and the harness-managed candidate runner without touching any hardcoded recipes or recipe registries.
   - Tested candidate iterative repair through optimistic concurrency patching (`workspace_fs` patch with `expected_hash`).

2. **Criterion 2: Split & Label Isolation Under Trusted Authority**
   - Verified that during K-fold evaluation, validation feature matrices provided to candidate `predict` operations strictly omit target labels (`"target"` column).
   - Candidate attempting to access validation target labels raises an immediate exception.
   - Ground truth labels remain exclusively within the trusted evaluator runtime.

3. **Criterion 3: Official Scorecard Tamper Immunity & Authority Enforcement**
   - Verified that untrusted agent roles attempting to publish `kind = "scorecard"` via `artifact_crud` are rejected (`"scorecard publication is restricted to the trusted evaluator API"`).
   - Official evaluation scorecards can only be authored and published by the trusted evaluator adapter, registered with `producer_authority = "evaluator:trusted"`, content-addressed in CAS, and indexed in SQLite.

4. **Criterion 4: Validation Query Budget & Diagnostic Slice Suppression**
   - Verified adaptive validation query budget enforcement: maximum 50 validation queries allowed per project/session to prevent holdout overfitting; the 51st query is rejected.
   - Diagnostic slices with sample counts below minimum support threshold (`max(20, 0.05 * N)`) are automatically suppressed with status `"suppressed"`, preventing micro-slice leakage while reporting well-supported slices.

5. **Criterion 5: Sandbox Isolation & Docker Fallback Integrity**
   - Verified that candidate execution defaults to `/workspace:ro`, task-local `/scratch:rw` and `/outputs:rw` tmpfs mounts, thread caps, and timeout guards.
   - Verified that if container runtime is unavailable, host Python execution fallback is strictly refused (`"container runtime unavailable; host python fallback refused"`).

---

## 4. Test Suite Execution Logs

| Suite | Command | Results |
| --- | --- | --- |
| `candidate_tools_p4` | `cargo test --locked --test candidate_tools_p4` | **9 passed; 0 failed** (finished in 5.47s) |
| `test_candidate_runner.py` | `PYTHONPATH=python/src pytest python/tests/test_candidate_runner.py` | **5 passed; 0 failed** (finished in 5.24s) |
| `test_protocol.py` | `PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py` | **3 passed; 0 failed** (finished in 0.004s) |
| `artifact_registry_p2` | `cargo test --locked --test artifact_registry_p2` | **14 passed; 0 failed** (finished in 2.06s) |
| `resource_broker_p3` | `cargo test --locked --test resource_broker_p3 -- --test-threads=1` | **13 passed; 0 failed** (finished in 19.85s) |
| `foundation` | `cargo test --locked --test foundation -- --test-threads=1` | **26 passed; 0 failed** (finished in 12.42s) |
| `agent_m4` | `cargo test --locked --test agent_m4` | **23 passed; 0 failed** (finished in 12.18s) |
| `recovery_cycles` | `cargo test --locked --test recovery_cycles -- --test-threads=1` | **8 passed; 0 failed; 2 ignored** (finished in 35.74s) |
| `release_m6` | `cargo test --locked --test release_m6` | **5 passed; 0 failed; 3 ignored** (finished in 1.40s) |
| **Total** | | **106 passed; 0 failed; 5 ignored** |

---

## 5. Artifacts and Evidence File List

- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_candidate_tools_p4.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_candidate_runner_python.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_python_protocol.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_artifact_registry_p2.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_resource_broker_p3.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_foundation.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_agent_m4.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_recovery_cycles.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/test_release_m6.log`
- `docs/evidence/v2/p4/20261014-p4-gate-closure/summary.md`
