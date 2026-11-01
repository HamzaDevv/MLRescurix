# Autonomous Model Creator - Session 3 Handoff

> **To Reviewer (Codex)**:
> This handoff document records the implementation completed in Session 3 addressing all 6 findings from `docs/AUTONOMOUS_MODEL_CREATOR_SESSION_2_REVIEW.md` in one substantial batch, guided authoritatively by `Major_refactor_docs/plans/FINAL_AUTONOMOUS_MODEL_CREATOR_MASTER_PLAN.md`.
>
> **Strict Operational Rule Followed**: All changes remain uncommitted and unpushed in the working tree for your independent review.

---

## 1. Resolution of the 6 Review Findings

### Finding 1: Provider Routing & Secret Fallthrough Closed
- **Files Touched**: [`crates/harness/src/main.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/main.rs), [`crates/harness/src/agents/gateway.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/agents/gateway.rs)
- **Problem**: In Session 2, `main.rs` maintained an unrouted dispatch path that bypassed `dispatch_gateway_with_routing` if `--routing-policy` was omitted. Additionally, secret resolution fell through to `OPENROUTER_API_KEY` when a custom provider's `secret_ref` was missing.
- **Resolution**:
  - Eliminated the unrouted dispatch bypass by constructing an effective `AgentRoutingPolicy` when `--routing-policy` is omitted, channeling **all** dispatches through `dispatch_gateway_with_routing`.
  - Fixed secret resolution so that when `prov.secret_ref` is specified, a missing secret fails closed and does **not** fall through to OpenRouter.
  - Added namespacing to model pricing (`register_namespaced_model_pricing`, `resolve_namespaced_model_pricing`) formatted as `{provider_key}::{model}` to prevent cross-provider pricing pollution.
  - Handled in-memory / test gateways with empty `endpoint_url` in `default_routing_policy_for_request` and `dispatch_gateway_with_routing`, ensuring non-empty URI validation without causing spurious endpoint mismatches.
  - Supported OpenRouter backend slugs in caller-specified routing preferences without false-positive URL rejection.

### Finding 2: Atomic In-Flight Role Spend Reservation & Concurrency Fencing
- **File Touched**: [`crates/harness/src/agents/gateway.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/agents/gateway.rs)
- **Problem**: `reserve_role_spend` only checked already completed transactions, leaving in-flight requests uncounted until completion and enabling concurrent requests to bypass `max_spend_millicents`.
- **Resolution**:
  - `reserve_role_spend` executes inside an immediate transaction (`rusqlite::TransactionBehavior::Immediate`) with a 10s busy timeout.
  - Inserts/updates `provider_accounting` with `status = 'reserved'`.
  - Both `query_role_spend` and `reserve_role_spend` sum over `status IN ('ok', 'reserved')`.
  - If a dispatch fails before completion (e.g. policy approval required or network failure), the in-flight reservation row is deleted (`DELETE FROM provider_accounting WHERE project=?1 AND request_id=?2 AND status='reserved'`), cleanly releasing the reserved budget without phantom failure rows polluting accounting counts.
  - Concurrency test `multi_provider_role_routing_and_context_bounds` executes two concurrent dispatches under a tight role budget ceiling, proving that atomic reservation allows exactly 1 request to proceed while the competing request is rejected.

### Finding 3: Assessment Lockbox Consumption Persistence & Instant Crash Recovery
- **Files Touched**: [`crates/harness/src/schema_custom_protocol.sql`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/schema_custom_protocol.sql), [`crates/harness/src/custom_protocol.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/custom_protocol.rs)
- **Problem**: `assessment_lockbox_consumptions` lacked receipt storage. A worker or host crash after consumption required re-evaluation (which is rejected by the one-shot consumption lock) or manual intervention. Moreover, caller-supplied labels were accepted without cross-verifying ground truth against registered datasets on disk.
- **Resolution**:
  - Added `receipt_json TEXT NOT NULL` column to `assessment_lockbox_consumptions` table in `schema_custom_protocol.sql`.
  - Added `receipt_json: String` to `AssessmentBinding` and `AssessmentRecord`.
  - Implemented `get_assessment_consumption_receipt(connection, project, dataset_id)` returning `Result<Option<EvaluationReceipt>>`.
  - Added authoritative verification in `evaluate_and_consume_assessment`: cross-checks caller-supplied labels against ground truth from the registered `scientific_datasets` CSV on disk using `extract_lockbox_ground_truth`. Any value mismatch fails closed before consumption.
  - Implemented `evaluate_and_consume_assessment_authoritative` to evaluate and consume directly from the registered dataset file without requiring the caller to supply labels.
  - Tolerated legacy/mock schemas without `path` column in `scientific_datasets` to support isolated unit tests.
  - Upon evaluating holdout rows, `evaluate_and_consume_assessment` atomically records the receipt JSON. If a crash occurs, reopening the database and querying `get_assessment_consumption_receipt` retrieves the exact receipt without re-running evaluation.

### Finding 4: Group & Temporal Split Validation on `CustomProtocol`
- **File Touched**: [`crates/harness/src/custom_protocol.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/custom_protocol.rs)
- **Problem**: `CustomProtocol` accepted `group_column` and `time_column` annotations but did not validate entity disjointness or temporal monotonicity.
- **Resolution**:
  - Added `row_groups: Option<HashMap<u64, String>>` and `row_timestamps: Option<HashMap<u64, i64>>` to `CustomProtocol`.
  - Added `validate_group_separation(&self, row_groups: &HashMap<u64, String>) -> Result<()>`:
    - Enforces mutual group disjointness across train, validation, test, and lockbox.
    - Validates within-fold group disjointness across fold train and fold validation sets.
  - Added `validate_temporal_ordering(&self, row_timestamps: &HashMap<u64, i64>) -> Result<()>`:
    - Enforces $\min(\text{val}) \ge \max(\text{train})$, $\min(\text{test}) \ge \max(\text{val})$, and $\min(\text{lockbox}) \ge \max(\text{dev})$.
    - Enforces $\min(\text{fold.val}) \ge \max(\text{fold.train})$ for every fold split.
  - `canonical(p)` automatically invokes both validators whenever groups or timestamps are supplied.

### Finding 5: Evidence-Bound Metric Assurance (`ReferenceEvidence`) & Unclosed Metrics
- **Files Touched**: [`crates/harness/src/trusted_evaluation.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/src/trusted_evaluation.rs), [`crates/harness/tests/trusted_evaluation.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/tests/trusted_evaluation.rs)
- **Problem**: `evaluate_with_spec_and_evidence` accepted a raw `golden_fixture_verified: bool`, allowing callers to self-certify as `reference_conformant`. Metamorphic checks also evaluated with default parameters rather than the spec's `decision_threshold` and `variant`. Metric support was a closed enum rejecting unfamiliar custom losses.
- **Resolution**:
  - Extended `Metric` enum with `Cauchy`, `Huber`, and untagged `Custom(String)` to support unfamiliar custom models and loss functions.
  - Created `ReferenceEvidence` struct requiring `reference_identity`, 64-hex `fixture_sha256`, finite `expected_result`, 64-hex `verified_receipt_sha256`, `metric_version`, and non-negative finite `tolerance`.
  - Cryptographically verifies `fixture_sha256` against `Sha256(labels_sha256:predictions_sha256)` digest and verifies evaluated score against `expected_result` within `tolerance`.
  - Metamorphic property checks (`check_spec_identity_property`, `check_spec_permutation_invariance_property`) evaluate directly with the spec's `decision_threshold`, `variant`, and `class_order`.
  - Enforced multiclass `class_order` dimension validation for vector prediction rows.

### Finding 6: Transitive Fit Scope Provenance & Row-Set Partial Overlap Rejection
- **Files Touched**: [`python/src/mlrescurix_worker/scientific/contracts/validator.py`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/python/src/mlrescurix_worker/scientific/contracts/validator.py), [`python/tests/test_v3_contracts_and_routing.py`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/python/tests/test_v3_contracts_and_routing.py)
- **Problem**: `check_fit_scope_compatibility` checked only single hash equality, missing partial row overlaps ($A \cap B \neq \emptyset$), ignoring `protocol_sha256` mismatch, and not checking whether `stateless` or `fixed_pretrained` scopes had fitted labels. Furthermore, `validate_transitive_provenance` did not pass row sets to `check_fit_scope_compatibility`.
- **Resolution**:
  - Enhanced `check_fit_scope_compatibility` to:
    - Enforce exact `protocol_sha256` identity between producer and consumer.
    - Check row-set partial overlap: `overlap = set(producer_label_rows).intersection(set(evaluated_label_rows))` and reject immediately on non-empty intersection.
    - Check feature exclusion overlap: `overlap = set(producer_feature_rows).intersection(set(consumer_exclusion_rows))` and reject immediately.
    - Verify that `stateless` and `fixed_pretrained` producers have zero fitted label rows.
  - Updated `validate_transitive_provenance` to collect and pass `producer_label_rows`, `evaluated_label_rows`, `producer_feature_rows`, and `consumer_exclusion_rows` across DAG ancestors, rejecting transitive partial label leakage and protocol mismatches.

---

## 2. Governed Local Source-Execution Vertical Slice

A complete end-to-end integration test was authored in [`crates/harness/tests/governed_source_execution.rs`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/crates/harness/tests/governed_source_execution.rs).

### Execution Details:
- **Unfamiliar Model & Loss**: An *Asymmetric Pseudo-Huber Spline Regressor with Cauchy Loss* ($L(r) = \frac{\gamma^2}{2} \ln(1 + (r/\gamma)^2)$), authored in Python without scikit-learn or PyTorch, fitting a non-linear continuous target ($y = 2.5 x_1^2 - 1.2 \sin(x_2) + 0.5$).
- **Lifecycle Flow (10 Stages)**:
  1. *Dataset Generation*: 30 synthetic samples with non-linear ground truth.
  2. *Controller Registration*: SQLite registration and hash verification.
  3. *Controller Protocol*: 20% lockbox carved (6 rows); 24 dev rows split into train (16), val (6), test (2). Protocol frozen with `custom_protocol::freeze`.
  4. *Snapshot Preparation*: `custom_snapshot::prepare_registered` yields `training.csv`, `validation_features.csv` (no target column; labels held out), and private `labels.json`.
  5. *Agent Authoring*: `program.json` manifest specifying `fit` and `infer` invocations.
  6. *Fit Execution*: Subprocess executes `solution.py fit` to produce `weights.json`.
  7. *Infer Execution*: Subprocess executes `solution.py infer` on `validation_features.csv` to produce `predictions.json`.
  8. *Independent Controller Scoring*: Controller loads private snapshot `labels.json` and evaluates predictions via `custom_snapshot::evaluate_predictions` and `trusted_evaluation::evaluate_with_spec`, producing MAE score ~0.42 and `provisional_spec_checked`. Valid `ReferenceEvidence` upgrades to `reference_conformant`.
  9. *Delivery Bundle Packaging & Anti-Tampering*: Packages candidate into bundle v2, including bytecode (`.pyc`) and 100% manifest tracking. Verifies `bundle::verify_bundle`. Rejects untracked physical files and symlinks.
  10. *Lockbox Assessment, Atomic Consumption & Crash Recovery*: Evaluates lockbox features, executes atomic consumption via `custom_protocol::evaluate_and_consume_assessment`, verifies rejection of second attempt, drops DB connection, reopens, and recovers exact receipt via `custom_protocol::get_assessment_consumption_receipt`.

---

## 3. Verification Record

### Test Commands and Results:
```bash
# 1. Complete Rust core test suite (library + custom_protocol + custom_snapshot + trusted_evaluation + governed_source_execution)
cargo test --locked -p mlrescurix-harness --lib --test custom_protocol --test custom_snapshot --test trusted_evaluation --test governed_source_execution

# Output:
# running 36 tests (lib) -> 34 passed, 2 Docker-ignored, 0 failed
# running 8 tests (custom_protocol) -> 8 passed, 0 failed
# running 3 tests (custom_snapshot) -> 3 passed, 0 failed
# running 1 test (governed_source_execution) -> 1 passed, 0 failed
# running 12 tests (trusted_evaluation) -> 12 passed, 0 failed
# Total: 58 passed, 2 ignored, 0 failed in 4.26s.

# 2. Rust agent & gateway integration test suite
cargo test --locked -p mlrescurix-harness --test agent_m4

# Output:
# running 26 tests (agent_m4) -> 26 passed, 0 failed in 25.40s.

# 3. Python contracts & routing test suite
PYTHONPATH=python/src:python/tests python3 -m pytest -q python/tests/test_v3_contracts_and_routing.py python/tests/test_architecture_consolidation.py

# Output:
# 25 passed in 22.04s.

# 4. Python custom evaluation & worker process test suite
PYTHONPATH=python/src:python/tests python3 -m pytest -q python/tests/test_custom_evaluation_review.py python/tests/test_custom_worker_process.py python/tests/test_custom_agent.py

# Output:
# 35 passed in 30.46s.
```

---

## 4. Known Boundaries & Honest Assessment

- `Fatal Functional Bug`: None. All production entry points compile cleanly and tests pass.
- `Shallow Verification`: Full Docker container execution tests (`execution::resources::tests::live_cgroup_oom_is_enforced...`) remain ignored as in previous sessions because the macOS host lacks a running Docker daemon. Local source execution is verified via host process isolation (`std::process::Command`).
- `Minor Robustness Risk`: The pure Python spline regressor uses fixed gradient descent iterations (1000 epochs) rather than adaptive convergence criteria. This is intentional for deterministic testing.
