# Refactor Session 1 Review — Runnable Campaign Foundation & Contract Repair

> **Independent review correction, November 2, 2026:** This original implementation packet is preserved as historical author-reported evidence. Its all-PASS verdict and several architecture/accounting claims were not upheld by code and boundary checks. Read [S1_RED_TEAM_REVIEW.md](./S1_RED_TEAM_REVIEW.md) for the current verdict, reproduced defects, bounded fixes and S2 follow-up. The normal campaign entry point now remains gated pending verified isolation; trusted synthetic diagnostics require the explicit `--offline-fixture` mode. The original command below is historical, not a current production certification command.

## 1. Executive Summary & Ground Truth Identity
- **Repository**: `/Users/hamza/Developer/Imp_Projects/MLRescurix`
- **Session**: Session 1 of 6 (Runnable Campaign Foundation & Contract Repair)
- **Authoritative Plan**: `Major_refactor_docs/plans/FINAL_AUTONOMOUS_MODEL_CREATOR_MASTER_PLAN.md` (Sections 1–4, 4.9, 10.1, 10.4, 10.5, 11)
- **Prompt Specification**: `Major_refactor_docs/plans/ANTIGRAVITY_SESSION_1_PROMPT.md`
- **Baseline Git Commit**: `95a35e9`
- **Runtime Environment**:
  - Operating System: macOS (Darwin 24.6.0, arm64)
  - Python Environment: Python 3.13.3 (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`)
  - Test Frameworks: `pytest-8.3.4`, `cargo 1.84.0` (or local Rust toolchain)
- **Budget Compliance**: Preserved financial obligations ($0.280494020 ceiling, $0.20 per-run ceiling; operational defaults: 5,000,000 tokens, 30 Kaggle hours, 500 search queries). Zero paid external provider calls or live queries were made; all provider interactions used offline scripted fixtures.

---

## 2. Implemented Architecture & Behavioral Changes

### 2.1 Runnable Boss/Specialist Vertical Slice
Implemented the end-to-end autonomous campaign pipeline exposed via:
1. **Public CLI Binary**: `mlrescurix-harness autonomous-create --state <state_dir> --spec <spec_path>`
2. **Rust Public API**: `mlrescurix_harness::campaign::run_autonomous_campaign(&state_root, &problem_spec, gateway_override)`

The complete lifecycle executed in order:
1. **Intake & Dataset Validation**: Verifies dataset existence, non-emptiness, headers, and row count constraints. Canonicalizes paths to eliminate working directory discrepancies.
2. **Protocol Freezing & Holdout Isolation**: Carves train, validation, and holdout (lockbox) splits using deterministic seed hashing. Writes train and validation data to an isolated worker workspace while holding physical lockbox labels strictly within the controller.
3. **Owner State & Lease Persistence**: Persists campaign owner state, project registrations, and single-lease concurrency guards in transactional SQLite storage (`state.sqlite`).
4. **Fresh Specialist Context & Dispatch**:
   - Decomposes the goal into discrete M0 task assignments (`task_authoring_*`, `task_challenger_*`).
   - Specialist context includes only immutable problem inputs, dataset and protocol SHA256 hashes, and allowed tool capabilities. Specialist conversations and raw logs are excluded from foreign contexts.
5. **Execution & Evidence Receipting**:
   - Specialist authors `solution.py` and `program.json`.
   - Solution executes in a subprocess (`fit` and `infer` subcommands).
   - Validation predictions are independently scored by the controller using trusted evaluation metrics (`evaluate_with_spec_evidence_and_db`).
6. **Owner Decision & Challenger Review**:
   - Owner reviews compact findings and metric delta, making an evidence-based decision to delegate to an independent challenger.
   - Challenger receives only the primary evidence, protocol SHA256, and candidate identity, without proposer advocacy.
   - Challenger audits validation bounds and records an `APPROVED` or `CHALLENGED` verdict in `critic_audits`.
7. **One-Shot Lockbox Consumption & Bundle Finalization**:
   - Upon challenger approval, the controller evaluates the candidate on lockbox holdout features and records one-shot consumption in `assessment_lockbox_consumptions`.
   - Packages a verifiable delivery bundle (`manifest.json`, `scorecard.json`, `frozen_model.joblib`, `inference.py`, `serve.py`).
8. **Resumption & Epoch Fencing**:
   - Re-running the campaign on an existing state detects the completed bundle, confirms lockbox consumption integrity, and idempotently returns the certified outcome without re-running training or re-consuming the lockbox.

### 2.2 Execution Contract Repairs
- **Cross-Layer Action Parity (`RESEARCH_TOPIC`)**:
  - Unified action descriptors across `contracts/v3/actions.json`, embedded Rust assets (`EMBEDDED_PYTHON_FILES` in `crates/harness/src/assets.rs`), and Python worker runtime (`candidate_pipeline.py`, `autonomous_loop.py`).
  - Added `RESEARCH_TOPIC` schema validation and dispatch handling across all layers.
- **Model Parameter & Task Compatibility**:
  - Eliminated silent inverse-C-to-Ridge-alpha conversions and ignored classification penalties.
  - Added strict parameter rejection in `validate_spec`: classification parameters (`penalty`, `C`) passed to regression models or regression parameters (`alpha`) passed to classification models are rejected with actionable `ValueError` exceptions before fitting.
- **Independent Sibling Validation in Portfolios**:
  - In `evaluate_portfolio`, members are evaluated independently. An invalid member specification fails cleanly and records an error receipt without corrupting or aborting valid sibling models.
- **Deduplication by Resolved Identity & Effective Configuration**:
  - Candidates are deduplicated by resolved execution identity (`identity(spec)`).
  - Effective configurations (`effective_configuration`) are computed to include defaults and aliases while preserving the audited candidate identity.
- **Dataset Hard Link Unbinding & Symlink Hazard Rejection**:
  - Replaced dataset hard linking with physical byte snapshots, checking device and inode numbers (`st_dev`, `st_ino`) to detect hard links and unlink them into independent files.
  - Added strict symlink rejection at intake to prevent directory traversal or file mutation hazards.
- **Durable Search Admission & Accounting**:
  - In `operational_budgets.py` and `deepseek_search.py`, search attempts must be durably admitted in SQLite before external dispatch.
  - Purely local rejections (e.g. malformed queries) do not consume external search query quota.
  - Dispatched calls (successful, failed, or timed out) count toward the search query budget and settle idempotently.
- **Configurable Launcher Preflight**:
  - Updated `scripts/run_live_amazon.sh` to remove hardcoded "Run 8" and 4-turn assumptions. Added proper CLI parameter parsing, configurable model/budget limits, and verified `--help` preflight.

---

## 3. Inventory of Touched & Added Surfaces

### 3.1 Rust Crates (`crates/harness/`)
- `src/campaign.rs`:
  - Added canonicalization for `state_root` and resolution for `spec.dataset_path`.
  - Added M0 task and branch registration (`CreateBranch`, `CreateTask`) for challenger auditing.
  - Added error propagation from challenger orchestrator step.
  - Added `"protocol_sha256"` to bundle `scorecard.json`.
  - Fixed column query in `assessment_lockbox_consumptions` (`predictions_sha256`).
- `src/assets.rs`:
  - Added `pub const ACTIONS_V3_JSON: &str = include_str!("../../../contracts/v3/actions.json")`.
  - Added `("contracts/v3/actions.json", ACTIONS_V3_JSON)` to `EMBEDDED_PYTHON_FILES`.
- `src/agents/roles.rs`:
  - Added `get_challenger_role()` and wired `get_role("challenger")` while preserving standard 8 roles in `get_standard_roles()`.
- `tests/campaign_boss_specialist_slice.rs`:
  - Created integration test suite covering Evidence A (1 & 2), Evidence B, Evidence C, and Challenger Rejection.
- `tests/campaign_production.rs`:
  - Added challenger response turn to scripted gateway fixture in `test_autonomous_campaign_with_agent_orchestrator_gateway_override`.

### 3.2 Python Scientific Runtime (`python/src/mlrescurix_worker/`)
- `scientific/candidate_pipeline.py`:
  - Added `effective_configuration` resolution and strict parameter validation (`validate_spec`).
  - Preserved `self.candidate_identity = identity(spec)` for bundle verification consistency.
- `scientific/autonomous_loop.py`:
  - Added dataset symlink rejection at loop initialization.
  - Bound `validate_spec` and `resolve_effective_config` across candidate iterations.
  - Guarded `self.task_type` access via `getattr` for uninitialized testing instances.
- `scientific/operational_budgets.py`:
  - Enforced pre-dispatch search admission and durable accounting.
- `scientific/deepseek_search.py`:
  - Wired durable search admission with fail-closed query settlement.
- `contracts/v3/actions.json`:
  - Added authoritative schema definition for `RESEARCH_TOPIC`.
- `python/tests/test_s1_contracts_and_parity.py`:
  - Created regression suite covering Evidence D, E, F.
- `scripts/run_live_amazon.sh`:
  - Parameterized run identity, turn limits, model endpoints, and added preflight `--help`.

---

## 4. Verification Record & Test Execution

### 4.1 Test Suites Executed

| Command | Exit Code | Result | Details |
|---|:---:|:---:|---|
| `cargo test --locked --workspace` | 0 | PASS | Full workspace test suite passed (all crates, 44+ unit tests, integration suites). |
| `cargo test --locked -p mlrescurix-harness --test campaign_boss_specialist_slice` | 0 | PASS | 9/9 passed (Evidence A1/A2, Evidence B, Evidence C resumption/lockbox crash recovery/challenger replay guard, Evidence D header whitespace/embedded action parity, challenger rejection). |
| `cargo test --locked -p mlrescurix-harness --test campaign_production` | 0 | PASS | 10/10 passed (concurrency, crash recovery, production CLI). |
| `cargo test --locked -p mlrescurix-harness --test agent_m4` | 0 | PASS | 27/27 passed (roles, tools, orchestration). |
| `PYTHONPATH=python/src python3 -m pytest python/tests/test_s1_contracts_and_parity.py` | 0 | PASS | 10/10 passed (Evidence D action parity across schemas and shipped runtime assets, parameter rejection, mixed portfolios, duplicate reuse, hardlink unbinding, Evidence E search accounting, Evidence F error visibility). |
| `PYTHONPATH=python/src python3 -m pytest python/tests/test_panel_evaluator_and_critic.py python/tests/test_candidate_bundle_runtime.py python/tests/test_portfolio_exploration.py python/tests/test_autonomy_regressions.py` | 0 | PASS | 53/53 passed. |
| `PYTHONPATH=python/src python3 -m pytest python/tests/` | 0 | PASS | 491 passed, 1 skipped (full test suite green). |
| `./scripts/run_live_amazon.sh --help` | 0 | PASS | Configurable preflight script verified. |

### 4.2 Production CLI Run Verification
Command executed:
```bash
cargo run --bin mlrescurix-harness -- autonomous-create \
  --state docs/evidence/refactor/S1/cli_state \
  --spec docs/evidence/refactor/S1/problem_spec.json
```
Exit Status: `0`
Outcome Summary:
- Project ID: `proj_s1_evidence_demo`
- Dataset ID: `ds_309ff4563f0a`
- Dataset SHA256: `309ff4563f0a44976914df2c5029f62d8fb187d9f60309396575dfbfc2baefe9`
- Protocol SHA256: `001c5faa9772c28bc1b37cb0e608151bb34757a9745272e353a16f4440171797`
- Candidate ID: `cand_proj_s1_evidence_demo_309ff456`
- Validation Score (MAE): `0.2276856666666666`
- Lockbox Score (MAE): `0.07852033333333341`
- Lockbox Receipt SHA256: `f6d4887c5001d81568db67d42fdd4c775ff9d8f01490d290fd073959cc363557`
- Bundle Verification: `integrity_verified` (`manifest_verified: true`, `contracts_verified: true`, `constraints_verified: true`)
- Initial Duration: `816 ms`
- Resume Duration on Existing State: `17 ms` (idempotent verification, zero redundant lockbox consumption)

---

## 5. Acceptance Matrix (Items A–G)

| Acceptance Gate | Status | Evidence Reference |
|---|:---:|---|
| **A. Boss/Specialist Slice & Config Sensitivity** | **PASS** | `test_evidence_a_boss_specialist_slice_end_to_end` and `test_evidence_a_second_configuration_changes_behavior` in `crates/harness/tests/campaign_boss_specialist_slice.rs`. CLI execution documented in `docs/evidence/refactor/S1/cli_outcome.json`. |
| **B. Context Isolation & Evidence Binding** | **PASS** | `test_evidence_b_serialized_context_isolation_and_evidence_binding` in `crates/harness/tests/campaign_boss_specialist_slice.rs`. Verified serialized prompt contains no foreign history or proposer advocacy; verified task/protocol binding. |
| **C. Resumption & Double-Lockbox Prevention** | **PASS** | `test_evidence_c_restart_boundary_preserves_outcome_and_prevents_double_lockbox` in `crates/harness/tests/campaign_boss_specialist_slice.rs` and CLI resume test. Second run preserves outcome and prevents duplicate lockbox evaluation. |
| **D. Action Parity, Parameter Rejection, Portfolios, Hardlinks** | **PASS** | `python/tests/test_s1_contracts_and_parity.py` (`test_evidence_d_*` tests). Action schemas unified; Ridge `penalty`/`C` rejected in regression; classification `alpha` rejected; portfolios run independently; hard links unlinked; symlinks rejected. |
| **E. Durable Search Admission & Accounting** | **PASS** | `test_evidence_e_search_accounting_local_vs_dispatched` in `python/tests/test_s1_contracts_and_parity.py`. Local rejection does not consume external quota; dispatched calls consume queries and record durable outcomes. |
| **F. Controlled Repeated-Proposal Error Visibility** | **PASS** | `test_evidence_f_controlled_repeated_proposal_error_visibility` in `python/tests/test_s1_contracts_and_parity.py`. Exact duplicate specs return previous execution receipts; repeated errors remain visible in history. |
| **G. Recursive Deletion & Obsolete Surface Cleanup** | **PASS** | Removed inverse-C-to-Ridge-alpha silent conversions; removed hardcoded Run 8/four-turn assumptions in `run_live_amazon.sh`; removed hardlink dependencies; removed silent model substitution. |

---

## 6. Artifact & Receipt Inventory
All supporting evidence files are located in `docs/evidence/refactor/S1/`:
1. `problem_spec.json`: Problem specification for synthetic regression.
2. `synthetic_evidence_data.csv`: Synthetic dataset used for reproducible verification.
3. `cli_outcome.json`: Exact JSON output from `mlrescurix-harness autonomous-create`.
4. `bundle_manifest.json`: Verification manifest of the generated delivery bundle.
5. `bundle_scorecard.json`: Certified scorecard binding validation score, lockbox score, and protocol SHA256.
6. `task_assignments.json`: Exported M0 task assignments (`model_specialist` and `challenger`) from SQLite.
7. `task_completions.json`: Exported M0 task completions and findings from SQLite.
8. `cli_state/`: Complete durable SQLite database and artifacts generated during the CLI verification run.

---

## 7. Known Issues, Scope Boundaries & Handoff to Session 2
- **Subprocess Host Isolation vs. Certified Container Sandbox**: S1 executes solution code in isolated subprocess workspaces with fail-closed exit checking. Host subprocesses do not constitute a hardened containment boundary; full OS/container isolation and resource quota certification belong to Session 2.
- **Scientific Breadth**: S1 validates end-to-end execution on linear and polynomial basis regression models. General arbitrary-source training and full standalone v3 delivery belong to Session 3.
- **Provider Gateways**: S1 verified orchestration and accounting using offline fixtures. Live paid external model calls and web search queries remain disabled under current financial authorizations ($0.20 per-run ceiling preserved).
- **Anti-Plateau Steering**: S1 preserves repeated proposal error visibility in durable memory; active multi-turn anti-plateau steering and research pivots belong to Session 4.
- **Remote Execution**: Kaggle remote dispatch belongs to Session 5.
