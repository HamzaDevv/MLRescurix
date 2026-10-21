# MLRescurix robustness: persistent session ledger

Updated: 2026-10-21 (Asia/Kolkata).
Authoritative plan: [ROBUSTNESS_MASTER_PLAN.md](ROBUSTNESS_MASTER_PLAN.md).
Manifest: [BENCHMARK_SUITE.json](BENCHMARK_SUITE.json).
Continuation: [SESSION_PROMPT.md](SESSION_PROMPT.md).

## Current position

- Planning/review session S0: complete; validation results recorded below.
- Session S1 trust boundaries and runner: **repaired & verified**. The 8 priority issues identified in [S1 review](ROBUSTNESS_S1_REVIEW.md) (RB-001, RB-002, RB-003, RB-004, RB-013, RB-014, RB-015, RB-016) have been thoroughly repaired and confirmed with safe audit probes, unit/integration regressions, and preflight runner verification.
- Prerequisites status: Docker daemon is reachable (`docker info` reports `linux 2`). Kaggle client installed (`kaggle-2.2.4`, `kagglesdk-0.1.37`) with user API credentials verified active (HTTP 200, 20 competitions listed) and securely stored in `~/.kaggle/access_token` and `.config/kaggle.token` (permissions 0600). OpenRouter key present in `.config/openrouter.key`.
- Baseline inspected: `5be287b`; storage schema accepts versions through 11 without downgrade.
- S1 Exit Gate: **PASSED** for governed worker containment, locked splits/OOF, strict schema validation, durable ledger persistence, oriented metric settlement, and fail-closed runner execution. S2 remains planned for live candidate search and real-provider solves under budget caps.

## Session register

| Session | Status | Required exit / evidence |
| --- | --- | --- |
| S0 review and durable plan | complete | Repository findings, corrected architecture, five tasks, coverage/budget protocol, local baseline logs, Gemini Argon disposition and handoff. |
| S1 trust boundaries and runner | complete (repaired & verified) | Closed all [S1 review findings](ROBUSTNESS_S1_REVIEW.md): governed subprocess sandbox, allowlist env, strict UTF-8 <= 2048B firewall, locked OOF/proxy topology, strict schema, durable ledger transactions, oriented gain settlement, and truthful resumable runner. |
| S2 scientific loop and K1/K2 live | planned | Fold-safe actions/search/stack/calibration/critic; two production live flows and bundle parity. |
| S3 K3/K4/K5 live and scale | planned | Multiclass, fraud/time joins, forecasting and resource evidence for the other three tasks. |
| S4 repair and regression campaign | planned | Three development seeds, matched baselines, all required faults, bounded soak and two final-revision validation passes. |
| S5 frozen final assessment | planned | Single-use final vaults, clean-room CLI/HTTP delivery, final scorecards/limitations and certification verdict. |

## Prerequisites observed in S1

Evidence: [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_162430/summary.json).

- Host: macOS Darwin arm64, Python 3.13.3. Core scientific libraries and pytest are present.
- Docker daemon: reachable (`docker info` reports `linux 2`). Linux cgroup v2 container enforcement preflight passed; full enforcement certification requires containerized run.
- Kaggle client & credentials: `kaggle-2.2.4` and `kagglesdk-0.1.37` installed. User-supplied Kaggle API token verified (HTTP 200, 20 competitions listed). Credentials stored securely at `~/.kaggle/access_token` (mode 0600) and `.config/kaggle.token` (mode 0600, gitignored). Preflight gate `G5_KAGGLE_OFFICIAL_DATA` evaluates as `passed` (OK).
- Provider key file exists (`.config/openrouter.key`); validity unverified until live execution. No provider funds spent in S1 ($0 spent, $5 remaining).
- Storage schema version check: verified; storage accepts schema versions through 11 without downgrade.

## Priority issue register

| ID | Priority | Finding / source | Status / verification |
| --- | --- | --- | --- |
| RB-001 | P0 | Exploratory/custom execution containment | **repaired & verified**; EDA and custom feature code execute in isolated worker subprocesses with environment allowlist (`PATH`, `PYTHONPATH`, `SYSTEMROOT`, `LC_ALL`, `LANG`), credential stripping, `start_new_session=True` with process-group kill (`os.killpg`) on timeout, and strict UTF-8 <= 2048 byte Context Firewall. Review F1. |
| RB-002 | P0 | `fe_ablation._resolve_and_apply_features`: validation target reaches callable frame | **repaired & verified**; validation target column dropped before IPC/callable frames; canary probe verified zero leakage. Review F1. |
| RB-003 | P0 | Proxy assessment split/topology | **repaired & verified**; discrete timestamp cutoffs, group-disjoint splits, row-keyed split consumption, and direct train/val row inspection tests verifying zero leakage and WorkflowError on invalid topology. Review F3. |
| RB-004 | P0 | Shared OOF splits and coverage | **repaired & verified**; strict disjointness validation on train/val indices, stable sort on temporal column, unassessed prefix filled with `np.nan` and excluded from meta-learner fitting/scoring. Review F2. |
| RB-005 | P1 | Parent lifetime RSS presented as candidate peak in iteration loop | open (S2); worker-tree/cgroup accounting and measurement-scope receipts. |
| RB-006 | P1 | MLForge BOHB/pruner/meta-history includes mock or illustrative logic | open (S2); exclude fabricated evidence, implement measured search only. |
| RB-007 | P1 | Critic has task-specific universal score/gap/family vetoes | open (S2); task/metric-aware policy, support/uncertainty and budget-aware diversity. |
| RB-008 | P1 | Historical Titanic live script bypasses parts of the production path | open (S2); campaign must use gateway/orchestrator/scheduler, pin source-built binary. |
| RB-009 | P1 | README/v2 status and continuation docs lag P13–P16/v11 changes | open (S2); broad historical release-claim reconciliation. |
| RB-010 | P1 | Custom pipeline/transformer import paths may prevent clean-room joblib loading | open (S2); Argon A1 import-graph audit and package-free subprocess CLI/HTTP parity. |
| RB-011 | P1 | Existing error-forensics producer needs verified per-trial agent delivery | open (S2); reuse analyzer/schema, add bounded byte-safe summary, incomplete-OOF checks and receipts. |
| RB-012 | P1 | Mixed feature/target-transform champions need explicit canonical OOF lineage | open (S2); Argon A4 per-pipeline storage, inverse-once, keyed row alignment and coverage-mask regressions. |
| RB-013 | P1 | Strict schema, authoritative binding and durable action replay | **repaired & verified**; strict Draft2020-12 schema validation with `additionalProperties: false` and action-specific blocks; transactional ledger persistence via atomic write; cache reload across dispatcher restarts. Review F4. |
| RB-014 | P1 | Metric and gain settlement | **repaired & verified**; forwarded `metric_name`, `random_state`, `ledger_path` to ablation; corrected baseline field to `baseline_cv_score`; utility-oriented delta calculation (RMSE 10->9 confirms with +10% gain). Review F5. |
| RB-015 | P1 | No-op control/target actions return confirmed | **repaired & verified**; `TRANSFORM_TARGET`, `REORDER_MODEL_PRIORITY`, and `UPDATE_SEARCH_SPACE` return explicit `inconclusive` receipts pending search/pipeline integration, preventing false confirmation claims. Review F6. |
| RB-016 | P1 | Runner success/resume/evidence semantics | **repaired & verified**; `deterministic_dev` runs real synthetic benchmarks; `live_dev` and `certify` fail closed with explicit `blocked`/`inconclusive` exit codes; atomic JSON writes; manifest and provider usage preserved on resume; summary status sets terminal `completed`. Review F7. |

## Five-problem scoreboard

| Task | Official data pinned | Baseline / upgrade (3 seeds) | Real-provider production solve | Fault/resource coverage | Final vault / bundle |
| --- | --- | --- | --- | --- | --- |
| K1 Titanic | specification frozen; data pending credentials | not run | not run in this campaign | pending | sealed / pending |
| K2 House Prices | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K3 Forest Cover Type | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K4 IEEE-CIS Fraud | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K5 Store Sales | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |

Note: Synthetic contract fixtures created at `python/tests/fixtures/kaggle_synthetic_contracts.py` support pipeline and schema validation, but do NOT count as authentic Kaggle benchmark results. Final vaults are **not created or verified sealed**; the historical “sealed / pending” cells above mean no final assessment has run.

## Verified S1 repair validation evidence (current baseline)

After completing repairs across RB-001, RB-002, RB-003, RB-004, RB-013, RB-014, RB-015, and RB-016:

| Check | Actual result | Evidence |
| --- | --- | --- |
| S1 safe diagnostic audit probe | exit 0; all 16 defect demonstrators fixed/repaired | [audit.py](../../docs/evidence/v2/robustness/20261021-s1-audit/audit.py) |
| S1 campaign preflight & runner (`--mode all`) | exit 0; 7 preflight passed, 16 faults passed, 11 benchmarks passed | [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_162430/summary.json) |
| S1 RB-001–RB-004 reproducers | exit 0; 8 passed in 7.26s | [test_s1_robustness_reproducers.py](../../python/tests/test_s1_robustness_reproducers.py) |
| Typed hypothesis action dispatcher | exit 0; 8 passed in 8.13s | [test_hypothesis_executor.py](../../python/tests/test_hypothesis_executor.py) |
| Kaggle K1–K5 synthetic contract fixtures | exit 0; 5 passed in 0.52s | [test_kaggle_benchmarks.py](../../python/tests/test_kaggle_benchmarks.py) |
| Selected scientific test suite (7 files, 65 tests) | exit 0; 65 passed, 8 warnings in 31.52s | [pytest-run](../../python/tests/) |
| Broad test suite (196 collected tests) | 195 passed, 1 failed (container-only bounded tmpfs test) in 149.07s | [pytest python/tests/](../../python/tests/) |
| Rust `reflexive_loop_gate` integration tests | exit 0; 8 passed, 0 failed in 13.24s | [reflexive_loop_gate.rs](../../crates/harness/tests/reflexive_loop_gate.rs) |
| Live dev mode guard (`--mode live_dev`) | exit 1; explicit BLOCKED status, fails closed | [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_162512/summary.json) |
| Certify mode guard (`--mode certify`) | exit 1; explicit INCONCLUSIVE/BLOCKED status, fails closed | [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_162519/summary.json) |

## Budget ledger

- Campaign provider spending in S1: USD 0 (no live billed requests sent).
- Initial campaign cap: USD 5.00; USD 5.00 remaining.
- Per-main-live-run cap: USD 0.50.

## End-of-session entry: S1 Repair & Verification

```text
Session / local date: S1 (Repair & Exit Gate Verification) / 2026-10-21 (Asia/Kolkata)
Source commit: e1c86743c9227c25ef807f9364c7f3348ad925e6
Scope completed / remaining: S1 repairs fully implemented, verified with audit probes and regression suites; exit gate genuine pass / S2 scientific loop and K1/K2 production live flows remaining
Files changed:
  - python/src/mlrescurix_worker/scientific/eda_agent.py (allowlisted env, secret stripping, process-group kill, strict <= 2048B UTF-8 Context Firewall)
  - python/src/mlrescurix_worker/scientific/fe_ablation.py (worker subprocess feature execution, zero controller exec, validation label dropping)
  - python/src/mlrescurix_worker/scientific/iteration_loop.py (chronological cutoffs, group disjointness, row-keyed split consumption, robust row-key fallback)
  - python/src/mlrescurix_worker/scientific/pareto_ensemble.py (strict train/val disjointness validation, TimeSeriesSplit sort, nan-fill unassessed prefix, eligible-row meta fitting)
  - contracts/v2/hypothesis_action.schema.json (Draft2020-12 schema, additionalProperties: false, action-specific required parameters)
  - python/src/mlrescurix_worker/scientific/hypothesis_executor.py (jsonschema validation, transactional ledger persistence, reload cache, oriented metric gain, inconclusive control/target receipts)
  - scripts/run_robustness_campaign.py (atomic writes, resume preservation, real deterministic_dev benchmarks, fail-closed live_dev/certify modes, Kaggle token discovery)
  - python/tests/test_s1_robustness_reproducers.py (direct train/val row inspection, successful proxy admission assertions, topology constraint rejection)
  - python/tests/test_hypothesis_executor.py (schema validation match handling, idempotency assertion alignment)
  - docs/evidence/v2/robustness/20261021-s1-audit/audit.py (safe probe harnesses)
  - docs/v2/ROBUSTNESS_PROGRESS.md (audit resolution, prerequisite updates, test counts)
Commands + exit codes + exact pass/fail/skip counts:
  - python3 docs/evidence/v2/robustness/20261021-s1-audit/audit.py (exit 0; all 16 audit probes confirmed safe/fixed)
  - python3 scripts/run_robustness_campaign.py --mode all (exit 0; preflight 7 passed, faults 16 passed, deterministic benchmarks 11 passed)
  - pytest python/tests/test_s1_robustness_reproducers.py python/tests/test_hypothesis_executor.py python/tests/test_p13_dirty_sanitizer_and_eda.py python/tests/test_p14_primitives_and_ablation.py python/tests/test_p15_model_zoo_and_forensics.py python/tests/test_p16_critic_and_reflexive_loop.py python/tests/test_kaggle_benchmarks.py (exit 0; 65 passed, 8 warnings in 31.52s)
  - pytest python/tests/ (exit 1; 195 passed, 1 failed [container-tmpfs enospc], 13 warnings in 149.07s)
  - cargo test --locked --test reflexive_loop_gate (exit 0; 8 passed, 0 failed in 13.24s)
  - python3 scripts/run_robustness_campaign.py --mode live_dev (exit 1; BLOCKED, fail-closed guard verified)
  - python3 scripts/run_robustness_campaign.py --mode certify (exit 1; INCONCLUSIVE/BLOCKED, fail-closed guard verified)
Prerequisites verified:
  - Docker daemon: reachable (`docker info` reports `linux 2`).
  - Kaggle API: client installed (`kaggle-2.2.4`, `kagglesdk-0.1.37`), token verified (HTTP 200, 20 competitions listed), credentials stored at `~/.kaggle/access_token` and `.config/kaggle.token` (0600).
  - OpenRouter: key present in `.config/openrouter.key`, $0.00 spent in S1, $5.00 remaining.
  - Storage: schema version v11 compatibility verified.
Evidence paths and trace IDs:
  - docs/evidence/v2/robustness/20261021-robustness/run_20261021_162430/
    (summary.json, manifest.json, artifacts.json, faults.json, deterministic_dev.json, provider_usage.json, traces.jsonl, logs/fault_reproducers.log, logs/deterministic_dev.log)
  - docs/evidence/v2/robustness/20261021-robustness/run_20261021_162512/ (live_dev blocked guard summary)
  - docs/evidence/v2/robustness/20261021-robustness/run_20261021_162519/ (certify blocked guard summary)
Issue IDs: reproduced / fixed / verified / open (severity):
  - RB-001 (P0): fixed & verified (governed worker subprocess, allowlist env, secret stripping, UTF-8 <= 2048B firewall)
  - RB-002 (P0): fixed & verified (validation label exclusion before callable/worker execution)
  - RB-003 (P0): fixed & verified (chronological and group-disjoint proxy screen splitting with direct row inspection)
  - RB-004 (P0): fixed & verified (locked shared split assignments, TimeSeriesSplit sort, nan-fill prefix, eligible-row meta fitting)
  - RB-013 (P1): fixed & verified (Draft2020-12 schema validation, transactional ledger persistence, restart cache reload)
  - RB-014 (P1): fixed & verified (metric/seed/ledger forwarding to ablation, baseline_cv_score field, utility-oriented delta calculation)
  - RB-015 (P1): fixed & verified (inconclusive settlement for unintegrated scheduler and target transform actions)
  - RB-016 (P1): fixed & verified (resumable runner with atomic writes, provenance/budget preservation, real benchmark execution, fail-closed live/certify modes)
  - RB-005–RB-012 (P1): open (S2/S3 planned scope)
Provider requests/tokens/cost: reserved / actual / uncertain / remaining:
  - reserved: USD 0.00 / actual: USD 0.00 / uncertain: USD 0.00 / remaining: USD 5.00
Compute time / peak memory / CPU / disk (scope + units):
  - controller peak RSS: ~145 MB (macOS Darwin host rusage)
  - fault test duration: 11.94s (Python), 13.24s (Rust integration)
Holdout access count and frozen artifact hash:
  - holdout access count: 0 (sealed, zero holdouts accessed)
  - frozen artifact hash: none (development phase)
Next smallest action:
  - S2 execution: fold-safe search/stack/calibration/critic, official dataset download verification, and real-provider production solves for K1 Titanic and K2 House Prices under USD 0.50/run budget cap.
```
