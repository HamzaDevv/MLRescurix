# MLRescurix robustness: persistent session ledger

Updated: 2026-10-21 (Asia/Kolkata).
Authoritative plan: [ROBUSTNESS_MASTER_PLAN.md](ROBUSTNESS_MASTER_PLAN.md).
Manifest: [BENCHMARK_SUITE.json](BENCHMARK_SUITE.json).
Continuation: [SESSION_PROMPT.md](SESSION_PROMPT.md).

## Current position

- Planning/review session S0: complete; validation results recorded below.
- Session S1 trust boundaries and runner: **complete**; all S1 gates passed, evidence recorded below.
- Baseline inspected: `5be287b`; storage schema accepts versions through 11.
- Historical P1–P16/M6 evidence remains available; it does not close this campaign.

## Session register

| Session | Status | Required exit / evidence |
| --- | --- | --- |
| S0 review and durable plan | complete | Repository findings, corrected architecture, five tasks, coverage/budget protocol, local baseline logs, Gemini Argon disposition and handoff. |
| S1 trust boundaries and runner | complete | EDA isolation, label firewall, split propagation, DSL, provenance/split manifests and repeatable runner. |
| S2 scientific loop and K1/K2 live | planned | Fold-safe actions/search/stack/calibration/critic; two production live flows and bundle parity. |
| S3 K3/K4/K5 live and scale | planned | Multiclass, fraud/time joins, forecasting and resource evidence for the other three tasks. |
| S4 repair and regression campaign | planned | Three development seeds, matched baselines, all required faults, bounded soak and two final-revision validation passes. |
| S5 frozen final assessment | planned | Single-use final vaults, clean-room CLI/HTTP delivery, final scorecards/limitations and certification verdict. |

## Prerequisites observed in S1

Evidence: [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_145922/summary.json).

- Host: macOS Darwin arm64, Python 3.13.3. Core scientific libraries and pytest are present.
- Docker CLI exists, but daemon connectivity failed (`Cannot connect to the Docker daemon`). Container cgroup v2 enforcement check marked `blocked`.
- Kaggle client and checked credential locations are absent. Official competition access marked `blocked`.
- Provider key file exists (`.config/openrouter.key`); validity unverified until live execution. No provider funds spent in S1.
- Storage schema version check: verified; storage accepts schema versions through 11 without downgrade.

## Priority issue register

| ID | Priority | Finding / source | Status / verification |
| --- | --- | --- | --- |
| RB-001 | P0 | `eda_agent.run_exploratory_python`: in-process exec, no enforced timeout, shallow source copy | **fixed & verified**; routed through isolated subprocess worker with hard timeout kill, zero parent DataFrame mutation, and column exclusion checks. Tested in `test_s1_robustness_reproducers.py`. |
| RB-002 | P0 | `fe_ablation._resolve_and_apply_features`: validation target reaches callable frame | **fixed & verified**; validation labels stripped (`val_df_no_target`) before candidate callables; canary test proves validation labels inaccessible in callable frames. Tested in `test_s1_robustness_reproducers.py`. |
| RB-003 | P0 | `iteration_loop.run_proxy_screen`: topology sampling followed by random train/validation split | **fixed & verified**; chronological split for temporal data and group-disjoint split for grouped data with zero-leakage assertions. Tested in `test_s1_robustness_reproducers.py`. |
| RB-004 | P0 | `pareto_ensemble._collect_oof_predictions`: independent random splits | **fixed & verified**; consumes locked shared `split_assignments` and topology group/temporal structures. Tested in `test_s1_robustness_reproducers.py`. |
| RB-005 | P1 | Parent lifetime RSS presented as candidate peak in iteration loop | open (S2); worker-tree/cgroup accounting and measurement-scope receipts. |
| RB-006 | P1 | MLForge BOHB/pruner/meta-history includes mock or illustrative logic | open (S2); exclude fabricated evidence, implement measured search only. |
| RB-007 | P1 | Critic has task-specific universal score/gap/family vetoes | open (S2); task/metric-aware policy, support/uncertainty and budget-aware diversity. |
| RB-008 | P1 | Historical Titanic live script bypasses parts of the production path | open (S2); campaign must use gateway/orchestrator/scheduler, pin source-built binary. |
| RB-009 | P1 | README/v2 status and continuation docs lag P13–P16/v11 changes | open (S2); broad historical release-claim reconciliation. |
| RB-010 | P1 | Custom pipeline/transformer import paths may prevent clean-room joblib loading | open (S2); Argon A1 import-graph audit and package-free subprocess CLI/HTTP parity. |
| RB-011 | P1 | Existing error-forensics producer needs verified per-trial agent delivery | open (S2); reuse analyzer/schema, add bounded byte-safe summary, incomplete-OOF checks and receipts. |
| RB-012 | P1 | Mixed feature/target-transform champions need explicit canonical OOF lineage | open (S2); Argon A4 per-pipeline storage, inverse-once, keyed row alignment and coverage-mask regressions. |

## Five-problem scoreboard

| Task | Official data pinned | Baseline / upgrade (3 seeds) | Real-provider production solve | Fault/resource coverage | Final vault / bundle |
| --- | --- | --- | --- | --- | --- |
| K1 Titanic | specification frozen; data pending credentials | not run | not run in this campaign | pending | sealed / pending |
| K2 House Prices | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K3 Forest Cover Type | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K4 IEEE-CIS Fraud | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |
| K5 Store Sales | specification frozen; data pending credentials | not run | not run | pending | sealed / pending |

Note: Synthetic contract fixtures created at `python/tests/fixtures/kaggle_synthetic_contracts.py` support pipeline and schema validation, but do NOT count as authentic Kaggle benchmark results.

## S1 validation evidence

| Check | Actual result | Evidence |
| --- | --- | --- |
| S1 campaign preflight & runner | exit 0; 5 passed, 2 blocked (Docker daemon, Kaggle credentials) | [summary.json](../evidence/v2/robustness/20261021-robustness/run_20261021_145922/summary.json) |
| S1 RB-001–RB-004 reproducers | exit 0; 8 passed in 5.21s | [test_s1_robustness_reproducers.py](../../python/tests/test_s1_robustness_reproducers.py) |
| Typed hypothesis action dispatcher | exit 0; 8 passed in 2.34s | [test_hypothesis_executor.py](../../python/tests/test_hypothesis_executor.py) |
| Kaggle K1–K5 synthetic contract fixtures | exit 0; 5 passed in 0.52s | [test_kaggle_benchmarks.py](../../python/tests/test_kaggle_benchmarks.py) |
| Full scientific test suite (P13–P16 + S1) | exit 0; 65 passed, 8 warnings in 18.73s | [pytest-run](../../python/tests/) |
| Rust `reflexive_loop_gate` integration tests | exit 0; 8 passed, 0 failed/ignored in 13.35s | [reflexive_loop_gate.rs](../../crates/harness/tests/reflexive_loop_gate.rs) |

## Budget ledger

- Campaign provider spending in S1: USD 0 (no live billed requests sent).
- Initial campaign cap: USD 5.00; USD 5.00 remaining.
- Per-main-live-run cap: USD 0.50.

## End-of-session entry: S1

```text
Session / local date: S1 / 2026-10-21 (Asia/Kolkata)
Source commit + dirty diff hash: 5be287b1abf4d3e3140510f4a633072e1519f820 + e053433c02999a1c8670fbd187ca08bd25ea419dcf80822f3a489d1d073358c5
Scope completed / remaining: S1 trust boundaries, reproducers, typed action contract, runner, and benchmark freezing complete / S2 scientific loop and K1/K2 live flows remaining
Files changed:
  - python/src/mlrescurix_worker/scientific/eda_agent.py
  - python/src/mlrescurix_worker/scientific/fe_ablation.py
  - python/src/mlrescurix_worker/scientific/iteration_loop.py
  - python/src/mlrescurix_worker/scientific/pareto_ensemble.py
  - contracts/v2/hypothesis_action.schema.json
  - python/src/mlrescurix_worker/scientific/hypothesis_executor.py
  - scripts/run_robustness_campaign.py
  - docs/v2/kaggle_benchmarks/README.md
  - python/tests/fixtures/kaggle_synthetic_contracts.py
  - python/tests/fixtures/__init__.py
  - python/tests/test_s1_robustness_reproducers.py
  - python/tests/test_hypothesis_executor.py
  - python/tests/test_kaggle_benchmarks.py
  - docs/v2/ROBUSTNESS_PROGRESS.md
Commands + exit codes + exact pass/fail/skip counts:
  - python3 scripts/run_robustness_campaign.py --mode all (exit 0; preflight 5 passed, 2 blocked; faults 16 passed, 0 failed)
  - pytest python/tests/ (exit 0; 65 passed, 8 warnings, 0 failed in 18.73s)
  - cargo test --locked --test reflexive_loop_gate -- --test-threads=2 (exit 0; 8 passed, 0 failed in 13.35s)
Data/split/image/binary/config hashes:
  - debug_binary_sha256: 09d31fefce88b90c74fdb47db8b211a761e8c95a281861788cba8c63f8d95166
  - dist_binary_sha256: 1a9f07a0d42129037feaa23326756cc2c53fcfd466e319a2b724f7e26cf8d8ad
  - benchmark_manifest_sha256: 58021d7b39b56f8f0414f5cfbcba7471239c43ba780fc239c57d7681ebbf8f47
  - python_worker_sha256: 3c57eb0567e7811ef1432f8eb4f9103c8c7c77e7740e2cf2d3dcf7d612e4d079
Evidence paths and trace IDs:
  - docs/evidence/v2/robustness/20261021-robustness/run_20261021_145922/
    (summary.json, manifest.json, artifacts.json, faults.json, provider_usage.json, traces.jsonl, logs/fault_reproducers.log)
Issue IDs: reproduced / fixed / verified / open (severity):
  - RB-001 (P0): reproduced / fixed / verified (subprocess worker isolation, timeout kill, zero parent mutation, label exclusion)
  - RB-002 (P0): reproduced / fixed / verified (validation labels stripped before candidate callable frames, canary verified)
  - RB-003 (P0): reproduced / fixed / verified (chronological and group-disjoint proxy screen splitting with zero-leakage assertion)
  - RB-004 (P0): reproduced / fixed / verified (locked shared split assignments consumed in Pareto ensemble)
  - RB-005–RB-012 (P1): open
Provider requests/tokens/cost: reserved / actual / uncertain / remaining:
  - reserved: USD 0.00 / actual: USD 0.00 / uncertain: USD 0.00 / remaining: USD 5.00
Compute time / peak memory / CPU / disk (scope + units):
  - controller peak RSS: 142 MB (macOS Darwin host rusage, not Linux cgroup v2)
  - fault test duration: 5.85s (Python), 13.35s (Rust integration)
Holdout access count and frozen artifact hash:
  - holdout access count: 0 (sealed, zero holdouts accessed)
  - frozen artifact hash: none (development phase)
Next smallest action / external prerequisites:
  - S2 execution: fold-local caching, quarantine ablations, metric-aware critic, and real-provider production solve for K1/K2
  - External prerequisites: Docker daemon connectivity for cgroup v2; Kaggle credentials for official dataset downloads
```
