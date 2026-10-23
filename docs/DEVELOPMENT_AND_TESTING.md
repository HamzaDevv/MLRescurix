# Development and Testing Guide

This document describes the reproducible offline development commands, supported environments, static verification checks, and exact validation limits for MLRescurix.

---

## 1. Supported Environments & Prerequisites

MLRescurix is engineered and tested for:
- **macOS Darwin arm64** (Apple Silicon)
- **Linux x86_64** (Ubuntu 22.04 LTS / Debian 12)

### Required Toolchains
- **Rust Toolchain**: Rust 1.80+ (stable, 2021 edition) with `cargo`.
- **Python Runtime**: CPython 3.11+ (tested with CPython 3.13) with `pip` and virtual environment support.

### Docker Runtime Status
- **Current Daemon Status**: The Docker daemon is currently **unreachable / not running** on this development host (`docker info` fails to connect to the unix socket `unix:///Users/hamza/.docker/run/docker.sock`).
- **Consequences**:
  - The unchanged Rust scheduler crash-recovery fixture (`crates/harness/tests/agent_m4.rs::p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions`) fails scientific container admission when Docker is unavailable. This test failure is expected and retained in `rust-docker-blocker.log`; it is **not** masked or counted as a pass.
  - Offline tests that do not depend on active Docker containers (unit tests, offline evaluation corrections, adapter verification, and local process group workers) execute and pass cleanly.

---

## 2. Reproducible Offline Test Commands

All commands below are strictly offline and make zero external network requests or paid LLM API calls.

### 2.1 Python Test Suites

```sh
# 1. Run the final affected Python regression suite (105 tests, ~50s)
# Validates architecture consolidation, autonomous loop, task adapters, governance, and prerequisite guards:
PYTHONPATH=python/src python3 -m pytest \
  python/tests/test_architecture_consolidation.py \
  python/tests/test_autonomous_agent_loop.py \
  python/tests/test_autonomy_regressions.py \
  python/tests/test_rust_gateway_adapter.py \
  python/tests/test_prerequisite_repairs.py \
  python/tests/test_panel_evaluator_and_critic.py \
  python/tests/test_robustness_runner_gates.py \
  python/tests/test_s1_followup_guards.py -q

# 2. Run the full Python test suite (325 passing, 2 skipped, ~220s)
PYTHONPATH=python/src python3 -m pytest python/tests -q
```

*Note on skips*: The 2 skipped tests require an explicitly bounded 64 MiB tmpfs mount and an active Docker daemon, respectively.

### 2.2 Rust Harness Test Suites

```sh
# 1. Run core Rust unit and integration tests (63 passed, 2 ignored, 1 filtered)
cargo test --locked --offline --lib --test agent_m4 --test reflexive_loop_gate \
  -- --skip p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions

# 2. Run installed-runtime integration recheck (1 passed)
# Verifies that embedded Python runtime assets materialize and support controller + human interface outside source tree:
cargo test --locked --offline --lib \
  assets::tests::installed_runtime_supports_common_controller_and_human_interface_without_checkout
```

*Note on ignored/filtered tests*:
- `2 ignored`: Explicit digest-pinned Linux Docker cgroup fixtures; require configured Linux container environment.
- `1 filtered`: The unchanged scheduler crash-recovery fixture (`p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions`) attempted and failed container admission because Docker is unavailable.

### 2.3 Staging & Contract Reproduction Commands

Reproduce official data imports and panel protocol commitments without fitting models or spending budget:

```sh
# 1. Reproduce hash-verified development staging for K1 (zero holdout labels scored)
python3 scripts/stage_benchmark_development.py \
  --task K1 \
  --destination /tmp/mlrescurix-development/K1.csv

# 2. Reproduce and verify frozen K5 panel protocol (byte-for-byte identical)
python3 scripts/freeze_k5_panel_contract.py --output /tmp/K5_PANEL_PROTOCOL.json
cmp docs/v2/K5_PANEL_PROTOCOL.json /tmp/K5_PANEL_PROTOCOL.json

# 3. Run campaign accounting read-only reconciliation
# Reconciles all 29 historical runs against the preserved USD 0.050724020 total:
python3 scripts/reconcile_campaign_accounting.py
```

### 2.4 Static Compilation Checks

```sh
# Verify Python syntax and bytecode compilation across all source and test trees
python3 -m compileall python/src python/tests scripts

# Verify Rust compilation in locked, offline mode
cargo build --locked --offline -p mlrescurix-harness
```

---

## 3. Validation Limits & Evidence Baseline

The validation baseline recorded in [`docs/evidence/v2/autonomy/20261023-architecture-consolidation/summary.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/autonomy/20261023-architecture-consolidation/summary.json) reflects the following exact counts:

| Check Category | Execution Command | Result | Duration | Scope & Limitations |
|---|---|:---:|:---:|---|
| **Full Python Suite** | `pytest python/tests -q` | **325 passed, 2 skipped** | 220.03s | Completed before final source guards. Skips: bounded 64m tmpfs and Docker daemon. |
| **Final Affected Python** | 8 affected test modules | **105 passed** | 50.17s | Includes 18 new architecture regressions and final source-binding guards. |
| **Rust Lib & Integration** | `cargo test --lib --test agent_m4 --test reflexive_loop_gate` | **63 passed, 2 ignored, 1 filtered** | ~67s | 2 ignored require Linux Docker; 1 filtered was attempted and failed due to unavailable Docker daemon. |
| **Installed-Runtime Recheck** | `assets::tests::installed_runtime...` | **1 passed** | 3.21s | Materializes Python files outside repo and verifies controller/human interface. |
| **Rust Build** | `cargo build --locked --offline` | **Passed** | ~4s | Zero compilation errors or broken dependencies. |
| **Python Compileall** | `python3 -m compileall` | **Passed** | ~2s | Zero syntax or indentation errors. |

### Critical Boundaries on What Tests Prove

> [!WARNING]
> **Synthetic Fixtures vs Scientific Quality**:
> - Offline tests utilize synthetic provider fixtures and stubbed critic approvals to prove **control wiring, state transitions, cryptographic identity checks, and boundary enforcement**.
> - They do **NOT** prove machine learning model generalization, scientific superiority, or benchmark quality on real data.

> [!CAUTION]
> **Host Execution vs Production Certification**:
> - Local worker evaluations, refits, and fresh-interpreter bundle inferences on the development host prove functional execution and packaging integrity.
> - They do **NOT** certify production clean-room delivery, multi-tenant scheduling, concurrent cgroup v2 containment, soak stability, or physical fault resilience.
> - The cached scientific Docker image (`mlrescurix-scientific@sha256:d094b6be...`) currently lacks `jsonschema==4.26.0` and `psutil==6.1.1` Linux wheels, preventing in-container bounded tmpfs certification.
