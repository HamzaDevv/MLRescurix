# Phase P12 Session 1 Evidence & Acceptance Closure: Gate 3, Warm Pool (P9.7) & Gate 5

**Date:** 2026-10-17  
**Scope:** Session 1 of 2-session completion plan (Decision D48 / D49)  
**Status:** **PASSED / CLOSED** for Gate 3, Warm Worker Pool (P9.7), and Gate 5 (Platform Memory Certification).  
**M6 Overall Status:** NOT READY pending Session 2 Gate 4 benchmarks.

---

## 1. Release Gate 3: Clean Installed End-to-End Delivery & Upgrade Verification

- **Wheel Build & Distribution Artifacts**:
  - Python worker wheel built via `python -m build --wheel`: `python/dist/mlrescurix_worker_protocol-0.1.0-py3-none-any.whl`.
  - Production harness compiled via `cargo build --release`: `target/release/mlrescurix-harness`.
- **Standalone Execution Outside Git Tree**:
  - Verified in `crates/harness/tests/installed_delivery_gate3.rs`:
    - Prefix created in disposable temp directory `/tmp/mlrescurix-gate3-verify-...` with dedicated virtualenv and release binary installed in `prefix/bin`.
    - Negative guard verified: `git status` fails inside workspace, proving zero dependence on git repository sources.
    - Dataset profiling executed standalone (`profile-dataset` CLI).
    - Candidate delivery bundle packaged standalone (`package-candidate-bundle` CLI) with all 13 contracts verified.
    - Standalone CLI inference executed (`python inference.py --input ... --output ...`).
    - Standalone HTTP serving executed (`python serve.py --port ...`) with `/health` (200 OK) and `/predict` (200 OK) verified.
    - Standalone bundle verification executed (`mlrescurix-harness verify-bundle --bundle ... --sandbox=false`) verifying manifest integrity, contract validity, and zero untracked file contamination.
- **Restart Recovery & Schema Upgrade Consistency**:
  - Initialized state via installed binary (`mlrescurix-harness init --state ... --policy ...`).
  - Verified automatic migration to `PRAGMA user_version = 10` without manual migration steps or schema version bump.
  - Repeatedly restarted and executed capabilities and inspection across simulated restart cycles with persistent integrity.
- **Cryptographic Anti-Tampering Negative Guard**:
  - Modifying bytes of bundle contracts (e.g. `model_schema.json`) triggers immediate tamper refusal upon verification.

---

## 2. Warm Worker Container Pool (P9.7)

- **Implementation**:
  - Authored `WarmWorkerPool` in `crates/harness/src/pool.rs`, wired to `crates/harness/src/execution.rs` and `crates/harness/src/main.rs`.
  - Maintains pre-warmed, idle Docker containers running `mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`.
  - Fallback simulated execution mode supported for deterministic development without a live Docker daemon.
- **Sub-100ms Task Dispatch Latency**:
  - Instant task dispatch via container `docker exec`.
  - Dispatch latency reduced from ~2s to `<50ms` (measured ~30–45ms in integration tests).
- **Two-Tier Clean State Isolation (Negative Guard b)**:
  - Host workflow directory and container `/tmp` and `/outputs` wiped before and after task execution.
  - Verified: files created in slot during task 1 are completely absent when task 2 executes on the same recycled container slot.
- **Graceful Shutdown & Reaping**:
  - Containers cleanly stopped and removed upon pool drop or harness exit.
- **Pool Exhaustion & Recovery (Negative Guard c)**:
  - Verified concurrent task load exceeding capacity (4 concurrent tasks across 2 slots) gracefully queues, dispatches sequentially, and recovers without task loss or deadlocks.

---

## 3. Platform Matrix Memory Certification (Gate 5 Closure)

- **Cgroup v2 Memory Accounting & Certification**:
  - Integrated real-time Linux cgroup v2 sampling across container slots:
    - `memory.current`: live container memory footprint.
    - `memory.peak`: watermark peak memory receipt.
    - `memory.events`: differential `oom_kill` accounting.
- **Deterministic Cap Enforcement & DiagnosticBridge**:
  - Verified in `test_cgroup_v2_memory_certification_and_diagnostic_bridge_oom`:
    - Container provisioned with 64 MiB memory ceiling.
    - Injected task allocating 128 MiB terminated with exit code 137.
    - Cgroup OOM detected via monotonic differential events and exit code pattern.
    - Structured `DiagnosticReceipt` emitted via `DiagnosticBridge` diagnosing cgroup OOM.
    - Container slot recycled and subsequent conforming task succeeds without pool degradation.
- **Status**: Platform Matrix Memory Certification (Gate 5) is formally **CLOSED**.

---

## 4. Verification Logs & Command Receipts

| Artifact | Description | Result |
| --- | --- | --- |
| `installed_delivery_gate3.log` | Gate 3 integration tests (`cargo test --locked --test installed_delivery_gate3`) | **3 passed, 0 failed** |
| `warm_pool_p97.log` | Warm pool & Gate 5 tests (`cargo test --locked --test warm_pool_p97`) | **5 passed, 0 failed** |
| `rust_lib_tests.log` | Library unit tests (`cargo test --locked --lib`) | **27 passed, 0 failed, 2 ignored** |
| `full_regression_python.log` | Full Python test suite (`pytest python/tests/`) | **120 passed, 1 deselected, 0 failed** |
| `release_binary_capabilities.json` | Release binary capabilities JSON payload | `state_schema_version = 10`, `standalone_install = true`, `warm_pool = passed` |

---

## 5. Architectural Invariants Preserved

1. **Retained `PRAGMA user_version = 10` strictly without bumping the schema version.**
2. **Kept historical `docs/evidence/m6/` completely untouched.**
3. **Enforced strict label isolation: zero target leakage during candidate runs.**
4. **Maintained deterministic verification: all generated handoff bundles pass `verify_bundle`.**
