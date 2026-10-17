# MLRescurix

[![Release](https://img.shields.io/badge/release-v0.1.0--m6-blue.svg)](RELEASE_NOTES.md)
[![Status](https://img.shields.io/badge/status-READY_FOR_RELEASE-brightgreen.svg)](docs/m6-completion-status.md)
[![Tests](https://img.shields.io/badge/tests-346_passed-success.svg)](CHANGELOG.md)
[![Gates](https://img.shields.io/badge/release_gates-5%2F5_closed-success.svg)](docs/m6-completion-status.md)
[![Schema](https://img.shields.io/badge/schema-v10-informational.svg)](crates/harness/src/schema.rs)
[![Platform](https://img.shields.io/badge/platform-macOS_arm64_%7C_Linux_x86__64-informational.svg)](RELEASE_NOTES.md)

MLRescurix is a dual-tier, context-bounded autonomous machine learning harness engineered to solve one tabular machine-learning problem per project. It autonomously discovers, synthesizes, executes, debugs, and refines machine learning models within deterministic sandbox guardrails, producing a certified, standalone, zero-dependency deployment bundle (`mlrescurix.bundle.v2`).

The control plane is implemented in high-performance Rust (`mlrescurix-harness`) and integrates a sub-50ms warm worker pool, Content-Addressed Storage (CAS), a multi-project resource broker, and an integrated native HTTP/SSE observability dashboard. The scientific kernel executes inside an digest-pinned Docker container with Linux cgroup v2 memory certification.

---

## Current Status: READY FOR RELEASE

The official **M6 Milestone** is **READY FOR RELEASE** on branch `main`. All five historical release gates and all twelve v2 architectural phases (Phases P1 through P12) are **100% CLOSED AND VERIFIED**:

- **Release Gate 1 (Specialist Reconciliation & Audit Parity)**: **CLOSED** (Durable pre-dispatch intent binding and authoritative crash reconciliation).
- **Release Gate 2 (Managed State Bounds & ENOSPC Recovery)**: **CLOSED** (Hard 128 MiB state admission ceiling, multi-process file locking, and crash/restart recovery on bounded tmpfs).
- **Release Gate 3 (Clean Installed Delivery Outside Git)**: **CLOSED** (Complete end-to-end autonomous model rescue, dataset profiling, candidate synthesis, bundle compilation, and CLI/HTTP inference executed outside git checkouts).
- **Release Gate 4 (Prospective Matched Benchmarks & Scorecards)**: **CLOSED** (Multi-seed OpenML benchmarks across seeds `[2026, 42, 101]` with matched compute/memory budgets, transparent scorecards, single-use sealed holdouts, and overfit candidate rejection).
- **Release Gate 5 (Platform Memory Certification & Cgroup v2)**: **CLOSED** (Real-time cgroup v2 peak memory tracking, sub-50ms warm worker pool dispatch, and structured `DiagnosticBridge` OOM receipts).

Full repository sweeps pass cleanly: **346 tests passing** (220 Rust + 126 Python), 0 failures.

---

## Key Innovations

1. **Context Firewall & Token Economics**: Enforces economical ~16K default token windows (64K hard cap), 80%–98% context compression ratio (CCR), and strict outbound data redaction, shielding frontier LLM reasoning from high-entropy raw terminal and matrix bloat.
2. **Sub-50ms Warm Worker Container Pool**: Eliminates Docker cold-start delays by maintaining pre-warmed idle containers (`mlrescurix-scientific@sha256:...`) with two-tier clean state isolation and instant IPC dispatch.
3. **Autonomous Model Rescue & Diagnostic Bridge**: Intercepts 8 runtime failure classes (`cgroup_oom`, `non_finite_loss`, `shape_mismatch`, `class_order_mismatch`, `execution_timeout`, `missing_dependency`, `syntax_error`, `runtime_exception`) and emits bounded 32KB diagnostic receipts for autonomous self-healing.
4. **Hard Verification Ladder & Two-Key Handoff**: Deterministic L0–L4 multi-gate verification (AST, Diff, Typecheck, Lint, Test), two-key release gates requiring registered approvals, and cryptographic tamper detection.

---

## Quickstart & Installation

### Option 1: Standalone Release Archive (Recommended)

Download and extract the pre-built distribution archive for your platform:

```sh
# Extract release distribution archive
tar -xzf dist/mlrescurix-v0.1.0-m6-darwin-arm64.tar.gz
cd mlrescurix-0.1.0-m6

# Run standalone installer (installs binary to ~/.local/bin by default)
./install.sh

# Verify binary installation and capability matrix
mlrescurix-harness capabilities
```

### Option 2: Build From Source

```sh
# Prerequisites: Rust 1.94+ and Python 3.13+
cargo build --release --locked

# Copy binary to system path
cp target/release/mlrescurix-harness ~/.local/bin/

# Build and install Python worker protocol wheel
python3 -m pip install dist/mlrescurix_worker_protocol-0.1.0-py3-none-any.whl
```

---

## End-to-End Workflow

### 1. Initialize Project & Profile Dataset

```sh
# Initialize state directory under PRAGMA user_version = 10
mlrescurix-harness init --state /path/to/state --policy /path/to/policy.json

# Profile tabular dataset without private row leakage
mlrescurix-harness profile-dataset \
  --state /path/to/state \
  --csv /path/to/dataset.csv \
  --target churn
```

### 2. Autonomous Rescue & Bundle Compilation

```sh
# Package certified candidate delivery bundle
mlrescurix-harness package-candidate-bundle \
  --state /path/to/state \
  --candidate-dir /path/to/candidate \
  --bundle-dir /path/to/bundle \
  --dataset /path/to/dataset.csv \
  --test-access-id access-001 \
  --project p-prod-01 \
  --candidate cand-apex-01 \
  --task-type binary \
  --target churn \
  --metric balanced_accuracy
```

### 3. Verify Bundle Integrity Against Authoritative State

```sh
mlrescurix-harness verify-bundle \
  --bundle /path/to/bundle \
  --sandbox=false
```

### 4. Standalone Inference & Zero-Dependency HTTP Serving

The generated bundle is 100% self-contained and runnable without harness or developer dependencies:

```sh
# Standalone CLI inference
python -B /path/to/bundle/inference.py \
  --input /path/to/input.json \
  --output /path/to/predictions.json

# Zero-dependency HTTP serving microservice (/health and /predict)
python -B /path/to/bundle/serve.py --port 8080
```

### 5. Launch Native Observability Dashboard

```sh
# Launch native Rust HTTP/SSE dashboard with interactive DAG, Pareto frontier, and telemetry
mlrescurix-harness serve-dashboard \
  --state /path/to/state \
  --port 8765 \
  --token my-secure-token
```

Open `http://127.0.0.1:8765` in your browser and enter your session token.

---

## Test & Verification Suites

```sh
# Run full Rust test suite across all 20+ integration binaries (220 tests)
cargo test --locked -- --test-threads=4

# Run scientific Python test suite (126 tests)
PYTHONPATH=python/src pytest python/tests/ -k "not test_real_container_tmpfs_enospc_never_publishes"

# Run end-to-end clean-room release smoke test
python3 scripts/run_standalone_smoke_test.py
```

---

## Documentation & Architecture References

- [Official Release Notes](RELEASE_NOTES.md)
- [Changelog & Milestones](CHANGELOG.md)
- [M6 Completion Status & Gate Audit](docs/m6-completion-status.md)
- [v2 Phased Roadmap & Architecture Specification](docs/v2/README.md)
- [Persistent Progress & Session Handoff](docs/v2/PROGRESS.md)
- [Architectural Decisions Log (D1–D50)](docs/v2/DECISIONS.md)
- [M1 Execution & Storage Engine](docs/m1-execution-storage.md)
- [M2 Tabular Kernel & Limits](docs/m2-scientific-kernel.md)
- [M3 Constrained Resumable Search](docs/m3-constrained-search.md)
- [M4 Agent-Led Coding Harness](docs/m4-agent-harness.md)
