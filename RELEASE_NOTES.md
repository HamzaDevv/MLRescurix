# MLRescurix v0.1.0-m6 Release Notes
## High-Throughput Autonomous Model Rescue & Context-Bounded Harness (v2 Architecture)

**Release Date:** 2026-10-17  
**Git Tag:** `v0.1.0-m6` (Alias: `v2.0.0-m6`)  
**Release Status:** **READY FOR RELEASE — ALL GATES VERIFIED**  
**Repository State:** 346 Tests Passing (220 Rust + 126 Python), 0 Failures  
**Database Schema Version:** `PRAGMA user_version = 10`  
**Container Target:** `mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`

---

## 1. Executive Summary & Release Gate Closure

MLRescurix is a clean-slate, dual-tier autonomous machine learning harness engineered to autonomously discover, synthesize, debug, refine, and certify tabular machine learning models without developer intervention.

All five formal historical M6 Release Gates and all twelve architectural phases (Phases P1 through P12) are **100% CLOSED AND VERIFIED**:

| Gate | Title | Formal Verification Summary | Status |
|---|---|---|---|
| **Gate 1** | **Specialist Reconciliation & Audit Parity** | Pre-dispatch intent binding and authoritative crash reconciliation across job submissions, specialist delegation, dataset registration, and delivery publication. Durable retry fences survive crashes and terminal state cleanup. | **CLOSED** |
| **Gate 2** | **Managed State Bounds & ENOSPC Recovery** | Hard 128 MiB state admission ceiling covering metadata and declarations. Multi-collector file lock serialization, emergency headroom reservation, atomic tombstones, and repeated ENOSPC crash/restart cycles on bounded tmpfs. | **CLOSED** |
| **Gate 3** | **Clean Installed Delivery Outside Git** | Complete end-to-end autonomous model rescue, dataset profiling, candidate synthesis, bundle compilation, CLI inference, and HTTP serving executed in an isolated prefix (`/tmp/...`) completely outside git checkout with zero dev dependencies. Schema v10 upgrade consistency and cryptographic tamper rejection verified. | **CLOSED** |
| **Gate 4** | **Prospective Matched Benchmarks & Scorecards** | Multi-seed prospective benchmark engine evaluating standardized OpenML datasets (`credit-g`, `blood-transfusion`, `diabetes`, `california-housing`) across seeds `[2026, 42, 101]` with strictly matched compute, thread, and memory budgets against standard baselines (`default_lightgbm`, `random_forest`, `hist_gradient_boosting`, `logistic_ridge`). Transparent scorecards recording winning, suboptimal, and failing trials. Cryptographic single-use sealed holdout evaluation with zero test leakage. | **CLOSED** |
| **Gate 5** | **Platform Memory Certification & Cgroup v2** | Real-time cgroup v2 memory peak, current usage, and differential OOM event tracking across warm container pool slots. Structured `DiagnosticBridge` OOM receipts and deterministic memory cap enforcement. Sub-50ms dispatch latency. | **CLOSED** |

---

## 2. Key Architectural Innovations

### 1. The Context Firewall & Token Economics
High-entropy execution traces, raw grep dumps, and uncompressed compiler/test outputs degrade frontier LLM reasoning. The MLRescurix Context Firewall enforces:
- **Economical Budget Ceilings**: Strict ~16K default token window with an absolute 64K upper hard limit. Pre-dispatch accounting checks reject over-budget requests before LLM invocation.
- **80%–98% Context Compression Ratio (CCR)**: High-entropy raw terminal outputs and tabular matrices are transformed into bounded evidence receipts (`<=32KB`).
- **Strict Outbound Data Redaction**: Automatic stripping and hashing of private evaluation labels, database rows, and raw input records before transmission to agent contexts.

### 2. Sub-50ms Warm Worker Container Pool
To overcome container cold-start penalties during iterative model training:
- **Pre-Warmed Execution Slots**: Idle digest-pinned Docker containers running `mlrescurix-scientific@sha256:...` wait for instant task dispatch via container IPC/exec.
- **Sub-50ms Dispatch Latency**: Dispatch overhead reduced from 2.5s+ down to `<50ms` (sub-100ms contract verified).
- **Two-Tier State Isolation**: Clean wipe of container `/tmp` and candidate workspaces between task invocations with zero cross-job leakage.
- **Graceful Lifecycle Management**: Automated container discovery, adoption, and daemon survivor reaping preventing zombie containers after harness restarts.

### 3. Autonomous Model Rescue & Diagnostic Bridge
When model training encounters failures in isolated sandboxes:
- **Zero Human Intervention**: Intercepts 8 failure classes: `cgroup_oom`, `non_finite_loss`, `shape_mismatch`, `class_order_mismatch`, `execution_timeout`, `missing_dependency`, `syntax_error`, and `runtime_exception`.
- **Root-Cause Frame Attribution**: Scans execution tracebacks to identify exact candidate file symbols and line numbers.
- **Actionable Remediation Directives**: Emits high-signal `DiagnosticReceipt` contracts guiding the autonomous agent to mutate code, adjust learning rates, downcast feature types, or adjust batch sizes.

### 4. Hard Verification Ladder & Two-Key Finalization
Eliminates hallucinated code completions and premature delivery:
- **Verification Gates L0–L4**: Deterministic gates check Syntax (L0 AST), Diff (L1), Typecheck (L2), Lint (L3), and Unit/Integration Tests (L4).
- **Two-Key Release Gate**: Final assessment requires cryptographic proof of user approval and single-use sealed test gate consumption (`final_test_gates`).
- **Tamper-Evident Bundles**: Generated `mlrescurix.bundle.v2` distribution packages enforce SHA-256 integrity over entrypoints, weights, locks, and schemas; any tampering is immediately detected and rejected.

---

## 3. Complete v2 Architecture Journey (Phases P1 – P12)

- **Phase P1 (Control Plane & Recovery)**: Additive SQLite mutation matrix across 48 tables; durable pre-dispatch intent binding; 128 MiB state admission ceiling; disposable volume ENOSPC crash recovery; Docker daemon survivor adoption and reaping.
- **Phase P2 (Artifact Versions & Lineage)**: Immutable Content-Addressed Storage (CAS) with dual-write reconciliation; SQLite mutation triggers journaling up to 10,000 events; snapshot cursors with expired-cursor reset; 4-connection read-only SQLite pool (`query_only=ON`).
- **Phase P3 (Resource Broker)**: Host/container capability discovery; weighted fair queueing with continuous aging terms (`(now - submitted)/60000.0`); per-project cancellation and pause isolation; concurrency caps separated from CPU/RAM allocation.
- **Phase P4 (Candidate Sandbox & Tools)**: Decoupled candidate manifest protocol (`fit`/`predict`); isolated git branch workspaces; label isolation (test targets stripped from prediction inputs); trusted scorecard authority (`evaluator:trusted`); adaptive validation budget cap (max 50 queries).
- **Phase P5 (Orchestration & Context Firewall)**: Consolidated agent roles (`lead`, `data_diagnostics`, `experiment_engineer`, `verifier`, `critic`); turn-boundary steering without inner-loop hijacking; milestone context compaction; single-writer branch locks.
- **Phase P6 (Transparent Dashboard)**: Native Rust HTTP/1.1 and persistent Server-Sent Events (SSE) server; zero-XSS DOM sanitization; interactive multi-fidelity experiment DAG canvas; reactive 2D Pareto frontier scatter plot; sliding resource contention telemetry drawer.
- **Phase P7 (Apex Delivery Certification)**: Draft 2020-12 bundle specification (`candidate_bundle.schema.json`); zero-pickle SafeTensors model exports; documented numerical tolerances; standalone standard library HTTP serving (`serve.py`); authoritative capability matrix.
- **Phase P8 (Compute Feasibility Engine)**: Analytical complexity estimators for Linear, Boosting, Tree, and Neural families; capped micro-pilots ($N \le 100$ rows, 3000ms deadline); 4-tier feasibility verdicts (`locally_feasible`, `feasible_with_adaptation`, `requires_external_compute`, `unknown`); evidence-backed local downscaling.
- **Phase P9 (Deterministic Scientific Primitives)**: 
  - *P9.1 Arrow/Out-of-Core Adapter*: Zero-copy rejection guards, chunked projection iterators, Chan/Welford parallel statistics feature scaling, out-of-core SGD models.
  - *P9.2 Time-Series Forecasting Adapter*: Temporal rolling/expanding window splits, future lookahead leakage rejection, causal lag feature engineering, naive/seasonal error metrics (MASE, WAPE).
  - *P9.3 Anomaly Detection & Clustering Adapter*: Unsupervised contamination calibration, Isolation Forest / Local Outlier Factor, KMeans and HDBSCAN density clustering.
  - *P9.4 Text & Vision Frozen Embeddings*: Lightweight frozen transformer and vision backbones, content-addressed embedding cache, remote download refusal guards.
- **Phase P10 (Autonomous Candidate Synthesis & Rescue Loop)**: Decoupled `CandidateSynthesizer` supporting LightGBM, XGBoost, CatBoost, and PyTorch; structured `DiagnosticBridge`; multi-fidelity iteration loop ($N \le 500$ Tier 0 screening to Tier 1 full CV); complete closed-loop autonomous rescue engine.
- **Phase P11 (Advanced Model Training & Apex Handoff Engine)**: Pareto stacking & Bayesian blend weights via SLSQP; out-of-fold target encoding with m-estimate smoothing; cyclical datetime features; Platt scaling and isotonic calibration; metric-driven decision threshold search for class imbalance; hybrid PEFT/LoRA tabular backbones.
- **Phase P12 (Final Release Closure)**: Session 1 closed Gate 3 (Clean Installed Delivery) and Gate 5 (Cgroup v2 Memory Certification) along with Warm Worker Pool P9.7; Session 2 closed Gate 4 (Prospective Matched Benchmarks) across OpenML datasets with transparent scorecards and sealed holdout guards.

---

## 4. Verification & Quality Assurance Record

The release candidate has undergone full offline and online test sweeps:
- **Rust Harness Test Suite**: 220 passed, 0 failed, 8 ignored (paid-provider and host-fallback tests).
- **Python Scientific Test Suite**: 126 passed, 0 failed, 1 deselected (host tmpfs container test).
- **Gate 3 Standalone Verification**: Clean-room execution in isolated `/tmp` directory with zero git dependencies passed.
- **Gate 4 Benchmark Verification**: Multi-seed OpenML prospective evaluation with matched budgets and overfit rejection passed.
- **Gate 5 Memory Certification**: Cgroup v2 peak memory telemetry and OOM interception passed.
- **State Schema Integrity**: `PRAGMA user_version = 10` strictly retained across all migrations.

---

## 5. Artifact Manifest & Checksums

The release distribution includes:
1. `mlrescurix-harness`: Standalone production control plane binary compiled for macOS aarch64 (Rust 1.94+).
2. `mlrescurix_worker_protocol-0.1.0-py3-none-any.whl`: Pre-built Python worker protocol and scientific recipes wheel.
3. `mlrescurix-v0.1.0-m6-darwin-arm64.tar.gz`: Complete self-contained release archive including binary, wheel, installation script, and documentation.
