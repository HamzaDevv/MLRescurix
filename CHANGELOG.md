# Changelog

All notable changes to the MLRescurix project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0-consolidation] - 2026-10-23

### Consolidated
- **Governed Agent Architecture**: Consolidated runtime into one feedback controller (`AutonomousAgentLoop`), pure scientific task adapters (`ClassificationAdapter`, `RegressionAdapter`, `PanelAdapter`), execution service (`ExecutionService`), governance service (`GovernanceService` with `CampaignBudgetAuthority`), contract-driven benchmark runner (`benchmark_runner.py`), and identity-bound delivery (`evidence.py`).
- **Common Human Control Interface**: Implemented `AgentInterface` in `agent_interface.py` providing turn-boundary `status`, `steer`, `pause`, `resume`, and `stop` operations under independent `.human_control.lock`.
- **Documentation Consolidation**: Consolidated and reorganized repository documentation into a concise, coherent set (`README.md`, `docs/ARCHITECTURE.md`, `docs/HUMAN_CONTROL_INTERFACE.md`, `docs/DECISIONS.md`, `docs/DEVELOPMENT_AND_TESTING.md`, `docs/STATUS_AND_ROADMAP.md`, `docs/CONTINUATION_GUIDE.md`, `docs/MIGRATION_MAP.md`).
- **Preserved Accounting & Contracts**: Retained canonical campaign budget ledger (USD 0.30 ceiling, USD 0.050724020 committed, USD 0.249275980 remaining, SHA-256 `827abd98...`), `BENCHMARK_SUITE.json`, `K5_PANEL_PROTOCOL.json`, and all historical evidence.

---

## [0.1.0-m6] - 2026-10-17

### Added
- **Release Gate 1 (Specialist Reconciliation & Audit Parity)**:
  - Durable pre-dispatch intent binding and authoritative crash reconciliation across job submissions, specialist delegation, dataset registration, and delivery publication (`crates/harness/src/agents/recovery.rs`).
  - Durable retry fences surviving cleanup and terminal states.
- **Release Gate 2 (Managed State Bounds & ENOSPC Recovery)**:
  - Hard 128 MiB state admission ceiling with immediate SQLite transaction reservation accounting covering declared output and log allocations.
  - Multi-process artifact collection serialization via durable lock file.
  - Recovery headroom and clean crash recovery under physical and tmpfs ENOSPC conditions.
- **Release Gate 3 (Clean Installed Delivery Outside Git)**:
  - Standalone delivery verification outside source checkouts (`crates/harness/tests/installed_delivery_gate3.rs`).
  - Self-contained embedded Python runtime assets (`crates/harness/src/assets.rs`) materializing into `<state>/runtime-assets/<hash>/python`.
  - Cryptographic tamper detection rejecting altered model schemas or untracked candidate files in `verify_bundle`.
- **Release Gate 4 (Prospective Matched Benchmarks & Transparent Scorecards)**:
  - Multi-seed prospective benchmark engine (`python/src/mlrescurix_worker/scientific/benchmarks.py`) evaluating OpenML datasets (`credit-g`, `blood-transfusion`, `diabetes`, `california-housing`) across seeds `[2026, 42, 101]`.
  - Matched compute, thread, and memory budgets against standard baselines (`default_lightgbm`, `random_forest`, `hist_gradient_boosting`, `logistic_ridge`).
  - Transparent trial scorecards recording winning, suboptimal, and failing trials.
  - Cryptographic single-use `SealedHoldout` with `SealedTestLeakageError` preventing evaluation data leakage.
  - Deterministic overfit candidate rejection (`overfit_candidate`).
- **Release Gate 5 (Platform Memory Certification & Cgroup v2)**:
  - Automated sampling of Linux cgroup v2 peak memory (`memory.peak`) and differential OOM events (`cgroup.events`).
  - Structured `DiagnosticBridge` OOM receipts.
  - Hard memory cap enforcement and certification verification.
- **Warm Worker Container Pool (P9.7)**:
  - Pre-warmed idle Docker container pool maintaining `mlrescurix-scientific@sha256:...` slots (`crates/harness/src/pool.rs`).
  - Sub-50ms task dispatch via container IPC/exec.
  - Two-tier state isolation wiping container `/tmp` and candidate workspaces between tasks.
  - Graceful pool shutdown and orphan container adoption/reaping.
- **Autonomous Candidate Synthesis & Iterative Refinement Loop (Phase P10)**:
  - Decoupled `CandidateSynthesizer` generating modular candidate code (`entrypoint.py`, `model.py`, `features.py`, `manifest.json`) supporting arbitrary ML families (LightGBM, XGBoost, CatBoost, PyTorch, Trees, Linear).
  - Structured `DiagnosticBridge` intercepting 8 failure classes (`cgroup_oom`, `non_finite_loss`, `shape_mismatch`, `class_order_mismatch`, `execution_timeout`, `missing_dependency`, `syntax_error`, `runtime_exception`) and emitting actionable remediation recommendations.
  - Multi-fidelity iteration loop with Tier 0 subsampled fast proxy screening ($N \le 500$, 15s deadline) and non-dominated Pareto frontier tracking.
  - Autonomous rescue engine (`AutonomousRescueEngine`) orchestrating dataset profiling, candidate generation, error repair, and certified bundle compilation without manual developer intervention.
- **Advanced Model Training & Apex Handoff Engine (Phase P11)**:
  - Multi-model Pareto stacking and constrained Bayesian blend weight optimization via SLSQP (`pareto_ensemble.py`).
  - Out-of-fold target encoding with m-estimate smoothing, frequency encoding, numerical feature crosses with division stability guards ($\epsilon = 10^{-5}$), and cyclical datetime encoding (`features.py`).
  - Platt scaling and isotonic calibration with Brier score evaluation and metric-driven decision threshold search for class imbalance (`calibration.py`).
  - Hybrid PEFT/LoRA multimodal tabular architecture with $>95\%$ frozen backbone parameters.
- **Deterministic Scientific Primitives (Phase P9)**:
  - Arrow out-of-core data adapter with chunked projection iterators and parallel Welford feature scaling (`arrow_adapter.py`).
  - Time-series forecasting adapter with entity-aware temporal splits, causal lag generation, and multi-step forecasting (`time_series.py`).
  - Unsupervised anomaly detection and clustering adapter with empirical score calibration and HDBSCAN density clustering (`anomaly_clustering.py`).
  - Text and vision frozen-embedding adapters with deterministic content-addressed caching (`frozen_embeddings.py`).
- **Compute Feasibility Engine (Phase P8)**:
  - Analytical complexity bounds and capped micro-pilots ($N \le 100$ rows) emitting 4-tier feasibility verdicts (`feasibility.py`).
  - Local evidence-backed adaptations (quantization, sparse matrices, progressive fidelity).
- **Transparent Dashboard & human Controls (Phase P6)**:
  - Native Rust HTTP/1.1 and persistent SSE server with zero-XSS DOM sanitization (`dashboard.html`, `server.rs`).
  - Interactive multi-fidelity experiment DAG canvas and reactive 2D Pareto frontier scatter plot.
  - Sliding resource contention and capacity telemetry drawer.
- **Resource Broker Across Projects (Phase P3)**:
  - Weighted fair queueing with continuous aging and stable deterministic tie-breaking (`broker.rs`).
  - Multi-project isolation where pause or cancellation of one project leaves sibling projects unaffected.
- **Immutable Artifact CAS & Lineage (Phase P2)**:
  - Content-Addressed Storage with dual-write reconciliation and project-scoped lineage.
  - SQLite mutation triggers atomically journaling up to 10,000 events per project.
  - Read-only 4-connection SQLite pool with zero waiting on exhaustion.

### Changed
- `docs/m6-completion-status.md` updated to **READY for release** with all 5 release gates formally closed.
- `crates/harness/src/main.rs` capabilities updated to report all 5 historical release gates as `closed_and_verified` and all v2 release gates as `passed`.
- `README.md` updated with official release badges, quickstart instructions, and distribution workflow.
- Embedded assets updated to include all Phase P9–P12 scientific modules and schemas.

### Fixed
- Hardened timing margins in `recovery_cycles.rs` (`leased_artifacts_survive_concurrent_gc_and_cancellation`) to prevent premature job timeouts during concurrent artifact collection under heavy multi-threaded test runner execution.
- Prevented daemon survivor container leaks across supervisor death.
- Eliminated all potential race conditions in concurrent multi-collector artifact collection through durable file-locking.
- Resolved all potential label leakage vectors through single-use sealed holdout gates and strict feature column dropping.

### Invariants Maintained
- `PRAGMA user_version = 10` preserved strictly without schema bump.
- Historical evidence under `docs/evidence/m6/` kept completely untouched.
- 100% test pass rate: 220 Rust + 126 Python tests passing (346 total tests).
