# MLRescurix Architecture Decision Register (ADR)

This document records the foundational architectural decisions, rationale, tradeoffs, active consolidation decisions, and superseded decisions across the development of MLRescurix.

---

## 1. Active Consolidation Decisions (D51–D60, 2026-10-23)

These decisions govern the consolidated architecture at baseline commit `441762f`.

| ID | Decision | Rationale | Tradeoffs & Revisit Trigger |
|---|---|---|---|
| **D51** | **Unified Feedback Controller (`AutonomousAgentLoop`)** | Consolidates fragmented specialist loops into a single multi-turn feedback controller. Actions are proposed via typed JSON schemas, executed under governance, measured against real cross-validation folds, and audited by an objective critic. | Eliminates bespoke per-task loops. Revisit if multi-agent parallel debate is empirically proven to outperform a single governed feedback controller under strict token limits. |
| **D52** | **Pure Scientific Task Adapters (`task_adapters.py`)** | Task adapters own topology, development data staging, panel coverage verification, and candidate defaults. They are strictly prohibited from implementing provider, budget, or worker loops. | Isolates task-specific ML semantics from system orchestration. Revisit when adding new problem modalities (e.g. graph ML, audio). |
| **D53** | **Unified Execution Service (`ExecutionService`)** | Centralizes admission, deadlines, process-group isolation, and artifact collection for trusted worker evaluations and isolated Docker scripts. Host evaluation is explicitly labeled as development execution; generated scripts execute inside Docker with no host fallback. | Prevents duplicate worker-spawning logic and credential leakage. Revisit when dedicated remote Kubernetes/Slurm execution backends are introduced. |
| **D54** | **Single Campaign Governance Boundary & Shared POSIX Lock** | Enforces the USD 0.30 campaign ceiling across all sessions, tasks, and retries via `CampaignBudgetAuthority` and `fcntl.flock` on `campaign_budget_ledger.json`. Non-fixture dispatches require `RustGatewayAdapter` with exact route matching and pre-dispatch reservations. | Guarantees zero budget overruns. Revisit if cross-machine distributed workers require distributed consensus instead of POSIX file locking. |
| **D55** | **Single Contract-Driven Benchmark Runner (`benchmark_runner.py`)** | Replaces duplicated K1–K5 execution paths with a single contract-driven runner. Follows a unified lifecycle: validate scientific contract -> validate pricing -> stage hash-bound development data -> run controller -> audit candidate -> refit -> package bundle. | Incomplete contracts (K1/K2 null thresholds, K4 missing fold provenance) block immediately before budget spend. Revisit when contracts are independently frozen. |
| **D56** | **End-to-End Cryptographic Lineage (`delivery_evidence.json`)** | Bundles must include a cryptographic audit receipt binding source SHA-256, data hash, fold hash, candidate specification, evaluation receipt, critic audit, model serialization digest, and applied human command IDs. | Eliminates unverified or mismatched candidate promotion. Revisit if compliance requirements mandate hardware TPM / HSM attestation. |
| **D57** | **Common Human Control Interface (`AgentInterface`)** | Provides a single status/steering/pause/resume/stop entry point for all tasks via durable inbox (`human_control.json`) and independent lock (`.human_control.lock`). Consumed at safe turn boundaries. | Allows concurrent human monitoring without blocking or corrupting active controller workspace locks. Revisit when building remote web/SSE frontends. |
| **D58** | **Retirement of Direct HTTP Provider Bypass Scripts** | Deprecates and fails closed legacy direct HTTP provider scripts (`run_kaggle_titanic_live.py`, `run_live_end_to_end_test.py`) before reading credentials. All live execution must route through the governed Rust gateway. | Closes unmonitored provider bypasses. Revisit trigger: none; direct bypasses are permanently retired. |
| **D59** | **Canonical Campaign Ledger Preservation** | Forbids minting new allowances by creating new campaign directories. Live CLI and benchmark paths must open the existing preserved ledger; missing files fail closed. | Prevents silent budget inflation. Revisit trigger: explicit authorized user reset of the entire campaign. |
| **D60** | **Strict Accounting Classification (Estimates vs Charges)** | Distinguishes calculated estimates (`cost_is_estimate=true`) and unverified local accounting from authentic provider invoices. Unused reservations on interrupted calls are marked `uncertain`. | Prevents premature release of funds or false billing certification. Revisit when authentic provider invoices are ingested. |

---

## 2. Superseded Decisions Register

The following decisions were adopted during earlier exploration or drafts and have been formally **superseded**:

| Old Decision / Practice | Superseded By | Reason for Reversal |
|---|---|---|
| **Direct HTTP Specialist Calls** (calling OpenRouter from Python without gateway fences) | **D54 / D58**: Rust gateway adapter with pre-dispatch intent binding and replay protection. | Unfenced HTTP calls risked double-billing and untracked failures during network interruptions. |
| **Ad-Hoc / Relaxed Quality Thresholds** (setting arbitrary thresholds in runner scripts) | **D55**: Contract-driven thresholds from `BENCHMARK_SUITE.json`; null thresholds fail closed. | Prevents artificial inflation of benchmark success rates; quality requirements must be independently frozen. |
| **Self-Approving Outbound Gateway Views** (controller approving its own provider messages) | **D54**: Separate operator inspection and approval (`approve-outbound-view` with actor and TTL). | Controller self-approval violated separation of duties and allowed unintended outbound data leakage. |
| **Silent Fold Reconstruction for K4** (generating replacement temporal folds when manifest mapping failed) | **D52 / D55**: Strict fold contract validation; missing index-space or gaps block execution. | Silently replacing frozen benchmark folds invalidates benchmark integrity and masks provenance defects. |
| **Treating Calculated Prices as Verified Charges** (asserting billed cost from token formulas) | **D60**: Labeling token-derived amounts as calculated estimates; verified charges remain null. | Token price formulas with rounding do not constitute verified provider invoices or binding settlement receipts. |
| **Inferring Production Certification from Offline Test Passes** (declaring M6 ready based on unit tests) | **D53 / D56**: Clear distinction between offline test validation and production clean-room certification. | Unit and synthetic tests verify logic and wiring; they do not certify real-world concurrency, hardware cgroups, or clean-room delivery. |

---

## 3. Foundational Architectural Decisions (D01–D50)

Historical decisions established during Phases P1–P12:

### Core Storage & Execution (D01–D10)
- **D01 (Safety Prerequisites First)**: Repair state, recovery, and release prerequisites before adding new runtime surfaces.
- **D02 (Dual-Tier Platform)**: Rust control plane for persistent scheduling, process management, and safety; Python workers for scientific estimation. Docker-only execution for untrusted candidate scripts.
- **D03 (Durable SQLite WAL)**: Shared SQLite database with Write-Ahead Logging (`WAL`) and `PRAGMA synchronous = FULL` for cross-process transaction safety.
- **D04 (Immutable Content-Addressable Storage)**: Store immutable artifacts in CAS (`cas/<hash[0..2]>/<hash>`) addressed by SHA-256.
- **D05 (Core Tool Families)**: Standardize on four core tool families (`workspace_fs`, `sandbox_exec`, `evaluate_submission`, `artifact_crud`) plus governed orchestration controls.
- **D06 (CLI Executable Contract)**: Standardize candidate model execution on CLI entrypoints (`fit` and `predict`) rather than rigid in-process Python class hierarchies.
- **D07 (Disposable Containers Before Warm Pools)**: Establish process and filesystem isolation using disposable containers first; optimize dispatch latency later.
- **D08 (Capacity-Driven Resource Dispatch)**: Schedule jobs against explicit CPU thread and memory caps rather than assuming unconstrained hardware.
- **D09 (Token Window Economics)**: Enforce ~16K economical token window default with a hard ceiling of 64K tokens, using context compaction.
- **D10 (Replay Protection)**: Replay only isolated, idempotently verifiable operations; unknown side effects fail with explicit uncertain status.

### Evaluation, Isolation & Governance (D11–D25)
- **D11 (Multi-Fidelity Search)**: Employ multi-fidelity successive halving (Tier 0 proxy screening followed by Tier 1 full cross-validation).
- **D12 (Deterministic Evaluator & Advisory Critic)**: Strict separation between the deterministic evaluator (which computes ground-truth metrics) and the LLM critic (which provides advisory reviews and policy vetoes).
- **D13 (Modality-Scoped Expansion)**: Restrict ML expansion to explicit task adapters with verified scientific contracts.
- **D14 (Embedded Runtime Assets)**: Embed Python worker assets inside the compiled Rust binary to allow execution outside repository checkouts.
- **D15 (Read-Only Workspace Execution)**: Mount candidate workspaces as read-only (`/workspace:ro`) with isolated tmpfs scratch spaces (`/scratch:rw`).
- **D16 (Fold-Level Label Isolation)**: Split data before preprocessing. Exclude validation labels from candidate feature transformers and inference inputs.
- **D17 (Legacy Pretrained Model Provenance)**: Treat unverified external models as untrusted legacy baselines requiring clean refits from raw data.
- **D18 (Two-Tier Summary Bounds)**: Enforce index digests $\le 512$ bytes for listings and summaries $\le 4096$ bytes for detail views to protect context windows.
- **D19 (Versioned Contract Checking)**: Enforce JSON Schema Draft 2020-12 validation on all tool inputs, completions, and artifacts.
- **D20 (Pre-Dispatch Intent Binding)**: Record tool intent in SQLite before execution; crashes before admission emit uncertain outcomes rather than duplicating side effects.
- **D21 (Authoritative Crash Reconciliation)**: Reconcile non-file operations (job submission, delegation, publication) from durable records on supervisor restart.
- **D22 (Unified Reservation Accounting & Headroom)**: Require pre-admission byte reservations for all disk writes under a 128 MiB policy ceiling with emergency recovery headroom.
- **D23 (Lease-Aware Archival & Fences)**: Compact historical action bodies while retaining SHA-256 tombstones and idempotency retry fences.
- **D24 (Per-Job Linux Cgroup v2 Limits)**: Probe kernel cgroup v2 support; refuse memory certification deterministically if cgroups are unavailable.
- **D25 (Docker Survivor Recovery)**: Track exact container IDs and owner nonces to safely discover, adopt, or reap orphaned worker containers after process crashes.

### Artifacts, Resources & Candidate Synthesis (D26–D50)
- **D26 (Unified Artifact Registry & CAS)**: Unify artifact management behind relational metadata tables (`artifacts`, `artifact_versions`, `artifact_lineage`) and CAS storage.
- **D27 (Durable Event Journal)**: Record mutations in `durable_events` with a monotonic cursor for real-time streaming and reconnect replay.
- **D28 (Pooled Query-Only Read Connections)**: Maintain four dedicated read-only SQLite connections (`PRAGMA query_only = ON`) to prevent reader starvation during SSE streaming.
- **D29 (Crate-Internal Evaluator Authority)**: Restrict scorecard publication authority to crate-internal evaluator modules; untrusted callers cannot forge trusted scorecards.
- **D30 (Revisioned Artifact Commands)**: Idempotently fence artifact mutations using `(project, command_id)` pairs.
- **D31 (CAS Staging Sweeps in GC)**: Clean orphaned temporary files from `cas/.staging/` during garbage collection without corrupting active blobs.
- **D32 (Effective Resource Capability Discovery)**: Clamp container resource limits to `min(host, vm)` to prevent host RAM overestimation on macOS Docker Desktop.
- **D33 (Resource Lease Lifecycle & Aging)**: Manage worker resource allocations (`reserved -> active -> recovering -> released`) with weighted fair queueing and aging.
- **D34 (Generic Tool Adapters & Schema P4)**: Implement versioned generic coding tools (`workspace_fs`, `sandbox_exec`, `dataset_query`, `artifact_crud`).
- **D35 (Task Assignment & Completion Schemas)**: Enforce structured task assignments and completion receipts for specialist roles.
- **D36 (Native Rust HTTP/SSE Observability Dashboard)**: Provide loopback-authenticated dashboard server for DAG visualization, Pareto frontiers, and telemetry.
- **D37 (Self-Contained Candidate Bundle `v2`)**: Package trained models into standalone distributions with frozen weights, source code, schemas, and serving recipes.
- **D38 (Analytical Feasibility & Micro-Pilots)**: Predict resource feasibility using complexity models and capped $N \le 100$ micro-pilots before heavy dispatch.
- **D39 (Out-of-Core Arrow & Time-Series Adapters)**: Support memory-mapped Apache Arrow datasets and causal time-series backtesting with lookahead leakage guards.
- **D40 (Unsupervised Anomaly & Clustering Adapter)**: Support Isolation Forests, Local Outlier Factor, and clustering with empirical score calibration receipts.
- **D41 (Frozen Text/Vision Embeddings Adapter)**: Support pretrained frozen transformers with content-addressed embedding caching and offline download refusal.
- **D42 (Agent-First Autonomous Synthesis Realignment)**: Empower the agent as primary ML engineer to generate arbitrary Python code within deterministic guardrails.
- **D43 (Autonomous Candidate Synthesis Engine)**: Generate modular candidate trees (`entrypoint.py`, `model.py`, `features.py`, `manifest.json`) across model families.
- **D44 (Structured Diagnostic Feedback Bridge)**: Intercept 8 failure classes (`cgroup_oom`, `non_finite_loss`, `shape_mismatch`, `execution_timeout`, etc.) with 32KB diagnostic receipts.
- **D45 (Multi-Fidelity Iteration & Pareto Frontier)**: Formulate falsifiable hypotheses, screen via Tier 0 proxy sampling, and optimize Pareto tradeoffs (metric, latency, memory).
- **D46 (Autonomous End-to-End Rescue Engine)**: Closed-loop pipeline from raw dataset profiling to self-healing repair and verified bundle compilation.
- **D47 (Pareto Ensembles & Feature Primitives)**: Stack non-dominated models using SLSQP bounded blending weights and leak-free out-of-fold feature encoders.
- **D48 (Core Scope Boundary)**: Define the formal boundary of MLRescurix at the self-contained bundle; downstream cloud serving is outside harness scope.
- **D49 (Clean Installed Delivery & Warm Pool)**: Verify standalone bundle compilation outside git checkouts; maintain sub-50ms pre-warmed container worker pools.
- **D50 (Prospective Matched Benchmarks)**: Multi-seed benchmark trials on OpenML datasets under matched compute budgets with single-use sealed holdouts.
