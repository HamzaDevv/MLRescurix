# MLRescurix

Governed dual-tier autonomous machine learning harness and context firewall for tabular and panel forecasting problems.

---

## 1. System Overview & Purpose

MLRescurix is engineered to solve machine learning problems within strict computational, environmental, and financial constraints. It integrates an authoritative Rust control plane (`mlrescurix-harness`) with a specialized Python scientific kernel (`mlrescurix_worker.scientific`).

### Architecture Baseline (Commit `441762f`)
The repository is consolidated into six core architectural components:
1. **Feedback Controller (`AutonomousAgentLoop`)**: Multi-turn adaptive controller executing schema-validated hypothesis actions, fold-local evaluations, and critic audits.
2. **Scientific Task Adapters (`task_adapters.py`)**: Dedicated adapters for Classification, Regression, and Panel Forecasting (`K5_PANEL_PROTOCOL.json`). Purely scientific; strictly zero provider or worker loops.
3. **Execution Service (`ExecutionService`)**: Lifecycle management, evaluation count caps ($\le 12$), process-group isolation, and Docker-only script execution with no host fallback.
4. **Governance Service (`GovernanceService`)**: Transactional POSIX-locked spend authority governing the single **USD 0.30 campaign ceiling**, pre-dispatch reservation math, 3-phase dispatch fencing, and Rust replay protection.
5. **Contract-Driven Benchmark Runner (`benchmark_runner.py`)**: Unified lifecycle for K1–K5 tasks, rejecting incomplete contracts before budget spend.
6. **Identity-Bound Evidence & Delivery (`evidence.py`)**: Standalone `mlrescurix.bundle.v2` distribution format with 13 contracts and cryptographic lineage in `delivery_evidence.json`.

---

## 2. Authoritative Budget Ledger

Financial governance is strictly enforced via a single POSIX-locked JSON ledger:
- **Ledger Location**: [`docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json)
- **SHA-256 Digest**: `827abd984dd960f91cd39725d7dcbdb78d47c114322a0358cbbb08a5b30d0b52`
- **Total Campaign Ceiling**: **USD 0.30** (Shared across all tasks, sessions, retries, and uncertain outcomes; **not** a per-problem allowance).
- **Cumulative Committed Spend**: **USD 0.050724020** across 29 historical runs (56 local turn records).
- **Recorded Remaining Budget**: **USD 0.249275980** (83.09% remaining).
- **Consolidation Baseline Spend**: **USD 0.00** (Zero paid provider calls, zero additional spend, zero holdout scoring).

> [!IMPORTANT]
> Historical ledger entries are **calculated estimates** based on registered token rates and rounding; they are **not verified provider invoices**.

---

## 3. Supported Workflows

### Workflow 1: Governed Development Agent
Execute an offline or governed development session using `scientific.agent_cli`:

```sh
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_cli \
  --development-csv /path/to/dev_data.csv \
  --folds /path/to/folds.json \
  --workspace /path/to/workspace \
  --project my_project \
  --session sess_001 \
  --target target_col \
  --task binary_classification \
  --metric balanced_accuracy \
  --direction maximize \
  --max-turns 3 \
  --fixture-actions /path/to/fixture_actions.json
```

### Workflow 2: Robustness Campaign Benchmarking
Execute benchmark tasks (K1–K5) through the contract-driven runner:

```sh
# Run benchmark in live_dev mode (requires verified contracts and pricing)
python3 scripts/run_robustness_campaign.py --mode live_dev --task K3
```

*Note*: K1/K2 block due to null quality thresholds; K4 blocks due to missing fold provenance.

### Workflow 3: Human Session Steering & Monitoring
Inspect, steer, pause, resume, or terminate any running task via `AgentInterface`:

```sh
# Inspect session status without provider calls (zero LLM tokens)
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/workspace status

# Submit guidance to active session
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/workspace steer \
  --message "Prune correlated features and evaluate gradient boosting." \
  --command-id steer-001

# Pause / Resume / Stop
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /path/to/workspace pause --command-id p-1
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /path/to/workspace resume --command-id r-1
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /path/to/workspace stop --command-id s-1
```

### Workflow 4: Bundle Verification & Standalone Serving
Delivered candidate models are 100% self-contained distributions runnable without git or harness dependencies:

```sh
# Verify candidate bundle integrity against authoritative state
mlrescurix-harness verify-bundle --bundle /path/to/bundle --sandbox false

# Standalone CLI inference
python3 -B /path/to/bundle/inference.py --input /path/to/input.json --output /path/to/preds.json

# Zero-dependency HTTP serving microservice (/health and /predict)
python3 -B /path/to/bundle/serve.py --port 8080
```

---

## 4. Setup and Installation

### Prerequisites
- **Rust Toolchain**: Rust 1.80+ (stable, 2021 edition)
- **Python**: CPython 3.11+ (tested with 3.13)
- **Docker**: Optional for host development; required for script isolation. (Verified **unavailable / daemon not running** on current host).

### Build from Source

```sh
# 1. Compile Rust control plane
cargo build --release --locked

# 2. Add binary to path
cp target/release/mlrescurix-harness ~/.local/bin/

# 3. Verify capabilities
mlrescurix-harness capabilities
```

---

## 5. Validation Status & Limits

Validation at baseline commit `441762f` (recorded in [`docs/evidence/v2/autonomy/20261023-architecture-consolidation/summary.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/autonomy/20261023-architecture-consolidation/summary.json)):

- **Full Python Suite**: **325 passed, 2 skipped, 10 warnings** (220.03s), run before final source guards.
- **Final Affected Python Suite**: **105 passed** (50.17s), including 18 new architecture regressions and final source-binding guards.
- **Rust Lib & Integration Suite**: **63 passed, 2 ignored, 1 filtered** (~67s). The filtered Docker-dependent scheduler fixture was first attempted and failed container admission; its log is retained in `rust-docker-blocker.log`.
- **Installed-Runtime Recheck**: **1 passed** (3.21s), exercising controller creation and human interface outside the git tree.
- **Static Checks**: Python `compileall` passed; `cargo build --locked --offline` passed.

> [!WARNING]
> Offline test passes establish architectural wiring, identity tracking, and boundary enforcement; they do **not** establish an entirely passing production suite, model quality, or production clean-room certification.

---

## 6. Explicit Unresolved Blockers

1. **K1 Final Contamination & Ticket Overlap**: Reused 178-row final partition and 18 shared Ticket values remain permanent historical records; staging cannot restore clean status.
2. **K1 & K2 Null Quality Thresholds**: Independent absolute quality requirements are absent; thresholds remain `null`, blocking benchmark execution.
3. **K4 Missing Provenance**: Authoritative fold index-space, prediction-context gaps, and per-origin training bounds are absent; execution remains blocked.
4. **K5 Unverified Feasibility & Quality**: Complete 1,782-entity observation calendar protocol is intact ([`docs/v2/K5_PANEL_PROTOCOL.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/K5_PANEL_PROTOCOL.json)); full-scale compute feasibility (2.9M+ rows) and model quality remain unverified.
5. **Absence of Provider Invoices**: Provider charge receipts are absent; recorded local amounts cannot be promoted to verified charges.
6. **Future Paid Calls Requirement**: Requires verified current route availability/pricing (`deepseek/deepseek-v4.1-flash-1017` via authorized routes `inference-net/fp8` or `deepinfra/fp8/flex`, no fallback) and independent content-bound operator authorization (`approve-outbound-view`).
7. **Human Steering Authority Limits**: Human steering cannot authorize uploads, approve models, alter frozen contracts, reset budgets, or clear ambiguous dispatches.
8. **Docker & Image Dependency Status**: Docker daemon was activated and verified passing scheduler crash-recovery and container isolation tests; images were cleared post-verification per operational policy. Scientific container base lacks Linux wheels for `jsonschema==4.26.0` and `psutil==6.1.1` in offline mode.
9. **Production Evidence Deficits**: Production scheduling, Linux cgroup containment, concurrent isolation, resource/fault/soak coverage, and clean-room delivery are not certified by offline tests.

---

## 7. Documentation Navigation

- [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md): Complete architecture specification, component boundaries, and contracts.
- [`docs/HUMAN_CONTROL_INTERFACE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/HUMAN_CONTROL_INTERFACE.md): Specification for `AgentInterface`, CLI usage, and approval boundaries.
- [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md): Architectural decision register (D01–D60) and superseded decisions.
- [`docs/DEVELOPMENT_AND_TESTING.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DEVELOPMENT_AND_TESTING.md): Reproducible test commands, environments, and validation limits.
- [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md): Detailed implementation status, blockers, and phased roadmap.
- [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md): Operating instructions and prime directives for continuing engineering work.
- [`docs/MIGRATION_MAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/MIGRATION_MAP.md): Mapping of removed narrative documents and minimal compatibility pointers.
- [`docs/v2/BENCHMARK_SUITE.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/BENCHMARK_SUITE.json): Machine contract for benchmark suite and split commitments.
- [`docs/v2/K5_PANEL_PROTOCOL.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/K5_PANEL_PROTOCOL.json): Machine contract for K5 Store Sales panel protocol.
- [`CHANGELOG.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/CHANGELOG.md) & [`RELEASE_NOTES.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/RELEASE_NOTES.md): Project changelog and historical release records.
