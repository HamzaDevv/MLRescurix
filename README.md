# MLRescurix

MLRescurix is a clean-slate, agent-led harness for solving one tabular machine-learning problem per project. A completed project is intended to deliver a final code artifact, an evidence-backed report card, and independently runnable inference for the same model or pipeline.

The control plane is built in Rust. M1 adds the local supervisor and managed storage; M2 adds registered CSV datasets, supervised Python CPU recipes, trusted evaluation, a one-time final-test gate, inference bundles, and a loopback dashboard. Native project state and workflow remain local and do not depend on MLflow, DVC, or a model provider.

## Current status

**M0 foundation, M1 safe execution/storage, M2 tabular workflow, and M3 constrained, resumable search are implemented.** M0 establishes the Rust workspace, versioned Rust/Python protocol contracts, fresh SQLite state, project/branch/task identities, fencing leases, receipts/events, timers, budget reservations/charges, checkpoints, and uncertain-action reconciliation. M1 adds a fair durable queue, singleton supervisor, bounded typed workers, deadlines and cleanup, environment leases, hashed artifact staging, pins, and garbage collection. M2 adds bounded CSV audit, dummy/linear/tree baselines, row-validated predictions, a trusted evaluator, frozen refit/test access, a hashed inference bundle, and a live local UI. M3 adds parameterized trusted recipes, Rust-owned experiment DAG and search sessions, managed Optuna sidecar (with offline deterministic fallback), bounded multi-fidelity screening, finalist confirmation across matched seeds, feasible Apex selection with tie tolerance, wall budget guards, and complete bundle size enforcement.

The scientific runner executes only registered built-in recipes with allowlisted parameters. It does not execute arbitrary/generated code, enforce a hard memory quota, provide a provider or autonomous agent, or certify unsupported host limits. CPU/RSS telemetry remains best effort on macOS. See the [M0 guide](docs/m0-foundation.md), [M1 guide](docs/m1-execution-storage.md), [M2 workflow guide](docs/m2-scientific-kernel.md), and [M3 search guide](docs/m3-constrained-search.md). Later milestones require their own discussion and explicit approval.

## Requirements

- Rust 1.94 or newer, with Cargo.
- Python 3.13 for the protocol package, scientific worker, and dashboard. The scientific profile is installed only in an owned environment.
- Network access on the first Cargo build, unless the Rust dependencies are already cached.

The Rust control plane and dashboard need no scientific dependencies. The managed scientific profile pins NumPy, pandas, SciPy, scikit-learn, and joblib; the optional search profile adds Optuna 5.0.0 and transitive dependencies in an isolated owned environment without altering global Python.

## Build and verify

```sh
cargo build --workspace --locked
cargo test --workspace --locked
PYTHONPATH=python/src python3 -m unittest discover -s python/tests -v
```

Cargo downloads only the pinned Rust workspace dependencies. The suites use temporary state and synthetic data. They cover shared protocol fixtures, lifecycle recovery, fair bounded probe execution, M2 dataset registration and final-test reservation, leakage-safe splits, classification/regression baselines, M3 parameterized search, multi-fidelity screening, finalist confirmation, Apex selection, budget protection, and standalone bundle inference.

If Cargo cannot write its default cache, direct it to a writable temporary location:

```sh
CARGO_HOME=/private/tmp/mlrescurix-cargo cargo test --workspace --locked
```

## Start a fresh local state

Create a policy file at a path you own. Each dimension is an integer; dimension names make their units explicit. For example:

```json
{"budget":{"calls":100,"tokens":50000,"wall_ms":3600000},"capacity":{"workers":3},"max_tasks":100}
```

Initialize a new, empty state directory. The CLI refuses to adopt an existing or unmarked directory:

```sh
cargo run -p mlrescurix-harness -- init --state /absolute/path/to/new-state --policy /absolute/path/to/policy.json
cargo run -p mlrescurix-harness -- capabilities
```

Submit a versioned command as JSON and inspect its durable receipt, project snapshot, events, or task recovery record. See the CLI and request examples in [the M0 foundation guide](docs/m0-foundation.md). For scientific search sessions:

```sh
cargo run -p mlrescurix-harness -- search-init --state /path/to/state --project <proj_id> --dataset <dataset_hash> --spec-json '<json>'
cargo run -p mlrescurix-harness -- search-run --state /path/to/state --project <proj_id> --session <session_id>
cargo run -p mlrescurix-harness -- search-inspect --state /path/to/state --session <session_id>
```

## Repository boundaries

`MLForge/` and `Redesigning.md` are reference material from the legacy project. They are excluded from Git and are not imported into MLRescurix. This repository tracks the new implementation, its contracts, tests, and current architecture/build plans only. Runtime databases, downloaded dependencies, generated artifacts, credentials, and local environments do not belong in source control.

## Project documents

- [Build plan](BUILD_PLAN.md)
- [System design and recovery requirements](SYSTEM_DESIGN.md)
- [M0 implementation and operation](docs/m0-foundation.md)
- [M1 implementation and operation](docs/m1-execution-storage.md)
- [M2 tabular workflow and limits](docs/m2-scientific-kernel.md)
- [M3 constrained, resumable search](docs/m3-constrained-search.md)
