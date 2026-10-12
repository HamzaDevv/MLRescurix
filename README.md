# MLRescurix

MLRescurix is a clean-slate, agent-led harness for solving one tabular machine-learning problem per project. A completed project is intended to deliver a final code artifact, an evidence-backed report card, and independently runnable inference for the same model or pipeline.

The control plane is being built in Rust. Isolated Python workers will own scientific computation in a later milestone. Native project state and workflow remain local and do not depend on MLflow, DVC, or a model provider.

## Current status

**M0 durable foundation is implemented.** It establishes the Rust workspace and lifecycle CLI, versioned Rust/Python protocol contracts, fresh SQLite state with migrations, project/branch/task identities, fencing leases, receipts and events, timers, budget reservations and charges, checkpoints, and uncertain-action reconciliation.

M0 uses scripted actors only. It does not train or evaluate models, run worker processes, enforce host resource limits, connect to providers, publish official scores, finalize models, or provide a dashboard. Those boundaries are documented in [docs/m0-foundation.md](docs/m0-foundation.md). Later milestones require their own discussion and explicit approval.

## Requirements

- Rust 1.94 or newer, with Cargo.
- Python 3.13 for the standard-library protocol package and its tests.
- Network access on the first Cargo build, unless the Rust dependencies are already cached.

The core Python protocol package has no scientific dependencies. Installing it does not modify a global Python environment; use a virtual environment if you want to install it.

## Build and verify

```sh
cargo build --workspace --locked
cargo test --workspace --locked
PYTHONPATH=python/src python3 -m unittest discover -s python/tests -v
```

Cargo downloads only the pinned Rust workspace dependencies. The integration suite uses temporary databases and synthetic actors. It covers the shared positive/negative protocol fixtures, ten project records with three parallel branch tasks, aggregate/project reservation races, deduplication, fencing, crash recovery, uncertain charges, timers, pending questions, and rejection of fake official evidence.

If Cargo cannot write its default cache, direct it to a writable temporary location:

```sh
CARGO_HOME=/private/tmp/mlrescurix-cargo cargo test --workspace --locked
```

## Start a fresh local state

Create a policy file at a path you own. Each dimension is an integer; dimension names make their units explicit. For example:

```json
{"budget":{"calls":100,"tokens":50000},"capacity":{"workers":3},"max_tasks":100}
```

Initialize a new, empty state directory. The CLI refuses to adopt an existing or unmarked directory:

```sh
cargo run -p mlrescurix-harness -- init --state /absolute/path/to/new-state --policy /absolute/path/to/policy.json
cargo run -p mlrescurix-harness -- capabilities
```

Submit a versioned command as JSON and inspect its durable receipt, project snapshot, events, or task recovery record. See the CLI and request examples in [the M0 foundation guide](docs/m0-foundation.md).

## Repository boundaries

`MLForge/` and `Redesigning.md` are reference material from the legacy project. They are excluded from Git and are not imported into MLRescurix. This repository tracks the new implementation, its contracts, tests, and current architecture/build plans only. Runtime databases, downloaded dependencies, generated artifacts, credentials, and local environments do not belong in source control.

## Project documents

- [Build plan](BUILD_PLAN.md)
- [System design and recovery requirements](SYSTEM_DESIGN.md)
- [M0 implementation and operation](docs/m0-foundation.md)
