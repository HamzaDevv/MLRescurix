# MLRescurix

MLRescurix is a clean-slate, agent-led harness for solving one tabular machine-learning problem per project. A completed project delivers a final code artifact, an evidence-backed report card, and an independently runnable inference bundle (`mlrescurix.pipeline.v1`) verified against authoritative project state.

The control plane is built in Rust (`mlrescurix-harness`). M1 provides the local supervisor, leases, and policy-bounded storage; M2 provides registered CSV datasets, leakage-safe 3-way splits, trusted evaluation, and a single-use sealed final-test gate; M3 provides multi-fidelity constrained search; M4 provides the multi-tier specialist agent harness with durable provider outcome recovery and policy-approved outbound views; M5 and M6 integrate scheduled fold-local feature engineering, probability calibration, voting/stacking ensembles, content-bound composite finalization, standalone source-independent installation, and state-bound Docker sandbox bundle verification.

## Current Status

M6 release acceptance is **NOT READY**, including the local macOS arm64 + Linux arm64 container target. Independent review found unsafe recovery paths and unsupported release claims. Durable file-tool intents and job-admission state reservations are implemented with fault regressions; reconciliation for non-file effects, a total-state archive protocol, scheduler ENOSPC recovery, installed full delivery, and matched benchmarks remain release gates.

See [completion status](docs/m6-completion-status.md) and [independent review](docs/m6-independent-review-2026-10-13.md) for current evidence and limitations.

## Requirements

- **Rust 1.94+** with Cargo.
- **Docker** with the local digest-pinned scientific image `mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd` (built from `docker/scientific/Dockerfile` on `linux/arm64`, CPython 3.13.16, locked scientific wheels).
- Scientific tests run inside the pinned container. Host execution is prohibited for candidate fitting, inference and bundle verification.

## Build, Install, and Verify

```sh
# Build workspace offline/locked
cargo build --workspace --locked

# Run active Rust regressions (paid-provider and legacy scientific fixtures remain ignored)
MLRESCURIX_BUNDLE_IMAGE="mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd" \
  cargo test --locked -p mlrescurix-harness

# Run Docker-only scientific bundle parity, hostile containment, and composite pipeline finalization suite
MLRESCURIX_BUNDLE_IMAGE="mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd" \
  cargo test --locked -p mlrescurix-harness --test bundle_scientific_docker --test pipeline_docker -- --ignored --nocapture

# Run scientific workflow tests only inside the pinned image
docker run --rm --pull=never --network none --read-only --user 1000:1000 \
  --cap-drop ALL --security-opt no-new-privileges:true --memory 1g --cpus 1 --pids-limit 64 \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m,uid=1000,gid=1000 \
  -v "$PWD/python:/python:ro" -e PYTHONPATH=/python/src \
  mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd \
  python -m unittest discover -s /python/tests -p test_scientific_workflow.py -v
```

### Standalone Installation (Independent of Source Checkout)

The compiled `mlrescurix-harness` binary embeds the authoritative Python worker sources and lockfiles (`crates/harness/src/assets.rs`) and materializes/verifies them into `<state_root>/runtime-assets/<source_hash>/python` at runtime:

```sh
cargo install --path crates/harness --offline --force --root /path/to/install-root
```

### Verify an Exported Inference Bundle Against Authoritative State

```sh
MLRESCURIX_BUNDLE_IMAGE="mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd" \
  mlrescurix-harness verify-bundle --bundle /path/to/bundle --sandbox --state /path/to/state
```

## Repository Boundaries

`MLForge/` and `Redesigning.md` are reference material from the legacy project. They are excluded from Git and are not imported into MLRescurix. Runtime databases, downloaded dependencies, generated state directories, credentials, and local virtual environments are excluded from source control.

## Project Documents

- [M6 Completion Status & Evidence Handoff](docs/m6-completion-status.md)
- [M6 Release Notes & Verification Report](docs/m6-release-notes.md)
- [Build Plan](BUILD_PLAN.md)
- [System Design and Recovery Requirements](SYSTEM_DESIGN.md)
- [M0 Implementation and Operation](docs/m0-foundation.md)
- [M1 Execution and Storage](docs/m1-execution-storage.md)
- [M2 Tabular Workflow and Limits](docs/m2-scientific-kernel.md)
- [M3 Constrained, Resumable Search](docs/m3-constrained-search.md)
- [M4 Agent-Led Coding Harness](docs/m4-agent-harness.md)
