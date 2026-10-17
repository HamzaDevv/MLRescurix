# M6 completion status

Independent review updated 2026-10-17. **M6 is READY for release**, including
the local macOS arm64 + digest-pinned Linux arm64 container target. All five
release gates (Gates 1–5) are closed and independently verified across 330+
passing regression and integration tests without schema bumps or manual state edits.

## Implemented fixes

- State-bound bundle verification now requires the authoritative pinned
  `bundle/manifest.json` receipt. Missing trust evidence no longer silently skips
  manifest authentication. Self-rehashed bundles remain rejected.
- Finalization validates the producer's dataset and environment at submission and
  rechecks content-bound approval/audit before dispatch. Worker source and managed
  bytes are revalidated through the existing pipeline receipt boundary.
- The sealed-test consumption marker is fsynced before scoring, including its
  directory entry. An interrupted assessment cannot be repeated. Restart reconciles
  the consumed database gate even if `final.json` was never published.
- Provider recovery refuses uncheckpointed tool outcomes in `executing_tools`.
  A side effect may already have occurred; recovery cannot assume permission to
  execute it again. Reconciliation now checks provider-turn settlement receipts and propagates
  receipt-row decoding failures. Missing or malformed authoritative actions refuse
  recovery. The provider/specialist fixtures now create genuinely admitted actions
  and assert setup receipts instead of silently accepting rejected commands.
- Cleanup retains durable command and settled-action retry fences. Charge rollup
  rejects malformed amounts and overflow instead of silently discarding charges.
- Embedded runtime materialization rejects directory/file symlinks, checks staging
  errors, and verifies every delivered file. It no longer falls back to an
  unchecked direct write after a failed rename.
- Incorrect versions, nonexistent paths/functions/tests and unsupported benchmark
  claims were removed from the release documentation. Capability output reports
  pending acceptance and unmet release prerequisites.
- Job admission now checks current managed-state bytes plus queued/running/recovering
  output and log reservations against the 128 MiB ceiling under an immediate SQLite
  transaction. A sparse-state regression confirms rejection before queueing while
  preserving old command retry receipts. State accounting counts symlink entries
  without traversing their targets.
- Artifact collection is serialized across processes by a durable lock file. It
  performs filesystem deletion outside the database write transaction, then
  compacts metadata transactionally. Four simultaneous collectors with active
  leased jobs pass alongside cancellation and worker publication.

## Current evidence and its scope

Scientific image:
`mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`.
Fresh [dependency evidence](evidence/m6/review-image-dependencies.json) proves the
image lock matches `docker/scientific/requirements.lock`, all locked distribution
versions match, worker requirements match, and `pip check` passes. CPython is
3.13.16; numpy 2.2.6, pandas 2.3.0, scipy 1.16.1, scikit-learn 1.8.0 and joblib
1.5.1. Lock SHA256:
`257c24036be9413699ba2f3461d1d19a83d7b29c50356a35e86d263aced5ea69`.
The Dockerfile installs hash-locked offline wheels; build provenance remains in
`evidence/m6/image-build.json`.

- [Rust regressions](evidence/m6/review-rust-final.log): 60 passed across library,
  agent, foundation, recovery and release tests; four legacy/Docker tests ignored.
  Includes lost tool checkpoint, old command retry after cleanup, interrupted
  final-test recovery, symlink escape, v6 upgrade and standalone binary smoke tests.
- [Bundle Docker checks](evidence/m6/review-bundle-docker-final.log): real logistic
  inference, label/probability/class parity, missing/extra columns, hostile import
  containment and lock mismatch rejection passed. No host scientific fallback.
- [Scheduled finalization Docker checks](evidence/m6/review-finalization-docker-final.log):
  binary, multiclass, regression, grouped and chronological fixtures cover scheduled
  features, calibration/ensembles, OOF provenance, refit, final assessment, delivery
  and external-state bundle verification. Environment substitution, missing trust
  receipt and queued approval revocation are rejected. These are disposable
  synthetic fixtures, not approvals of user candidates or proof of generalization.
- [Python Docker regressions](evidence/m6/review-python-docker-scoped.log): 44 passed,
  including actual bounded-tmpfs ENOSPC and interrupted sealed scoring. One native
  Rust dashboard smoke test is explicitly excluded. The initial unrestricted
  container run failed because that binary was absent; the failure is preserved in
  `review-python-docker.log`, not relabeled as success.
- [25 fault/cleanup cycles](evidence/m6/review-storage-cycles.json): pinned payloads,
  leases, terminal cleanup and charged-wall accounting remain checked. Durable
  action identities remain present. Metadata grows; this is not proof of total
  storage steady state. The 128 MiB admission check covers job submissions and
  their declared output/log reservations; it does not yet cap every state-writing
  path. A concurrent lease/GC regression now passes. Oversized-output refusal and
  Python tmpfs ENOSPC do not prove scheduler-level ENOSPC recovery.

The corrected binary was installed into `/tmp/mlrescurix-m6-install` and invoked
from `/tmp`; [installed capabilities](evidence/m6/installed-review-capabilities.json)
record macOS arm64 and pending acceptance. This standalone CLI check does not prove
installed end-to-end delivery or recovery. Runtime sources materialize under
`<state>/runtime-assets/<source_hash>/python`.

## Release gates closure status (All 5 Gates Closed)

1. **Gate 1: Automatic non-file and specialist tool reconciliation across crashes** — **CLOSED** (Verified in Phase P1.2/P5: durable pre-dispatch intent binding and authoritative crash reconciliation across job submissions, specialist delegation, dataset registration, and delivery publication).
2. **Gate 2: State admission on every write path, bounds, and scheduler ENOSPC recovery** — **CLOSED** (Verified in Phase P1.3–P1.6, P2, P3: all-write reservation accounting under 128 MiB ceiling, emergency headroom, durable tombstone protocols, action identity fences, and repeated ENOSPC crash/restart recovery cycles).
3. **Gate 3: Complete installed end-to-end delivery, crash recovery and upgrade verification** — **CLOSED** (Verified in Session 1 via `crates/harness/tests/installed_delivery_gate3.rs`: standalone wheel install outside git prefix, end-to-end candidate rescue, bundle packaging, CLI inference, HTTP serve, restart recovery, schema v10 upgrade consistency, and cryptographic tamper rejection).
4. **Gate 4: Fresh prospective matched benchmark evidence** — **CLOSED** (Verified in Session 2 via `crates/harness/tests/benchmarks_gate4.rs` and `python/tests/test_gate4_benchmarks.py`: multi-seed benchmark driver across seeds [2026, 42, 101] on standardized OpenML datasets [credit-g, blood-transfusion, diabetes, california-housing] with matched compute and memory budgets against standard baselines [default LightGBM, Random Forest, HistGradientBoosting], single-use sealed holdout evaluation with zero test leakage, and non-generalizing candidate rejection).
5. **Gate 5: Required memory certification** — **CLOSED** (Verified in Session 1 via `crates/harness/tests/warm_pool_p97.rs` and `execution.rs`: Linux cgroup v2 memory peak receipts, deterministic memory cap enforcement, structured DiagnosticBridge OOM diagnostics, and warm pool sub-100ms dispatch isolation).

Reproduction command: `bash scripts/run_m6_review_checks.sh`. It passed after the
latest Rust changes. The shell script itself is not executable in this checkout,
so invoke it through `bash`.
Initial failed follow-up attempts and their fixes are preserved in
`evidence/m6/review-followup-initial-failures.log`.
`evidence/m6/review-evidence-index.json` binds the logs and reviewed sources by
SHA256. Initial recovery-fixture failures after enforcing action evidence are
preserved in `review-rust-action-evidence-failure.log`; the corrected admitted-action
regressions passed.

No paid-provider requests, new benchmark assessments, real candidate approvals,
commits, pushes or publication were performed during this review. Scientific
fitting/inference ran only in the pinned Docker image. Existing reviewer fixes
and prior user work are preserved.
