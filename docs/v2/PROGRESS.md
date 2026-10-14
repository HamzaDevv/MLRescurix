# Persistent progress and session handoff

Last updated: 2026-10-14 (Asia/Kolkata).
Baseline source commit: `abf1f235030a79f9a606b429e34608431e37603d`.

## Current state

- Current work completed:
  - P1.1 mutation and side-effect inventory authored at `docs/v2/P1_MUTATION_MATRIX.md`
    covering all 48 SQLite tables across M0/M1/M2/M4, 11 filesystem mutation paths, and runner entrypoints.
  - P1.2 scheduler job-submission reconciliation implemented in `crates/harness/src/agents/recovery.rs`
    with pre-dispatch intent binding and authoritative job reconciliation.
  - P1.2 non-file tool reconciliation implemented in `crates/harness/src/agents/recovery.rs`
    for specialist delegation (`delegate_task`, `delegate`, `schedule_specialist`),
    dataset registration (`register_dataset`), and delivery publication (`request_finalization`, `publish_delivery`, `publish_bundle`).
  - P1.3 disk allocation budgets, reservation accounting, and ENOSPC recovery implemented across
    `crates/harness/src/execution.rs`, `storage.rs`, `assets.rs`, `bundle.rs`, and `agents/{context,gateway,orchestrator,workspace}.rs`.
  - P1.4 lease-aware archival and tombstones, compact retry fences, and interrupted publication reconciliation implemented across
    `crates/harness/src/execution.rs`, `crates/harness/src/storage.rs`, `crates/harness/src/agents/recovery.rs`, and `crates/harness/src/main.rs`.
  - Comprehensive fault regressions authored covering:
    1. Unpinned terminal job artifact collection and SHA-256 artifact tombstones.
    2. Pinned artifact preservation (`pinned = 1`) and terminal job pruning tombstones with active lease / active task protections.
    3. Command request body compaction beyond active window (`MAX_ACTIVE_COMMAND_RAW_BODIES = 24`) with payload hash idempotency retries.
    4. Settled action compaction beyond `MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT = 64` with durable `action_fences` preventing action identity reuse.
    5. Interrupted publication crash recovery adopting valid existing final directory outputs after rename.
    6. Leaked bundle staging directory sweep (`staging/bundle_verification_*` and `workflows/<p>/.bundle_stage`).
    7. Interrupted delivery publication pre-admission crash recovery with explicit uncertain outcomes and rollback of unreserved finalizing state.
  - P1.5 complete: Docker cgroup v2 capability detection and per-job limits, certification refusal, validated peak/event receipts, atomic exact container ID and ownership metadata, and fenced daemon survivor adoption/reaping.
  - P1.6 complete: repeated crash/restart recovery under actual ENOSPC on a disposable bounded volume.
  - Full `cargo test --locked` passed; live pinned-Docker cgroup/OOM/survivor checks, six end-to-end scientific pipeline modes, and real ENOSPC regression passed. See the fresh evidence below.
  - Architecture decisions D20–D25 recorded in `docs/v2/DECISIONS.md`.
  - Historical M6 evidence preserved untouched.
  - Runtime implementation: P1.1–P1.6 complete and verified. P2 (P2.1–P2.6 and full P2 Roadmap Acceptance Gate) complete and verified. P3 (P3.1–P3.5 and full Phase P3 Acceptance Gate) complete and verified. P4 (P4.1–P4.3 and full Phase P4 Acceptance Gate) complete and verified:
    1. P4.1: Versioned generic tool adapters: `workspace_fs` (read, write, patch, list, glob, search with optimistic concurrency SHA-256 hash checks), `sandbox_exec` (isolated execution with `/workspace:ro`, `/scratch:rw` tmpfs, `/outputs:rw` tmpfs, thread caps, and honest fallback refusal when Docker is unavailable), `artifact_crud` (publish, get, list, lineage, pin, unpin, tombstone), `discover_capabilities` (tools, container status, image ref, presets, limits), `dataset_query` (schema inspection without leaking label/holdout values), and `experiment_tree` (DAG visibility). Legacy `file_*` tools retained for backward compatibility.
    2. P4.2: Executable candidate protocol & manifest: `manifest.json` schema (`task_type`, `entrypoint`, `supported_operations`, `stable_row_keys`, `class_order`, `raw_input_schema`, `source_hash`), CLI `fit`/`predict` protocol, candidate prediction validation (probabilities in `[0, 1]`, row sum $= 1.0 \pm 10^{-3}$, class alignment), and diagnostic slices with minimum support threshold (`max(20, 0.05 * N)`).
    3. P4.3: Decoupled candidate runner: candidate evaluation without modifying any recipe registry, harness-driven cross-validation (`StratifiedKFold`, `KFold`, `GroupKFold`, `TimeSeriesSplit`), fold-level label isolation (validation features provided strictly without target column), and baseline validation. Decoupled `recipe` in `experiment_spec.schema.json` and `execution.rs::migrate()` to allow arbitrary recipes of length $\le 128$ while preserving `"hist_boosting"` in DDL for backwards compatibility.
    4. Phase P4 Acceptance Gate passed: agents create and fix arbitrary candidate code without modifying a recipe registry; evaluation, scoring, and budgets remain under trusted authority; split/label isolation strictly enforced; untrusted agents cannot publish or forge `kind = "scorecard"` artifacts; adaptive validation query budget (max 50 queries per project) prevents holdout overfitting. Verified across 9 integration tests in `crates/harness/tests/candidate_tools_p4.rs`, 5 unit tests in `python/tests/test_candidate_runner.py`, and 89 passed tests across full regression suites (106 total passed tests). Fresh logs in `docs/evidence/v2/p4/20261014-p4-gate-closure/`. Phase P4 is closed.
- Release: M6 remains **NOT READY**, as recorded in `docs/m6-completion-status.md`.
  - Authorization: P4.1–P4.3 and Phase P4 Acceptance Gate were authorized and completed in this session. P5 or later phases are NOT begun.
  - Next recommended slice: Phase P5 (when explicitly authorized). M6 release gates remain independently open.

| Phase | Status | Evidence / next step |
| --- | --- | --- |
| P0 | Complete | This doc set, archived inputs, static audit and documentation validation |
| P1 | complete | P1.1–P1.6 verified; see `20261014-p1-review-resources/` evidence. M6 release remains independently NOT READY. |
| P2 | complete | P2.1–P2.6 and full P2 roadmap acceptance gate complete and fully evidenced. Populated upgrade preservation, partial CAS/DB reconciliation, protected references under concurrent GC, and event restart/reconnect consistency verified across 14 tests in artifact_registry_p2 and full regression suites. Evidence in `20261014-p2-gate-closure/`. M6 remains independently NOT READY. |
| P3 | complete | P3.1–P3.5 and full Phase P3 Acceptance Gate verified across 13 tests in resource_broker_p3 and full regression suites (88 total passed tests). Fresh logs and timelines in `20261014-p3-gate-closure/`. M6 remains independently NOT READY. |
| P4 | complete | P4.1–P4.3 and full Phase P4 Acceptance Gate verified across 9 tests in candidate_tools_p4, 5 tests in test_candidate_runner.py, and full regression suites (106 total passed tests). Fresh logs in `20261014-p4-gate-closure/`. M6 remains independently NOT READY. |
| P5 | Planned | Context/task recovery on generic tool substrate |
| P6 | Planned | Read/resource views after P2/P3; new semantics/steering after P4/P5 |
| P7 | Planned | Installed delivery, prospective benchmarks, release matrix |
| P8 | Planned | Pilot-calibrated feasibility reports |
| P9 | Optional/planned | Approve and certify each adapter/backend separately |

Allowed implementation statuses: planned, in_progress, blocked, implemented_unverified,
verified, complete. Complete requires the phase gate and evidence; code presence
or a specialist's assertion alone does not meet it. P0 is documentation complete.

## M6 acceptance register

| Gate | Status at planning baseline | Owner in this roadmap |
| --- | --- | --- |
| 1: automatic non-file/specialist reconciliation | Closed (verified in P1.2: job submission, delegation, registration, publication) | P1.2, P5 |
| 2: all-write state bounds/archive/scheduler ENOSPC | In progress (P1 managed paths are bounded and restart/ENOSPC recovery verified; P2 still owns broader registry and lifecycle bounds) | P1.3–P1.6, P2 |
| 3: installed end-to-end delivery/recovery/upgrade | Open | P7.5 |
| 4: fresh prospective matched benchmark evidence | Open | P7.6 |
| 5: required memory certification | In progress (required certification enforced on verified cgroup v2 Docker hosts; platform matrix and broader delivery gate remain) | P1.5, P3, P7.3 |

## Session ledger

| Date | Scope | Outcome |
| --- | --- | --- |
| 2026-10-14 | P0 planning and two-blueprint review | Saved source specs, corrected unsafe/unsupported claims, created phased architecture and continuation docs; no runtime tests or implementation |
| 2026-10-14 | Argon planning feedback refinement | Confirmed local generator/gateway/composite touchpoints; added scoped fold execution, rescue provenance, container identity, outbound policy and compact digests; corrected replay/privacy/quota/version claims; preserved review and regenerated packet |
| 2026-10-14 | P1.1 inventory & P1.2 scheduler submission reconciliation | Authored P1_MUTATION_MATRIX.md; implemented pre-submission intent binding & crash reconciliation in recovery.rs; added 7 fault regressions in agent_m4.rs; added D20; full locked Rust suite passed; captured evidence in docs/evidence/v2/p1/20261014-p1-reconcile/ |
| 2026-10-14 | P1.2 specialist delegation & non-file reconciliation | Extended pre-dispatch intent binding & authoritative crash reconciliation to delegation, dataset registration, and delivery publication; added 11 fault regressions in agent_m4.rs; added D21; full locked test suites passed; captured evidence in docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/ |
| 2026-10-14 | P1.3 disk allocation budgets, reservation accounting, & ENOSPC recovery | Implemented unified reservation accounting (StorageWriteKind) under 128 MiB policy with zero silent exemptions, 4 MiB emergency headroom, 2 MiB WAL reserve, configurable category budgets via storage_budgets.json, scheduler POSIX ENOSPC (code 28) staging purge and reservation release, saturation cancellation without DB deadlocks (TransactionBehavior::Immediate), and 5 restart/cleanup cycles; added D22; full locked suite passed; captured evidence in docs/evidence/v2/p1/20261014-p1-storage-accounting/ |
| 2026-10-14 | P1.4 lease-aware archival/tombstones, compact retry fences, & publication recovery | Implemented bounded archival & SHA-256 tombstones, active lease & pinned artifact protections, command body compaction to SHA-256 hash beyond 24 commands, settled action in-place compaction beyond 64 actions with durable action_fences, interrupted publication adoption for valid final directories, leaked bundle staging sweep, and delivery publication rollback of unreserved finalizing state; added D23; full locked suite passed (78 passed, 0 failed, 7 ignored); captured evidence in docs/evidence/v2/p1/20261014-p1-archival-fencing/ |
| 2026-10-14 | P1.5–P1.6 cgroup resource receipts, Docker survivor recovery, and actual ENOSPC fault cycle | Added per-job cgroup v2 limits and host feature detection; refused required certification on unsupported hosts; collected/validated memory peak and events before teardown; atomically persisted exact Docker IDs and owner/state-root/project/lease/generation/intent labels; discovered survivors after restart and retained leases until confirmed exit; added mismatch, stop/remove failure, still-running, malformed-receipt, unsupported-host and race regressions; verified real ENOSPC recovery on disposable volume. Full locked suite, live pinned-image cgroup/OOM/survivor tests, six-mode pinned pipeline and ENOSPC test passed. Added D24–D25; user_version remains 9; M6 evidence untouched. Evidence: `docs/evidence/v2/p1/20261014-p1-review-resources/`. |
| 2026-10-14 | P2.1 artifact registry & versioned contract design | Authored read-only inventory of M0–M6 artifact tables, filesystem paths, and APIs; designed CAS write protocol (`<state>/cas/<hash[0..2]>/<hash>`), target schema (`user_version = 10` DDL: `blobs`, `artifacts`, `artifact_versions`, `artifact_lineage`, `artifact_annotations`, `project_selections`, `durable_events`), two-tier summary bounds (<=512 B `index_digest`, <=4096 B `summary_text`), non-destructive migration strategy with dual-write compatibility, and lease-aware GC; user_version remains 9; no runtime edits; added D26; captured design in `docs/v2/P2_1_ARTIFACT_DESIGN.md`. |
| 2026-10-14 | P2.2 runtime CAS, schema v10 migration & artifact registry | Implemented CAS module (`crates/harness/src/cas.rs`) with pre-admission reservation checks, SHA-256 validation, read-only permissions (`0o444`), directory manifests; additive schema migration to `user_version = 10` (`schema_m7_artifacts.sql`) with backfill from `managed_artifacts` and `scientific_datasets`; relational registry (`crates/harness/src/artifacts.rs`) with two-tier summary bounds (`index_digest` <= 512 B, `summary_text` <= 4096 B), evaluator-only scorecard authority, lineage, annotations, and Apex selection; dual-write backward compatibility in `Supervisor::finish` and `pin_artifact`. Decoupled ephemeral `jobs` GC from persistent versions. All 5 tests in `artifact_registry_p2` passed; 6 regression suites passed (90 tests passed, 0 failed). Captured evidence in `docs/evidence/v2/p2/20261014-p2-artifact-registry/`. |
| 2026-10-14 | P2.3–P2.5 artifact queries, durable event journal, and read pool | Added bounded inspection and lineage queries, pin/unpin, evaluator-only scorecard capability, trigger-backed atomic durable event journal with cursor snapshot/replay/reset behavior, and four shared query-only reader connections with non-blocking acquisition. Eight artifact tests plus recovery/foundation/release regressions passed in the prior session; evidence in `docs/evidence/v2/p2/20261014-p2-inspection-journal-readers/`. |
| 2026-10-14 | P2.6 revisioned successor/tombstone commands | Added `publish_successor_command` and `tombstone_artifact_command` using the shared project revision and existing durable `(project, command_id)` request/receipt fences. Successors atomically publish an immutable version, advance current pointer, add `supersedes` lineage and increment revision; tombstones retain version/lineage/bytes and refuse pinned, selection-history-referenced, or actively produced artifacts. Tombstone event trigger is installed idempotently on already-v10 states. `artifact_registry_p2` (10 passed), `recovery_cycles` (8 passed, 2 ignored), `foundation` (26 passed), and `release_m6` (5 passed, 3 ignored) passed serial rerun; one initial foundation timing failure under concurrent Cargo runs passed in isolated and serial reruns. Evidence in `docs/evidence/v2/p2/20261014-p26-successor-tombstone/`. P2 remains in progress; M6 remains NOT READY. |
| 2026-10-14 | P2 Roadmap Gate Closure | Verified all remaining P2 acceptance checks: populated upgrade preservation (retaining `final_test_gates`, `search_sessions`, `trial_results`, `pipeline_receipts`), partial CAS/DB commit reconciliation (`cas/.staging` sweep in `collect_artifacts`, blob adoption on retry, SHA-256 corruption repair), protected references under concurrent GC (pinned artifacts, Apex selection, lineage survive GC), and event restart/reconnect consistency across process boundaries. 14 `artifact_registry_p2` tests passed; full locked suites passed (`foundation`, `recovery_cycles`, `release_m6`, `agent_m4`, `search_m3`, `lib`, python protocol). Evidence in `docs/evidence/v2/p2/20261014-p2-gate-closure/`. P2 complete; M6 remains NOT READY. |
| 2026-10-14 | P3.1 capability discovery, presets, headroom, & infeasible-request rejection | Discovered effective host and container-VM CPU, memory, disk, and accelerator capabilities; clamped container jobs to container-VM limits (`min(host, vm)`); configured headroom (1 core, 512 MiB RAM, 64 MiB disk); added bounded presets (`MICRO`, `STANDARD`, `HEAVY`, `CUSTOM`); early infeasible rejection in submit() with zero SQLite jobs rows inserted. 5 tests passed in resource_broker_p3; full regression suites passed. Evidence in `docs/evidence/v2/p3/20261014-p31-capability-discovery/`. M6 remains NOT READY. |
| 2026-10-14 | P3.2–P3.5 & Phase P3 Acceptance Gate Closure | Implemented atomic capacity reservations and dimension distinction (workers vs cpu_threads vs memory_bytes), durable resource leases and transitions (reserved -> active -> heartbeated -> recovering -> reaped -> released), weighted fair queueing with aging term and stable tie ordering, thread caps across execution backends and scikit-learn n_jobs via _leased_threads(), pressure backoff via disk headroom and MLRESCURIX_PRESSURE_BACKOFF, finalization/control-plane headroom preservation (1 worker, 1 CPU thread, 512 MiB RAM), per-project cancellation and pause isolation, and safe Docker survivor discovery. All 13 tests in resource_broker_p3 passed, including 50-job 10x5 stress test and acceptance gate; 88 passing tests across all full repository regression suites. Added D33. Fresh logs in docs/evidence/v2/p3/20261014-p3-gate-closure/. Phase P3 is closed; M6 remains NOT READY. |

## Update at the end of every implementation session

### Implementation Session 2026-10-14 (P1.1 & P1.2 Slice)

```text
Date / phase / task IDs: 2026-10-14 / P1 / P1.1 (mutation matrix), P1.2 (scheduler submission recovery)
User-authorized scope: P1.1 mutation/effect inventory and P1.2 scheduler job-submission reconciliation.
Starting commit and pre-existing dirty files: abf1f235030a79f9a606b429e34608431e37603d; dirty: README.md, docs/v2/
Changes and source paths:
  - docs/v2/P1_MUTATION_MATRIX.md: Comprehensive inventory of 48 SQLite tables across M0-M4, 11 filesystem mutation paths, runner entrypoints, distinguishing verified source from proposed improvements.
  - crates/harness/src/agents/recovery.rs: canonicalize_value, build_scheduler_submission_intent (pre-dispatch cryptographic binding to project, task, turn, call_id, action, canonical arguments hash, dataset hash, ready environment identity, protocol hash, and planned JobSpec hash into tool_outcomes), reconcile_scheduler_submission (authoritative reconciliation against admitted job without duplicates, explicit unconfirmed_submission_crash on missing admission, conflicting spec/argument rejection, stale generation fencing).
  - crates/harness/tests/agent_m4.rs: p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions (7 fault regressions covering crashes before submission, crashes after admission, crashes after outcome before checkpoint, duplicate recovery, conflicting payloads, stale generations, and authoritative single reservation/charge settlement).
  - docs/v2/DECISIONS.md: Recorded D20 and P1.2 resolution details.
Decisions added or revised: D20 added (pre-dispatch intent binding and authoritative job reconciliation for scheduler submission).
Exact verification commands and exit results:
  - cargo test --locked --test agent_m4 -- p1_scheduler_job_submission (exit code 0; 1 passed)
  - cargo test --locked --lib (exit code 0; 4 passed)
  - cargo test --locked --test agent_m4 (exit code 0; 22 passed)
  - cargo test --locked --test foundation (exit code 0; 26 passed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 4 passed, 1 ignored)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p1/20261014-p1-reconcile/review-rust-p1.log
  - docs/evidence/v2/p1/20261014-p1-reconcile/p1-scheduler-recovery.log
  - docs/evidence/v2/p1/20261014-p1-reconcile/full-locked-harness.log
  - Historical M6 evidence preserved in docs/evidence/m6/ untouched.
Implemented but unverified: None in this slice (all implemented code paths verified by fault regressions and suite).
Failure/recovery scenarios still open:
  - Specialist worker crash / handoff reconciliation for non-file tools (remaining P1.2).
  - Storage exhaustion & ENOSPC rollback across WAL / temporary staging (P1.3–P1.6).
  - Candidate / scientific execution on host remains strictly prohibited; dockerized container tests require active Docker engine.
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 9; schema_m6_recovery.sql tool_outcomes table utilized).
  - Exactly one reservation and one charge per admitted job guaranteed; zero double-billing or duplicate queue entries on crash recovery.
Next small actionable slice:
  - Remaining P1.2 specialist reconciliation or P1.3 disk allocation budgeting.
```

### Implementation Session 2026-10-14 (P1.2 Specialist Delegation & Non-File Reconciliation)

```text
Date / phase / task IDs: 2026-10-14 / P1 / P1.2 (specialist delegation, dataset registration, and delivery publication recovery)
User-authorized scope: Complete P1.2 non-file tool reconciliation (specialist delegation, dataset registration, delivery publication).
Starting commit and pre-existing dirty files: abf1f235030a79f9a606b429e34608431e37603d; dirty: README.md, docs/v2/, docs/evidence/v2/, scripts/build_argon_review_packet.py
Changes and source paths:
  - crates/harness/src/agents/recovery.rs:
      * build_delegation_intent & reconcile_delegation: Durable pre-dispatch cryptographic binding to project, task, turn, call_id, action, generation, canonical inputs, and target child task identity. Pre-admission crashes deterministically emit explicit unconfirmed_delegation_crash with effect_outcome="uncertain". Post-admission crashes reconcile authoritative child task state without duplicate child tasks or double billing. Rejection on conflicting arguments, target task identity reuse, or stale generations.
      * build_dataset_registration_intent & reconcile_dataset_registration: Binds canonical inputs, dataset SHA256, path, and target dataset identity. Pre-admission crashes yield unconfirmed_registration_crash with effect_outcome="uncertain". Post-admission crashes recover authoritative dataset record without duplicate entries. Contradictory content is rejected.
      * build_delivery_publication_intent & reconcile_delivery_publication: Binds candidate_id, target_bundle_id, finalize_job_id, canonical arguments, generation, and project policy. Validates approved critic audit or content decision. Pre-admission crashes yield unconfirmed_publication_crash with effect_outcome="uncertain". Post-admission crashes reconcile authoritative finalize job status without duplicate jobs.
      * recovery::execute: Dispatches intent building and reconciliation across all non-file mutating tool families ("delegate", "delegate_task", "schedule_specialist", "register_dataset", "request_finalization", "publish_delivery", "publish_bundle").
  - crates/harness/src/agents/orchestrator.rs:
      * Added schedule_specialist_with_ids to pass custom child task and branch IDs.
      * Added dispatch routing for register_dataset, delegate_task, and publish_delivery / request_finalization.
  - crates/harness/src/agents/roles.rs:
      * Added delegate_task, register_dataset, publish_delivery, publish_bundle capabilities to specialist roles.
  - crates/harness/src/agents/tools.rs:
      * Added tool descriptors for register_dataset, delegate_task, and publish_delivery.
  - crates/harness/tests/agent_m4.rs:
      * Added p1_delegation_and_registration_crash_reconciliation_and_fault_regressions covering 11 fault modes across delegation, dataset registration, and delivery publication.
      * Increased supervisor poll loop bound in p1_scheduler_job_submission to 250 to ensure reliable concurrent test execution.
  - docs/v2/DECISIONS.md:
      * Recorded D21 and updated P1 resolution status to complete for P1.2.
Decisions added or revised: D21 added (pre-dispatch intent binding and authoritative crash reconciliation for specialist delegation, dataset registration, and delivery publication).
Exact verification commands and exit results:
  - cargo test --locked --test agent_m4 -- p1_ (exit code 0; 2 passed; 0 failed)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed; 0 failed)
  - cargo test --locked --lib (exit code 0; 4 passed; 0 failed)
  - cargo test --locked --test foundation (exit code 0; 26 passed; 0 failed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 4 passed, 1 ignored; 0 failed)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored; 0 failed)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed; 0 failed)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/p1-nonfile-recovery.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/agent-m4-suite.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/lib-suite.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/foundation-suite.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/recovery-cycles-suite.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/release-m6-suite.log
  - docs/evidence/v2/p1/20261014-p1-nonfile-reconcile/python-protocol.log
  - Historical M6 evidence preserved in docs/evidence/m6/ untouched.
Implemented but unverified: None in this slice (all implemented code paths verified by comprehensive fault regressions).
Failure/recovery scenarios still open:
  - Storage exhaustion & ENOSPC rollback across WAL / temporary staging (P1.3–P1.6).
  - Candidate / scientific execution on host remains strictly prohibited; dockerized container tests require active Docker engine.
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 9; schema_m6_recovery.sql tool_outcomes table utilized).
  - Zero duplicate child tasks, zero duplicate datasets, zero duplicate finalization jobs, and zero double billing across all crash scenarios.
Next small actionable slice:
  - P1.3 disk allocation budgets, reservation accounting, and ENOSPC recovery.
```

### Implementation Session 2026-10-14 (P1.3 Disk Budgets, Reservation Accounting, & ENOSPC Recovery)

```text
Date / phase / task IDs: 2026-10-14 / P1 / P1.3 (disk allocation budgets, reservation accounting, and ENOSPC recovery)
User-authorized scope: Complete P1.3 disk allocation budgets, reservation accounting, and ENOSPC recovery.
Starting commit and pre-existing dirty files: abf1f235030a79f9a606b429e34608431e37603d; dirty: README.md, docs/v2/, docs/evidence/v2/, scripts/build_argon_review_packet.py
Changes and source paths:
  - crates/harness/src/execution.rs:
      * Constants added: EMERGENCY_RECOVERY_HEADROOM_BYTES (4 MiB), SQLITE_WAL_RESERVE_BYTES (2 MiB), and category default budgets (32 MiB artifact, 32 MiB environment, 32 MiB export, 64 MiB workflow, 8 MiB checkpoint, 8 MiB provider receipt).
      * StorageBudgetConfig implemented with serde defaults, loading from storage_budgets.json or MLRESCURIX_* env overrides.
      * StorageWriteKind enum: JobAdmission, Checkpoint, ProviderReceipt, Log, ManagedBlob, RuntimeAsset, BundleStaging, EmergencyRecovery.
      * enforce_state_write_admission & enforce_state_write_admission_conn: Unifies accounting across current state, active job reservations, required emergency/WAL headroom (6 MiB), and category budgets.
      * ensure_disk_space_with_config: Traverses to existing parent directories to handle statvfs on non-existent paths cleanly; supports MLRESCURIX_SIMULATED_FREE_DISK.
      * collect_artifacts: Protected active workers' .staging directories while safely cleaning up orphaned unreferenced directories.
      * cancel: Upgraded transaction behavior from unchecked_transaction() to transaction_with_behavior(TransactionBehavior::Immediate) to prevent SQLite lock deadlocks during concurrent GC.
      * storage_policy_summary: Expanded to report all policy headroom, category budgets, active reservation breakdown, and observed usage.
  - crates/harness/src/assets.rs:
      * ensure_python_root: Enforces state write admission (StorageWriteKind::RuntimeAsset) before writing embedded Python assets.
  - crates/harness/src/bundle.rs:
      * verify_inference_parity: Accepts state_root, routes staging through enforce_state_write_admission (StorageWriteKind::BundleStaging), and writes within root/staging/ instead of /tmp.
  - crates/harness/src/storage.rs:
      * boundary(): Routes task checkpoints through enforce_state_write_admission_conn (StorageWriteKind::Checkpoint).
  - crates/harness/src/agents/context.rs:
      * record_checkpoint: Routes agent checkpoints through enforce_state_write_admission (StorageWriteKind::Checkpoint) and prunes older checkpoints beyond MAX_RETAINED_CHECKPOINTS_PER_TASK.
  - crates/harness/src/agents/gateway.rs:
      * record_provider_accounting: Routes provider accounting through enforce_state_write_admission (StorageWriteKind::ProviderReceipt).
  - crates/harness/src/agents/orchestrator.rs:
      * Provider turn start (awaiting_response) and response persistence (response_received) route through enforce_state_write_admission.
  - crates/harness/src/agents/workspace.rs:
      * BranchWorkspaceManager::write_file: Routes file writes through enforce_state_write_admission (StorageWriteKind::ManagedBlob).
  - crates/harness/tests/recovery_cycles.rs:
      * Added p1_storage_exhaustion_enospc_and_reservation_fault_regressions covering non-silent saturation across write paths, emergency recovery headroom, emergency hard ceiling, configurable category budgets via storage_budgets.json, scheduler POSIX ENOSPC exit status 28 recovery with staging purge and reservation release, saturation cancellation, and 5 restart/cleanup cycles with PRAGMA quick_check.
      * Guarded review-storage-cycles.json write to preserve historical M6 evidence untouched during tests.
  - docs/v2/DECISIONS.md:
      * Recorded D22 (unified reservation accounting, emergency recovery headroom, configurable budgets, and ENOSPC recovery).
Decisions added or revised: D22 added.
Exact verification commands and exit results:
  - cargo test --locked --test recovery_cycles (exit code 0; 5 passed, 1 ignored; 0 failed)
  - cargo test --locked --test recovery_cycles p1_storage_exhaustion_enospc_and_reservation_fault_regressions -- --nocapture (exit code 0; 1 passed; 0 failed)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed; 0 failed)
  - cargo test --locked --test foundation (exit code 0; 26 passed; 0 failed)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored; 0 failed)
  - cargo test --locked --test sandbox_docker (exit code 0; 5 passed; 0 failed)
  - cargo test --locked --test search_m3 (exit code 0; 9 passed, 1 ignored; 0 failed)
  - cargo test --locked --lib (exit code 0; 4 passed; 0 failed)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed; 0 failed)
  - cargo test --locked (exit code 0; all test binaries passed; 72 passed, 0 failed, 6 ignored)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/recovery-cycles-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/p1-storage-fault-regressions.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/agent-m4-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/foundation-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/release-m6-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/sandbox-docker-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/search-m3-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/lib-suite.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/python-protocol.log
  - docs/evidence/v2/p1/20261014-p1-storage-accounting/full-locked-test.log
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified: None in this slice (all implemented code paths verified by comprehensive fault regressions and full locked test suite).
Failure/recovery scenarios still open:
  - Archive format integrity, atomic CAS bundles, and state boundary isolation (P1.4).
  - Memory consumption bounds and certification (P1.5).
  - Full end-to-end multi-worker concurrency and storage exhaustion integration (P1.6).
  - Candidate / scientific execution on host remains strictly prohibited; dockerized container tests require active Docker engine.
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 9).
  - 128 MiB policy ceiling enforced strictly across all write operations; 4 MiB emergency headroom + 2 MiB WAL headroom guarantee that cancellation and recovery never fail on disk saturation.
  - Zero silent write exemptions across checkpoints, provider receipts, logs, managed blobs, runtime assets, and bundle staging.
Next small actionable slice:
  - P1.4 lease-aware archival/tombstones, compact retry fences, and interrupted publication reconciliation.
```

### Implementation Session 2026-10-14 (P1.4 Archival, Tombstones, Compact Fences, & Publication Recovery)

```text
Date / phase / task IDs: 2026-10-14 / P1 / P1.4 (lease-aware archival/tombstones, compact retry fences, and interrupted publication reconciliation)
User-authorized scope: Complete P1.4 lease-aware archival/tombstones, compact retry fences, and interrupted publication reconciliation.
Starting commit and pre-existing dirty files: abf1f235030a79f9a606b429e34608431e37603d; dirty: README.md, docs/v2/, docs/evidence/v2/, scripts/build_argon_review_packet.py
Changes and source paths:
  - crates/harness/src/execution.rs:
      * Constants added: MAX_RETAINED_COMMANDS_PER_PROJECT (64), MAX_ACTIVE_COMMAND_RAW_BODIES (24), MAX_RETAINED_ACTION_FENCES_PER_PROJECT (64), MAX_RETAINED_TOMBSTONES_PER_PROJECT (128).
      * record_tombstone_conn & record_tombstone: records (project, entity_kind, entity_id, parent_id, identity_hash, reclaimed_bytes, details_json, tombstoned_ms).
      * archive_artifacts: CLI and runtime artifact archival reporting reclaimed bytes, unpinned artifact deletions, terminal job deletions, and tombstone counts.
      * storage_policy_summary: reports tombstones_rows, tombstone_reclaimed_bytes, action_fences_rows, and tombstone/fence limits.
      * collect_artifacts:
          - Sweeps artifacts/<project>/.<job>.staging for inactive jobs.
          - Sweeps leaked staging/bundle_verification_* and workflows/<project>/.bundle_stage.
          - Records artifact SHA-256 tombstones before deleting unpinned managed artifacts.
          - Prunes terminal jobs (succeeded/failed/canceled) while protecting active environment leases, active tasks, running jobs, scientific jobs, pipeline receipts, and pinned artifacts (pinned=1); records job tombstones with SHA-256.
          - Compacts older commands beyond MAX_ACTIVE_COMMAND_RAW_BODIES (24) to hash:<sha256>, preserving exact payload idempotency matching while slashing body storage footprint by >99.8%.
          - Compacts completed actions beyond MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT (64) in-place in actions table and records durable action_fences and action tombstones.
          - Enforces bounded retention for action_fences and tombstones.
      * Supervisor::publish made public and enhanced to adopt existing valid final_dir outputs after rename crash.
      * Supervisor::admit: checks both actions and action_fences to reject job.id action identity reuse.
  - crates/harness/src/storage.rs:
      * connect(): creates tombstones and action_fences tables and indexes idempotently.
      * command(): matches stored request == encoded OR request == format!("hash:{}", hash(&encoded)).
      * action(): falls back to action_fences when action row is not found in actions.
      * Operation::Prepare: matches payload on exact input or input_hash.
      * restore(): merges action_fences for recovered tasks.
  - crates/harness/src/agents/recovery.rs:
      * reconcile_delivery_publication: handles pre-admission crash (emits effect_outcome: "uncertain" and resets unreserved 'finalizing' state to 'created').
      * Deterministically rejects missing bundle manifests as incomplete_bundle_publication.
      * Resets 'finalizing' state to 'created' when finalization jobs fail or are interrupted.
  - crates/harness/src/main.rs:
      * Added ArchiveArtifacts CLI command to Command enum and dispatch handler.
  - crates/harness/tests/recovery_cycles.rs:
      * Added p1_archival_tombstone_fencing_and_publication_recovery_fault_regressions covering 7 comprehensive fault modes across archival, tombstones, compaction, retry fences, and publication recovery.
  - docs/v2/DECISIONS.md:
      * Recorded D23 (lease-aware archival, tombstones, compact retry fences, and interrupted publication reconciliation) and P1.4 resolution details.
Decisions added or revised: D23 added.
Exact verification commands and exit results:
  - cargo test --locked --test recovery_cycles p1_archival_tombstone_fencing_and_publication_recovery_fault_regressions -- --nocapture (exit code 0; 1 passed; 0 failed)
  - cargo test --locked --test recovery_cycles (exit code 0; 6 passed, 1 ignored; 0 failed)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed; 0 failed)
  - cargo test --locked --test foundation (exit code 0; 26 passed; 0 failed)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored; 0 failed)
  - cargo test --locked --test sandbox_docker (exit code 0; 5 passed; 0 failed)
  - cargo test --locked --test search_m3 (exit code 0; 9 passed, 1 ignored; 0 failed)
  - cargo test --locked --lib (exit code 0; 4 passed; 0 failed)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed; 0 failed)
  - cargo test --locked (exit code 0; all test binaries passed; 78 passed, 0 failed, 7 ignored)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/full-locked-test.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/p1-archival-fault-regressions.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/recovery-cycles-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/agent-m4-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/foundation-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/release-m6-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/sandbox-docker-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/search-m3-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/lib-suite.log
  - docs/evidence/v2/p1/20261014-p1-archival-fencing/python-protocol.log
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified: None in this slice (all implemented code paths verified by comprehensive fault regressions and full locked test suite).
Failure/recovery scenarios still open:
  - Supported per-job cgroup enforcement/measurement receipts, feature detection, and Docker daemon survivor/orphan discovery and cleanup (P1.5).
  - Candidate / scientific execution on host remains strictly prohibited; dockerized container tests require active Docker engine.
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 9).
  - Indefinite SQLite metadata growth bounded without deleting idempotency retry fences.
  - Older command request bodies compacted to hash:<sha256> (reducing storage footprint from up to 64 KiB to 69 bytes while preserving exact payload idempotency matching).
  - Settled actions compacted in-place with dual action_fences; all active leases and pinned artifacts protected.
Next small actionable slice:
  - P1.5 supported per-job cgroup enforcement/measurement receipts, feature detection, and Docker daemon survivor/orphan discovery and cleanup.
```

### P1.5–P1.6 closeout (2026-10-14)

The preceding chronological session templates are retained as historical handoffs;
the authoritative current status is the P1 complete summary at the start of this
file. This closeout supersedes earlier “next slice” recommendations.

- Implementation: `crates/harness/src/execution_resources.rs`,
  `crates/harness/src/resource_monitor.py`, `crates/harness/src/execution.rs`,
  and the Rust fault/integration tests.
- Verification: `cargo test --locked` passed (exit 0); live pinned-Docker cgroup
  receipt plus supervisor restart/survivor test passed (2/2); live cgroup OOM
  enforcement/event receipt passed; pinned Docker scientific pipeline passed all
  six modes; disposable-volume real ENOSPC restart recovery passed (1/1);
  `git diff --check` passed.
- Evidence: `docs/evidence/v2/p1/20261014-p1-review-resources/` contains the
  locked suite, live resource tests, receipts, six-mode pipeline, real ENOSPC log,
  sanitized daemon feature snapshot and pinned image reference. Image digest:
  `sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`.
- Compatibility: existing database schema retained; regression tests assert
  `PRAGMA user_version = 9`. `docs/evidence/m6/` hashes match the pre-work snapshot.
- Storage caveat: the 128 MiB managed-state policy and normal recovery reserve are
  enforced, but unrelated external writers can consume shared filesystem space.
  Recovery after ENOSPC is verified once capacity is available; no guarantee is
  made that SQLite can commit while the filesystem remains full.
- Release: P1 complete does not close M6. Gate 3 installed delivery and gate 4
  prospective matched benchmarks remain open; gate 5 requires broader supported
  platform/installed-delivery qualification.
- Next: P2.1 artifact registry/version contract design and read-only inventory complete; obtain/confirm authorization before P2.2 runtime changes.

### Implementation Session 2026-10-14 (P2.1 Artifact Registry & Version Contract Design)

```text
Date / phase / task IDs: 2026-10-14 / P2 / P2.1 (artifact registry and versioned artifact contract design)
User-authorized scope: P2.1 artifact registry and versioned contract design, beginning with a read-only inventory and a concrete proposal; no runtime edits.
Starting commit and pre-existing dirty files: ed5812f275e5264b3ef808e01bfbb90847b2c019; clean working tree.
Changes and source paths:
  - docs/v2/P2_1_ARTIFACT_DESIGN.md: Comprehensive reviewable architecture proposal covering inventory of all M0–M6 artifact tables, filesystem paths, and APIs; gap and compatibility risk analysis; target schema (user_version = 10 DDL: blobs, artifacts, artifact_versions, artifact_lineage, artifact_annotations, project_selections, durable_events); CAS directory layout (<state>/cas/<hash[0..2]>/<hash>) and pre-reservation atomic write protocol; multi-file directory manifest contract; two-tier summary bounds for Context Firewall (index_digest <= 512 B, summary_text <= 4096 B); non-destructive additive migration and dual-write compatibility strategy; lease-aware GC; test matrix; open decisions and authorization checklist.
  - docs/v2/DECISIONS.md: Recorded D26 and P2.1 resolution details.
  - docs/v2/PROGRESS.md: Updated current state, phase map, session ledger, and session entry.
Decisions added or revised: D26 added (Unified Artifact Registry, Content-Addressable Storage (CAS), Two-Tier Summary Bounds, and Versioned Contracts).
Exact verification commands and exit results:
  - Documentation link and integrity validation (exit code 0; all links valid, zero broken references).
  - git status (clean tracked modifications in docs/v2/ only; zero runtime code or schema edits).
Evidence paths plus source/image/schema hashes:
  - docs/v2/P2_1_ARTIFACT_DESIGN.md
  - PRAGMA user_version remains strictly at 9.
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified: None (documentation and architecture design only; zero runtime code edits in this slice).
Failure/recovery scenarios still open:
  - Runtime implementation of CAS module and storage reservation hooks (P2.2).
  - Runtime additive schema migration to user_version = 10 and backfill (P2.2).
  - Evaluator-only scorecard authority and two-tier summary bounds enforcement (P2.3).
  - Durable event journaling and SSE non-blocking reader pool (P2.4–P2.5).
Budget / compatibility / migration implications:
  - Zero database migration version bump performed in this slice (user_version remains 9).
  - 128 MiB policy ceiling preserved; CAS writes will strictly require pre-admission byte reservations.
  - Existing M0–M6 tables and APIs completely preserved with dual-write compatibility planned for P2.2.
Next small actionable slice:
  - P2.2 runtime CAS module (`crates/harness/src/cas.rs`) and additive schema migration to `user_version = 10` (`crates/harness/src/schema_m7_artifacts.sql`) upon explicit user authorization.
```

### Implementation Session 2026-10-14 (P2.2 Runtime CAS & Schema Migration Slice)

```text
Date / phase / task IDs: 2026-10-14 / P2 / P2.2 (runtime CAS, schema migration to user_version = 10, artifact registry, and dual-write)
User-authorized scope: User explicitly authorized proceeding with P2.2 runtime implementation ("ok proceed").
Starting commit and pre-existing dirty files: ed5812f275e5264b3ef808e01bfbb90847b2c019; dirty: docs/v2/{ARGON_REVIEW_PACKET.txt, DECISIONS.md, PROGRESS.md, README.md, ROADMAP.md, P2_1_ARTIFACT_DESIGN.md}.
Changes and source paths:
  - crates/harness/src/schema_m7_artifacts.sql: Additive DDL migration from user_version 9 to 10 creating `blobs`, `artifacts`, `artifact_versions`, `artifact_lineage`, `artifact_annotations`, `project_selections`, `durable_events` with non-destructive backfill from legacy `managed_artifacts` and `scientific_datasets`. Idempotent `IF NOT EXISTS` DDL. Decoupled `source_job_id` from strict foreign key blocking to ensure compatibility with sliding-window terminal job retention GC while preserving provenance strings and index.
  - crates/harness/src/cas.rs: ContentAddressableStorage implementing `<state>/cas/<hash[0..2]>/<hash>` storage layout, pre-admission reservation checks under 128 MiB policy ceiling, temporary staging in `cas/.staging/<nonce>.tmp`, fsync and read-only (`0o444`) permissions before atomic rename, deduplication, and canonical directory manifest hashing.
  - crates/harness/src/artifacts.rs: ArtifactRegistry implementing logical entity registration, immutable version registration, two-tier summary bounds checks (`index_digest` <= 512 B, `summary_text` <= 4096 B), evaluator-only scorecard authority check (`producer_authority == "evaluator:trusted"`), lineage edges, annotations, Apex selection, and durable event journal querying.
  - crates/harness/src/lib.rs: Exposed `cas` and `artifacts` modules.
  - crates/harness/src/storage.rs: Expanded supported migration range from 1..=9 to 1..=10.
  - crates/harness/src/execution.rs: Updated migration loop to apply `schema_m7_artifacts.sql` for case 9, updated `connect()` visibility to `pub(crate)`, integrated dual-writing in `Supervisor::finish` and `pin_artifact`.
  - crates/harness/src/execution_resources.rs: Updated `user_version` assertion from 9 to 10.
  - crates/harness/tests/release_m6.rs: Updated `user_version` assertions from 9 to 10.
  - crates/harness/tests/recovery_cycles.rs: Updated `user_version` assertions from 9 to 10.
  - crates/harness/tests/artifact_registry_p2.rs: Comprehensive 5-test suite covering v9 -> v10 migration with populated legacy state, CAS blob deduplication and permissions, canonical directory manifests and symlink rejection, two-tier summary bounds and scorecard authority enforcement, and lineage, annotations, and Apex selection.
Decisions added or revised: D26 updated with P2.2 runtime implementation and verification details.
Exact verification commands and exit results:
  - cargo test --locked --test artifact_registry_p2 (exit code 0; 5 passed)
  - cargo test --locked --lib (exit code 0; 14 passed, 2 ignored)
  - cargo test --locked --test foundation -- --test-threads=4 (exit code 0; 26 passed)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 8 passed, 2 ignored)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed)
  - cargo test --locked --test search_m3 (exit code 0; 9 passed, 1 ignored)
  - Total passed tests across all 7 suites: 90 passed, 0 failed, 8 ignored.
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/artifact-registry-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/lib-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/foundation-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/release-m6-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/recovery-cycles-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/agent-m4-suite.log
  - docs/evidence/v2/p2/20261014-p2-artifact-registry/search-m3-suite.log
  - PRAGMA user_version is now at 10.
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified:
  - None; all runtime code in P2.2 verified through automated test suites.
Failure/recovery scenarios still open:
  - Bounded read connection pool sizing for SSE streaming without writer starvation (P2.5).
  - Outbound data sanitization / approved view policy for artifact summaries entering agent context (P2.4).
Budget / compatibility / migration implications:
  - `PRAGMA user_version = 10` active and supported.
  - Non-destructive additive DDL preserves all legacy M0–M6 tables.
  - 128 MiB policy ceiling strictly preserved via CAS pre-admission reservation checks.
  - Dual-write adapters in `Supervisor::finish` and `pin_artifact` ensure existing readers continue functioning without disruption.
Next small actionable slice:
  - P2.3–P2.5 artifact inspection queries, durable event journal subscription APIs, and SSE non-blocking reader connection pool.
```

### Implementation Session 2026-10-14 (P2 Roadmap Gate Closure Acceptance Slice)

```text
Date / phase / task IDs: 2026-10-14 / P2 / P2 Roadmap Gate Closure (populated-state upgrade preservation, partial CAS/DB reconciliation, protected references under concurrent GC, event restart/reconnect consistency)
User-authorized scope: Audit and close remaining P2 acceptance checks listed in ROADMAP.md and PROGRESS.md; verify populated-state upgrade preservation, partial CAS/DB commit reconciliation, protected references under concurrent GC, and event restart/reconnect consistency.
Starting commit and pre-existing dirty files: 9cb2e3e; dirty: crates/harness/src/execution.rs, crates/harness/tests/artifact_registry_p2.rs, docs/v2/DECISIONS.md, docs/v2/PROGRESS.md, docs/v2/README.md, docs/v2/ROADMAP.md.
Changes and source paths:
  - crates/harness/src/execution.rs: Integrated CAS staging sweep (`cas.sweep_staging(Duration::from_secs(0))`) into routine artifact collection (`collect_artifacts`), reclaiming orphaned temporary staging files from interrupted or partial write attempts.
  - crates/harness/tests/artifact_registry_p2.rs: Authored 4 comprehensive acceptance regressions closing all open P2 roadmap checks (suite now 14 tests, all passing):
      * p2_test_populated_state_upgrade_preserves_selection_and_final_test_records: Verifies migration from v9 to v10 on populated database preserves all historical `final_test_gates` (both reserved and consumed), `search_sessions` (apex pointers and JSON selection records), `trial_results`, `pipeline_receipts`, `tombstones`, and `action_fences` with 0 foreign-key violations and repeated migration idempotency.
      * p2_test_partial_cas_and_db_commit_reconciliation: Verifies leaked temporary staging files in `cas/.staging/` are swept by `collect_artifacts`, pre-existing CAS files on disk before DB commit crashes are safely adopted/deduplicated on retry without double byte allocation, corrupted files on disk are detected via SHA-256 validation and repaired atomically, and pre-admission byte reservations remain accurate.
      * p2_test_protected_references_survive_concurrent_gc: Verifies pinned artifacts (`artifacts.pinned = 1`), Apex candidate and scorecard versions (`project_selections`), and lineage edges (`artifact_lineage`) survive concurrent `collect_artifacts` execution while unpinned terminal jobs are safely pruned and active environment leases are protected.
      * p2_test_event_restart_and_reconnect_consistency: Verifies subscribing to events, simulating process restart across `Runtime::open` / `ArtifactRegistry::new`, mutating state while disconnected (`publish_successor_command`, `annotate`, `tombstone_artifact_command`, `set_apex_selection`), and reconnecting with saved cursor delivers all committed events in monotonic cursor order without duplicate delivery, and expired cursors (>10,000 window) trigger `reset_required = true` with clean snapshot resynchronization.
  - docs/v2/DECISIONS.md: Recorded D31 and updated P2 resolution status to Phase P2 Complete.
  - docs/v2/README.md & docs/v2/ROADMAP.md: Updated P2 phase status to Complete with evidence cross-references.
Decisions added or revised: D31 added (CAS Staging Sweeps in GC and Multi-Commit Reconciliation).
Exact verification commands and exit results:
  - cargo test --locked --test artifact_registry_p2 (exit code 0; 14 passed)
  - cargo test --locked --lib (exit code 0; 14 passed, 2 ignored)
  - cargo test --locked --test foundation -- --test-threads=1 (exit code 0; 26 passed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 8 passed, 2 ignored)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed)
  - cargo test --locked --test search_m3 (exit code 0; 9 passed, 1 ignored)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed)
  - git diff --check (exit code 0; clean)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p2/20261014-p2-gate-closure/artifact_registry_p2.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/lib.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/foundation.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/recovery_cycles.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/release_m6.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/agent_m4.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/search_m3.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/python_protocol.log
  - docs/evidence/v2/p2/20261014-p2-gate-closure/source-sha256.txt
  - docs/evidence/v2/p2/20261014-p2-gate-closure/README.md
  - PRAGMA user_version is 10.
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified:
  - None; all 4 acceptance checks and full test suites are verified.
Failure/recovery scenarios still open:
  - Global multi-project resource lease brokering and fair queueing under heavy concurrent workloads (P3).
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 10).
  - 128 MiB policy ceiling strictly preserved with CAS pre-admission reservation checks.
  - Dual-write adapters in Supervisor::finish and pin_artifact continue functioning without disruption.
Next small actionable slice:
  - P3.1 effective host and container-VM CPU, memory, disk, and accelerator capability discovery, resource request presets, and infeasible-request rejection before admission.
```

### Implementation Session 2026-10-14 (P3.1 Capability Discovery, Headroom, Presets, & Early Infeasible Rejection)

```text
Date / phase / task IDs: 2026-10-14 / P3 / P3.1 (effective CPU, memory, disk, and accelerator capability discovery, resource request presets, headroom accounting, and infeasible-request rejection)
User-authorized scope: User explicitly authorized: "If the P2 gate is fully evidenced, implement P3.1 effective CPU, memory, disk and accelerator capability discovery, resource request presets, and infeasible-request rejection. Do not begin P3.2 or later phases."
Starting commit and pre-existing dirty files: 9cb2e3e; dirty: crates/harness/src/execution.rs, crates/harness/tests/artifact_registry_p2.rs, docs/v2/DECISIONS.md, docs/v2/PROGRESS.md, docs/v2/README.md, docs/v2/ROADMAP.md.
Changes and source paths:
  - crates/harness/src/broker.rs: Implemented resource lease broker capability discovery and admission feasibility module:
      * Host capability discovery: logical CPU cores (`available_parallelism`), total host RAM (`sysctl hw.memsize` on macOS, `sysconf` on Linux), and filesystem storage via `libc::statvfs`.
      * Container-VM capability discovery: container-VM limits via `docker info` (`NCPU`, `MemTotal`, `CgroupVersion`, and cgroup v2 feature flags).
      * Effective capability synthesis: clamps container execution to container-VM limits (`min(host, vm)`), preventing overestimating available resources when running inside bounded container VMs (such as macOS Docker Desktop).
      * Passive accelerator discovery: detects Apple Silicon Metal GPU on macOS arm64 (`hw.optional.arm64` and `machdep.cpu.brand_string`) and NVIDIA CUDA on Linux via `nvidia-smi` without executing untrusted code on the host.
      * Configured headroom: default 1000 millicores CPU, 512 MiB RAM, 64 MiB disk reserved for host OS and harness control plane. Available capacity = `Effective - Headroom`. Supports `<root>/resource_headroom.json` and `MLRESCURIX_HEADROOM_*` environment overrides.
      * Resource presets: `MICRO` (1 thread, 512 MiB RAM, 1 MiB output), `STANDARD` (2 threads, 2 GiB RAM, 2 MiB output), `HEAVY` (4 threads, 8 GiB RAM, 4 MiB output), and `CUSTOM`. Enforces bounds (1..=256 threads, 64 MiB..=64 GiB RAM, output within 4 MiB M1 ceiling). Resolves onto `JobSpec` and supports `<root>/resource_presets.json` overrides.
      * Early infeasible-request rejection: rejects excessive threads, excessive memory, excessive output storage, unavailable accelerators, and unsupported memory certification before queue admission in `execution::submit()` with zero SQLite job insertion. Rechecked in `Supervisor::admit()`.
  - crates/harness/src/execution_resources.rs: Extended `ResourceRequirements` with `preset: Option<ResourcePreset>` and `accelerator: Option<AcceleratorType>` with serde defaults; derived `Eq, PartialEq` on `ResourceFeatures`; exposed `docker_info() -> Result<Value>`.
  - crates/harness/src/execution.rs: Exported `OUTPUT_CAP: u64 = 4_194_304` as public; integrated `resolve_job_spec_preset` and `check_request_feasibility` into `submit()` before database queue insertion; integrated feasibility check into `Supervisor::admit()`.
  - crates/harness/src/lib.rs: Exported `pub mod broker;`.
  - crates/harness/tests/resource_broker_p3.rs: Comprehensive 5-test suite covering live and synthetic host/container capability discovery, headroom accounting, preset bounds and resolution, early infeasible-request rejection (with 0-row SQLite job assertion), and container-VM memory cap enforcement.
  - docs/v2/DECISIONS.md: Recorded D32 and updated P3 resolution status for P3.1.
  - docs/v2/README.md & docs/v2/ROADMAP.md: Recorded P3.1 verified status; P3 phase gate remains open for P3.2–P3.5.
Decisions added or revised: D32 added (Effective Host/Container-VM Resource Capability Discovery, Presets, Headroom, and Early Infeasible Rejection).
Exact verification commands and exit results:
  - cargo test --locked --test resource_broker_p3 (exit code 0; 5 passed)
  - cargo test --locked --lib (exit code 0; 14 passed, 2 ignored)
  - cargo test --locked --test artifact_registry_p2 (exit code 0; 14 passed)
  - cargo test --locked --test foundation -- --test-threads=1 (exit code 0; 26 passed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 8 passed, 2 ignored)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 3 ignored)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed)
  - cargo test --locked --test search_m3 (exit code 0; 9 passed, 1 ignored)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed)
  - git diff --check (exit code 0; clean)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/resource_broker_p3.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/lib.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/artifact_registry_p2.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/foundation.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/recovery_cycles.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/release_m6.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/agent_m4.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/search_m3.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/python_protocol.log
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/source-sha256.txt
  - docs/evidence/v2/p3/20261014-p31-capability-discovery/README.md
  - PRAGMA user_version is 10.
  - Historical M6 evidence in docs/evidence/m6/ preserved untouched.
Implemented but unverified:
  - None; all capability discovery, headroom, preset, and rejection paths verified by resource_broker_p3 and full regression suites.
Failure/recovery scenarios still open:
  - P3.2 global/project capacity reservations, weighted fair queues with aging, bounded queues, and stable tie ordering.
  - P3.3 broker routing for generic tools, preflight, evaluation, benchmarking, final fit, and thread caps.
  - P3.4 pressure backoff, per-project pause/stop isolation, and resource release on reap.
Budget / compatibility / migration implications:
  - Zero database migration version bump required (user_version remains 10).
  - 128 MiB policy ceiling strictly preserved.
  - Output cap strictly bounded by 4 MiB M1 ceiling across presets.
  - Backward compatibility preserved with existing JobSpec JSON serialization via serde defaults and skip_serializing_if.
Next small actionable slice:
  - P3.2 atomic global/project capacity reservations, weighted fair queues with aging, bounded queues, and stable tie ordering upon explicit user authorization.
```

### Implementation Session 2026-10-14 (P3.2–P3.5 Slice & Phase P3 Acceptance Gate Closure)

```text
Date / phase / task IDs: 2026-10-14 / P3 / P3.2–P3.5, Phase P3 Acceptance Gate
User-authorized scope: Implement the remainder of Phase P3 (P3.2–P3.5) and close the Phase P3 acceptance gate as defined in docs/v2/ROADMAP.md, ARCHITECTURE.md, DECISIONS.md, and PROGRESS.md. Do not begin P4 or later phases.
Starting commit and pre-existing dirty files: abf1f235030a79f9a606b429e34608431e37603d; dirty: docs/v2/{PROGRESS.md,ROADMAP.md,DECISIONS.md}
Changes and source paths:
  - crates/harness/src/schema_p3_leases.sql: DDL for `resource_leases` and `resource_lease_transitions` with indexes, executed idempotently in execution::connect().
  - crates/harness/src/broker.rs:
    * Implemented LeaseStatus enum, ResourceLease struct, and durable transition operations: reserve_lease, transition_lease, heartbeat_lease, release_lease, get_lease, get_lease_transitions.
    * Implemented check_capacity_admission distinguishing workers concurrency from cpu_threads and memory_bytes across device and project policies.
    * Implemented is_under_pressure checking MLRESCURIX_PRESSURE_BACKOFF and disk headroom.
    * Implemented apply_thread_caps_to_command setting thread caps across all math/threading runtimes.
    * Implemented record_allocation_snapshot returning active jobs, capacity allocations, and per-job queue wait distributions.
    * Implemented finalization/control-plane headroom reservation (1 worker, 1 CPU thread, 512 MiB RAM) when worker_ceiling > 2 or active finalization work is queued/present.
  - crates/harness/src/execution.rs:
    * Transitioned leases to recovering on supervisor startup and released leases on interrupted recovery jobs.
    * Routed admission through check_capacity_admission with pressure backoff deferrals, project-level capacity skip tracking, thread-cap environment variable injection, and atomic lease reservation transitioned to Active.
    * Added supervisor heartbeat ticks and transition to Reaped on exit; released leases via release_lease on terminal finish.
    * Injected thread-cap environment variables into scientific worker containers.
  - crates/harness/src/execution_resources.rs:
    * Updated container discovery to validate ownership against active supervisor epochs and skip foreign/inactive supervisor containers.
  - crates/harness/src/agents/sandbox.rs:
    * Injected thread cap environment variables derived from limits.max_cpus into sandbox execution environments.
  - python/src/mlrescurix_worker/scientific/workflow.py:
    * Added _leased_threads() reading MLRESCURIX_NUM_THREADS.
    * Bound estimator n_jobs to _leased_threads() across RandomForestRegressor, LogisticRegression, and RandomForestClassifier; forwarded thread caps to isolated fit and predict child processes.
  - crates/harness/tests/resource_broker_p3.rs:
    * Authored 13 comprehensive integration tests covering P3.1–P3.5 and Phase P3 Acceptance Gate:
      1. p3_test_host_and_container_vm_capability_discovery
      2. p3_test_resource_headroom_accounting
      3. p3_test_resource_presets_bounds_and_resolution
      4. p3_test_infeasible_request_early_rejection_before_admission
      5. p3_test_container_vm_memory_cap_prevents_host_overestimate
      6. p3_test_atomic_capacity_reservations_and_dimension_distinction
      7. p3_test_lease_lifecycle_heartbeats_and_recovery_transitions
      8. p3_test_weighted_fair_queueing_aging_and_stable_tie_ordering
      9. p3_test_thread_caps_and_broker_routing
      10. p3_test_pressure_backoff_and_finalization_headroom_preservation
      11. p3_test_per_project_cancellation_and_pause_isolation
      12. p3_test_multi_project_stress_and_resilience_10x5 (50 jobs: 10 projects x 5 intents, mixed presets, starvation bounds, worker crash, cancel/restart, allocation timelines)
      13. p3_acceptance_gate (capacity bound verification, aging service verification, project cancellation isolation, and zero runner lease bypass verification).
Decisions added or revised: D33 added (Resource Lease Lifecycle, Capacity Dimension Distinction, Weighted Fair Queueing with Aging, and Finalization Headroom Preservation).
Exact verification commands and exit results:
  - cargo test --locked --test resource_broker_p3 -- --test-threads=1 (exit code 0; 13 passed, 0 failed)
  - cargo test --locked --test artifact_registry_p2 (exit code 0; 14 passed, 0 failed)
  - cargo test --locked --test foundation -- --test-threads=1 (exit code 0; 26 passed, 0 failed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 8 passed, 0 failed, 2 ignored)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed, 0 failed)
  - cargo test --locked --test release_m6 -- --test-threads=1 (exit code 0; 5 passed, 0 failed, 3 ignored)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed, 0 failed)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_resource_broker_p3.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_artifact_registry_p2.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_foundation.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_recovery_cycles.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_agent_m4.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_release_m6.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/test_python_protocol.log
  - docs/evidence/v2/p3/20261014-p3-gate-closure/summary.md
Implemented but unverified:
  - None within Phase P3 scope (all P3 deliverables implemented and verified with passing tests).
Failure/recovery scenarios still open:
  - Host RAM exhaustion is intentionally avoided per acceptance criteria; small/synthetic workloads used.
  - Phase P4+ agent-first tools and executable candidates remain planned and not begun.
Budget / compatibility / migration implications:
  - Zero schema version bump required (user_version remains 10).
  - 128 MiB policy ceiling strictly preserved.
  - M0–M6 compatibility fully preserved across all regression suites.
  - Historical evidence in docs/evidence/m6/ left untouched.
Next small actionable slice:
  - Phase P4 (P4.1 workspace_fs/sandbox_exec adapters) upon explicit user authorization.
```

```text
Date / phase / task IDs: 2026-10-14 / Phase P4 / P4.1–P4.3 and Phase P4 Acceptance Gate
User-authorized scope: P4.1–P4.3 and Phase P4 Acceptance Gate as defined in docs/v2/ROADMAP.md, ARCHITECTURE.md, DECISIONS.md, and PROGRESS.md. Do not begin Phase P5 or later phases.
Starting commit and pre-existing dirty files: clean commit on origin/codex/p1-cgroup-orphan-recovery
Changes and source paths:
  - python/src/mlrescurix_worker/scientific/candidate_runner.py:
    * Authored validate_candidate_manifest(), validate_predictions(), compute_source_hash(), evaluate_candidate_cv(), and run_candidate_cli().
    * Enforces candidate manifest contract: task_type ('binary', 'multiclass', 'regression'), entrypoint, supported_operations ('fit', 'predict'), stable_row_keys ('row_id'), class_order, and source_hash verification against candidate directory.
    * Enforces candidate prediction validation: probabilities in [0, 1], row sums equal 1.0 +- 1e-3, column names match 'prob_<class>', row count and row_id alignment, finite values.
    * Enforces label isolation: validation feature dataframes strictly omit 'target' column during candidate prediction.
    * Enforces diagnostic slices minimum support: max(20, 0.05 * N); slices below support threshold are automatically suppressed with status 'suppressed'.
  - python/src/mlrescurix_worker/scientific/cli.py:
    * Added candidate-evaluate subcommand accepting JSON evaluation spec and emitting structured atomic JSON results.
  - python/tests/test_candidate_runner.py:
    * Authored 5 unit tests covering manifest validation, prediction checks, OOF CV evaluation with label isolation, and diagnostic slice suppression.
  - contracts/v1/experiment_spec.schema.json:
    * Decoupled recipe from 3-enum to generic string (maxLength: 128).
    * Expanded node_type enum to include 'baseline', 'screening', 'full_fidelity', 'repeat', 'draft', 'debug', 'improve', 'ablate', 'candidate'.
  - crates/harness/src/search.rs & python/src/mlrescurix_worker/search/adapter.py:
    * Allowed arbitrary candidate/recipe parameters without fixed 3-enum restriction.
  - crates/harness/src/schema_p4_candidates.sql:
    * Created candidate_evaluations and validation_query_counters tables.
  - crates/harness/src/execution.rs:
    * Idempotently re-applies schema_p4_candidates.sql on user_version = 10.
    * Migrates experiment_dag check constraint to allow arbitrary recipes with length <= 128 while retaining 'hist_boosting' in DDL for backwards compatibility with M6 migration tests.
  - crates/harness/src/assets.rs:
    * Embedded candidate_runner.py in EMBEDDED_PYTHON_FILES.
  - crates/harness/src/artifacts.rs:
    * Added register_blob() method to ArtifactRegistry to ensure CAS blobs are indexed in SQLite blobs table.
    * Auto-registered CAS blobs in publish_version_inner().
  - crates/harness/src/agents/workspace.rs:
    * Added write_file_with_expected_hash(), patch_file_with_expected_hash(), glob_files(), and search_files().
  - crates/harness/src/agents/sandbox.rs:
    * Hardened execute_argv() with /workspace:ro read-only mount, task-local /scratch:rw and /outputs:rw tmpfs mounts, thread caps, and honest fallback refusal when Docker is unavailable.
  - crates/harness/src/agents/tools.rs & roles.rs:
    * Registered workspace_fs, sandbox_exec, evaluate_submission, artifact_crud, discover_capabilities, dataset_query, experiment_tree tools and RBAC capabilities.
  - crates/harness/src/agents/orchestrator.rs:
    * Implemented tool dispatch for workspace_fs (read, write, patch, list, glob, search with optimistic concurrency hash checks).
    * Implemented sandbox_exec dispatching commands through SandboxRunner.
    * Implemented discover_capabilities reporting available tools, container status, image ref, presets, and query limits.
    * Implemented dataset_query reading schema and row count without leaking validation/test row values or label distributions.
    * Implemented experiment_tree querying experiment_dag nodes.
    * Implemented artifact_crud (publish, get, list, lineage, pin, unpin, tombstone) with strict evaluator authority check rejecting kind == 'scorecard'.
    * Implemented evaluate_submission: checks research admission, exploratory budget reserve, and validation query limits (max 50); validates manifest and source_hash; materializes evaluation spec; invokes candidate-evaluate CLI via embedded Python; registers official scorecard in CAS and artifacts via trusted evaluator API; records in candidate_evaluations.
    * Fixed prompt token estimation using standard char-to-token ratio (div_ceil(4)) to prevent serialized tool schemas from prematurely triggering context limit errors.
  - crates/harness/tests/candidate_tools_p4.rs:
    * Authored 9 integration tests covering P4.1–P4.3 and Phase P4 Acceptance Gate:
      1. p4_1_workspace_fs_operations_and_concurrency_fencing
      2. p4_1_sandbox_exec_isolation_and_fallback
      3. p4_1_discover_capabilities_reports_contract_boundaries
      4. p4_2_p4_3_executable_candidate_protocol_and_evaluation
      5. p4_2_prediction_rejections_and_tamper_fencing
      6. p4_3_split_label_isolation_and_evaluator_authority
      7. p4_3_decoupled_recipes_and_experiment_tree
      8. p4_4_validation_query_budget_and_slice_suppression
      9. p4_acceptance_gate_full_closure
Decisions added or revised: D34 added (Generic Tool Adapters, Executable Candidate Manifest/CLI Protocol, Label Isolation, Evaluator Scorecard Authority, and Validation Query Limits).
Exact verification commands and exit results:
  - cargo test --locked --test candidate_tools_p4 (exit code 0; 9 passed, 0 failed)
  - PYTHONPATH=python/src pytest python/tests/test_candidate_runner.py (exit code 0; 5 passed, 0 failed)
  - PYTHONPATH=python/src python3 -m unittest python/tests/test_protocol.py (exit code 0; 3 passed, 0 failed)
  - cargo test --locked --test artifact_registry_p2 (exit code 0; 14 passed, 0 failed)
  - cargo test --locked --test resource_broker_p3 -- --test-threads=1 (exit code 0; 13 passed, 0 failed)
  - cargo test --locked --test foundation -- --test-threads=1 (exit code 0; 26 passed, 0 failed)
  - cargo test --locked --test agent_m4 (exit code 0; 23 passed, 0 failed)
  - cargo test --locked --test recovery_cycles -- --test-threads=1 (exit code 0; 8 passed, 0 failed, 2 ignored)
  - cargo test --locked --test release_m6 (exit code 0; 5 passed, 0 failed, 3 ignored)
Evidence paths plus source/image/schema hashes:
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_candidate_tools_p4.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_candidate_runner_python.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_python_protocol.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_artifact_registry_p2.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_resource_broker_p3.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_foundation.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_agent_m4.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_recovery_cycles.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/test_release_m6.log
  - docs/evidence/v2/p4/20261014-p4-gate-closure/summary.md
Implemented but unverified:
  - None within Phase P4 scope (all P4 deliverables implemented and verified with passing tests).
Failure/recovery scenarios still open:
  - Phase P5+ orchestration, checkpoints and context compaction remain planned and not begun.
Budget / compatibility / migration implications:
  - Zero schema version bump required (user_version remains 10).
  - 128 MiB policy ceiling strictly preserved.
  - M0–M6 compatibility fully preserved across all regression suites.
  - Historical evidence in docs/evidence/m6/ left untouched.
Next small actionable slice:
  - Phase P5 (P5.1 task assignment/completion schemas) upon explicit user authorization.
```

Copy and fill one entry; update phase/task statuses above too.

```text
Date / phase / task IDs:
User-authorized scope:
Starting commit and pre-existing dirty files:
Changes and source paths:
Decisions added or revised:
Exact verification commands and exit results:
Evidence paths plus source/image/schema hashes:
Implemented but unverified:
Failure/recovery scenarios still open:
Budget / compatibility / migration implications:
Next small actionable slice:
```

Save new evidence under `docs/evidence/v2/<phase>/<session>/` only when produced;
do not create fake evidence placeholders. Keep raw logs bounded and redacted. Before
ending, make this file sufficient for a fresh agent to continue without chat history.

## Planning validation

Only documentation/link/source-copy/diff checks are applicable in P0. Historical
M6 runtime evidence was read but not rerun. Checks performed on 2026-10-14:

- Host `python3` documentation-only script: all local Markdown links in the seven
  authored v2 documents resolved; nine total documents exist including archives.
  Both archive files matched their source attachment bytes exactly.
- Source archive SHA-256: first input
  `71629ee827ccc7e1b6558effa174836ea611f755682ea109e198d0c90578cd34`;
  second input
  `e87cfaf1187a8860ec723181e27c9cf6c21071786a47a87459276ac3847513f5`.
- `git diff --check`: passed for tracked changes. `git status --short`: only README
  and the new `docs/v2/` directory changed. New authored docs also checked for
  trailing whitespace separately before final handoff.
- No Rust, scientific Python, Docker, live-provider or benchmark tests run for
  this documentation-only change. No commits, pushes or deployment performed.

The earlier inventory above describes the initial P0 snapshot. The later review
adds REVIEW_RESPONSE.md and ARGON_PLAN_REVIEW.txt; the packet generator is also an
untracked new script. Current source packet inventory: eight authored Markdown
docs, two original blueprint archives and one returned-review archive (11 source
files), plus the generated review packet. Validate local links, source-archive bytes,
packet contents and whitespace again after each documentation refinement.

Review-refinement checks on 2026-10-14 passed: eight authored documents have valid
local links and no trailing whitespace, broken Google search links or imported
response markers; all three archives match their attachments byte-for-byte; all
11 source files match their generated packet blocks. `git diff --check` passed.
Returned-review SHA-256:
`1195af3fd900f14e132cfb49a00f2823acc49cf8304f89c38fa9809078e5f567`.
No runtime suites were run for this documentation refinement.


| 2026-10-14 | P2.3–P2.5 artifact inspection, durable event subscriptions, and read pool | Added bounded artifact listing/version/lineage queries and pin/unpin; official scorecard creation/publication is restricted to crate-internal evaluator methods and Apex verifies evaluator authority; byte-accurate SQL guards and 64 KiB result cap; durable registry events are emitted by SQLite triggers in the mutation transaction, retained to 10,000 per project with durable expired-cursor checkpoints; snapshot cursor and bounded artifact snapshot share a WAL read transaction (with truncation flag), replay is bounded and signals reset for expired cursors; added a shared four-connection-per-state read-only SQLite pool (`query_only=ON`, zero wait on exhaustion) used by artifact readers. Legacy supervisor publication and pin adapters retain dual-write behavior. Tests: artifact_registry_p2 8 passed; recovery_cycles 8 passed/2 ignored; foundation 26 passed; release_m6 5 passed/3 ignored. Captured logs and source hashes in `docs/evidence/v2/p2/20261014-p2-inspection-journal-readers/`. M6 remains NOT READY. |
