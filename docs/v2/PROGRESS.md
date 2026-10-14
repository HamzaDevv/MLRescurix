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
- Runtime implementation: P1.1–P1.6 complete and verified for the covered regressions and supported Docker runtime.
- Release: M6 remains **NOT READY**, as recorded in `docs/m6-completion-status.md`.
- Authorization: user authorized completion of P1.5 and P1.6, including final commit and push.
- Next recommended slice: P2.1 artifact registry/version contract design and read-only inventory; obtain/confirm authorization before P2 runtime changes.

| Phase | Status | Evidence / next step |
| --- | --- | --- |
| P0 | Complete | This doc set, archived inputs, static audit and documentation validation |
| P1 | complete | P1.1–P1.6 verified; see `20261014-p1-review-resources/` evidence. M6 release remains independently NOT READY. |
| P2 | Planned | Artifact contracts after P1 safety prerequisites |
| P3 | Planned | Extend existing reservations; no unconditional 50-job claim |
| P4 | Planned | Add executable candidate adapter; keep old API working |
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
- Next: P2.1 artifact registry/version contract design and read-only inventory;
  runtime changes need the corresponding user authorization.

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
