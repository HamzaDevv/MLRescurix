# Architecture decision register

Initial proposed defaults, 2026-10-14. These are planning decisions, not claims of
implementation or irrevocable user approvals. Record reversals with evidence.

| ID | Decision | Reason / revisit trigger |
| --- | --- | --- |
| D01 | Repair safety prerequisites before new runtime surfaces | M6 recovery/storage/release gates remain open |
| D02 | Preserve Rust control plane and Docker-only candidate execution initially | Existing isolation requirement; native runtime is a separately certified backend |
| D03 | Keep shared SQLite WAL with FULL durability and converge writes | Cross-project leases and retry fences need durable atomicity; shard only after measured contention and cross-DB recovery design |
| D04 | Immutable blobs/versions, mutable annotations and revisioned selection | CRUD must preserve evidence and distinguish pin/baseline/Apex |
| D05 | Four core coding tool families plus governed controls and optional conveniences | Small generic surface without losing delegation/evaluation/finalization authority |
| D06 | CLI executable contract beside legacy class adapter | Support arbitrary ML/legacy repair without a new universal Python ABC bottleneck |
| D07 | Disposable container per job first; warm pools later | Prove isolation/accounting before optimizing import latency |
| D08 | Capacity-driven dispatch, weighted queues before DRF | 10x5 is a workload/cap scenario, not a minimum hardware guarantee |
| D09 | Keep ~16K economical default with <=64K complete-window cap | Larger windows increase cost; tokenizer must match provider |
| D10 | Replay only known safe operations with full input identities | Resetting files cannot reverse arbitrary effects; zero waste is not guaranteed |
| D11 | Best-first/multi-fidelity default, optional measured MCTS | Search strategy should earn complexity through matched experiments |
| D12 | Deterministic evaluator and independent advisory critic | An LLM does not own official scoring or prove zero leakage |
| D13 | Capability-scoped modality expansion | Local resources and evaluation validity vary by task/backend |
| D14 | Embedded trusted runtime remains; locked optional environments added | Source-independent installs and optional libraries are compatible |
| D15 | Read-only source execution by default; reviewed staging diffs optional | Source writes use workspace_fs; output cleanup alone never proves a job safe to retry |
| D16 | Harness-controlled per-fold execution with exposure provenance | Isolate fold inputs/state; agent access to development labels still limits claims of independent OOF evidence |
| D17 | Pretrained rescue starts as an unverified legacy baseline | Clean refit resets all learned state; untouched disjoint evaluation needs provenance, not just an attestation |
| D18 | Compact index digests plus on-demand summaries and total token caps | <=512 bytes/<=4 KiB are storage bounds, not exact token counts |
| D19 | Versioned contract generation/checking and outbound payload policy | Keep v1 meaning stable; validate/count/hash actual sanitized provider messages and approved views |
| D20 | Pre-dispatch intent binding and authoritative job reconciliation for scheduler submission | Crash before submission must yield explicit uncertain/rejected outcome rather than blind replay; crash after admission reconciles existing job/reservation without duplicates |
| D21 | Pre-dispatch intent binding and authoritative crash reconciliation for specialist delegation, dataset registration, and delivery publication | Non-file mutating tools cryptographically bind parameters, generation, policy, and target identities before execution; pre-admission crashes emit explicit uncertain outcomes; post-admission crashes reconcile authoritative state without duplicate child tasks or double billing |
| D22 | Unified reservation accounting, emergency recovery headroom, configurable budgets, and ENOSPC recovery | All state writes (jobs, checkpoints, provider receipts, logs, managed blobs, runtime assets, bundle staging) route through pre-admission accounting under the 128 MiB policy; 4 MiB emergency headroom and 2 MiB SQLite/WAL reserve ensure cancellation and crash cleanup never fail on saturated storage; ENOSPC exits recover cleanly with staging purge and lease/reservation release; zero silent write exemptions. |
| D23 | Lease-aware archival, tombstones, compact retry fences, and interrupted publication reconciliation | Finite-state bounding must not delete idempotency retry fences or break historical action invariants; command bodies hashed in-place beyond active window; completed actions compacted in-place with dual action_fences; SHA-256 tombstones track reclaimed bytes; active leases/pinned artifacts protected; publication staging crashes adopt valid final directories or reset unreserved finalizing states. |
| D24 | Per-job memory certification requires measured cgroup v2 evidence | Apply Docker memory/swap/CPU/PID limits to every scientific job; probe daemon/kernel support before admission; required certification is refused deterministically if any required cgroup feature is absent. A trusted in-container monitor captures before/after counters and peak/events, fsyncs a receipt before exit, and the supervisor validates and persists that receipt before reporting certified success. |
| D25 | Docker survivor recovery uses exact IDs and fenced ownership metadata | Persist the exact created container ID and owner nonce, state-root digest, project/job, lease epoch, generation, intent revision and spec identity before start. On restart, discover by harness labels, compare against durable job/lease state, and only stop/remove a verified owned survivor by exact ID. Keep leases reserved until inspect confirms exit; ambiguous or mismatched ownership is retained for explicit reconciliation. |
| D26 | Unified Artifact Registry, Content-Addressable Storage (CAS), Two-Tier Summary Bounds, and Versioned Contracts | Unify fragmented M0–M6 artifact stores behind hash-addressed CAS (`<state>/cas/<hash[0..2]>/<hash>`) and a relational registry (`blobs`, `artifacts`, `artifact_versions`, `artifact_lineage`, `artifact_annotations`, `project_selections`, `durable_events`); enforce strict SQLite CHECK bounds for `index_digest` (<=512 B) and `summary_text` (<=4 KiB); restrict scorecard authority to `evaluator:trusted`; migrate `user_version` from 9 to 10 additively with dual-write compatibility. |
| D27 | Journal Events at the SQLite Mutation Boundary; Replay from Snapshot Cursor | Registry events are inserted by database triggers in the same transaction as artifact/version/lineage/annotation/selection/pin mutations, covering legacy dual-write adapters as well as registry calls. Retain 10,000 events per project and a durable per-project expired-through cursor so global AUTOINCREMENT gaps do not cause false resets. Capture artifact snapshot and cursor in one short read transaction; expired cursors request a fresh snapshot. |
| D28 | Four Shared Query-Only Readers with Non-Blocking Acquisition | Share one pool per canonical state root with four dedicated read-only connections (`query_only=ON`); return busy/exhausted immediately for the streaming caller to retry on a later tick. Every query is bounded, each transaction ends before returning, and SSE callers retain only a cursor between polls. |
| D29 | Evaluator Authority Is an In-Crate Capability, Not a Caller Label | The public artifact API cannot register or publish an official scorecard by passing `evaluator:trusted`. Only crate-internal evaluator methods can create trusted scorecard artifacts/versions, and Apex selection verifies the referenced version is an evaluator-authorized scorecard. General callers may register claimed scorecards, but cannot publish them as official. |
| D30 | Revisioned Artifact Commands Reuse Durable Command Receipts | Successor and tombstone mutations compare the shared project revision and persist the exact request and receipt in the existing `(project, command_id)` fence in the same transaction. Exact retries return the original receipt; payload reuse is rejected. Tombstones preserve immutable versions and lineage and refuse pins, selection history, and active producer jobs; physical CAS deletion stays with reference-aware GC. |

## Resolve in the owning phase

- P1: exact control-state/archive/artifact/environment/export quotas and emergency
  headroom. Preserve current 128 MiB policy until an explicit versioned policy changes it.
  Argon proposes 2 GiB per-project CAS/export capacity; treat that as an experiment,
  not an adopted default. Specify combined vs separate limits, staging/transient
  duplication, environment bytes and device/global disk caps before choosing numbers.
  *Resolved in P1.2 (2026-10-14, Phase Complete)*:
  1. Scheduler job submission recovery binds canonical arguments, dataset SHA256,
     environment identity, protocol SHA256, and planned JobSpec SHA256 into `tool_outcomes`
     before dispatch. Crash after admission recovers authoritative `JobView` with zero
     duplicate jobs or double reservations; crash before admission yields explicit
     `unconfirmed_submission_crash` (`effect_outcome: "uncertain"`); conflicting inputs
     or specs are rejected; stale generations fence deterministically (`stale_generation_fenced`).
  2. Specialist delegation (`delegate_task`, `delegate`, `schedule_specialist`), dataset
     registration (`register_dataset`), and delivery publication (`request_finalization`,
     `publish_delivery`, `publish_bundle`) bind target identities (`target_task_id`,
     `target_dataset_id`, `target_bundle_id`/`finalize_job_id`), canonical argument SHA256,
     task generation, and project policy into `tool_outcomes` prior to dispatch. Pre-admission
     crashes deterministically emit explicit uncertain outcomes (`unconfirmed_delegation_crash`,
     `unconfirmed_registration_crash`, `unconfirmed_publication_crash` with `effect_outcome: "uncertain"`).
     Post-dispatch crashes reconcile authoritative rows from `agent_tasks`, `scientific_datasets`,
     and `jobs` without duplicate child tasks, duplicate datasets, or duplicate finalization jobs.
     Target identity reuse, contradictory content, and stale task generations fence deterministically.
  *Resolved in P1.3 (2026-10-14, Phase Complete)*:
  1. Unified reservation accounting across managed write operations (`StorageWriteKind`: `JobAdmission`,
     `Checkpoint`, `ProviderReceipt`, `Log`, `ManagedBlob`, `RuntimeAsset`, `BundleStaging`, `EmergencyRecovery`).
     Each covered path performs bounded pre-admission checks; policy limits are clamped to the 128 MiB
     ceiling. These reservations cannot control unrelated processes writing to the same filesystem.
  2. Recovery headroom is reserved by the policy under normal managed admission. It is not an absolute
     guarantee against external filesystem exhaustion; on actual ENOSPC, retain uncertain job/lease state
     until the storage failure is resolved, then reconcile from durable evidence.
  3. Configurable category budgets: `StorageBudgetConfig` loaded from `storage_budgets.json` or
     `MLRESCURIX_*` environment overrides provides independent limits for artifacts (32 MiB default),
     environments (32 MiB), exports (32 MiB), workflows (64 MiB), checkpoints (8 MiB), and provider
     receipts (8 MiB).
  4. POSIX ENOSPC crash recovery: worker exit status 28 purges disposable staging directories, releases
     all held reservations and environment leases, and settles job state without metadata corruption
     or losing pinned champion artifacts.
  5. Saturation cancellation: running jobs cancel cleanly under storage saturation using emergency headroom
     with `TransactionBehavior::Immediate`, preventing SQLite upgrade deadlocks during concurrent GC.
  *Correction recorded in P1.5 closeout:* earlier text overstated the isolation of emergency/WAL headroom.
  External writes can consume free blocks on the shared device. P1.6's disposable-volume regression proves
  restart recovery after real ENOSPC and available space; it does not claim recovery commits succeed while
  the device remains completely full.
  *Resolved in P1.5 (2026-10-14, Phase Complete):* Docker daemon feature detection and per-job cgroup v2
  memory/swap/CPU/PID limits; deterministic refusal for requested certification when support is absent;
  validated pre-teardown peak/event receipts; atomic exact-ID ownership metadata; restart survivor discovery
  fenced by owner, state root, project, lease and generation; retain leases until confirmed exit.
  *Resolved in P1.6 (2026-10-14, Phase Complete):* repeated restart/fault-cycle proof on a disposable bounded
  volume with real ENOSPC, preserving pinned output and retry fences and checking SQLite integrity/version.
  *Resolved in P1.4 (2026-10-14, Phase Complete)*:
  1. Lease-aware archival and tombstone accounting:
     - Implemented `archive_artifacts()` and `collect_artifacts()` with tombstone accounting (`tombstones` table).
     - Pruned terminal jobs and deleted unpinned managed artifacts are recorded with SHA-256 digests, parent IDs, and reclaimed byte sizes.
     - Active environment leases (`environment_leases`), active tasks (`json_extract(data, '$.status') = 'active'`), running jobs, and pinned artifacts (`pinned = 1`) are strictly protected from reclamation.
  2. Compact retry fences and bounded history growth:
     - Indefinite SQLite metadata growth bounded without deleting idempotency retry fences.
     - Older command request bodies beyond `MAX_ACTIVE_COMMAND_RAW_BODIES` (24) compacted in-place to `hash:<sha256>`, slashing body footprint from up to 64 KiB to 69 bytes while preserving exact payload matching on retry.
     - Completed actions beyond `MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT` (64) compacted in-place in `actions` (hashing input, stripping raw responses, emptying budgets) and mirrored into durable `action_fences` to preserve action identity reuse fencing without breaking historical row-count assertions.
  3. Interrupted publication reconciliation:
     - Staging crash boundary in `Supervisor::publish()`: if crash occurs after atomic `fs::rename(&stage, &final_dir)`, `publish()` verifies fences and directory manifest, adopting existing valid outputs instead of failing with destination collisions.
     - Leaked staging cleanup: `artifacts/<project>/.<job>.staging`, `workflows/<project>/.bundle_stage`, and `staging/bundle_verification_*` directories from interrupted runs are safely purged during artifact collection.
     - Delivery publication crash boundary: `reconcile_delivery_publication()` checks final test gate reservation status; unreserved 'finalizing' state is safely rolled back to 'created' with `effect_outcome: "uncertain"`.
- P2: kind schema extension/visibility policy, event retention, reader pool sizing
  and migration compatibility lifetime.
  *Resolved in P2.1 Design (2026-10-14, Proposal Complete)*:
  1. Kind schema extension & visibility: defined 10 canonical artifact kinds (`data_profile`,
     `research_note`, `code`, `run_log`, `diagnostic_spec`, `prediction`, `scorecard`,
     `apex_bundle`, `checkpoint`, `model_weights`) with extensible namespaced `schema_version`;
     visibility controlled via `'public'|'project'|'internal'`; official scorecard authority
     strictly restricted to `evaluator:trusted` (agent claims remain explicitly unofficial).
  2. Two-tier summary bounds for Context Firewall: `index_digest` <= 512 UTF-8 bytes for
     low-overhead discovery/DAG listing/context injection; `summary_text` <= 4096 UTF-8 bytes
     for on-demand inspection, enforced via SQLite `CHECK` constraints.
  3. CAS layout and atomic write protocol: `<state>/cas/<hash[0..2]>/<hash>` with pre-admission
     byte reservations under 128 MiB policy ceiling, temporary staging in `cas/.staging/<nonce>.tmp`,
     fsync and read-only (`0o444`) permissions before atomic rename.
  4. Directed lineage graph: `artifact_lineage` tracking `'derived_from'`, `'evaluates'`,
     `'packages'`, `'supersedes'` edges across immutable version IDs.
  5. Event journal & retention: `durable_events` with monotonic autoincrement cursor; sliding
     retention window (default 10,000 per project) with cursor snapshot checkpoints.
  6. Read connection pool sizing: 4 dedicated read-only connections (`PRAGMA query_only = ON`)
     to prevent writer starvation during SSE streaming.
  7. Migration compatibility & implementation: `PRAGMA user_version = 10` implemented
     in `schema_m7_artifacts.sql` with non-destructive additive DDL and backfill from
     `managed_artifacts` and `scientific_datasets`. Ephemeral job GC compatibility ensured
     by decoupling `source_job_id` from strict foreign key blocking while preserving
     provenance strings and indexes. Dual-write adapters in `Supervisor::finish` and
     `pin_artifact` guarantee full backward compatibility. Verified in `artifact_registry_p2`
     and harness regression suites.
- P3: effective Docker VM capacity probing, initial fair-queue weights, admission
  request presets, supported cgroup/kernel/platform matrix.
- P4: prediction wire format, argv vs shell surface, environment provisioning tool,
  dataset query restrictions and API names. Read-only SQL alone is not file isolation:
  deny external file readers/extensions/network using the sandbox capability boundary.
  Argon proposes 50 full-validation queries and minimum slice support of
  max(20, 5% of validation rows). These are uncalibrated starting hypotheses;
  choose task/privacy/budget-aware limits and count screening/overlapping diagnostics.
  Neither value guarantees privacy or prevents validation overfitting.
- P5: provider tokenizers, summary bounds, prompt versions and search strategy.
  Preserve OutboundFilter and approved-view controls; its current heuristics are
  insufficient to certify privacy for arbitrary new tool output or source content.
- P6: UI framework and packaging, loopback session lifecycle and artifact renderer.
- P7: safe-export support per family and target benchmark protocol.
- P8/P9: estimate calibration, remote data policy, licensed weights, accelerator
  backends and thresholds for enabling optimization experiments.

Do not choose per-project databases, weaken fsync, expose a host Python fallback
or remove branch ownership locks based only on a blueprint's throughput claims.


### P2.3–P2.5 implementation decisions (2026-10-14)

D27–D29 are implemented in `crates/harness/src/schema_p2_events.sql`, `artifacts.rs`, and `read_pool.rs`. Evaluator authority is restricted to crate-internal scorecard methods; public callers cannot forge it with a string. Event payloads and metadata have UTF-8 byte limits; event pages and artifact query results cap at 64 KiB and 128 rows. Snapshot/replay API state is independent of HTTP transport; the snapshot reports when its 128-artifact page is truncated, and `list_artifacts` provides keyset pagination. The dashboard SSE route remains a later P6 surface. Reader contention returns a retryable error without waiting on an occupied slot. The four regression logs are under `docs/evidence/v2/p2/20261014-p2-inspection-journal-readers/`. These changes do not close any M6 release gate.
