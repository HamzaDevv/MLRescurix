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
| D31 | CAS Staging Sweeps in GC and Multi-Commit Reconciliation | `collect_artifacts` sweeps orphaned temporary files from `cas/.staging/`; CAS deduplication adopts valid pre-existing blobs across partial DB commit crashes without double accounting; hash validation rejects and repairs corrupted blobs. |
| D32 | Effective Host/Container-VM Resource Capability Discovery, Presets, Headroom, and Early Infeasible Rejection | Discover host CPU/RAM/disk and container-VM limits via `docker info`; clamp container workloads to container-VM limits (`min(host, vm)`); detect hardware accelerators passively (Apple Metal on macOS arm64, NVIDIA CUDA via nvidia-smi); enforce configurable headroom (default 1 core, 512 MiB RAM, 64 MiB disk); provide bounded presets (`MICRO`, `STANDARD`, `HEAVY`, `CUSTOM`); deterministically reject infeasible requests before queue admission in `execution::submit` with 0 rows added to SQLite `jobs`. |
| D33 | Resource Lease Lifecycle, Capacity Dimension Distinction, Weighted Fair Queueing with Aging, and Finalization Headroom Preservation | Track worker resources via durable `resource_leases` and `resource_lease_transitions` across `reserved -> active -> heartbeated -> recovering -> reaped -> released`; distinguish worker concurrency caps from CPU threads and memory allocations at global device and project policy scopes; route generic tools, preflight, evaluation, benchmarking, final fit, inference checks, and environment jobs through thread-capped execution (`MLRESCURIX_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, `POLARS_MAX_THREADS` and scikit-learn `n_jobs`); implement weighted fair queueing with continuous aging term (`submitted_ms / 60,000`) and deterministic tie-breaking (`ORDER BY project, submitted_ms, id`); enforce pressure backoff under storage saturation and `MLRESCURIX_PRESSURE_BACKOFF`; preserve 1 worker slot, 1 CPU thread, and 512 MiB RAM headroom for finalization/control-plane work; isolate per-project cancellation and pause without interrupting sibling projects; match persisted container IDs and owner labels against active supervisor epochs to skip foreign supervisor workers during orphan discovery. |

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
   8. Complete P2 Roadmap Gate Closure (2026-10-14, Phase P2 Complete): verified populated-state
      upgrade preservation (retaining `final_test_gates`, `search_sessions`, `trial_results`, and
      `pipeline_receipts`), partial CAS/DB reconciliation with `cas/.staging/` sweep in
      `collect_artifacts`, protected references under concurrent GC (pinned artifacts, Apex selections,
      and lineage edges survive), and event restart/reconnect consistency across process boundaries.
      Verified in 14 tests in `artifact_registry_p2` and full harness regression suites; captured in
      `docs/evidence/v2/p2/20261014-p2-gate-closure/`.
- P3: effective Docker VM capacity probing, initial fair-queue weights, admission
  request presets, supported cgroup/kernel/platform matrix.
  *Resolved in P3.1 (2026-10-14, Slice Complete)*:
  1. Discovery of effective capabilities across host and container-VM: host CPU (`available_parallelism`), host total RAM (`sysctl hw.memsize` / `sysconf`), disk (`statvfs`), and passive accelerator discovery (Apple Metal on macOS arm64, NVIDIA CUDA on Linux via `nvidia-smi`). Docker container-VM limits probed via `docker info` (`NCPU`, `MemTotal`, `CgroupVersion`, cgroup v2 features). Container effective capacity clamped to `min(host, vm)` to prevent host RAM overestimation.
  2. Configured headroom: default 1000 millicores CPU, 512 MiB RAM, 64 MiB disk reserved for host OS and harness control plane. Available capacity = `Effective - Headroom`. Configurable via `<root>/resource_headroom.json` and `MLRESCURIX_HEADROOM_*` environment variables.
  3. Bounded presets: `MICRO` (1 thread, 512 MiB RAM, 1 MiB output), `STANDARD` (2 threads, 2 GiB RAM, 2 MiB output), `HEAVY` (4 threads, 8 GiB RAM, 4 MiB output), and `CUSTOM`. Validates 64 MiB..=64 GiB floor/ceiling and 4 MiB M1 output cap. Resolves onto `JobSpec` and supports `<root>/resource_presets.json` overrides.
  4. Early infeasible-request rejection: rejects excessive threads, memory, output disk, unavailable accelerators, or uncertified memory requirements before queue admission in `execution::submit()` with zero SQLite job insertion. Rechecked in `Supervisor::admit()`. Verified in `tests/resource_broker_p3.rs` and documented in `docs/evidence/v2/p3/20261014-p31-capability-discovery/`.
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

### P2 Gate Closure decisions (2026-10-14)

D31 is implemented in `crates/harness/src/execution.rs` and verified in `crates/harness/tests/artifact_registry_p2.rs`. Orphaned CAS staging files are swept during routine artifact collection (`collect_artifacts`). Populated states upgrade to v10 preserving all historical selection and final test records with zero FK violations. Protected artifact references survive concurrent GC runs without data loss. Reconnecting after restart delivers all committed events in monotonic cursor order with clean snapshot reset on expired cursors. These results close all remaining P2 roadmap acceptance checks. Evidence captured in `docs/evidence/v2/p2/20261014-p2-gate-closure/`. M6 remains independently NOT READY.

### P3.1 implementation decisions (2026-10-14)

D32 is implemented in `crates/harness/src/broker.rs`, `execution.rs`, and `execution_resources.rs`. Capability discovery reports host and container-VM limits, clamping container execution to `min(host, vm)` to ensure macOS Docker Desktop VM limits (e.g. 8 GiB) prevent admitting 10+ GiB container jobs. Hardware accelerators are probed passively without executing untrusted code on host. Resource presets (`MICRO`, `STANDARD`, `HEAVY`, `CUSTOM`) and configured headroom are bounded and validated. Infeasible requests are deterministically rejected in `submit()` before database insertion, leaving SQLite `jobs` untouched. Verified across 5 tests in `crates/harness/tests/resource_broker_p3.rs` and full locked regression suites. Evidence captured in `docs/evidence/v2/p3/20261014-p31-capability-discovery/`. M6 remains independently NOT READY.

### P3.2–P3.5 & Gate Closure decisions (2026-10-14)

D33 is implemented in `crates/harness/src/broker.rs`, `execution.rs`, `execution_resources.rs`, `schema_p3_leases.sql`, and `python/src/mlrescurix_worker/scientific/workflow.py`.
1. **Resource Lease Lifecycle**: Defined tables `resource_leases` and `resource_lease_transitions` with indexes, executed idempotently on schema `user_version = 10`. Leases track transitions across `reserved -> active -> heartbeated -> recovering -> reaped -> released` with full durable transition history and recovery reconciliation on supervisor restart.
2. **Capacity Dimension Distinction**: Distinct global and project limits for worker concurrency (`workers`), CPU allocations (`cpu_threads`), and RAM allocations (`memory_bytes`). Verified that exhausting CPU threads or RAM keeps worker slots free for smaller jobs and skips only the saturated project without blocking sibling projects.
3. **Thread Cap Enforcement**: Enforced across execution backends via `apply_thread_caps_to_command()` setting `MLRESCURIX_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `POLARS_MAX_THREADS`. Forwarded into scientific container workflows and applied to scikit-learn estimators (`RandomForestRegressor`, `LogisticRegression`, `RandomForestClassifier`) via `_leased_threads()`.
4. **Weighted Fair Queueing with Aging & Stable Ties**: Jobs are ordered by weighted service ratio with continuous aging term `(now - submitted_ms) / 60000.0` ensuring older jobs receive priority and bound starvation. Stable tie-breaking orders deterministically by `project, submitted_ms, id`.
5. **Pressure Backoff & Finalization Headroom Preservation**: `is_under_pressure()` defers non-finalization queue admissions when free disk drops below headroom or `MLRESCURIX_PRESSURE_BACKOFF=1` is asserted. Preserves 1 worker slot, 1 CPU thread, and 512 MiB RAM headroom for finalization/control-plane work when `worker_ceiling > 2` or active finalization work is queued/present.
6. **Per-Project Isolation & Safe Discovery**: Project cancellation and pause terminate and reap only the target project's jobs while sibling projects remain running and succeed cleanly. Docker survivor discovery verifies container ownership against active supervisor epochs before stopping/removing containers, skipping foreign supervisor workers.
7. **Acceptance Gate Verification**: Verified across 13 integration tests in `crates/harness/tests/resource_broker_p3.rs` (including 50-job 10x5 stress test) and full repository regression test suites (88 total passed tests). Fresh logs captured in `docs/evidence/v2/p3/20261014-p3-gate-closure/`. Phase P3 is closed. M6 remains independently NOT READY.

### P4.1–P4.3 & Gate Closure decisions (2026-10-14)

D34 is implemented in `crates/harness/src/agents/`, `crates/harness/src/schema_p4_candidates.sql`, `crates/harness/src/execution.rs`, `contracts/v1/experiment_spec.schema.json`, `python/src/mlrescurix_worker/scientific/candidate_runner.py`, and `python/src/mlrescurix_worker/scientific/cli.py`.
1. **Generic Tool Adapters**: Implemented versioned generic agent tools replacing legacy hardcoded recipes:
   - `workspace_fs`: read, write, patch, list, glob, and search with optimistic concurrency hash fencing (`expected_hash` SHA-256 checks reject stale edits). Legacy `file_*` tools retained for backward compatibility.
   - `sandbox_exec`: executes commands with `/workspace:ro` read-only mount, task-local `/scratch:rw` tmpfs, `/outputs:rw` tmpfs, thread caps, and timeout guards. Enforces fallback isolation: if container runtime is unavailable, host Python execution fallback is strictly refused.
   - `artifact_crud`: publish, get, list, lineage, pin, unpin, tombstone. Agent roles are strictly prohibited from publishing `kind = "scorecard"` (`"scorecard publication is restricted to the trusted evaluator API"`).
   - `discover_capabilities`: reports available tools, sandbox container status, pinned scientific image, resource presets, validation query limits (max 50), and state limits (128 MiB).
   - `dataset_query`: inspects feature columns, types, and row count without leaking validation/test row values or label distributions.
   - `experiment_tree`: queries DAG nodes (`id`, `recipe`, `fidelity`, `node_type`, `parent_id`, `created_ms`) directly from `experiment_dag`.
2. **Executable Candidate Manifest & Protocol**: Defined `manifest.json` schema (`candidate_id`, `name`, `task_type` (`binary`, `multiclass`, `regression`), `entrypoint`, `supported_operations` (`fit`, `predict`), `stable_row_keys` (`row_id`), `class_order`, `raw_input_schema`, `source_hash`), CLI `fit`/`predict` invocation contract, and candidate prediction validation (probabilities in `[0, 1]`, summing to $1.0 \pm 10^{-3}$, matching class order, finite numbers).
3. **Fold-Level Label Isolation & Split Drivers**: Harness-managed cross-validation (`StratifiedKFold`, `KFold`, `GroupKFold`, `TimeSeriesSplit`). During `predict`, the validation feature slice is provided strictly without the target label column; candidates attempting to inspect or leak ground truth labels trigger immediate validation abort.
4. **Validation Query Budgets & Diagnostic Slices**: Implemented `validation_query_counters` in SQLite limiting full-validation queries to 50 per project/session to prevent adaptive holdout overfitting; the 51st query is rejected. Diagnostic slices enforce minimum support threshold `max(20, 0.05 * N)`; slices below threshold are automatically suppressed with status `"suppressed"`.
5. **Scorecard Authority & CAS Registration**: Official evaluation scorecards are registered exclusively by the trusted evaluator adapter with `producer_authority = "evaluator:trusted"`, stored in CAS, registered in the `blobs` table, and recorded in `candidate_evaluations`.
6. **Recipe Decoupling & Backward Compatibility**: Decoupled `recipe` in `experiment_spec.schema.json` and `execution.rs::migrate()` to allow arbitrary candidate recipes of length $\le 128$ while preserving `"hist_boosting"` in DDL for M6 migration compatibility. Schema remains at `user_version = 10`.
7. **Acceptance Gate Verification**: Verified across 9 integration tests in `crates/harness/tests/candidate_tools_p4.rs`, 5 unit tests in `python/tests/test_candidate_runner.py`, and 89 passed tests across full repository regression suites (106 total passed tests). Fresh logs captured in `docs/evidence/v2/p4/20261014-p4-gate-closure/`. Phase P4 is closed. M6 remains independently NOT READY.

### P5.1–P5.6 & Gate Closure decisions (2026-10-14)

D35 is implemented in `contracts/v2/task_assignment.schema.json`, `contracts/v2/task_completion.schema.json`, `crates/harness/src/schema_p5_orchestration.sql`, `crates/harness/src/agents/task_contract.rs`, `roles.rs`, `workspace.rs`, `context.rs`, `orchestrator.rs`, `crates/harness/src/search.rs`, and `crates/harness/src/execution.rs`.
1. **Versioned Task Schemas & Durable Ledgers (P5.1)**:
   - Authored strict Draft 2020-12 schemas for `task_assignment` (specifying goals, hypotheses, immutable inputs, allowed tools, branch ownership, resource limits, expected artifacts, acceptance criteria) and `task_completion` (recording verdicts, candidate IDs, metric deltas, protocols, failures, bounded learnings $\le 4096$ bytes, next actions).
   - Created SQLite tables `task_assignments`, `task_completions`, `steering_directives`, and `user_approvals` via `schema_p5_orchestration.sql`, migrated idempotently under `PRAGMA user_version = 10`.
2. **Consolidated Roles & Specialist Registry (P5.1)**:
   - Added `get_consolidated_default_roles()` providing 5 core default roles: `lead`, `data_diagnostics`, `experiment_engineer`, `verifier`, and `critic`.
   - Normalizes legacy/slashed role IDs (`data/diagnostics` -> `data_diagnostics`, `experiment/engineer` -> `experiment_engineer`) while retaining all 8 specialist roles in `get_standard_roles()` for backward compatibility.
3. **Branch Workspace Isolation & Lock Ownership (P5.1)**:
   - Exposed `BranchWorkspaceManager::acquire_lock()` wrapping durable `recovery::BranchLock::acquire()`.
   - Each concurrent mutation task is allocated an isolated branch workspace where writes are completely segregated from peer branches, protected by durable single-writer lock ownership.
4. **DAG Node Traversal & Comparative Multi-Fidelity Search (P5.2)**:
   - Implemented `get_experiment_dag_node()` returning node properties, children, and parent evidence (`primary_metric`, `model_sha256`, `predictions_sha256`).
   - Implemented `evaluate_comparative_strategies()` evaluating bounded multi-fidelity best-first search against uniform random baseline and exploratory MCTS/UCB1 strategies, incorporating explicit metric reward, wall-clock cost penalties, and adaptive validation query limits ($\le 50$).
5. **Request Accounting, Token Bounds, & Tool Pair Validation (P5.3)**:
   - Configured economical ~16K token default (`16,384`) with hard ceiling $\le 64\text{K}$ (`65,536`).
   - Enforced pre-dispatch essential state accounting (system prompt, schemas, constraints, reserves) with immediate refusal if essential state does not fit.
   - Enforced tool pairing validation: every tool call argument must parse as valid JSON, and every tool response must match an active preceding tool call.
   - Outbound filter sanitization (`OutboundFilter`) redacts raw tabular data streams before provider dispatch.
6. **Milestone Compaction & State Preservation (P5.4)**:
   - `AgentContextManager::maybe_compact()` prunes historical messages under budget pressure while strictly preserving: active hypothesis, current steering directive, constraints, user approvals, final-test identity, committed achievement IDs with durable evidence SHAs, and paired assistant tool calls.
   - Context restore verifies durable evidence SHA integrity in `.m2/evidence/`, rejecting tampered evidence with `"Durable evidence hash mismatch"`.
7. **Turn-Boundary Steering & Explicit User Approvals (P5.5)**:
   - Directives from human operators or agent critics (`apply_steering`, `apply_critic_steering`) are journaled into `steering_directives` and increment `handoff_version`.
   - The orchestrator drains pending directives at turn boundaries prior to outbound request construction, without interrupting inner tool loops.
   - Operations unlocking final test access or candidate deployment handoff require explicit registered user approvals (`user_approvals` ledger with action types `"deployment_handoff"` or `"final_test"`), rejecting unauthorized finalizations.
8. **Phase P5 Acceptance Gate Closure (P5.6 & Gate)**:
   - Verified across 11 integration tests in `crates/harness/tests/collaboration_p5.rs` and 162 tests across repository regression suites (173 total passed tests, 0 failures). Evidence captured in `docs/evidence/v2/p5/20261014-p5-gate-closure/`. Phase P5 is closed. M6 remains independently NOT READY.

### P6.1–P6.6 & Gate Closure decisions (2026-10-15)

D36 is implemented in `crates/harness/src/server.rs`, `crates/harness/src/assets/dashboard.html`, `crates/harness/src/main.rs`, `crates/harness/src/lib.rs`, `python/src/mlrescurix_worker/scientific/dashboard.py`, and `crates/harness/tests/dashboard_p6.rs`.
1. **Native Rust HTTP/1.1 & SSE Engine (P6.1 & P6.6)**:
   - Implemented `DashboardServer` in `crates/harness/src/server.rs` using native Rust (`std::net`, `std::io`, `rusqlite`, `serde_json`, `sha2`) with zero external networking dependencies.
   - Binds strictly to loopback (`127.0.0.1`) with session token authentication (`Authorization: Bearer <token>` or `Cookie: mlr_session=<token>; SameSite=Strict; HttpOnly`).
   - Validates `Host` header against `127.0.0.1:<port>` and `localhost:<port>`, and enforces strict `Origin` and CSRF header verification on all mutation requests.
   - Enforces 64 KiB request body ceiling on command ingress, draining excess bytes up to `Content-Length` before returning 413 to prevent TCP reset RST on macOS.
   - Read-only queries execute against the P2 4-connection read pool (`query_only=ON`).
   - Serves revisioned project snapshots at `GET /v1/projects/:id/snapshot`, persistent Server-Sent Events at `GET /v1/projects/:id/events?after=:cursor` emitting monotonic durable events from `durable_events`, revision-fenced mutation command ingress at `POST /v1/projects/:id/commands`, CAS blob streaming at `GET /v1/projects/:id/artifacts/:id` with path traversal protections, and embedded dashboard HTML at `GET /`.
2. **Safe Artifact Renderer & Zero-XSS DOM Construction (P6.2)**:
   - Removed all untrusted `innerHTML` interpolation across frontend assets (`crates/harness/src/assets/dashboard.html` and `python/src/mlrescurix_worker/scientific/dashboard.py`).
   - All dynamic text, hypotheses, steering directives, notes, and labels are injected exclusively via safe DOM properties (`textContent`) and structured SVG DOM element nodes.
   - Diagnostic slice visualizer explicitly renders status badges for suppressed micro-slices (`status: "suppressed"` when sample size falls below minimum support limit `max(20, 0.05 * N)`).
3. **Interactive Animated Experiment DAG Canvas (P6.3)**:
   - Interactive SVG canvas with zoom, pan, and dynamic layout rendering the multi-fidelity experiment tree (`experiment_dag` joined with `trial_results`).
   - Visual nodes styled by fidelity (screening 20%/50% vs full 100%), recipe, and lifecycle stage (`baseline`, `screening`, `full_fidelity`, `candidate`, `ablate`).
   - Pulsing ring animations denote actively running trials; color-coded border glow indicates feasibility and apex candidacy.
   - Node traversal allows clicking any node to highlight its full lineage path back to root baseline while expanding parent evidence (`primary_metric`, `model_sha256`, `predictions_sha256`).
4. **Reactive Pareto Frontier & Dynamic Leaderboard Explorer (P6.3)**:
   - 2D scatter visualization plotting primary metric against training latency/fidelity with animated Pareto frontier curves.
   - Interactive hover tooltips displaying seed, recipe parameters, fold metrics, and artifact hashes.
   - Sortable Leaderboard table clearly distinguishing official scorecards (`evaluator:trusted`) from unverified agent claims.
5. **Sliding Resource & Contention Drawer (P6.4)**:
   - Telemetry drawer rendering real-time system metrics: allocated vs. measured CPU threads, memory bytes with smooth CSS gradient meters, queue wait times, active worker lease timers, and cross-project disk pressure backoff alerts.
6. **Turn-Boundary Steering & Control Center (P6.5)**:
   - Steering input deck for injecting human directives (`apply_steering`) and critic directives, verified with turn-boundary acknowledgment.
   - Revisioned project notes and intent editor: note edits issue explicit revision-fenced project commands and advance `handoff_version`; merely editing Markdown cannot silently mutate the locked evaluation protocol.
   - Action triggers for candidate pinning (`pin_artifact`), project pause/resume/cancel, and holdout access unlock requests (`user_approvals`). Stale revisions are deterministically rejected with HTTP 409 conflict.
7. **Reconnect & Replay Resilience, Large DAGs, & Accessibility (P6.6)**:
   - Client automatically recovers full state from snapshot plus monotonic SSE cursor after network interruption.
   - Expired cursor (`after < expired_through_cursor`) triggers an `event: reset` frame instructing the client to perform a clean state reload.
   - Large DAG rendering tested with 120+ nodes completing under responsive time bounds.
   - Adheres to `@media (prefers-reduced-motion: reduce)` accessibility preference, disabling continuous CSS/SVG animations.
8. **Phase P6 Acceptance Gate Closure (P6.6 & Gate)**:
   - Verified across 9 integration tests in `crates/harness/tests/dashboard_p6.rs`, 8 unit tests in `python/tests/test_search.py`, and 165 tests across repository regression suites (182 total passed tests, 0 failures). Evidence captured in `docs/evidence/v2/p6/20261015-p6-gate-closure/`. Phase P6 is closed. M6 remains independently NOT READY.

### P7.1–P7.7 & Gate Closure decisions (2026-10-15)

D37 is implemented in `contracts/v2/candidate_bundle.schema.json`, `python/src/mlrescurix_worker/scientific/export.py`, `python/src/mlrescurix_worker/scientific/serve.py`, `python/src/mlrescurix_worker/scientific/benchmarks.py`, `python/src/mlrescurix_worker/scientific/workflow.py`, `python/src/mlrescurix_worker/scientific/bundle.py`, `crates/harness/src/assets.rs`, `crates/harness/src/bundle.rs`, `crates/harness/src/main.rs`, `crates/harness/tests/release_p7.rs`, and `python/tests/test_p7_scientific_delivery.py`.
1. **Executable Candidate Bundle Manifest & Dependency Isolation (P7.1)**:
   - Authored strict Draft 2020-12 schema `contracts/v2/candidate_bundle.schema.json` validating format `mlrescurix.bundle.v2`.
   - Packages frozen model weights (`frozen_model.joblib`), zero-pickle SafeTensors (`weights.safetensors`), training source (`training/source/workflow.py`), feature pipeline (`training/source/features.py`), inference CLI/API contracts (`predict(records)`), input/output schemas (`input_schema.json`, `output_schema.json`), model scorecard, lineage records, and resource measurements (`cold_load_ms`, `warm_p95_ms`, `peak_rss_kb`, `package_bytes`).
   - Clean separation of isolated inference dependencies in `inference.lock` strictly rejecting build tools, test frameworks (`pytest`, `harness`), and compilers.
2. **Native / SafeTensors Model Exports & Modality Tolerances (P7.2)**:
   - Implemented `export_model`, `SafeTensorsPredictor`, and `verify_export_parity` in `python/src/mlrescurix_worker/scientific/export.py`.
   - Modality-specific documented tolerances (`DEFAULT_TOLERANCES`: classification probabilities `rtol=1e-5, atol=1e-7`; continuous regression `rtol=1e-5, atol=1e-6`; tree discrete `rtol=1e-4, atol=1e-5`), avoiding hardcoded 1e-6 across disparate modalities.
   - Strict sandbox boundary protection: export directory path must not escape sandbox bounds (`..` in path components rejected).
   - SafeTensors zero-pickle inference engine executing purely via NumPy array arithmetic without unpickling risks.
3. **Target Verification, Sealed Final Assessment, & Zero Leakage (P7.3)**:
   - Single-use sealed holdout evaluation strictly enforced via authoritative state gate (`final_test_gates` status `consumed`) and verified user approval token (`user_approvals` ledger with action types `final_test` or `deployment_handoff`).
   - Strict zero-leakage verification deterministically rejecting bundles whose golden examples (`examples_input.json`) contain rows from the sealed holdout test split (`splits.test` in authoritative `dataset_manifest.json`).
4. **Standalone HTTP Serving Recipe (P7.4)**:
   - Implemented `python/src/mlrescurix_worker/scientific/serve.py` providing `/health` and `/predict` endpoints using Python standard library `http.server` with zero external server framework dependencies.
   - Employs locked inference-only dependencies (`inference.lock`). Rejects empty bodies, malformed JSON, and requests violating input schema contracts (`400/422`).
5. **Installed Standalone Workflow Lifecycle (P7.5)**:
   - Embedded worker assets (`crates/harness/src/assets.rs`) enabling standalone execution outside repository checkout.
   - Verified full lifecycle: fresh project initialization, dataset registration, crash/resume across simulated failures, candidate finalization into bundle, and standalone CLI prediction (`python inference.py --input <in> --output <out>`).
   - Upgraded populated state databases via `execution::migrate()` preserving historical consumed gates and approvals without schema version bump (`PRAGMA user_version = 10`).
6. **Prospective Matched Benchmarks & Suboptimal Trial Transparency (P7.6)**:
   - Implemented `python/src/mlrescurix_worker/scientific/benchmarks.py` running matched benchmarks comparing random, search, and agent baselines across fixed seeds (`[2026, 42, 101]`) under matched compute budgets.
   - Transparently records suboptimal and failing trials alongside winning candidates, guaranteeing honest reporting without synthetic metric inflation.
7. **Capability Matrix & Release Audit (P7.7 & Gate)**:
   - Published authoritative capability matrix in `mlrescurix-harness capabilities` bound to immutable source hash (`source_sha256`) and container image hash (`mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd`).
   - Formally audited all 5 historical M6 release gates (preserved as `not_ready`) and all 7 v2 release gates (P1–P7 recorded as `passed`).
   - Verified across 8 integration tests in `crates/harness/tests/release_p7.rs`, 4 unit tests in `python/tests/test_p7_scientific_delivery.py`, and 175 tests across repository regression suites (187 total passed tests, 0 failures). Evidence captured in `docs/evidence/v2/p7/20261015-p7-gate-closure/`. Phase P7 is closed. M6 remains independently NOT READY.

### P8.1–P8.5 & Gate Closure decisions (2026-10-15)

D38 is implemented in `contracts/v2/hardware_profile.schema.json`, `contracts/v2/dataset_metadata.schema.json`, `contracts/v2/feasibility_report.schema.json`, `contracts/v2/execution_plan.schema.json`, `python/src/mlrescurix_worker/scientific/feasibility.py`, `crates/harness/src/feasibility.rs`, `crates/harness/src/schema_p8_feasibility.sql`, `crates/harness/src/assets.rs`, `crates/harness/src/agents/tools.rs`, `crates/harness/src/agents/roles.rs`, `crates/harness/src/agents/orchestrator.rs`, `crates/harness/src/main.rs`, `crates/harness/tests/feasibility_p8.rs`, and `python/tests/test_p8_feasibility.py`.
1. **Feasibility & Execution Plan Artifacts (P8.1)**:
   - Defined strict Draft 2020-12 typed schemas for `HardwareProfile`, `DatasetMetadata`, `FeasibilityReport`, and `ExecutionPlan`.
   - Reused P3 resource broker discovery (`broker.rs`), capturing effective CPU, RAM, disk headroom, Docker VM/accelerator limits, and memory certification support.
   - Accurately accounts for dataset cardinality expansion (e.g. one-hot encoding), calculating transformed expansion factors and dense memory footprints.
2. **Analytical Estimators & Capped Micro-Pilots (P8.2)**:
   - Model family-specific analytical complexity bounds for Linear, Boosting, Tree, Neural, and Custom architectures.
   - Accurately accounts for CV fold counts, final refit multiplier, feature expansion factor, and concurrent worker lease allocations.
   - Implemented capped micro-pilots ($N \le 100$ rows, low-epoch probes) with strict timeout (default 3000ms) and peak RSS memory guards. Clamps requested rows to $\le 100$ rows and detects memory cap violations (`status='memory_exceeded'`).
3. **Transparent Multi-Tier Feasibility Reporting (P8.3)**:
   - Deterministic four-tier feasibility verdicts emitted prior to heavy dispatch: `locally_feasible`, `feasible_with_adaptation`, `requires_external_compute`, `unknown`.
   - Reports include full documented assumptions, confidence bounds (`high`, `medium`, `low`), predicted runtime range, predicted peak RAM range, and measured pilot trial IDs. Calibrates predictions using micro-pilot results (raising confidence to `high` when pilot succeeds).
4. **Evidence-Backed Local Alternatives (P8.4)**:
   - Evaluates and suggests concrete local adaptations: quantization (float64 to float32 downcasting), sparse linear baselines (scipy CSR matrices), progressive fidelity screening (20% subsampling for candidate exploration), downsampled screening, smaller tree/boosting architectures, and chunked streaming ingestion.
   - Strictly prohibits quiet semantic degradation by documenting quality impacts and requiring explicit negotiation.
5. **Constraint Negotiation & Safe Downscaling (P8.5)**:
   - Deterministic constraint negotiation protocols generating immutable `ExecutionPlan` artifacts.
   - Enforces safe local downscaling without purchasing external compute. Strictly preserves locked evaluation split protocols (`locked_split_protocol_preserved = true`), disallowing modifications to evaluation partitions.
   - Heavy training dispatch feasibility gating (`enforce_heavy_dispatch_feasibility`) strictly refuses tasks with `requires_external_compute` before queue admission.
6. **Phase P8 Acceptance Gate Closure (Gate)**:
   - Verified across 11 integration tests in `crates/harness/tests/feasibility_p8.rs`, 6 unit tests in `python/tests/test_p8_feasibility.py`, and 185 tests across repository regression suites (202 total passed tests, 0 failures). Evidence captured in `docs/evidence/v2/p8/20261015-p8-gate-closure/`. Phase P8 is closed. M6 remains independently NOT READY.

