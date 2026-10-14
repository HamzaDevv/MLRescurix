# Phased implementation roadmap

Updated 2026-10-14 after both supplied Argon blueprints and local source review.
Each task ID is stable for progress tracking. Proposed module names may change;
keep their ownership and contracts. No time estimates are asserted before sizing
the implementation slices. Do not undertake the entire roadmap in a single session.

## Shared completion rules

For cross-language changes, identify the authoritative schema, generator and both
validators. `scripts/generate_contracts.py` currently generates the v1 worker frame
schema/shared fixtures; extend it or add a versioned generator/checker as needed.
Do not put incompatible v2 meanings into `contracts/v1/` or assume running the
current generator creates Rust types. Keep worker transport changes separate from
candidate/artifact schemas when their versioning needs differ.

Every implemented slice needs changed-source references, meaningful tests of its
boundary, exact command/results, source/image/schema identities, known limitations,
and a next-session handoff. Tests must check failure/restart/concurrency behavior
where relevant. Update capabilities honestly; metadata registration is not proof
of execution, integrity is not parity, and screening is not certified feasibility.

Use `cargo test --locked -p mlrescurix-harness` for relevant Rust regressions and
the digest-pinned scientific image for Python/scientific fitting or inference.
The existing Python tests use unittest; do not assume a host `pytest` invocation
is the authoritative test command. Select suites appropriate to the slice and
run integrated gates at phase completion. Preserve old evidence; the existing
`scripts/run_m6_review_checks.sh` overwrites M6 evidence files, so revise its output
handling or capture fresh phase-specific logs before using it for new acceptance.

## P0 — Audit and durable planning

**Outcome:** shared scope, corrected contracts and files that survive chat loss.

- P0.1 Review source, M6 handoff and both supplied blueprints.
- P0.2 Preserve input specifications, create audit/architecture/roadmap/decisions.
- P0.3 Create status tracker and paste-ready continuation prompt; link from README.

**Gate:** documentation links resolve; current runtime status remains unchanged;
plan clearly distinguishes proposed/implemented/verified work. Documentation only.

## P1 — Recovery, storage and release prerequisites

**Depends on:** P0. **Outcome:** trustworthy state transitions before new surfaces.

**Touchpoints:** `storage.rs`, `execution.rs`, `core.rs`, agent recovery/orchestrator/
context, `assets.rs`, bundle/finalization code; foundation/recovery/release tests.

- P1.1 Inventory every SQLite/filesystem mutation and runner entrypoint. Produce a
  mutation/effect matrix with ownership, reservation and recovery classification.
  Converge authoritative writes behind a command service incrementally; retain
  bounded queues and cross-process serialization until callers are migrated.
- P1.2 Extend durable tool intent/outcome reconciliation to scheduler submission,
  specialist delegation, registration and delivery publication. Bind call identity
  to project/task/turn/generation/policy plus canonical inputs. Add provider
  response-before-tools persistence and explicit unknown dispatch/billing state.
  Pure cache results require dataset/split/environment/protocol identities too.
- P1.3 Route checkpoints, provider receipts, logs, managed blobs, runtime assets
  and bundle staging through reservation accounting. Include SQLite/WAL growth and
  emergency recovery headroom. Enforce existing 128 MiB policy; specify configurable
  artifact/environment/export budgets without silently exempting writes.
  The current ceiling covers managed state, not only control metadata. Separate
  quotas require an explicit policy migration, project/global limits, device disk
  accounting and recovery headroom before artifacts leave that ceiling.
- P1.4 Add lease-aware archival/tombstones, compact retry fences and interrupted
  publication reconciliation. Indefinite unique-history retention cannot fit a
  finite cap: archive under explicit bounded policy, or stop admission honestly.
- P1.5 Implement supported per-job cgroup enforcement/measurement receipts and
  feature detection. Collect peak/events before cgroup removal; prove refusal when
  required certification is unavailable. Preserve Docker-only candidate execution.
  Persist exact container IDs and owner/state-root/project/lease/generation labels.
  After supervisor death, discover Docker-daemon survivors and verify ownership
  and lease epochs before adopting/reaping them; release only after confirmed exit.
- P1.6 Prove scheduler ENOSPC recovery on disposable bounded storage, repeated
  restart/cleanup and cancellation without losing protected artifacts or fences.

**Gate:** fault injection before/after effect/result/checkpoint cannot duplicate
known effects; unknown irreversible effects refuse. Concurrent writers cannot
over-admit state. ENOSPC leaves recoverable receipts. Orphan workers are reaped
before lease release. Memory enforcement is measured on declared supported targets.

**Status (2026-10-14): Complete for P1.1–P1.6.** Fault/restart coverage, actual
disposable-volume ENOSPC, digest-pinned Docker pipeline, cgroup peak/OOM receipts,
and daemon survivor recovery are recorded under
`docs/evidence/v2/p1/20261014-p1-review-resources/`. Broader M6 delivery and
benchmark gates remain open as tracked in `PROGRESS.md`.

**First implementation slice:** P1.1 plus one non-file operation from P1.2 (job
submission), with a crash after effect but before outcome checkpoint. This is a
reviewable starting point; do not begin by deleting tool APIs or rebuilding schemas.

**Remaining M6 mapping:** release gate 1 -> P1.2/P5 recovery; gate 2 -> P1.3/P1.4/
P1.6; gate 5 -> P1.5/P3/P7. Installed full delivery (gate 3) and fresh matched
benchmarks (gate 4) remain P7 gates; P1 completion alone is not M6 release readiness.

## P2 — Artifact versions, lineage and events

**Depends on:** P1. **Outcome:** one artifact authority for agents and humans.

**Touchpoints:** new artifact service/migrations/contracts; storage snapshots/events;
existing managed artifacts and scientific registration adapters.

- P2.1 Add blobs, logical artifacts, immutable versions, lineage edges, annotations
  and project selection references. Use project-scoped foreign keys; allow kind
  extensions through versioned schemas. Allocate migrations from actual latest
  version (currently 9); update all guards, fixtures and embedded assets.
  Define versioned cross-language schemas and fixtures, including <=512-byte index
  digests and <=4-KiB on-demand summaries; bound tokens and total query output too.
- P2.2 Implement reserved staging/hash/fsync/publication/commit/reconciliation and
  lease-aware GC. Add bounded manifest handling for multi-file models/directories.
- P2.3 Add query/lineage/pin/successor/tombstone commands with revision/idempotency
  checks. Restrict official scorecards to evaluator authority. Bridge M6 records;
  do not replace all scientific tables before readers and writers are migrated.
- P2.4 Define durable event schemas, snapshot cursor, replay retention and reset
  behavior. Publish registry mutation and event in the same database transaction.
  Project artifact/annotation summaries entering provider context must pass typed
  outbound policy and the existing approved-view/sanitization boundary.
- P2.5 Add bounded read connection pool and short transactions, observing WAL
  checkpoints and contention. Measure needed pool size; avoid holding DB reads for SSE.

**Gate:** old states upgrade without lost selection/final-test records; partial
CAS/DB commits reconcile; protected references survive concurrent GC; stale updates
and forged scores reject; restart/reconnect delivers committed events consistently.

**Status (2026-10-14): Phase P2 Complete.**
P2.1 design and P2.2 schema-v10/CAS/dual-write evidence remain in `docs/v2/P2_1_ARTIFACT_DESIGN.md` and `docs/evidence/v2/p2/20261014-p2-artifact-registry/`. P2.3 inspection queries and pin/unpin enforce bounded outputs and summary byte limits; public callers cannot self-assert official evaluator authority, and Apex selection checks the scorecard source. P2.4 SQLite triggers atomically journal registry writes; subscriptions capture a snapshot cursor and replay retained events with explicit reset after expiry. P2.5 shares four query-only connections per state root and refuses excess readers without waiting. P2.6 adds project-revision and durable command-id fences to successor/tombstone commands, preserves immutable history, protects selection/pin/active-job references, and journals tombstones atomically. Test logs are in `docs/evidence/v2/p2/20261014-p2-inspection-journal-readers/` and `docs/evidence/v2/p2/20261014-p26-successor-tombstone/`. Full P2 roadmap acceptance gate closed: populated upgrade preservation, partial CAS/DB reconciliation with `cas/.staging` sweep in `collect_artifacts`, protected references under concurrent GC, and restart/reconnect event consistency across process boundaries verified in `crates/harness/tests/artifact_registry_p2.rs` (14 passed) and full regression suites; evidence captured in `docs/evidence/v2/p2/20261014-p2-gate-closure/`. M6 remains independently NOT READY.

## P3 — Resource leases across projects

**Depends on:** P1, P2 telemetry. **Outcome:** ten projects/five intents per project
remain responsive within device capacity; concurrency scales down when necessary.

**Touchpoints:** broker module, existing execution reservations/supervisor, sandbox
and capability probes; resource contracts and isolated runner tests.

- P3.1 Discover effective host/container-VM CPU, memory, disk and accelerator
  capabilities. Configure headroom and request presets; reject infeasible requests.
- P3.2 Add atomic global/project capacity reservations, weighted fair queues with
  aging, bounded queues and stable tie ordering. Distinguish concurrency caps from
  CPU/RAM allocation. Persist lease/heartbeat/recovery transitions.
- P3.3 Route generic tools, preflight, evaluation, benchmarking, final fit, inference
  checks and environment jobs through the broker. Apply thread caps everywhere,
  including estimator n_jobs and nested fold/HPO execution.
- P3.4 Add pressure backoff, per-project pause/stop isolation, orphan reconciliation
  and resource-release-on-reap. Preserve capacity for finalization/control-plane work.
  Match persisted container IDs/owner labels against active epochs; do not kill
  another supervisor's valid worker merely because its container has harness labels.
- P3.5 Test ten projects x five queued intents, mixed-size jobs, starvation bounds,
  worker crash and cancel/restart; attach aggregate allocation/resource timelines.

**Gate:** admitted sums stay within declared capacity; a large queued job receives
service under the documented policy; one project's cancellation leaves others
running; no runner bypasses leases. Report actual active count and queue waits.
Use synthetic/small jobs; do not intentionally exhaust the user's host RAM.

**Status (2026-10-14): P3.1 implemented and verified; P3 phase gate remains open for P3.2–P3.5 broker leases, fair queueing, and multi-project tests.**
P3.1 is implemented in `crates/harness/src/broker.rs`, `execution.rs`, and `execution_resources.rs`. It discovers host and container-VM capabilities (`min(host, vm)`), passively probes accelerators (Apple Metal on macOS arm64, NVIDIA CUDA on Linux), configures headroom (default 1 core, 512 MiB RAM, 64 MiB disk), bounds request presets (`MICRO`, `STANDARD`, `HEAVY`, `CUSTOM`), and deterministically rejects infeasible requests before admission in `execution::submit` (0 rows added to SQLite `jobs`). Verified across 5 tests in `crates/harness/tests/resource_broker_p3.rs` and full locked regression suites. Evidence in `docs/evidence/v2/p3/20261014-p31-capability-discovery/`. M6 remains independently NOT READY.

## P4 — Agent-first tools and executable candidates

**Depends on:** P2, P3. **Outcome:** agents can create/fix arbitrary candidate code
without modifying a recipe registry, while evaluation and budgets remain trusted.

**Touchpoints:** tools/workspace/sandbox, contracts, custom preflight/runner,
scientific workflow/search, experiment DAG and delivery adapter.

Include `agents/gateway.rs::OutboundFilter`, versioned schemas/generation/fixtures,
`scientific/scheduled_pipeline.py`, `calibration.py`, `ensemble.py`,
`search/adapter.py` and `docker/scientific/image.ref` in compatibility review.

- P4.1 Add versioned workspace_fs/sandbox_exec/evaluate_submission/artifact_crud
  adapters; retain existing operations during migration. Add capability discovery.
  Keep governed delegation/lifecycle/finalization controls. Optional dataset_query
  and experiment_tree are convenience surfaces using the same authority boundaries.
  Default to immutable `/workspace:ro`, bounded task-local scratch/outputs and
  reviewed source commits through workspace_fs. Bind scientific development and
  candidate prediction to the same approved digest-pinned environment profile;
  separately verify any distinct delivery inference profile.
- P4.2 Define executable candidate manifest and CLI fit/predict protocol, raw-input
  schemas, stable row keys, class order, source/model/environment bindings and
  timeout/output expectations. Support old CustomEstimator via an adapter.
  Add row/class-aligned prediction tables and a versioned parameter/search-space
  adapter for executable candidates. Preserve OOF calibration and composite
  inference through scheduled_pipeline/ensemble adapters without importing
  candidate code into trusted scoring; calibration already accepts OOF arrays.
- P4.3 Decouple experiment identity from built-in recipe enum across SQL, Rust,
  contracts and Python dispatch. Keep lifecycle states validated; distinguish
  search operation (draft/debug/improve/ablate) from run status and fidelity.
- P4.4 Run candidate fitting/prediction inside isolated jobs; trusted evaluator
  receives only validated outputs. Never import arbitrary candidate code into a
  label-bearing trusted process. Evaluate OOF provenance and bounded diagnostics.
  A trusted split driver freezes source/config and runs fit/predict per fold with
  only that fold's training labels and feature-only held rows, isolated fit state
  and no shared writable cross-fold cache. Record prior coding-agent label exposure;
  per-fold execution does not prove absence of memorized labels embedded in source.
  Use untouched outer validation or nested cross-fitting to evaluate fitted
  calibration/stacking. Limit adaptive validation queries, minimum slice support,
  overlapping diagnostics and output precision; thresholds alone do not prove privacy.
- P4.5 Add no-dependency custom model and legacy-code rescue vertical slices.
  Add optional locked library profiles later; environment provisioning is separate
  from offline training. Keep embedded trusted runtime assets for standalone use.
  Distinguish unfitted code repair from pretrained artifact rescue. A clean refit
  resets preprocessing and all learned state; warm starts retain prior exposure.
  Otherwise report an unverified legacy baseline until provenance supports a
  disjoint untouched holdout. Record attestations and their limits explicitly.
- P4.6 Preflight schema/missingness/category behavior, reload parity and batch/row
  consistency where the task requires it. Invalid input may be rejected according
  to contract; do not force every model to predict on Inf/all-null input.

**Gate:** an unregistered algorithm completes fit/evaluate/delivery through the
same trusted boundary; existing candidates still work; row/class/source substitution
rejects; candidate process cannot access evaluation labels or mutate official scores.
Every fold-local transform uses the permitted training split.

## P5 — Orchestration, checkpoints and context

**Depends on:** P4. **Outcome:** bounded collaborative research survives steering,
crashes and long experiment runs without losing scientific state.

**Touchpoints:** agent context/gateway/roles/orchestrator/recovery/workspaces and
versioned task/handoff contracts.

- P5.1 Introduce versioned task assignment/completion schemas. Consolidate default
  roles without deleting specialized capabilities. Give each concurrent mutation
  task its own branch/workspace; branch lock remains useful for ownership.
- P5.2 Maintain experiment DAG operations and parent evidence. Begin with bounded
  best-first/multi-fidelity selection; add MCTS/UCB as an optional strategy with
  explicit reward, costs and adaptive-validation assumptions, then compare fairly.
- P5.3 Enforce provider-specific complete request accounting before every dispatch.
  Retain economical ~16K default; configurable hard upper window <=64K. Bound
  schemas/logs/handoffs and response reserve; refuse when essential state cannot fit.
  Check typed outbound projection, approved views and OutboundFilter for every new
  tool/summary/steering surface. Hash/count the actual serialized provider payload;
  reject broken JSON/tool pairing or lost essential state after sanitization.
- P5.4 Compact at task/milestone boundaries and budget pressure. Store durable
  summaries with evidence IDs; preserve current steering, pending tool pairs,
  constraints, approvals and final-test identity. Recover without replaying LLM
  requests whose responses are already durable.
- P5.5 Define pause/cancel/resume semantics; reconcile parallel workers and provider
  receipts. Optional epoch checkpoints bind code/data/environment/seeds and include
  optimizer/RNG state where needed; not every algorithm supports mid-fit resume.
- P5.6 Version/test role prompts for hypothesis quality, fold locality, honest
  evidence, escalation and independent critique. Agent critic complements deterministic
  checks; neither can prove absence of every domain-specific leakage source.

**Gate:** long mocked-provider runs remain bounded; restore latest steering and
unfinished jobs; duplicate delegation/result delivery does not duplicate charges
or work; parallel branches cannot overwrite one another. Compare token overhead
against the current orchestrator. Paid-provider evidence is a separate approved run.

## P6 — Transparent dashboard and human controls

**Depends on:** P2–P3 for read/resource views (P6.1–P6.4), with existing M6 data or
explicit placeholders for unavailable new data; P4–P5 for new candidate/agent
semantics and steering (P6.5). P6.6 verifies each available slice, then the integrated
workflow. **Outcome:** users see actual work, resources and evidence
and can steer safely. Build API first, then one vertical UI view at a time.

**Touchpoints:** Rust HTTP/SSE service (axum candidate), embedded UI assets,
scientific dashboard compatibility bridge, artifact and lifecycle command services.

- P6.1 Port authenticated snapshots/events/commands to Rust with cursor replay,
  revisions, request caps, project authorization and Host/Origin/CSRF checks.
- P6.2 Add safe research/artifact renderer and code/log viewer; remove untrusted
  innerHTML interpolation. Validate diagnostic specs and render bounded summaries.
- P6.3 Add actual experiment DAG/search operations, leaderboard/Pareto view,
  diagnostic gallery and research feed. Show feasible/unverified/failed separately.
- P6.4 Add fifth resource drawer: allocated vs measured CPU/RAM, queue wait,
  running slots, backend capacity and contention across projects.
- P6.5 Add revisioned notes, pin/baseline actions, prune/cancel and steering deltas.
  A note editing “constraints” must issue an explicit project-intent command;
  merely editing Markdown cannot silently alter the locked evaluation protocol.
- P6.6 Test reconnect/restart, expired cursor, event duplication, large DAGs,
  malicious notes/plot payloads, slow clients and reduced-motion accessibility.

**Gate:** UI recovers from snapshot plus cursor; mutation receipts agree with CLI/
agent behavior; unsafe rendering/project crossing reject; animations reflect committed
state. No unsupported “zero lag” guarantee. Embed built UI for standalone installs.

## P7 — Apex delivery and release certification

**Depends on:** P1–P6. **Outcome:** a complete agent-first tabular release with
working standalone output and scoped claims.

**Touchpoints:** bundle/finalization/verifier, runtime assets, install/migration
tests, benchmark drivers and release capabilities/docs.

- P7.1 Version bundle manifest for executable candidates; include weights,
  training source, ingestion/preprocess/postprocess, inference CLI/API, schemas,
  separate training/inference locks, score/model cards, lineage and measurements.
- P7.2 Prefer supported native/ONNX/safetensors exports where parity holds. These
  formats do not sandbox custom Python or certify the complete bundle. Record
  numerical tolerances per adapter; do not hardcode 1e-6 for every modality.
- P7.3 Verify final reload/raw-input parity/cold-warm latency/peak memory/size on
  the declared target. Preserve bound approvals and single-use sealed assessment.
  Golden examples are synthetic/explicitly permitted; no private-data packaging.
- P7.4 Optional serving recipe adds health/predict and locked runtime dependencies;
  benchmark service overhead independently. “Zero dependency inference” is not
  plausible for arbitrary scientific models; exclude harness/LLM dependencies instead.
- P7.5 Prove installed binary outside checkout -> research/custom candidate ->
  crash/resume -> finalization -> verified bundle -> standalone prediction. Upgrade
  populated version-9 states and older supported states without changing consumed gates.
- P7.6 Predeclare representative datasets/splits/seeds/metrics/budgets and compare
  random/search/agent baselines with matched total costs, including refit, evaluation,
  calibration, downloads and LLM costs. Preserve disappointing/failing trials.
- P7.7 Publish capability matrix and evidence index bound to source/image hashes.
  Explicitly decide each of the five M6 release gates and new v2 gates.

**Gate:** no manual state edits; runnable exact frozen bundle; honest unknown/failure
outcomes; installed/recovery/storage/benchmark scenarios demonstrate their specific
claims. No superiority/statistical-equivalence claim from convenient synthetic parity.

## P8 — Compute feasibility and local alternatives

**Depends on:** P3, P4, P7. **Outcome:** estimate before spending substantial training
budget, explain limits and choose evidence-backed local alternatives.

- P8.1 Define HardwareProfile, DatasetMetadata, FeasibilityReport and ExecutionPlan
  artifacts. Reuse broker discovery; include Docker VM/accelerator capability limits.
- P8.2 Add family-specific analytical ranges and capped micro-pilots, accounting
  for CV/final fit/benchmark cost, transformed data expansion and concurrent leases.
- P8.3 Report local feasible / feasible with adaptation / requires external compute /
  unknown. Include assumptions, confidence, predicted ranges and measured pilot IDs.
- P8.4 Suggest streaming, smaller models, sparse linear baselines, frozen embeddings,
  progressive fidelity and distillation where appropriate. Compare actual quality
  and complete inference cost; a tiny head still pays encoder latency.
- P8.5 Negotiate changed goals/latency, downloads, external teacher/data transfer or
  cloud cost explicitly. Existing authorized local constraints may guide automatic
  downscaling; do not quietly change semantics or purchase resources.

**Gate:** estimates are calibrated against bounded real runs; misleading guarantees
are absent; remote/cloud calls never occur simply because the agent recommends them;
unsupported backends return useful alternatives rather than crash.

## P9 — Separately gated expansion and optimizations

**Depends on:** P8. Each slice is optional and independently scoped.

- P9.1 Out-of-core/Arrow data adapter: feature-only split views, ingestion fingerprints,
  byte limits and memory-mapped reuse. Measure actual preprocessing/model copies;
  Arrow mmap does not make arbitrary training zero-copy.
- P9.2 Time-series adapter: horizons, feature availability, entity/temporal splits,
  rolling backtests, naive baselines and correct forecasting delivery interface.
- P9.3 Anomaly/clustering adapter: task-defined utility, optional labels, stability
  and baseline protocols; internal cluster metrics alone do not certify usefulness.
- P9.4 Text/vision frozen-embedding adapters before broad fine-tuning. Budget model
  downloads/licensing, caching, full preprocessing/encoder inference and privacy.
- P9.5 Small-model fine-tuning: proven accelerator/runtime profile, measured
  sequence/batch/optimizer/activation memory, checkpoint and quantized export parity.
- P9.6 Remote runner: durable remote job IDs, scoped credentials, data policy,
  reservations/cost/shutdown, artifact transfer validation and uncertain-outcome
  reconciliation. A trusted remote evaluator is a separately reviewed boundary.
- P9.7 Warm worker experiment: compare disposable containers vs pooled isolated
  jobs; certify per-job process/filesystem/cgroup isolation, cancellation, reset,
  memory accounting and import/fork/thread compatibility. No privileged fallback.
  Enable only if measured savings justify increased complexity.
- P9.8 Bandits/offline RL last: logged propensities/support assumptions, OPE limits,
  environment adapters and task-specific safety review; no real-world policy control
  or claimed reward improvement from offline metrics alone.
- P9.9 Optional drift reports/retraining proposals: task-calibrated thresholds,
  privacy/count windows and explicit lifecycle policy. PSI >0.20 is not a universal
  retraining trigger; changed training still needs a fresh valid evaluation protocol.

**Gate per adapter:** complete fit/predict/evaluate/package vertical slice, negative
scientific cases, resource-failure recovery, locked supported environment, delivery
parity and declared limitations. Unsupported task types remain rejected.

## Recommendation priority

Ship model rescue and transparent evidence first; they distinguish the product
without needing exotic ML. Next add reproducible environment profiles, better
diagnostics and compute estimates. Add out-of-core/time-series/frozen embeddings
before small-LLM/RL ambitions. Benchmark MCTS, warm pools and DB sharding as optional
optimizations; do not make them prerequisites for a useful tabular coding harness.
