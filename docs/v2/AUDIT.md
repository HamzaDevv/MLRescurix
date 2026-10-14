# Repository audit and blueprint reconciliation

Static review on 2026-10-14 at `abf1f235030a79f9a606b429e34608431e37603d`.
The checkout was clean and its origin matches the user-supplied GitHub repository.
This audit read local source and historical evidence; it did not rerun runtime,
Docker, provider or benchmark suites. Historical passes are not fresh certification.

## Findings that affect implementation

| Source | Verified observation | Plan consequence |
| --- | --- | --- |
| `docs/m6-completion-status.md` | Five release gates remain open, including installed full delivery and matched benchmarks | Carry all five gates into P1/P7; the blueprint's Phase 1 covers only portions |
| `storage.rs`, `execution.rs` | Runtime has a bounded writer thread, but execution and agent modules also open writable SQLite connections | Inventory and converge authoritative mutations; do not assume a single writer already owns every write |
| `execution.rs::migrate`, `storage.rs::Runtime::open` | Latest supported database version is 9 | Allocate new migration numbers when implementing; update every version guard and embedded migration asset |
| `contracts/v1/`, `scripts/generate_contracts.py`, `contracts.rs`, Python `protocol/` | Generator writes v1 frame schema to Rust-facing/Python locations and shared adversarial fixtures; Rust decoding is separately implemented | Extend versioned generation/checks and both validators; preserve v1 compatibility rather than placing every new v2 schema into v1 |
| `agents/gateway.rs::OutboundFilter` | Provider adapter sanitizes message content/tool arguments heuristically; schema fields are passed through | Extend typed outbound projection, approved-view policy and coverage for tool/artifact/note surfaces; do not claim heuristic redaction guarantees confidentiality |
| `scientific/calibration.py`, `ensemble.py`, `scheduled_pipeline.py`, `search/adapter.py` | Calibrator accepts OOF arrays; ensemble inference uses model objects; scheduled fitting and search dispatch are recipe-specific | Preserve prediction-table calibration, introduce executable-parent/composite and search-space adapters, and verify full pipeline cost/parity |
| `schema_m6.sql`, `search.rs`, `scientific/workflow.py` | SQL recipe enum is fixed; Python built-in fitting/parameter dispatch is also recipe-specific | Removing SQL CHECK alone cannot support arbitrary candidates; add a candidate adapter end to end |
| `agents/tools.rs` | File tools, sandbox Python, agent-authored registered tools, scheduling and delegation already exist | Introduce generic aliases/adapters gradually; preserve lifecycle/delegation controls and useful project tools |
| `custom/contract.py`, `custom/preflight.py`, `custom/runner.py` | Custom-estimator/transformer class contracts exist | Add executable-directory support beside existing contracts; retire old support only with compatibility evidence |
| `agents/sandbox.rs::SandboxRunner::execute` | Docker script execution uses a read-only workspace and writable bounded scratch; default image is `python:3.13-slim` | Do not assume writable workspace or universal digest pinning; require approved pinned profiles for scientific jobs |
| `execution.rs` | Worker/thread capacity and several BLAS environment limits already exist; unsupported dimensions are rejected | Extend admission to memory/accelerators/storage and route every runner through it |
| `agents/recovery.rs` | Durable file intents/outcomes and branch lock exist; unknown effects refuse replay | Add authoritative reconciliation for non-file effects; preserve refusal where evidence is absent |
| `agents/context.rs` | 16,384-token default, 4,096 reserve, checkpointing and compaction already exist; estimator uses bytes/4 | Improve accounting and summaries rather than replacing the entire memory system |
| `scientific/dashboard.py` | Session authentication, Host checks, command revisions, snapshots and EventSource already exist | Rust service replaces a working event bridge; preserve auth/revision semantics and fix unsafe HTML rendering |
| `bundle.rs`, `scientific/bundle.py`, `workflow.py` | Existing state-bound manifest/parity/final-test protections | Extend delivery versions without creating a second trust authority |

Paths in this table are relative to `crates/harness/src/` or
`python/src/mlrescurix_worker/` unless prefixed with `docs/`, `contracts/`,
`scripts/` or `docker/`.

## Keep, revise, defer

**Keep:** agent-authored Python, independent evaluation, resource leases,
artifact-backed transparency, milestone memory, feasibility negotiation and task
adapters. These match the existing product goal and can be added incrementally.

**Revise:**

- Four generic *coding/data* tools are a compact surface, not the entire harness
  API. Delegation, completion, status, cancellation and finalization still need
  governed control commands. Generic tools must not bypass scheduling or scoring.
- A worktree reset cannot undo an arbitrary shell side effect, remote job, SQLite
  mutation or consumed holdout. Reconcile by authoritative receipt; retry only
  isolated replayable operations. Provider dispatch without a saved response can
  have an uncertain billed outcome. Promise reduced duplicate work, not zero waste.
- SHA-256 addresses immutable blobs, not mutable artifact identity. Separate
  records, versions, blobs, references and many-to-many lineage. Scorecards need
  evaluator authority; an agent artifact cannot self-certify official metrics.
- A single `is_apex` column does not enforce one current Apex or explain a Pareto
  set. Use a revisioned project selection record pointing to a verified candidate.
- A new local HTTP service needs authentication, project authorization, safe
  rendering and cursor replay. Loopback binding alone does not supply these.
- Hidden validation labels do not prevent repeated-score overfitting or leakage
  through detailed residuals. Budget adaptive submissions and diagnostics; preserve
  an untouched final test. Data agents may inspect training labels, not raw sealed
  evaluation residuals that reconstruct labels.
- A 128 MiB total-state ceiling conflicts with large weights/runtime assets and
  permanent retry fences. Enforce the current policy first; design explicit
  separate control-state, artifact, environment and export budgets before expansion.
- Ten projects with five scripts each means 50 queued/runnable intents. Fifty
  simultaneously running jobs are conditional on admitted resource demand. Memory
  limits are caps, not reservations of physical RAM; summed demands are a planning
  risk, not proof that container limits allocate 50 GiB immediately.
- Reserve host/VM capacity by measured capabilities and configurable headroom;
  fixed “2 cores + 2 GiB” is unsuitable for every device. Docker's Linux VM budget
  can be lower than macOS host RAM. MPS on the host is not exposed merely because
  Apple Silicon exists.
- CPU thread settings already exist in one path; apply them to all backends.
  Fractional CPU quotas still require at least one library thread and are not an
  integer thread allocation. Required resource claims need measurement provenance.
- Context limits include schemas, message overhead and response allowance. Bytes/4
  is not a hard token guarantee. Compaction must preserve tool-call/result pairs,
  latest steering and pending actions, not merely the earliest 300 request characters.
- Three seeds are configurable repeated runs, not an automatic confidence interval
  or proof of generalization. Choose confirmation protocols by task and budget.
- A formula cannot return an exact GPU-hour deficit. Use bounded pilots and report
  estimate ranges, confidence and assumptions. Unsupervised internal metrics and
  offline RL OPE cannot universally certify downstream usefulness or policy safety.

**Defer:** warm zygote workers, broad remote execution, RL and small-LLM training
until the corresponding capability and scientific acceptance gates pass. Shared
containers enlarge the isolation problem; a child directory and privilege drop
do not establish per-job namespace, process or memory separation.

## External design checks

The following primary sources informed these corrections, accessed 2026-10-14:

- Python 3.13 warns that mixing threads with `fork()` is unsafe. A preloaded
  scientific-library daemon therefore needs a dedicated compatibility experiment:
  [Python os.fork documentation](https://docs.python.org/3.13/library/os.html#os.fork).
- Cgroup `memory.max` is enforcement; `memory.peak` is measurement of the cgroup
  and descendants. A shared container's lifetime peak is not a job-specific
  certificate. Kernel/version/reset support must be checked:
  [Linux cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).
- WAL allows reader/writer concurrency, but long readers can prevent checkpoint
  completion and grow WAL files. Bound read transactions and size the pool from
  measured load rather than hardcoding 16 connections:
  [SQLite WAL documentation](https://www.sqlite.org/wal.html).

## Priority defects and evidence gaps

P1 must investigate non-file uncertain outcomes, universal state growth and
scheduler ENOSPC, supported cgroup measurement, and direct mutation ownership.
P6 must eliminate untrusted `innerHTML` interpolation while preserving the current
session and command protections. P7 must prove installed end-to-end recovery,
upgrade and matched prospective benchmarks. These are design/review findings;
new test evidence must be attached before labeling any repair complete.

## Second Argon blueprint review

The second input claims repository-informed findings and proposes seven tools,
native sandbox fallback, per-project SQLite, Arrow ingestion, MCTS, six roles,
rescue mode and a fifth resource pane. Both original inputs are archived verbatim.

**Adopted additions:** resource matrix in P6.4, explicit rescue slice in P4.5,
optional dataset query/tree convenience tools in P4.1, native-format export and
service recipes in P7, optional training checkpoints in P5.5, Arrow/out-of-core
adapter in P9.1 and drift reports in P9.9.

**Repository claims corrected:**

- `bash_exec` is not the current sandbox tool in `tools.rs`; `sandbox_python` is.
- Dashboard reads `state.sqlite`/Rust snapshots and already uses SSE EventSource;
  it is not just a static `mlrescurix.db` polling page disconnected from Rust.
- Compaction is also called from orchestrator paths and
  `ensure_context_within_bounds()` runs before dispatch. The weakness is token
  estimation/preservation, not total absence of a dispatch bound.
- SandboxLimits values are configurable struct defaults, not proof all real jobs
  have immutable 30-second/512 MiB limits. Measure effective operation-specific limits.
- Branch flock serializes writes in one branch; independent branches can run in
  parallel. Removing it without replacement ownership weakens recovery.
- Embedded runtime assets enable standalone execution. Additional locked library
  profiles do not require abandoning embedded trusted code.
- Actual migration wiring is in `execution.rs::migrate`, with version guards in
  `storage.rs`; renaming a v7/v8 table and wiring only `core.rs` is insufficient.

**Rejected as defaults:**

1. Switching SQLite from FULL to NORMAL while advertising the same crash/power-loss
   durability. WAL NORMAL can lose committed transactions on power loss; keep FULL
   for authoritative receipts unless a separate durability decision changes the
   product contract. [SQLite synchronous documentation](https://www.sqlite.org/pragma.html#pragma_synchronous).
2. Per-project databases before profiling. This splits atomic global resource
   admission across databases and introduces another recovery problem. A semaphore
   hierarchy alone also loses admission state on restart; persist leases.
3. macOS host fallback using setrlimit/RSS polling as equivalent isolation. Resource
   ceilings do not supply label/filesystem/network boundaries, and polling can miss
   peaks. A new backend must independently pass the existing sandbox threat model.
4. Loading candidate `pipeline_entry.py` in a label-bearing trusted evaluator.
   Execute predictions in the untrusted job and score validated outputs separately.
   Candidate evaluation must not repeatedly consume the sealed final lockbox.
5. Arrow as guaranteed “zero RAM duplication.” Memory mapping can reuse input pages,
   but transformations, pandas conversion and training buffers can allocate copies;
   benchmark end-to-end peak use. [Arrow memory interfaces](https://arrow.apache.org/docs/python/memory.html).
6. One BPE encoding as exact counting for every provider. Select provider/model
   tokenizer and overhead policy; reject unsupported hard guarantees.
7. Universal batch/row equality as proof of zero leakage, universal successful
   inference on invalid input, or every ensemble component requiring individually
   positive ablation. These are task-specific checks; ensembles can gain jointly.
8. Guaranteed <3 GiB fine-tuning, four-minute training/2.4ms inference, 8–12x search
   speedups, <40ms worker startup, or automatic drift-triggered retraining. Preserve
   these as proposed experiments, not release promises. Full encoder cost matters.

The second document's six-phase master prompt omits the open M6 release repair
sequence and combines too many incompatible changes. Use SESSION_PROMPT.md with
one ROADMAP.md task scope instead of executing either master prompt literally.

## Review of Argon's planning feedback

The later planning review is archived as `ARGON_PLAN_REVIEW.txt`; its accepted
changes and necessary corrections are recorded in [review disposition](REVIEW_RESPONSE.md).
Local checks confirmed the missing contract/gateway/composite-pipeline touchpoints.
Fold-specific execution, model-rescue provenance, daemon-container identity,
bounded index digests and earlier read-only dashboard slices are incorporated.
The replacement files were merged selectively: broken Google search links,
changed storage-cap scope, blanket replay/privacy guarantees and incompatible v1
schema instructions were not copied into the authoritative docs.
