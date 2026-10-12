# M4 implementation handoff

This document replaces the outstanding review loop with an implementation task.
The user authorized proceeding and reviewer fixes on 2026-10-12. Implement the
remaining M4 repairs below; do not request another approval of this specification.
M4 acceptance, live-provider expenditure, candidate finalization approval, and M5
are separate boundaries. Do not describe M4 as complete until its evidence gates
are satisfied.

## Work already applied by the reviewer

- Removed `ToolRegistry::invoke_tool`'s host-Python fallback. Unavailable sandbox
  now refuses before workspace writes; execution failures also clean up the
  temporary input file.
- Tool registration rejects conflicting descriptors for the same project/name/
  version, with identical registration remaining idempotent. All descriptor
  fields are compared; the original record is preserved.
- Added a deterministic refusal test and registration/conflict assertions.
  The diagnostic fixture reports unavailable container execution as UNVERIFIED.
  These changes do not implement immutable snapshot execution or schema validation.
- Marked the existing M4 document as incomplete rather than accepted.

Reviewer verification on 2026-10-12:

```text
cargo test --workspace --locked unavailable_sandbox_refuses_tool_before_any_state_or_workspace_write -- --nocapture
  1 passed; remaining tests filtered out.
cargo test --workspace --locked --test agent_m4 m4_agent_authors_tests_and_reuses_diagnostic_tool_and_adapts_decision -- --exact --nocapture
  1 passed (registration/conflict/refusal assertions).
  UNVERIFIED: diagnostic execution/adaptation requires an available container runtime.
docker version --format '{{.Server.Version}}'
  exit 1: Cannot connect to the Docker daemon.
git diff --check
  exit 0.
```

No full Rust/Python suite, generated Python execution, live-provider call, commit
or push was performed by the reviewer in this repair turn.

Preserve these uncommitted changes. Do not commit or push without user instruction.
Keep `MLForge/` and `Redesigning.md` ignored; do not copy, import, execute, migrate,
or publish legacy material. Do not modify global Python environments, run generated
code on the host, create automations, or advance into M5 in this repair task.

## Implement in this order

1. **Authoritative admission and recovery.** Integrate provider calls, tool runs,
   delegated tasks, and custom experiments with existing M0 actions, generations,
   leases, reservations, charges, receipts, and restore paths. Preserve prepared,
   dispatched, responded, terminal outcomes and the explicit outcome_unknown
   branch. Deduplicate publication and reconcile uncertain work before retry.
   Every provider attempt consumes its own allowance; persist successful usage
   and conservative ceiling settlement for unresolved attempts, distinguishing
   measured usage, estimates, and unknown remote billing. Fail closed on admission
   or accounting errors. Make query_budget report current remaining allowances.
   Existing M1 already atomically reserves job wall_ms and capacity at project
   and device scopes; it is not capacity-only. Reuse reservations/charges, not
   imaginary existing project_ledger/device_ledger tables. Add protected
   finalization allowance alongside the existing soft wall-budget guard.
   Hierarchical session/trial reservations must transfer or encumber allowances
   atomically, settle actual consumption, release unused allowance, and avoid
   double counting at both device and project scopes. Include finalization work
   in the shared scheduler and accounting.

2. **Real sandbox execution and tool contracts.** Use a managed scientific image
   pinned by digest, matching Python >=3.13,<3.14 and the repository dependency
   locks; a bare pinned Python 3.11 image is insufficient. No network or host
   credentials, non-root UID, dropped capabilities, no-new-privileges, immutable
   source/input mounts, bounded scratch/memory/CPU/PIDs/wall time, streaming
   bounded stdout/stderr, and bounded artifact export before container removal.
   Allow training labels only in training jobs; validation/test inference receives
   features and stable row IDs without protected labels. Reject symlink/path
   escapes for reads, writes, patches, snapshots, execution and artifact export.
   Register ownership and execution identity durably before launch; use runtime
   labels plus leases/generations to reconcile create/start/crash races. Cleanup
   may stop only owned stale/fenced containers: NEVER kill every mlr-prefixed
   container on startup, since another healthy supervisor may own it. Keep
   reservations/capacity until stop is confirmed, preserving uncertain accounting
   where execution/publication is unresolved. Implement cancellation and cleanup
   without orphaning daemon-side containers. Preserve fail-closed refusal.
   Freeze and verify the tool source snapshot/environment, validate input before
   invocation and output after invocation, enforce role permissions at dispatch,
   and provide implemented retrieval/audit tools to the appropriate roles. Avoid
   shared mutable .tool_input.json inputs for concurrent invocations.

3. **Usable agent loop and durable context.** Wire explicit provider configuration
   and a bounded turn loop into agent-run; fixtures remain an explicit offline
   mode, not the production default. Actually schedule isolated specialist
   branches/tasks and ingest durable outcomes at turn boundaries. Research stays
   adaptive; no mandatory Lead -> Diagnostic -> Model -> Critic scientific chain.
   Reconstruct checkpoints through M0 restore with all current Context/Checkpoint
   fields, model/environment identities, user intent and constraints, approval
   state, pending questions/actions, handoff version, generations, event cursor,
   jobs/calls, next action/stop conditions, and durable evidence identities.
   Verify artifact contents and typed semantic evidence links, not existence.
   Reload budgets from authoritative state. Bound the entire serialized provider
   request (tools, messages, retrieved evidence and completion reserve), with
   provider-aware counting or conservative bounds. Latest four turns and
   achievement summaries are preferences, not exemptions from the hard bound.
   Spill historical content to durable retrieval; if essential intent cannot fit,
   refuse oversized dispatch rather than dropping constraints or approvals.

4. **Outbound policy and custom evidence.** Enforce project upload policy at the
   final outbound boundary, including messages, receipts, tool arguments and
   source/code text. A typed wrapper is not evidence that arbitrary agent text
   contains no rows. Summary statistics, class counts, column names and slices
   also require policy authorization; typed does not mean public. Agents cannot
   self-authorize views or upload waivers. Preserve policy-approved user intent
   and coding context while preventing raw row leakage under prohibited policy.
   Register CustomCandidateSpec in the Rust experiment/action ledger before
   sandbox fitting. Bind dataset bytes, actual split assignments, code snapshot,
   environment and trial identity. Require exact row IDs, task-dependent shapes,
   class alignment, finite regression/probability values, valid classification
   labels, and normalized probabilities. String class labels are valid; do not
   require every prediction to be numeric. Malformed/provenance-invalid evidence
   is rejected and receives no official score. Structurally valid evidence that
   exceeds constraints is scored as infeasible. Trusted supervisor telemetry
   measures fit/reload/full pipeline prediction time and peak resource usage;
   candidate self-reports cannot certify feasibility. The evaluator neither
   imports candidate code nor unpickles its artifacts. Measure the complete
   runtime bundle, including source, preprocessing, weights and dependencies
   under a declared accounting convention.

5. **Finalization and M3 durability.** A critic audit plus a user approval receipt
   must bind the candidate, exact source/environment, declared constraints,
   evaluation protocol and relevant spec/intent revisions. Approval is created by
   a trusted user-facing path, never an agent or a fabricated token. Atomically
   close research and fence/cancel/drain exploratory work before refit exposes
   validation labels. Preserve permanent closure on failure. Refit on train plus
   validation in isolation, freeze the actual artifact, and recheck measured
   feasibility before reserving/consuming the sealed test gate. Durable test
   access transitions prevent a crash from reopening or repeatedly evaluating
   the test. No automatic Git merge. Deliver the approved source, locked runtime,
   full inference pipeline, lineage, scorecard, hash manifest, input schema and
   sample verification data; independently verify standalone inference in a
   fresh sandbox. Honest no-improvement can deliver the best verified feasible
   baseline under the declared protocol; no invented gain, seed or slice gates.
   Complete M3 ask-to-experiment mapping and optimizer recovery, durable wall
   deadlines, content-verified cache reuse, managed-environment refusal instead
   of host fallback, dataset/actual split/source/environment identity, declared
   split-aware training-only fidelity, finalist confirmation and retained-evidence
   GC pins. Time/group splits require their own fidelity rules; generic random
   stratification must not violate them.

## Acceptance and evidence

Implement meaningful positive, rejection, crash and race checks for each contract.
Cover immutable conflicting registrations, actual snapshot-content verification,
schema/RBAC errors, symlink escapes, output/disk bounds, timeout/orphan cleanup,
healthy-owner preservation, provider uncertainty/retries/accounting failure,
checkpoint reconstruction after compaction and steering, invalid custom evidence,
forged/missing/stale approval, concurrent duplicate finalization, pre-test frozen
artifact infeasibility, sealed-test crash recovery, cache corruption, search
optimizer recovery, deadline persistence and shared-budget races.

Exercise ten active projects with three isolated researcher branches each under
four shared workers. Check real scheduled work, bounded allocation, branch/data
isolation, progress/fairness, cancellation and recovery, not just directories.
Use scripted provider fixtures for local regression without external calls.
Container-dependent tests must distinctly report UNVERIFIED/skipped if Docker is
unavailable; passing refusal tests do not certify isolation or adaptation. Do not
run legacy/custom host-execution tests as a substitute. Record exact commands,
outputs, counts, environment identities and limitations. The quoted earlier
44-Rust/20-Python runs are supplied claims, not reviewer-verified current results;
`unittest discover` without a listing mechanism executes tests, not just discovery.

After local repair checks, report a compact implemented/unverified evidence map
and the remaining operational gate. Do not send another specification approval
request. A live-provider acceptance run requires separate user authorization with
specific provider/model, enforceable aggregate token and currency ceilings,
attempts across all specialists/retries included, and no invented pricing. The
previous 50,000-token/$0.50 estimate is not authorization or a universal price.
Prove evidence-driven hypothesis adaptation and an authenticated finalization
approval; until then distinguish fixture acceptance from live acceptance.

## Prompt for the next Antigravity session

Continue implementation of MLRescurix at
`/Users/hamza/Developer/Imp_Projects/MLRescurix`. The user has authorized proceeding
with M4 repairs; implement instead of returning another proposal for approval.
Read `docs/m4-repair-handoff.md` as the authoritative repair task, inspect current
uncommitted reviewer changes, and preserve them. Complete its ordered repairs and
acceptance checks without generated/untrusted host Python execution, global
environment changes, legacy reuse, commits, pushes or M5 scope. Keep scientific
research adaptive. Report actual executed evidence and explicit unverified gates;
do not claim M4 complete from fixture passes. Ask only for genuinely separate
live-spend or candidate-finalization authorization when concrete work reaches that
gate. M5 planning starts after M4 acceptance, not as a substitute for these repairs.
