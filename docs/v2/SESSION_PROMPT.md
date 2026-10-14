# Paste-ready continuation prompt

Use this in a fresh local coding session. Replace `[PHASE_OR_TASK]` with the work
you authorize; P1 is now complete. The recommended next slice is a P2.1 artifact registry/version contract design and read-only inventory.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix.
Authorized scope for this session: [PHASE_OR_TASK, e.g., P2.1 artifact registry/version contract design].

Read docs/v2/README.md, PROGRESS.md, AUDIT.md, ROADMAP.md, ARCHITECTURE.md
and DECISIONS.md. Then read docs/m6-completion-status.md and the relevant
source/tests. The Argon files in docs/v2 are archived suggestions; the corrected
roadmap governs implementation. Inspect git status and preserve prior user work.
Check applicable repository instructions before edits.

Implement the authorized slice directly after a concise scope/acceptance recap.
Do not ask again for routine reversible fixes within that scope. Do not implement
every later phase. When scope is missing, establish the next concrete slice from
PROGRESS.md and obtain phase authorization before runtime edits.

Preserve authoritative scoring, split/label isolation, final-test single use,
state-bound finalization, epoch/intent fencing, durable retry fences, offline
training, approved digest-pinned scientific environments and prohibition on
candidate execution on the host. Generic coding tools still require resource
leases. Unknown effects must reconcile; never reset-and-replay arbitrary commands.

Read docs/v2/REVIEW_RESPONSE.md for the latest Argon review disposition. Default
sandbox execution uses /workspace:ro with task-specific bounded scratch/outputs;
source edits use workspace_fs or a validated staging diff. Read-only source alone
does not make an interrupted job replayable. Verify exact Docker container identity,
state-root ownership and lease epochs before survivor adoption/cleanup and release.
For executable candidates, isolate per-fold inputs/fit state, record prior label
exposure, and enforce rescue provenance and diagnostic/query budgets. Preserve
OutboundFilter, typed outbound projection and approved-view controls; validate,
hash and count the actual provider payload after sanitization.

Use additive migrations from the actual current schema version, update every
version guard/asset and test populated-state upgrade. Keep compatibility adapters
until replacement vertical slices pass. No live provider/cloud spending, data
transfer, commits, pushes or deployment without applicable authorization.

Synchronize authoritative schemas, generation/checking and both Rust/Python
validators in the owning slice. The current scripts/generate_contracts.py handles
v1 worker frames/fixtures; extend it or add versioned tooling as needed. Do not
overwrite contracts/v1 semantics with incompatible v2 contracts or assume all new
artifact/candidate contracts require a worker-frame change.

Run meaningful tests appropriate to this slice. Scientific Python fitting,
inference and parity tests run only inside the approved pinned Docker image.
Existing Python tests use unittest. Preserve historical evidence and use fresh
docs/evidence/v2/<phase>/<session>/ logs; do not overwrite M6 evidence accidentally.
Report unsupported required constraints as refusal, not success.

Before ending, update PROGRESS.md with task statuses, exact changes/tests/results,
evidence identities, limitations and next slice; update DECISIONS.md when design
changes. Report implemented/verified/unverified separately. Do not claim phase
completion until ROADMAP.md's acceptance gate passes. Leave a self-contained
handoff that a fresh coding agent can resume without this conversation.
```
