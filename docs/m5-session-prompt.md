# M5 session prompt and reviewer findings

## Current acceptance status (2026-10-12)

The supplied report's claim that every M4 repair is implemented is unsupported.
The reviewer applied concrete fixes rather than requesting another specification.
Preserve all current uncommitted work. M4 is not yet accepted, and its remaining
implementation dependencies must not be silently inherited by M5 experiments.
M5 research/design can start now; experimental execution needs those dependencies.

### Reviewer fixes applied

- Replaced invalid, ignored provider reservation SQL with M0 Claim/Prepare/
  Dispatch/Response/Commit and MarkUnknown/Resolve commands. Admission now checks
  device/project budgets and capacity through the serialized authoritative writer.
  Accounting errors propagate. Exhausted budgets cannot dispatch a provider call.
- Removed malformed synthetic Action records and direct charge overwrites from
  provider telemetry. M0 owns settlement and durable charges. Removed fabricated
  per-token currency prices; currency budgeting without a configured price now
  refuses rather than claiming enforcement. Internal HTTP retries are limited to
  one attempt so retries cannot bypass per-attempt reservations.
- Counted complete serialized requests, including tool schemas/arguments, using
  conservative byte bounds; filtered advertised tools by role; rejected malformed
  tool arguments; aligned retrieval and critic tools with their published schemas.
- Fixed container-relative input filenames, lowered tool deadlines to the actual
  descriptor limit, rejected successful tools with malformed JSON, and replaced
  millisecond-only input/container names. Streaming output now counts all emitted
  bytes and triggers termination instead of silently discarding excess bytes.
- Rejected symlink reads/writes/listing and missing evidence even when a refs row
  exists; context restore propagates hard-bound failures. Research-state queries
  and budget ledger queries fail on SQL errors. Finalization's state update rejects
  concurrent duplicate closure, though full finalization still needs repair.
- Enforced prediction shape, valid string/numeric classification labels, exact
  probability class order, bounds and row normalization. Missing trusted latency/
  memory telemetry and unsupported constraints cannot certify feasibility.
- Corrected the M4 acceptance banner and added focused regression checks.

### Executed reviewer evidence

`cargo test --workspace --locked --test agent_m4 --lib --test foundation -- --nocapture`:
16 M4 tests, 2 library tests, 24 foundation tests passed. Container execution in the
diagnostic fixture explicitly remains UNVERIFIED; a refusal pass is not an execution
pass. `PYTHONPATH=python/src python3 -m unittest discover -s python/tests -p
test_custom_evaluation_review.py -v`: four evaluator-only tests passed. No generated
candidate Python was executed on the host. No full Python suite, M3 search execution,
live-provider expenditure, commit or push was performed by the reviewer.

### Remaining M4 implementation dependencies

1. Delegation still inserts an agent_tasks row; it does not run the specialist,
   await a durable outcome or resume the parent. Add unique authoritative tasks,
   isolated snapshot-based branches, fair supervisor scheduling and actual outcomes.
2. Agent initialization still starts a fresh context/turn counter; agent checkpoints
   are not integrated with M0 restore and lack full intent/spec/approval/fencing
   fields. Persist provider response content before settlement and reconstruct it
   after crashes. Preserve essential user constraints during compaction, rather
   than discarding earlier user messages. Implement protected finalization allowance
   at atomic admission, not merely a displayed 20% subtraction. Integrate currency
   ceilings with configured provider/model pricing, keeping unknown billing explicit.
3. Sandbox image is a bare Python digest, not a verified scientific dependency
   image. Container registration/ownership, lease-aware startup reconciliation,
   artifact export, confirmation of removal and shared supervisor allocation are
   unfinished. Verify managed image contents and test timeout/output-limit/crash
   behavior with Docker; do not infer containment from command flags.
4. Registered tools still accept snapshot directory existence without verifying
   its contents and can use mutable matching branches. Input writes can mutate
   the purported snapshot. Freeze at registration, verify hashes, stage inputs
   separately, pin environment identity, and implement the declared JSON Schema
   subset strictly (including enums/items/bounds) or use a complete validator.
5. Outbound protection still uses CSV pattern scanning; typed policy-approved views
   and the final upload-policy boundary remain unimplemented. Two-line/JSON/source
   text can carry prohibited rows. Do not perform live uploads until policy is enforced.
6. Custom candidates still lack authoritative pre-execution registration and full
   provenance/telemetry. Bind dataset bytes, actual split assignment, trial, source
   and environment; reject invalid evidence before official score publication.
7. Finalization still checks candidate IDs in unbound approval/audit rows and only
   changes state/computes a hash. Implement verified feasible candidate selection,
   source/environment/spec/protocol-bound user approval and independent audit,
   atomic closure/draining, sandbox refit, frozen-artifact feasibility, durable
   sealed-test access and standalone delivery. CLI approve is a trusted local
   operation, not proof of content-bound approval. No automatic Git merge.
8. M3 optimizer ask/mapping recovery, actual split/source/environment identities,
   declared split-aware fidelities and complete retained-evidence pins are unfinished.
   Cache reuse must require verified model AND prediction hashes; empty hashes must
   not substitute existence checks. Managed interpreter presence alone is not a
   ready environment lease. Add crash/race checks for these behaviors.

## Paste into the next Antigravity session

Continue MLRescurix in `/Users/hamza/Developer/Imp_Projects/MLRescurix` for the M5
session. Read `docs/m5-session-prompt.md` and `docs/m4-repair-handoff.md`, inspect
the current diff, and preserve all uncommitted reviewer fixes. The user authorized
progress and fixes: implement the remaining M4 dependencies directly, with real
regression evidence, rather than returning another proposal approval request.
Do not claim M4 accepted from fixture counts or from a changed project state.

Begin M5 official-source research/design in parallel with that work. Develop
matched ablations for feature engineering, calibration and constrained ensembles.
Fit calibration, learned weights and stacking only on disjoint training-only or
out-of-fold development predictions; preserve the protected validation and sealed
test boundaries. Support valid voting/averaging and bounded stacking with explicit
parent lineage. Consider CPU boosting only after verifying official documentation,
dependencies and pinned managed environment compatibility. No GPU without approval.

Make Rust authoritative for admission, budgets, scheduler, leases/fencing,
isolation, evaluation and finalization. Reuse the shared scheduler, register every
experiment/ensemble parent, and measure complete delivered inference cost and
feasibility. Keep scientific choices adaptive, not a mandatory reasoning sequence.
Predeclare a small matched equal-compute comparison against M3, counting search,
training and confirmation work fairly and reporting provider expenditure separately.
An honest no-gain result can deliver the best verified feasible baseline.

Do not execute M5 comparisons until the required M4/M3 paths pass their acceptance
checks. If Docker remains unavailable, continue code, fixtures and M5 design,
report containment as UNVERIFIED, and stop container-dependent execution cleanly.
Request separate concrete authorization only for live-provider spending or final
candidate approval when ready; do not fabricate approvals or prices. Keep research
closed before refit reveals protected validation labels, consume sealed test only
under the durable gate, and verify standalone inference in a clean sandbox.

Keep MLForge/ and Redesigning.md ignored; do not reuse or execute legacy material.
No untrusted/generated host Python, global Python environment changes, commits,
pushes or automations. Existing custom-agent Python tests execute generated modules
on the host: replace that path with sandbox tests rather than blindly rerunning the
full Python suite. Finish with an implemented/unverified evidence map and M5
findings, with exact commands and limitations.
