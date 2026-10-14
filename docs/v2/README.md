# MLRescurix v2 implementation hub

Created 2026-10-14. Planning baseline: `abf1f235030a79f9a606b429e34608431e37603d`.

Build an agent-first ML coding harness that delivers the best verified feasible
candidate discovered within a declared budget, its training/inference code,
weights, scorecard, provenance and independently runnable pipeline. Begin with
tabular ML; expand through evaluated task/backend adapters and compute negotiation.

## Read and continue

1. [Progress and current handoff](PROGRESS.md): authoritative implementation status.
2. [Repository audit and blueprint corrections](AUDIT.md): what exists and what changes.
3. [Phased roadmap](ROADMAP.md): ordered slices, dependencies and acceptance gates.
4. [Target architecture and contracts](ARCHITECTURE.md): boundaries shared by phases.
5. [Decision register](DECISIONS.md): rationale and unresolved choices.
6. [Next-session prompt](SESSION_PROMPT.md): paste into a local coding agent.
7. [Original supplied blueprint](ARGON_BLUEPRINT.md): preserved reference, not acceptance evidence.
8. [Second repository-informed blueprint](ARGON_REPO_BLUEPRINT.md): additional reference reviewed in AUDIT.md.
9. [Argon review packet](ARGON_REVIEW_PACKET.txt): one paste-ready file containing this doc set, both original blueprints, and the review prompt. Regenerate with `python3 scripts/build_argon_review_packet.py` after edits.
10. [Review disposition](REVIEW_RESPONSE.md): accepted additions and corrections to Argon's replacement documents.
11. [Argon planning review](ARGON_PLAN_REVIEW.txt): verbatim returned feedback and proposed replacements.
12. [P1 mutation and effect matrix](P1_MUTATION_MATRIX.md): inventory of all 48 SQLite tables and filesystem mutations.
13. [P2.1 artifact registry design](P2_1_ARTIFACT_DESIGN.md): authoritative inventory, CAS write protocol, and target versioned contracts.

P1 (P1.1–P1.6) is complete for the acceptance scope recorded in `PROGRESS.md`.
P2.1–P2.6 are implemented and verified for recorded tests: schema v10 CAS, artifact inspection queries, durable event replay, four pooled query-only readers, and revision-fenced successor/tombstone commands. The wider P2 compatibility and recovery gate remains open.
M6 remains **NOT READY** until its independent delivery and benchmark gates close.
The progress ledger records runtime work and evidence; this document remains the
roadmap. Existing project handoffs remain the authority on M6 release evidence. This roadmap supersedes the
supplied blueprint's suggested implementation order and literal SQL/prompt examples;
it does not change current capability or release claims.

## Phase map

| Phase | Outcome | Dependency |
| --- | --- | --- |
| P0 | Reviewable plan and durable continuation docs | Complete in this session |
| P1 | Recoverable, bounded control plane and M6 prerequisite repairs | P0 |
| P2 | Versioned artifact registry and durable event contracts | P1 |
| P3 | Global CPU/RAM/resource lease broker | P1; P2 for telemetry publication |
| P4 | Generic coding tools and executable candidate contract | P2, P3 |
| P5 | Durable agent collaboration and bounded context | P4 |
| P6 | Transparent local dashboard and human collaboration | P2–P3 for read/resource views; P4–P5 for new candidate/agent semantics and steering |
| P7 | Standalone Apex delivery and release evidence | P1–P6 |
| P8 | Measured feasibility estimator and local alternatives | P3, P4, P7 |
| P9 | Task adapters, remote execution and optional warm workers | P8; separate gates per extension |

P1–P7 form the agent-first tabular release. P8 adds resource-aware planning. P9
expands scope selectively; “universal” describes extensible contracts, not a claim
that every ML problem fits on a laptop. Implement one small slice at a time and
update progress, decisions and evidence before ending every implementation session.
