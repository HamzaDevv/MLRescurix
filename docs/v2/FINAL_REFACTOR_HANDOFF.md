# Architecture consolidation handoff — 2026-10-23

Prerequisite implementation is described in [PREREQUISITE_REPAIRS_20261023.md](PREREQUISITE_REPAIRS_20261023.md); exact checks are in `docs/evidence/v2/autonomy/20261023-prerequisite-repairs/summary.json`. Read the current branch/worktree first. Historical evidence and the shared ledger are authoritative records, not reusable passing scores. The preceding architecture has not been broadly consolidated in this session.

## Target architecture for the following session

Build one agent controller that chooses actions from measured feedback and delegates scientific semantics to task adapters. Adapters own classification/regression/panel contracts, metrics, valid transformations and topology; they must not implement separate provider, budget or worker-control loops.

Use one execution service for trusted evaluation/refit and isolated generated scripts. It owns lifecycle, admission, deadlines, cgroup/container or explicitly development-only process execution, artifact collection and honest resource receipts. Keep trust levels explicit: trusted evaluator predictions determine scores; generated script metrics are advisory.

Use one governance service for request identities, content-bound outbound authorization, provider route/pricing validation, shared reservations, durable response recovery, idempotent settlement and uncertain liability. Preserve and migrate the current JSON ledger conservatively if persistence changes. Never create a fresh USD 0.30 allowance, double-count historical spend, or treat calculated prices as verified charges. Runtime receipt databases must survive interrupted dispatches.

Bind evidence and delivery to source/config, data/stage, frozen split/topology, candidate specification, measured evaluation, approved audit and model/artifact identities. A delivered model must match the independently measured and approved candidate. Trace IDs and immutable receipts must survive restarts and packaging.

Replace the duplicated K1–K5 control paths in `scripts/run_robustness_campaign.py` with **one contract-driven runner**. Resolve task-specific loading/staging/joins through adapters, then run a common lifecycle: contract validation → hash-bound development inputs → governed controller → trusted evaluation/audit → identity-bound delivery/evidence. Keep public CLI compatibility or document explicit migration; do not retain alternate paid dispatch paths.

## Preserve these repairs

- `development_stage.py`: pinned source/split import, exact selected order, dedicated worker inputs, explicit trusted-import access history. Final isolation is not restored by changing a split.
- `panel_contract.py` and `K5_PANEL_PROTOCOL.json`: exact entities/origins/horizons/observation history/embargo, known-at-origin provenance and explicit missing/unseen behavior. Incomplete contracts fail. The reproducible metadata importer is `scripts/freeze_k5_panel_contract.py`.
- `CampaignBudgetAuthority`: stable terminal request states, lock/fsync publication, idempotent settlement, estimate residuals retained as uncertain.
- Rust dispatch: the live adapter must keep the canonical preserved campaign ledger (no fresh allowance via a new campaign directory); matching sufficient reservation under the same campaign lock, request/scope/store binding, durable response fence, no ambiguous redispatch, exact route without fallback.
- Separate operator inspection/approval; controller staging cannot authorize uploads. Actor labels are not OS authentication. Preserve deployment identity separation as an explicit prerequisite.
- Candidate acceptance uses trusted measurements, audit identity and bundle/model identity; scripts cannot self-approve.
- Environment allowlists, descendant termination/reaping, partial/unavailable measurement labels and bounded script repair locations.

## Blockers that consolidation must retain

K1 final contamination/regenerated history remains. K1/K2 lack independent absolute quality requirements; thresholds stay null. K4's frozen index-space/gap/training bounds/subset mapping lack authoritative provenance. K5 now has a complete source-bound observation-calendar/entity protocol using deterministic calendar predictors only; full-scale resource feasibility and repaired-revision model quality remain unverified. Preserve its manifest row/index counts and do not substitute the old June-only history or publication-unknown covariates. Provider charge verification lacks invoices/charge receipts, even where local bookkeeping matches. Future outbound requests lack operator authorization. Current model availability/pricing has not been checked in this offline session.

Linux Docker allocation/OOM/recovery fixtures passed; full campaign production scheduling, per-job CPU/resource measurements, concurrent isolation, required fault/soak coverage and clean-room delivery remain unverified. The cached scientific image also lacks `jsonschema`; worker metadata now declares the tested `jsonschema==4.26.0` and `psutil==6.1.1`, but the matching hashed Linux wheels/lock and rebuilt image remain unavailable offline, preventing the separate bounded-tmpfs scientific publication test from starting. Preserve every limitation; source cleanup cannot retroactively validate historical benchmark observations.

## Suggested implementation sequence

1. Map duplicated controller, transport, execution, adapter and evidence responsibilities. Define service contracts and identities before moving code. Add meaningful compatibility/recovery tests around those boundaries.
2. Route existing controller actions through shared execution/governance interfaces, preserving safety gates and existing public behavior. Migrate persistence only with before/after ledger equality and crash/replay tests.
3. Implement classification/regression/panel adapters and a single benchmark runner. Block incomplete contracts before initializing paid work. Remove obsolete bypass paths as they become unreachable.
4. Validate offline provider fixtures, real local worker failures, concurrency, crash boundaries, resource receipts and standalone delivery. Update progress and continuation documents with exact results. Commit and push validated changes.

## Paste-ready prompt

```text
Work in /Users/hamza/Developer/Imp_Projects/MLRescurix. Read applicable AGENTS.md, docs/v2/FINAL_REFACTOR_HANDOFF.md, docs/v2/PREREQUISITE_REPAIRS_20261023.md, POST_LIVE_REVIEW_20261023.md, ROBUSTNESS_PROGRESS.md, SESSION_PROMPT.md, AUTONOMY_IMPLEMENTATION_20261022.md, ROBUSTNESS_MASTER_PLAN.md, BENCHMARK_SUITE.json, the latest prerequisite summary, and the existing shared campaign ledger. Inspect the branch/worktree and preserve subsequent work and all historical evidence.

Perform the final architecture consolidation: one agent controller, task adapters, one execution service, one governance service, and identity-bound evidence/delivery. Replace duplicated benchmark control paths with one contract-driven runner. Follow the handoff's migration sequence and preserve staging, frozen split identities, panel validation, independent outbound approval, conservative shared-budget recovery and trusted candidate acceptance. Do not weaken gates to pass tests or erase K1 contamination.

Use offline tests only. No paid provider calls, five-task live campaign, final scoring, or provider availability/pricing claims. The total provider authority remains USD 0.30 across every session/retry/uncertain outcome; read the ledger, previously USD 0.050724020 committed and USD 0.249275980 recorded remaining. Never reset it. Preserve deepseek/deepseek-v4.1-flash-1017 through OpenRouter inference-net/fp8 with no fallback; future live use requires current availability/pricing verification and explicit outbound authorization. Never expose credentials or commit datasets, runtime databases or model binaries.

Keep external scientific, billing, authorization and production-evidence blockers explicit. Implement meaningful regressions and relevant Python/Rust checks; distinguish implemented, verified and blocked work. Update durable progress/continuation docs, commit and push validated changes, and report exact validation counts, commit/push result and remaining limitations. Do not claim production certification from offline tests.
```
