# Documentation Migration Map

This document records the consolidation of the repository's documentation at baseline commit `441762f`. Redundant, obsolete, and conflicting narrative documents have been removed, with their still-relevant architectural decisions, operational procedures, provenance, and limitations transferred into the consolidated document set.

---

## 1. Consolidated Canonical Documentation Set

The consolidated documentation set resides in [`docs/`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs) and the repository root:

| Canonical Path | Purpose & Scope |
|---|---|
| [`README.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/README.md) | Repository overview, purpose, supported workflows, quickstart setup, and navigation. |
| [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) | Governed dual-tier architecture: feedback controller, task adapters, execution service, governance service, contract benchmark runner, and identity-bound evidence/delivery. |
| [`docs/HUMAN_CONTROL_INTERFACE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/HUMAN_CONTROL_INTERFACE.md) | Common `AgentInterface` API/CLI: status inspection, steering directives, pause, resume, stop, and strict approval boundaries. |
| [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) | Architectural decision register: historical foundational decisions (D01–D50), active consolidation decisions (D51–D60), and superseded decisions. |
| [`docs/DEVELOPMENT_AND_TESTING.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DEVELOPMENT_AND_TESTING.md) | Offline development commands, supported environments, Docker status, and precise validation limits. |
| [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) | Baseline implementation status, campaign ledger values, explicit scientific/environmental blockers, and phased roadmap to production certification. |
| [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md) | Instructions and non-negotiable prime directives for continuing engineering work safely. |
| [`docs/MIGRATION_MAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/MIGRATION_MAP.md) | This document: mapping of removed narrative files and compatibility pointers. |

---

## 2. Migration Table: Removed Documents to Consolidated Destinations

| Removed File | Primary Topics | Where Relevant Content Was Transferred |
|---|---|---|
| `BUILD_PLAN.md` | Early initial M0/M1 planning notes (Oct 12) | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) (D01–D10) |
| `Redesigning.md` | Unstructured conversational transcript | Discarded as obsolete brainstorming; core concept retained in [`README.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/README.md) |
| `SYSTEM_DESIGN.md` | Early design specification (Oct 12) | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) and [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/AGENT_FIRST_BLUEPRINT.md` | Preliminary agent harness blueprint | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) and [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) (D42) |
| `docs/m0-foundation.md` | Milestone 0 narrative | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) (D01–D03) |
| `docs/m1-execution-storage.md` | Milestone 1 execution and storage notes | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) (D04, D22, D23) |
| `docs/m2-scientific-kernel.md` | Milestone 2 scientific kernel notes | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) §2.2 |
| `docs/m3-constrained-search.md` | Milestone 3 search notes | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) (D11, D45) |
| `docs/m4-agent-harness.md` | Milestone 4 harness notes | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) §2.1 |
| `docs/m4-repair-handoff.md` | Early repair handoff | Superseded by [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/m5-session-prompt.md` | Obsolete milestone 5 session prompt | [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md) |
| `docs/m6-session-prompt.md` | Obsolete milestone 6 session prompt | [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md) |
| `docs/m6-release-notes.md` | Historical M6 release draft | Retained in [`RELEASE_NOTES.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/RELEASE_NOTES.md) |
| `docs/m6-review-next.md` | Historical review notes | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/m6-independent-review-2026-10-13.md` | Historical review | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) and [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/reports/LIVE_TEST_REPORT.md` | Outdated live test claims (superseded) | Replaced by accurate baseline in [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/ARGON_BLUEPRINT.md` | Raw Argon blueprint (Oct 14) | Adopted decisions incorporated into [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/ARGON_REPO_BLUEPRINT.md` | Second Argon blueprint (Oct 14) | Adopted decisions incorporated into [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/ARGON_PLAN_REVIEW.txt` | Raw review feedback | Adopted amendments (A1–A5) recorded in [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/ARGON_REVIEW_PACKET.txt` | Transient review prompt payload | Discarded as obsolete |
| `docs/v2/REVIEW_RESPONSE.md` | Response to Argon review | Incorporated into [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/ROBUSTNESS_ARGON_REVIEW.md` | Review disposition | Incorporated into [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/ROBUSTNESS_ARGON_REVIEW_SOURCE.md` | Source document for review | Discarded as duplicate |
| `docs/v2/ROBUSTNESS_REPORT_SOURCE.md` | Source text for report | Discarded as duplicate |
| `docs/v2/ROBUSTNESS_S1_REVIEW.md` | S1 review narrative | Incorporated into [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/ROBUSTNESS_S1_FOLLOWUP.md` | S1 follow-up review | Incorporated into [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/AUTONOMY_IMPLEMENTATION_20261022.md` | Autonomous loop implementation record | Incorporated into [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) and [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/AUTONOMOUS_LOOP_REVIEW_20261022.md` | Review of autonomous loop | Incorporated into [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/K1_TITANIC_RUN_REVIEW.md` | K1 evaluation erratum & leakage review | Facts permanently recorded in [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) §4 and [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md) |
| `docs/v2/ROBUSTNESS_MASTER_PLAN.md` | Master plan for robustness campaign | Replaced by [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/AUDIT.md` | Static audit from Oct 14 | Incorporated into [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/P1_MUTATION_MATRIX.md` | Early mutation inventory | Architectural model summarized in [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) |
| `docs/v2/P2_1_ARTIFACT_DESIGN.md` | Early artifact registry design | Standardized in [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) §2.6 and [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |

---

## 3. Minimal Compatibility Pointers

To ensure existing external links, historical references, and tools do not break, minimal compatibility pointers are preserved at the following legacy paths:

| Pointer Path | Points To |
|---|---|
| `docs/m6-completion-status.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/README.md` | [`README.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/README.md) & [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) |
| `docs/v2/ARCHITECTURE.md` | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) |
| `docs/v2/HUMAN_CONTROL_INTERFACE.md` | [`docs/HUMAN_CONTROL_INTERFACE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/HUMAN_CONTROL_INTERFACE.md) |
| `docs/v2/DECISIONS.md` | [`docs/DECISIONS.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/DECISIONS.md) |
| `docs/v2/SESSION_PROMPT.md` | [`docs/CONTINUATION_GUIDE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/CONTINUATION_GUIDE.md) |
| `docs/v2/PROGRESS.md` & `ROBUSTNESS_PROGRESS.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/ROADMAP.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/ARCHITECTURE_CONSOLIDATION_20261023.md` | [`docs/ARCHITECTURE.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/ARCHITECTURE.md) & [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/FINAL_REFACTOR_HANDOFF.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/PREREQUISITE_REPAIRS_20261023.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |
| `docs/v2/POST_LIVE_REVIEW_20261023.md` | [`docs/STATUS_AND_ROADMAP.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/STATUS_AND_ROADMAP.md) |

---

## 4. Preserved Authoritative Machine Records & Evidence

The following files are authoritative machine-consumed contracts, historical evidence, or runtime configurations. They are preserved in place and **never modified or deleted**:

- [`docs/evidence/**`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence): All historical test and campaign evidence logs, JSON summaries, and traces.
- [`docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json): Preserved campaign ledger (SHA-256 `827abd984dd960f91cd39725d7dcbdb78d47c114322a0358cbbb08a5b30d0b52`).
- [`docs/v2/BENCHMARK_SUITE.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/BENCHMARK_SUITE.json): Machine contract consumed by benchmark runner and tests.
- [`docs/v2/K5_PANEL_PROTOCOL.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/K5_PANEL_PROTOCOL.json): Machine contract consumed by panel adapter.
- [`docs/v2/kaggle_benchmarks/README.md`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/v2/kaggle_benchmarks/README.md): Mathematical formulations of competition metrics.
- [`docs/m5-comparison-results.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/m5-comparison-results.json) & [`docs/m6-comparison-results.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/m6-comparison-results.json): Historical comparison benchmark records.
