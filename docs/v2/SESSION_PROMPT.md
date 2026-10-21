# Paste-ready continuation prompt

Current campaign: the 2026-10-21 robustness refinement, starting at S1.
The old P9-only/v10 instruction is historical and no longer the continuation target.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix.
The user requested a refined architecture, implementation over 3–5 sessions, five
different Kaggle problems, real-provider end-to-end tests, trace inspection,
maximum memory/CPU monitoring, vulnerability testing and a bounded fix→run→fix loop.

First inspect git status/current branch and read:
1. docs/v2/ROBUSTNESS_PROGRESS.md — actual status, prerequisites and issue register.
2. docs/v2/ROBUSTNESS_MASTER_PLAN.md — current architecture, S1–S5 gates and budgets.
3. docs/v2/ROBUSTNESS_ARGON_REVIEW.md — accepted amendments A1–A5 and corrections.
4. docs/v2/BENCHMARK_SUITE.json — five-task planning manifest, not an executed runner.
5. docs/evidence/v2/robustness/20261021-plan-review/summary.md — scoped review baseline.
Consult existing ARCHITECTURE.md/DECISIONS.md and actual source for unchanged contracts.

S0 is documentation/review only. Existing local checks passed 44 Python scientific
tests and 8 Rust reflexive integration tests. No upgrade fix, five-task campaign,
current real-provider solve or final vault evaluation was completed in S0.
Source baseline reviewed: 5be287b. Storage currently accepts schema versions through
11; inspect actual migrations and never downgrade to v10 to match old handoffs.
Preserve historical evidence, existing user changes and bundle/schema compatibility.

Implement the next unfinished S1 slice:
- Reproduce RB-001–RB-004 with synthetic fixtures: exploratory exec timeout/isolation,
  callable validation-label exposure, topology-safe proxy splitting and shared
  ensemble/full-CV split assignments. Fix the actual reachable paths.
- Add strict typed hypothesis action schema/dispatcher integrated with existing
  primitives, research ledger and paired ablation; retain governed sandbox coding.
- Add a resumable campaign preflight/runner with explicit skipped/blocked outcomes,
  immutable manifests, source-built binary/image provenance and worker resource traces.
- Prepare official Kaggle data and freeze metric/split/quality definitions before
  adaptive search. K1 Titanic, K2 House Prices, K3 Forest Cover Type, K4 IEEE-CIS Fraud,
  K5 Store Sales. Synthetic cases do not count as authentic Kaggle benchmark results.

At S0 Docker CLI could not reach the daemon; Kaggle client/checked credential paths
were absent. A provider key file existed, but validity was unverified and its content
was not inspected. Recheck prerequisites without printing secrets. Continue independent
contract work when external prerequisites are unavailable.

Live tests must use the production provider gateway/orchestrator/scheduler/evaluator
path, current configured model and pricing, and the existing authorized outbound-view
policy. No provider keys in workers. Reserve request cost first; initial campaign cap
USD 5 and per-main-live-run cap USD 0.50, further bounded by existing policy. Never
bypass runtime approvals or silently increase budgets. All limits are in the manifest.

Measure worker-tree/cgroup peak bytes and CPU time/quota, controller metrics separately,
plus disk/I/O, stage latency, provider cost/tokens and event lineage. Linux cgroup v2
is required for memory-enforcement certification; macOS host RSS is not a substitute.

Honor Argon A1–A5: fresh-environment bundle imports, existing forensics producer with
bounded mid-loop receipts, legal fidelity within fold training boundaries, complete
pipeline per champion and canonical OOF units/row IDs, direction-aware metric gain.
The original Argon formula is wrong for minimization; use the corrected specification.

Run fix→reproducer→affected regression within declared retry/trial/time/cost bounds.
Do not use final holdout labels in this loop. Require the five-task/feature/fault
matrix, three matched development seeds and two complete final-revision development
validation passes before S5. Freeze the selected artifact and consume each new
dataset-campaign final vault once. A failed final gate stays failed.

End every session by updating ROBUSTNESS_PROGRESS.md with exact changes, commands,
exit/pass/fail/skip counts, evidence/trace IDs, issue status, remaining budgets,
holdout accesses and the next smallest action. Update the manifest when policy
changes. Advance sessions based on passed gates, not elapsed session count.
```
