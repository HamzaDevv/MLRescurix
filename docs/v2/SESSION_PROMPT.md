# Paste-ready continuation prompt

Current campaign: the 2026-10-21 robustness refinement, advancing to S2.
S0 (planning) and S1 (trust boundaries, reproducers, typed actions, runner repair) are complete and verified.

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
6. docs/v2/ROBUSTNESS_S1_REVIEW.md — audit review of e1c8674 whose findings are now repaired.
Consult existing ARCHITECTURE.md/DECISIONS.md and actual source for unchanged contracts.

S1 repair status: COMPLETE & VERIFIED.
- RB-001/RB-002: Governed execution boundary enforces safe allowlist env, secret stripping,
  isolated worker subprocess execution (zero controller exec), process-group kill on timeout,
  and strict UTF-8 <= 2048 byte Context Firewall.
- RB-003/RB-004: Locked evaluation splits validated for strict train/val disjointness, stable
  temporal sorting, nan-filled unassessed temporal prefix, and eligible-row meta fitting.
- RB-013: Strict Draft2020-12 schema validation, atomic ledger persistence, restart replay.
- RB-014: Forwarded metric_name, seed, and ledger to ablation; baseline_cv_score field used;
  utility-oriented delta calculation (RMSE 10->9 confirms improvement).
- RB-015: Inconclusive receipts for unintegrated control and target transformations.
- RB-016: Resumable runner with atomic writes, provenance/budget preservation, real
  deterministic_dev benchmarks, and fail-closed live_dev/certify modes.
- Prerequisites: Docker daemon reachable (linux 2); Kaggle client installed and credentials
  verified active (HTTP 200, 20 competitions listed) at ~/.kaggle/access_token and .config/kaggle.token (0600);
  OpenRouter key present in .config/openrouter.key ($0 spent, $5 remaining).

Implement the next session slice (S2: Scientific loop and K1/K2 production live):
- Fold-safe candidate feature mutation, model search, stacking, and calibration.
- Task/metric-aware critic policy (address RB-007) and genuine error-forensics delivery (RB-011).
- Official Kaggle data download verification for K1 Titanic and K2 House Prices.
- Production real-provider live solves for K1 and K2 using the gateway/orchestrator/scheduler
  path, under the per-run cap USD 0.50 and campaign cap USD 5.00.
- Clean-room bundle parity verification (Argon A1).

Live tests must use the production provider gateway/orchestrator/scheduler/evaluator
path, current configured model and pricing, and the existing authorized outbound-view
policy. No provider keys in workers. Reserve request cost first; initial campaign cap
USD 5 and per-main-live-run cap USD 0.50, further bounded by existing policy. Never
bypass runtime approvals or silently increase budgets. All limits are in the manifest.

Measure worker-tree/cgroup peak bytes and CPU time/quota, controller metrics separately,
plus disk/I/O, stage latency, provider cost/tokens and event lineage.

Honor Argon A1–A5: fresh-environment bundle imports, existing forensics producer with
bounded mid-loop receipts, legal fidelity within fold training boundaries, complete
pipeline per champion and canonical OOF units/row IDs, direction-aware metric gain.

End every session by updating ROBUSTNESS_PROGRESS.md with exact changes, commands,
exit/pass/fail/skip counts, evidence/trace IDs, issue status, remaining budgets,
holdout accesses and the next smallest action.
```
