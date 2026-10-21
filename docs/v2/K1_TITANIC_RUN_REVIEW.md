# K1 Titanic run review and status correction

Review date: 2026-10-21 (Asia/Kolkata)

## Finding

K1 is **not a valid benchmark pass**. The two live development runs both opened and scored the same 178-row final partition. The first run reported 1.000 CV and 1.000 holdout accuracy. Antigravity then found and fixed target/surrogate-ID leakage in `FeatureEngineeringGraphTransformer`, and the second run reported 0.8243 CV balanced accuracy and 0.7472 holdout accuracy. Because the first run exposed labels/score before the fix, the second score is a repeat use of an already-consumed holdout. Preserve both runs as exploratory history; do not present either as a clean final assessment.

The user's concern about Titanic being a familiar benchmark is reasonable: domain familiarity can make an agent's suggestions less novel. The available evidence cannot quantify that effect. The demonstrable reason to reject the claimed score is the target-leakage defect plus repeated final-partition use.

## Evidence and discrepancies

- `run_20261021_172942/metrics.json` records 1.000 CV and holdout accuracy on 713 development / 178 holdout rows; its trace records three paid specialist requests costing USD 0.00061408.
- `run_20261021_173653/metrics.json` records 0.824311 CV score, 0.747191 holdout accuracy, and 0.742214 holdout balanced accuracy on the same 713 / 178 partition; its trace records three paid specialist requests costing USD 0.00092344.
- `scripts/run_robustness_campaign.py` slices development rows for fitting, then explicitly reads `Survived` from `holdout_indices` and computes holdout metrics. This occurs in both runs. The saved run summaries nevertheless report `holdout_accesses: 0` and `budget_spent_usd: 0.0`.
- The authoritative campaign ledger records USD 0.00153752 billed across the two runs, USD 0.00 reserved/uncertain, and USD 0.29846248 remaining under the USD 0.30 cap. Use the ledger, not the run-summary counters, for campaign spend.
- The live code calls the OpenRouter specialist helper directly and runs the scientific engine in-process. The Rust command uses `verify-bundle --sandbox false`; that is bundle integrity verification, not proof of sandbox isolation or the complete production gateway/orchestrator/scheduler path.
- The Critic's `APPROVED` is advisory; the K1 runner's pass condition is bundle verification plus a fixed CV threshold and does not parse/enforce the Critic decision.
- Reported repository test results are historical Antigravity claims. This review did not rerun them.

## K2 readiness

K2 House Prices is pinned independently. Verified archive, train/test/description/sample-submission hashes and split-manifest SHA256 match `BENCHMARK_SUITE.json`. The split contains 1,168 development rows and 292 final rows; the development folds do not intersect the final partition, and fold train/validation indices are disjoint from the final partition.

Proceed with **one bounded K2 development-only experiment** under the existing campaign authority. Keep the 292-row final partition closed: do not load it, score it, inspect its target, export it to a model/provider, or report a final metric. Freeze the candidate, source, configuration, and selection rule before any separate single-use final evaluation. Keep the campaign cap at USD 0.30 total, charge prior USD 0.00153752, and cap this K2 run at USD 0.05. Use only OpenRouter model `deepseek/deepseek-v4.1-flash-1017`, route `inference-net/fp8`, no fallback.

Before any paid request, fix/verify run-level spend and holdout accounting so saved summaries reconcile with the authoritative budget ledger and any final-partition access is recorded before reading labels. Describe the K2 work as a development experiment; do not call it production certification while RB-008 and cgroup/resource gates remain open.

## Additional fairness audit of the resubmitted report

The resubmitted attachment describes the same runs, not a new clean evaluation. Read-only checks reconfirm all K1 archive/file/manifest hashes match the pinned manifest, 713 development and 178 final indices are disjoint and cover all 891 rows, and each recorded fold partitions the development indices without final overlap. Hash agreement establishes consistency with the local pin, not independent provenance certification.

Further blockers found in source inspection:

- The runner reads `development_indices` but never passes `development_folds` into `execute_reflexive_loop`. That method constructs its own splitter from inferred topology. The claim that the reported score uses the locked five folds is unsupported by this execution path.
- `execute_reflexive_loop` calls `fe_graph.fit_transform(clean_df, y)` before model cross-validation. This graph includes target-aware text selection, categorical encoding and interaction selection. They must fit inside each outer training fold; internal encoding folds alone do not establish outer-fold isolation. The 0.8243 score is not accepted as leakage-free CV.
- Ablation records add fixed constants 0.038, 0.024 and 0.018 to the baseline and assign fixed fold-win counts. Those values are not measured removal/refit experiments. Preserve them as invalid historical evidence and replace this behavior with measured paired fold comparisons or an explicit unmeasured status.
- Diagnostics are quoted into the architecture prompt, but neither response configures the scientific engine. Hypotheses in the improvement epoch explicitly name Titanic columns and interactions. Three paid specialist responses do not demonstrate general autonomous problem solving.
- There are 18 shared `Ticket` values across development/final rows. This requires reconciliation with the promised frozen family/ticket group-disjoint split; nonoverlapping row indices alone do not prove group isolation. No target values were inspected for this identifier check.
- The reported primary development score is balanced accuracy, whereas the frozen K1 primary metric is accuracy. Threshold tuning over 81 choices on the same OOF predictions makes this a selected development score, not an independent assessment.

Revised next action: continue in Antigravity with an **offline evaluation-correction slice first**, before any K2 paid run. Consume the locked folds with explicit original-to-development row mapping, move learned preprocessing inside outer folds, eliminate invented ablation evidence, enforce the task-specific primary metric/direction, and make provider suggestions traceably influence validated execution or label them advisory. Reconcile the K1 grouping contract without resplitting the consumed final partition. Verify these corrections with relevant checks and retain failures. Then proceed to the bounded K2 development-only experiment already specified; keep its 292 final rows closed. No new provider cost was incurred during this review.
