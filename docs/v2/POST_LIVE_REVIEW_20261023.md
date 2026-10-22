# Post-live independent review — 2026-10-23

The Antigravity report contains useful development observations, but its all-five-pass, zero-physical-holdout-access, exact-billed-cost, and production-governance conclusions are not established. This review supersedes those conclusions. Original run evidence and campaign ledger are preserved without rewriting historical metrics.

## Definite findings and repairs

- The runner accepted locally chosen thresholds instead of the locked benchmark policy. Quality gates now read the policy. K1/K2 have null thresholds and remain inconclusive until acceptance criteria are independently frozen. K3/K4/K5 use 0.75 / 0.85 / 0.55 respectively.
- K4 silently generated replacement temporal folds when frozen manifest mapping failed. This fallback is removed. Missing index space, temporal bounds, or gap now blocks; overlapping/duplicated row folds fail validation. The current K4 manifest requires repair and independent revalidation; its old pass does not validate corrected execution.
- Candidate acceptance now requires matching measured evaluation identity and score, approved audit identity, threshold, serialized model identity, and bundle verification together.
- Pricing verification was asserted in a dated policy snapshot. The snapshot again requires current verification before live calls. No new pricing or provider availability is asserted by this review.
- The gateway controller staged and approved its own outbound view. Review repairs separate staging from approval, retain request-bound durable receipts, and label price-derived charges as estimates. An authorized external approval or independently validated scoped policy is required before further live use.
- Scientific model requests must be honored or rejected. Unsupported families, altered weights, and unsupported parameter keys must not be silently replaced or discarded.
- An unavailable feature-importance vector remains unavailable. The critic handles None without manufacturing importances. Coefficients and split gain are distinct measurements; a coefficient magnitude must not be described as tree gain.
- Panel forecasting needs explicit known-at-forecast covariates, horizon and entity validation in addition to chronological splits. Incomplete protocol metadata must block; a generic regression score is not complete forecasting certification.
- The generated-script artifact fallback previously used unrestricted tar extraction. It now copies only regular bounded members, rejects path traversal, links, sparse members and excess bytes/entries, and kills a timed-out artifact stream. Offline adversarial archive regressions cover the boundary.
- Runtime Kaggle CSV copies, generated model bundles, harness databases, and bulky evaluation requests stay local. Versioned receipts, source code, tests, and handoff documents remain reviewable in Git.

## Evidence interpretation and remaining work

The five historical CV observations are K1 0.7980 accuracy, K2 0.1378 log-price RMSE, K3 0.8094 accuracy, K4 0.8826 ROC-AUC, and K5 0.4147 RMSLE. These are observations on the older dirty revision, not a passing campaign on the repaired code. K1 has documented final-partition contamination and a regenerated split. Do not relabel it as a fresh sealed benchmark.

The prior zero holdout counter records no holdout scoring, not proof of zero physical label reads. K1/K2/K3/K5 source files were materialized before development slicing. Reports now distinguish materialized labels from labels used for selection/scoring. Dedicated development-only staging and clean final-partition isolation remain release work.

The preserved shared ledger records USD 0.050724020 committed and USD 0.249275980 remaining from USD 0.30. Newer gateway charges were calculated from registered token rates with rounding; they were not verified provider invoices. Reconcile actual provider receipts before treating these as exact billed amounts. Do not reset the ledger or assume that estimated charges are zero. This review adds USD 0 provider spend and does not score final holdouts.

Campaign budget admission remains outside the Rust gateway receipt transaction; the new receipt table is a replay fence, not an atomic production budget gate.

A Rust transport adapter is not proof that the standalone campaign traverses the full Rust scheduler, per-job reservations, deployment policy, and Linux cgroup containment. Child-process memory watchdogs protect the controller but do not fix a native backend crash. Missing worker RSS/CPU measurements cannot be called successful resource validation. Production certification remains incomplete.

Live evaluation of the repaired revision, adversarial/development perturbations, actual dynamic-script error repair, all required faults/soak tests, and clean-room bundle execution remain necessary after these prerequisites are resolved. Never force extra turns or spend as a proxy for autonomy.

## Validation

Final review commands and results are recorded in `docs/evidence/v2/autonomy/20261023-post-live-review/`. Historical test and campaign logs remain separate. No provider calls are needed for offline regressions.

Final checks passed:

- `python3 -m pytest python/tests -q`: **283 passed, 1 skipped**, 10 warnings, 198.43 seconds. The skip requires an explicitly bounded 64m container tmpfs. Warnings are native-library/dependency convergence or API notices; the earlier LightGBM crash did not recur in this run.
- `cargo test --locked --test agent_m4 --test reflexive_loop_gate --lib`: **60 passed, 2 ignored**. The ignored checks require an explicitly configured digest-pinned live cgroup Docker environment.
- `cargo build --locked -p mlrescurix-harness`: passed after the final source changes.
- Last affected panel regressions: **7 passed**; artifact tool regressions: **10 passed**, including actual local Docker execution.
- Python compilation and whitespace checks passed. Staged files were checked for credential patterns and copied runtime datasets/models/databases; none were found.

These checks validate the offline repairs. They do not rerun or certify the five paid historical benchmark results. No acceptance thresholds or historical metrics were lowered to make tests pass.
