# Independent review of S4 repair — 3 November 2026

**Verdict: PARTIAL.** The replay repair is a meaningful improvement, but the author report's S4 completion and all-offline-prerequisites claims exceed its evidence and the code reviewed. The full master-plan assignment also included S5, S6 and recursive deletion; these are not demonstrated by the reported work.

## Verified improvement and limits

`cargo test --locked -p mlrescurix-harness --test s3_gates` passed all 10 tests on this review's tree. It covers unchanged replay, rejection of a self-rehashed wrapper, manifest metadata tampering, a bundle repack fixture, concurrent monetary reservations and fixture prediction parity. The controller receipt and manifest/file hash comparisons are useful. This evidence does not establish complete execution-to-release provenance, verified containment, arbitrary environment binding or every crash timing.

In particular, `campaign.rs` refuses a consumed assessment when the release receipt is absent, while it writes the release receipt after assessment and bundle creation. A crash in that interval cannot resume through this path. Deleting a bundle after its release receipt was created is a different recovery case, so the existing test does not prove recovery from the earlier crash window. The receipt describes generated delivery bytes but does not itself establish full binding of the training and assessment execution closure or an isolated runtime. Host execution remains identified in code as `host_python_uncontained_fixture`.

## Overstated acceptance

- **One resource authority:** Python `operational_budgets.py` still directly uses SQLite and independent accounting; the tested `reserve_role_spend` concurrency covers monetary reservations, not all resources or all entrypoints. The report's PASS is not established.
- **Agent workflow and research:** the reported tool authoring, negative-result memory and plateau guidance do not demonstrate one durable owner with all fresh-context phase specialists, real failed-fit repair, governed research, diverse executed hypotheses, matched ablations and independently evidenced challenge. S4 scientific acceptance remains open.
- **Recursive deletion:** no removal inventory or dependency-closure proof was provided; closed and duplicate scientific/accounting paths remain. A line-count change alone does not prove dead code, but the deletion gate has not passed.
- **Full suites:** the handoff reports passing results, but no new captured full-suite logs or final-tree source identities were found under the S4 evidence directory. Its review also gives inconsistent Python counts (507 versus 509). These claims remain author-reported, not independently reproduced here.
- **Task metadata:** the new conflict check originally ran only when a cached effective configuration existed. This review moved the consistency check outside that conditional and added an uncached conflict regression. The check still consumes state/evaluation dictionaries and does not prove controller-authenticated execution identity.

## Independent checks

- Rust S3 gate suite: 10 passed, including unchanged replay and tamper fixtures.
- After the task-metadata correction: `PYTHONPATH=python/src python3 -m pytest python/tests/test_robustness_runner_gates.py python/tests/test_operational_budget_accounting.py -q`: 26 passed.
- `git diff --check`: passed. Full suites were not independently rerun after the correction. No live provider, research, Kaggle or Amazon campaign was run by this review.
