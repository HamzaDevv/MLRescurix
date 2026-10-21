# Review baseline and validation evidence — 2026-10-21

Scope: architecture/report review, persistent planning and existing local regression
checks. No runtime implementation changes, provider requests, Kaggle downloads or
new final holdout evaluations occurred.

Inspected source commit: `5be287b1abf4d3e3140510f4a633072e1519f820`. Working tree was clean at the start of review.
Environment and source-report hashes: [preflight.json](preflight.json).
Gemini Argon arithmetic checks: [argon-review-checks.json](argon-review-checks.json).

## Actual checks

| Command | Exit | Result | Log |
| --- | --- | --- | --- |
| `PYTHONPATH=python/src python3 -m pytest python/tests/test_p13_dirty_sanitizer_and_eda.py python/tests/test_p14_primitives_and_ablation.py python/tests/test_p15_model_zoo_and_forensics.py python/tests/test_p16_critic_and_reflexive_loop.py -q` | 0 | 44 passed; 9 warnings; 53.33 s | [Python baseline](python-baseline.log) |
| `cargo test --locked --test reflexive_loop_gate -- --test-threads=2` | 0 | 8 passed; 0 failed/ignored; 50.30 s | [Rust baseline](rust-baseline.log) |
| `docker info --format '{{.OSType}} {{.CgroupVersion}}'` | 1 | Docker daemon inaccessible | [Environment preflight](preflight.json) |

The Python warnings include physical-core discovery and logistic solver convergence.
An installed pytest async plugin also emits a configuration deprecation. These tests
are a scoped baseline; they do not certify all repository tests, the new review
findings, worker maximum memory, CPU utilization or real-provider behavior.

## Current limitations

Kaggle client and checked credentials are absent. Official competition data/rules,
readable evaluation definitions, dataset hashes and final split commitments still
need preparation. A provider key file exists but was not read, printed or validated;
model availability/pricing and outbound-view approval are unverified.

Argon's proposed unsigned gain formula was checked arithmetically: RMSE 10→9 yields
a negative result, whereas the corrected minimization direction yields a positive
benefit. This is an arithmetic review, not a runtime metric-gate fix.

## Durable next action

Follow [the current session prompt](../../../../v2/SESSION_PROMPT.md) and start S1:
reproduce/fix the EDA isolation, feature-callable label and topology propagation
findings; implement the resumable test runner and typed action contract. See the
[ledger](../../../../v2/ROBUSTNESS_PROGRESS.md) for the full issue register and
[master plan](../../../../v2/ROBUSTNESS_MASTER_PLAN.md) for S1–S5 gates.
