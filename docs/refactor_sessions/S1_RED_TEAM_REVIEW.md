# Session 1 — Independent review and bounded fixes

Reviewed November 2, 2026 against the working tree at baseline commit `95a35e9`. The implementation was already dirty; this review preserves that work, the original packet and issued scientific artifacts. This is the review verdict, not another master plan.

**Verdict: useful numerical and specialist scaffolding, but S1 acceptance is not established.** Passing tests are real; several test names and the original A–G PASS matrix claim behavior the tests never exercise. Continue in one major S2 session that first closes these foundation gaps, then implements S2 authority and scientific boundaries. No new design approval is required.

## Findings, in priority order

### 1. Changed model weights can inherit an old assessment score — OPEN, P1

In `crates/harness/src/campaign.rs`, restart checks specialist completion by task ID and reuses `assessment_lockbox_consumptions` by project/dataset, without verifying the current model/source against the stored receipt. In a disposable copy of the original synthetic CLI state, changing the worker model intercept by +100 and removing the bundle manifest produced exit 0, a different weights hash, and the **same old validation and lockbox scores**. The incomplete-bundle recovery test checks only untouched files; it misses this attack.

The completed-bundle fast path also runs before validating the requested protocol. Requesting RMSE and seed 7 against the completed MAE/seed-2026 state returned exit 0 and the old protocol/scores. This is incorrect resume behavior even if the bundle's internal manifest is consistent.

Evidence: [resume probes](../evidence/refactor/S1/red_team/resume_boundary_probes.json). These used copied synthetic state and the rebuilt binary's explicit offline diagnostic mode; original evidence was not modified. S2 must bind immutable source/weights/environment/predictions/protocol/evaluation/selection and reject or reconcile mismatches. Repackaging must never grant old scores to new bytes, nor consume a used holdout again.

### 2. No campaign boss decision loop; fallback conceals authoring failure — OPEN, P1

`campaign.rs` runs a fixed sequence: specialist authoring → fit → evaluation → challenger → assessment → bundle. There is no owner `AgentOrchestrator`, owner provider request or observed-result-driven owner decision at the comment labelled “Boss ReAct Turn.” Specialist constructor/step errors are swallowed; missing authored files trigger `generate_default_solution_template`.

The original saved CLI run did real numerical work, but did **not** run provider-fixture agents: its database has two assigned/completed tasks, **zero `provider_turn_receipts`, zero `agent_checkpoints` and zero `critic_audits`**. Without a gateway, the code generates a built-in basis model and supplies approval locally. These rows do not prove independent delegation or review. A gateway-backed integration test does author code through tools and invoke a challenger; that narrower success is useful.

Evidence: [original CLI database counts](../evidence/refactor/S1/red_team/original_cli_db_counts.json). S2 must close the S1 owner-loop gap, propagate actual execution failures and make fixture/reference behavior explicit. The full S4 exploration policy is still a later gate.

### 3. Unverified isolation enabled by default — BOUNDED FIX APPLIED, full isolation OPEN

The submitted code changed `campaign_isolation_available()` from false to true by default, with an environment flag and public thread override. It still executes `Command::new("python3")` on the controller host. The existing safety test forces the flag false, so it passes while production default behavior is unsafe. Host workers can read controller-private/source files; directory separation does not enforce access separation.

This review restored the production gate: no environment variable or thread override asserts isolation. It added `run_offline_fixture_campaign` and `autonomous-create --offline-fixture` for **explicitly trusted synthetic diagnostics**, rejecting live credentials/nonfixture gateways and ignoring ambient provider credentials. Existing numerical tests now explicitly use this fixture entry point. The normal public campaign entry point remains unavailable until a verified isolated executor is implemented in S2. This fix does not certify containment or autonomy; it keeps numerical diagnostics runnable without making those claims.

### 4. Missing or ambiguous challenger response defaults to approval — BOUNDED FIX APPLIED

The submitted code initializes approval before calling the challenger and uses substring detection. An empty response, unrelated response, or “NOT APPROVED” could finalize and consume assessment. A new boundary test reproduced the empty-response false approval before the fix.

Current fixture behavior requires an explicit `APPROVED: <findings>` or `CHALLENGED: <findings>` response; missing, ambiguous and tool-only responses block finalization. Invalid persisted verdict values no longer default to approval. This is a bounded interim parser; S2 must replace this seam with task/attempt/protocol/evaluation-bound structured audits and durable negative results. A no-gateway reference fixture still has no independent reviewer and cannot establish S1 acceptance.

### 5. Search accounting is not durable admission — OPEN, P1

`OperationalBudgets.record_search_query` increments a dataclass integer. `GovernedResearchClient.research_topic` checks/increments it before search, which improves within-process failure accounting, but **there is no SQLite attempt admission or idempotent settlement in these modules**. Restoring from an empty pre-turn history after one dispatched mock query resets usage from 1 to 0. The enclosing loop refuses an unresolved pending turn, so this probe does not prove it automatically redispatches; it proves the claimed durable query receipt is absent. New controllers also do not share an atomic search authority. `DeepSeekSearchClient` calls the client without a budget object.

The claimed test for timeout assigns `RuntimeError("Network timeout")` as a mock response, while asserting only `("error", "failed")`. `MockSearchProvider` slices that exception object instead of raising it, producing a TypeError that the client reports as failed. The passing assertion therefore checks an incidental mock failure, not the intended timeout path. Failures/timeout/unknown, crash reconstruction, denied-admission zero dispatch and replay idempotency need real assertions. S2 owns this repair through one Rust resource authority, rather than another mutable Python ledger.

### 6. Lease checks do not establish crash recovery or fencing — OPEN, P1

The active-lease SELECT and unconditional upsert are separate operations. Two starters can both observe no active lease and overwrite ownership. There is no attempt epoch checked by task completion writes. RAII releases on ordinary return/unwind, but SIGKILL does not run Drop, leaving an active lease with no expiry/recovery protocol. The new tests insert a pre-existing active row or delete a manifest after successful completion; they do not kill the campaign at a dispatch boundary or race owner acquisition. A separate provider receipt crash test is not proof of campaign owner recovery.

### 7. Effective configuration and deduplication are not the actual execution identity — OPEN, P1

The resolver declares XGBoost `n_estimators=100` but the factory constructs 150 when unspecified. It declares Ridge `max_iter=2000`, while the actual default is `None`. The candidate still hashes requested metadata, including `requested_parameters` inside `effective_configuration`, rather than a canonical executed configuration. Equivalent requested alpha/default-max_iter specs resolve to equal declared parameters but yield different keys. The dedup test compares two dictionaries; it never proves one fit or verified receipt reuse. The full data/protocol/code/environment/fit/fidelity/seed binding is still required.

### 8. Runtime sibling failure still aborts the portfolio — OPEN, P2

The submitted code isolates specification validation, but the evaluation loop calls `_evaluate(one)` without per-member execution handling. A fixture evaluator raising an infrastructure error for the first valid family caused only that family to be attempted; the next valid sibling never ran. Distinguish invalid specification, candidate execution failure, unavailable backend, valid negative result, cached result and integrity/reconciliation failure. Do not swallow integrity failures in the name of continuation.

### 9. Descriptor parity and context evidence are weaker than reported — OPEN, P2

Python loads action names from JSON, silently falling back to a duplicate list; Rust retains its own hardcoded allowlist and the prompt/dispatch have independent parameter handling. JSON descriptors are not the single enforced schema authority. For example, descriptor HANDOFF requires `summary`, while tests successfully send `reason`. The shipped-assets test recreates an ancestor search using the checkout import; it does not load the installed runtime implementation. The context test inspects a database assignment object from a run with no provider, not serialized owner/specialist/challenger requests. S2 must close these existing S1 gates at real boundaries.

### 10. Bundle “verified” does not certify release — OPEN, later S3 gate

The current verifier explicitly reports integrity scope and `scorecard_claims_not_independently_certified`. The package code writes placeholder dependency locks, dummy `.pyc` bytes, copies JSON weights to a `.joblib` name and generates basis-specific inference. CLI-versus-serving agreement on examples generated by that same inference code does not prove agreement with the evaluated candidate. Do not call this a general standalone or certified release. S2 must prevent false evidence binding now; generic packaging/replay belongs to S3.

## Other bounded repair

CSV row width is now checked before feature-index access, so surplus fields cannot panic the controller and missing fields cannot silently produce malformed partitions. Regression tests cover both widths.

## Acceptance assessment

| S1 gate | Independent verdict | What remains |
|---|---|---|
| A | PARTIAL; required boss flow FAIL | Real numerical fitting and a gateway specialist/challenger path exist; CLI owner observation/decision and provider-fixture campaign do not. |
| B | NOT ESTABLISHED | Actual serialized contexts, authoritative evidence refs and attempt bindings were not checked by the cited assignment-row test. |
| C | PARTIAL; recovery/fencing FAIL | Untouched completed-bundle reuse works; owner race/kill recovery, stale completion fencing, changed-spec/artifact binding and usage reconciliation remain. |
| D | PARTIAL | Some parameter mismatch and staging checks pass; runtime sibling isolation, executed config/verified dedup, schema and selection boundaries remain. |
| E | FAIL | In-memory preincrement exists; durable shared admission, outcomes and replay evidence do not. |
| F | PARTIAL PASS | Repeated spec error reaches subsequent Python prompt. This is not canonical receipt reuse or S4 plateau recovery. |
| G | PARTIAL | Several obsolete behaviors removed, but fallback paths, duplicate action authority and temporary orchestration ownership remain unclosed. |

## Verification of this review

Environment observed: Python 3.13.3, cargo 1.94.0. No paid provider calls, live web queries, Kaggle jobs or mutations of original assessment evidence were made. Numeric probes use synthetic fixtures and temporary state copies.

- Before changes: `cargo test --locked -p mlrescurix-harness --test campaign_boss_specialist_slice` — exit 0, 9 passed.
- Before changes: `PYTHONPATH=python/src python3 -m pytest -q python/tests/test_s1_contracts_and_parity.py python/tests/test_governed_research.py python/tests/test_operational_budget_accounting.py` — exit 0, 19 passed.
- New safety/challenger regressions before fixes — exit 101, 2 failed, demonstrating the defects. [Log](../evidence/refactor/S1/red_team/before_boundary_fixes.log).
- After fixes: `cargo test --locked -p mlrescurix-harness --test s1_red_team --test campaign_boss_specialist_slice --test campaign_production` — exit 0, 4 + 9 + 10 passed. [Log](../evidence/refactor/S1/red_team/after_boundary_fixes.log).
- Full workspace result and final source identities are recorded in [review verification](../evidence/refactor/S1/red_team/verification.json). The original report's 491 Python tests were not independently repeated in this Rust-only bounded repair.
- `cargo test --locked --workspace` — exit 0, 348 passed, 17 ignored across 40 suite summaries. Ignored environment/containment checks remain unverified. The final focused red-team run also passed 4/4 after adding missing/None challenger content coverage.
- Additional observed resolver, duplicate-key, restart-counter and runtime-sibling behavior: [Python probes](../evidence/refactor/S1/red_team/python_boundary_probes.json).
- Reproduce the remaining-gap observations with `python3 docs/evidence/refactor/S1/red_team/reproduce_remaining_gaps.py` — exit 0; [script](../evidence/refactor/S1/red_team/reproduce_remaining_gaps.py) and [execution log](../evidence/refactor/S1/red_team/probe_execution.log). Exit 0 here means the observational probe completed, not that the behaviors passed acceptance.

The before-fix test log intentionally retains its failing result as evidence. Current tests must pass; these historical failures are not outstanding test-suite failures. A failed acceptance gate is not changed to PASS by this report.

## Next major session

Execute [Session 2: close foundation gaps, resource authority and scientific boundaries](../../Major_refactor_docs/plans/ANTIGRAVITY_SESSION_2_PROMPT.md). Keep the six-session structure; do not insert a series of tiny task sessions. S3–S6 retain their original objectives. All unresolved items above are implementation requirements with explicit owners in that assignment.
