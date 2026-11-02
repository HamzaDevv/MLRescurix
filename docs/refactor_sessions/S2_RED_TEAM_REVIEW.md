# Session 2 independent review — useful progress, acceptance remains partial

Reviewed November 3, 2026 against the actual working tree, the S2 author packet, gate JSON, implementation and offline tests. The author's **all A–G PASS** conclusion is not upheld. This review does not certify production execution or a new Amazon run. The canonical master plan remains the implementation authority.

## What improved

The owner now makes provider-backed decisions after validation, with scripted delegate-versus-stop trajectories. Requested metric, dataset and split mismatches are rejected on resume. Changed weights cannot simply inherit the consumed assessment. Lease acquisition uses an immediate transaction and increments an epoch. Search admission uses a durable SQLite transaction, and timeout fixtures exercise TimeoutError correctly. Canonical recipe hashing and ordinary sibling-failure isolation have meaningful focused coverage. Production remains disabled without a verified isolation backend.

These are useful changes. They do not establish one Rust resource authority, a general durable owner loop, hostile-worker isolation or immutable source-to-score-to-release provenance.

## Defects reproduced and repaired in this review

1. **Missing owner authorization consumed assessment.** A provider response with `content=None` at either the first owner decision or the post-challenger decision reached successful finalization. Unknown prose also defaulted to success. Substring matching admitted negated or ambiguous decisions. Added explicit action/rationale parsing, rejection of missing/tool-only/ambiguous replies, an approved-audit prerequisite and explicit post-audit finalization. This is an interim strict text adapter; it does not replace the required versioned owner contract or general ReAct state machine.
2. **Malformed challenger JSON was accepted.** `{"verdict":"APPROVED"}` sufficed. Supplied task/candidate/protocol/score/schema fields were silently discarded and replaced with the current identity. Require a nonempty rationale, reject contradictory supplied bindings and malformed findings, and persist the parsed structured findings/task information. Minimal JSON and explicit prose remain temporary compatibility adapters; fully mandatory attempt/source/evaluation binding is still open.
3. **The real loop lost pre-turn search attempts on reconstruction.** S2 tests manually passed matching project IDs to budget instances, but `AutonomousAgentLoop` called `OperationalBudgets.load()` without its project or stable database. Each constructor generated a new random project. Bind the bridge to its actual project and workspace-local database; a fresh loop constructor now sees the previous admission. This does not unify controllers across different workspaces or migrate earlier orphaned random-ID stores. Preserve those old records for reconciliation rather than guessing their ownership.
4. **Unknown execution was swallowed as a sibling failure.** Injecting `RecoveryRequired("remote dispatch outcome unknown")` still invoked the next sibling. Explicit recovery exceptions now propagate through the sibling handler. The surrounding loop can record a rejection; the test asserts that no further sibling executes. A complete typed failure taxonomy and durable unknown-attempt recovery remain required.
5. **Misleading fixture environment identity.** Evaluation identities said `host_python_isolated`; they now say `host_python_uncontained_fixture`. Actual production isolation remains gated.

Reproducer tests were run against the submitted implementation first: three Rust tests and two Python tests failed. After repair they pass. Historical before-fix logs are retained, not current failing acceptance results.

## Remaining material findings for Session 3

### P1 — Source can still be rebound to a previously consumed score

The weights check protects weights alone. The assessment reuse branch ignores the stored full receipt, and completed specialist evidence is selected by task success without matching the current source/environment/prediction closure. A disposable CLI probe generated a reference candidate, changed its inference source to emit `1000000.0`, deleted the bundle manifest, and reran the campaign. It returned exit 0 with the **same validation score, same holdout score and same weights**, exported the changed source, and that source produced predictions different from the scored predictions. No historical state or real dataset was modified.

See [observations](../evidence/refactor/S2/red_team/remaining_gap_observations.json) and [reproducer](../evidence/refactor/S2/red_team/reproduce_remaining_gaps.py). Prevent changed source/program/dependencies/environment/preprocessing/weights/predictions/selection from reusing evidence. Do not consume the assessment again to repair a broken release. Fast-path scorecard claims also require controller provenance, not merely a self-updated manifest.

### P1 — W2's single resource authority was not implemented

Search writes its own Python SQLite database. Tokens and Kaggle hours remain Python counters/turn restoration, while money has existing Rust/Python accounting paths. The campaign policy derives token allowance from money and uses a fixed call count, independently of the configured 5M/30h/500 operational policy. No evidence establishes a single Rust transaction admitting all applicable resources or importing historical obligations. This was a central S2 outcome, not an optional S3 improvement.

Search also still dispatches with `budgets=None`: the mock probe records one provider call and success without admission. The legacy client defaults to no budget. Replaying a contradictory settlement changes an already `success` attempt to `timeout`; `settle_search()` is not immutable/idempotent settlement. Read/schema failures may be reduced to zero usage or warnings. Move authority once, preserving unknown obligations and denied-dispatch semantics. Connectivity alone must not claim enforcement.

### P1 — Lease fencing is a separate check, not atomic mutation protection

`verify_active_lease()` precedes completion/assessment writes in separate operations. Another owner can take over between check and write. Several assignments, runtime commands and filesystem writes lack fences. Heartbeats occur only at selected checkpoints, so a still-live owner in a long fit or provider call can exceed 30 seconds and lose the lease while continuing to write shared artifacts. PID liveness alone has host/PID-reuse limits and does not reconcile surviving workers or unknown provider effects.

The named concurrency test makes sequential acquisitions on one connection. The named SIGKILL test inserts a timestamped row; the immediate-recovery test waits for `true` to exit, then inserts a lease for its dead PID. These are valid helper checks, **not an actual killed campaign**, concurrent reservation race or stale completion race. Require real competing controllers and kill/restart at dispatch/completion boundaries.

### P1 — W3 isolation is absent

Removing targets from feature CSVs is good schema separation. Host Python still shares the controller's filesystem identity and can read labels, SQLite and source data by absolute path; directory names are not access control. `lockbox_features.csv` is present in the mutable training workspace before final selection. No actual worker security boundary or independent fit/inference capability was exercised by the new gate. The production refusal remains correct, and an unavailable backend must be **NOT_VERIFIED**, not Gate D PASS. Reuse existing snapshot/isolated-inference/resource machinery rather than creating another host runner.

### P1 — Owner and challenger foundations are narrower than reported

The owner is a two-step post-fit branch, not yet a durable loop choosing authoring, repair, a different hypothesis, handoff or finalization from evidence. S2's two trajectories use the same generated data and score with different scripted responses; they prove action branching, not observation-dependent policy quality. The new tests do not capture and assert actual serialized request isolation. The authoring path still silently generates a built-in model when the specialist merely says `[FINAL]` or `reference` without files. The challenger receives summary fields rather than scoped primary source/prediction/evaluation evidence; stamping a JSON envelope does not establish an independent audit. No-gateway diagnostics also write local approval and boss-success rows. Keep those diagnostics explicitly distinct from autonomous acceptance.

### P2 — Contract and installed-runtime evidence is incomplete

Rust dynamically checks action names but still applies its own receipt parameter allowlist rather than each canonical descriptor schema. Python implements a partial handwritten schema validator and finds descriptors by walking filesystem ancestors; its package-data declaration only includes protocol JSON. The offline probe built a real wheel using `pip wheel --no-index --no-deps --no-build-isolation` in a disposable copy: the wheel contains no `actions.json`, and an isolated import from it fails with `FileNotFoundError: Missing canonical actions schema`. Embedded Rust extraction and standalone installed Python are different paths. The owner choices are not covered by the scientific action list. Validate all supported entrypoints against versioned descriptors, using actual built/extracted assets away from the checkout; remove silent aliases or document validated translations.

### P2 — Gate evidence overstates what was measured

Gate A's JSON lists 24 validation rows, while the 30-row fixture uses six assessment rows and a 75/25 development split, leaving six validation rows. Gate D claims `controller/val_ground_truth`; the implementation writes `controller/private/val_labels.json`, with no tested physical isolation. The gate JSON summaries lack generated source/binary identities and machine-captured receipts proving their detailed claims. Gate G inventories three bridges but does not demonstrate recursive deletion. Preserve author evidence as historical claims and regenerate subsequent evidence from executed assertions/receipts.

## Corrected gate assessment

| Gate | Independent status | Basis |
|---|---|---|
| A | PARTIAL | Provider-backed branch exists; invalid authorization fixed here; general owner loop, actual context assertions and independent primary-evidence audit remain |
| B | PARTIAL | Atomic acquisition and epoch helper exist; atomic fenced writes and real campaign kill/race recovery unproven |
| C | FAIL for full S2 authority | Durable Python admission is useful; stable loop binding fixed here; bypass, settlement and unified Rust authority remain |
| D | NOT_VERIFIED for containment | CSV target stripping and production refusal pass; hostile-worker boundary unavailable |
| E | PARTIAL / release binding FAIL | Changed spec/weights rejection and ordinary duplicate/sibling tests pass; changed inference source inherits old assessment |
| F | PARTIAL / wheel import FAIL | Dynamic names and basic Python checks exist; real built wheel lacks action schema and cannot import the loop in isolation; cross-language parameter parity incomplete |
| G | PARTIAL | Bridge inventory exists; replacement/deletion closure not established |

## Verification and scope

A `cargo test --locked --workspace` run started before these repairs exited **101** at `reflexive_loop_gate::test_reflexive_orchestrate_end_to_end_and_sqlite_v11_sync`: measured CV **0.8148** was below its **0.815** assertion. The working tree changed while that run was executing, so it is not a frozen pre-repair snapshot. A subsequent run passed this test; the discrepancy needs diagnosis rather than a claim of a proven cause. Do not lower the threshold or tune a test-specific model merely to pass. The failed run did not reach later targets; the resource-broker suite also passed separately.

Machine-readable commands, exit statuses, source/toolchain/binary identities and counts are recorded in [verification.json](../evidence/refactor/S2/red_team/verification.json). Logs sit beside it. The submitted focused Python suite independently passed **60 tests**; after these repairs it passed **62**. The focused Rust boundary suites passed after repair. The subsequent full Rust workspace run exited **0**; its counts and logs are recorded in the verification manifest. Full Python-suite acceptance, paid provider behavior, live search, remote Kaggle and hostile container execution are not claimed. Workspace tests may build/install local runtime artifacts; no new paid provider call, live research query or Kaggle job was requested by this review.

Session 2 is **PARTIAL**, not a release milestone. Proceed with [the major Session 3 assignment](../../Major_refactor_docs/plans/ANTIGRAVITY_SESSION_3_PROMPT.md): close the specific authority/provenance boundaries as prerequisites of the open contained train→infer→score→standalone delivery slice. Preserve six major sessions and historical financial/scientific obligations. No design approval is needed for those already-authorized local changes.
