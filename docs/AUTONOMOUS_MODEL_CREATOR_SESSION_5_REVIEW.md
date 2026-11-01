# Autonomous Model Creator — Session 5 independent review

**Verdict: work in progress; the campaign is disabled pending a real execution boundary.** The handoff materially overstates completion. It adds an entry point, atomic assessment transaction, and useful tests, but does not establish the autonomous ML system described in the master plan. This review supersedes conflicting completion claims in the Session 5 handoff.

## Findings

1. **Critical: host execution is not isolation.** `campaign.rs` runs agent-supplied Python using `Command::new` with the controller user's filesystem, network, and inherited environment. It can access the original dataset, validation labels, controller state, and credentials. Stripping labels from a workspace does not prevent access elsewhere. Separate fit/infer subprocesses, and tests that inspect workspace contents, do not establish a trust boundary. The reviewer disabled `run_autonomous_campaign` before it creates state or executes code; the four campaign and intake demonstrations are ignored until a verified sandbox is wired. The sandbox must be fail-closed with explicit mounts, environment, network and resources, and tested with adversarial code against real secrets and labels.

2. **Critical: unsafe controller initialization.** The campaign recursively removes an existing `controller` directory whenever its `OWNED` marker is absent. An existing unrelated or damaged directory can therefore be deleted. Remove this behavior and require safe explicit initialization before enabling the campaign.

3. **Critical: credential restriction bypassed.** The campaign independently reads `OPENROUTER_API_KEY` using a substring endpoint check, bypassing the exact-host restriction fixed in Session 4. A lookalike endpoint can receive the key. Route all implicit credential resolution through one verified authority; avoid secrets in problem specs and child environments.

4. **Reference authority remains self-certified.** The public registration API accepts caller-supplied scores, fixture digests, and receipt digests without proving a trusted reference. Immutable SQLite rows do not validate their origin. The included test invents and registers its own authority. The reviewer removed the conformance elevation; matching caller evidence remains provisional. Independently admit content-bound reference artifacts and bind the metric version to the evaluated specification before restoring certification.

5. **Provider recovery is unsafe under concurrency and incomplete.** `reconcile_gateway_receipts` can delete a legitimate reservation between reservation and receipt creation, changes live awaiting requests to unknown without proving a crash, and can settle a response at zero when pricing lookup fails. Its synthetic state test does not exercise these races or a real dispatch crash. Use durable ownership/leases and explicit dispatch state, preserve uncertain reservations, and only settle from authoritative endpoint-specific pricing and recorded usage. Campaign role reservation IDs and per-turn gateway accounting also require end-to-end audit.

6. **Rigid fallback and model delivery do not satisfy the goal.** Agent errors are swallowed and a fixed polynomial/sine or logistic template is substituted; its fit/infer protocol, numeric CSV assumptions, JSON weights and reconstructed bundle inference restrict the solution family. A scripted provider fixture does not establish autonomous research, multiple independent specialists, flexible model families, optimization, ensemble search or correct general artifact parity. A JSON file copied as `frozen_model.joblib` is not a joblib artifact. CSV records are reserialized with `join(",")`, corrupting quoted fields. Lockbox, source/row provenance and real artifact selection need evidence through the protected runtime.

7. **Recovery and work-package claims are overstated.** The process-kill test demonstrates a SQLite transaction/process termination in a fixture, not end-to-end crash recovery across provider dispatch, model execution, scoring and delivery. Kaggle remote GPU execution and live provider calls have not been verified; local subprocess is not secure fallback. Existing adapters, schemas, and persistence tests do not prove W0–W9 are implemented and enforced, particularly scientific search, multimodal work, multi-agent workflow, or release-quality general delivery. No benchmark validates the broad model-quality claims.

## Reviewer changes

- Disabled the new campaign public entry point until isolated execution is actually implemented; added a fail-closed regression and left the affected demonstration tests explicitly ignored.
- Prevented caller-created reference records from elevating assurance to `reference_conformant`; adjusted its test to expect provisional assurance.
- Retained the handoff as historical evidence; it should not be used as a current capability statement.

## Next work

Replace host subprocess execution with a real governed worker and test filesystem, environment, network, resources, label access and adversarial code. Remove destructive initialization and centralize exact-host credential handling. Repair accounting races and recovery, use independently admitted reference data, wire source/split provenance, preserve agent-authored model and inference behavior without template substitution, and demonstrate real provider/Kaggle workflows and portable artifacts. Audit remaining master-plan packages with evidence and explicit gaps before calling this complete.

## Follow-up fixes

- Controller initialization now refuses an existing unowned path instead of deleting it. The campaign remains disabled, so this is defense in depth rather than an end-to-end initialization test.
- Implicit OpenRouter credentials now require the shared exact HTTPS authority check in campaign, gateway dispatch, and agent run. Explicitly supplied credentials remain caller-directed.
- Gateway reconciliation preserves in-flight receipts and reservations, including stored responses: without a durable dispatch lease and endpoint-bound price evidence it cannot safely infer ownership or final cost. It reports manual reconciliation rather than freeing budget or settling at zero. This deliberately restricts recovery; a complete lease-based recovery protocol is still required.
- Removed the fabricated fixed-model fallback and the JSON-as-joblib copy. Agent code and its manifest must exist. CSV output escapes special fields and preserves feature whitespace.
- These fixes do not make the campaign operational. The sandbox, general agent-authored delivery and inference parity, trusted reference admission, durable recovery, research and multi-agent scientific search, and verified live provider/Kaggle workflows remain unfinished. The hard-coded model delivery assumptions in the disabled prototype still require replacement, not merely test changes.
