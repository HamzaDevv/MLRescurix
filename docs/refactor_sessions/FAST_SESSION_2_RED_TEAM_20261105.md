# Fast session 2 — independent review

The session adds useful ledger migration and old-writer fencing, selected provider and Kaggle admission integration, source binding, accounting replay, CAS integrity and scheduler recovery work. The author records full-suite snapshots (522 Python passed, 1 skipped; Rust workspace exit 0 with ignored tests), but those receipts are not independently bound to this final tree and are not live or overall acceptance evidence.

Remaining risks and work:

- Resource unification is not established across all production paths. Python retains a separate campaign monetary authority, fallback operational database and optional Kaggle budget wiring; no evidence proves every caller is routed through one transaction. Token reservation and gateway provider accounting can both represent one request; reconciliation and double counting need end-to-end review. Some settlement errors are logged rather than propagated. Failed admission after the separate campaign reservation may leave a reservation behind. Legacy migration covers its specified tables, not all possible external liabilities.
- Cancellation cannot force-stop a remote Kaggle kernel. This review changes cancellation settlement from `failed` to `outcome_unknown`, retaining the requested hours. The updated integration test confirms a subsequent job is refused when that uncertain reservation exhausts headroom. Later remote completion/reconciliation and quota telemetry remain necessary.
- This review removes a duplicated completion allowance in the governance token reservation: its prompt bound previously included the completion bound, which was then separately reserved again. This remains a conservative estimated admission, not live metering proof.
- The independent evidence still does not establish an owner-controlled research workflow with trained diverse models and a non-Amazon task; comprehensive containment, atomic fencing, provenance, replay, holdout, delivery, and recursive obsolete-code removal also require real entrypoint evidence.
- No live Amazon campaign was performed. The historical campaign ledger reports $0.01038 remaining against a $0.05 per-run ceiling. Actual admission also depends on current policy, liabilities and resource wiring; no live budget increase or provider call was made in this review.

Validation during this review is reported with the commit response; test results remain offline and do not prove remote execution or final acceptance.
