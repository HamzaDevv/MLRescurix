# Disposition of Argon's planning review

Reviewed and incorporated on 2026-10-14. Source: [verbatim review](ARGON_PLAN_REVIEW.txt).
This is a documentation refinement; runtime implementation remains unstarted.

## Accepted and integrated

- Cross-language contract touchpoints: existing v1 schemas/generator/fixtures,
  Rust contracts and Python protocol validators are explicit in the plan.
- Outbound provider boundary: new tool outputs, artifact summaries and user notes
  must pass typed projection, approved views and existing sanitization.
- Composite/search compatibility: scheduled_pipeline, ensemble and search adapters
  need executable-candidate integration; calibration already works with OOF arrays.
- Explicit harness-driven fold fit/predict: immutable source/config, scoped per-fold
  training labels and feature-only held rows, verified model/prediction bindings.
- Rescue provenance: distinguish refitted source from supplied pretrained state
  and default unverifiable legacy performance to an unverified baseline.
- Container recovery: persist exact container IDs plus owner/project/lease/epoch
  metadata; reconcile Docker-daemon survivors before releasing resources.
- Artifact summaries: small index digests for lists and detailed summaries on demand,
  with independent total query/context token bounds.
- Dashboard consistency: architecture now includes the resource drawer as its fifth
  view; read/resource slices can start after P2/P3, while new semantics/steering wait
  for P4/P5. Read views can show existing M6 data without fabricating future state.
- Default read-only source mounts and scoped writable scratch/outputs are explicit;
  reviewed writable staging remains an optional generic coding capability.

## Corrections to the supplied replacements

| Review proposal | Resolution in authoritative docs |
| --- | --- |
| All new v2 contracts must flow through the existing v1 generator | Current generator covers worker frames/fixtures only. Extend checking/generation deliberately and use new schema versions; do not silently change v1 meaning |
| Calibration assumes built-in estimator objects | `fit_from_oof` already accepts probability arrays and labels; focus changes on executable-parent inference, scheduled fitting and search-space coupling |
| 128 MiB is preserved as a control-only ceiling | Existing policy covers managed state. Enforce it first, then explicitly migrate to split quotas before placing blobs outside it |
| Read-only workspace makes sandbox execution safely repeatable | Reconcile prior publication, caches, scoring queries and surviving containers; retry only classified isolated operations with immutable inputs |
| Fold containers prevent all leakage | Mount only one fold's inputs and isolate fit state. The coding agent may know development labels; track that exposure and preserve untouched outer assessments |
| Minimum slice sizes prevent label reconstruction | Thresholds/coarsening reduce exposure; overlapping queries can still reveal information. Apply aggregate information/query policy, no row-level sealed residuals |
| Clean refit or user attestation is sufficient for independent legacy assessment | Clean refit resets learned state; prior exposure of evaluation data still invalidates independence. Record provenance and attestation uncertainty |
| Reap every surviving labeled container | Verify state-root owner, exact container ID and current lease epoch; do not kill workers owned by another live supervisor |
| <=512 bytes means approximately 120 tokens | Byte limits are not provider token guarantees. Count tokens and cap total list/context output |
| Arbitrary fixed CAS/query/slice defaults | Record 2 GiB, 50 queries and max(20, 5%) as uncalibrated proposals in DECISIONS.md; choose limits in owning phases |
| Google search URLs replace local document links | Preserve local file links; do not import response marker `<<>>` into documents |

Keep original prediction validation requirements: missing/extra/duplicate rows,
probabilities, NaNs, class order and exact source/model/split identities still reject.
OOF used to train stacking/calibration cannot also be independent evidence of the
fitted composite; use outer validation or properly nested cross-fitting.

## Continuation

Recommended starting implementation remains P1.1 plus P1.2 scheduler job submission
reconciliation. No later runtime phase was authorized by receiving review text.
Stable P0–P9 and task IDs remain intact. Review findings are planning requirements,
not new test passes or closed M6 gates. The regenerated review packet includes this
disposition and the review archive for another independent review if desired.
