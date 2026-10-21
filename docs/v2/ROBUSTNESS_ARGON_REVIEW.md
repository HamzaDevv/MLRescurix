# Gemini Argon review: disposition and specification amendments

Reviewed 2026-10-21 against source baseline `5be287b` and the
[refined master plan](ROBUSTNESS_MASTER_PLAN.md). Original review preserved in
[ROBUSTNESS_ARGON_REVIEW_SOURCE.md](ROBUSTNESS_ARGON_REVIEW_SOURCE.md).

Verdict: the five integration concerns are useful. Accept them with the corrections
below. The numerical architecture ratings and guarantees are opinions, not test
evidence. The review assessed the supplied blueprint; it did not establish that the
current repository satisfies the blueprint.

## Disposition of the five edge cases

| Argon concern | Disposition | Repository basis and final requirement |
| --- | --- | --- |
| Custom-transformer unpickling outside the package | Accept; proposed remedies need qualification | `pipeline.py` defines `FullInferencePipeline` and other custom classes; `autonomous_rescue.bundle_candidate` writes joblib and emits `inference.py` calling `joblib.load`. Audit every custom class/import dependency, including ensemble/candidate classes, rather than copying just two files. Test fresh-process loading without the checkout or installed worker package. |
| OOF error-forensics producer missing from blueprint | Accept integration requirement; producer already exists | `scientific/error_forensics.py` has `ErrorForensicsAnalyzer.analyze`; `contracts/v2/error_forensics.schema.json` and portfolio/critic paths also exist. Reuse these and prove per-completed-trial delivery to the agent; do not add a second competing analyzer or assume the current wiring covers every trial. |
| Fidelity breaks group/time semantics | Accept; amend the exact rule | Existing topology-aware sampler is followed by random proxy splitting, while ensemble OOF uses independent random folds. Fix the entire evaluation path. Fidelity changes only legal training resources inside locked splits; validation assignments remain fixed for a comparable rung. |
| Per-champion preprocessing and OOF target scale | Accept | Different feature versions need independent fitted preprocessing + model + inverse-target behavior. Store immutable row/fold/class/feature/transform lineage. Stacking input predictions use canonical target units; scoring may then apply the declared metric transform. |
| Metric direction and near-zero denominators | Accept problem; reject the supplied formula as written | `(ensemble-single)/abs(single-dummy)` still rewards a worse RMSE and penalizes an improvement. Include an explicit direction sign, a declared normalization and absolute tolerance near zero. |

## A1. Serialization: the actual invariant

Copying `primitives.py` into a file with a new name does not change a pickled object's
original `__module__`. Inlining definitions in `inference.py` alone has the same problem.
An exporter must deliberately preserve/resolve the serialized import graph, or
serialize under a stable bundle-local module path in a fresh process. That graph may
include the pipeline wrapper, sanitizer, feature transformers, ensemble wrapper,
target inverse, calibrator and model-library imports.

Choose and document one export profile in S2:

- A versioned, self-contained bundle runtime with stable module names and a verified
  loading strategy; or
- An explicitly supported declarative/export format that covers the whole required
  pipeline and passes numerical parity tests.

Avoid global alias hacks that accidentally load a developer checkout. If module aliases
are necessary for backward compatibility, bind them only to verified bundled code
and test in an isolated subprocess. Cloudpickle or skops is not an automatic fix for
custom imports, portability or trust: support/version/import requirements still need
proof. No format migration is required merely because Argon suggested one.

Acceptance: create a fresh environment from the bundle's pinned dependencies, clear
`PYTHONPATH`, keep the repository off `sys.path`, do not install `mlrescurix_worker`,
copy the bundle outside the checkout, and exercise CLI plus HTTP on raw records.
Compare predictions/probabilities/class order with the frozen candidate, including
unseen categories, missing fields, target inverse and calibration. Verify file hashes
before loading executable serialized content. Say “independent of MLRescurix runtime,”
with explicit Python/library dependencies.

## A2. Mid-loop forensics contract

Extend/reuse `ErrorForensicsReport` for the full artifact and add a compact agent view
after a valid complete development evaluation. Keep the serialized UTF-8 view within
1,024 bytes; enforce byte size after serialization, including references. Bound top-k
lists and preserve valid JSON. Full details remain in an authorized hashed artifact.

Required fields: schema/version, candidate/source/split references, metric/direction,
coverage, `train_val_gap` with its definition, worst supported slices, ranked feature
evidence and optional interaction candidates, uncertainty/status and full-report hash.
Use `null`/`not_computed` for unavailable evidence, never a synthetic zero.

`zero_gain_features` is a candidate list with method, support and uncertainty, not a
deletion instruction. Native split gain is not permutation importance. A high-error
slice does not prove an interaction pair improves a model. Permutation/interaction
probes must be budgeted; admit at most bounded top-k probes and settle resulting
actions through paired fold-local ablation. A scalar metric alone cannot justify them.

No pruned, failed or incomplete prediction vector receives a complete OOF report.
Time-based CV may intentionally leave an initial history prefix unassessed: record
the expected eligible validation rows and coverage mask, never fill them with zero.
Partial-rung diagnostics are labeled partial and cannot replace full-rung assessment.
Aggregate summaries pass existing outbound data/privacy policy before reaching the
provider. They contain no raw rows or holdout-derived observations.

## A3. Fidelity resource contract

Prefer iteration/epoch budgets when the estimator exposes meaningful intermediate
results and measured savings. Otherwise use deterministic nested training subsets
within the training side of the existing fold. Some models require refitting and
have no valid learning-curve extrapolation; record that cost and limitation.

- Temporal subsets use training history strictly before the fold cutoff, keep the
  required lag lookback and gap, and preserve date/entity boundaries. A contiguous
  tail window is allowed only within that legal history. Keep the assessment horizon
  unchanged; never sample the combined train/validation frame.
- Group subsets contain whole training groups with recorded selection/coverage.
  Keep minimum rare-class support; changing group composition can still change
  distribution and counts, so do not claim universally unbiased group features.
- If fold count is a fidelity axis, select fixed fold subsets, report their IDs and
  score uncertainty, and compare candidates only at matching rungs. All champions
  promoted for stacking/final selection must complete the common full assessment.
- The proposed N=50,000 cutoff has no repository evidence and is not an invariant.
  Pick the resource axis from task semantics and measured cost, not row count alone.

## A4. Champion and prediction contracts

Each champion binds its own raw-input schema, sanitizer configuration, feature-set
version, fitted preprocessing, estimator, target transform/inverse, calibration and
class order. The final ensemble invokes those complete independent pipelines.

Prediction artifacts contain row IDs, fold IDs, eligible-row mask, canonical units,
class order, source/pipeline hashes and prediction/probability column semantics.
Join by immutable keys, not file position. Reject duplicate/missing/extra eligible
rows, transform mismatch, class mismatch, non-finite output and unequal assessment
coverage. Apply inverse target transformation exactly once before canonical storage;
the inverse may be `exp`, `expm1` or another declared mapping. House Prices can then
apply its log-price metric to canonical positive price predictions. Fit stack/calibration
parameters with the independent assessment protocol in the master plan.

## A5. Corrected gain calculation

Let `d = +1` for maximization and `d = -1` for minimization:

```text
benefit = d * (score_new - score_old)
relative_gain = benefit / max(abs(score_old), epsilon)
```

This is the master plan's default. Near-zero baselines additionally require a
predeclared absolute improvement and uncertainty assessment. Negative or unbounded
metrics need metric-specific practical tolerances, not a claim that every percentage
has the same meaning. Slice audits use the same direction and explicit support rules.

If a task deliberately adopts dummy-baseline normalization, the corrected form is:

```text
normalized_benefit = d * (score_new - score_old) / max(abs(score_old - score_dummy), epsilon)
```

That changes what “0.5%” means; freeze the chosen denominator before search and do
not switch conventions to pass a gate. It is not uniformly an error-reduction rate.
For RMSE 10 → 9 with dummy RMSE 20, Argon's expression gives about -10% despite
improvement. Multiplying the numerator by -1 gives +10%; 10 → 11 correctly gives
-10%. Arithmetic cases are saved in
[argon-review-checks.json](../evidence/v2/robustness/20261021-plan-review/argon-review-checks.json).

## Claims not accepted as established facts

- Vault locking alone does not guarantee fold-local preprocessing, safe mounts or
  absence of target exposure. The current source findings demonstrate why those
  boundaries need separate regressions.
- A pooled OOF bootstrap does not eliminate model-selection bias. The claimed >80%
  compute saving is unmeasured here; BOHB/LC-PFN reference stubs do not establish it.
- JSON actions constrain requests but do not ensure complete numerical determinism,
  correct statistical choices or safe execution on their own.
- “Zero dependency,” “100% implementation-ready,” and 10/10 ratings do not certify
  a working clean-room bundle or vulnerability-free harness.

All accepted amendments are part of S1–S5 gates in the master plan and persistent
ledger. This review changed documentation only; no implementation fix or live
benchmark is claimed by accepting a recommendation.
