# M5 independent review and M6 release handoff

Reviewed 2026-10-12. M5 scientific components exist, but full milestone acceptance
is not established. M6 preparation can proceed while the remaining dependencies
are completed. Preserve the working tree and previous local commits.

## Reviewer repairs

- Target encoder now uses positional rows, including non-contiguous dataframe
  indices, and the fold-training mean for unseen held-out categories. Previously
  the unseen-category fallback included each held-out row's own label.
- Comparison calibration OOF now fits every feature transform inside its outer
  training fold. Previously features were encoded once before outer CV, carrying
  outer held-out labels into calibration training predictions.
- No-scaling numeric transforms now impute; all-missing columns retain their
  shape; new inference missingness no longer crashes the indicator transform.
- Voting rejects invalid weights/model counts/duplicate parents, aligns parent
  probability class columns and rejects invalid probabilities or unknown labels.
  Stacking validates lineage/model counts, meta-feature dimensions and class order.
- Calibration rejects invalid probability shapes/bounds/normalization, missing
  training classes, unfitted inference and conflicting prediction class order.
- Boosting Python integer parameters now reject non-integer values consistently
  with Rust. The stacking fixture now generates real cross-validated predictions
  instead of calling fitted models on their own training rows.
- Bundle measurement includes the selected ensemble or calibrator. Comparison
  script is explicitly exploratory and unequal-compute, without claiming a durable
  sealed-test gate. Historical results are retained but marked invalid acceptance
  evidence. No new benchmark/test-set assessment was performed by the reviewer.
- Specialist delegation creates unique identities through M0 CreateBranch/CreateTask
  instead of directly overwriting existing task/lease rows. It copies and verifies
  parent workspace contents, reports provider failures and turn-limit outcomes,
  and does not turn errors into successful completion. This remains synchronous,
  not a complete supervisor-scheduled durable delegation implementation.

## Executed evidence

- `cargo test --workspace --locked --lib --test agent_m4 --test foundation --test search_m3 -- --nocapture`:
  52 tests passed before adding the delegation regression.
- `cargo test --workspace --locked --test agent_m4 --test sandbox_docker -- --nocapture`:
  17 agent tests and 5 actual Docker tests passed after delegation changes.
  Docker tests covered read-only workspace/writable scratch, non-root/cap drop,
  denied outbound network, timeout termination and output-bound enforcement.
- `PYTHONPATH=python/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s python/tests -p 'test_m5*.py' -v`:
  12 tests passed, using trusted fixtures/models only. No generated candidate host
  execution, full Python suite, live-provider spend, commit or push was performed.
- `git diff --check` passed; MLForge/ and Redesigning.md remain ignored.

## Remaining acceptance gaps

1. Scientific tables `scientific_ensembles`, `feature_transforms` and
   `calibration_records` are created but have no runtime callers. Connect features,
   calibration and ensembles to authoritative registered experiments, receipts,
   source/environment/split identities and recovery. A local parent ID list is not
   verified lineage. Bare arrays labelled OOF are not proof of disjoint training.
2. Current comparison bypasses the Rust scheduler, admission and final-test gate.
   Five configurations are not equal compute: M3 has 15 CV fits plus 5 refits,
   whereas M5 has 3 base fits plus 3 calibration OOF fits and derived candidates.
   The consumed synthetic holdout cannot become fresh confirmation evidence.
   Run a predeclared bounded comparison on fresh data through authoritative paths.
3. Resource-aware feasible selection, matched ablations, seed confirmation and
   clean-sandbox standalone bundle parity remain unverified. Report complete
   pipeline cost, including transforms, calibrators, all parents and load cost.
4. Agent finalization still closes research and hashes/copies a branch without
   performing the full source/environment/spec/protocol-bound approval, draining,
   refit, trusted feasibility, durable sealed-test assessment and delivery workflow.
   Copy errors are ignored in that path. Repair before release certification.
5. Outbound filter remains pattern-based and retains sample rows. It is not a typed
   policy-approved upload boundary. Tool execution can still fall back to a mutable
   branch. Provider response/checkpoint recovery, currency budgeting, finalization
   allowance and specialist scheduling/reconciliation need end-to-end evidence.
6. Joblib ensemble loaders are only suitable for trusted verified artifacts;
   arbitrary candidate artifacts must be loaded within the sandbox. Array/schema
   checks cannot certify provenance, budget enforcement or containment.

## Prompt for Antigravity

Proceed with the M6 release-quality session in
`/Users/hamza/Developer/Imp_Projects/MLRescurix`. Read BUILD_PLAN.md (M6),
SYSTEM_DESIGN.md, docs/m6-session-prompt.md and the M4/M5 handoffs. Preserve existing
work. The user authorizes direct implementation and defect repairs; do not return
another proposal-only loop. Complete the dependencies listed above alongside M6.

Implement release-quality bundle verification, documentation, bounded resource
fault tests, representative tabular benchmarks and clean install/recovery scenarios.
Provide an end-to-end supported task that finishes without manual state/code edits.
Verify standalone inference from raw input, hashes and manifests in a clean managed
sandbox, including feature engineering/calibration/ensemble parents. Fail explicitly
on tampering, unavailable requirements or unverifiable constraints.

Keep Rust authoritative for admission, budgets, leases/fencing, scheduler, evidence,
evaluation and finalization. Demonstrate crash/restart recovery, stale-result denial,
concurrent reservations, project cancellation isolation, orphan cleanup, disk-full
publishing, lease-safe GC and protected artifact retention. Establish policy-bounded
steady-state storage using disposable bounded fixtures, without exhausting the host.

Repair M5 integration and run a genuinely matched, predeclared comparison through
those authoritative paths using fresh datasets/seeds. Cover binary, multiclass,
regression, unseen categories, missingness and group/time splits. Report feasible
model rate, quality, time, resource/provider costs, failures and storage growth.
No-gain outcomes are valid; do not recycle the invalid historical comparison as
release evidence. Do not widen data downloads, compute or provider spend silently.

Keep research closed before final refit exposes protected labels. Require verified
content-bound user approval and an independent audit; sealed-test consumption must
be durable and nonrepeatable. No approvals or monetary prices may be fabricated.
Do not claim release ready until required scenarios pass with exact evidence.

Never run generated/untrusted code or deserialize arbitrary candidate artifacts on
the host. Preserve ignored legacy directories. No global Python changes, GPU usage,
live-provider spending, commits, pushes, publication or automations without explicit
applicable authorization. The previously exposed credential must be replaced before
further live calls; never print or copy it.

Finish with a supported-feature matrix, implemented/verified/unverified evidence,
exact reproducible commands, remaining limitations and a concrete release decision.

## Current continuation after independent review, 2026-10-13

Read `docs/m6-completion-status.md` and
`docs/m6-independent-review-2026-10-13.md` before resuming. Their NOT READY decision
supersedes the pasted implementer completion report. Preserve all existing
reviewer fixes, including the newly retained command/action retry fences,
unknown-tool-outcome refusal, consumed-before-scoring marker, required external
manifest receipt, finalization environment/approval validation and runtime-asset
symlink rejection.

Implement durable per-tool outcome reconciliation and interrupted delivery
recovery. Then implement enforceable total-storage admission and safe retention
of retry identities, with exact accounting and genuine concurrent lease/GC and
scheduler ENOSPC fault coverage. Do not restore unsafe deletion to manufacture a
storage plateau. Reinstall and prove supported end-to-end recovery/delivery.
Only after these prerequisites pass, persist a prospective benchmark protocol and
reproducible driver with matched total compute, then run fresh representative
benchmarks. The retained 15-pair JSON is unequal-compute historical evidence and
is explicitly not release-acceptance evidence.

Keep scientific fitting/inference inside the locked digest-pinned Docker image;
never use host fallback. Keep previous authorization boundaries for paid calls,
real user-candidate approvals, sealed test gates and publication. Finish with
exact evidence, remaining limitations and an honest release decision. Implement
directly instead of returning another proposal-only loop.

### Follow-up completed 2026-10-13

File tools now persist before/after hash intents and outcome digests; recovery
reconciles exact file bytes only after authoritative provider action settlement.
Non-file tools still refuse unknown outcomes without authoritative external
evidence. Job admission enforces the 128 MiB state ceiling against current state
and active output/log reservations. Concurrent artifact collectors are serialized
and tested against live leases and cancellation. These fixes do not establish
total-state bounds across every writer or scheduler-level ENOSPC recovery.

The corrected binary was installed under `/tmp/mlrescurix-m6-install` and run
from `/tmp` on macOS arm64. `bash scripts/run_m6_review_checks.sh` passed, including
the digest-pinned scientific Docker image. The exact new evidence is in
`docs/evidence/m6/review-{rust-final,bundle-docker-final,finalization-docker-final,python-docker-scoped}.log`,
`review-image-dependencies.json`, and `installed-review-capabilities.json`.
M6 remains NOT READY. Do not run fresh benchmarks until complete recovery,
storage and installed delivery gates pass.
