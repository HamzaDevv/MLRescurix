# Independent M6 review: next work

> Current follow-up: [2026-10-13 independent review](m6-independent-review-2026-10-13.md)
> and [completion status](m6-completion-status.md). Release acceptance remains pending.

Reviewed 2026-10-12. M6 release acceptance is pending. The supplied implementer
report's approval is not supported by the actual bundle-verification boundary or
by complete scientific execution integration. Continue M6 completion rather than
inventing M7 or publishing the current build.

## Repairs made by the reviewer

- Removed the host Python subprocess from Rust `verify-bundle --sandbox`. It now
  requires `MLRESCURIX_BUNDLE_IMAGE` pinned as `repository@sha256:<64 hex>` and uses
  the existing hardened Docker runner with bounded resources/output/deadline. It
  copies inputs into disposable staging, checks the original manifest identity,
  revalidates the staged bytes, and never modifies the original bundle.
- Python bundle verifier now refuses candidate execution and directs callers to
  the rebuilt Rust sandbox verifier. Its previous dynamic import was host
  execution; its schema check also caught its own failure exception, accepting
  missing/extra columns that were not rejected.
- Integrity-only verification now reports `integrity_verified`, with parity
  explicitly skipped. Manifest bytes count toward total bundle size. Untracked
  `__pycache__` content is no longer exempt, and Python root/nested symlinks are
  rejected before reads. Required unknown constraints fail; unsupported memory
  without any required threshold remains disclosed as unsupported.
- Restored the historical M3 migration. Added transactional version 7 to rebuild
  the recipe CHECK constraint for existing version-6 states, preserving dependent
  table references, checking foreign keys and remaining idempotent.
- Ensemble registration validates weight shape/finiteness/nonnegativity/positive
  sum and rejects an agent/caller assertion of official feasibility. Registration
  metadata is not evaluation of a complete delivered ensemble pipeline.
- Paid-provider test is ignored by default; possessing a key is not authorization
  to spend money during ordinary `cargo test`.
- Benchmark driver refuses before consuming final-test gates if the required pinned
  image is absent or the binary is stale. Historical metrics remain available,
  but the host parity/release claims are marked invalid acceptance evidence.
- README and release-notes banner now reflect pending release acceptance. Rust
  export tests now explicitly verify integrity, not pretend to prove sandbox parity.

## Reviewer evidence

`cargo test --workspace --locked -- --nocapture`: 66 passed, 1 paid-provider test
ignored. Includes 7 release tests, existing-version-6 upgrade regression, and 5
live Docker isolation/timeout/output tests. A final staging revalidation edit was
followed by the targeted library verification regression.

`PYTHONPATH=python/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s python/tests -p 'test_m[56]*.py' -v`:
18 passed, including safe refusal, symlink and unknown-constraint regressions.
`git diff --check` passed; MLForge/ and Redesigning.md remain ignored.

No live-provider requests, new benchmark/test-set assessments, commits or pushes
were performed by the reviewer. Arbitrary bundle code was not executed on the host.
Scientific-image parity was not executed: only a bare Python image is locally
available, and that is not the required managed scientific dependency environment.

## Remaining release work

1. Provide a digest-pinned scientific image with the dependency versions in
   `python/pyproject.toml`. Prove the versions inside it. Run actual bundle parity
   and missing/extra-column rejection there, including a deliberately hostile
   bundle and denial of labels/secrets/network/host writes. Do not turn safe refusal
   or integrity checks into parity success. Validate dependency-lock compatibility.
2. Scientific registration APIs and schemas are present, but neither metadata nor
   the recipe-only benchmark proves actual feature/calibration/ensemble execution,
   verified OOF provenance, source/environment binding or full-pipeline feasibility.
   Integrate those paths into registered, scheduled, resumable jobs and evaluate
   them through the trusted boundary. Parent hash strings must refer to verified
   managed bytes, not merely be nonempty. Include genuine matched ablations.
3. The report calls identical scores statistical parity without statistical evidence.
   Six synthetic runs and their total disk size are not proof of generalization or
   storage steady state. Demonstrate bounded repeated cleanup/restart cycles,
   lease/GC races, disk-full publication, fresh representative datasets/seeds and
   explicit matched compute accounting. Preserve consumed test gates and past data.
4. Close remaining M4 integration requirements: durable specialist scheduling and
   outcome recovery, policy-approved outbound views, provider/checkpoint recovery,
   currency/finalization reservation accounting and content-bound approval/audit.
   Branch closure/snapshotting alone is not verified refit/test/delivery finalization.
5. Bundle manifests establish internal consistency, not independent authenticity.
   A caller can replace a model and regenerate its own hashes/expected predictions.
   Bind verification to an externally trusted candidate/source/environment manifest
   and durable approvals. Sandbox parity compares examples, not complete feasibility
   or fresh latency/memory certification. Keep verification claims scoped accordingly.
6. Show clean install/upgrade/recovery with declared supported platforms and an
   end-to-end supported task. Do not describe a deterministic fixture as proof of
   the complete autonomous coding-agent product. Record unsupported features honestly.

## Paste into the next implementation session

Continue M6 release completion in `/Users/hamza/Developer/Imp_Projects/MLRescurix`.
Read `docs/m6-review-next.md` and the existing M6 handoff. Preserve reviewer fixes
and all existing work. Implement and fix directly; no proposal-only approval loop.

First close the real sandbox bundle-parity path using a digest-pinned scientific
image with verified locked dependencies. Rebuild the Rust binary and test correct
inference, probability/class parity, schema rejection and hostile bundle containment
inside Docker. Keep host execution prohibited and safe refusals explicit.

Then connect feature/calibration/ensemble execution and OOF provenance to the
Rust scheduler, admission, registered artifacts and trusted evaluation. Verify full
pipeline cost and feasibility; registration must never certify them by assertion.
Complete content-bound finalization and the remaining agent integration dependencies.

Prove install/upgrade/recovery and policy-bounded storage across repeated fault and
cleanup cycles. Produce fresh, predeclared matched benchmarks within explicit
small compute/download budgets after prerequisites pass. Do not reuse consumed
holdouts as new confirmation, claim statistical equivalence from identical scores,
or certify release from aggregate fixture counts.

Keep the boundaries in prior handoffs: no legacy reuse, arbitrary host execution,
global Python changes, paid-provider spending, GPU usage, candidate approval,
commits, pushes, publication or automations without applicable user authorization.
The exposed credential must be replaced before any further live calls.

Finish with exact implemented/verified/unverified evidence, supported-platform and
feature matrices, and a release readiness decision justified by those scenarios.
