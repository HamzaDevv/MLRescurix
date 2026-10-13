# M6 release notes

**Release acceptance: NOT READY, independently reviewed 2026-10-13.**
The previous READY statement and equal-fit benchmark claim were incorrect.
[Current completion status](m6-completion-status.md) is authoritative;
[independent review](m6-independent-review-2026-10-13.md) records the fixes and
remaining gates.

M6 implements scheduled feature engineering, calibration and ensemble pipelines
with managed OOF receipts, content-bound approvals, Docker finalization and
state-bound bundle verification. The scientific image is pinned in
`docker/scientific/image.ref`; its lock is `docker/scientific/requirements.lock`.
Actual in-image versions are recorded in
[evidence/m6/review-image-dependencies.json](evidence/m6/review-image-dependencies.json).

This review repairs retry-fence deletion, unknown tool replay, interrupted sealed
assessment reuse, missing manifest authentication, finalization environment
substitution and runtime-asset symlink escape. It preserves the required-memory
certification refusal and the prohibition on host bundle execution.

The follow-up adds durable file-tool intents/outcomes with before/after byte
reconciliation, enforces a 128 MiB scheduler job-admission reservation against
current state plus active output/log bounds, and serializes concurrent artifact
collectors without holding the database writer during filesystem deletion. Fresh
Rust and Docker regression evidence is linked from the completion status. A fresh
binary install runs outside the checkout and still reports acceptance pending.

The happy-path Docker fixtures are functional evidence. Automatic recovery of
unknown non-file side effects, total-state admission on every write path and an
archive/tombstone protocol, scheduler-level disk-full recovery, installed full
delivery certification and fresh reproducible matched benchmarks remain open.
The historical benchmark JSON is retained with its acceptance
status withdrawn; recorded unequal fit counts cannot support the previous
matched-compute superiority claim.

Verification commands and exact evidence are in the completion status and
`scripts/run_m6_review_checks.sh`. Neither offline fixtures nor test totals imply
complete autonomous-product acceptance. Native Linux hosts, amd64 containers and
paid-provider operation remain uncertified by this review. No release was
published.
