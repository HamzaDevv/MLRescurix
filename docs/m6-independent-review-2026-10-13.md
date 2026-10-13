# Independent review of the M6 completion report

The supplied report does not justify M6 release acceptance. I reviewed its claims
against the repository and evidence, implemented the defects found, and ran local
control-plane tests plus Docker-only scientific tests. The decision is **NOT READY**.

## Defects repaired

| Boundary | Defect | Repair and regression |
| --- | --- | --- |
| Bundle authenticity | Missing pinned manifest receipt silently skipped authentication | Require authoritative receipt; Docker pipeline fixture deletes it and checks refusal |
| Finalization | A caller could select an environment different from the approved producer; queued decisions were not rechecked | Bind dataset/environment and validate current decisions at admission; substitution and queued revocation regressions |
| Sealed assessment | Crash after scoring started could leave the gate reserved and allow retry | Persist and fsync consumption before scoring; Docker fault injection and Rust recovery without a final result |
| Tool recovery | Side effect followed by a lost checkpoint could be executed twice | Refuse unknown tool outcomes; regression preserves an already-written file rather than overwriting it |
| Reconciliation | Rejected settlement receipts, malformed rows and absent authoritative actions could be ignored | Check applied status, propagate decoding failures and require action evidence; repair fixtures to use admitted actions |
| Cleanup | Deleting old command/action identities removed durable idempotency fences | Preserve identities; retry-after-80-commands and retained-action cycle regressions |
| Charge rollup | Malformed JSON and overflow could silently undercount consumption | Return an error instead of dropping or saturating amounts |
| Installed assets | Symlinked directories/files could redirect writes outside state; rename failure was ignored | Reject symlinks, fail staging safely and verify all bytes; external victim-file regression |

## Report corrections

The report listed wrong package versions and CPython 3.13.12 rather than 3.13.16.
It cited nonexistent lock/evidence paths, finalization functions and
`crates/harness/tests/matched_benchmarks_docker.rs`. Its fit-count claim conflicts
with its own benchmark JSON: values include 8, 9 and 11. Its storage proof relied
on deleting retry fences. Its alleged GC race test is sequential and its Rust
ENOSPC-named test checks oversized output. The tests are renamed to describe their
actual coverage; Python Docker tmpfs ENOSPC remains real, separately scoped evidence.

The finite-cycle rerun reports metadata growth while preserving durable identity.
Removing a safety requirement to make the graph plateau would invalidate the proof.
The 128 MiB constant was only reported, not enforced; capability/policy output now
makes this limitation explicit.

README, release notes, completion status, session handoff and CLI capability output
now reflect pending release acceptance. Historical evidence is retained, including
the initial Python container run's native-dashboard-binary failure. The explicitly
scoped 44-test Docker rerun passed. See
[completion status](m6-completion-status.md) for exact logs, counts and release gates.

## Limits of this review

A safe refusal is not successful recovery. Missing automatic tool reconciliation,
indefinite metadata bounds, fresh installed full delivery and matched benchmark
validation remain open. The synthetic finalization fixtures have content-bound
fixture approvals; they do not authorize a real candidate. No paid calls, new
benchmarks or publication were performed. The original independent review in
`m6-review-next.md` remains preserved.
