# Paste-ready continuation prompt

Use this in a fresh local coding session for the next resource phase. P2 is
implemented through P2.6, but its broader acceptance gate remains open. Verify
that prerequisite before starting P3 runtime work.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix.
Authorized scope: audit and close only the remaining P2 acceptance checks listed
in docs/v2/ROADMAP.md and docs/v2/PROGRESS.md. If the P2 gate is fully evidenced,
implement P3.1 effective CPU, memory, disk and accelerator capability discovery,
resource request presets, and infeasible-request rejection. Do not begin P3.2 or
later phases.

Start by reading docs/v2/README.md, PROGRESS.md, ROADMAP.md, DECISIONS.md,
ARCHITECTURE.md, P2_1_ARTIFACT_DESIGN.md, and docs/m6-completion-status.md.
Inspect git status and recent history before changing anything. Preserve all
existing work and historical evidence. Check repository instructions. Current
expected schema is user_version 10; verify the checkout and current P2 evidence.

P2 acceptance checks still requiring explicit evidence include populated-state
upgrade preservation, partial CAS/filesystem/DB commit reconciliation, protected
references under concurrent GC, and event restart/reconnect consistency. First
determine which are already proven by existing evidence. Work on P3.1 only if
the complete P2 gate passes. Otherwise address only the smallest remaining P2
acceptance gap in this authorized scope, leave P3 unchanged, and report any gate
that needs additional authorization or external setup.

For P3.1, derive concrete acceptance checks from the roadmap and current runtime.
Measure effective host and container-VM limits rather than assuming host totals;
preserve required memory-certification refusal on unsupported environments.
Bound and validate resource presets, account for configured headroom, and reject
requests that cannot fit declared capacity before admission. Do not consume large
host resources or intentionally exhaust RAM. Do not run candidate/scientific code
on the host; approved digest-pinned Docker environments remain mandatory.

Preserve trusted evaluator authority, split/label isolation, final-test single
use, generation/intent fencing, durable retry fences, resource leases, outbound
filtering and approved-view controls. Keep managed state within the 128 MiB
ceiling with zero silent write exemptions. Maintain M0–M6 compatibility through
the existing adapters. Do not claim M6 readiness.

Run locked tests for the chosen work and relevant compatibility suites. Capture
fresh logs under `docs/evidence/v2/p2/<session>/` or
`docs/evidence/v2/p3/<session>/`, as applicable; leave `docs/evidence/m6/`
untouched. Update PROGRESS.md with exact results, evidence, limitations and next
slice; update DECISIONS.md when design changes. Regenerate
`docs/v2/ARGON_REVIEW_PACKET.txt` with
`python3 scripts/build_argon_review_packet.py` after source-document changes.
Do not mark a phase complete until its roadmap gate passes. Report implemented,
verified and unverified items separately.
```
