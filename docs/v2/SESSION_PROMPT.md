# Paste-ready continuation prompt

Use this in a fresh local coding session for the remainder of Phase P3 (P3.2–P3.5).
P2 is complete and verified, and P3.1 (capability discovery, presets, headroom,
and infeasible-request rejection) is complete and evidenced.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix.
Authorized scope: implement the remainder of Phase P3 (P3.2–P3.5) and close the
Phase P3 acceptance gate as defined in docs/v2/ROADMAP.md, ARCHITECTURE.md,
DECISIONS.md, and PROGRESS.md. Do not begin P4 or later phases.

Start by reading docs/v2/README.md, PROGRESS.md, ROADMAP.md, DECISIONS.md,
ARCHITECTURE.md, and docs/m6-completion-status.md. Inspect git status and recent
history before changing anything. Preserve all existing work and historical evidence.
Current expected schema is user_version 10; verify the checkout and existing P1-P3.1
evidence.

Deliverables for completing Phase P3:
1. P3.2: Atomic global/project capacity reservations, weighted fair queues with aging,
   bounded queues, and stable tie ordering. Distinguish concurrency caps from CPU/RAM
   allocation. Persist lease, heartbeat, and recovery transitions.
2. P3.3: Route generic tools, preflight, evaluation, benchmarking, final fit, inference
   checks, and environment jobs through the broker. Apply thread caps everywhere,
   including estimator n_jobs and nested fold/HPO execution.
3. P3.4: Pressure backoff, per-project pause/stop isolation, orphan reconciliation,
   and resource-release-on-reap. Preserve capacity for finalization/control-plane work.
   Match persisted container IDs/owner labels against active epochs; do not kill
   another supervisor's valid worker merely because its container has harness labels.
4. P3.5: Multi-project stress and resilience verification (ten projects x five queued
   intents, mixed-size jobs, starvation bounds, worker crash, and cancel/restart).
   Attach aggregate allocation/resource timelines and queue waits.
5. P3 Acceptance Gate: Admitted sums stay within declared capacity; a large queued
   job receives service under the documented policy; one project's cancellation
   leaves others running; no runner bypasses leases. Use synthetic/small jobs;
   do not intentionally exhaust host RAM.

Preserve trusted evaluator authority, split/label isolation, final-test single use,
generation/intent fencing, durable retry fences, resource leases, outbound filtering,
and approved-view controls. Keep managed state within the 128 MiB ceiling with zero
silent write exemptions. Maintain M0–M6 compatibility through the existing adapters.
Do not claim M6 readiness.

Run locked tests for the chosen work and relevant compatibility suites. Capture fresh
logs under `docs/evidence/v2/p3/<session>/`; leave `docs/evidence/m6/` untouched.
Update PROGRESS.md with exact results, evidence, limitations, and next slice; update
DECISIONS.md when design changes. Regenerate `docs/v2/ARGON_REVIEW_PACKET.txt` with
`python3 scripts/build_argon_review_packet.py` after source-document changes. Do not
mark a phase complete until its roadmap gate passes. Report implemented, verified,
and unverified items separately.
```
