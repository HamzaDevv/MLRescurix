# Paste-ready continuation prompt

Use this in a fresh local coding session for Phase P4 (Agent-first tools and executable candidates).
Phases P1, P2, and P3 are complete, evidenced, and their acceptance gates are closed.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix.
Authorized scope: implement Phase P4 (P4.1–P4.3) and work toward the Phase P4 acceptance gate as defined in docs/v2/ROADMAP.md, ARCHITECTURE.md, DECISIONS.md, and PROGRESS.md. Do not begin P5 or later phases.

Start by reading docs/v2/README.md, PROGRESS.md, ROADMAP.md, DECISIONS.md,
ARCHITECTURE.md, and docs/m6-completion-status.md. Inspect git status and recent
history before changing anything. Preserve all existing work and historical evidence.
Current expected schema is user_version 10; verify the checkout and existing P1-P3 evidence.

Deliverables for Phase P4:
1. P4.1: Versioned workspace_fs, sandbox_exec, evaluate_submission, and artifact_crud adapters;
   retain existing operations during migration. Add capability discovery. Keep governed
   delegation/lifecycle/finalization controls. Optional dataset_query and experiment_tree
   convenience surfaces using the same authority boundaries. Default to immutable /workspace:ro,
   bounded task-local scratch/outputs, and reviewed source commits through workspace_fs.
   Bind scientific development and candidate prediction to the same approved digest-pinned
   environment profile; separately verify any distinct delivery inference profile.
2. P4.2: Define executable candidate manifest and CLI fit/predict protocol, raw-input schemas,
   stable row keys, class order, source/model/environment bindings, and diagnostic slice evaluation.
3. P4.3: Custom runner implementation, fallback isolation, evaluation without recipe modifications,
   and baseline candidate validation.
4. P4 Acceptance Gate: Agents can create and fix arbitrary candidate code without modifying a
   recipe registry; evaluation, scoring, and budgets remain under trusted authority; split/label
   isolation, single-use final test gate, and generation/intent fencing are strictly enforced.

Preserve trusted evaluator authority, split/label isolation, final-test single use,
generation/intent fencing, durable retry fences, resource leases, outbound filtering,
and approved-view controls. Keep managed state within the 128 MiB ceiling with zero
silent write exemptions. Maintain M0–M6 compatibility through the existing adapters.
Do not claim M6 readiness.

Run locked tests for the chosen work and relevant compatibility suites. Capture fresh
logs under `docs/evidence/v2/p4/<session>/`; leave `docs/evidence/m6/` untouched.
Update PROGRESS.md with exact results, evidence, limitations, and next slice; update
DECISIONS.md when design changes. Regenerate `docs/v2/ARGON_REVIEW_PACKET.txt` with
`python3 scripts/build_argon_review_packet.py` after source-document changes. Do not
mark a phase complete until its roadmap gate passes. Report implemented, verified,
and unverified items separately.
```
