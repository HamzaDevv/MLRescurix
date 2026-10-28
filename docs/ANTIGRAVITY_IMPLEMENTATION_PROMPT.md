# Antigravity implementation prompt

Paste the following into Antigravity with this repository open.

---

Work in `/Users/hamza/Developer/Imp_Projects/MLRescurix`. Implement the next unfinished session of the persistent architecture plan; do not stop at writing another plan.

Read `docs/IMPLEMENTATION_REVIEW_20261028.md` first, then `docs/ARCHITECTURE_BUILD_HANDOFF.md`, `docs/GENERAL_ML_DEVELOPMENT_PLAN_20261028.md`, `docs/CONTINUATION_GUIDE.md`, `docs/STATUS_AND_ROADMAP.md`, and applicable repository instructions. The review supersedes the historical completion claims in the handoff. Inspect the actual dirty working tree before editing. Existing uncommitted changes are intentional work to review and preserve, not files to reset. The architecture handoff records the latest product scope and overrides older cart/recommendation priorities.

The goal is a general ML engineering system that can handle custom neural architectures, sequential/deep models and compute-heavy training through a modular, resource-aware, resumable pipeline. The cart transformer was only an example. Do not implement it as a required feature or choose the five benchmark problems; I will provide five distinct problems later. Use small clearly labeled engineering fixtures for integration tests now.

Target three implementation sessions, two only if all gates can actually be met. Start with Session 1 unless the tracker and verifiable evidence show it complete. Implement the session's full vertical slice, run relevant tests, fix discovered issues, and update the persistent tracker. If context or environment limits prevent completion, record exact unfinished work and continue from it next session; never label mocks, CPU tests, or scaffold code as live GPU certification.

Extend the existing Rust governance/execution paths as the authoritative control plane; use Python for ML ecosystem worker operations. Move prototype responsibilities into existing Rust services only when an integrated, tested replacement is ready. Do not create a parallel privileged orchestrator, bypass the governed gateway, silently coerce custom tasks into tabular models, or import generated model code into the privileged process. Keep task adapters, training plugins, and execution backends independent. Add real optional PyTorch custom training and bounded data loading; implement resource preflight, full training-state checkpoints and correct resume, durable Kaggle job reconciliation, bounded OOM recovery, budgeted tuning, and complete inference bundles according to the plan's session gates.

Use the researched shortlist selectively: PyTorch, optional Accelerate, Optuna, suitable streaming/Arrow readers, uv environment locks, official Kaggle CLI/API, optional Transformers/PEFT. Verify compatible released Python/CUDA/package versions against actual workers. Do not install every shortlisted framework or assume FSDP, quantization, TPU support, or distributed execution works without evidence. Keep the lightweight core install working.

Fix the identified Apex eligibility/pinning and metric-consistency issues early. Integrate OpenRouter primary inference and DeepSeek search/fetch plus policy-controlled explicit/fallback inference through existing governance, correct accounting and credential separation. Review the existing provider/search/Kaggle prototypes instead of assuming they are integrated.

Interpret “never choke” as bounded memory/time/retries, durable recovery, resource-based offloading and actionable `needs_resources` results. Do not promise arbitrary models fit finite GPUs. Do not reduce sequence length or model semantics silently. Checkpoint publication must survive a new remote job; local files inside an active notebook alone do not prove remote durability. Treat separate Kaggle accounts as independently authorized batch workers, not a single shared-memory cluster or automatic doubled quota.

Implement and test offline by default. This prompt does not authorize a new paid API budget, live Kaggle jobs/data uploads, competition submissions, or historical holdout evaluation. Preserve the historical ledger/folds and reconcile changed checkpoint bindings explicitly. Complete offline implementation and fault tests before listing a concrete live verification request if needed. Public documentation research is allowed; never expose keys or upload provider credentials to workers.

At the end, update `docs/ARCHITECTURE_BUILD_HANDOFF.md` and relevant canonical continuation/status docs with changed paths, actual tests/results, version locks, evidence locations, measured capabilities, pending live/GPU checks, remaining defects and the exact next session task. Report what works, what was tested, and what remains. No fake completion claims and no unrelated rewrite.

Begin now with source inspection and implementation of the next unfinished session.
