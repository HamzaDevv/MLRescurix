# MLRescurix: remaining engineering plan

Reviewed 2026-10-29. This is a dependency-ordered plan, not a claim that every ML problem can be made error-free or run on a finite resource allocation. The user will choose five distinct real problems later; the fixtures below are engineering tests, not those benchmarks. Read `CORE_ARCHITECTURE_GROUND_TRUTH.md` and the latest independent review first.

## 1. Establish one safe local execution path (highest priority)

- Keep custom admission disabled until the following tests and review pass. Establish a threat model for user/generated code, trusted control state, sealed data, and worker outputs. Create per-job immutable input snapshots and narrowly scoped read-only mounts and writable output/scratch mounts. Do not mount a shared workflow, holdout or authoritative manifests writable into development workers.
- Bind raw snapshot bytes and canonical split/protocol to the registered dataset and job before execution, without a hash/reopen race. Stream CSV records with explicit limits and correct row semantics; use atomic, per-job development artifacts and cleanup. Check multiline/escaped quotes, blank records, malformed data, different dataset identities, concurrency, symlinks, mutation and interrupted writes. Test that a hostile worker cannot discover holdout rows or alter another job's state.
- Make a trusted evaluator independently compute or validate metrics from candidate predictions and frozen validation inputs. Atomically bind job, candidate, dataset, split/protocol, source, environment and weights to evaluation and promotion; never trust a worker claim alone. Check stale, fabricated, missing and cross-job/cross-dataset receipts and replay attempts. Keep final holdout separately gated.
- Make the container image buildable and usable from a clean machine for supported CPU platforms with a dependency lock and documented acquisition procedure. Add GPU-specific profile/probes separately. Test a real local synthetic train → interruption → resume → evaluation → export, plus crash and timeout paths. Re-enable admission only after independent review and the adversarial tests pass.

## 2. Govern every provider and search request

- Route OpenRouter primary and configured DeepSeek fallback/explicit requests through one authoritative gateway. Disallow silent ungoverned direct fallback. Reserve against a defensible upper bound per potentially billable attempt, persist dispatch intent, treat ambiguous failures as uncertain until reconciled, and propagate settlement errors. Avoid claiming estimated costs are invoices. Verify routes, models, prices and actual API capabilities at integration time.
- Treat web search as a separately verified capability. DeepSeek model API access alone does not establish search API availability; use a supported provider or clearly report unavailable. Apply content/privacy rules to all outbound routes.
- Exercise retries, duplicate IDs, concurrency, failures after dispatch, stale pricing, no authority configured, missing credentials and exhausted budgets with offline mock transports. Do not make paid calls during this implementation phase.

## 3. Build a real remote execution adapter

- Integrate one backend interface with the existing supervisor: preflight, stage, submit, inspect/reconcile, collect and explicit cancellation capabilities. Implement a real, supported Kaggle CLI/API transport, not just `KaggleApi` mocks. Persist job intent and attempt identity before submission; reconcile unknown results before retry; fence leases transactionally.
- Stage a verified worker, compatible Python/dependencies and permitted data, with bounded secret scanning, explicit file allowlists, no symlinks/path escape, and manifest/hash validation. Validate downloaded archives and paths before extraction. Implement checkpoint publication and new-job resume only where Kaggle actually supports it; otherwise surface the limitation.
- Discover real quota, accelerator, runtime and account capabilities rather than relying on hard-coded defaults. Respect competition rules and provider terms; never promise that a second account increases allowed competition quota or bypasses restrictions. Treat both account profiles as optional configuration, with no cross-account access assumptions.
- Test offline with an injected transport: ambiguous submission, stale leases, restart, retries, cancellation limitations, quota exhaustion, malicious artifacts, tampering and secret leakage. Record the exact capabilities that still require an authorized live Kaggle run.

## 4. Resource-aware training and tuning

- Define resource requests and backend capabilities for CPU, RAM, VRAM, disk, time, accelerator and network; admit only when measured or conservatively estimated capacity fits. Bound retries and return `needs_resources` with actionable details when work cannot fit. Exercise memory pressure, OOM evidence and recovery; exit 137 alone is not proof of OOM.
- Implement checkpoint identity, atomic writes and resumability for each supported candidate, including optimizer, scheduler, sampler and RNG state when applicable. State precisely which CPU/GPU cases have deterministic replay versus best-effort resume. Verify interruption, corruption, mismatch, and continuation without silently restarting.
- Integrate Optuna with durable trial and lease identities under the same authority as execution. Hard trial deadlines require killable supervised processes; detecting elapsed time after a blocking function returns is insufficient. Reserve and settle real projected costs and resource quotas before dispatch, including parallel trials and retries; avoid cross-machine shared SQLite assumptions. Test process restarts and concurrent reservations.

## 5. Task adapters, delivery, and offline readiness

- Define stable task, candidate, training and prediction contracts independent of backend. Demonstrate tabular and sequential data plus an extensible custom-network path with small synthetic fixtures. Avoid flattening sequences or claiming that GRU prototype tests prove transformer, transfer-learning, multimodal or arbitrary-network support. Add optional PyTorch/PEFT/Transformers only for real use cases and versioned environments.
- Export the actual preprocessing and inference graph with explicit schema, dependencies, model weights and provenance. Check adversarial input, prediction parity, clean independently provisioned environment, missing dependency behavior and integrity. Document supported formats and measured RAM/latency; do not call a package-dependent bundle zero-dependency.
- Run focused tests and then relevant full Rust and Python suites, lint/format checks appropriate to the repository, reproducible image checks, and end-to-end fault injection. Update the capability matrix with commands, evidence, limitations and mock-only versus live-only status. Preserve historical accounting and holdouts. Ask for independent code review before declaring readiness.

## Live phase, after offline gates

With the user, choose five distinct real problems and their metrics, data rights, validation and resource limits. Separately define each concrete remote job, permitted data, credentials, quota/time and budget; then run small live preflight, Kaggle offload/resume and provider tests where authorized. Record measured results and failures. No test can prove universal success on every future Kaggle or ML problem.
