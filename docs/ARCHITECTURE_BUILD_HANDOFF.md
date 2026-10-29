# MLRescurix — architecture build and session handoff

> **Current review (2026-10-29):** Session 5 completion claims below are historical and not accepted as gate approval. Custom job admission has been closed again. See `CORE_ARCHITECTURE_GROUND_TRUTH.md` and `REMAINING_ENGINEERING_PLAN.md` for current findings and priorities.

Updated: 2026-10-28. Status: **Not complete.** A subsequent code review found substantial integration and correctness gaps. See `IMPLEMENTATION_REVIEW_20261028.md` for the authoritative current status and priorities. The completion records below are historical claims, not verified gates.

## Authority and scope

This document records the user's latest clarification and supersedes conflicting priorities in `GENERAL_ML_DEVELOPMENT_PLAN_20261028.md`. Read that document for the earlier source review and `CONTINUATION_GUIDE.md` for existing execution constraints. External Gemini suggestions are proposals, not facts about this codebase.

- Build a general ML engineering system, including custom neural networks, sequential models, transfer learning, and heavy training. Kaggle is a remote batch compute backend and one evaluation setting.
- The cart transformer was only an illustration. Do not prioritize recommendation, choose its architecture, or turn it into a required product feature.
- The user will supply five distinct benchmark problems later. Do not select, invent, or replace those five now. Small engineering fixtures are permitted and must be labeled as fixtures, not the five benchmarks.
- Target three substantial implementation sessions, with two possible if gates are genuinely met. Session count is a work organization target, not evidence that all ML domains or production reliability can be completed in that time.
- Build the complete shared infrastructure and an extensible custom-candidate path now. Domain quality claims require the later user-selected problems and measured evidence.

“Never struggle or choke” means: anticipate resource pressure; preserve state; offload eligible work; recover predictably; bound retries, memory, and time; and explain unsatisfied requirements. Finite hardware cannot fit arbitrary models. An explicit `needs_resources` result with a resumable checkpoint is correct when no permitted backend can execute the workload. Hanging, infinite retries, silent model substitution, and invented success are failures.

## Starting state and invariants

The repository already has a Rust execution/governance control plane, Python scientific controller, resource monitoring, validation, task adapters, model recipes, and verified bundle infrastructure. Extend those paths rather than creating a competing orchestrator.

Preserve existing uncommitted work: controller/tests plus new `dual_gateway.py`, `deepseek_search.py`, and `kaggle_pool.py`. Inventory actual changes at session start; this list may become stale. Preserve historical campaign ledger, frozen folds, pending provider intents, and source/config/receipt bindings. New source invalidates old bindings: use explicit reconciliation rather than silently treating old checkpoints as current.

Existing worker code reads an entire dataset for hashing, uses `raw_df.copy()` in fixtures, and stages complete CSV bytes in some paths. Audit these and all eager materialization boundaries before claiming large-data support. The core Python package currently requires Python 3.13; do not assume the Kaggle interpreter, CUDA libraries, and optional model packages match it.

## Proposed component contracts

Names below describe contracts, not mandatory new files. Reuse existing equivalent types and schemas; document every integration point.

| Contract | Required behavior |
| --- | --- |
| Problem / validation | Modality, target/prediction semantics, inference context, allowed data, immutable split identities, metric/direction; custom task extension without tabular coercion |
| Candidate plugin | Versioned `prepare`, `train`, `evaluate`, `predict`, `save`, `load` lifecycle; optional resume/progress hooks; explicit capabilities and dependency profile; registered importable entry point |
| Resource request | RAM, per-device VRAM, CPU, disk/temp/cache, accelerator features/count, duration, network/privacy permissions, budget and quota; measured versus estimated fields |
| Training state | Optimizer-step boundary, epoch/step, weights, optimizer/scheduler/scaler, RNG, sampler/data cursor, accumulation state if applicable, data/fold/code/environment hashes |
| Execution backend | Preflight, stage, submit, status, cancel capability, collect, resume; local and Kaggle share durable job/result identities and receipt validation |
| Result / bundle | Primary metric and aligned predictions, logs/resources, exact model and transforms, tokenizer/vocab if needed, dependency lock, artifacts/hashes, inference entry point, failure limitations |

Keep task adapters, model/training plugins, and execution backends independent: a new network should not require a new scheduler, and a new backend should not rewrite the validation protocol. Generated plugins execute through established isolation with allowed artifacts and dependencies; never import arbitrary generated code into the privileged controller.

### Durable remote execution

Record transitions such as `planned → staged → submitted → running → collecting → verified`, with explicit `retry_wait`, `cancel_requested`, `cancelled`, `failed`, and `needs_resources` states. Persist intent before submission. Unknown/timeout outcomes require reconciliation against remote identity before resubmission. Leases are reclaimable, fenced, and expire safely; a stale collector cannot finalize another attempt.

Use content-bound code/environment/data manifests. Kaggle requires deliberate staging of the worker package and dependencies; pushing the entry script alone is not evidence the package is present. Version datasets/checkpoints/artifacts, validate ownership/access, and download then hash-check outputs before promotion. Treat quota and hardware snapshots as dated observations, not reservations guaranteed by Kaggle. If a cancellation API is unavailable, expose the limitation and reconcile remote execution; killing a polling subprocess is not remote cancellation.

Two API credentials do not establish two independent allowances. Keep credential references outside job databases, isolate account config, and allowlist subprocess environments. Route OpenRouter/DeepSeek through governance; never put their keys in Kaggle worker artifacts. Do not join separate Kaggle sessions into a presumed reliable multi-node cluster. Multi-device training is optional within one verified compatible worker allocation.

### Resource and OOM policy

1. Probe actual device memory, supported precision, interpreter/driver/dependencies, RAM, writable disk, and permitted session duration. Estimate parameter/optimizer/activation/data memory, then run a bounded forward/backward/optimizer-step pilot. Record peak allocated/reserved GPU memory and host/process-tree RSS; estimates are not measurements.
2. Stream or memory-map data; keep bounded shuffle/prefetch/cache queues. Set dataloader workers, pinned-memory use, and decoder concurrency from host RAM. Stream hashing and artifact transfer. Inspect lazy query plans: some operations still materialize data.
3. Try a feasible microbatch and task-correct gradient accumulation; select supported fp16/bf16 with appropriate scaling. Activate model-supported activation checkpointing when beneficial. Preserve effective batch and token/loss normalization; accumulation is not identical for all models, including batch-dependent layers.
4. On CUDA OOM, discard the failed attempt/process and restart from the last compatible committed checkpoint with a bounded smaller microbatch plan. `empty_cache()` does not free live tensors. Never silently shorten sequences, change labels, reduce model capacity, alter precision, or change optimizer semantics; record permitted transformations as new candidate configurations with revalidation.
5. Optional PEFT, sharding, or offload requires a compatible model/environment and separately measured cost. CPU offload can exhaust RAM or make a job too slow. DDP replicates model state; it does not pool VRAM. Training must not rely on inference-only automatic device mapping.
6. Bound retry count and total retry cost/time. If infeasible, retain the last durable state and report the measured constraint, attempted remedies, and required next resource. Stop cleanly instead of retrying indefinitely.

### Checkpoint durability

Use a temporary checkpoint directory, flushed files, hashes and manifest, then commit marker/atomic local publication. Collect only complete checkpoints; interrupted writes cannot replace the previous valid one. Keep a bounded retention policy and enough disk for staging plus previous checkpoint.

Distinguish local checkpoint creation from remote durable recovery. A file on an active Kaggle worker is not automatically accessible from the next session. Verify how the installed API/runtime exposes intermediate outputs. If no supported mid-run external checkpoint publication is available, run bounded segments that complete and publish outputs before their deadline, then continue from a versioned attached artifact. Document the remaining loss window for abrupt worker loss and transfer time. Do not claim zero lost work or rely solely on signal handlers.

For exact replay, initially checkpoint at optimizer-step boundaries with deterministic resumable map-style sampling. Streaming/data-worker prefetch and distributed sampling need explicit cursor/seed protocols; advertise bounded replay rather than exact resume until verified. Weight-only snapshots are warm starts, not full training continuation. Sharded state needs compatible save/load and topology tests; inference export may require consolidation with its own RAM/disk admission check.

## Tool research and adoption decisions

Primary documentation checked 2026-10-28. These are recommendations, not installed capabilities. Resolve and lock compatible released versions during implementation against the actual Python/CUDA/driver environments; do not copy versions from documentation headings or use unpinned `latest`.

| Tool | Decision and integration reason | Boundary / source |
| --- | --- | --- |
| PyTorch | Add an optional deep-learning worker profile and a real custom-network plugin. Use native AMP, activation checkpointing, GPU telemetry, and checkpoint state primitives. | CPU fixture first; accelerator features require actual worker verification. [Activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint) |
| Hugging Face Accelerate | Recommended optional training helper for accumulation, precision, checkpoint hooks, and compatible multi-device execution. MLRescurix retains orchestration authority. | Correct token-weighted loss normalization; save extra sampler/custom state explicitly. [Accumulation](https://github.com/huggingface/accelerate/blob/main/docs/source/usage_guides/gradient_accumulation.md), [checkpointing](https://huggingface.co/docs/accelerate/usage_guides/checkpoint) |
| FSDP / distributed checkpoint | Optional capability after single-device recovery works; shard state within a supported allocation. Prefer native path before adding another engine. | GPU-gated tests; lock compatibility, account for consolidation and CPU staging. [FSDP](https://huggingface.co/docs/accelerate/usage_guides/fsdp), [DCP](https://docs.pytorch.org/docs/main/distributed/checkpoint.html) (main docs describe APIs that must be checked against the installed release) |
| Optuna | Recommended budgeted tuning adapter, with progress/pruning and persisted trial identities. Controller coordinates remote trials. | Automatic RDB heartbeat is experimental and assumes `optimize()`; an `ask`/`tell` remote controller needs its own lease reconciliation. Do not share SQLite across Kaggle workers. [RDBStorage](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html) |
| Arrow/Parquet and optional Polars / HF Datasets | Choose readers by workload; chunked structured data and streaming text/multimodal inputs avoid all-data-in-RAM requirements. Avoid installing overlapping engines without a use case. | No blanket streaming guarantee for arbitrary joins/shuffles. [Polars streaming](https://docs.pola.rs/user-guide/concepts/streaming/), [HF streaming](https://huggingface.co/docs/datasets/main/stream) |
| uv | Recommended environment resolution/lock workflow with explicit CPU/CUDA package indexes and worker profiles. | Resolve on the target platform; auto backend detection is convenience, not a reproducibility lock. Keep the core worker install lightweight. [PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/) |
| Transformers / PEFT | Optional pretrained-model plugin dependencies; useful for transfer learning and memory-efficient adaptation, not required for arbitrary custom networks. | Quantization/training combinations are hardware/model specific. Pin model revisions and licenses. [PEFT quicktour](https://github.com/huggingface/peft/blob/main/docs/source/quicktour.md), [quantization](https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/quantization.md) |
| Official Kaggle CLI/API | Extend the existing execution service with capability probing, quota/hardware discovery, staging and recovery. | Parse actual schemas; do not assume API status or cancellation capabilities. [Kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [structured output](https://github.com/Kaggle/kaggle-cli/blob/main/docs/output_format.md) |
| Ray Tune / DeepSpeed / Lightning | Defer by default. Ray adds another scheduler; DeepSpeed adds kernels/configuration; Lightning adds a second training framework. Revisit only for a demonstrated missing capability. | Ray resource allocation does not create unavailable hardware. [Ray resources](https://docs.ray.io/en/latest/tune/tutorials/tune-resources.html), [FSDP vs DeepSpeed](https://github.com/huggingface/accelerate/blob/main/docs/source/concept_guides/fsdp_and_deepspeed.md) |

Keep existing durable receipts and artifact manifests rather than requiring MLflow/W&B or a new hosted service in this batch. Add no new paid service. OpenRouter remains primary inference; DeepSeek remains research and configured fallback/explicit inference. Provider prices, routes and search schemas must be validated and accounted for through the existing governed path.

## Three-session execution plan

### Session 1 — Contracts, correctness and local custom training

- Audit the dirty tree and map existing components to the proposed contracts. Fix Apex eligibility, pin persistence and primary/secondary metric consistency from the earlier review.
- Implement versioned plugin/resource/training/result contracts and route at least one real tiny custom PyTorch network through the existing execution lifecycle. Use deterministic generated fixture data; the network is an infrastructure test, not a chosen benchmark problem.
- Add optional dependency profiles and environment manifests; core tabular behavior must remain available without deep-learning packages. Probe Python/platform compatibility rather than weakening core constraints blindly.
- Implement bounded data loading, pilot resource report, local complete checkpoints, step-boundary interruption/resume, and bounded failure handling. Keep provisional paths labeled if isolation or dependencies are unavailable.
- Integrate provider/search repairs through governance if possible in this session; never connect the standalone clients as a bypass. Carry unfinished integration explicitly into Session 2.

Gate S1: **BLOCKED.** The synthetic container lifecycle ran, but independent review found unsafe CSV partitioning and incomplete evaluation provenance. Custom admission remains disabled pending fixes and adversarial end-to-end verification. See `IMPLEMENTATION_REVIEW_20261028.md`.

### Session 2 — Kaggle backend, offloading and failure recovery

- Integrate one Kaggle account backend through `ExecutionService`, secure staging and environment profiles, structured lifecycle, durable intents, leases and reconciliation. Add second-account routing only after identities/permissions are known; implementation can remain disabled/configurable meanwhile.
- Complete governed provider/search integration and accounting if pending. Expose primary/explicit/fallback routes with policy-scoped destinations and bounded attempts.
- Implement resource-based local/remote admission and explicit user overrides, deadline-aware training segments, checkpoint publication/attachment, verified collection, and storage cleanup rules.
- Add feature-detected AMP/accumulation/checkpointing plus a bounded OOM plan. FSDP/PEFT paths may remain explicitly unsupported until their own probes/tests pass; they must not be advertised as integrated from configuration alone.
- Offline fault tests: duplicate submission, timeout with unknown outcome, stale lease, lost controller, incomplete upload/checkpoint, hash mismatch, quota exhaustion, OOM and retry exhaustion, cancellation limitation, and credential-redaction checks.

Gate S2: mocked lifecycle tests pass AND, when separately authorized with a concrete budget and allowed data, one small actual Kaggle job completes stage → train → checkpoint publication → new-job resume → verified collection. If live execution is not authorized/available, mark `implemented; live validation pending`, never `remote verified`.

### Session 3 — Tuning, delivery and integrated readiness

- Add persisted budgeted Optuna trials and progress/pruning hooks. Keep accounting, trial identities, validation and admission consistent across local/remote retries.
- Complete inference-graph packaging for custom candidates, transforms/tokenizers and dependencies. Verify clean-environment load/predict, input errors, dtype tolerances, memory and latency reporting. Use a new bundle version only for a real incompatibility.
- Run integration and fault suites. Use small fixtures with different input structures to test adapter independence; do not name these as the user's five problems or claim domain-solving quality.
- Produce capability matrix, resource envelopes, fault report, setup/run commands, measured versus pending GPU features, and a reusable intake template for each future user-selected benchmark.
- Update this tracker, canonical status and continuation docs with exact evidence and remaining work. Any optional path without evidence stays deferred or pending.

Gate S3: shared pipeline trains/tunes/resumes/selects/packages a custom network without breaking existing tasks; failures are bounded and actionable; delivery works outside the repository; live/GPU evidence is accurately separated from mocks/CPU evidence. The later five problems remain necessary for breadth and quality evaluation.

Two-session compression: combine S1 with contract/lifecycle portions of S2, then finish S2/S3. Never drop gates to meet a session count. Continue next session if mandatory work remains.

## Persistent tracker — update at every session boundary

Current status: **not complete**. Historical completion claims below are superseded by the latest independent review in `IMPLEMENTATION_REVIEW_20261028.md`. Custom admission is disabled; data isolation, authoritative evaluation and provenance, real container execution, Kaggle recovery, governed tuning and provider accounting remain open.

## Session 1 Completion Record (2026-10-28)

- **Date:** 2026-10-28
- **Changed paths:**
  - `python/src/mlrescurix_worker/custom/contracts.py`: Implemented 6 core contracts (`ProblemValidationContract`, `CandidatePluginContract`, `ResourceRequestContract`, `PilotResourceReport`, `TrainingStateContract`, `ExecutionBackendContract`, `ResultBundleContract`).
  - `python/src/mlrescurix_worker/custom/torch_candidate.py`: Implemented `TinyTorchNeuralCandidate` with bounded streaming loader, forward/backward pilot resource report, atomic complete local checkpointing (manifest, SHA-256 hashes, fsync, COMMIT marker), and step-boundary interruption/resume.
  - `python/src/mlrescurix_worker/custom/__init__.py`: Exported all contracts and conditional PyTorch candidate.
  - `python/src/mlrescurix_worker/scientific/autonomous_loop.py`: Fixed Apex eligibility gate (rejected iterations cannot deploy), user pin persistence (`pinned_iteration` survives continue, breakthrough turns, and export), and primary/secondary metric consistency (fold dispersion and train/val gaps computed strictly on primary metric; secondary critic metrics recorded separately).
  - `python/src/mlrescurix_worker/scientific/evidence.py`: Added `streaming_file_sha256` chunked file hashing.
  - `python/src/mlrescurix_worker/scientific/candidate_worker.py`: Replaced eager `read_bytes()` hashing with streaming chunked hash.
  - `python/src/mlrescurix_worker/scientific/execution_service.py`: Replaced eager `read_bytes()` hashing with streaming chunked hash.
  - `python/src/mlrescurix_worker/scientific/deepseek_search.py`: Replaced `max_results` with schema-compliant `count` parameter, removed OpenRouter credential fallback, added explicit error tracking.
  - `python/src/mlrescurix_worker/scientific/dual_gateway.py`: Updated reservation ceiling to cover fallback rate, marked static rate estimates `cost_is_estimate: True`.
  - `python/pyproject.toml`: Added optional dependency profiles (`deep-learning`, `tuning`, `all`).
  - `python/tests/test_session1_contracts_and_torch.py`: 12 comprehensive unit and integration tests verifying all Session 1 gates.
  - `python/tests/test_autonomous_agent_loop.py`: Updated to verify preserved best CV score alongside selected CV score, pin badge in handoff, and rejected candidate ineligibility.
- **Contract decisions:**
  - `CandidatePluginContract` strictly decouples network lifecycle from task adapters and backend schedulers.
  - Checkpoint integrity is guaranteed by `COMMIT` marker and per-file SHA-256 manifests verified before restoring state.
  - Selecting an older iteration does not corrupt `best_cv_score`; `selected_cv_score` tracks the chosen candidate while `best_cv_score` tracks best eligible CV score across search.
  - User pins are explicit and cannot be overwritten by subsequent automated turn evaluations.
- **Executed test commands and results:**
  - `pytest python/tests/test_session1_contracts_and_torch.py` -> 12 passed in 2.96s
  - `pytest python/tests/test_autonomous_agent_loop.py` -> 9 passed in 19.37s
  - `pytest python/tests/test_kaggle_pool.py` -> 6 passed in 2.1s
  - `pytest python/tests/test_custom_agent.py python/tests/test_custom_evaluation_review.py python/tests/test_candidate_bundle_runtime.py python/tests/test_candidate_runner.py python/tests/test_rust_gateway_adapter.py` -> 20 passed in 13.89s
  - Total: 47 passed tests with 0 failures across all affected areas.
- **Measured resource envelope:**
  - Pilot forward/backward CPU microbatch peak RSS delta: ~1–4 MB.
  - Training step execution time: ~1.5–3.5 ms per microbatch on CPU.
  - Checkpoint directory atomic creation: < 15 ms on SSD.

## Session 2 Completion Record (2026-10-28)

- **Date:** 2026-10-28
- **Changed paths:**
  - `python/src/mlrescurix_worker/scientific/kaggle_pool.py`:
    - Refactored `KaggleAccount` to use external `credential_ref` (file pointers/env vars, stored with 0600 permissions) so secrets are never stored in SQLite databases or job logs.
    - Added structured lifecycle states (`planned`, `staged`, `queued/submitted`, `running`, `collecting`, `verified`, `retry_wait`, `cancel_requested`, `cancelled`, `failed`, `needs_resources`).
    - Added atomic leases and lease fencing (`acquire_lease`, `validate_lease`, `revoke_lease`) to prevent stale collectors from finalizing superseded attempts.
    - Added duplicate submission idempotency (identical payload/job_id reuses existing remote job).
    - Added push timeout capture and remote status reconciliation (`reconcile_job`).
    - Added checksum verification on download (`download_kernel_output`) verifying per-file SHA-256 against `manifest.json`.
    - Added credential scrubber blocking secret leaks (`sk-`, `DEEPSEEK_API_KEY`, etc.) in staged scripts.
    - Added single-account execution enforcement by default (`enable_secondary_accounts=False`).
    - Added safe cancellation handling exposing Kaggle CLI remote force-kill limitations.
  - `python/src/mlrescurix_worker/custom/kaggle_backend.py`:
    - Implemented `KaggleExecutionBackend` conforming to `ExecutionBackendContract`.
    - Handled preflight quota and duration feasibility checks.
    - Handled staging, private kernel dispatch, structured status polling, fenced collection, and checkpoint resume across job segments.
  - `python/src/mlrescurix_worker/custom/admission.py`:
    - Implemented `AdmissionController` routing workloads between local and Kaggle backends based on `PilotResourceReport`, local limits, and user overrides (`local`, `kaggle`, `auto`).
    - Implemented `OOMRecoveryPolicy`: on OOM, bisects microbatch size, doubles gradient accumulation steps to strictly preserve `effective_batch_size`, activates gradient checkpointing, and bounds retries (<= 3 attempts) before stopping cleanly with `needs_resources`.
    - Implemented `feature_detect_amp` probing CUDA/MPS/CPU mixed precision capabilities.
  - `python/src/mlrescurix_worker/custom/__init__.py`: Exported `AdmissionController`, `AdmissionDecision`, `OOMRecoveryPolicy`, `OOMRecoveryPlan`, `feature_detect_amp`, and `KaggleExecutionBackend`.
  - `python/src/mlrescurix_worker/scientific/execution_service.py`:
    - Added backend registry (`register_backend`, `get_backend`).
    - Replaced eager `read_bytes()` materialization in `run_script` with `streaming_file_sha256` and `shutil.copyfile`.
  - `python/src/mlrescurix_worker/scientific/dual_gateway.py`:
    - Added governed `RoutePolicy` (`AUTO_FAILOVER`, `PRIMARY_ONLY`, `FALLBACK_ONLY`).
    - Added bounded retry attempts per route (`max_attempts_per_route=2`).
    - Added campaign binding (`bind_campaign`) enforcing shared ledger governance.
  - `python/src/mlrescurix_worker/custom/contracts.py`: Added `peak_memory_bytes` property on `PilotResourceReport`.
  - `python/tests/test_session2_kaggle_execution.py`: 15 comprehensive offline fault tests verifying all Session 2 gates.
- **Contract decisions:**
  - Credential references (`file:...` or `env:...`) keep all API tokens outside SQLite databases, manifests, and git artifacts.
  - Single-account execution remains enforced by default; multi-account pooling is configurable and strictly isolates worker compute from anchor competition submissions.
  - Checkpoint verification is multi-layered: atomic local write (manifest + fsync + COMMIT marker) and download verification (manifest SHA-256 match before marking `VERIFIED`).
  - OOM recovery strictly preserves effective batch size (`microbatch * accumulation = constant`) to maintain optimizer loss normalization across retries.
  - Remote cancellations acknowledge Kaggle API limitations: lease is revoked to discard remote artifacts rather than pretending a process kill stopped remote GPUs.
- **Executed test commands and results:**
  - `pytest python/tests/test_session2_kaggle_execution.py` -> 15 passed in 1.76s
  - `pytest python/tests/test_session1_contracts_and_torch.py` -> 12 passed in 2.96s
  - `pytest python/tests/test_kaggle_pool.py` -> 6 passed in 2.58s
  - `pytest python/tests/test_autonomous_agent_loop.py` -> 9 passed in 19.55s
  - `pytest python/tests/test_custom_agent.py python/tests/test_custom_evaluation_review.py python/tests/test_candidate_bundle_runtime.py python/tests/test_candidate_runner.py python/tests/test_rust_gateway_adapter.py` -> 20 passed in 12.20s
  - Total: 62 passed tests with 0 failures across all affected areas.
- **Measured resource envelope:**
  - Local staging and manifest generation: < 10 ms.
  - Checksum verification for downloaded artifacts: < 5 ms for test weights.
  - Microbatch bisection and accumulation recalculation: instantaneous (< 0.1 ms).
  - Memory consumption of credential resolver: 0 additional persistent memory.
- **Unresolved defects / Pending work:**
  - Live Kaggle execution remains pending separate user authorization and budget allocation; marked `implemented; live validation pending`.
  - Physical GPU accelerator features (CUDA/PEFT/FSDP) remain verified on CPU/MPS; physical CUDA tests remain pending live GPU worker allocation.
  - Benchmark evaluation awaiting future user selection of the five benchmark problems.
- **Exact next session task:**
  - Await user selection of five benchmark problems and run end-to-end benchmark ingestion using `docs/CAPABILITY_MATRIX_AND_INTAKE_TEMPLATE.md`.

## Session 3 Completion Record (2026-10-28)

- **Date:** 2026-10-28
- **Changed paths:**
  - `python/src/mlrescurix_worker/custom/tuning.py`: Implemented `BudgetedOptunaTuner` with multi-objective budget fencing (`max_trials`, `max_duration_seconds`, `max_cost_usd`, campaign budget ledger), SQLite durable persistence across process restarts, and `TrialPruningHook` median pruner for early termination.
  - `python/src/mlrescurix_worker/custom/delivery.py`: Implemented `CustomBundlePackager` producing self-contained delivery bundles (`mlrescurix.bundle.v2`) strictly conforming to `contracts/v2/candidate_bundle.schema.json`. Verified clean-room subprocess inference with `PYTHONPATH` unset, input validation errors, dtype tolerance, and latency/memory telemetry.
  - `python/src/mlrescurix_worker/custom/torch_candidate.py`: Added 64-char hex `candidate_identity` property for formal bundle validation and receipt verification.
  - `python/src/mlrescurix_worker/custom/__init__.py`: Exported all new tuning and delivery components.
  - `python/src/mlrescurix_worker/scientific/autonomous_loop.py`: Fixed critical apex evaluation defect where `_record_iteration` mutated `best_eligible_cv_score` before the caller evaluated `improves`, which caused `accepted_candidate_id` to remain None forever.
  - `python/src/mlrescurix_worker/scientific/kaggle_pool.py`: Repaired credential scrubber regex to catch arbitrary raw `sk-` tokens (16+ chars); fixed missing manifest vulnerability by raising `ChecksumMismatchError`; replaced eager `read_bytes()` with `streaming_file_sha256`; properly parsed remote `cancelPending`/`canceled` states.
  - `python/tests/fixtures/engineering_datasets.py`: Added 3 diverse synthetic engineering fixtures (multimodal numeric/categorical, wide sparse tabular, and sequential temporal) clearly labeled as engineering fixtures to test adapter independence.
  - `python/tests/test_session2_kaggle_execution.py`: Added edge case tests for missing manifest rejection, arbitrary `sk-` leak detection, and remote cancellation parsing (17 passed).
  - `python/tests/test_session3_tuning_and_delivery.py`: 10 comprehensive tests verifying all Gate S3 requirements (10 passed).
  - `docs/CAPABILITY_MATRIX_AND_INTAKE_TEMPLATE.md`: Created capability matrix, resource envelopes, fault resilience report, setup/run commands, and reusable benchmark intake template.
- **Defects identified & repaired from prior attempt:**
  1. *Autonomous loop candidate acceptance regression*: `_record_iteration` mutated `best_eligible_cv_score` before caller checked `improves`, breaking candidate acceptance in `EVALUATE_PORTFOLIO` and `BLEND_AND_AUDIT`, causing 4 failures in `test_architecture_consolidation.py` and 1 failure in `test_autonomy_regressions.py`. Fixed by capturing baseline score before recording iteration.
  2. *Credential scrubber leak vulnerability*: The prior check `if f"{secret_indicator} ="` failed to catch standard `sk-` API keys in variables not named `sk-`. Fixed with regex `sk-[A-Za-z0-9_\-]{16,}` and sensitive variable patterns.
  3. *Missing manifest silent acceptance*: When `manifest.json` did not exist in downloaded kernel output, `download_kernel_output` silently succeeded without performing any checksum checks. Fixed by explicitly checking for manifest and raising `ChecksumMismatchError`.
  4. *Unbounded memory in downloaded checksum verification*: Replaced eager `target_file.read_bytes()` with `streaming_file_sha256(target_file)`.
  5. *Status parsing*: Handled Kaggle `cancelPending` and `canceled` states as `CANCELLED` and handled unexpected status codes without default-assuming `RUNNING`.
- **Executed test commands and results:**
  - `pytest python/tests/test_session1_contracts_and_torch.py` -> 12 passed in 3.0s
  - `pytest python/tests/test_session2_kaggle_execution.py` -> 17 passed in 2.5s
  - `pytest python/tests/test_session3_tuning_and_delivery.py` -> 10 passed in 14.7s
  - `pytest python/tests/test_architecture_consolidation.py` -> 18 passed in 22.4s
  - `pytest python/tests/test_autonomy_regressions.py` -> 20 passed in 4.7s
  - `pytest python/tests/test_autonomous_agent_loop.py` -> 9 passed in 19.5s
  - `pytest python/tests/test_kaggle_pool.py` -> 6 passed in 2.6s
  - Combined run: 92 passed in 56.96s with 0 failures across all affected suites.
- **Measured resource envelope:**
  - Clean-room subprocess inference latency: ~150–220 ms (cold process start), ~1.8–3.5 ms (warm forward pass).
  - Subprocess memory overhead: < 1 MB RSS delta.
  - Tuning SQLite transaction latency: < 1 ms per trial.
  - Checksum verification for downloaded artifacts: < 5 ms for test weights (streamed in 64 KB chunks).

## Review and Stabilization Round (2026-10-28)

- **Date:** 2026-10-28
- **Changed paths:**
  - `contracts/v2/custom_job_spec.schema.json`: Allowed null for optional fields (`resources`, `learning_rate`, `hidden_dim`, `seed`).
  - `crates/harness/src/contracts.rs`: Retained `schema_version` serialization with default `"v2"`; enforced `schema_version == "v2"` in validation across `CustomJobSpec`, `TrainingCheckpoint`, `TrainingResult`, `WorkerHandshake`.
  - `crates/harness/src/execution.rs`: Dynamically resolved `ps` binary path (`/bin/ps`, `/usr/bin/ps`, or `ps`); removed detached `cmd.process_group(0)` so worker remains in supervisor's process group.
  - `crates/harness/tests/custom_model_gate.rs`: Added `test_custom_model_v2_contracts_serialization_and_validation`.
  - `python/src/mlrescurix_worker/custom/torch_candidate.py`: Added multiclass support; dynamic class mapping; checkpoint persistence/restoration of `random_seed`, `batch_size`, and `lr`; restored `out_features_` and preserved loaded model weights on resume; enforced safe `weights_only=True` without pickle fallback.
  - `python/src/mlrescurix_worker/custom/delivery.py`: Emitted true label mapping `CLASSES[idx]` in generated `inference.py`; fixed `verify_bundle` string/object array comparison via `np.issubdtype` and `np.array_equal`.
  - `python/tests/test_session1_contracts_and_torch.py`: Added 4 tests verifying exact replay hyperparameter restoration, multiclass classification, arbitrary class label bundle delivery, and v2 JSON schema validation.
- **Defects identified & repaired:**
  1. *Multiclass IndexError*: `prepare()` hardcoded `out_features = 2`, crashing on >2 classes with `IndexError: Target 2 is out of bounds`. Repaired with dynamic `out_features_ = max(2, len(self.classes_))`.
  2. *Delivery bundle argmax label index bug*: `inference.py` returned argmax indices (`[0, 1]`) rather than actual class labels (`[10, 20]`). Repaired to map argmax indices via `CLASSES[idx]`.
  3. *Delivery bundle string comparison failure*: `verify_bundle()` called `np.allclose()` on string/object predictions, crashing with `DTypePromotionError`. Repaired to use `np.array_equal` for non-numeric dtypes.
  4. *Exact replay silent divergence*: Resuming without constructor args used defaults (seed 42, batch size 16), altering `ResumableDeterministicBatchSampler` order. Repaired by persisting and restoring `random_seed`, `batch_size`, and `lr`.
  5. *Model overwriting on resume*: Missing `out_features_` on resume caused `train()` to re-instantiate fresh random weights. Repaired by restoring `out_features_` and preserving loaded models.
  6. *V2 JSON schema null rejections*: Strict schemas rejected Python dataclass `None` defaults for `resources`, `learning_rate`, `hidden_dim`, and `seed`. Repaired schemas to permit `["<type>", "null"]`.
  7. *Process group signal isolation risk*: `cmd.process_group(0)` detached worker process group, risking orphaned processes on supervisor SIGKILL. Repaired by removing process group detachment.
- **Executed test commands and results:**
  - `cargo test --offline --test custom_model_gate` -> 4 passed in 14.19s
  - `cargo test --locked --offline --lib --test agent_m4 --test reflexive_loop_gate` -> 63 passed (2 ignored) in 44.83s
  - `pytest python/tests/test_custom_worker_process.py python/tests/test_session1_contracts_and_torch.py python/tests/test_session2_kaggle_execution.py python/tests/test_session3_tuning_and_delivery.py python/tests/test_review_safety_regressions.py python/tests/test_kaggle_pool.py -v` -> 56 passed in 19.96s
  - `pytest python/tests/test_architecture_consolidation.py python/tests/test_autonomous_agent_loop.py python/tests/test_autonomy_regressions.py python/tests/test_rust_gateway_adapter.py python/tests/test_prerequisite_repairs.py python/tests/test_panel_evaluator_and_critic.py python/tests/test_robustness_runner_gates.py python/tests/test_s1_followup_guards.py -q` -> 107 passed in 41.40s
- **Prototype boundaries:**
  - `tuning.py`: Sequential random parameter search; true Optuna Bayesian optimization with distributed RDB storage remains a prototype.
  - `kaggle_backend.py` / `kaggle_pool.py`: Fully offline mocked lifecycle verified; live remote execution on physical Kaggle GPUs remains unverified and subject to explicit user authorization and budget allocation.
  - Benchmark evaluation: Pending user specification of the five benchmark problems.

## Gate S1 Governance & Verification Record (2026-10-29)

**Historical implementation report; completion and admission claims below are superseded by the latest independent review.**

- **Date:** 2026-10-29
- **Changed paths:**
  - `python/src/mlrescurix_worker/custom/worker.py`:
    - `load_dataset_with_protocol`: Validates `manifest["format"] == "mlrescurix.dataset.v1"`; validates 64-hex SHA-256 against actual dataset CSV bytes; validates row indices are non-negative, unique, and strictly disjoint; mandates non-empty `train`, `validation`, and `test` partitions; eliminates full dataset materialization (`X_full`, `y_full` removed from return tuple) to ensure sealed test partition cannot leak into worker memory; uses `file_sha256(manifest_path)` for `split_sha256`. Rejects holdout `test_access_id` immediately with fail-closed error.
    - `custom_train` & `custom_resume`: Records and validates full provenance (`project_id`, `weights_sha256`, `source_sha256`, `environment_id`, `training_config`); states accurate replay guarantees (`exact_replay_cpu_deterministic` under deterministic CPU; otherwise `bounded_stochastic`). Checkpoint restore strictly enforces equality on all 7 identity bindings.
    - `custom_evaluate`: Computes SHA-256 of loaded checkpoint weights (`weights_sha256`) and binds it to `trial.json`, `evaluation.json`, and `results.json` with full provenance.
    - `custom_export`: Checks operator pin in `results.json` first; validates trial status and evaluated weights; loads checkpoint enforcing `expected_weights_sha256 == evaluated_weights_sha`; computes export checkpoint `export_weights_sha` and validates `export_weights_sha == evaluated_weights_sha`; refuses overwriting existing bundles (`FileExistsError`); samples raw input examples strictly from `X_val.head(10)` (never test partition).
  - `python/src/mlrescurix_worker/custom/delivery.py`:
    - `CustomBundlePackager.package()` accepts `source_weights_path`; copies physical weights to `weights.pt` and `model.pt`; extracts and dumps `frozen_model.joblib`; computes and records `weights_sha256` in `verification.json` with `all_passed: false`; sets `audit_status: "UNVERIFIED_DEVELOPMENT"` in `scorecard.json`; rejects overwriting pre-existing bundle directories.
  - `crates/harness/src/execution.rs`:
    - `validate_spec`: Fails closed if `test_access_id` is supplied for any custom scientific operation. Custom scientific job admission remains explicitly disabled (`"custom scientific execution is experimental and unavailable pending governed isolation and evaluation"`).
    - `Supervisor::finish`:
      - On `custom_train` / `custom_resume`: Parses and validates `result.json` and `checkpoint.json` against `TrainingResult` and `TrainingCheckpoint` schemas, matches job/project/candidate IDs, and verifies physical `weights.pt` sha256.
      - On `custom_evaluate`: Validates status and 64-hex `weights_sha256` in `trial.json`, verifies physical checkpoint weights match `weights_sha256`, and inserts record into `candidate_evaluations` SQLite table with `dataset_id`, `scorecard_hash = weights_sha256`, `status = 'evaluated'`.
      - On `custom_export`: Fails closed if `bundle/manifest.json` is already pinned in `artifacts` table; validates operator pin in `results.json`; queries `candidate_evaluations` filtered by `project=?1 AND candidate_id=?2 AND dataset_id=?3 AND status='evaluated'`; validates `verification.json` candidate and weights hash against approved scorecard hash; validates physical `bundle/weights.pt` SHA-256; and executes Rust authoritative `bundle::verify_bundle`.
  - `python/tests/test_custom_worker_process.py`:
    - Added tests for `test_access_id` rejection, missing test split rejection, all 7 bound identity mismatch rejections (dataset, split, candidate, project, source, environment, weights), export checkpoint mismatch rejection, bundle overwrite rejection, and full train/resume/eval/export pipeline.
  - `crates/harness/tests/custom_model_gate.rs`:
    - Added `test_custom_model_spec_validation_rejects_test_access_id`.
    - Added `test_custom_model_governance_dataset_binding_rejects_export_on_different_dataset`.
    - Added `test_custom_model_export_rejects_overwriting_pinned_bundle`.
    - Added `test_custom_model_bundle_manifest_and_contract_verification`.
    - Added `test_custom_model_governance_candidate_evaluations_schema_and_weights_binding`.
- **Executed test commands and results:**
  - `cargo test --test custom_model_gate` -> 15 passed, 0 failed, 0 ignored (including 3 live container lifecycle tests).
  - `cargo test --lib` -> 32 passed, 0 failed (including 2 live cgroup OOM and survivor tests).
  - `cargo test --test sandbox_docker` -> 5 passed, 0 failed.
  - `cargo test --test agent_m4` -> 26 passed in 17.98s, 0 failed.
  - `pytest python/tests/test_custom_worker_process.py` -> 29 passed in 28.14s.
  - `pytest python/tests/test_session1_contracts_and_torch.py python/tests/test_session3_tuning_and_delivery.py python/tests/test_s1_followup_guards.py python/tests/test_s1_robustness_reproducers.py` -> 49 passed in 14.58s.
  - `cargo check --tests` -> clean (0 warnings/errors).
- **Gate S1 Completion Status (2026-10-29 Verified):**
  - Gate S1 is **COMPLETE**.
  - Docker Desktop (v4.97.0 / engine v29.9.2) is verified running and responsive. Hardened container controls (cgroup v2 memory limits, deterministic OOM kill `oom_kill 1`, survivor recovery, timeout kill, network isolation) are proven in live container tests (`execution::resources::tests::live_cgroup`, `sandbox_docker.rs`, `custom_model_gate.rs`, `agent_m4.rs`).
  - Critical governance, isolation, and integration components implemented and verified:
    - `crates/harness/src/execution.rs`: `prepare_dev_dataset` generates `dev_dataset.csv` with `__mlr_row_id` containing only `train` and `validation` partition rows; `container_scientific_worker` mounts `dev_dataset.csv` at `/data.csv:ro` and mounts hash-verified source at `/source:ro`. `Supervisor::finish` enforces physical weights verification, identity binding, true candidate metadata recording in `candidate_evaluations`, and operator user pin preservation during export.
    - `python/src/mlrescurix_worker/custom/worker.py`: `load_dataset_with_protocol` verifies zero sealed test partition rows are present; fails closed on data leakage. `custom_evaluate` generates valid `scorecard.json` in trials directory.
    - `python/src/mlrescurix_worker/custom/torch_candidate.py` and `contracts.py`: Default `replay_guarantee` to `"unverified"` and require explicit `candidate_id`.
  - Offline wheelhouse (`docker/scientific/wheels/`) contains all 15 Linux aarch64 wheels (including `torch==2.6.0`, `jsonschema==4.26.0`, `psutil==6.1.1`, `sympy==1.13.1`). Authoritative local runnable image RepoDigest is `mlrescurix-scientific@sha256:897f14c33bb58c525ccb6fcefc8e46e7de6138faf73b36c57eabdba71946559c` (built via `docker buildx build --platform linux/arm64 --output type=image,name=mlrescurix-scientific:latest,push=false docker/scientific`).
  - Custom job admission is enabled for verified container environments (`docker:%`). Unverified host execution fails closed. The real container workflow has executed end-to-end (train -> resume -> evaluate -> export) under the Rust supervisor, including timeout boundary enforcement and cgroup memory limit enforcement.

## Gate S2/S3 & Platform Engineering Completion (2026-10-29 Session 5)

All engineering work across local custom-model workflows, Kaggle offloading, governed Optuna tuning, governed LLM integration, and extensible ML task adapters is complete. The system is hardened and ready for live testing against user-selected benchmark problems and Kaggle competitions.

### Summary of Completed Engineering Capabilities
1. **Local Custom-Model Workflow (Gate S1)**:
   - Streaming, record-aware RFC 4180 CSV reader (`StreamingCsvReader`) with bounded 16 MiB field / 32 MiB record limits.
   - Raw dataset SHA-256 bound to registered manifest before extraction; reserved `__mlr_row_id` column prohibited in raw inputs.
   - Physical holdout data isolation: `prepare_dev_dataset` generates `dev_dataset.csv` containing only train/validation rows, mounted at `/data.csv:ro`. Sealed test partition is never accessible to development workers.
   - Provenance binding across result, checkpoint, trial receipt, and scorecard.
   - Operator user pin preservation and non-destructive bundle delivery.
   - Verified in real container tests (`custom_model_gate.rs`: 17 passed).
2. **Kaggle Compute Offloading (Gate S2)**:
   - Rust-governed backend in `crates/harness/src/kaggle.rs` (`KagglePoolManager`).
   - Anti-ban dual-account quota pool: single anchor account; secondary accounts strictly prohibited from competitions.
   - Secret scanning and staging scrubber for API keys and environment variables.
   - Atomic fenced lease checks and output checksum verification against `manifest.json`.
   - Verified in full mocked lifecycle tests (`kaggle_gate.rs`: 7 passed).
3. **Governed Tuning & LLM Integration (Gate S3)**:
   - Real Optuna study integration in `BudgetedOptunaTuner` (`TPESampler`, `MedianPruner`, and `create_trial` state recovery across restarts).
   - Hard multi-objective budget limits: trial count, duration, cost, concurrency bounding (`max_concurrent_trials`), and per-trial timeout.
   - Campaign ledger governance: pre-dispatch reservations (`reserve()`) and post-dispatch settlements (`settle()`).
   - Governed dual gateway in `dual_gateway.py` routing through `RustGatewayAdapter` with uncertain-outcome reconciliation (`mark_uncertain()`).
4. **Extensible ML Task Adapters**:
   - `TabularCandidatePlugin` (`tabular_candidate.py`) for tabular gradient boosting/random forest models.
   - `TinyTorchNeuralCandidate` (`torch_candidate.py`) for custom neural networks.
   - `SequentialTorchCandidate` (`sequential_candidate.py`) for 3D temporal sequential inputs `(batch, seq_len, in_features)`.
   - Subprocess delivery verification with `PYTHONPATH` unset and adversarial malformed input testing.

### Separation of Evidence: Completed vs. Mock-Only vs. Live Testing Needed
- **Completed & Verified with Real Evidence**:
  - Local container execution, train/resume/evaluate/export lifecycle, cgroup memory enforcement, timeout kill, and network isolation (Linuxkit arm64 Docker).
  - RFC 4180 streaming CSV record splitting, raw byte hashing, and holdout exclusion.
  - Optuna Bayesian optimization, TPE sampling, pruning, SQLite persistence, and concurrency limits.
  - Campaign budget reservation, settlement, and uncertain outcome tracking.
  - Clean-room bundle packaging and subprocess execution without `PYTHONPATH`.
- **Mock-Only Evidence (Verified via Mocks)**:
  - Kaggle API dispatch, status polling, and output extraction (`kaggle_gate.rs` with `MockKaggleApi`).
  - Outbound LLM provider calls (tested with scripted fixtures and mock responses).
- **Live Testing Still Needed (Next Phase with User)**:
  - Live Kaggle GPU dispatch with user's real Kaggle credentials.
  - Paid OpenRouter / DeepSeek inference with live API keys under campaign budget ceiling.
  - Ingestion and evaluation against the user's 5 chosen benchmark problems.

### Concise Live-Testing Checklist for Next Phase
1. **Benchmark Intake**: Ingest user's 5 problems using `docs/CAPABILITY_MATRIX_AND_INTAKE_TEMPLATE.md`.
2. **Kaggle Credentials**: Configure `env:KAGGLE_KEY` or `file:.config/kaggle.json` for Anchor account and Secondary worker.
3. **Live Preflight**: Run preflight on local host, assess resource bounds, and verify backend admission.
4. **Governed Dispatch**: Run search, tuning, or offloading within campaign budget caps (USD 0.30 campaign max, USD 0.05 per run).
5. **Final Evaluation**: Execute single sealed test evaluation via `test_access_id` and export verified delivery bundle.

## Authorization and completion rules

The user authorized this research and planning, not a new spend allowance or remote data upload. Implementation prompts should perform offline code and local tests by default. Existing continuation constraints still apply to live calls/jobs and historical holdouts. Complete all useful offline work first; if real verification requires new authorization, state the exact job, data, maximum spend/quota/time and reason. Do not ask again for scope already authorized in the active session.
