# M4: Agent-Led Tabular ML Coding Harness

This document describes MLRescurix's M4 milestone: an agent-led tabular machine learning coding harness with provider-backed orchestration, reusable specialist roles, agent-authored project tools, sandboxed execution, evaluator separation, achievement context management, and permanent finalization closure.

---

## 1. Milestone Assertion & Product Intent

A provider-backed lead agent can formulate evidence-backed hypotheses, delegate to reusable specialist roles, author and test novel Python utilities, execute bounded ML experiments, receive independent official evaluation from the trusted control plane, interpret the results, and continue or stop—all while strictly preserving user intent, budgets, isolation, and recoverability.

### Separation of Concerns

| Layer | Responsibility | Authority & Constraints |
|---|---|---|
| **Agent Layer** (Lead & Specialists) | Research direction, hypothesis framing, multi-file code authoring, diagnostic creation, delegation, stopping decisions. | Operates in sandboxed workspaces on isolated branches. No access to unmasked credentials or private test labels. Generated outputs remain untrusted until officially evaluated. |
| **Control Plane** (Rust Harness & Trusted Evaluator) | Budgets, leases, contracts, process isolation, scheduling, data registration, official metrics, artifact verification, finalization closure. | Authoritative and non-bypassable. Evaluator never executes untrusted code, never imports candidate modules, and never deserializes untrusted model artifacts. |

---

## 2. Reusable Specialist Roles

Rather than mandatory sequential phases, M4 provides 8 reusable specialist roles available for delegation:

1. **`lead`**: Research director orchestrating problem framing, hypothesis progression, specialist delegation, and stopping decisions.
2. **`auditor`**: Data specialist inspecting column schemas, missingness, class imbalance, and potential target leakage.
3. **`feature_engineer`**: Designs, implements, and validates preprocessing pipelines (encodings, scalers, imputers) fitted strictly on training splits.
4. **`model_specialist`**: Implements tabular estimators (linear models, tree ensembles, and custom estimators adhering to `CustomEstimator`).
5. **`search_engineer`**: Configures and executes M3 multi-fidelity hyperparameter searches.
6. **`diagnostic_engineer`**: Authors reusable Python diagnostic tools (slice metrics, residual distributions, calibration checks).
7. **`verifier`**: Subjects candidate estimators to preflight tests (shapes, NaN handling, single-row inference, joblib round-trip).
8. **`critic`**: Independent pre-freeze reviewer stress-testing evidence and validating robustness before any finalization request.

---

## 3. Two Distinct Tool Categories

M4 supports two cleanly separated tool categories:

### Category A: Trusted Harness Tools
Implemented by the trusted control plane and exposed via standard tool call interfaces:
- `file_read`: Read a file from the branch workspace.
- `file_write`: Write a file in the branch workspace.
- `file_patch`: Apply exact text replacements in a workspace file.
- `file_list`: List workspace files with byte counts and SHA-256 hashes.
- `sandbox_python`: Execute Python scripts or tests in the hardened sandbox.
- `register_project_tool`: Register an agent-authored project tool descriptor.
- `invoke_project_tool`: Execute a registered project tool and receive a typed `ToolExecutionReceipt`.
- `inspect_dataset`: Inspect registered dataset schemas and manifest metadata.
- `delegate`: Delegate a subtask with an assigned branch and hypothesis.
- `query_budget`: Query remaining project budget dimensions (calls, tokens, wall_ms).
- `request_finalization`: Submit a candidate for final freeze and test evaluation.

### Category B: Agent-Authored Project Tools
Python utilities authored by the agent inside its branch (e.g. `diagnostics/missingness_checker.py`, `diagnostics/slice_evaluator.py`):
- **Descriptor Specification**:
  - `name` & `version`: Unique semantic identification.
  - `input_schema` & `output_schema`: JSON Schemas for arguments and returns.
  - `entrypoint`: Script path within the branch workspace.
  - `source_hash`: SHA-256 merkle hash of the branch source tree.
  - `environment_id`: Managed execution profile (e.g. `scientific`).
  - `provenance`: Author role, task ID, and creation timestamp.
- **Execution Contract**: Invoked inside the sandbox; writes JSON receipt; outputs validated against `output_schema`.
- **Privilege Boundary**: Authoring a tool creates zero privileges. Tools cannot modify host files, register host handlers, access private labels, or exceed sandbox resource limits.
- **Cross-Branch Reuse**: Implemented via immutable snapshots (`export_immutable_snapshot` and `import_snapshot`) rather than exposing live workspace directories across branches.

---

## 4. Provider Gateway & Outbound Redaction Filter

- **OpenAI / OpenRouter Adapter (`OpenAICompatibleAdapter`)**:
  - Configurable endpoint URL, model, and API key.
  - Exponential backoff retry logic on transient HTTP failures.
  - Outbound calls dispatched with bounded timeouts via `curl`.
- **Scripted Provider Fixture (`ScriptedProviderFixture`)**:
  - Queues deterministic responses for offline regression tests and CI without external network access.
- **Outbound Data Redaction Filter (`OutboundFilter`)**:
  - Enforces provider upload policy across all outbound messages, tool receipts, and previews.
  - Detects multi-line raw tabular data rows and redacts them (`[REDACTED_ROW_DATA: ... raw tabular rows omitted]`) while preserving column schemas and aggregate statistics.
  - Enforces character bounds (16 KiB ceiling per message).
- **Provider Accounting**:
  - Records request ID, model, prompt tokens, completion tokens, latency, and estimated cost in SQLite `provider_accounting`.

---

## 5. Achievement Context Management & Compaction

- **Achievement Boundaries**:
  - Significant milestones (`data_audited`, `baseline_evaluated`, `tool_authored`, `finalization_requested`) are recorded into SQLite `achievements`.
  - Achievement boundaries trigger an authoritative state snapshot in `agent_checkpoints`.
- **Structured Context Compaction**:
  - Triggers when context tokens approach `max_context_tokens - reserve_tokens` (reserving 4 KiB for tool outputs and next response).
  - Preserves: system prompt, active hypothesis, all committed achievements, scientific findings, and recent conversation turns.
  - Compresses verbose tool logs into a structured markdown index.
- **Turn-Boundary Steering**:
  - User guidance injected strictly at turn boundaries.
  - Automatically supersedes prior uncommitted handoffs and increments the handoff epoch version.
- **Historical Retrieval**:
  - `retrieve_historical_findings(keyword)` allows agents to inspect and learn from earlier failed hypotheses.

---

## 6. Hardened Sandbox & Workspace Isolation

- **Host Sandbox Capability Detection (`SandboxDetector`)**:
  - Probes Docker engine (`docker version`).
  - If Docker is offline, returns `SandboxStatus::Unavailable` and **strictly refuses** untrusted code execution.
  - Host Python is **never** substituted for untrusted agent code.
- **Container Isolation Controls**:
  - `--network none`: Complete external network isolation.
  - `--read-only`: Read-only container root filesystem.
  - `--tmpfs /tmp:rw,noexec,nosuid,size=64m`: Bounded in-memory scratch space.
  - `--memory 512m --cpus 1.0 --pids-limit 64`: Strict resource limits.
  - Host directory exclusion: Host sentinel files outside the state root are never mounted or accessible.
  - Path traversal rejection: Path components with `..`, root `/`, or symlinks are rejected deterministically.

---

## 7. Custom Candidate Contracts & Evaluator Separation

- **Python Contracts**:
  - `CustomEstimator`: ABC defining `fit(X, y)`, `predict(X)`, and optional `predict_proba(X)`.
  - `CustomTransformer`: ABC defining `fit(X, y)` and `transform(X)`.
- **Preflight Verification (`run_preflight_checks`)**:
  - Validates shape parity, 1.0 probability sums, single-row inference, joblib round-trip serialization, and NaN stability before experiment submission.
- **Strict Label Isolation**:
  - Candidate models only receive validation feature matrices (`X_val`); validation labels are never exposed to the candidate container.
  - Candidate generates row-keyed predictions in `predictions.json`.
  - Trusted evaluator reads predictions, checks row count and alignment against private labels, and computes official metrics.
  - Untrusted candidate code is **never** imported or unpickled by the evaluator process (`allow_pickle=False`).

---

## 8. Permanent Finalization Closure

- Prior to finalization, the independent `critic` role audits development evidence, multi-seed stability, and slice metrics.
- Submitting `request_finalization` atomically transitions the project to `finalizing` in SQLite and locks research admission.
- Any subsequent attempts to submit search trials, custom experiments, or exploratory jobs are deterministically rejected (`"research admission is closed"`).
- A failed or completed finalization permanently fences the project from reopening research, preventing test set data snooping.

---

## 9. Verification & Test Evidence

M4 is verified by a multi-tier test suite:
- **Rust Integration Suite (`tests/agent_m4.rs`)**:
  - 8 specialist roles and capabilities.
  - Sandbox detection and honest refusal when container runtime is offline.
  - Path traversal and symlink rejection.
  - Agent authors, tests, registers, and reuses diagnostic tool, adapting research decision based on result.
  - Cross-branch immutable snapshot export and import isolation.
  - Achievement checkpointing, structured compaction, steering fencing, and finding retrieval.
  - Outbound data redaction filter.
  - Critic finalization and permanent research closure.
  - 10 projects and 3 branches concurrency and fairness.
- **Rust Foundation & M3 Search Tests (`tests/foundation.rs`, `tests/search_m3.rs`)**:
  - 24 M0/M1 tests + 10 M3 search tests passing.
- **Python Scientific & Worker Suite (`python/tests/`)**:
  - 20 unit tests covering custom contracts, preflight verification, isolated runner, official evaluation with slice diagnostics, and dashboard snapshot smoke.
