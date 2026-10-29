# MLRescurix: architecture and current evidence

Reviewed 2026-10-29 against the working tree based on commit `1d4fe51`. This is a scoped source and test review, not a complete security audit. The reviewed changes extend beyond that baseline commit. Current evidence supersedes older completion claims in the session records. See `REMAINING_ENGINEERING_PLAN.md` for the plan and `IMPLEMENTATION_REVIEW_20261028.md` for historical review details.

## Objective and boundaries

MLRescurix aims to help develop and evaluate general ML models and tackle Kaggle problems, including custom neural and sequential models. Rust is the preferred control-plane language; Python is appropriate for scientific and training libraries. The user will select five distinct benchmark problems later. A cart transformer was an example, not a mandatory target. Finite resources prevent an unconditional guarantee that any problem will run or succeed: the desired behavior is bounded execution, resumable progress where supported, and a clear `needs_resources` or unsupported result.

This repository has a Rust harness and Python scientific worker, with different degrees of integration across paths. The presence of a module, contract, mock or synthetic result does not establish end-to-end operation for arbitrary workloads.

## Current components

| Component | Code | Status and limits |
| --- | --- | --- |
| Supervisor, state and agent tooling | `crates/harness/src/execution.rs`, `storage.rs`, `agents/` | Existing Rust control, SQLite records, scheduling, container execution and agent tools. Specific paths need separate verification; diagrams must not be read as proof that every route shares the same authority. |
| Scientific worker | `python/src/mlrescurix_worker/scientific/` | Existing tabular and panel modeling, experiments, diagnostics and finalization paths. Their prior benchmark records are historical, not the five future user-selected tasks. |
| Custom candidates | `python/src/mlrescurix_worker/custom/` | Contracts and tiny PyTorch, tabular and GRU sequential prototypes exist. Synthetic/plugin tests do not prove that every plugin is wired through the governed supervisor, supports arbitrary architectures, or is production ready. |
| Custom job admission | `crates/harness/src/execution.rs` | **Disabled** after spec validation pending isolation and authoritative evaluation. The recent synthetic container lifecycle had passed before the protective gate was restored, but passing it does not establish the trust boundary. |
| CSV splitting | `execution.rs`, `StreamingCsvReader` and `prepare_dev_dataset` | New streaming record parser, dataset-manifest hash comparison, and split extraction exist; quoted multiline CSV and adversarial cases have tests. This alone does not establish immutable registered inputs or sealed holdout isolation throughout execution. |
| Kaggle | `crates/harness/src/kaggle.rs`, `python/src/mlrescurix_worker/scientific/kaggle_pool.py` and related backend | Rust pool has lifecycle, account and lease prototypes; `KaggleApi` has a mock implementation in tests, with no production Rust implementation found in the repository. Quota defaults are local assumptions, not evidence of current Kaggle allocation. Live GPU execution, secure checkpoint transfer and scheduler integration are unverified. Two credentials do not by themselves establish that using multiple accounts is allowed under applicable terms. |
| Tuning | `python/src/mlrescurix_worker/custom/tuning.py` | Optional Optuna ask/tell, TPE and recovery code plus a random fallback exist. Sequential `optimize` checks trial timeout only after the objective returns; optional authority hooks and fixed estimated reservation are not proof of hard time, money, or concurrent dispatch limits. |
| Providers and search | `scientific/dual_gateway.py`, `rust_gateway_adapter.py`, `deepseek_search.py`, Rust gateway | OpenRouter/DeepSeek routing code exists. Optional authority and adapter hooks allow direct calls, so universal governance is not established. Verify endpoint support, configured model IDs and pricing before actual use. Do not assume DeepSeek API credentials supply a web-search endpoint. |
| Delivery | `custom/delivery.py`, `crates/harness/src/bundle.rs` | Packaging and verification code exist; some artifacts require Python packages such as Torch, joblib or NumPy. A clean `PYTHONPATH` test is not a zero-dependency or independently provisioned clean-machine test. Identity and authoritative evaluation must be verified per path. |

## Trust and resource findings

1. `container_scientific_worker` mounts the entire workflow directory writable into the container. A development-only CSV mount does not prove the worker cannot read sealed data or mutate trusted files elsewhere in that directory. The generated development file also uses a shared workflow path; concurrent jobs and failed partial writes need isolation. The raw file is hashed and reopened for parsing, so immutable snapshot and time-of-check/time-of-use handling still matter.
2. Evaluation and promotion must bind authoritative dataset, protocol, job, candidate, source, environment and weights to an independently checked result. A self-reported worker score, matching worker-generated hashes or a passing fixture alone cannot prove this. Re-review the new `Supervisor::finish` checks and their negative cases before reopening admission.
3. The locally present pinned Docker image was used in historical synthetic tests. Its ignored wheelhouse is not included in Git; reproducible provisioning on a clean machine, target platform compatibility and GPU support are not established.
4. Tuning and provider routes should be treated as experimental until every dispatch is governed, reservations cover each potentially billable attempt, uncertain outcomes are reconciled, and subprocess time/resource limits actually terminate work. Provider billing is not established by local estimates.
5. Claims of universal sealed-data safety, exact replay, 80–98% context compression, all-job governance, zero-dependency delivery, complete support for all modalities, or guaranteed success lack evidence for the whole system. Apply those claims only to a defined path with a reproducible measurement.

## Budgets and past tests

The historical robustness campaign used a USD 0.30 ceiling and a specific ledger. This is a campaign constraint, not evidence that every future project must use that limit or that every code path enforces it. Do not modify or reuse historical ledgers, frozen holdouts, or accounting records without the appropriate project authorization. No new paid provider calls, Kaggle jobs, private-data uploads, or holdout use were part of this review.

On 2026-10-29, before restoring the custom admission gate, `cargo test --offline --test kaggle_gate --test custom_model_gate` reported 7 Kaggle mock tests and 17 custom-gate tests passing; `python3 -m pytest python/tests/test_session3_tuning_and_delivery.py python/tests/test_ml_task_adapters.py -q` reported 15 passing. These results apply to that working-tree state and those tests; they do not establish live service readiness. Three custom container admission tests are intentionally ignored while the gate is closed. Consult the latest review record for subsequent checks.

## Handoff instructions for an external reviewer

Read this file, `REMAINING_ENGINEERING_PLAN.md`, the actual code, and the latest review before proposing changes. Distinguish existing, tested, mock-only, and proposed capabilities; cite concrete paths and tests. Do not assume that the historical session completion reports or pre-gate synthetic results are current gate approvals. Avoid rewriting the system or adding another scheduler unless an identified gap requires it.
