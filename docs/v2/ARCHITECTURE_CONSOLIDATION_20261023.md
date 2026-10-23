# Architecture consolidation and human control — 2026-10-23

This record updates [FINAL_REFACTOR_HANDOFF.md](FINAL_REFACTOR_HANDOFF.md). Historical scientific evidence, frozen manifests, K1 contamination, provider receipts and the shared campaign ledger remain authoritative historical records. No paid requests or final scoring are authorized or performed by this session.

## Active architecture

`AutonomousAgentLoop` is the shared feedback controller. It chooses actions from the task context, measured history and human guidance. Classification, regression and panel adapters in `task_adapters.py` own topology, frozen development loading/mapping, deterministic calendar derivation, panel coverage validation and candidate defaults; they call the existing fold-local scientific kernel. Adapters contain no provider, budget or worker-control loops.

`ExecutionService.evaluate(session, spec, operation)` handles both trusted evaluation and accepted-candidate refit. It owns worker admission, request/result identities, durable result publication, deadlines, environment restrictions, sampling and process-group cleanup. `run_script(session, params)` handles registered scripts through the existing Docker backend with no host fallback. Generated script results remain advisory. Host training is explicitly development execution; Linux cgroups, concurrent production containment and unavailable measurements must not be inferred from a successful host subprocess.

`GovernanceService.reservation/dispatch` owns controller admission, stable request dispatch, response checkpointing and conservative settlement through `CampaignBudgetAuthority` and `RustGatewayAdapter`. Rust still enforces exact content/route approval, reservation scope/store binding and the durable response fence under the shared campaign lock. These are backends of the shared governance boundary, not competing dollar authorities. The JSON ledger format and values have not migrated. Paid CLI and benchmark paths open the preserved ledger; missing authority fails rather than initializing replacement funds. Evidence campaign names do not select new paid allowances. Explicit offline CLI fixtures use a local fixture budget.

`scripts/run_robustness_campaign.py --mode live_dev --task K1|K2|K3|K4|K5` retains its public command line and delegates to `benchmark_runner.run_benchmark`. Old Python `run_live_dev*` methods remain compatibility delegates. There is one lifecycle: validate scientific contract → validate current pricing → stage hash-bound development inputs and frozen folds → governed controller → trusted measured evaluation/audit → accepted-model refit → identity-bound bundle and evidence. K1/K2 missing independent requirements and K4 missing provenance block before gateway initialization or reservations. All task metrics, directions and thresholds come from the contract. No replacement folds, inferred gaps or new acceptance requirements are generated.

Legacy `run_kaggle_titanic_live.py` and `run_live_end_to_end_test.py` direct provider helpers now fail before reading credentials. Historical local scientific compatibility kernels and deterministic test helpers remain available, but they are not alternate paid agent entry points. The generic development CLI imports service contracts directly rather than depending on the campaign script.

The Rust embedded-runtime asset list includes all new service/interface modules plus staging and panel dependencies. An installed-runtime regression materializes them outside the checkout and exercises controller creation, identity binding and queued human guidance/pause without provider calls.

## Identity and recovery

Controller bindings now include worker/Rust source fingerprints in a checkout (available Python assets with an explicit scope after installation), the transport binary digest, execution limits, pricing, staging lineage, data, folds, topology, target/metric and provider route. Benchmark topology also binds the complete task contract. Worker requests carry this binding and workers recheck the source before execution. Results are atomically published with request and payload digests and a sealed evaluation identity; stale/foreign/corrupt cached results fail. Accepted audits bind the specific evaluation receipt. Restart and refit check candidate specification, score, audit and measured receipt together. Old checkpoints lacking these bindings must be preserved and explicitly reconciled; there is no automatic migration, redispatch or re-certification of historical observations.

`delivery_evidence.json` is included in the bundle file manifest. It carries source/config/data/split lineage, candidate/evaluation/audit identities, exact serialized model digest, session and trace digest, stable request IDs/provider receipt digests and applied human-command IDs. Trace events have content identities. Runtime gateway databases remain local and must survive interruptions; packaging digest references is not a backup of those databases. A shared OS user can alter local files: hashes and actor labels do not provide adversarial OS authentication. Independent upload-operator deployment identity separation remains required.

Exported inference examples now exclude the target column. Cross-task integration caught the prior examples/API mismatch through a fresh interpreter importing the packaged inference code.

## Common human interface

[HUMAN_CONTROL_INTERFACE.md](HUMAN_CONTROL_INTERFACE.md) specifies the stable local `AgentInterface` API and CLI. Every task uses the same status, steer, pause, resume and stop interface. Commands bind to the session/config identity, have idempotent caller-selected IDs and durable content receipts, and are consumed at turn boundaries. The independent inbox lock allows input while a controller owns its workspace. Applied IDs and guidance are checkpointed before another provider dispatch. This interface cannot change frozen inputs, clear ambiguous requests, raise budgets, approve uploads or approve models.

## Validation and limitations

Validation: full Python suite **325 passed / 2 skipped / 10 warnings (220.03s)** before final source-binding guards; final affected suite **105 passed (50.17s)** includes the final guards and 18 new architecture regressions. Final Rust checks **63 passed / 2 ignored / 1 filtered**; the filtered Docker-dependent scheduler fixture was first attempted and failed admission, with its log retained. Rust build, Python compilation and whitespace checks passed. Exact commands and scope limits are in the evidence summary.

Exact commands, counts and retained failed attempts are in `docs/evidence/v2/autonomy/20261023-architecture-consolidation/summary.json`. Offline providers and stubbed critic approval in synthetic delivery tests prove wiring and identity checks, not scientific quality. Local worker training/refit and fresh-interpreter bundle inference are real. All synthetic files, models and runtime databases remain outside Git.

The current Docker daemon is unreachable and `open -a Docker` reports no application. The Rust scheduler crash-recovery fixture therefore cannot admit its scientific container job. This environmental failure is retained without changing its assertion. The prerequisite session's two successful Linux cgroup fixtures remain historical evidence; they were not rerun here.

Unresolved external blockers remain:

- K1 contaminated/regenerated final history and Ticket overlap; staging cannot restore cleanliness.
- K1/K2 independently justified absolute quality requirements are absent; thresholds remain null.
- K4 authoritative fold index-space, gap/units, training bounds and complete subset mapping are absent.
- K5 retains all 1,782 entity tuples, complete observation calendar and 2,893,968 / 2,920,698 / 2,947,428 frozen training rows, three exact 15-date horizons, zero declared embargo and deterministic calendar predictors only. Full-scale resource feasibility and repaired-revision quality remain unverified.
- Invoices/provider-charge receipts are absent. Recorded/calculated amounts and uncertain liabilities cannot be promoted to verified charges or released on this basis.
- Future uploads lack independent content-bound authorization; current model/route availability/pricing was not verified offline. Preserve `deepseek/deepseek-v4.1-flash-1017` via `inference-net/fp8`, no fallback.
- Full production scheduling, per-job CPU/resource measurements, concurrent isolation, required fault/soak coverage and clean-room delivery remain unverified. The cached scientific image lacks `jsonschema`; matching hash-verified Linux wheels/lock for declared `jsonschema==4.26.0` and `psutil==6.1.1`, and a rebuilt image, remain unavailable offline. The bounded-tmpfs scientific publication test cannot be certified from host tests.

Campaign ceiling stays USD 0.30, committed USD 0.050724020, recorded remaining USD 0.249275980. This session adds USD 0, makes zero paid provider calls and performs zero final scoring. No production certification or new benchmark quality pass is claimed.
