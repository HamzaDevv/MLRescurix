# Autonomous development agent implementation — 2026-10-22

Latest prerequisite repairs and unresolved blockers: [PREREQUISITE_REPAIRS_20261023.md](PREREQUISITE_REPAIRS_20261023.md). Next architecture session: [FINAL_REFACTOR_HANDOFF.md](FINAL_REFACTOR_HANDOFF.md). Historical findings below are preserved.

Historical implementation handoff. Current review and continuation status are in
[POST_LIVE_REVIEW_20261023.md](POST_LIVE_REVIEW_20261023.md).

This records the earlier implementation handoff. It supersedes earlier claims that the
five scripted benchmark runs established complete autonomy. Historical evidence
and the $0.30 shared campaign ledger have been preserved. No paid requests or final
holdout scoring were performed in this implementation session.

## Delivered behavior

The controller selects one structured action per AI turn, executes it, and returns
measured results, parameters, errors, resource observations, and critic flaws to
the next decision. An AI provider chooses the next action; there is no fixed
three-call diagnostic/architect/critic sequence. Tests use explicit offline
provider fixtures; they do not count as live AI evidence.

Actions cover diagnostics, paired feature experiments, family evaluation,
weighted ensemble audit, dynamic Python writing/execution, finalization, and
handoff. A supported feature experiment compares the same pinned folds before
and after the transform, retains an improved transform in the versioned pipeline,
and uses it during later evaluation, refitting, and inference. Refuted transforms
are not silently applied. Model requests and positive blend weights are honored;
unavailable optional families fail rather than silently substituting a model.

CandidatePipeline fits imputation, categorical/frequency encoding, transforms,
and estimators inside each training fold. The critic receives actual paired
train/validation metrics and actual fold scores for the ensemble. Classification
critic metrics are balanced accuracy; regression critic metrics are R2; primary
metrics retain their own exact definitions (including RMSE versus log-price RMSE
versus RMSLE). Missingness error slices are measured from validation predictions.
Feature-gain importance remains explicitly unavailable, never a uniform invented
vector. The legacy critic's unmeasured feature-dominance check is not certified.

Only an approved candidate with a finite measured score can be packaged. A
canonical specification hash binds evaluation, audit, refit, serialized model,
and bundle metadata. The bundle carries the accepted development-CV receipt;
a new arbitrary split of development data is not presented as a sealed final
score. Candidate bundles include a minimal runtime and preserve raw categorical
inputs in standalone classification and regression inference. Legacy bundle
paths are retained and separately checked.

Generated Python is written to a dedicated script registry, syntax checked,
hashed, and run through Docker with no host fallback. Only that script and a
separate development-data directory are mounted read-only. The container has no
network, a read-only root, a non-root user, dropped capabilities, and CPU/memory/
PID/time/output/artifact limits. A supervisor retains bounded tmpfs artifacts
until they are copied and hashed. Results include stdout, stderr, status,
artifact manifests, and available kernel resource receipts. Scripts are advisory:
their self-reported metrics cannot approve a candidate or replace pinned folds.
Free-form stdout/stderr stay in local receipts unless an outbound view is
explicitly approved; subsequent AI turns receive execution status/error classes.

Trusted model training and accepted-candidate refitting run in separate worker
process groups with a CPU-time limit, wall timeout, an observed RSS watchdog,
and Linux address-space limits. Continuous sampling separates registered workers
from the controller, keeps persistent CPU counters, and reports unavailable data
truthfully when OS permissions deny access. These development workers are not
claimed to have production cgroup containment. Docker script resource receipts
are separately labeled and do not claim host process-tree measurements.

A durable intent precedes provider dispatch. Request IDs are stable per turn.
Provider receipts are saved before settlement; turns checkpoint after tool work.
Interrupted/uncertain dispatches stop for reconciliation instead of automatically
rebilling or repeating side effects. Corrupt, incompatible, or changed-data/fold
checkpoints fail closed. Exclusive workspace locks prevent concurrent controllers
from replaying a stale state. This is conservative recovery, not an exactly-once
provider API guarantee. Automatic recovery reconciliation is not implemented.

Each session writes `autonomous_loop_state.json`, `autonomous_trace.jsonl`,
`handoff.json`, and `HANDOFF.md`. State binds data bytes, fold indices, topology,
metric, model route, session, and resource policy. Runs have at most 20 turns,
12 candidate evaluations, an 80,000 conservative requested-token bound, and the
configured run dollar limit. Live calls require verified input/output price
ceilings; uncertain charges remain committed under the existing shared campaign
authority. The campaign authorization remains $0.30 total, not per session.

## Running and continuing

Use the repository's Python environment and development-only inputs:

```sh
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_cli --help
```

The CLI requires `--development-csv`, `--folds`, `--workspace`, `--project`,
`--session`, and `--target`. Fold files contain development-local
`train_indices`/`val_indices`; pass authoritative topology for temporal/group
constraints. The default performs diagnostics without provider dispatch.
`--fixture-actions` is explicitly offline. `--development-live` explicitly selects
the development transport and requires current verified pricing in the benchmark
manifest. It does not satisfy production certification. Docker must be running
with a locally available scientific image before RUN_PYTHON can succeed.

Resume a turn-limited session with the same workspace/session/data/folds and a
higher max-turn count (up to 20). HANDOFF and explicit stop remain terminal.
For a pending dispatch, inspect its durable provider receipt and the budget
ledger, reconcile whether settlement/tool effects occurred, and preserve the
original evidence. Do not delete checkpoints or clear pending_turn merely to
retry. There is deliberately no automatic unsafe reset/migration of v2 states.

## Remaining release blockers

1. **Production Rust adapter:** the Python development loop is not wired through
   the Rust gateway/orchestrator/scheduler. The campaign runner now blocks live
   certification rather than bypassing this requirement. Revalidating pricing
   alone does not unblock it. Implement and test a shared governed adapter with
   outbound-view approval and one authoritative spend path.
2. **Forecasting:** generic row regression cannot certify panel forecasting,
   horizon semantics, or known-at-forecast covariates. K5/forecasting topology
   explicitly hands off until a real panel evaluator is connected. Vision,
   audio, general unsupervised tasks, and long-document modeling are unsupported
   by the built-in candidate backend; isolated analysis scripts and explicit
   handoff are available. There is no universal-problem-solving claim.
3. **Live environment:** the Docker daemon is unavailable here; the live Docker
   integration test skips. `data/official` is absent in this checkout. Restore
   the pinned official files/manifests and validate actual fold mappings and
   joins before any benchmark campaign. Earlier runner paths read full source
   CSVs before dev selection; those blocked paths still need a development-only
   staging boundary before production is enabled.
4. **Native dependency stability:** two full-suite attempts terminated with
   exit 139 inside LightGBM `basic.py::__init_from_np2d`, including an attempt
   with native thread limits set at startup. No full-suite pass is claimed.
   Trusted worker crashes are contained and returned as agent feedback, but the
   legacy direct native-training paths still need environment/dependency repair.
5. **Certification evidence:** feature-dominance auditing, production cgroup
   resource/fault/soak evidence, clean final partitions, the five live autonomous
   development runs, and final frozen assessments remain outstanding. Existing
   historical benchmark scores are not retroactively repaired by source changes.

## Verification record

See the final validation section below for the exact completed test commands and
results. Offline provider/tool fixtures establish control-flow contracts, not live
provider quality. Fresh subprocess bundle tests exercise actual inference, and
worker limit/crash tests terminate real subprocesses.

## Final validation — this implementation session

- Focused autonomy, script tool, monitoring, runner, new standalone candidate
  bundles, and affected legacy bundle compatibility checks: **53 passed,
  1 skipped** in 29.38 seconds. The skipped check is live Docker integration.
  Log: `docs/evidence/v2/autonomy/20261022-development-agent/python-tests.log`.
- Rust bundle integrity/constraint checks:
  `cargo test -p mlrescurix-harness --lib bundle::tests --offline`:
  **2 passed**, 27 filtered out. Log in the same evidence directory.
- After the final workspace-path/request-ID hardening, the autonomy regression
  suite was rerun separately; see the final regression log in that directory.
- Python compilation and `git diff --check` passed.
- Full Python suite attempted twice; both exited **139**, with the native
  LightGBM stack described above. The second attempt set OMP/BLAS/MKL/VECLIB
  thread limits before interpreter startup. Saved crash output is in the same
  evidence directory. These attempts do not establish a full-suite pass.
- Docker CLI exists but its daemon is unreachable. Attempts to start installed
  Docker Desktop via Launch Services failed with `kLSNoExecutableErr`; no Docker
  installation or environment replacement was attempted.
- Reconciled existing ledger: billed **$0.014604020**, reserved **$0.00**,
  uncertain **$0.00**, remaining **$0.285395980** of **$0.30**. Additional campaign
  API spend this session: **$0**. No final holdout scoring, commits, or pushes.

Do not mark S1/S3/S4/S5 complete on these results. The accepted-candidate tests
with a stubbed critic verify identity/control flow; they do not establish model
quality or real-provider autonomous performance on the five benchmark tasks.

Final hardening also pins accepted-bundle inference dependencies to the versions
actually used for fitting, including requested native model libraries, rather
than copying unrelated historical package versions. The final autonomy regression
run passed **19 tests**. Final affected bundle checks (new classification and
regression runtime plus three legacy compatibility cases) are recorded in
`bundle-final-regressions.log`.
