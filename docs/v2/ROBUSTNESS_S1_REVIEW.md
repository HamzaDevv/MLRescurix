# S1 implementation review: partial, requires repair

Reviewed 2026-10-21 against commit `e1c8674` and the supplied S1 completion report.
The checkout was clean before this audit. No runtime code was changed in this review.

The implementation is useful progress, but **S1 has not met its exit gate**. Existing
tests exercise selected happy paths and do not prove the trust boundaries, durable
execution or production split propagation claimed in the report. Repair S1 before
advancing to paid K1/K2 production solves. Docker and official data remain external
prerequisites; the defects below can be repaired independently.

## Findings and required acceptance checks

### F1 — P0: exploratory and custom execution are not a governed sandbox (RB-001 reopened)

[`eda_agent.py`](../../python/src/mlrescurix_worker/scientific/eda_agent.py) starts
`python -I` with the inherited process environment. The audit injected a harmless
dummy variable and the worker read it. A real provider environment credential would
also be inherited. The worker has host filesystem/network access, unbounded StringIO
capture and no memory/CPU/process limit. Timeout kills the direct child, not a
proven process tree. `-I` isolates Python startup configuration, not OS capabilities.
Actual secret reads, external egress and fork attacks were not attempted.

Separately, `CUSTOM_SANDBOX_CODE` reaches the in-process `exec` in
[`fe_ablation.py`](../../python/src/mlrescurix_worker/scientific/fe_ablation.py).
A harmless feature returning `os.getpid()` proved it executes in the controller.
Removing a target column does not contain arbitrary code or protect host files.

The 2048-byte claim also fails: `print('界' * 3000)` returned **5892 UTF-8 bytes**.
The truncation slices characters, after unlimited output is already collected.

Required: one governed execution boundary for both paths; fail closed when it is
unavailable, allowlisted environment, isolated mounts/network, bounded streaming
capture, process-tree cleanup and enforceable job limits. Test dummy credentials,
approved/rejected file paths and egress, descendant cleanup and multibyte output.
Do not run untrusted production candidate code on this host while containment is
unverified. Linux enforcement certification remains pending Docker/cgroup access.

### F2 — P1: OOF topology and coverage remain unsafe (RB-004 reopened)

[`pareto_ensemble.py`](../../python/src/mlrescurix_worker/scientific/pareto_ensemble.py)
accepts `split_assignments` without checking train/validation overlap, bounds,
coverage, topology or dataset-row identity. The audit supplied an overlapping fold
and it was accepted. Temporal fallback applies `TimeSeriesSplit` to input row order,
without sorting/checking the time column or applying a locked gap. With descending
timestamps, training rows were later than validation rows. Its unassessed initial
temporal prefix was filled with four zero predictions, and synthesis fits/scores
the meta model against all labels. Missing predictions must not become observations.
Group fallback can silently revert to random folds when groups are insufficient.

Required: evaluator-owned row-keyed split manifest; validate folds before executing
any candidate; no silent random fallback; explicit OOF coverage masks; meta fitting
and evaluation only on eligible rows, with the separate nested/crossfit assessment
required by the master plan. The new optional argument alone does not establish
propagation through production full-CV, stacking and calibration.

### F3 — P1: proxy split is only a partial repair (RB-003 reopened)

[`iteration_loop.py`](../../python/src/mlrescurix_worker/scientific/iteration_loop.py)
now handles simple chronological or group splitting, but still creates its own
25% split instead of consuming the locked assessment split. It slices rows rather
than whole timestamps, allows equal timestamps across the boundary, applies no
purge/gap and prioritizes temporal rules over simultaneous group constraints.
An insufficient group count can fall through to random splitting. These are source
findings; panel/gap/combined-topology reproducers remain required.

The two new proxy tests only assert `candidate_id`, even if the receipt reports a
failed candidate. Replace these with direct observation of train/assessment row IDs,
successful evaluation and assertions for every declared topology constraint.

### F4 — P1: strict action validation and durable idempotency are not implemented (RB-013)

[`hypothesis_executor.py`](../../python/src/mlrescurix_worker/scientific/hypothesis_executor.py)
does not load or apply its JSON Schema. An unknown top-level property, unknown
parameter, negative fold count and non-integer seed passed validation together.
The schema itself needs action-specific required parameters and supported-method
enums; a union of optional parameter properties is not a strict per-action contract.

`source_hash`, `topology_hash`, `split_id` and `budget_id` are accepted as supplied,
without checking the governed objects. `ledger_path` is stored but unused; no
ledger file was created and a new dispatcher had an empty cache. Process-local
memoization is not durable replay protection. No production dispatcher integration
was demonstrated by the new tests.

Required: schema and semantic validation at the authoritative boundary, durable
action ID/payload hash/settlement transactions, restart and conflict tests, and
budget/split/source binding. Do not create a second independent Python authority
in place of the existing Rust ledger and evaluator.

### F5 — P1: actual gain settlement contradicts the corrected metric policy (RB-014)

The dispatcher ignores `metric_name` and does not forward the action seed to ablation.
It reads `base_cv_score`, while ablation emits `baseline_cv_score`, so its denominator
silently defaults to 0.5. Ablation represents RMSE as **negative RMSE**, already
oriented for maximization. Applying another minimization sign reverses improvement.
Using the real receipt shape for RMSE 10→9 (`baseline_cv_score=-10`, delta=+1), the
audit got **refuted**, relative gain **-2.0**, instead of +0.10. The ablation was
mocked for this settlement probe; this is not a measured model improvement.

Required: one explicit raw-metric versus oriented-utility contract, metric-specific
scoring, correct baseline field, and end-to-end tests for minimization, maximization,
negative scores and near-zero values. Forward the locked seed/splits to ablation.

### F6 — P1: three actions confirm work they do not perform (RB-015)

`REORDER_MODEL_PRIORITY`, `UPDATE_SEARCH_SPACE` and `TRANSFORM_TARGET` construct
confirmed receipts and return. There is no scheduler update, search-space mutation,
fitted target transform or inverse-transform artifact. The target-action probe
returned confirmed without creating any ledger/configuration artifact. Lack of
input mutation alone is not an error: the missing configured candidate pipeline is.

Required: implement actual governed state transitions and candidate pipeline lineage,
or return an explicit unsupported/inconclusive result. Confirm target transformation
only with fitting, inverse-once prediction parity and canonical metric units.

### F7 — P1: runner can falsely succeed and overwrite resumed evidence (RB-016)

[`run_robustness_campaign.py`](../../scripts/run_robustness_campaign.py) accepts
`deterministic_dev`, `live_dev` and `certify` but executes none of them. All three
printed “Run Complete,” returned normally, and wrote no summary in isolated audit
directories. `all` only runs preflight and the two small pytest files. Unsupported
modes must be explicit skipped/blocked/non-success outcomes, never certification.

On resume, `run_preflight` replaces the manifest, artifact record and provider usage
with a fresh zero-spend/$5-remaining baseline. It does not check revision/config
identity or preserve prior cost reservations. Summary remains `in_progress` even
after CLI completion. Writes are not transactional and concurrent/resumed runs are
not locked. Resource time-series files are declared but not populated by these paths.
These resume/resource findings are source inspection, not a paid-run experiment.

Required: immutable identity checks, atomic receipts and budget reconciliation,
explicit terminal aggregate status, implemented-mode dispatch, concurrency protection,
resource traces and resume tests that prove existing evidence/spending survive.

### F8 — P2: reported provenance, test scope and frozen benchmark claims overreach

G2 passes on binary **existence**, not a build tied to source. G3 is a hardcoded
dictionary marked verified, without opening/migrating a database. G6 proves only
key presence, not a usable provider gateway. Official data credential presence would
not prove download authorization, pinned data, metric snapshots or frozen splits.
The ledger's binary hashes differ from the linked artifact summary; reconcile build
timing/identities rather than selecting a convenient hash. No final vault exists
merely because the ledger calls it “sealed.”

The reported 65 tests correspond to seven selected scientific/S1 files. This audit
re-ran those files: **65 passed, 9 warnings, 51.23s**. Repository-wide collection found
**196 tests**. Therefore `pytest python/tests/ → 65 passed` is not a demonstrated
full-suite result. The historical Rust 8-pass result was not re-run in this review.

Keep K1–K5 as specifications; data hashes, locked row assignments and final vault
creation remain pending. Retain synthetic fixtures as contract checks only.

## Evidence and continuation

- [Executable audit probes](../evidence/v2/robustness/20261021-s1-audit/audit.py)
- [Observed results](../evidence/v2/robustness/20261021-s1-audit/observations.json)
- [Repository test collection](../evidence/v2/robustness/20261021-s1-audit/collection.log)
- [Supplied S1 report](../evidence/v2/robustness/20261021-s1-audit/S1_REPORT_SOURCE.txt)

Audit command: `PYTHONPATH=python/src python3 docs/evidence/v2/robustness/20261021-s1-audit/audit.py`.
It exited 0, meaning the diagnostic probes completed; it deliberately demonstrates
defects and is **not** a passing robustness gate. Submodel process calls and the
metric ablation receipt are mocked as documented. Host probes use dummy values only.
Provider spend and holdout accesses in this review: **0**.

Continue with S1 repair: containment first, evaluator-owned splits/OOF coverage next,
then schema/durable actions/metrics and runner truthfulness. Reuse RB-001–RB-012;
new IDs RB-013–RB-016 above cover the dispatcher and runner defects. Keep the accepted
RB-002 callable argument-level label fix, but do not call it a complete security
boundary while arbitrary custom code executes in the controller.
