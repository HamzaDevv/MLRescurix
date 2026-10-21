# MLRescurix v2: refined architecture and robustness campaign

Updated: 2026-10-21 (Asia/Kolkata). Review baseline: `5be287b`.
Status: implementation specification and test plan; upgrade certification is **NOT VERIFIED**.

This is the current plan for the requested 3–5 implementation sessions. It supersedes
the implementation order and unsupported guarantees in the [supplied report](ROBUSTNESS_REPORT_SOURCE.md)
and the old P9 continuation prompt. Historical release receipts remain historical evidence.
Track actual completion in [ROBUSTNESS_PROGRESS.md](ROBUSTNESS_PROGRESS.md); use
[BENCHMARK_SUITE.json](BENCHMARK_SUITE.json) for the five-problem test manifest.
The [Gemini Argon disposition](ROBUSTNESS_ARGON_REVIEW.md), amendments A1–A5, is part
of this specification: clean-room serialization, bounded mid-loop forensics,
fidelity resources, per-champion prediction lineage and corrected metric direction.

## 1. Outcome and scope

Deliver the best verified feasible model found within a declared budget, with its
preprocessing, reproducible inference, provenance, scorecard and limitations.
Retain Rust as the authoritative control plane and Python as the scientific worker.
Cover data cleaning, EDA, feature hypotheses, search, model improvement, critique,
and standalone handoff. Instrument training and tests throughout this campaign.
Post-deployment drift services and ongoing model operations remain outside scope.

Five Kaggle tasks establish evidence for specific tabular and forecasting workflows;
they cannot establish universal competence, absence of vulnerabilities, or Kaggle
Grandmaster performance. Unsupported modalities and budget-infeasible tasks must
produce an explicit refusal or alternative, never a fabricated success.

## 2. Repository-informed review

These findings are source inspection, not demonstrated exploits. Paths are relative
to the repository root. New behavior below still requires implementation and evidence.

| Report proposal / claim | What the repository shows | Final refinement |
| --- | --- | --- |
| Build sanitizer, topology, EDA and primitives | `scientific/{dirty_sanitizer,semantic_sieve,eda_agent,primitives,fe_ablation}.py` already exist; P13–P16 tests exist | Extend the existing vertical slice; do not restart or duplicate it. |
| Typed hypothesis executor already exists | No `scientific/hypothesis_executor.py` was found at this baseline | Add a strict action contract and dispatcher integrated with current hypothesis/ablation receipts. |
| Port production BOHB from MLForge | `MLForge/.agents/skills/mlforge-dev/scripts/bohb_search.py` calls `objective_stub` and synthesizes scores | Reuse existing iteration/search control paths. Implement measured multi-fidelity promotion; call it BOHB only if that algorithm is actually implemented. |
| LC-PFN prunes 85% at epoch 4 | `lcpfn_pruner.py` has a simplified slope fallback; models do not share epoch semantics | Optional experimental pruner after measured comparison. Default to conservative, metric-aware successive halving with a no-pruning control. |
| Warm-start from historical OpenML champions | `shared/meta_db.py` contains illustrative seeded champions and unscaled distance | Admit only measured, provenance-bound history with compatible task/metric/schema; normalize descriptors. Defer warm-start if evidence is absent. |
| Sealed holdout guarantees safe EDA | `eda_forensics_suite` sanitizes and profiles whatever frame/path it receives; it does not itself seal the vault | Pass an authorized development view; enforce the boundary in the control plane and worker mounts. |
| Topology-aware proxy implies safe evaluation | `iteration_loop.run_proxy_screen` samples by topology, then uses random `train_test_split` | Preserve time/group restrictions in both sampling AND splitting. No silent random fallback. |
| Topology-aware ensemble | `ParetoEnsembleSynthesizer._collect_oof_predictions` constructs shuffled KFold/StratifiedKFold | Consume immutable split assignments shared by all candidates, stacking and calibration. |
| Fold-safe feature callables | `fe_ablation._resolve_and_apply_features` passes whole train/validation frames, including target, to callable paths | Remove validation labels before any candidate callable; labels stay in the evaluator. Reject ambiguous callable contracts. |
| EDA sandbox timeout and read-only frame | `run_exploratory_python` uses in-process `exec`, a shallow frame copy, and does not enforce `timeout_s` | Route exploration through a bounded worker; prove timeout termination, label isolation, filesystem/network limits and no mutation of the source dataset. Output truncation is not isolation. |
| Peak memory is already measured per trial | `iteration_loop.get_peak_rss_bytes` reads controller `RUSAGE_SELF` lifetime peak | Add worker-tree/cgroup measurements and explicit measurement scope; do not attribute parent lifetime RSS to candidate training. |
| Existing live script proves full harness behavior | `scripts/run_kaggle_titanic_live.py` calls provider HTTP directly, uses a GitHub mirror, Python engines and `dist/` binary | Keep as historical smoke tooling. New certification must traverse the production gateway, orchestrator, scheduler, worker, evaluator and artifact publication path. |
| Current status docs describe current runtime | README/progress contain old M6/v10/test totals; current storage accepts versions through 11 | Record commit, source-built binary, image, lock versions and actual schema. Never downgrade to v10 to satisfy stale docs. |

MLForge is an ignored reference tree, not an importable runtime dependency. Inspect
licensing, correctness and dependencies before transferring any algorithm. Existing
tests passing does not resolve the cross-component findings above.

## 3. Scientific corrections to the supplied report

1. **Split before learned processing.** Only fixed, explicitly configured row-local
   parsing may precede splitting. Learned type inference, sentinels inferred from
   distributions, imputation, category vocabularies, feature selection, MI, SVD,
   target transforms and encoders fit inside each training partition. Lock group/time
   semantics using schema and declared prediction context before inspecting outcomes.
2. **Cache by evaluation boundary.** Replace “MI computed exactly once” with a bounded
   cache keyed by dataset, training row IDs, fold, seed, transform and algorithm version.
   Reuse that fold's statistics for plots; never reuse a full-data supervised statistic
   across CV folds. Cap feature pairs, rows and memory before allocating an O(p²) matrix.
3. **Correlation and MI are signals, not verdicts.** A strong valid predictor is not
   necessarily leakage; low MI or non-positive permutation importance is not proof of
   uselessness. Hard-drop confirmed forbidden fields or constant/duplicate inputs;
   otherwise use fold-local paired ablation. Include an XOR-only-signal regression.
   Quarantine is reversible, versioned and family-specific; it must not remove useful
   interactions from every baseline. Correlation-based deletion needs an ablation too.
4. **Missingness mechanisms are hypotheses.** Observed data generally cannot prove
   MAR versus MNAR. Report patterns and uncertainty. Test missingness indicators by
   ablation, rather than claiming definitive MCAR/MAR/MNAR classification.
5. **Match uncertainty to dependence.** Use paired stratified bootstrap for suitable
   independent classification rows, group bootstrap for entities, and time-block
   resampling for dependent temporal errors. Start with 1,000 replicates and record
   seed, invalid/single-class replicates and CI method. A CI on selected OOF predictions
   is conditional on selection; it does not remove adaptive-search bias or replace
   independent final evaluation. Nested CV is optional when selection-risk and budget
   justify it, not triggered solely by N < 10,000.
6. **Replace universal numeric vetoes.** Fold std 0.05, train–validation gap 0.10 and
   slice drop 15% have different meanings across accuracy, RMSE and RMSLE. Define
   metric direction, absolute/relative units, baseline and tolerances in ProjectSpec.
   Small slices get uncertainty/insufficient-support status. Do not universally veto a
   dataset because one family dominates, a text column is unused, or fewer than three
   families were affordable. Record explored families and skipped alternatives.
7. **Avoid stacking and calibration optimism.** Meta-learners, probability calibrators
   and threshold selection need a separate development calibration partition or a
   correctly nested cross-fit schedule. Fitting weights and reporting score on the same
   OOF target vector is optimistic. All base predictions must share row IDs, fold IDs,
   class order and topology. Prove the meta-training data cannot depend on the outer
   assessment labels. Freeze calibration and threshold before final holdout access.
8. **Make complexity measurable.** Default ensemble promotion requires a direction-aware
   relative improvement of at least 0.5% over the best feasible single model on paired
   development assessment: `(new-old)/max(abs(old), eps)` for maximization, the reverse
   for minimization. Declare eps and a practical absolute tolerance near zero. Treat
   uncertain gains as inconclusive; enforce inference p95, memory and artifact budgets.
   This is a configurable starting policy, not a guarantee of statistical significance.
9. **Stop early for the right objective.** Meeting latency alone cannot skip model
   improvement. Stop when the frozen quality target AND all feasibility constraints
   are satisfied, improvement has plateaued, or a declared budget is exhausted.
10. **Describe delivery honestly.** “Independent of MLRescurix” still requires Python
    and pinned model libraries. Preserve existing bundle manifests and filenames unless
    a versioned migration is necessary. Include dependency lock, hashes, input/output
    schemas, raw-input smoke vectors, preprocessing and limitations. Standard-library
    HTTP and FastAPI are alternative serving profiles, not interchangeable zero-dependency
    claims. Load only trusted, verified serialized models; pickle/joblib is executable.
11. **Measure latency promises.** Separate warm dispatch, IPC, data loading, model fitting
    and prediction latency. Publish p50/p95 with host/image/sample count and cold/warm
    distinction. `<50 ms` is a measured target for dispatch, not an unconditional end-to-end SLA.
12. **Verification is layered.** AST/type/lint/unit checks validate code properties, not
    statistical validity or sandbox containment. Require independent metric recomputation,
    leakage tests, resource enforcement and clean-room inference as separate gates.

The fold-local processing and independent assessment requirements follow the
[scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
and [nested evaluation example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
These sources support the evaluation principles; runtime dependencies remain pinned
to the repository locks, not the latest documentation version.

## 4. Refined lifecycle and ownership

```text
ProjectSpec + capability/budget preflight
  -> dataset provenance + immutable row IDs + split/vault commitment
  -> development-only EDA + fold-local preprocessing
  -> frozen simple baseline + bounded hypothesis/feature ablations
  -> measured multi-fidelity candidate search
  -> optional cross-fitted ensemble/calibration + independent development critique
  -> candidate/code/config freeze + verification
  -> single final holdout evaluation per dataset campaign
  -> verified standalone bundle + evidence report
```

Rust owns ProjectSpec, authorization, immutable splits, budget admission, job leases,
canonical evaluation receipts, artifact publication, vault consumption and recovery.
Python workers own transformations, model fit/predict and bounded exploratory outputs.
The trusted evaluator owns labels, row/class alignment and official score computation.
Worker-supplied scores are advisory until independently recomputed.

The provider gateway is the only authorized route for live agent requests. Model
workers receive no provider/Kaggle credentials and no general network access. Dataset
downloads happen in a separate preparation step with explicit provenance. Final
holdout features are passed only through the trusted prediction/evaluation boundary;
labels are never mounted in candidate or exploratory worker namespaces.

### Ten typed actions, without removing the coding harness

Add `contracts/v2/hypothesis_action.schema.json` and a dispatcher, ideally
`scientific/hypothesis_executor.py`. Use strict enums, `additionalProperties: false`,
bounded parameters, allowed column IDs and task checks. Reuse existing research ledger
and ablation contracts. DSL actions are a safe default; sandboxed candidate code remains
an explicitly governed capability needed for richer problems.

Every action includes `action_id`, `parent_candidate_id`, `source_hash`, `split_id`,
`topology_hash`, `seed`, `action`, `parameters`, `budget_id` and `rationale_ref`.
Receipt includes versioned transform lineage, fitted-fold references, paired metric
delta, resource measurements and `confirmed/refuted/inconclusive/failed` settlement.
Same action ID and payload replay idempotently; changed payload with same ID is rejected.

| Action | Preconditions / acceptance behavior |
| --- | --- |
| APPLY_SKEW_TRANSFORM | Fit on fold training only; Box-Cox requires positive values; use domain-safe alternatives and finite-output checks. |
| ENCODE_HIGH_CARD_CAT | Nested OOF target encoder; preserve group/time semantics, class order, seeded training noise and unseen-level fallback. |
| DECOMPOSE_QUASI_ID | Fit counts/aggregates on training only; raw IDs never become lookup keys to validation labels. |
| EMBED_SHORT_TEXT | Fit vocabulary/SVD on training only; cap vocabulary, nnz and dense allocations; empty/unseen-text behavior tested. |
| ADD_MNAR_INDICATOR | Name retained for compatibility; receipt says missingness hypothesis, not proven MNAR. |
| ADD_INTERACTION | Bounded train-derived candidate pairs, reversible quarantine access; guard division and overflow; retain only on paired development evidence. |
| DELETE_FEATURE | Reversible feature-set version; forbidden inputs can be hard-dropped, uncertain features require ablation. |
| REORDER_MODEL_PRIORITY | Control-plane scheduling action within approved families and remaining budget; it is not a transformer. |
| UPDATE_SEARCH_SPACE | Control-plane validated bounds/version; cannot expand resource, permission or holdout access. |
| TRANSFORM_TARGET | Regression only; fit inside fold, validate domain and inverse transform, score in specified metric space. |

JSON validation controls syntax and allowed choices; numerical determinism still
depends on libraries, seeds, threads and hardware. Record tolerances and environment.

## 5. Five selected Kaggle problems

Official competition pages were checked on 2026-10-21. Some evaluation pages return
only a JavaScript shell to automated readers. Metric definitions below are the intended
protocol; Session 1 must archive the readable official evaluation/data descriptions,
confirm definitions and file fingerprints before locking any benchmark. Data access,
rule acceptance and credentials are not verified by opening a public page.

| ID / official source | Task, target and primary metric | Protocol and features exercised |
| --- | --- | --- |
| K1 [Titanic](https://www.kaggle.com/competitions/titanic) | Binary; `Survived`; accuracy ↑. Secondary balanced accuracy/log loss. | Derived family/ticket group-disjoint development and final split, with group mapping frozen from input metadata. Missing data, Name text, Ticket/Cabin quasi-IDs, unseen categories, feature ablation and repair. This grouped score is a robustness assessment, not a Kaggle leaderboard estimate. |
| K2 [House Prices](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) | Regression; `SalePrice`; RMSE of natural log prices ↓, with positive price predictions. | Fixed random folds from labeled training data. Skew/domain-safe target transformation, categorical encoding, missingness, paired ensemble parsimony and train/serve parity. Do not silently replace log(price) metric with log1p(price). |
| K3 [Forest Cover Type](https://www.kaggle.com/competitions/forest-cover-type-prediction) | Seven-class; `Cover_Type`; accuracy ↑. Secondary macro-F1/log loss. | Stratified folds, explicit class order 1–7 and ID exclusion. Numeric/one-hot mixtures, rare-class slices, probability alignment, batching and large unlabeled inference. Official labeled train is small; do not call this a large-training-memory benchmark. |
| K4 [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | Imbalanced binary; `isFraud`; ROC-AUC ↑. Secondary PR-AUC/log loss/calibration. | Chronological splits on `TransactionDT`, transactional/identity join validated on `TransactionID`, explicit time gap and group-overlap diagnostics. Wide mixed features, category explosion, rare-event sampling, memory limits, evaluator integrity. Proxies preserve time and join coverage. |
| K5 [Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) | Panel forecasting; `sales`; RMSLE ↓. Secondary MAE/WAPE. | Forecast next 15 dates jointly across selected `(store_nbr,family)` series; three expanding origins and a final contiguous 15-day vault. Causal lags/rolling features, known-at-forecast covariates, joins and seasonal-naive baseline. Time boundaries use dates, never arbitrary row counts. |

K4/K5 begin with deterministic bounded subsets (manifest records selected entities,
date range, row count, seed and join coverage). Full-data scale is a separate result;
subset success does not certify full-data memory behavior. Pin and process full input
streaming or with admitted memory; never read all rows merely to choose the proxy.
If full scale is infeasible, report the largest measured feasible scale and refuse
the larger request. Do not silently substitute a mirror, OpenML variant or synthetic data.

Only labeled competition training data supply local development/holdout targets.
Kaggle test files are unlabeled inference/schema checks. Leaderboard submission is
optional, outside this local acceptance protocol, and is not performed automatically.
These tasks do not validate vision, long-document text, reinforcement learning or
all unsupervised adapters; existing adapters need their own certification profiles.

## 6. Session plan (default five; compress only after gates pass)

| Session | Concrete implementation and experiment scope | Exit evidence |
| --- | --- | --- |
| S1 — trust boundaries and repeatable tests | Reconcile source/image/schema, add campaign preflight/runner, fix EDA isolation, target exposure and group/time split propagation. Add strict DSL contract/dispatcher and fault regressions. Prepare official datasets/split commitments. Run deterministic small-scale baselines. | No candidate access to validation/holdout labels, enforced EDA timeout, shared topology through proxy/full CV/ensemble; manifests and raw logs; no unexplained regression failures. |
| S2 — scientific improvement and first live slice | Finish fold-local EDA/cache/quarantine and action ablations. Integrate actual multi-fidelity results with bounded restartable scheduler. Cross-fitted stacking/calibration and metric-aware critic. Run K1/K2 through real production provider/orchestrator and bundle path. | Both tasks have matched baseline/upgraded development results, all ten actions have meaningful success/rejection coverage, live role/tool/repair receipts, raw-input bundle parity. |
| S3 — task diversity and scale | Extend adapters only where K3–K5 require it. Exercise class order, time joins/lags/horizons, minority preservation and resource admission. Run production live flows for K3/K4/K5 and concurrent isolation tests. | All five tasks have live provider + real training evidence, each with trace/resource receipts or an explicit failed/blocked gate. No fake scores or silent topology fallback. |
| S4 — fix → run → fix campaign | Run 3 development seeds per task with matched baseline/upgrade budgets, fault matrix, constrained-memory variants, restart/SSE replay and soak. Fix reproducible defects; rerun failing case and affected cross-task checks. | Complete pass/fail/not-applicable matrix, issue ledger, bounded spending, no unresolved critical/high findings or unexplained telemetry gaps. |
| S5 — freeze and certify | Freeze source/config/split manifests after development passes. Run final regressions, clean-room bundle CLI/HTTP and model validation. Consume each task's final vault once, produce final report and limitations. | Five task scorecards, no test-driven retuning, resource/cost summary, vulnerability dispositions, reproducible bundle and evidence hashes. |

Three-session route: combine S1+S2 and S3+S4 only if their exit gates actually pass;
keep S5 independent. Sessions are resumable work units, not guaranteed durations.
If gates fail, document carryover instead of declaring completion to meet the count.
LC-PFN, speculative meta-history and additional modalities are not blockers for the
initial robust release. Start with S1, not a wholesale scientific-kernel rewrite.

## 7. Experiment discipline and bounded repair loop

1. Capture original failure, exact source/config/data/split hashes, seed, resource
   profile and trace. Assign a durable issue ID and suspected boundary.
2. Write the smallest regression that exposes the defect. Implement a targeted fix.
3. Rerun the failing case, then affected cross-task checks; compare equivalent budgets.
4. Promote only if correctness and constraints pass. A quality tie or inconclusive
   improvement retains the simpler verified feasible candidate.
5. Repeat within budget; default maximum three repair attempts per issue/run, twelve
   candidate evaluations per problem/seed, and three consecutive non-improvements.
   These are independent ceilings; the first exhausted bound stops the loop.
6. After critical/high fixes, require two consecutive complete development validation
   passes on the final source revision. Each pass includes all five tasks at the
   frozen development seed 2026 in upgraded mode plus required integration/fault
   checks; it need not repeat the whole three-seed matched comparison. Keep each
   invocation as a separate receipt. Then freeze and assess the sealed final sets.

Default development seeds: `[2026, 42, 101]`. Use a single committed final split per
task, independent of development seeds; seed-specific folds are derived only inside
the remaining development rows. Compare the pre-upgrade baseline and upgraded system
on identical development rows/folds/seeds, CPU/memory and wall/trial budgets. Record
baseline implementation commit and provider policy separately; a deterministic baseline
has zero provider spend. Repeated seeds are variability checks, not independent datasets.

One production live solve per task may also be one of the matched upgraded runs;
label that overlap rather than counting it twice. Keep API resilience/fault injection
receipts separate from organic provider failures. A canned response or fixture never
counts as real-provider evidence. Measured model scores must be recomputed by the
trusted evaluator from predictions, not copied from LLM text or worker JSON.

Final holdout labels stay inaccessible throughout the repair loop. Select the
final candidate/seed/ensemble rule using development evidence, then freeze its hash.
One atomic vault consumption per dataset campaign evaluates only that frozen artifact.
A failed final gate is reported; it does not authorize another tuning loop on that
holdout. Any new campaign needs a genuinely untouched final set and explicit lineage.
Predeclare a deterministic degraded-baseline delivery policy if needed; do not select
between candidates after seeing final outcomes.

## 8. Required feature and vulnerability coverage

Five datasets alone will not naturally trigger every feature. The runner must combine
natural task traces with explicit fault and contract cases, linked to these IDs.
Each case needs an expected outcome, actual receipt and source revision.

| Case group | Task anchors | Required assertions |
| --- | --- | --- |
| A1 roles/delegation/steering/context | K1, K4; lightweight full-role pass on all | Lead delegates through registered roles, result lineage is durable, role cannot bypass permissions, steering applies once at turn boundary, bounded context preserves pending tools. |
| A2 ten actions and hypothesis loop | K1, K2, K4 | Every action has valid execution and invalid/precondition case, paired ablation/settlement, idempotent retry, reversible feature version and budget-bound stop. Include XOR, negative target, unknown categories and empty text. |
| A3 autonomous diagnosis/repair | K1 plus K3/K5 affected paths | Deliberate syntax, shape, class-order and timeout failures; trace failure→diagnosis→agent patch→gate→successful rerun. No manual patch credited as autonomous. |
| H1 evaluator/vault | All | Workers cannot read validation labels; forged score and misaligned/duplicate/missing row IDs fail; label access and repeated/concurrent vault access fail closed; crash after consumption cannot reopen tuning access. |
| H2 topology/OOF/calibration | K1, K3, K4, K5 | No group overlap where group-disjoint; no future labels or future-derived transforms; all stages share splits; class probabilities align; separate meta/calibration assessment proven. |
| H3 leases/ledger/restart/storage | K4/K5 plus disposable harness state | Kill at dispatch, fit, publication and finalization; reconcile without duplicate job/charge/publication; ENOSPC and 128 MiB managed-state admission tested; reconnect SSE without lost canonical events. |
| H4 artifact and standalone delivery | All | CAS/source/manifest tamper fails, path traversal/symlink escape refused, changed data/schema rejected or diagnosed, installed bundle outside checkout reproduces raw-input predictions and HTTP contract. |
| R1 resource containment | K3 inference; K4/K5 fit; synthetic faults | Worker-child allocation/CPU/fork pressure stays in configured cgroup; OOM/timeout produce bounded diagnostics; sibling job survives; no orphan process, stale lease or disk growth after cleanup. |
| P1 provider resilience and economics | All live flows; transport fixtures | Timeout/429/5xx retries bounded with jitter/deadline; honor Retry-After when available; 401/403 invalid credentials/config stop; malformed JSON/truncated actions fail safely; ambiguous billed request is reconciled or counted conservatively. |
| V1 prompt/data injection | K1 text, K4 categorical; synthetic fixtures | Malicious text/column names/EDA logs cannot change role, tools, budget, outbound data policy, filesystem mounts or holdout permission. Canary secret never appears in model-worker environment, outbound payload or committed evidence. |
| V2 malicious code/exfiltration | Isolated disposable candidate workers | Attempt network egress, host-file reads, arbitrary path writes, fork bomb, infinite loop and model-artifact substitution. Verify denial using receipts; never aim probes at external hosts or other user projects. |
| Q1 useful modeling and uncertainty | All | Beat a task-appropriate dummy/seasonal-naive development baseline; compare with the pre-upgrade simple baseline under frozen tolerances; inspect slices, calibration where applicable and CI assumptions. A failed run is not a low numeric score. |

Existing optional features outside this release profile receive explicit
`not_in_release_profile` status and cannot inherit certification from these tasks.
Required cases cannot be waived by marking them not applicable. Document residual
severity and owner; no open critical/high finding is compatible with certification.

## 9. Trace, CPU, memory and cost protocol

Every run records run/project/task/candidate/attempt/parent IDs, git SHA and dirty diff
hash, dataset/split/topology/action/config hashes, image digest, binary hash, locks,
seed, monotonic timestamps and event sequence. Trace spans cover provider request,
role/delegation, tool validation, queue/admission, worker dispatch, preprocessing,
fit/predict, evaluator, hypothesis settlement, critique, repair and publication.
Store canonical event/receipt references; dashboard screenshots are supporting evidence.

Collect samples every 250 ms where practical, plus mandatory stage boundaries. Record
gaps and collector overhead. Use kernel `memory.peak` for peaks that sampling could
miss; for warm pools reset the per-job accounting boundary or use a fresh cgroup.
The existing `crates/harness/src/resource_monitor.py` provides before/after cgroup
receipts; extend with time-series measurements rather than replacing enforcement.

| Measurement | Required definition |
| --- | --- |
| Memory | Worker/cgroup memory.current and memory.peak, limit, swap, OOM events; controller/worker-tree RSS separately; units bytes. Child allocations and native model libraries must be included. Python tracemalloc is supplementary only. |
| CPU | cgroup cpu.stat usage delta, wall interval, quota/period, throttling and CPU time. `quota_utilization_pct = 100 * delta_usage_usec / (delta_wall_usec * allocated_cores)`. Report raw CPU seconds and host core count too. |
| Disk / I/O | Managed ledger + WAL/SHM + CAS + staging totals separately from raw datasets/export; reservation high-water mark, read/write bytes and cleanup residuals. 128 MiB is a managed-state boundary, not total training memory. |
| Latency | Queue, cold start, warm dispatch, training, evaluation and serving p50/p95; benchmark batch sizes 1, 32 and 1,024 where applicable, 10 warmups + 100 timed calls, declared concurrency and hardware. |
| LLM | Requested/actual model ID, provider request ID, prompt/output/reasoning/cache usage with provider-specific non-overlap semantics, retries, latency, billed cost and estimate basis. Missing cost is unknown, never zero. |
| Reliability | Success/failure/timeout/OOM by stage; repairs attempted/accepted, invalid actions, orphan count, lease leaks, hash mismatches, recovery latency, event replay completeness and reproducibility tolerance. |

Linux cgroup v2 is the enforcement/certification target. On this macOS host, report
host metrics and Docker-VM metrics separately. Docker daemon availability and image
pinning must pass first. Never label macOS RSS-only observations as Linux memory
certification. Kernel accounting semantics are described in the
[cgroup v2 documentation](https://docs.kernel.org/admin-guide/cgroup-v2.html).

### Initial campaign limits

These are conservative starting defaults in the manifest, not permission to exceed
existing project/provider policy. Capability preflight can reduce scale or refuse.

- Serial main task execution, 2 CPU cores per job; BLAS/OpenMP/model thread limits
  agree. One concurrent-isolation test admits two jobs only if combined leases fit.
- K1/K2/K3: 2 GiB worker memory; K4/K5: 4 GiB. Reserve at least 30% effective host/VM
  memory and never oversubscribe. A constrained fault profile uses 512 MiB in a fresh
  disposable container; expected refusal/OOM is success only when contained and diagnosed.
- Per problem/seed/mode wall caps: 10/15/20/30/30 minutes respectively. Same caps for
  baseline and upgrade; count preprocessing/OOF/calibration in the cap. Campaign wall
  ceiling 20 hours including retries; resumable across sessions, not an estimate.
- Live-provider ceiling: USD 0.30 total campaign across all sessions/runs/retries, USD 0.05 per main live task run, 20
  requests including retries and 80,000 total requested tokens per run. Ceiling applies
  at whichever bound is exhausted first. Reserve worst-case request cost before sending;
  if pricing/accounting is unavailable, refuse the next billed request. Change caps
  explicitly in the manifest and ledger; model ID/prices must be current at execution.
- Raw-data staging cap 20 GiB; retained experiment artifacts 8 GiB; managed state
  128 MiB under its existing accounting contract. Preflight joins, covariance/MI,
  TF-IDF, dense conversions and ensemble copies. Archive selected evidence, delete
  disposable intermediates through existing ownership/GC rules, and record refusals.
- Requested model: `deepseek/deepseek-v4.1-flash-1017` through OpenRouter, pinned
  to `inference-net/fp8`, with fallbacks disabled. Reserve USD 0.05 contingency within
  the USD 0.30 total; revalidate endpoint pricing before live use. This budget may
  restrict coverage; report incomplete evidence instead of increasing the cap.
- Provider transport: at most 2 retries per request, 60-second attempt timeout and
  180-second total request deadline, all within run caps. No unbounded fix/run loops.

## 10. Evidence layout and final acceptance

Proposed runner: `scripts/run_robustness_campaign.py` (to implement in S1). It should
support preflight, deterministic development, live development, faults and final
certification modes, resumable run IDs and nonzero exit on required skipped/failed
gates. The manifest is a planning input until this runner and its schema exist;
do not document speculative command-line flags as working commands.

```text
docs/evidence/v2/robustness/<campaign>/<run_id>/
  manifest.json             # immutable config/data/split/environment commitments
  summary.json              # explicit pass/fail/blocked/inconclusive per gate
  traces.jsonl              # redacted canonical event references and span timing
  resources.jsonl           # time samples, scope, units and collector health
  provider_usage.json       # redacted request IDs, tokens, retry and cost accounting
  metrics.json              # evaluator results, baselines, fold/slice/CI definitions
  faults.json               # expected vs observed failure/containment/repair receipts
  artifacts.json            # source/binary/image/output hashes and bundle references
  logs/                     # bounded, redacted execution logs
```

Raw data, credentials, full provider prompts/responses and private row records stay
outside committed evidence. Store heavyweight outputs under owned runtime storage
with manifest hashes; keep reproducible small receipts in docs. Persist a redacted
counterexample for every fixed issue.

Final acceptance requires all of the following:

1. Five authentic dataset manifests and official metric definitions are locked; all
   five complete the production live flow with independent model scores and trace IDs.
2. The 3-seed matched development table and two complete post-fix validation passes
   are available; required feature/fault coverage has no unexplained skips.
3. No validation-label exposure, unauthorized final-label access, forged evaluation,
   containment escape or unresolved critical/high issue. Unsupported paths fail closed.
4. CPU/memory/time/disk/cost caps are enforced and measured; peaks include workers;
   no orphan jobs, resource leaks or unexplained trace gaps. Performance limitations
   remain explicit even if a simpler feasible model is delivered.
5. Quality tolerances are frozen before final access. Direction-normalized relative
   development non-inferiority tolerance defaults to 0.5%, with explicit near-zero
   absolute tolerance per metric; paired uncertainty and slices are reported. If an
   upgrade loses, retain the simpler verified baseline and mark the improvement claim
   failed. “Robust” does not imply every proposed algorithm improved accuracy.
6. Each dataset's final frozen candidate passes its declared final quality/feasibility
   criteria on its single-use holdout, or the certification gate remains failed.
7. Bundle integrity, pinned dependency installation outside the checkout, raw-record
   CLI/HTTP predictions and malformed-request handling pass; scorecard, model card,
   limitations, data lineage and reproduction instructions are complete.

## 11. Continuation and current evidence

At every session end, update the progress ledger with source revision/diff, files,
actual commands, exit codes, seeds, evidence paths, open issue severity, spend/time
remaining and the next smallest action. Do not edit historical receipts to make
current status look green. Re-read git status and current code before resuming.

Current review evidence is under
[`docs/evidence/v2/robustness/20261021-plan-review/`](../evidence/v2/robustness/20261021-plan-review/).
It contains only environment checks and existing local regression results. The
five-problem campaign, live-provider certification, new containment regressions and
upgrade fixes are not complete in this planning session.

Dataset preparation should follow the [official Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli).
If access is unavailable, record the blocked prerequisite and continue independent
synthetic/contract work; do not claim a synthetic substitute is a Kaggle result.
