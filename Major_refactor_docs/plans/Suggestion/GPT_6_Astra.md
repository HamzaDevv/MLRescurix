**MLRescurix should first become a reliable experimental system, then a more expansive model builder.** The handoff describes useful foundations—separate authority and execution tiers, independently recomputed predictions, candidate identities, and bundle verification—but does not yet demonstrate autonomous scientific progress.

My central recommendation is to replace the loosely guided turn loop with a persistent experiment controller that distinguishes scientific hypotheses, execution failures, evaluation evidence, and spending authority. Adding more models before fixing those contracts would mostly enlarge the failure surface.

I reviewed only the supplied handoff. I did not inspect its referenced repository, artifacts, provider records, or datasets. Below, **reported** means asserted by the handoff; architectural judgments and proposed thresholds are my recommendations, not verified implementation facts.

**1. Prioritized diagnosis**

| Priority | Reported evidence | My assessment and plausible alternatives |
|---|---|---|
| **P0: Scientific evidence is insufficiently protected** | Repeated selection on two folds; no holdout; one candidate’s score attributed to another in Run 6 | The best development score cannot support a generalization claim. Independent metric recomputation also cannot establish that preprocessing, training, or row alignment was leak-free. |
| **P0: Execution and proposal contracts disagree** | Research allowed in Rust but rejected by Python; classification parameters passed to Ridge | This is primarily an interface defect. A capable planner cannot reliably operate against contradictory capabilities and parameter schemas. |
| **P0: Spending feasibility is opaque** | Approximately $0.01038 remaining under the reported campaign ceiling; substantial unresolved liability | Conservative reservations may be correct. The defect is incomplete reconciliation and misleading operational headroom, not necessarily excessive financial strictness. |
| **P0: Data isolation is incomplete** | Writable hard links between staged data and scratch | This establishes a mutation opportunity, not corruption. Hashes detect changes; they do not prevent mutation or restore lost data. |
| **P1: Feedback does not change search behavior** | Seven consecutive vetoed neural variations; repeated invalid parameters | The controller lacks enforced recovery transitions and durable experiment memory. Prompt advice alone is insufficient. |
| **P1: Search space is narrow and poorly factored** | Fixed text compression and limited transforms/models | Representation restrictions may matter more than missing neural architectures. Several boosting libraries do not constitute several fundamentally different hypotheses. |
| **P2: Remote execution and research remain unproven** | No demonstrated live research or Kaggle execution | These are useful capabilities, but neither is required to establish a strong local baseline or reliable autonomous recovery. |

Several handoff explanations deserve direct challenge:

- **A larger train–validation gap does not establish that a candidate is worse.** A model can improve validation error while fitting training data more closely. Conversely, an underfit model can have a reassuringly small gap. The veto may have rejected useful candidates.
- **Repeated identical scores need investigation before blaming planning.** Possibilities include ignored parameters, normalization to the same effective configuration, stale cache keys, reused predictions, deterministic repeated training, or score rounding. Compare resolved configurations and prediction hashes.
- **The identical Run 6 and Run 7 winning scores do not establish independent replication.** They could reflect a faithful rerun, reuse of a prior artifact, or unintended caching.
- **Run 8’s proposed parameter patch is scientifically risky.** Logistic-regression `C` and Ridge `alpha` are not interchangeable experimental intent. Silently ignoring `penalty` is worse than returning a structured validation error. An explicit compatibility translation can exist, but must be documented, opt-in, and recorded.
- **“Unused in these runs” does not mean “untouched.”** The remaining source rows may have been inspected, used historically, duplicated across partitions, or exposed through aggregate analysis.
- **The first 75,000 rows may be unrepresentative.** Ordering by category, time, seller, or ingestion batch could affect the results.
- **Zero disconnected counters do not prove zero external activity.** Conversely, existing adapters do not prove that remote execution works.
- **Single-threaded CPU execution is a throughput choice, not inherently an autonomy defect.**
- **More budget would permit more experiments, including more repeated mistakes.** It does not fix experimental selection.

Evidence most likely to change my assessment: raw traces and receipts; exact commit plus working-tree diff; effective candidate configurations; cache keys; row-indexed predictions; preprocessing fit boundaries; critic formula and thresholds; source-data lineage; and provider reconciliation records.

**2. Architecture for meaningful, extensible experimentation**

Preserve the Rust/Python separation, but make authority boundaries explicit.

| Component | Responsibility | Must not control |
|---|---|---|
| Task/protocol service | Dataset identity, task type, prediction-time information, split topology, metric, release criteria | Model selection based on lockbox feedback |
| Research planner | Hypotheses, expected mechanisms, diagnostic questions, experiment proposals | Spending approval, evaluator labels, policy changes |
| Experiment controller | Persistent graph, validation, deduplication, scheduling, recovery, stopping | Financial-limit overrides |
| Capability registry | Versioned preprocessing, estimators, losses, execution requirements, parameter schemas | Implicit parameter reinterpretation |
| Execution broker | Local/remote jobs, isolation, resource leases, cancellation, artifacts | Holdout access or direct provider credentials |
| Trusted evaluator | Prediction alignment, metric computation, comparison receipts, lockbox access | Executing untrusted model code with evaluator credentials |
| Resource authority | Atomic reservations, actual usage, unresolved liabilities, reconciliation | Scientific conclusions |
| Evidence store | Immutable events, lineage, artifacts, reproducible materialized state | Mutable “latest score” as source of truth |

Start with a transactional event store and artifact store. A distributed workflow platform is unnecessary until workload scale justifies it.

The core object should be an **experiment specification**, not an action name with loosely interpreted arguments:

```text
ExperimentSpec
  task_contract_hash, dataset_version, protocol_hash
  hypothesis_id, parent_experiment_ids
  mechanism, expected_observation, falsification_condition
  changed_axes, comparison_baseline
  preprocessing_graph
  estimator_plugin + version + resolved_parameters
  loss, target_transform, prediction_postprocessing
  fidelity, fold_ids, seeds
  source_hash, dependency_lock_hash, environment_digest
  resource_request, data_access_policy
```

Useful interfaces:

```text
registry.validate(spec) -> ValidatedSpec | StructuredErrors
controller.propose(spec) -> Scheduled | CachedResult | Rejected(reason)
authority.reserve(request, idempotency_key) -> Lease | Denied(reason)
broker.submit(validated_spec, lease) -> JobHandle
evaluator.evaluate(prediction_manifest, protocol) -> EvaluationReceipt
controller.observe(receipt_or_failure) -> UpdatedGraph + NextDecision
```

Maintain two identities:

- **Experiment identity:** effective scientific configuration, data, protocol, code, fidelity, seeds, and relevant environment.
- **Attempt identity:** an individual execution or retry.

A repeat with a new seed is a replication, not a new scientific paradigm. A retry after infrastructure failure is not a new hypothesis.

**Factor the search space into meaningful axes.**

| Axis | Practical first experiments | Expansion when evidence supports it |
|---|---|---|
| Data quality/preprocessing | Missingness indicators, robust scaling, categorical handling, unit normalization, label and duplicate diagnostics | Task-specific cleaning plugins; explicit noisy-label strategies |
| Text representation | Word and character TF-IDF directly into sparse linear models; hashing; configurable SVD | Frozen pretrained embeddings; subword models; fine-tuned encoders |
| Structured features | Length/count features; reusable quantity/unit extraction; interactions supported by error analysis | Learned feature representations; cross-fitted target encoding |
| Model family | Constant baseline, Ridge/ElasticNet, one strong GBDT, existing MLP | Additional trees, kernel approximations, task-appropriate neural models |
| Neural architecture | Existing MLP with controlled optimization and regularization | FT-Transformer, text CNN, pretrained text encoder, multimodal fusion |
| Target/loss | Raw-target versus log-target fitting; MAE, MSE, Huber where applicable | Quantile/distributional objectives; carefully tested metric surrogates |
| Ensemble | Fixed simple average of complementary candidates | Constrained blends and properly cross-fitted stacking |

For Amazon, **sparse word/character text regression is a higher-priority missing baseline than TabNet**. Compressing all text to 16 components may erase useful lexical and numerical information. That is a hypothesis to test, not an established cause.

Tabular-model research supports comparing strong tree and neural baselines rather than presuming architectural complexity wins. The relevant studies also concern particular benchmarks; they do not predict the winner on this catalog dataset. [papers.neurips.cc](https://papers.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html?utm_source=chatgpt.com)

Use matched comparisons: same data, folds, and reasonable resource allocation; initially change one substantive axis. After identifying promising axes, test their interactions. Avoid an exhaustive Cartesian product.

For extensibility:

- Plugins declare supported tasks/modalities, typed parameters, fit/predict/serialize contracts, dependency hashes, resource needs, and determinism limitations.
- Generated code enters a **quarantine workflow**: build, synthetic contract tests, bounded execution, artifact inspection, then campaign eligibility.
- No arbitrary package installation or network access during training. Resolve dependencies through a separate controlled build step.
- Run untrusted code without credentials, host mounts, or evaluator access. Stronger isolation such as microVMs is appropriate for hostile-code assumptions; a Python subprocess alone is inadequate.
- Mount data read-only from an immutable snapshot. Writable outputs belong to a separate filesystem.
- Record hardware and libraries. Require exact reproducibility where feasible and predefined numerical/statistical tolerances where GPU nondeterminism prevents bitwise parity.
- Load model artifacts in isolated inference workers. An integrity hash does not make a serialized object safe.

Future tasks should use modality-specific adapters and protocol requirements: grouped classification, chronological forecasting, image augmentation, text tokenization, and multimodal alignment. Do not pretend one tabular pipeline generalizes merely by accepting a different filename.

**3. Recovery, exploration, and stopping**

Replace undifferentiated `rejected` turns with structured outcomes:

```text
invalid_spec | infrastructure_failure | resource_denied
integrity_failure | scientific_negative | promising | inconclusive
```

Each outcome implies a different action.

| Trigger | Required response |
|---|---|
| Invalid parameter | Return valid names/types and task context; permit one bounded repair before execution |
| Invalid member of a portfolio | Preserve independent valid candidates; reject the malformed member |
| Exact completed duplicate | Return its prior evidence without retraining |
| Repeated prediction hash despite changed parameters | Check effective configuration, caching, and parameter application |
| Two consecutive near-duplicates | Require a different representation, loss, feature mechanism, or model family |
| Two gap-based warnings | Inspect fold variance, learning curves, leakage, and training-score comparability |
| Infrastructure failure | Bounded retry with backoff and distinct attempt accounting |
| Budget denial | Reconcile, reduce fidelity, or stop; never reinterpret the limit |
| Several valid experiments without useful progress | Compare underexplored axes, research a specific uncertainty, then pivot or stop |

These are proposed defaults, configurable in campaign policy.

Maintain durable hypothesis records containing:

- The mechanism and evidence motivating the experiment.
- What would support or contradict it.
- Changed factors and controlled factors.
- Observations, uncertainty, execution status, and cost.
- Whether the result refutes the hypothesis, merely fails to support it, or is uninterpretable.
- The resulting next action.

For example: “Increasing SVD dimensionality improved both paired folds but increased memory substantially; test sparse linear text next to determine whether compression is unnecessary.”

Use a lightweight portfolio scheduler initially. A reasonable starting allocation is 50% refinement, 30% underexplored mechanisms, and 20% replication/diagnostics, excluding protected finalization resources. These are scheduling preferences, not spending entitlements.

Research when it can resolve a decision: unexplained model behavior, unknown loss support, a representation bottleneck, an unfamiliar modality, or a persistent runtime failure. Do not search after every disappointing score.

Stop when no feasible high-value experiment remains, the authorized resource envelope is exhausted, evidence is too weak to proceed safely, or the preregistered plateau criterion is met. Preserve a best eligible incumbent and an explicit “no reliable improvement” outcome.

**4. Web research with defensible boundaries**

Separate research into three stages:

1. **A private analyst identifies an abstract question.**
2. **An isolated researcher searches without dataset access.**
3. **The planner receives cited claims and proposed experiments through a constrained interface.**

A permissible query might be “character n-gram regression for product price prediction.” It should not contain product records, proprietary schema names, customer identifiers, label examples, internal paths, or verbatim error payloads containing data.

Protection should include:

- Approved abstract task descriptions and query templates.
- Local checks for sensitive literals and identifiers as defense in depth.
- Cumulative disclosure limits: many individually innocuous queries can reveal sensitive context.
- No direct search egress from training workers.
- The same disclosure discipline for LLM prompts, telemetry, and remote dataset uploads.
- A recorded policy for whether external pretrained models and data are permitted.

Search results, PDFs, repositories, and code snippets are untrusted. NIST describes indirect prompt injection as an attack through attacker-controlled resources; a 2 KB summary limit does not establish a security boundary. [CSRC](https://csrc.nist.gov/glossary/term/prompt_injection?utm_source=chatgpt.com)

The researcher should have no dataset secrets, execution privileges, ledger access, or ability to change policy. Source instructions cannot authorize downloads, commands, or new destinations. Retrieved code follows the same quarantine process as generated code. Constrain fetch destinations, redirects, file sizes, and content types.

For every adopted claim, preserve source URL, retrieval time, content hash, relevant excerpt, confidence, applicability limitations, and the experiment it motivated. Prefer original papers and official documentation; corroborate consequential claims.

**Accounting must distinguish attempts from results.**

Record:

```text
request_id, retry_of, provider, sanitized_query_hash
reserved_units, dispatch_state, provider_request_id
success/failure/timeout, result_count
usage_evidence, estimated_charge, verified_charge, unresolved_amount
```

A dispatched timeout consumes an attempted-query unit and may carry financial uncertainty. A local rejection is recorded but is not an external request. A retry is another attempt. Empty results are still successful requests if the provider processed them. Track document fetches separately if they have distinct costs or limits.

**5. Resource governance and reconciliation**

The generous operational profile can be:

- 5,000,000 model tokens.
- 30 Kaggle accelerator hours.
- 500 dispatched search attempts.

These are **upper bounds, not an instruction or authorization to consume them**. Existing campaign and per-run financial limits remain binding unless explicitly amended.

Before a campaign, present the effective constraint set: operational ceilings, financial limits, unresolved liabilities, provider limits, local CPU/RAM/disk limits, and execution deadlines. With the handoff’s reported headroom, a substantial live campaign is not yet demonstrably feasible.

Admission should require every applicable resource constraint to pass:

```text
accounted_usage + outstanding_reservations
+ unresolved_exposure + proposed_reservation <= authorized_limit
```

For money, maintain disjoint categories so estimated usage and retained uncertainty are not counted twice. Use fixed-precision amounts, transactional reservations, unique operation IDs, and a single authoritative state machine.

A useful lifecycle is:

```text
reserved → dispatched → settled_verified
                     → settled_estimated_with_uncertainty
                     → failed_charge_unknown
reserved → cancelled_before_dispatch
```

A provider-reported charge above the reservation must still be recorded accurately. Flag the breach and suspend further admission; never cap recorded reality to make the invariant appear satisfied.

**Reconcile historical liabilities without erasing history:**

1. Preserve the original ledger and its hash.
2. Match each request to generation/job IDs, timestamps, account, model/provider route, receipts, and any retries.
3. Fetch provider usage evidence where available.
4. Distinguish actual account charge from upstream cost, token-derived estimate, and purchased-credit fees.
5. Append a reconciliation event that supersedes the prior estimate and releases only the evidenced excess reservation.
6. Keep unmatched or ambiguous liabilities unresolved.
7. Recompute balances from events and verify that replay produces the same totals.

OpenRouter documents response usage accounting and generation-level retrieval with cost and token metadata. That makes evidence-based reconciliation plausible, but does not establish that these historical requests remain retrievable or that MLRescurix captured the required IDs. [GitHub](https://github.com/OpenRouterTeam/docs/blob/main/cookbook/administration/usage-accounting.mdx?utm_source=chatgpt.com)

No automatic expiry should turn uncertainty into zero. Missing receipts, a timeout, or “the model is usually cheap” are insufficient evidence.

Prevent future accumulation through bounded request sizes, constrained routing, current pricing snapshots, provider-ID capture, timely reconciliation, and stopping when unresolved exposure reaches a configured limit.

For Kaggle:

- Verify the specific account’s available quota and supported telemetry.
- Reserve bounded job duration before submission.
- Count running and idle accelerator allocation, retries, and overlapping sessions according to provider semantics.
- Persist remote job IDs; implement polling, cancellation, orphan detection, and confirmed termination.
- If termination or usage is uncertain, retain exposure.
- Do not mark enforcement active until demonstrated.
- Do not rotate accounts to evade quotas.

Kaggle’s documentation describes variable weekly quota and separately running sessions. A local “30 hours” counter cannot establish either available quota or actual consumption. [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage?utm_source=chatgpt.com)

**6. Scientific validation and score provenance**

First freeze the prediction contract:

- What entity is predicted, at what time, with what information available?
- Which grouping or temporal constraints apply?
- What exact metric, aggregation, weighting, and invalid-prediction policy apply?
- Which external data and pretrained models are allowed?

For SMAPE, specify its scale, denominator, zero handling, clipping, and any positivity constraints explicitly. Training loss and evaluation metric are separate decisions. Log-target fitting may help, but is not equivalent to optimizing SMAPE.

Use three evaluation layers:

| Layer | Purpose | Access |
|---|---|---|
| Search development | Frequent comparisons and diagnostics | Planner can observe results |
| Confirmation | Limited assessment of a frozen shortlist on stronger splits/seeds | Release results only after the batch is fixed |
| Final lockbox | One release assessment of the frozen selected artifact | Evaluator only |

All learned preprocessing must fit inside the training partition: imputation, scaling, vocabulary, IDF, SVD, feature selection, target encoding, and learned representations. Early stopping needs an inner training validation split rather than unrestricted use of the outer evaluation fold. Pipeline construction helps enforce this separation. [scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/common_pitfalls?utm_source=chatgpt.com)

Further requirements:

- Split by product/entity groups if near-duplicates or repeated products would cross partitions.
- Use chronological evaluation when predicting future outcomes.
- Keep fold assignments fixed for paired comparisons; changing folds repeatedly is not a cure for adaptive overfitting.
- Treat confirmation data as development data once its results guide subsequent search.
- Fit ensemble weights on development evidence; assess them on independent confirmation data or with proper nested cross-fitting. A score on the OOF predictions used to optimize weights is a selection score.
- Preserve a separate validation record for each protocol and fidelity. Scores from different splits or training sizes are not interchangeable.
- For formal evaluation of the *research procedure*, use outer held-out tasks/folds and rerun the entire selection procedure inside each training partition. Nested evaluation addresses selection bias but costs substantially more. [scikit-learn 1.9.1 documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html?utm_source=chatgpt.com)

Every evaluation receipt should bind:

```text
candidate/artifact hash
dataset and protocol hashes
ordered row IDs and prediction hash
fold training membership and preprocessing lineage
metric implementation/version
per-fold and aggregate scores
seed, fidelity, environment
evaluation tier and eligibility decision
```

The selected candidate’s report must join through its immutable evaluation ID. A global “latest verified score” should never determine a winner’s reported result.

Crucially, recomputing an OOF metric only verifies arithmetic and alignment. It does not prove those predictions were generated without label access. Execute fold training and inference in controlled stages; withheld labels stay with the evaluator.

**Holdout use:** establish that the proposed lockbox was never used for training, selection, debugging, or label-driven analysis; account for historical exposure and cross-split duplicates. Freeze the exact final artifact before evaluation. If it fails, report failure. Further tuning requires a new independent final assessment; the old lockbox is now consumed.

A successful holdout supports performance for that sampling process and frozen artifact. It does not establish resistance to future distribution shift, causal validity, fairness, operational reliability, or general autonomous competence. Confidence intervals do not remove selection bias or fix an inappropriate split.

**7. Staged implementation and acceptance criteria**

The paths below are proposed targets based on names in the handoff, not files I inspected. New names are illustrative.

| Stage | Proposed changes | Meaningful acceptance tests |
|---|---|---|
| **A. Repair contracts and accounting** | Shared `contracts/action.schema.json`; registry-driven prompt/action dispatch; typed validation in `candidate_pipeline.py`; event replay in `operational_budgets.py`; candidate-bound reporting | Research action traverses both tiers; `C`/`penalty` receive structured errors without training; valid portfolio members survive; restart/replay does not lose or double-count receipts |
| **B. Protect data and evidence** | Replace writable hard-link staging; strengthen `candidate_worker.py`; add immutable `evidence_store.py`; extend `trusted_evaluation.rs` | Mutation attempts cannot alter source data; worker cannot read withheld labels or credentials; row permutation, missing rows, duplicate IDs, NaNs, and artifact tampering are rejected |
| **C. Implement recovery** | Add `experiment_spec.py`, `experiment_store.py`, `scheduler.py`, `recovery.py`; refactor `autonomous_loop.py`; split critic hard failures from diagnostic warnings | Replayed Run 6 pattern triggers a substantive pivot after two warnings/near-duplicates; exact duplicates incur zero new training; restart retains hypothesis state and incumbent |
| **D. Expand cheap scientific breadth** | Versioned transform/model/loss plugins; configurable text pipelines; experiment comparison reports | At least four distinct mechanisms complete under a bounded campaign; changes reach estimator configuration; matched ablations establish attribution |
| **E. Connect external services** | Repair `governed_research.py`; add reconciliation adapter; complete `kaggle_pool.py` and transport lifecycle | Search success, empty result, timeout, retry, and unknown-charge cases all reconcile correctly; remote cancellation and orphan recovery demonstrated on synthetic data |
| **F. Establish release evaluation** | Extend `custom_protocol.rs`; confirmation/lockbox controls; provenance-aware `bundle.rs` verification | Frozen candidate evaluated without planner label access; report reproduces exactly from receipts; failed release criteria cannot be overridden by the planner |
| **G. Demonstrate generality** | Task adapters and held-out benchmark campaigns | Same controller operates on regression, grouped classification, and temporal prediction without dataset-name branches |

Dependency order: **A and B before paid autonomy; C before increasing search length; D before expensive architectural expansion; E before claiming external capability; F before generalization claims.**

Add fault-injection tests for crash-after-dispatch, duplicate callbacks, delayed receipts, concurrent reservations, and cancellation uncertainty. These are more valuable than tests that merely assert an action name exists.

Suggested system-level targets:

- 100% of launched jobs have validated specs, leases, and terminal or explicitly unresolved states.
- Zero silently ignored parameters.
- Zero exact duplicate retraining unless explicitly designated replication.
- Zero unexplained candidate/score mismatches.
- Identical ledger and incumbent state after replay.
- No third consecutive near-duplicate without a documented diagnostic exception.
- No lockbox access from planner or training workers.
- Every final scientific claim traceable to an evaluation receipt.

Do not equate “more distinct model names” with autonomy. Evaluate whether the agent uses evidence to select different mechanisms and handles negative results correctly.

**8. Fresh Amazon live-test plan**

Run a staged campaign, not an unrestricted launch.

**Before spending**

Verify:

1. Exact source snapshot, dependency locks, and working-tree changes.
2. Offline action/schema/recovery tests, including the untested parameter patch’s replacement.
3. Dataset provenance, row ordering, duplicates, target validity, permitted external data, and prediction-time features.
4. Reproduced baseline metric on a fixed local fixture.
5. Candidate-bound score reports and replay-safe token accounting.
6. Read-only staging and evaluator isolation.
7. Effective financial headroom and reconciliation status.
8. Real provider connectivity and charge-evidence capture.
9. Kaggle data-transfer authorization, privacy settings, quota, and cancellation behavior if remote execution is planned.
10. Finalization resources reserved inside the existing authorized envelope.

If available headroom cannot fund a coherent test, return **blocked by authorized budget**, with the estimated shortfall. Do not quietly open a new campaign to bypass the old ceiling.

**First campaign sequence**

Use a modest, preregistered set rather than spending the whole operational allowance:

1. **Reproduction:** rerun the reported incumbent configuration if its exact artifacts become available. Otherwise establish a new named baseline and explicitly avoid claiming historical replication.
2. **Text hypothesis:** word/character TF-IDF with regularized linear regression, avoiding mandatory SVD.
3. **Structured hypothesis:** reusable quantity/unit/count features with a strong GBDT.
4. **Representation hypothesis:** configurable SVD dimensionality versus sparse or frozen-embedding alternatives.
5. **Loss hypothesis:** compare raw-target and log-target training under the same primary SMAPE evaluation.
6. **Neural hypothesis:** test the existing MLP fairly; add a more complex architecture only if evidence justifies the cost.
7. **Combination hypothesis:** assess a simple blend only if candidate errors are sufficiently complementary.
8. **Confirmation:** evaluate the frozen shortlist using the stronger validation protocol.
9. **Release:** refit/freeze the selected artifact and conduct the one-time lockbox assessment if eligibility criteria are met.

Use low-fidelity screening, then promote candidates to comparable higher-fidelity evaluation. Do not eliminate slower-learning neural candidates solely from tiny training budgets; preserve a small exploration allowance.

**Capture the complete decision trace**

Record prompts and action responses with appropriate private storage; model/provider identifiers; hypotheses and citations; requested and resolved parameters; every rejection and repair; candidate/attempt IDs; dataset/protocol/code hashes; jobs and seeds; row-indexed predictions; evaluator receipts; critic reasons; incumbent transitions; resource reservations and settlements; and the final stopping rationale.

Capture concise decision rationales and evidence, not supposed access to hidden model reasoning.

**Define success on separate axes**

- **Engineering success:** the system executes valid, meaningfully diverse experiments, repairs seeded faults, avoids duplicates, survives restart, preserves data, and accounts for all external operations.
- **Scientific success:** a frozen candidate achieves a preregistered practically meaningful improvement over a matched baseline on independent assessment, with acceptable uncertainty and subgroup behavior.
- **Useful negative result:** no improvement, but reproducible evidence identifies ineffective mechanisms and justifies stopping.
- **Failure:** repeated malformed proposals, ineffective pivots, unexplained score reuse, leakage, wrong-candidate reporting, unauthorized spending, or lockbox-guided tuning.

Choose the minimum meaningful SMAPE improvement before results arrive, based on application value. There is no defensible reason here to promise that the next run will beat 62.4663. That number is only comparable under the same protocol and data.

**If the first run falls short, use its failure mode to choose the next experiment.**

| Observation | Next action |
|---|---|
| Historical score cannot be reproduced | Audit code/data/configuration/environment and cache lineage before optimization |
| Large train–validation gap but consistent validation improvement | Diagnose leakage and instability; use confirmation evidence rather than automatic veto |
| Sparse text wins | Explore lexical coverage, character features, regularization, and targeted structured features |
| Sparse and dense representations make different errors | Test a constrained blend on independent assessment |
| All model families underperform | Investigate labels, parsing, task definition, split representativeness, and unavailable prediction-time information |
| Training curves improve with more data | Increase training data within the frozen protocol |
| Neural model adds cost without reliable benefit | Retain the simpler candidate and redirect resources |
| Provider uncertainty blocks progress | Reconcile charges or pause; changing model architecture does not solve accounting |
| Holdout fails | Report the failure, retire that holdout from future final claims, and obtain a new independent assessment |

The immediate milestone should be **one auditable campaign that learns from failure and finishes honestly**. Broader architectures, more searches, and more compute should follow demonstrated control of that process.