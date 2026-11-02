My assessment is that MLRescurix needs a more reliable experimental control system before it needs a larger model zoo. Its next milestone should be a campaign that executes distinct hypotheses, diagnoses failures, preserves evidence, and stops honestly—even when it finds no improvement.

I reviewed only the supplied handoff. I did not inspect the repository, underlying artifacts, provider account, or datasets. Below, **reported evidence** means a claim or excerpt in the handoff; **assessment** means my interpretation; implementation details and thresholds are **proposals**, not verified capabilities.

1. **Prioritize scientific integrity, executable contracts, and recovery.**

   The reported runs support a narrower capability claim than “autonomous ML researcher”: MLRescurix can reportedly execute constrained portfolios, recompute development metrics, and package an accepted candidate. They do not establish reliable long-horizon research, remote execution, or generalization.

   | Priority | Reported evidence | Assessment and plausible alternatives |
   |---|---|---|
   | **P0: Evaluation validity** | Repeated selection on two development folds; no holdout; Run 6 score bound to the wrong candidate. | The system can optimize an increasingly biased development signal. Correctly recomputing predictions verifies arithmetic, not leak-free training or generalization. |
   | **P0: Data and execution isolation** | Writable hard links share the staged dataset’s inode; Python execution actions exist. | This is a credible integrity risk, not evidence of corruption. Actual risk depends on process permissions, mounts, accessible files, and hash-check timing. |
   | **P0: Resource authority and reconciliation** | Approximately $0.18604 remains uncertain against a $0.28049402 campaign ceiling. | Retaining unresolved liabilities is defensible. Missing reconciliation and poor feasibility reporting are the operational defects. Low estimated cost does not establish that reservations are safe to release. |
   | **P1: Proposal-to-execution contract** | Run 8 loses two turns to regression/classification parameter mismatches. | This is primarily an interface failure. A model should receive the actual task-specific schema, and invalid proposals should fail before training dispatch. |
   | **P1: Recovery and experiment memory** | Seven turns repeatedly revisit vetoed neural configurations. | Prompt advice is insufficient. The controller needs explicit duplicate detection, failure classification, and enforced recovery transitions. |
   | **P1: Selection policy** | Candidates around 61.73 SMAPE are rejected while an accepted candidate scores 62.47. | The gap veto may be scientifically miscalibrated. A larger training gap does not establish worse generalization when validation error is lower. |
   | **P2: Exploration breadth** | Fixed transforms, 16-dimensional text compression, mostly tree libraries plus Ridge and an MLP. | Breadth is limited, but adding transformers will not repair the higher-priority failures. Sparse text baselines and larger training samples may provide more value first. |
   | **P2: Research and remote execution** | No recorded searches or Kaggle usage; disconnected counters; missing research action dispatch. | These capabilities remain unproven. Missing dispatch cannot alone explain zero research when the provider was also unconfigured. |

   Several handoff explanations deserve direct challenge:

   - **Identical scores do not establish repeated effective training.** They could reflect duplicate configurations, ignored parameters, cached results, deterministic convergence, or broken parameter application. Compare resolved parameters, prediction hashes, cache receipts, and training logs.
   - **The train–validation gap needs an audit.** Check its formula, units, metric direction, evaluation mode, dropout handling, and train/validation population differences. A low-gap underfit model can perform poorly.
   - **Fixed enums may not be the entire extensibility limit.** `WRITE_PYTHON` and `RUN_PYTHON` reportedly exist. What remains unknown is whether generated code can enter the governed evaluation and packaging path.
   - **A persistent DAG is useful, but not sufficient.** A linear loop with durable experiment memory and good recovery can outperform a poorly governed branching planner.
   - **Run 7’s exact match to Run 6 is not automatically independent replication.** It could be a legitimate deterministic reproduction or reuse of an existing artifact. The packet does not establish which.
   - **Unused rows are not automatically an untouched holdout.** Prior access, preprocessing, duplicate products, source ordering, and previous experiments must be checked.
   - **Hashes and bitwise sample parity have limited meaning.** They support artifact integrity and sampled inference consistency; they do not certify the training procedure, billing, or production suitability.

   The most decision-changing additional evidence would be:

   | Evidence | What it would resolve |
   |---|---|
   | Frozen source revision plus working-tree diff for each run | Which fixes and capabilities actually executed |
   | Traces containing resolved configurations, cache events, prediction hashes, and evaluator receipts | Whether experiments were distinct, parameters took effect, and scores belong to selected candidates |
   | Split manifests and preprocessing fit boundaries | Whether the apparent gains may involve leakage or inappropriate validation topology |
   | Provider request IDs, usage records, and billing exports | How much uncertain liability can legitimately be reconciled |
   | Dataset provenance and prior-access records | Whether any remaining source data can support a credible lockbox |
   | Matched ablations over representations and training sizes | Whether text compression, sample size, or model capacity is the dominant modeling limit |

2. **Build an extensible experiment system around the existing authority boundary.**

   Retain Rust’s responsibility for permissions, resource reservations, data-access capabilities, evaluation receipts, and release decisions. Let Python own scientific proposals and training implementations.

   A practical execution sequence is:

   ```text
   Frozen problem contract + capability inventory
       → persistent hypothesis planner
       → schema validation and duplicate check
       → resource admission
       → isolated experiment execution
       → independent evaluation receipt
       → scientific review and recovery
       → candidate selection and release assessment
   ```

   Start with a transactional experiment registry and content-addressed artifact storage. A distributed workflow platform is unnecessary initially.

   Every campaign needs a frozen **problem contract** describing:

   - Task type, target semantics, prediction-time inputs, metric definition and direction.
   - Grouping or temporal structure and allowed validation topology.
   - Data permissions, approved external services, and external-data restrictions.
   - Resource limits, deployment constraints, and stopping rules.
   - Confirmation and holdout access policy.

   Every hypothesis should state a mechanism and a falsifiable comparison. For example:

   > “Low-rank compression may discard product identifiers and quantities. Compare sparse word/character text features plus Ridge against the compressed-text baseline on the same folds.”

   That is a research hypothesis. “Try a different learning rate” is a local tuning action.

   Use a versioned plugin contract rather than adding another large enum:

   ```text
   PluginManifest:
     plugin_id, version, source_digest, dependency_lock_digest
     supported_tasks, accepted_input_types, parameter_schema
     resource_requirements, network_permissions, serialization_format

   Plugin:
     validate(spec, task_metadata) -> ValidationResult
     fit(training_partition, inner_validation, resolved_spec) -> ModelArtifact
     predict(model_artifact, features) -> Predictions
     export(model_artifact) -> InferenceBundle
   ```

   A recipe composes preprocessing, representation, estimator or architecture, target transform, objective, and postprocessing. The registry should expose compatible combinations to the planner.

   **Parameter handling should be strict and helpful.** I would replace the reported `C → 1/C` Ridge patch and silent `penalty` discard. Ridge and logistic regression use different objectives and parameter contracts; an inverse substitution is not a general semantic conversion. Return `unsupported_parameter`, identify the actual estimator, and provide its valid schema. Any supported migration must be explicit, versioned, and recorded. [Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [LogisticRegression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

   Validate each portfolio member separately so one invalid configuration does not discard valid siblings.

   Use the following exploration order, adapted to the task:

   | Axis | First comparisons | Later options when justified |
   |---|---|---|
   | Preprocessing | Missing-value indicators, appropriate imputation, categorical handling, scaling for scale-sensitive models, duplicate handling | Cross-fitted target encoding, learned numerical representations |
   | Text | Word and character TF-IDF with sparse linear models; ablate compression; compare several feasible SVD ranks | Frozen pretrained embeddings, subword CNNs, transformer fine-tuning |
   | Model families | Dummy baseline, regularized linear model, one strong GBDT, random forest or ExtraTrees, current MLP | Additional families motivated by residual behavior |
   | Neural architectures | Validate the existing MLP/ResNet and its training procedure | FT-Transformer, TabNet, text–tabular fusion, task-specific architectures |
   | Losses and target transforms | Standard task loss; robust alternatives; log-target modeling where valid | Quantile objectives, carefully tested metric surrogates, uncertainty-aware objectives |
   | Ensembles | Fixed simple averages of complementary candidates | Constrained weights and properly cross-fitted stacking |

   Four GBDT libraries are not four strongly orthogonal hypotheses. Representation changes, loss changes, data-size experiments, and genuinely different inductive biases often provide more useful diversity. Published tabular benchmarks also support strong simple neural baselines and find no universally superior model family. [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959).

   Begin with controlled comparisons that change one major factor. Combine promising factors afterward and test interactions. Avoid searching the full Cartesian product.

   As an initial scheduling policy, allocate roughly 50% of discretionary exploration to promising branches, 30% to underexplored axes, and 20% to replication, diagnostics, and ablations. These are configurable allocations, not spending authority or mandatory quotas.

   **Reproducibility and isolation must travel with every plugin:**

   - Freeze the source revision **and dirty diff**, dependency lock, pretrained-weight revision, seeds, resolved defaults, precision, hardware, and thread settings.
   - Mount immutable inputs read-only under a separate worker identity. Use independent copies or verified copy-on-write clones for writable scratch.
   - Give workers no ledger access, provider credentials, holdout labels, or unrestricted network.
   - Separate fold workspaces. A generated program must not recover one fold’s validation labels from another fold’s training artifacts.
   - Install approved dependencies in a separate build stage; run training without package-install privileges.
   - Load untrusted models and serializers inside the sandbox. The privileged evaluator should consume validated prediction arrays, not execute arbitrary artifact code.

   Maintain distinct identities:

   ```text
   recipe_id    = hash(data snapshot, contract, resolved pipeline,
                       parameters, source, dependencies, ensemble dependencies)

   execution_id = hash(recipe_id, fold, seed, fidelity, runtime,
                       declared replication identity)

   evaluation_id = hash(execution_id, prediction artifact, evaluator,
                        metric contract, evaluated row identities)
   ```

   A new seed is a replication, not a new scientific hypothesis. GPU reproducibility also needs declared tolerances and supported environments; PyTorch does not promise identical results across releases and platforms. [PyTorch reproducibility guidance](https://docs.pytorch.org/docs/stable/notes/randomness.html).

3. **Make recovery a controller responsibility.**

   Give every attempt a typed outcome: invalid specification, duplicate, execution failure, resource interruption, integrity rejection, evaluated result, or inconclusive result. Record the proposed configuration and the resolved configuration separately.

   Suggested initial recovery rules are:

   | Trigger | Required response |
   |---|---|
   | Invalid parameter or incompatible component | Reject before training; return the relevant schema; permit one repair |
   | Same invalid pattern twice | Suspend free-form configuration of that component and use its validated configurator |
   | Exact duplicate | Return the existing receipt without retraining |
   | Near-duplicate after repeated non-improvement | Require an explicit replication or diagnostic rationale; otherwise choose another axis |
   | Two consecutive scientific warnings in one branch | Inspect learning curves, residuals, split behavior, and effective configuration |
   | Three evaluated, non-improving local variants | Leave that local neighborhood; research or explore an underrepresented hypothesis |
   | NaNs or optimizer divergence | Check inputs and loss domains; make one bounded stabilization attempt |
   | OOM or timeout | Reduce fidelity, batch size, or model size; distinguish resource failure from poor predictive quality |
   | Budget denial | Reconcile, choose a cheaper admitted action, or checkpoint and stop |
   | Leakage, unauthorized access, or invalid provenance | Quarantine the result and repair the cause before further promotion |

   The counts are defaults to test. They should not be confused with statistical guarantees.

   Separate **hard integrity vetoes** from **scientific warnings**. Leakage, changed data, mismatched rows, and unauthorized execution are hard failures. A training gap is usually diagnostic evidence. Keep a candidate with a concerning gap in the research archive; require stronger confirmation rather than automatically declaring it inferior.

   Research when it can answer a specific decision:

   - A capability or implementation question blocks an experiment.
   - Observed failures contradict the current hypothesis.
   - Learning curves suggest an unfamiliar optimization problem.
   - A promising modeling axis has no validated implementation.
   - A plateau remains after distinct, valid experiments.

   Prefer local schemas, logs, and pinned documentation for simple API errors. Searching the web should not become a substitute for examining execution evidence.

   Every hypothesis record should contain its mechanism, parent comparison, expected outcome, failure criterion, experiment receipts, result, limitations, and next decision. “No improvement under these conditions” is a useful outcome. An OOM is not evidence that an architecture cannot work.

   Keep durable structured memory across restarts. An eight-turn context window can summarize current activity, but it should not be the experiment database. Persist rejected ideas, failure causes, coverage gaps, and the reason each branch was abandoned.

4. **Treat web research as a restricted evidence pipeline.**

   The research service should receive a sanitized task abstraction, not arbitrary planner text containing dataset material:

   ```text
   ResearchIntent
     → approved public task description
     → query admission and reservation
     → provider request
     → restricted retrieval
     → cited evidence extraction
     → testable hypothesis
   ```

   For example, “methods for combining short text and tabular features in positive-valued regression” may be sufficient. Private rows, proprietary column names, customer identifiers, filenames, rare values, and distinctive statistics generally are not needed.

   Privacy controls should include:

   - A deterministic allowlist of information permitted in outbound research.
   - Separation between the component that can inspect private data and the component that constructs public queries.
   - Local screening for secrets, identifying strings, and encoded content.
   - Explicit policy for third-party processing and retention.
   - Protected internal query logs; public reports should not reproduce sensitive rejected queries.

   Regex redaction alone cannot establish privacy. The LLM gateway needs equivalent controls because research is only one possible egress route.

   Retrieved pages, PDFs, snippets, code, and model cards remain untrusted. Use a fetcher with no credentials or private mounts; block private-network destinations, recheck redirects, and limit content size and type. Do not execute page JavaScript or downloaded code.

   Evidence extraction should return structured claims with source URLs, retrieval times, content hashes, relevant passages, and uncertainty. It should have no tools that train models, alter policy, spend money, or access datasets. Proposed actions must still pass the normal admission boundary.

   A 2 KB summary is a context-management feature, not an injection defense. Filters and instruction separation can reduce risk, but permissions must be enforced outside the model. [OWASP prompt-injection guidance](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html).

   Count requests and charges separately:

   | Outcome | Operational accounting | Financial accounting |
   |---|---|---|
   | Blocked locally | Record rejection; no dispatched query | No provider exposure |
   | Successful request | Count the dispatched attempt and success | Settle from provider evidence |
   | HTTP error | Count the dispatched attempt and failure | Record verified charge, verified zero, or uncertainty |
   | Timeout after dispatch | Count attempt; outcome unknown | Retain uncertain exposure |
   | Retry | Count another physical request | Reserve its possible additional liability |
   | Cache hit | Record logical research use and cache provenance | No new provider charge |

   Track logical questions, physical requests, fetched pages, and provider billing units separately. One “query” may consume several provider credits. Report successful, failed, pending, and uncertain requests explicitly.

5. **Unify resource admission while preserving financial authority and history.**

   Represent policy as a resource vector:

   ```text
   model tokens
   campaign USD / run USD / operation USD
   search requests / provider credits
   Kaggle quota units / accelerator session time
   local CPU time / RAM / disk
   campaign wall time / concurrency
   ```

   The proposed 5 million tokens, 30 accelerator hours, and 500 searches can be configurable operational ceilings. They do not override existing dollar limits, actual provider quota, or data permissions.

   Before starting a campaign, calculate a feasible envelope using current price evidence, context sizes, bounded outputs, planned repetitions, available quota, and uncertainty already committed. Report the binding constraint and estimated useful work it permits.

   At the handoff’s reported values:

   ```text
   $0.280494020 ceiling
   − $0.084074020 recorded billed amount, potentially estimated
   − $0.186040000 uncertain liability
   = $0.010380000 headroom
   ```

   That is not evidence of enough funding for a substantial new campaign. It may permit a small action, depending on its reservation requirements. The correct result of preflight may be **financially infeasible under current authority**.

   Use an authority-owned lifecycle:

   ```text
   PLANNED → RESERVED → DISPATCHED
                         ├─ VERIFIED_SETTLEMENT
                         ├─ VERIFIED_NO_CHARGE
                         └─ UNCERTAIN → RECONCILED
   ```

   Admission must be atomic across concurrent workers. For each operation, committed exposure consists of settled cost plus the remaining possible liability; do not double-count a reservation after transferring it into an uncertain state. The campaign must remain within its authorized ceiling at dispatch.

   Reserve against bounded input/output sizes, approved route prices, request options, and runtime limits. Estimated average cost is useful for planning, but is not necessarily an enforceable upper bound.

   **Reconcile existing entries through evidence:**

   1. Preserve the original ledger and its cryptographic history.
   2. Match entries to local request IDs, provider generation IDs, timestamps, model routes, and token receipts.
   3. Retrieve provider usage/cost records and, where available, account billing exports.
   4. Append reconciliation events identifying the evidence, matched liability, revised amount, and released reservation.
   5. Leave unmatched or ambiguous amounts unresolved.

   OpenRouter documents a generation metadata endpoint with request identifiers and cost fields, making it a useful reconciliation source. Whether a record is sufficient for final settlement should be defined by the authority’s evidence policy; local token estimates are weaker evidence. [OpenRouter generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation).

   Do not release uncertainty merely because an entry is old, the response failed, or estimated usage was small. Likewise, do not create a fresh campaign to escape existing liabilities. If adequate evidence is unavailable, the liability remains unresolved unless an explicitly authorized accounting decision changes its treatment.

   Recovery must handle a crash after dispatch but before receipt persistence. Exactly-once local state transitions are achievable; exactly-once remote requests may not be. Use provider idempotency where supported and reconcile ambiguous requests before blindly retrying.

   For Kaggle:

   - Verify the authorized account’s current quota and accelerator availability.
   - Define whether the internal budget means session-hours, device-hours, or provider quota units; record actual hardware and conversion.
   - Reserve a maximum runtime, maintain leases and heartbeats, and verify termination.
   - Include session startup and idle time where the provider consumes quota.
   - Keep unknown remote state reserved until resolved.
   - Use approved accounts without circumventing provider limits.

   Kaggle documents variable quota and account-side usage monitoring, so a configured “30 hours” cannot establish availability. [Kaggle GPU usage guidance](https://www.kaggle.com/docs/efficient-gpu-usage).

   Verify dataset and notebook privacy independently before upload. Private hosting still involves third-party processing. Use explicit metadata and read-back checks rather than relying on defaults. [Kaggle kernel metadata](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md).

6. **Separate exploration, confirmation, and final assessment.**

   Reusing fixed development folds is useful for paired comparisons, but repeated adaptation makes the selected score optimistic. More folds alone do not eliminate this effect. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).

   Use three evaluation levels:

   | Level | Purpose | Access |
   |---|---|---|
   | Exploration | Feature development, tuning, diagnostics | Planner receives development feedback |
   | Confirmation | Compare a small, frozen shortlist under a prespecified protocol | Limited assessments; every use recorded |
   | Lockbox | Assess one frozen final candidate against a prespecified comparator and release rule | Independent evaluator; no search or tuning feedback |

   Choose split topology from the deployment question. New-product performance may require product-group separation; future-price prediction may require chronological splits. Random row splitting is appropriate only when its independence assumptions are credible.

   The old 75,000 rows should remain development data. Audit the remaining source pool before assigning any portion to confirmation or lockbox use. Check provenance, historical usage, duplicates, product groups, and ordering. If untouched data cannot be established, obtain a new evaluation source rather than relabeling exposed data.

   Within every training fold, fit imputers, vectorizers, SVD, scalers, supervised encoders, and feature selection using training data only. Early stopping, calibration, and tuning need inner training partitions. A frozen external per-row embedding can be cached differently from a corpus-fitted transformation; record that distinction. [Scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html).

   For more rigorous assessment of the **research procedure**, use nested evaluation: run a frozen search procedure inside each outer training partition and evaluate its chosen result once on the outer test partition. Merely evaluating an already globally tuned configuration across outer folds does not assess the full adaptive search. This is more expensive, so introduce it for finalist procedures and cross-dataset benchmarking. [Nested cross-validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

   Every score receipt should bind:

   ```text
   candidate and execution identity
   training population and split identity
   evaluated row identity/order
   prediction artifact hash
   metric implementation, units, direction, and aggregation
   target transformation and prediction postprocessing
   evaluator revision
   result status, score, uncertainty, and evaluation level
   ```

   Validate missing rows, duplicate predictions, nonfinite values, and target-domain violations before scoring. The independent evaluator must recompute the metric, but execution isolation must establish which labels the model could access.

   For SMAPE, freeze the exact formula, scale, zero-denominator treatment, and any prediction clipping. Do not assume the reported number fully specifies the metric.

   Select candidates by ID and evaluation level. Generate reports, scorecards, and bundles from that same selection record. An out-of-order completion must never replace the selected candidate’s score.

   For ensembles, fitting blend weights on OOF predictions and reporting performance on those same rows is still selection. Assess the ensemble through a further validation layer, proper meta-level cross-fitting, or the untouched confirmation stage.

   Use paired uncertainty estimates appropriate to independent units—product groups or temporal blocks where needed. Fold means are correlated, and confidence intervals do not erase adaptive selection bias. Predeclare a practically meaningful improvement threshold instead of promoting every floating-point gain.

   Before opening the lockbox, freeze the artifact, preprocessing, postprocessing, comparator, metric, acceptance rule, and data-access policy. Do not use the lockbox for early stopping, ensemble selection, clipping thresholds, or repeated release attempts. A failed assessment consumes its status as untouched feedback.

   A successful holdout supports performance for that frozen candidate on the evaluated population, with sampling uncertainty. It does not prove robustness to future distribution shift, all datasets, rare subgroups, production failures, or training reproducibility. A model retrained afterward is a different artifact and must be described accordingly.

7. **Implement in gated stages with meaningful acceptance criteria.**

   All paths below are proposed repository-relative changes based on names in the packet. I have not inspected or modified them.

   | Stage | Proposed components | Meaningful tests and acceptance criteria |
   |---|---|---|
   | **A. Repair current execution** | `autonomous_loop.py`, `candidate_pipeline.py`, `operational_budgets.py`, `governed_research.py`, gateway action definitions, run script | Generate action definitions from one schema. Invalid parameters never reach fitting. A bad portfolio member does not abort valid siblings. Resume reproduces token totals and selected-candidate identity. Research traverses the complete mock path. |
   | **B. Protect evidence and resources** | Rust resource authority and reconciliation adapter; input staging; `trusted_evaluation.rs`; selection/report export | Attempt in-place input mutation and cross-fold label access. Inject crashes before/after dispatch and settlement. Concurrent reservations never exceed limits. Reconciliation preserves original entries. Out-of-order results cannot misattribute scores. |
   | **C. Add durable experimental control** | `experiment_contract.py`, `experiment_registry.py`, `recovery_policy.py`, `experiment_scheduler.py` | Replay duplicate, invalid-parameter, veto, timeout, and budget-denial scenarios. Exact duplicates consume no training compute. Repeated failures trigger bounded repair or a pivot within two planner decisions. Resume requires no manual state reconstruction. |
   | **D. Broaden low-cost science** | `plugins/preprocessing/`, `plugins/text/`, `plugins/estimators/`, `plugins/ensembles/` | Execute sparse text, engineered-feature GBDT, and validated MLP comparisons on identical protocols. Include ablations. Every result has resolved configuration and controller-bound score provenance. |
   | **E. Establish validation separation** | `protocol_builder.py`, confirmation service, isolated lockbox service, updated bundle scorecard | Deliberately leaky features and label-access attempts are blocked or detected. Planner cannot access lockbox labels. A frozen candidate and baseline complete one assessment with an uncertainty report. |
   | **F. Integrate research and Kaggle** | Research admission/fetch/extraction modules; `kaggle_pool.py`, `kaggle_output_transport.py` | A small real search has a provider receipt. Failed and uncertain requests remain visible. Injection fixtures cannot invoke privileged actions. A synthetic private remote job verifies actual hardware, artifact retrieval, cancellation, and quota accounting. |
   | **G. Add custom neural plugins and generality tests** | `plugins/neural/`, isolated plugin build process, benchmark suite | A new architecture runs through the existing contract without editing core orchestration. The same controller handles additional task types without Amazon-specific branching. |

   Suggested dependency order:

   ```text
   A → B → C → D
       B → E
       B + C → F
       C + D + E → G
   ```

   Some work can proceed concurrently, but custom generated models should wait for isolation, identity, and evaluation contracts.

   Keep these tradeoffs explicit:

   - Stronger isolation increases setup time but protects both data and score validity.
   - Nested evaluation reduces exploratory throughput; use it where the evidentiary value justifies the cost.
   - Deterministic settings can reduce performance; report their cost and supported scope.
   - Strict schemas constrain proposals but make execution dependable. Extensibility belongs in registered plugins.
   - A persistent registry adds state-management work but prevents repeated spending and lost failure knowledge.

   Do not make “better Amazon score” the sole architectural acceptance test. Include a separate benchmark set with classification, grouped regression, temporal prediction, and another modality as support matures. Report failures and baseline-relative results across all tasks. Reserve some datasets from framework tuning to reduce meta-overfitting.

8. **Run a fresh Amazon campaign in two parts: operational proof, then scientific assessment.**

   A fresh run should start from a frozen implementation snapshot and a new run identity while retaining the existing campaign’s financial obligations.

   **Before spending:**

   - Establish exact code, diff, dependencies, model route, available plugins, and capability status.
   - Complete the offline gates above, including parameter rejection, recovery, restart, and score-selection tests.
   - Reconcile historical entries where evidence permits; compute remaining headroom without clearing unresolved liabilities.
   - Produce an admitted resource envelope that includes final evaluation and packaging, not just exploration.
   - Verify source-data provenance, permissions, hashes, duplicates, target semantics, and split topology.
   - Freeze the evaluation contract and keep legacy-development comparisons separate from new-protocol results.
   - Verify search connectivity with a minimal approved query before claiming research capability.
   - Verify Kaggle on synthetic data before uploading Amazon data, if remote execution is needed.
   - Remove the inherited four-turn setting as the default campaign stop, while retaining wall-time, budget, and no-progress watchdogs.

   If current authority cannot fund a meaningful campaign, report that outcome. Do not treat an intentionally tiny smoke test as an autonomy demonstration.

   **Part A: prove operational behavior.**

   Use development data only. Demonstrate a valid training action, explicit invalid-parameter feedback, duplicate reuse, a recorded research request, checkpoint/resume, and candidate-bound reporting. Recovery edge cases should mostly be injected offline; there is little value in paying to rediscover known failures.

   Test Kaggle separately unless it is necessary for the scientific plan. An unused GPU is not a failure when CPU experiments are more appropriate.

   **Part B: execute a hypothesis-driven search.**

   A reasonable initial Amazon sequence is:

   1. Establish a dummy baseline and a reproducible current-pipeline reference.
   2. Compare direct sparse word/character text plus Ridge against SVD compression.
   3. Test structured text extraction—such as quantities or units—only where supported by the data.
   4. Compare a strong GBDT using suitable structured features and text representations.
   5. Reproduce the current MLP with verified parameter application and inner early stopping.
   6. Test an appropriate target transform or robust loss while keeping evaluation on the frozen primary metric.
   7. Run a learning-curve comparison using additional eligible training data.
   8. Test frozen embeddings or a different neural architecture if earlier evidence justifies the cost.
   9. Consider a simple ensemble only when candidate residuals show useful complementarity.

   This is a starting portfolio, not a mandatory script. Each result should determine the next action through the recorded hypothesis and recovery policy. Neural complexity should earn its compute allocation.

   **Capture a trace that distinguishes intention from execution:**

   - Campaign policy, feasible budget envelope, and stop conditions.
   - Hypothesis, expected mechanism, comparison, and falsification criterion.
   - Proposed and resolved configurations, recipe IDs, duplicates, and cache hits.
   - Source, environment, data, split, model, and prediction hashes.
   - Training curves, resource failures, and effective training settings.
   - Provider request IDs, usage receipts, reservations, uncertainties, and reconciliations.
   - Sanitized research queries, sources, extracted claims, and resulting experiments.
   - Independent evaluation receipts and selection decisions.
   - Restart events, recovery transitions, and final reason for stopping.

   Record concise decision rationales and evidence references; hidden chain-of-thought is unnecessary. Store sensitive operational traces under appropriate access controls.

   Use separate success criteria:

   | Dimension | Success | Failure or inconclusive result |
   |---|---|---|
   | Operational reliability | Every dispatched operation has a terminal or explicitly uncertain state; resume and accounting remain consistent | Unexplained requests, lost usage, changed inputs, or score misattribution |
   | Autonomy | Several materially distinct hypotheses execute; failures cause bounded repair or a justified pivot | Repeated invalid proposals or near-duplicates without diagnostic progress |
   | Scientific learning | Controlled comparisons support clear conclusions, including negative findings | Incomparable splits, ineffective parameter changes, or uninterpretable failures |
   | Predictive value | Frozen candidate exceeds a prespecified baseline by a practically meaningful margin with appropriate uncertainty | Improvement is insufficient, uncertain, or absent on confirmation/holdout |
   | Delivery | Selected artifact, inference behavior, and reported scores agree | Packaging passes but model/evaluation provenance does not |

   Provisionally, require at least three materially different hypotheses and one useful diagnostic or ablation in the first adequately funded campaign. That demonstrates a minimum exploration behavior; it does not establish broad researcher capability.

   Decide the next experiment from the failure:

   - **Sparse text wins:** investigate compression loss and complementary structured features.
   - **More training data helps:** prioritize eligible data expansion before architectural complexity.
   - **Better development scores disappear on confirmation:** examine selection bias, duplication, and topology.
   - **Training improves while validation worsens:** test capacity, regularization, loss, and shift hypotheses.
   - **Everything underfits:** investigate feature sufficiency, target semantics, and optimization.
   - **Infrastructure fails:** repair and reproduce the affected comparison before making scientific claims.
   - **The holdout falls short:** record the failure and retire it as untouched feedback; use new evaluation evidence for a later release attempt.

   The strongest outcome from the next run would be a trustworthy, reproducible campaign that learns something decisive. A lower development SMAPE without that evidence would remain a limited result.