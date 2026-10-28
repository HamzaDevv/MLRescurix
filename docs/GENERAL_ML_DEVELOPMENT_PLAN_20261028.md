# MLRescurix: General ML Development Plan

Date: 28 October 2026. Planning document; does not change runtime policy or authorize paid calls, uploads, submissions, or benchmark evaluation.

**Scope update:** Read [ARCHITECTURE_BUILD_HANDOFF.md](ARCHITECTURE_BUILD_HANDOFF.md) for the user's subsequent clarification and current implementation order. The cart transformer is illustrative only; no recommendation-first requirement applies. The user will select five benchmark problems later. Build shared custom-model, memory, offloading, checkpoint/recovery, tuning and delivery infrastructure first. The domain ordering and recommendation example below are historical proposals, superseded where they conflict with that handoff.

## 1. Product direction

Build an evidence-driven ML engineering system that can formulate a task, research approaches, implement and evaluate candidates, improve them under resource constraints, and deliver a reproducible model. Kaggle is a demanding evaluation environment and a compute backend; it should not define the core product.

The intended interface is: **data + objective + prediction context + constraints → validated model + reproducible training/inference + evidence + recorded limitations**.

Two operating profiles share the same engine:

- **Competition:** exact competition metric, rules and external-data policy, submission format, inference limits, development folds, and leaderboard records.
- **Application:** business acceptance criteria, representative evaluation data, latency/throughput/cost requirements, deployment environment, monitoring, and retraining policy.

Success on difficult competitions is valuable evidence. It does not logically establish success on every easier task: a competition does not test production reliability, all modalities, changing business objectives, or every data distribution. Test ordinary application tasks explicitly alongside competitions.

“Any Kaggle problem” is a long-term extensibility objective, not a credible first-release guarantee. Unsupported tasks should produce a precise capability gap and an implementation proposal rather than silently using a tabular substitute.

## 2. What the five documents contribute

| Input | Retain | Correct or qualify |
| --- | --- | --- |
| Conversation journey | LLM as scientist; deterministic workspace; resource awareness; iteration history; selectable handoff | Historical test counts and implementation assertions are not current certification |
| Apex architecture | Primary/fallback providers; Kaggle integration; small recommendation model example | Account roles do not establish platform permission; latency and model-quality examples are hypotheses |
| Transformer guide | Transfer learning, configurable pooling/optimization, distillation, batching | Recipes are experiments, not universal defaults; example code requires review and numerical tests |
| Kaggle playbook | Early baseline plus validation, leakage checks, domain research, fast iteration, OOF ensembles | No universally best model family, mandatory ensemble size, fixed clipping range, or always-correct CV scheme |
| Gemini addendum | Persistent prediction artifacts, fold-specific preprocessing, compression candidates, deployment profiles | It did not inspect this repository; its proposed paths and claims about absent functionality are not implementation facts |

The plan below extends existing components rather than introducing the addendum's parallel `src/validation`, `src/apex`, and `src/workers` architecture.

## 3. Code-grounded starting point

Inspected the current working tree, including pre-existing uncommitted changes. Existing edits were left intact.

| Area | Observed state | Implication |
| --- | --- | --- |
| Controller | `scientific/autonomous_loop.py` has an adaptive action loop, diagnostics, script execution, iteration recording, Apex history, explicit iteration selection, and handoff | Extend and harden this controller |
| Task coverage | `task_adapters.py` and `agent_cli.py` expose classification, regression, and panel forecasting | Ranking, recommendation, neural text/vision, and other modalities need real adapters and delivery paths |
| Validation | Locked folds, temporal/group checks, fold-local preprocessing, and critic receipts already exist | Add missing audits and generalize contracts; do not rebuild validation from zero |
| Models | `model_zoo.py` supports native optional GBDTs, linear models, neural tabular models, and explicit fallback metadata | Record actual executed family; native optional packages are not guaranteed by core dependencies |
| OOF | `ModelOofResult` includes predictions, probabilities, fold metrics, and fitted estimators | Persist and bind these artifacts consistently across controller iterations; OOF support is not absent |
| Embeddings | `frozen_embeddings.py` includes deterministic synthetic encoders initialized from random weights | Useful interface fixtures; not evidence that pretrained BGE/CLIP embedding extraction is implemented |
| Execution | Execution service and Rust resource/sandbox machinery exist | Integrate remote execution under the same lifecycle and evidence contracts |
| Providers | Governed live dispatch requires `RustGatewayAdapter`; the new `dual_gateway.py` is a separate client with static prices and broad fallback | Implement provider expansion inside the governed path rather than plugging in a bypass |
| Search | New `deepseek_search.py` has real endpoint URLs but sends `max_results` instead of documented `count`, can use an OpenRouter key against DeepSeek, and collapses failures into empty results | Repair schema, credential separation, explicit error states, billing, and integration |
| Kaggle | Existing runner plus new pool scaffold; pool stores raw API keys in SQLite, assumes local quota counters, inherits the parent environment, and lacks a unified governed worker lifecycle | Treat as an integration prototype, not a verified compute service or compliance mechanism |
| Handoff | Bundle verification accepts joblib or safetensors; existing pipeline/ensemble code and parity checks exist | Audit the actual model/transform graph before deciding whether a breaking v3 format is necessary |

The README records historical Python/Rust test results and explicitly distinguishes offline wiring from model quality and production certification. Those historical counts are not fresh results of this review.

### Immediate correctness issues worth fixing first

1. **Apex eligibility:** `select_apex_iteration()` checks that a spec exists, but does not itself require an approved audit and finite score. Exploration history may include rejected iterations; deployable selection needs an explicit eligibility gate.
2. **Selected versus best:** selecting an older iteration overwrites `best_cv_score`; new evaluations also update accepted state. Separate best observed, best eligible, explicitly pinned, and deployment-selected identities. Exercise pin → continue search → resume → refit → export.
3. **Metric consistency:** `_record_iteration()` prefers critic fold scores and critic train score while using the primary `cv_score`. `candidate_pipeline.py` may use a different secondary critic metric. Compute fold dispersion and train/validation gaps within the same metric, and record secondary diagnostics separately.
4. **Failover accounting:** static token rates are estimates, yet `dual_gateway.py` marks their result `cost_is_estimate=False`. Reservations must cover allowed attempts and actual routes; ambiguous paid timeouts must remain accounted for.
5. **Policy integration:** current historical campaign policy pins a route and disallows fallback. The desired general provider policy needs an explicit, versioned policy change; preserve old campaign restrictions and receipts.

## 4. Verified service constraints

### DeepSeek and OpenRouter

DeepSeek currently documents Search and Fetch endpoints using the same DeepSeek bearer key. Search uses `count`, with documented limits; unknown fields are ignored. Its guide lists request-based billing, so search/fetch require budget accounting in addition to LLM tokens. See [DeepSeek Search & Fetch](https://api-docs.deepseek.com/guides/search/).

Current direct chat documentation lists `deepseek-flash` and `deepseek-v4-pro`; do not rely on the scaffold's old `deepseek-chat` / `deepseek-reasoner` mappings. Resolve configured model IDs and supported parameters against current provider documentation/catalogs. See [DeepSeek chat reference](https://api-docs.deepseek.com/api/create-chat-completion/).

OpenRouter supports routing and usage reporting. Preserve the requested model, served model, provider, request identity, usage, and reported cost when available; distinguish a local estimate from provider-reported usage and reconciled billing. See [OpenRouter support documentation](https://openrouter.ai/support).

Proposed allocation:

- OpenRouter: primary reasoning/coding, with model selection based on measured task performance and budget.
- DeepSeek: research search/fetch; direct inference as an explicitly configured fallback or explicitly selected provider.
- Deterministic tools: data profiling, fold checks, metric computation, training, artifact inspection, and packaging.

Neither service's credential is valid for the other. Provider failover must not imply permission to transmit data to a new destination.

### Two Kaggle API credentials

Determine account identities before scheduling. Two credentials for one account share that account's quota. Credentials belonging to different people do not automatically grant permission to pool compute, share competition artifacts, or act on their behalf.

For competitions, check each competition's rules and team/data-sharing permissions. Published Kaggle rules include one account per participant and restrictions on private sharing outside teams. An account that does not submit can still participate through training or sharing; “compute-only worker” is not a general exemption. See [Kaggle example competition rules](https://www.kaggle.com/c/connectx/rules).

Use explicit owner authorization and permitted collaboration. Start with one account; enable the second for eligible work only after ownership, collaboration, dataset access, and platform permissions are established. Do not build random naming, encryption, or staggered requests as mechanisms for concealing account relationships. Backoff is for reliable API use.

The official CLI documents JSON quota output. Probe installed capability and parse its actual schema; the current pool does not implement that telemetry. Actual quota, concurrency, session limits, accelerator availability, and internet permissions must be discovered rather than inferred from `N × 50 hours`. See [official CLI output documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/output_format.md) and [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage).

Two 16 GB GPUs do not become a single 32 GB device under ordinary data parallelism: model replicas still need to fit per device. Use measured feasibility and explicit sharding only where supported. See [PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

## 5. Shared architecture to build toward

```text
Problem contract + data manifest + resource/privacy policy
                         |
              Validation and baseline
                         |
      Research → hypothesis → implementation/spec
                         |
          Governed execution backend interface
               /                    \
       Local sandbox           Kaggle batch job
               \                    /
       Evaluation receipts + prediction artifacts
                         |
        Critic → experiment history → selection
                         |
        Refit / compress / package / verify
                         |
    Competition submission OR application deployment
```

Keep Rust responsible for execution authority, admission, accounting, and durable control state; keep Python responsible for scientific computation. The LLM proposes work and interprets evidence; tools enforce measurable contracts.

### Contracts to extend

- **ProblemSpec:** task/modality, target/labels, prediction unit and time, available-at-inference fields, group/entity keys, horizon, metric definition/direction, acceptance requirements, and allowed data sources.
- **ResourcePolicy:** wall time, RAM/VRAM, storage, concurrency, provider/search spend, accelerator quota, inference batch size, latency/throughput, network and privacy permissions.
- **ValidationSpec:** row identity, dataset fingerprint, exact folds/origins, group/temporal constraints, preprocessing fit boundaries, untouched evaluation set, and test-covariate access policy.
- **CandidateSpec:** built-in recipe or sandboxed custom training entry point; dependency environment; preprocessing/model/postprocessing graph; seeds; checkpoint/resume interface.
- **IterationRecord:** parent candidate(s), hypothesis, code/environment/data/fold hashes, primary and secondary metrics, OOF locations, resource measurements, failures, audit status, and artifact identities.
- **SelectionRecord:** chosen iteration, choice reason, user pin, quality/latency profile, validation eligibility, and immutable artifact binding.

Do not force every task into a giant universal transformer DSL. Support tested recipes for common tasks plus a sandboxed custom candidate interface for unfamiliar architectures, objectives, and competitions. Custom candidates must obey the same validation, resources, and delivery contracts.

## 6. Phased implementation backlog and acceptance gates

### Phase 0 — Establish a trustworthy baseline (P0)

- Inventory source, tests, fixtures, native dependencies, integration gaps, and existing uncommitted changes.
- Repair Apex eligibility, primary/secondary metric mixing, selection/pinning state, and history/resume integrity.
- Create capability statuses: implemented, fixture-only, optional dependency missing, integrated, measured on real data.
- Keep historical campaign ledger, frozen folds, contamination disclosures, and unresolved thresholds intact.
- Reconcile existing K1/K2 missing quality requirements, K4 provenance gaps, K5 full-scale feasibility, and image/runtime dependency gaps as separate tracked work. Do not let those old benchmark contracts define all new application tasks.

**Gate:** offline regressions demonstrate approved-only deployment selection, stable user pins, correct minimize/maximize metrics, and reproducible resume/export. Unsupported capabilities fail explicitly.

### Phase 1 — Governed providers and durable Kaggle execution (P0)

- Add primary/fallback provider adapters through the existing governed gateway, with versioned model/route policies, current price ceilings, bounded retries, schema validation, circuit breaking, and per-attempt receipts.
- Fallback only for classified transient failures and permitted destinations; configuration/auth/schema errors must be surfaced. Reconcile uncertain outcomes rather than blindly replaying them.
- Repair DeepSeek Search/Fetch request and response handling; use only DeepSeek credentials; cache by query/source version; retain citations, fetch dates, and full source artifacts behind bounded summaries.
- Distinguish unavailable research from a successful search with no matches. Treat fetched code/text as untrusted evidence, not executable instructions.
- Refactor Kaggle runner/pool into a backend used by `ExecutionService`: preflight → stage → dispatch → reconcile → collect → verify → cancel/finish.
- Add quota snapshots, atomic worker leases, durable job IDs, idempotent retry/recovery, structured status parsing, output hashes, checkpoint budgets, artifact size limits, and redacted logs.
- Use credential references instead of raw secrets in the pool database. Pass only the selected Kaggle credentials and required environment; never upload OpenRouter/DeepSeek keys with worker code.
- Pin worker environments and allowlisted inputs; discover accelerator support and session limits. Checkpoint periodically; a signal handler cannot guarantee recovery from forced termination.
- Exercise the first account end to end before eligible use of the second account.

**Gate:** offline provider fault tests pass; subsequently authorized small real jobs show submission/status/output/recovery with matching artifact hashes and reconciled spend/quota. Kaggle deployment is batch compute, not an assumed always-on production endpoint.

### Phase 2 — Complete the general tabular scientific loop (P1)

- Add a reusable problem intake/preflight report and first baseline under the same validation contract.
- Extend metric support with explicit prediction semantics: labels, probabilities, raw margins, rankings, transformed targets, weights, and class order.
- Audit duplicates, groups, time boundaries, label availability, target leakage, imbalance, and train/serving schema mismatch.
- Make adversarial validation an optional diagnostic where test covariates are permitted. Report uncertainty and drifting features; do not automatically reject at AUC 0.55 or drop predictive timestamps.
- Persist predictions keyed by dataset/row/fold/origin/class and candidate identity. Rolling-origin evaluation may predict a row more than once; a single row-indexed OOF vector is insufficient.
- Run genuine linear, LightGBM, CatBoost, XGBoost, and supported neural baselines when installed. Make fallback identity visible and selectable.
- Add budgeted hyperparameter search and multi-fidelity pruning, caching, deterministic ablations, failure repair, and explicit no-progress stopping.

**Gate:** binary, multiclass, regression, grouped, and temporal datasets complete intake → baseline → measured improvement attempt → selected bundle, with correct fold-local transforms and explicit resource reports.

### Phase 3 — Evidence-driven research and selection (P1)

- Replace the controller's narrow recommended three-turn sequence with task-aware planning and evidence-based stopping.
- Introduce logical research/planning/implementation/evaluation/critique roles. They can be sequential in one controller; additional concurrent agents are not a prerequisite.
- Record each hypothesis, expected benefit, cost, prerequisites, and refutation evidence. Use failed experiments to avoid repeating work.
- Distinguish metric improvement from uncertainty, compute cost, and deployment feasibility. Maintain a Pareto frontier and selectable quality/latency profiles.
- Use paired comparisons on identical validation units, appropriate temporal/group uncertainty methods, and a final untouched evaluation protocol. Fold standard deviation is descriptive, not a universal shakeup probability.
- Add OOF-based greedy blending/stacking, calibration, thresholds, and postprocessing with additional cross-fitting or nested evaluation as appropriate. Fitting blend weights and reporting quality on the same OOF rows can overfit.
- Prioritize by expected information or improvement per unit cost; cache research and expensive reusable features. Stop on budget exhaustion, repeated non-improvement, or sufficient acceptance evidence.

**Gate:** improvements reproduce across predefined evaluation units; rejection/stop decisions are explainable; leaderboard feedback cannot silently alter frozen validation.

### Phase 4 — Broader model-building capability (P1/P2)

Deliver each lane as a vertical slice: real data, baseline, validation, improvement, resource profiling, standalone inference.

| Order | Capability | Initial slice |
| --- | --- | --- |
| 1 | Recommendation/ranking | Popularity/co-visitation → item retrieval → small SASRec where sequence order exists → optional LambdaRank reranker |
| 2 | Text | TF-IDF + linear baseline → genuine pretrained frozen embeddings → small encoder fine-tuning |
| 3 | Vision/multimodal | Genuine pretrained image embeddings → tabular fusion → task-specific fine-tuning where justified |
| 4 | Forecasting depth | Seasonal naive and lagged trees → global sequence models, hierarchical/probabilistic outputs when required |
| 5 | Additional domains | Detection, segmentation, audio, graphs, anomaly detection/clustering integration, then simulation/optimization/RL via dedicated contracts |

For cart recommendation, distinguish an ordered event history from an unordered basket. Use causal attention only for ordered sequences, time-respecting splits, training-only item mappings, explicit negative sampling, cold-start handling, and Recall/NDCG/MRR at specified K. Evaluate retrieval recall separately from reranker quality. The proposed ~0.7–1M parameter transformer is a reasonable candidate budget for a small vocabulary; vocabulary growth and serving requirements change feasibility. Measure CPU latency instead of promising 5–15 ms.

Transformer recipes should expose pretrained checkpoint/version, pooling, head, optimizer schedule, selective fine-tuning, precision, and checkpointing. Add LLRD, layer reinitialization, LoRA/DoRA, AWP, and DAPT incrementally when a benchmark supports them. Do not turn every technique on by default.

**Gate:** each advertised lane has at least one real-data result, leakage audit, reproducible training recipe, resource report, and clean-environment handoff. Coverage claims expand only as lanes pass.

### Phase 5 — Complete inference graphs and application delivery (P1/P2)

- Audit the existing bundle format against a concrete fold-ensemble/preprocessing graph. Extend compatible manifests first; introduce v3 only for an actual incompatible contract.
- Package model(s), fold-specific transforms, tokenizer/item vocabulary, retrieval index, calibration/blend/postprocessing graph, schema, environment lock, training recipe, hashes, selected iteration, and limitations.
- Support both full-fold inference and full-data refit as explicitly different artifacts; verify each independently. Refitting is a new training event, not identical weights from the selected iteration.
- Add golden-input parity with dtype/hardware-appropriate tolerances, finite/shape/schema checks, edge cases, cold starts, and latency/memory/throughput measurements on a named target machine.
- Evaluate quantization, pruning, distillation, or compatible checkpoint merging as new candidates with separate evidence. Never imply guaranteed 98–99% quality retention.
- Include model/data cards, licensing, permitted representative test inputs, deployment examples, monitoring definitions, rollback, and reproducible retraining instructions.
- Add application-specific checks: offline/online feature consistency, feedback-label delay, subgroup performance where relevant, drift alerts, and training-serving skew.

**Gate:** a recipient outside the repository can reproduce inference from the bundle, match expected behavior, measure the promised resource envelope, and identify the exact selected model and dependencies.

### Phase 6 — Demonstrate breadth and competitive strength (P2)

- Maintain a development suite and a separate sealed evaluation suite across tabular, grouped data, time series, recommendation, text, vision, and multimodal tasks; include ordinary business datasets.
- Use previously unseen tasks and record research exposure. Public competition solutions can contaminate agent evaluation even if labels remain sealed.
- Compare the same compute/data allowance against simple baselines, tuned domain baselines, and a suitable strong AutoML baseline where applicable.
- Measure task success rate, baseline-relative quality, cost, GPU hours, wall time, stability, autonomous completion, manual interventions, and bundle correctness.
- For competitions, record private/final results when available and their context. One public leaderboard score or one winning task is insufficient for a Grandmaster-level system claim.
- Run interruption, OOM, provider failure, quota depletion, resource isolation, and clean-room delivery exercises. Retain failures as evidence.

**Gate:** a preregistered diverse evaluation report demonstrates quality and reliable delivery under fixed budgets. State which task families remain unsupported.

## 7. Changes to the Gemini addendum before adopting it

1. **Retain Iteration 0**, but make split choice follow prediction context and supplied authority. Generate folds only for new unfrozen tasks; audit existing frozen folds without replacing them.
2. **Retain persistent OOF artifacts**, but use row/fold/origin identities and valid masks. Include repeated or partially covered temporal predictions explicitly.
3. **Replace universal Apex penalty constants** with task-specific acceptance criteria, uncertainty reports, and Pareto selection. Metric scales and minimize/maximize direction differ.
4. **Use residual correlation as a search heuristic**, not a 0.85/0.92 whitelist. For multiclass/ranking tasks define suitable residuals; promote only on independently measured incremental value.
5. **Reject universal custom-Hessian generation.** Non-smooth/global metrics such as F1 and QWK do not automatically provide stable per-row twice-differentiable objectives. Use validated surrogates, finite-difference checks, correct target-transform chain rules, and the exact metric for selection. The supplied positive SMAPE Hessian approximation is not an exact general Hessian.
6. **Treat GeM/LLRD/reinitialization/AWP code as examples.** GeM clamping discards negative hidden-state information; generalized pooling needs domain-appropriate semantics and constrained parameters. The LLRD example lowercases parameter names but contains an uppercase `LayerNorm.weight` test; broad regex matching also needs architecture coverage. Test zero-layer reinitialization explicitly because Python `layers[-0:]` selects all layers.
7. **Do not validate merged fold checkpoints using their old OOF predictions.** Each merged constituent may have trained on another constituent's validation rows. Evaluate merging/distillation inside an outer split or on a genuinely untouched set. Teacher predictions and pseudo-label data require equally careful boundaries.
8. **Make DAPT/test pseudo-labeling opt-in**, based on competition permissions or the application protocol. Never use evaluation labels; do not assume test covariates are available in production.
9. **Treat TIES/SLERP as experimental compression**, compatible with specific weight spaces and checkpoints; they cannot merge arbitrary heterogeneous models or guarantee ensemble quality. See the [TIES paper](https://arxiv.org/abs/2306.01708).
10. **Remove concealment-based Kaggle advice**, fixed hardware routing, combined-VRAM assumptions, and zero-ban-risk claims. Private datasets and encrypted shards do not change sharing permissions.
11. **Use prediction-distribution checks diagnostically.** Requiring a KS match against OOF predictions would reject valid distribution shift. Use schema/parity/finite checks as packaging gates and distribution comparisons as contextual evidence.
12. **Avoid “zero-token/zero-dependency” guarantees.** Deterministic execution reduces LLM involvement, but tool output still occupies context and ML inference generally needs explicit runtime dependencies.

## 8. Recommended first implementation batch

The next batch should deliver one trustworthy vertical slice, not all model families:

1. Fix the iteration-selection and metric-consistency issues in the existing controller.
2. Design the versioned provider policy and extend governed dispatch with current DeepSeek model IDs, accounted fallback, and repaired search/fetch.
3. Refactor one Kaggle backend through the existing execution service, with explicit credential/data policy, quota discovery, durable jobs, and verified artifacts.
4. Persist aligned OOF artifacts and run one complete general tabular problem through research, iteration, explicit selection, and standalone delivery.
5. Build the small recommendation example as the first new non-tabular vertical slice.

After this batch, prioritize real pretrained text/vision adapters and complete inference-graph delivery. Defer exotic surgery, large teacher models, fold merging, and broad simulation/RL coverage until their prerequisite lanes have evidence.

For future work, define separate LLM/search spend, accelerator-hour, storage, and wall-time budgets per project. Keep the historical $0.30 campaign ledger intact; this plan does not grant a new allowance or change current execution authorization.

## 9. Review scope

This is a source/document review plus targeted offline verification, not a full runtime, security, legal, or model-quality certification. Official service documentation was consulted for planning; no project credentials were used, no remote training jobs were launched, and no holdout was scored. Source files were not modified by this planning task.

Fresh targeted verification: `PYTHONPATH=python/src python3 -m pytest python/tests/test_autonomous_agent_loop.py python/tests/test_kaggle_pool.py -q` — **15 passed in 21.54 seconds**. These tests do not prove platform permission, complete provider integration, robust iteration pinning under all future actions, or model quality. The preserved campaign ledger SHA-256 matched the continuation guide: `827abd984dd960f91cd39725d7dcbdb78d47c114322a0358cbbb08a5b30d0b52`.
