# MLRescurix independent review

Sections tagged **[Handoff]** restate what the packet claims. **[Inference]** is my reading of it. **[Rec]** is my recommendation. I haven't seen the repository or traces, so anything not in the packet is an assumption and is marked as one.

## 0. How much weight the evidence can bear

**[Handoff, internally consistent]** I re-did the ledger arithmetic. Committed = billed + uncertain holds in Run 6 ($0.00619 + $0.03872 = $0.04491), and the headroom deltas across Runs 6, 7 and 8 reconcile to the cent. The numbers are coherent with each other, but I can't tell if they match reality.

**[Inference]** Three calibration points change how to read the packet:

1. **The evidence about exploration behaviour is one 9-turn run.** Runs 7 and 8 were capped at 4 turns by the launch script. That is a configuration choice, not agent behaviour. Every "why the agent didn't explore" factor in §7 of the handoff is a hypothesis from n=1, one model, one dataset.
2. **The three runs are not independent samples.** Run 7's winning score equals Run 6's turn-1 score to about 14 digits (62.46631892131646). Yet the candidate IDs differ (`cand_accf1e98…` vs `cand_4dd30caf…`). Either IDs include non-semantic fields such as session or lineage, or the configs differ in a parameter that has no effect. Either way, "candidate identity equals semantic identity" is unproven.
3. **"Neural tabular wins" is not established.** CatBoost scored 62.4834 against 62.4663, a difference of 0.017 SMAPE points. With two folds and no seed-variance estimate, that is almost certainly inside the noise. Run 8's 62.78 versus 62.47 can't be ranked either.

## 1. Prioritized diagnosis

Ranked by how much each limits autonomy and trustworthy results:

**D1. The headline number can't support a longer campaign.** This is the top priority.
- **[Handoff]** Two pinned folds, 37.5k train rows each, selection repeated on the same folds, no holdout, and `HYPOTHESIS_TRANSFORM` greedily accepts changes when CV improves.
- **[Inference]** Beyond adaptive overfitting, three further risks are plausible:
  - **Scale transfer.** Training on about 6% of the 600k-row pool means rankings (especially neural versus GBDT, and text feature value) may not hold at full scale.
  - **Slice bias.** The dev set is the contiguous rows 0–74,999. If the file has any ordering (by category, time, ID), the dev set isn't representative.
  - **Leaky preprocessing.** TF-IDF/SVD or high-cardinality encoding fit outside the fold leaks. The packet doesn't say.
- **No baseline is reported.** I can't tell whether 62.5 SMAPE is good, and there's no constant or sparse-linear reference.
- **Evaluator limits.** **[Inference]** The Rust evaluator proves the reported metric matches the predictions. It doesn't prove the predictions were made without validation labels, because the worker holds all labels. That is a smaller guarantee than the architecture diagram implies.

**D2. Campaign mechanics consume LLM turns.**
- **[Handoff]** Run 8 lost two of four turns to parameter errors. Run 6 spent seven turns on near-identical submissions. `BLEND_AND_AUDIT` appears in no trace, even though out-of-fold predictions for five families existed after Run 6's turn 1.
- **[Inference]** Blending existing OOFs, deduplicating, and validating parameters are deterministic jobs. The harness should do them, and LLM turns should go to hypotheses. This ranks above model-zoo expansion because, in the one run with room, the agent used one family after turn 1. The zoo's size wasn't what bound it.

**D3. Feedback and memory are weak.**
- **[Handoff]** An 8-turn window, and the veto message content is unspecified.
- **[Inference]** The vetoed candidates had better CV (61.73 vs 62.47). A critic that rejects on train-validation gap alone, with no way to check the candidate on fresh data, converts a measurement into an outright rejection. I don't know how "gap" is computed (in-sample vs. out-of-fold, relative vs. absolute points).

**D4. Budget authority is fragmented and misleading.**
- **[Inference]** At the packet's own estimated rate, 60,302 tokens cost $0.00619, or about $0.10 per million tokens. So 5M tokens is about $0.51, above the $0.28 campaign cap. The USD cap, not the token budget, is the binding constraint, and the user isn't shown that.
- Per-turn holds (about $0.0050 on average in Run 6) are roughly 7× actual estimated spend (about $0.0007). Spend isn't what is exhausting the campaign. Held reservations are.

**D5. Capability breadth (enums, SVD-16, five estimators).** This is real but second-order. **[Inference]** A 16-component SVD of free text discards a lot, and the cheapest test is a sparse linear model on the full TF-IDF matrix with a log target. If catalog text embeds structured fields such as quantity or unit (an assumption to check), numeric extraction is likely worth more than any neural tweak.

**D6. Integrity and contract drift.**
- The hard link is a plausible risk, not an observed corruption.
- I don't know what `WRITE_PYTHON`/`RUN_PYTHON` do. If they execute agent code in the worker with write access, the "rigid enum" and isolation claims both change.
- **[Handoff]** The `RESEARCH_TOPIC` mismatch between Rust and Python, and the Master Plan's false claim about Run 8, show that prose claims drift from code. The packet also contradicts itself on whether an unknown action "crashes" or becomes a rejected turn.

### Alternatives to the handoff's explanations

| Handoff explanation | Plausible alternative | What distinguishes them |
|---|---|---|
| Agent trapped in a veto loop | Duplicate submissions, or hyperparameters silently ignored. Four turns scored exactly 61.7489, three exactly 61.7296, which real LR/dropout changes would not produce | Candidate hashes and effective configs for turns 2–8 |
| Rigid zoo blocked exploration | Weak feedback, a flash-tier model, or prompt anchoring | The turn-3 prompt as sent; replaying the same state with another model |
| Critic protects against overfit | Gap is the wrong statistic for a 37.5k-row training half | Confirmation-split scores of the vetoed candidates |
| Neural beat GBDT | Statistical tie | Seed-variance estimate |
| Planned ensemble was lost to the cap | Unevidenced; the packet shows no blend plan in the trace | The turn-8 model output |

## 2. Experiment architecture

**Principle [Rec]:** the agent proposes declarative specs, the harness executes them, and Rust scores predictions against labels the worker can't reach. Widen what is allowed behind that boundary in order of expected value, not sophistication.

### Axes and what to try first

The hypothesis taxonomy is generic: **representation, model family, target/loss, regularization/capacity, features, ensemble, data scale.**

1. **Baseline ladder.** Constant, sparse TF-IDF + linear (log target), GBDT on structured features, GBDT with SVD text. Each rung is a hypothesis about where signal lives. Run it offline on CPU first (zero LLM cost).
2. **Target and loss.** Log or power transforms, metric-aware objectives, and prediction post-scaling for SMAPE bias. These apply to any skewed positive target.
3. **Text.** Word and char n-gram TF-IDF → SVD of varying size → cached offline sentence embeddings → fine-tuned small transformer (GPU, later).
4. **Structured extraction.** A generic "parse numbers with units from text" step.
5. **Ensembling.** Harness-run OOF blending and seed averaging, with no LLM turn.
6. **Neural models.** MLP on embeddings plus tabular features first. FT-Transformer, TabNet and custom nets come later, only where simpler models show the signal is under-used. At 37.5k training rows, tuned GBDTs often match neural tabular models (a general tendency, not tested here). Neural models are likely to matter most on text and image representations.

### Extensibility in three tiers

- **Tier 1: registry components.** Each component declares `name, version, task_types, modalities, param_schema, fit/predict contract, dependency pins`. The prompt's allowed-parameter text is generated from the registry per task type, so regression never sees `C`. Adding a family is a reviewed code change with no prompt edits.
- **Tier 2: pipeline specs.** A candidate is a small DAG (Representation → Model → Postprocess) of registry components. `candidate_id = H(canonical spec, component versions, data-view hash, seed, env lock hash)`. Canonicalize defaults before hashing so equivalent configs collide on purpose.
- **Tier 3: agent-written plugins.** The interface is `fit(train_view, params, rng) → state`, `predict(state, eval_view) → preds`. The harness drives each fold. The sandbox gets a train view with labels and an eval view without them, no network, a read-only mount, CPU/memory/time limits and a pinned image digest. Plugin code hash goes into the candidate ID. Static checks are advisory only. **[Rec]** Defer Tier 3 until Tiers 1–2 are validated, and first learn what `WRITE_PYTHON` does today.

### Reproducibility

- Derive seeds from the candidate ID and record the environment lockfile hash.
- Re-execute the final candidate and compare predictions. Run 6/7 suggests determinism on this one machine. That says nothing about Kaggle or other hardware, so claim bitwise parity only on matching hardware.
- Cache embeddings as artifacts keyed by (data hash, model hash).

### Generalizing across datasets

- A `TaskSpec` carries modality, column roles, metric, direction and split rules (group, time, stratify).
- A deterministic data profiler runs before any LLM call: types, cardinality, missingness, text lengths, duplicates, group structure and target distribution. Its output goes to the agent.
- Include a CI test that runs the loop offline on two or three small non-Amazon datasets (a classification set, a regression set with another metric). If any step fails on them, Amazon is hardcoded somewhere.

## 3. Recovery and exploration

A **steering controller** is deterministic harness code, not a prompt suggestion. It reads the experiment ledger and emits a steering block and hard gates.

**Failure handling**
- **Invalid parameters:** validate against the registry schema, then give structured feedback and one or two repair attempts per turn. Repairs cost tokens but don't consume a turn.
- **On the untested `C → alpha = 1/C` patch:** I'd drop it. Ridge's penalty and LogisticRegression's `C` aren't equivalent, so silently translating hides the agent's misunderstanding and makes the recorded config differ from the effective one. Reject with a repair message listing valid parameters. If you do translate, record both `requested` and `effective` and flag it.
- **Duplicates:** if a canonical spec was already evaluated, return the cached result and count it as a stalled turn. Near-duplicates (same axis, small parameter distance, score within noise) three times means a declared local plateau.
- **Vetoes:** return the structured reason with remedies (weight decay, early stopping, width, more data). Once a confirmation split exists, route a vetoed candidate there instead of rejecting it. A confirmed better candidate can be promoted, and the veto threshold isn't loosened.

**States and triggers**
- **EXPLORE** until at least N distinct axes have been covered. **EXPLOIT** refines the incumbent. **RECOVER** triggers on any of:
  - k non-improving turns (improvement must exceed a *measured* noise floor, taken from seed repeats of the incumbent)
  - two vetoes in one family
  - two duplicates
  - a repeated error class
- **In RECOVER:** run harness diagnostics first (learning curves, residuals by segment, error analysis). These are free and private. If diagnostics don't suggest a direction, either research or pivot to the least-covered axis.
- **Hard rule:** a veto or plateau in one family requires a *declared structural change* or a pivot.

**When to research**
1. A bounded landscape phase at campaign start.
2. In RECOVER, after diagnostics.
3. For unfamiliar failure classes.

Never for hyperparameter tweaking. Every request states a question and the decision it informs, and afterward the agent logs adopted or rejected plus the reason.

**Recorded rationale.** Each turn carries `diagnosis, hypothesis, axis, parent, expected_effect, alternatives_rejected, outcome, lesson, next`. The harness writes facts (scores, hashes, dispositions). The agent writes rationale, labelled `agent_stated`. A handoff generator may state only what is joined to ledger entries by candidate ID.

## 4. Web research design

**Pipeline:** request → policy gate → query check → reserve → search → quarantine reader → evidence note → ledger.

**Private-data protection**
- Per-dataset `research_policy`: `disabled`, `generic_only` (default) or `reviewed`.
- Queries are built from an abstract task card (task type, modality, metric, technique names). They never include column names, values, row counts or IDs.
- Add an exact-match check against a harness-held hash set of column names and long cell values, in addition to regex. Log every outbound query. **[Inference]** Semantic leakage via rephrasing can't be fully excluded, which is why the default is generic-only.
- **Benchmark integrity for the Amazon test:** since the dataset is public, searches may surface competition write-ups or leaderboard solutions. If the goal is to test the agent's research ability, decide the policy up front: block those domains or dataset names, and log domains retrieved.

**Untrusted results and injection**
- Results are data. A **quarantined reader** call with no action authority converts snippets or pages into a schema note: `{claim, technique, applicability, source_url, retrieved_at, content_hash, confidence}`, capped at 2 KB.
- The planner sees notes labelled "untrusted external summary," never raw text.
- Adoption goes through registry or spec validation, so injected instructions can't trigger an action the schema doesn't allow. Never execute web-derived code outside the plugin sandbox.
- Start with snippets only. Add fetching later, restricted to an allowlist (for example arXiv and library docs).

**Accounting**
- Reserve one query at dispatch and settle on the response. This fixes the packet's "count only after success" gap.
- Classes:
  - `succeeded`
  - `failed_nonbilled`: only when the provider explicitly confirms a pre-processing rejection
  - `uncertain`: timeouts, 5xx, resets; counts as consumed
- Retries are new queries. Add a normalized-query cache and caps per phase (for example landscape ≤10% of the total, ≤5 per stall) so 500 queries can't vanish in one loop.

## 5. Resource governance

**Constraint:** the authorized cap stays $0.280494. Nothing here clears, resets or raises it.

**Structure [Rec]**
- Rust remains authority. One content-hashed `campaign_policy` holds tokens, USD, searches, accelerator hours, turns and wall-clock.
- Per-run caps are derived as a fraction of the remaining campaign allowance, not hardcoded. At startup, refuse to run if `POLICY_AMAZON_LIVE_20261102.json`, `governance.py` or `operational_budgets.json` disagree, and print the diff.
- Show the owner the implied conversion ("5M tokens ≈ $0.51 at the observed rate; the binding limit is $0.28"). Whether to authorize anything different is the owner's decision, outside this design.
- Hide any dimension from the agent unless it's connected and enforced. Today's `enforced: False` counters shouldn't appear as budgets.

**Ledger states (append-only, hash-chained):** `reserved → settled_provider_reported | settled_estimated | uncertain → reconciled_released | reconciled_confirmed`.

**Evidence-based reconciliation of the existing $0.18604**
1. Export the provider's activity or usage records for the key and time window.
2. A dry-run tool matches ledger entries to provider generation or request IDs, computes `reserved − actual` per entry, and proposes releases. It never writes.
3. After operator approval, append a signed `reconcile` event citing the evidence file hash. Rust verifies per-entry coverage and caps each release at the held amount. The history stays intact and both snapshots are logged.
4. Entries without evidence stay `uncertain`.

**Caveat:** if old ledger entries didn't record provider request IDs (I can't tell from the packet), per-entry matching is impossible. Only an aggregate comparison to the account's total spend for that window remains. That gives an upper bound on actual spend but not a per-entry release. Check this first.

**Preventing the ratchet going forward**
- Size reservations from measured prompt tokens, a set `max_tokens` and model prices, not a flat hold.
- Settle from provider-reported usage or cost when present, and release the difference immediately. Verify against the provider's current docs what OpenRouter returns for this route.
- Record generation IDs at dispatch so later reconciliation is possible.
- Keep `uncertain` only for ambiguous outcomes, with a reconcile-by date.
- A provider-side credit limit on the API key is an independent backstop. That is an owner setting.
- **Wrap-up reserve:** carve out a finalization allowance up front. At about 85% of any dimension, switch to blend, bundle and handoff. Stop reasons distinguish `soft_limit_wrapped_up` from `hard_limit`.
- **Kaggle hours:** reserve at kernel push (timeout × accelerator), settle from logs, and mark `enforced: true` only once reservation and quota reads are wired. Uploading private data requires an explicit per-dataset consent flag.
- **Invariants to test:** `billed + reserved + uncertain ≤ cap` at all times, and `Σ receipts = Σ settled`.

## 6. Scientific validation

**Partitions, assigned by Rust at campaign start**
- **Lockbox:** sealed, sampled so it resembles the deployment distribution, with group-aware splitting if duplicates or product families exist.
- **Confirmation set:** disjoint, with limited queries.
- **Search pool:** everything else, used for the fidelity ladder.
- Sizes should come from the SMAPE standard error (bootstrap), not rules of thumb. Illustratively, a 600k pool could give tens of thousands of rows each to lockbox and confirmation.

**Check before calling the 525k rows untouched:** the packet says they weren't used in *these* runs. I need to know whether Runs 1–5, earlier agents, or humans looked at them for evaluation, EDA or tuning. If so, they aren't a pristine lockbox. Also check row ordering, near-duplicate text and group structure in `train.csv`. The existing 75k slice has been selected against heavily and can't be the lockbox.

**Reducing adaptive overfitting**
- **Fidelity ladder:** cheap repeated-seed screening on subsamples → top-k on larger slices → confirmation → lockbox.
- **Paired comparisons** on identical folds. An improvement counts only if it exceeds the measured noise floor.
- **Reusable-holdout mechanism:** the agent sees coarse feedback (confirmed / not confirmed, with bounded query count). Raw confirmation scores stay with the harness. Thresholdout-style noise is a reasonable choice. Don't overstate its guarantees.
- **Trial counter:** record candidates and hypotheses evaluated. Report the gap between the best dev score and its confirmation score as an observed winner's-curse estimate.
- **Leakage controls:** all fitting (TF-IDF, SVD, encoders, scalers, target encoding) inside the training fold, with nested folds for target-dependent encodings. Group duplicates.
- **Canaries in CI:**
  - *Label-permutation:* after many candidates on shuffled labels, dev shows apparent gains while confirmation and lockbox show none.
  - *Planted leak:* a feature that encodes the target must be caught.
- **Critic:** treat the gap as a diagnostic, combined with confirmation results and learning curves.

**Provenance.** Every score in a handoff is joined by `candidate_id`. No `latest_*` fields. A property test compares every handoff field against the scorecard and ledger, which would have caught the Run 6 defect class generically.

**Lockbox protocol.** Pre-register the final candidate hash (including the final blend spec) and the baseline set before opening. Open once, log it, and report a confidence interval. A second open voids it as an untouched holdout.

**What a successful holdout proves:** one pre-registered candidate's performance estimate, with sampling error, on the lockbox distribution, assuming no upstream leakage (duplicates spanning pool and lockbox would break it).

**What it doesn't prove:** production readiness, robustness to distribution shift, latency, calibration, behavior on SMAPE-sensitive low-price segments (report segment metrics), or anything about candidates that weren't pre-registered. `production_certified: false` should stay false unless owner-defined release criteria beyond the holdout are met.

## 7. Staged implementation plan

File names below are from the packet where known, otherwise proposed.

### Immediate fixes (no new live spend)

| # | Change | Acceptance |
|---|---|---|
| 0.1 | **Replay harness:** a scripted fake gateway driven by recorded `autonomous_trace.jsonl` responses | Run 8's responses replay deterministically; Run 6's replay surfaces the duplicate pattern |
| 0.2 | **Single action contract:** `schemas/actions.json` generates the Rust `ALLOWED_ACTIONS` and the Python `ACTIONS`. Remove `RESEARCH_TOPIC` from the prompt until it dispatches | CI test: both sets equal; prompt-offered ⊆ dispatchable; unknown action yields a rejected turn, not a crash |
| 0.3 | **Registry-generated param schemas** per task type; replace the silent `C→alpha` patch with reject-and-repair | Replay of Run 8's turn 1 yields a repair message, not a lost turn |
| 0.4 | **Candidate canonicalization and dedupe,** recording `requested` and `effective` params | Same canonical spec yields the same ID across sessions; a cache hit reruns nothing |
| 0.5 | **Data protection:** replace `os.link` with a copy (clonefile on APFS), `chmod 0444`, hash check at turn start and end | In-place write to scratch leaves master hash unchanged; mismatch aborts the run |
| 0.6 | **Handoff generator joined by candidate_id** | Property test across all fields |
| 0.7 | **Search accounting:** reserve at dispatch in `governed_research.py` | Test that timeout, 500 and network error each consume budget |
| 0.8 | **Budget consistency check** at startup; hide unenforced dimensions | Startup fails on any disagreement |
| 0.9 | **Reconciliation dry-run** (read-only) | Produces a per-entry or aggregate-only report, per what the ledger recorded |
| 0.10 | **Offline measurement on local CPU:** constant, sparse-linear and GBDT baselines; seed variance of the incumbent; whether train-fold fitting is leak-free | A baseline table and a noise floor exist before any further LLM spend |

### Larger changes

| Phase | Scope | Key tests / acceptance |
|---|---|---|
| **1: Loop control** | `experiment_ledger.py`, `steering.py`, `digest.py` (constant-size experiment table replacing the sliding window), `auto_blend.py`, wrap-up mode | Run 6 replay forces a pivot by turn 5; zero repeat evaluations; clean bundle at soft limit |
| **2: Validation** (`custom_protocol.rs`, `trusted_evaluation.rs`) | Partitions, confirmation oracle, lockbox seal and open-once, trial counter, canaries | Permutation and planted-leak canaries; second lockbox open fails; partition reproducible from the seal |
| **3: Registry and representations** | Registry, sparse text, log-target and metric-aware options, cached embeddings, an optional `NumericExtract` step | A new component is added by registry change only; two non-Amazon datasets run end to end in CI |
| **4: Live research** | Reader, policy gate, accounting, mock provider first | Injected-instruction page changes no action; no schema tokens in outbound queries; failed queries counted |
| **5: Plugins, torch models, Kaggle** | Sandbox, embedding models, kernel reservation | Sandbox cannot reach network or eval labels; plugin results reproduce from the code hash; killed kernel settles correctly |
| **6: Other modalities** | Image, time-series adapters | New adapter requires no agent-loop changes |

**Tradeoffs.** Phases 0–2 add engineering time before any visible capability gain, but they make later results believable. I put research (4) before plugins (5) because it is cheaper and registry-bound, but research is less useful until Tiers 1–2 can express what is found. The fidelity ladder and confirmation split cost wall-clock time. The reusable holdout trades some information for protection.

## 8. Fresh Amazon live-test plan

### Before spending (all zero-cost)

1. **Headroom gate.** From the packet's figures, recent runs committed about $0.0023–$0.0050 per turn, so $0.0104 allows roughly **2–4 turns**. A 30-turn run at the same hold rate needs about $0.07–$0.15. At estimated actual spend, about $0.02–$0.05 would be real cost. So the run depends on the reconciliation result, or on Phase 0's reservation sizing, which should reduce holds. If neither frees enough headroom, a meaningful live test is not possible under the current authorization. Say so rather than running a 3-turn test and calling it a result.
2. All Phase 0–2 tests green; replay tests pass; canaries pass; data hashes verified.
3. Baseline ladder and seed-variance noise floor measured offline.
4. Run from a **clean commit**. The packet's working tree is dirty, so record the commit and the diff hash.
5. Pre-register the success criteria (hashed) and seal the lockbox.
6. Dry run through the full pipeline to a bundle with the fake gateway.

### Trace to capture, per turn

- Prompt as sent (redacted) and its hash; raw response; parsed action.
- Validation and repair events.
- Canonical spec hash, `requested` vs `effective` params, code and env hashes, seeds.
- Per-fold scores, wall time, tokens (provider-reported and local).
- Generation IDs and ledger events (reserve, settle, release).
- Steering state and decision; agent rationale labelled `agent_stated`.
- Critic output; confirmation-oracle query count; data hashes.

### Success criteria

**Process (all required)**
- Zero turns lost to schema or parameter errors.
- No duplicate evaluations except cache hits.
- At least three axes and two representation or family types beyond the incumbent.
- Handoff tokens equal receipts; ledger conservation holds; data hash constant.
- A terminal state with a bundle from a soft-limit stop.
- If no natural stall occurs, a separate injected-stall test to verify RECOVER fires with a logged rationale.

**Outcome**
- The primary comparison is confirmation-split SMAPE against the offline baseline ladder on identical splits, with improvement beyond the noise floor.
- The lockbox is opened once, at the end, with a confidence interval.
- Matching the best scripted baseline within noise at lower human effort, with a coherent hypothesis trail, is a defensible result. Beating 62.47 on development folds is not the criterion.

### If the first run falls short

| Observation | Likely cause | Next step |
|---|---|---|
| Loops, dupes or errors persist | Harness gap | Fix and replay; no spend |
| Agent ignores or superficially satisfies steering | Model or prompt limit | Replay the same stalled state with a stronger model for a few turns (small cost); try a constrained axis-menu prompt |
| Diverse exploration, no gain over baselines | Baselines already strong, or data limit | Inspect diagnostics and error segments before concluding anything about the agent |
| Dev gain, no confirmation gain | Selection bias | Reduce adaptivity (fewer queries, coarser feedback) |
| Early halt on budget | Reservation sizing | Fix reservation logic; no policy changes without the owner |

For comparing arms (steering on vs. off), single runs can't support conclusions. Run 7 reproducing Run 6 suggests low sampling randomness, so repeats need varied seeds or temperature to mean anything. Plan at least three per arm.

## Evidence that would change my view

| Evidence | What it would change |
|---|---|
| Candidate hashes and effective configs, Run 6 turns 2–8 | Whether the loop was resubmission or ignored parameters (D3 vs D2) |
| The turn-3 prompt as sent | Whether the agent saw the veto reasons, which weighs prompt against model capability |
| What `WRITE_PYTHON` / `RUN_PYTHON` actually do, and their file permissions | Whether Tier 3 already exists and whether isolation is the more urgent problem |
| Whether TF-IDF/SVD and `ENCODE_HIGH_CARD_CAT` fit inside folds | Whether D1 includes outright leakage |
| Critic gap definition and CatBoost's disposition in Run 6 | How much to blame the critic |
| Target transform, loss, and baseline SMAPEs | How much headroom there is in representation and loss choices |
| Row order, duplicates and groups in `train.csv`; whether anything looked at the 525k rows earlier | Whether the lockbox is pristine and the split design |
| Whether ledger entries recorded provider request IDs | Whether per-entry reconciliation is possible |
| Sampling settings (temperature) | How much repeat runs tell you |
| What happens when an unknown action is returned (rejected turn vs process crash) | Severity of the `RESEARCH_TOPIC` mismatch |

If you can share the Run 6 trace, the candidate hashes and the two or three code files behind `WRITE_PYTHON`, the critic and the fold fitting, I can firm up the diagnosis considerably.