# MASTER IMPLEMENTATION PROMPT: MLRESCURIX AGENT-FIRST AUTONOMOUS DATA SCIENCE HARNESS (P13–P17)

## 1. ROLE & MISSION OBJECTIVE
You are a Staff Rust & Python Machine Learning Systems Engineer working on **`MLRescurix`**—a dual-tier, context-bounded autonomous ML harness built in **Rust** (`crates/harness`) and **Python 3.13** (`python/src/mlrescurix_worker`), backed by a sub-50ms warm Docker sandbox pool, Content-Addressed Storage (CAS), and SQLite state persistence (`state.sqlite` under a 128 MiB admission ceiling).

Your mission is to refactor `MLRescurix` from a **linear, single-pass compiler pipeline** into a **Reflexive, Agent-First Autonomous Data Science Organization** that operates like an elite human data science team:
1. **M1 Orchestrator (Principal Data Scientist / Project Owner):** Maintains a persistent **Research Ledger**, formulates explicit hypotheses, delegates experimental sprints to specialized M2 sub-agents, and reflects on adversarial critiques in a closed loop until convergence.
2. **M2-EDA (Exploratory & Visual Forensics Agent):** Inspects raw data, cleans dirty types, runs exploratory Python scripts, generates visual distribution plots + compact statistical digests, and locks an immutable **Dataset Topology Contract** (target, IDs, entity groups, timestamps, semantic column roles).
3. **M2-FE (Feature Engineering & Ablation Scientist):** Writes hypothesis-driven feature engineering experiments in isolated git worktrees, tests them via fast paired CV ablations, and commits only verified, leakage-free features.
4. **M2-Model (Multi-Paradigm Modeling & Residual Forensics Agent):** Trains diverse model families (`CatBoost`, `LightGBM`, `XGBoost`, `HistGradientBoosting`, `PenalizedLinear`, `TabNet`/`MLP`), evaluates across leakage-proof CV splits, and produces a structured **Error Forensics Report** (worst slices, false positives/negatives, learning curve gaps).
5. **M2-Ensemble (Pareto Blending & Calibration Agent):** Optimizes convex/rank ensembles across diverse Pareto-frontier models and calibrates Out-of-Fold (OOF) decision thresholds.
6. **M2-Critic (Adversarial Staff Reviewer — `critic-audit`):** Exercises hard veto power before packaging. Audits every candidate for target memorization, train-val overfitting, dropped text/quasi-ID signals, model monoculture, and slice degradation, emitting a `CritiqueReceipt` that forces M1 to reflect and iterate.

---

## 2. ARCHITECTURAL VULNERABILITIES YOU MUST ELIMINATE
Your implementation must permanently resolve these 7 empirical failure modes uncovered during live stress testing:
1. **Quasi-Identifier Catastrophic Memorization:** Never allow naive in-fold target encoding on high-cardinality columns (e.g., `Name`, `Ticket`), which causes 100% training memorization and collapses validation balanced accuracy to `0.5000`.
2. **Silent Target Misbinding:** Eliminate all `target_col = "target"` or `df.columns[-1]` fallbacks in `iteration_loop.py` and `autonomous_rescue.py` (which scored `Embarked` instead of `Survived`).
3. **ID Column Contamination:** Eliminate hardcoded `"row_id"` checks in `bundle_candidate()`. Surrogate keys (`PassengerId`, UUIDs, monotonic indices) must be deterministically detected and isolated from the feature matrix.
4. **Dirty/Mixed-Type Fragility:** Automatically parse and coerce dirty currency strings (`"$1,200.50"`), percentages (`"15.2%"`), accounting negatives (`"(450)"`), and universal missing sentinels (`"?"`, `"-999"`, `"N/A"`, `"null"`, `"none"`) prior to schema inference.
5. **Entity & Temporal Grouping Blindness:** Automatically detect repeated entity identifiers (patients, users, shared tickets/accounts) and temporal ordering columns, routing CV to `StratifiedGroupKFold`, `GroupKFold`, or `PurgedTimeSeriesSplit`.
6. **Extreme Class Imbalance Starvation in Tier 0:** Upgrade `MultiFidelityIterationEngine` ($N \le 500$) so rare events ($0.01\%\text{--}1\%$) are never starved to zero positives during fast proxy screening.
7. **Out-of-Core OOM Kills ($> 4\text{ GB}$):** Replace eager Pandas full-table materialization with streaming Polars / DuckDB / PyArrow execution respecting `cgroup v2` memory limits.

---

## 3. TARGET REPOSITORY FILE MAP & MODIFICATIONS

### 3.1. Contracts & Schemas (`contracts/v2/`)
Create or update the following JSON schemas:
- `contracts/v2/dataset_topology.schema.json` *(NEW)*: Immutable data contract locked after EDA.
- `contracts/v2/research_ledger.schema.json` *(NEW)*: Persistent state of M1's hypotheses, ablations, best CV scores, and reflections.
- `contracts/v2/critique_receipt.schema.json` *(NEW)*: Structured adversarial audit output from `M2-Critic`.
- `contracts/v2/error_forensics.schema.json` *(NEW)*: Residual slice analysis and failure cohort report from `M2-Model`.
- `contracts/v2/candidate_synthesis.schema.json` *(UPDATE)*: Support multi-family model specifications and preprocessing graph serialization.

### 3.2. Rust Control Plane (`crates/harness/src/`)
- `storage.rs`: Bump `PRAGMA user_version = 11`. Add tables `dataset_topology_contracts`, `research_ledger_epochs`, `feature_ablations`, and `critique_receipts`. Enforce 128 MiB CAS budget with automatic eviction of rejected Tier-0 scratch artifacts.
- `agents/mod.rs` & `agents/orchestrator.rs`: Implement the **M1 Reflexive State Machine**:
  `Discovery (M2-EDA)` $\rightarrow$ `Lock Topology Contract` $\rightarrow$ `Hypothesis Generation` $\rightarrow$ `M2-FE Ablation Sprint` $\rightarrow$ `M2-Model Multi-Family Sprint` $\rightarrow$ `M2-Ensemble` $\rightarrow$ `M2-Critic Audit` $\rightarrow$ `(If VETO: M1 Reflection & Next Epoch | If APPROVED: Bundle & Verify)`.
- `rescue.rs`: Wire CLI commands `profile-dataset`, `propose-hypothesis`, `critic-audit`, and `validate-temporal-split` directly to the new Python agent tools.
- `dashboard.rs`: Add SSE endpoints `/api/ledger`, `/api/critiques`, and `/api/eda-artifacts` to stream EDA charts, feature ablation lift tables, and Critic veto/approval logs in real time.
- `verify.rs`: Extend `verify-bundle` to test `inference.py` and `serve.py` against **raw, uncleaned input vectors** (containing raw strings, missing sentinels, and ID columns) to prove zero train-serve skew.

### 3.3. Python Scientific Kernel (`python/src/mlrescurix_worker/`)
Create and refactor the following modules:
- `scientific/dirty_sanitizer.py` *(NEW)*: Zero-copy Polars/Arrow dirty string cleaner and sentinel normalizer.
- `scientific/semantic_sieve.py` *(NEW)*: Information-theoretic 4-stage column classifier & target/entity/temporal resolver.
- `scientific/primitives.py` *(NEW)*: Verified, leakage-proof building blocks (`NestedOOFEmpiricalBayesEncoder`, `QuasiIdDecomposer`, `LightweightTextEmbedder`, `InformativeMissingnessImputer`, `MIGatedInteractionGenerator`).
- `scientific/eda_agent.py` *(NEW)*: M2-EDA execution harness generating visual plots + compact $< 4\text{ KB}$ Markdown/JSON `EdaDossier`.
- `scientific/fe_ablation.py` *(NEW)*: Paired Baseline-vs-Hypothesis CV evaluator for M2-FE.
- `scientific/model_zoo.py` *(NEW)*: Unified multi-paradigm estimator factory (`CatBoost`, `LightGBM`, `XGBoost`, `HistGradientBoosting`, `PenalizedLinear`, `TabNet`/`MLP`) + `ErrorForensicsAnalyzer`.
- `scientific/critic_engine.py` *(NEW)*: Deterministic statistical auditor + LLM adversarial critic producing `CritiqueReceipt`.
- `scientific/out_of_core.py` *(NEW)*: DuckDB / Polars streaming engine and memory-aware reservoir sampler.
- `scientific/iteration_loop.py`, `candidate_runner.py`, `autonomous_rescue.py`, `pareto_ensemble.py`, `calibration.py` *(REFACTOR)*: Wire into the locked `DatasetTopologyContract` and reflexive loop.

---

## 4. DETAILED SPECIFICATION OF AGENT TOOLS (CONTEXT-FIREWALL SAFE)

Equip the M1 Orchestrator and M2 Sub-Agents with the following high-leverage tools. **Crucial Rule:** No tool may return raw DataFrame dumps or $> 8\text{ KB}$ of text to the LLM context window. All outputs must be structured, compressed digests.

### Tool 1: `eda_forensics_suite(dataset_uri, problem_statement, explicit_target=None)`
Executed by **M2-EDA** inside the warm Docker pool:
1. **Dirty Data Sanitization Pass (`dirty_sanitizer.py`):**
   - Scans string/object columns in Polars/PyArrow.
   - Maps universal sentinels `{"?", "-999", "-9999", "N/A", "n/a", "NA", "null", "NULL", "None", "none", "nan", "NaN", "", " "}` to native `null`.
   - Strips currency/formatting (`$`, `€`, `£`, `¥`, `,`), converts accounting parentheses `"(1,234.50)"` $\rightarrow$ `-1234.50`, and parses percentages `"15.4%"` $\rightarrow$ `0.154`. If $\ge 85\%$ of non-null values in a string column coerce cleanly to float, promote the column to `Float64` and record the coercion rule for `bundle_v2`.
2. **Multi-Signal Target Resolution:**
   - Priority 1: `explicit_target` from CLI or user prompt.
   - Priority 2: Exact lexical match in problem statement against column headers.
   - Priority 3: Canonical target lexicon match (`survived`, `target`, `label`, `y`, `churn`, `fraud`, `default`, `outcome`, `class`, `price`, `salary`, `revenue`, `is_*`, `has_*`).
   - Priority 4: If ambiguous, M2-EDA inspects candidate columns (excluding monotonic IDs and high-missingness attributes) and locks the target in `dataset_topology.json`. **Never blindly fall back to `df.columns[-1]`.**
3. **4-Stage Information-Theoretic Semantic Column Sieve (`semantic_sieve.py`):**
   For every non-target column $X$ with $N$ non-null rows and $U$ unique values, compute:
   - Uniqueness ratio: $\rho_{\text{unique}} = U / N$
   - Effective support mass: $M_k = \frac{1}{N} \sum_{c=1}^{U} n_c \cdot \mathbb{I}(n_c \ge k)$ where $k = \max(10, \lceil 2 / p_{\min} \rceil)$
   - Sub-token vocabulary coverage ($C_{\text{cov}}$): Fraction of rows sharing whitespace/punctuation sub-tokens `[\s,\.;:/\-_]+` that recur $\ge k$ times.
   - Digit-masked template collapse ($\tau_{\text{mask}}$): $1 - (U_{\text{masked}} / U)$ after replacing `\d+` with `#`.
   - Monotonicity ($\mu_{\text{mono}}$) & uniform step check ($\Delta x_i = 1$).
   - **Deterministic Classification Rules:**
     - `DROP_SURROGATE_ID`: $(\rho_{\text{unique}} \ge 0.98 \text{ or } \mu_{\text{mono}} = 1.0 \text{ with } \Delta x_i = 1) \land C_{\text{cov}} < 0.05 \land \tau_{\text{mask}} < 0.10$. Action: Hard exclude from `feature_cols`; register in `id_cols`.
     - `QUASI_ID_DECOMPOSE`: $(0.05 < \rho_{\text{unique}} < 0.98 \text{ or } M_k < 0.50) \land (\tau_{\text{mask}} \ge 0.50 \text{ or median cluster size } \in [2, 15])$. Action: Ban raw target/one-hot encoding; route to `QuasiIdDecomposer` and flag as candidate `entity_group_col`.
     - `SHORT_TEXT_EMBED`: $\rho_{\text{unique}} > 0.20 \land (C_{\text{cov}} \ge 0.40 \text{ or median word count } \ge 2.5)$. Action: Ban raw categorical encoding; route to `LightweightTextEmbedder` (extract recurring sub-tokens like `"Mr."`, `"Mrs."`, `"Dr."`, surnames + Char/Word TF-IDF SVD).
     - `SAFE_CATEGORICAL`: $M_k \ge 0.80 \text{ or } U \le \min(250, 0.05 N)$. Action: Route to One-Hot ($U \le 15$) or `NestedOOFEmpiricalBayesEncoder`.
     - `TEMPORAL_AXIS`: Parses as ISO-8601 date/timestamp or Unix epoch in plausible range `[1970..2035]`.
     - `NUMERIC_CONTINUOUS` / `NUMERIC_DISCRETE`: Clean numerical columns.
4. **Visual & Statistical Dossier Generation:**
   - Saves diagnostic SVG/PNG plots to `.m2/artifacts/eda/` (Target distribution, top 15 bivariate mutual information / target rate charts, missingness correlation matrix).
   - Returns a $< 4\text{ KB}$ `EdaDossier` JSON/Markdown summary to M1 containing exact quantiles, missingness correlation with target, detected quasi-IDs, and the locked `DatasetTopologyContract`.

---

### Tool 2: `run_exploratory_python(code_snippet, timeout_s=15)`
Allows **M2-EDA** or **M2-FE** to behave like a human data scientist testing a quick hypothesis in a Jupyter cell:
- Executes `code_snippet` inside the warm Docker container against a read-only view of the dataset.
- **Context Firewall Wrapper:** Intercepts `stdout`, converts any printed DataFrames into compact Markdown tables (max 15 rows $\times$ 10 columns + summary stats), captures exceptions cleanly, and returns the digest in $< 2\text{ KB}$.

---

### Tool 3: `ablate_feature_hypothesis(hypothesis_id, candidate_features_py, fidelity="auto")`
Executed by **M2-FE** to verify whether a new feature group genuinely helps before committing it:
- Evaluates `candidate_features_py` against the current best baseline feature set using the exact locked CV splitter (`StratifiedKFold`, `StratifiedGroupKFold`, `GroupKFold`, or `PurgedTimeSeriesSplit`).
- Computes:
  1. $\Delta \text{CV Score}$ (mean and fold-by-fold win rate: e.g., improved on 4/5 folds).
  2. $\Delta \text{Train-Val Gap}$: $(\text{Score}_{\text{train}} - \text{Score}_{\text{val}})$.
  3. Top split/permutation importances of the newly introduced features.
- **Auto-Gate:** If $\Delta \text{CV} \le 0$ or if $\Delta \text{Train-Val Gap}$ spikes above $+0.08$ (overfitting), the tool marks the ablation as `REJECTED` in the `ResearchLedger` so M2-FE can discard or refine the feature instead of polluting `features.py`.

---

### Tool 4: `train_and_diagnose_models(model_families, feature_commit_hash, tier="tier1")`
Executed by **M2-Model**:
- Trains a specified portfolio of diverse model families on the validated feature set:
  - `catboost`: Native ordered boosting & categorical combinations.
  - `lgbm`: Fast histogram GBDT with colsample/subsample regularization.
  - `xgboost`: Depth-wise / loss-guided regularized GBDT.
  - `hist_gbdt`: Scikit-learn native Histogram GBDT.
  - `penalized_linear`: `LogisticRegression` (L1/L2/ElasticNet) or `Ridge`/`ElasticNet` wrapped with `RobustScaler` + `SimpleImputer` + missingness indicators (essential anchor for small-$N$ and sparse SVD text features).
  - `neural_tabular`: Lightweight Residual MLP or TabNet architecture.
- Automatically computes **Out-of-Fold (OOF) Predictions** and runs the **Error Forensics Analyzer (`ErrorForensicsReport`)**:
  - Fits a shallow depth-3 surrogate decision tree on the absolute OOF residuals $\vert{}y - \hat{y}_{\text{oof}}\vert{}$ to identify the **Top 3 Worst-Performing Data Slices** (e.g., `"Pclass == 3 AND Sex == 'male' AND Age_is_nan == 1 -> Error Rate 38.2%"`).
  - Reports feature split-gain concentration (Gini/Herfindahl index of feature importances).
  - Reports train-vs-validation metric gap per fold.

---

### Tool 5: `critic_audit_candidate(candidate_id)`
Executed by **M2-Critic** (`critic-audit` CLI / sub-agent). Combines **Deterministic Hard Assertions** with **Adversarial Agent Review**:
1. **Hard Assertion 1 — Target Memorization & Leakage Veto:**
   - VETO if $\text{Metric}_{\text{train}} - \text{Metric}_{\text{val}} > 0.12$ (or if $\text{Metric}_{\text{train}} > 0.99$ while $\text{Metric}_{\text{val}} < 0.75$).
   - VETO if any single derived feature accounts for $> 65\%$ of total model split gain AND originates from a `QUASI_ID_DECOMPOSE` or high-cardinality column.
2. **Hard Assertion 2 — Unexploited Signal Veto:**
   - Inspects `dataset_topology.json` vs. `features.py`. If any column classified as `SHORT_TEXT_EMBED` (e.g., `Name`) or `QUASI_ID_DECOMPOSE` (e.g., `Ticket`, `Cabin`) was completely dropped without extracting sub-tokens, prefixes, or group aggregations, VETO with remediation: *"Column `<col>` was dropped without decomposition/embedding."*
3. **Hard Assertion 3 — Model Monoculture Veto:**
   - VETO if fewer than 3 distinct model families have been evaluated in Tier 1 before attempting final packaging.
4. **Hard Assertion 4 — Topology & Imbalance Veto:**
   - VETO if `entity_group_col` or `temporal_col` is non-null in `dataset_topology.json` but standard random `KFold`/`StratifiedKFold` was used.
   - VETO if validation balanced accuracy $\le 0.52$ on classification or $R^2 \le 0.01$ on regression.
5. **Output (`CritiqueReceipt`):**
   - Status: `"APPROVED"` or `"VETO"`.
   - `blocking_flaws`: List of exact architectural or statistical failures.
   - `error_slice_hypotheses`: Concrete recommendations for M1 to task M2-FE or M2-Model in the next epoch based on the `ErrorForensicsReport`.

---

## 5. BULLETPROOF SCIENTIFIC PRIMITIVES (`scientific/primitives.py`)

Implement these reusable, scikit-learn `BaseEstimator` / `TransformerMixin` compatible classes so `M2-FE` can import them in `features.py` with zero data leakage and seamless `joblib` serialization into `mlrescurix.bundle.v2`:

### 5.1. `NestedOOFEmpiricalBayesEncoder`
Prevents target encoding memorization:
- During `fit_transform(X_train, y_train)`:
  - Splits `(X_train, y_train)` into $K_{\text{inner}} = 5$ stratified/grouped folds.
  - For each inner fold, computes category posterior statistics strictly from the other $K_{\text{inner}}-1$ folds using **Empirical Bayes shrinkage**:
    $$\hat{y}_c = \frac{n_c \bar{y}_c + m \cdot \bar{y}_{\text{global}}}{n_c + m}$$
    where smoothing weight $m = \max\left(10.0, \frac{\sigma^2_c}{\tau^2_{\text{global}}}\right)$.
  - Adds small deterministic Gaussian noise $\mathcal{N}(0, 0.01 \cdot \text{std}(y))$ during training transform to prevent trees from isolating exact float values of low-count categories.
- During `transform(X_val / X_test)`:
  - Maps categories using the full training fold posterior lookup table; unseen categories or categories with $n_c < 2$ smoothly map to $\bar{y}_{\text{global}}$.

### 5.2. `QuasiIdDecomposer`
Extracts group and structural signal from columns like `Ticket`, `Cabin`, `ZipCode`, `DeviceID` while dropping the raw string:
- **Group Size / Frequency Feature:** `<col>__group_size` (number of rows sharing the exact value across the dataset or training fold).
- **Group Shared Covariate Aggregations:** If numeric columns exist (e.g., `Fare`), computes `<col>__<num>_mean`, `<col>__<num>_per_member = <num> / <col>__group_size`.
- **Structural Prefix & Mask Extraction:** Extracts leading alphabetic prefix (`"PC 17599"` $\rightarrow$ `"PC"`, `"C85"` $\rightarrow$ `"C"`), structural mask (`"AA-####"`), string length, and digit ratio, then routes the low-cardinality prefix to `NestedOOFEmpiricalBayesEncoder`.

### 5.3. `LightweightTextEmbedder`
Extracts signal from `SHORT_TEXT_EMBED` columns (`Name`, `JobTitle`, `Description`) at zero LLM token cost:
- **Sub-Token Indicator Extraction:** Tokenizes on `[\s,\.;:/\(\)\[\]\-_]+`, selects the top 25 recurring tokens with highest Mutual Information with the target $y$ on `X_train` (deterministically capturing honorifics like `"Mr"`, `"Mrs"`, `"Miss"`, `"Master"`, `"Dr"` or key domain words), and creates binary/categorical features.
- **Surname / Prefix Cluster Size:** Extracts the leading comma-separated or whitespace token (e.g., family surname in `Name`) and computes cluster size (`family_surname_count`).
- **In-Fold Char + Word TF-IDF SVD:** Fits `FeatureUnion([TfidfVectorizer(analyzer="word", ngram_range=(1,2), max_features=2500), TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=2500)])` followed by `TruncatedSVD(n_components=16, random_state=42)` strictly inside the training fold.

### 5.4. `InformativeMissingnessAndInteractions`
- Creates binary `<col>__is_nan` indicators for every column with missingness rate $\in [0.5\%, 99.5\%]$.
- Generates candidate pairwise numeric interactions ($A / (\vert{}B\vert{} + \epsilon)$, $A \times B$, $A - B$) among the top 8 numeric features, scores them via fast Mutual Information (`mutual_info_classif` / `mutual_info_regression`) on the training fold, and retains only the top $K \le 10$ interactions that exceed baseline parent MI.

---

## 6. LEAKAGE-PROOF TOPOLOGY & TIER-0 RARE-EVENT SAMPLING (`candidate_runner.py` & `iteration_loop.py`)

### 6.1. Automatic CV Splitter Selection
Read `dataset_topology.json`:
1. If `temporal_col` is present:
   - Sort dataset chronologically by `temporal_col`.
   - Use **`PurgedTimeSeriesSplit(n_splits=5, embargo_pct=0.02)`** so rolling window features never leak across the train/validation boundary.
2. Else if `entity_group_col` is present (or a quasi-ID column exhibits high target homogeneity across repeated keys):
   - Use **`StratifiedGroupKFold(n_splits=5)`** for classification or **`GroupKFold(n_splits=5)`** for regression.
3. Else:
   - Use **`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`** for classification (or binned-target `StratifiedKFold` for skewed continuous regression).

### 6.2. Topology-Preserving Tier-0 Proxy Sampler ($N \le 500$, $\le 15\text{s}$)
Upgrade `MultiFidelityIterationEngine` so Tier 0 never fails on imbalanced, grouped, or temporal datasets:
- **Rare-Event Floor (`Minimum-Minority Stratified Subsampling`):** If classification minority class prevalence $p_{\min}$ yields fewer than $k_{\min} = 30$ minority samples in a 500-row sample, retain up to $\min(N_{\text{minority}}, 100)$ minority samples and sample $400$ majority samples, passing `sample_weight` / prior-adjustment offset so probabilities remain calibrated to the true prior.
- **Group-Preserving Sampling:** Sample whole `entity_group_col` clusters rather than individual rows.
- **Temporal Tail Slicing:** For time-series datasets, take the most recent contiguous $N = 1,000$ rows rather than a random uniform sample.

---

## 7. THE M1 REFLEXIVE ORCHESTRATION LOOP (`autonomous_rescue.py` & `crates/harness/src/agents/`)

Implement the main autonomous loop in `AutonomousRescueEngine.run()` and synchronize every transition to `state.sqlite`:

```python
# Pseudocode Specification for AutonomousRescueEngine.run()
def run_agent_first_loop(
    dataset_uri: str, user_prompt: str, budget: RescueBudget
) -> BundleArtifact:
    # 1. PHASE 1: Exploratory Data Science & Topology Lock (M2-EDA)
    eda_dossier = m2_eda_agent.execute_forensics(dataset_uri, user_prompt)
    topology = lock_dataset_topology_contract(eda_dossier)  # Persisted to state.sqlite
    ledger = ResearchLedger.initialize(topology, eda_dossier)

    # 2. PHASE 2: Baseline Anchor (Epoch 0)
    baseline_fe = m2_fe_agent.synthesize_safe_baseline(topology)
    baseline_eval = m2_model_agent.evaluate_families(
        families=["lgbm", "catboost", "penalized_linear"],
        features_py=baseline_fe,
        topology=topology,
        tier="tier1",
    )
    ledger.record_epoch(epoch=0, fe=baseline_fe, eval_result=baseline_eval)

    # 3. PHASE 3: Reflexive Hypothesis -> Ablation -> Modeling -> Critic Loop
    for epoch in range(1, budget.max_epochs + 1):
        # M1 examines EDA Dossier, Error Forensics (worst slices), and any previous Critic Veto
        hypotheses = m1_orchestrator.propose_hypotheses(ledger)

        # M2-FE runs fast paired ablations in isolated git worktrees (.m2/worktrees/epoch_{epoch})
        accepted_fe = m2_fe_agent.run_ablation_sprint(
            base_features_py=ledger.best_features_py,
            hypotheses=hypotheses,
            topology=topology,
        )

        # M2-Model trains diverse model families & generates ErrorForensicsReport
        model_results = m2_model_agent.train_and_diagnose(
            families=m1_orchestrator.select_model_families(ledger, topology),
            features_py=accepted_fe,
            topology=topology,
        )

        # M2-Ensemble blends Pareto-optimal models (SLSQP + Rank Averaging) & tunes OOF threshold
        ensemble_candidate = m2_ensemble_agent.blend_and_calibrate(
            pareto_candidates=ledger.get_pareto_frontier(model_results),
            topology=topology,
        )
        ledger.record_epoch(epoch=epoch, fe=accepted_fe, eval_result=ensemble_candidate)

        # M2-Critic performs Adversarial Audit (Hard Assertions + LLM Staff Review)
        critique: CritiqueReceipt = m2_critic_agent.audit(
            candidate=ensemble_candidate,
            topology=topology,
            eda_dossier=eda_dossier,
            ledger=ledger,
        )
        ledger.record_critique(epoch=epoch, critique=critique)

        if critique.status == "APPROVED":
            break
        else:
            # M1 logs structured reflection on Critic's blocking_flaws & error_slice_hypotheses
            m1_orchestrator.reflect_and_update_steering(ledger, critique)

    # 4. PHASE 4: Zero-Skew Packaging & Two-Key Cryptographic Verification
    bundle = bundle_candidate(ledger.apex_candidate, topology)
    verify_bundle_strict(bundle, raw_dataset_sample=eda_dossier.raw_holdout_vectors)
    return bundle
```

---

## 8. OUT-OF-CORE EXECUTION & `mlrescurix.bundle.v2` ZERO-SKEW PACKAGING

1. **Polars / DuckDB Memory Guard (`scientific/out_of_core.py`):**
   - Before loading any file, inspect file size and container `memory.max` (`cgroup v2`).
   - If estimated in-memory footprint $> 35\%$ of available container RAM, execute `dirty_sanitizer` and `semantic_sieve` via **Polars LazyFrame (`pl.scan_csv` / `pl.scan_parquet`)** and **DuckDB out-of-core SQL aggregations** (`approx_count_distinct`, streaming quantiles).
   - For training when row count $> 1,000,000$: retain $100\%$ of minority class instances (if imbalanced) and apply stratified/grouped reservoir downsampling to fit safely within 50% of container memory, adjusting log-odds predictions via prior shift calibration.
2. **Self-Contained `mlrescurix.bundle.v2` Pipeline Graph:**
   - `frozen_model.joblib` must encapsulate a unified `FullInferencePipeline` object containing:
     `DirtyDataSanitizer` $\rightarrow$ `ColumnDropper(id_cols)` $\rightarrow$ `FeatureEngineeringGraph` (`NestedOOFEmpiricalBayesEncoder`, `QuasiIdDecomposer`, `LightweightTextEmbedder`, `InformativeMissingnessAndInteractions`) $\rightarrow$ `ParetoEnsemblePredictor` $\rightarrow$ `CalibratedThresholdCategorizer`.
   - `inference.py` and `serve.py` must accept **raw, uncleaned JSON/CSV rows** (including `PassengerId`, raw `Name`, `"$1,200"`, `"?"`) and produce identical predictions to CV evaluation within $10^{-6}$ float tolerance.

---

## 9. PHASED EXECUTION CHECKLIST & VERIFICATION GATES

Execute implementation in the following order, running `cargo test` and `pytest` after each phase:

- [x] **Step 1 (Schemas & Storage):** Create `dataset_topology.schema.json`, `research_ledger.schema.json`, `critique_receipt.schema.json`, and `error_forensics.schema.json` in `contracts/v2/`. Migrate `crates/harness/src/storage.rs` to `user_version = 11`.
- [x] **Step 2 (Sanitizer, Semantic Sieve & M2-EDA):** Implement `dirty_sanitizer.py`, `semantic_sieve.py`, and `eda_agent.py`.
  - *Verification Test:* Run `profile-dataset` on `titanic.csv` and a synthetic dirty dataset (`"$1,200"`, `"15%"`, `"-999"`, shuffled target column index). Assert `Survived` is chosen as target (never `Embarked`), `PassengerId` is classified as `DROP_SURROGATE_ID`, `Name` as `SHORT_TEXT_EMBED`, `Ticket` and `Cabin` as `QUASI_ID_DECOMPOSE`, and `EdaDossier` size is $< 4\text{ KB}$.
- [x] **Step 3 (Leakage-Proof Primitives & M2-FE Ablation Engine):** Implement `primitives.py` (`NestedOOFEmpiricalBayesEncoder`, `QuasiIdDecomposer`, `LightweightTextEmbedder`, `InformativeMissingnessAndInteractions`) and `fe_ablation.py`.
  - *Verification Test:* Feed a column of 100% unique random strings into `NestedOOFEmpiricalBayesEncoder` + `LightGBM`. Assert validation balanced accuracy does NOT collapse to `0.5000` and train-val gap remains $< 0.05$.
- [x] **Step 4 (Topology-Aware CV, Multi-Paradigm Zoo & Error Forensics):** Implement `model_zoo.py`, `ErrorForensicsAnalyzer`, and upgrade `candidate_runner.py` & `iteration_loop.py` with `StratifiedGroupKFold`, `PurgedTimeSeriesSplit`, and Tier-0 Minimum-Minority Stratified Subsampling.
  - *Verification Test:* Test Tier-0 sampling on a synthetic dataset with $0.1\%$ positive rate ($N = 50,000$). Assert Tier-0 sample ($N = 500$) contains $\ge 25$ positive instances and completes in $< 15\text{s}$.
- [x] **Step 5 (M2-Critic, M1 Reflexive Loop & Bundle V2 Parity):** Implement `critic_engine.py`, wire the reflexive loop in `autonomous_rescue.py` and `crates/harness/src/agents/`, and update `verify.rs`.
  - *End-to-End Acceptance Test:* Run full autonomous rescue on `titanic.csv`. Assert:
    1. M2-EDA locks `Survived` as target and extracts `Name` honorifics + `Ticket` group features.
    2. If a naive candidate is injected in Epoch 1, `M2-Critic` issues a `VETO` (`CritiqueReceipt`) and M1 self-corrects in Epoch 2.
    3. Final Pareto ensemble blends $\ge 2$ distinct model families, achieves CV Balanced Accuracy $\ge 81.5\%$, and passes `verify-bundle` on raw uncleaned test records.