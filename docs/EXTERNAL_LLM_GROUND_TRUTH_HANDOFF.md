# MLRescurix Ground-Truth Architectural & Empirical Handoff

**Target Document:** `docs/EXTERNAL_LLM_GROUND_TRUTH_HANDOFF.md`  
**Date:** November 2, 2026  
**Subject:** Ground-truth system review, verified run analysis (Amazon Runs 6, 7, and 8), architectural bottlenecks, and request for independent external plan  
**Audience:** Independent External LLM / Senior Systems & Machine Learning Architect  
**Repository State:** Inspected at commit `95a35e9` with active working-tree modifications  

**Review note (November 2):** This packet combines historical run artifacts and the current working tree; they are different snapshots. The trace timestamps below have been corrected. Claims about billing, privacy, data corruption, and causes of limited exploration are qualified. Local paths are evidence references; external reviewers cannot open them unless files are supplied. Ledger figures are estimates or unresolved liabilities, not proof of provider charges. No provider reconciliation or fresh live run was performed for this review.

---

## Executive Summary & System Review Context

This document is an evidence-based review packet prepared for an independent external LLM. It details the ground-truth architecture, empirical execution history, operational accounting, and failure modes of **MLRescurix**—an autonomous machine learning system designed to independently explore, evaluate, and produce apex machine learning models under strict governance.

Prior internal agent reports have at times overstated capability, conflated planned features with live execution, or reported inaccurate metrics. It draws on source code, run traces, transaction ledgers, agent logs, and bundle scorecards. Some interpretations and historical details remain unverified; the reviewer should challenge them.

---

## 1. User Goals & Desired Autonomy

### 1.1 The Ultimate Objective
The user's goal is an autonomous ML campaign owner capable of accepting:
1. A raw problem statement and dataset.
2. An authorized evaluation contract (metric, direction, validation topology).
3. Configurable compute and operational budgets.

From this input, the system must autonomously:
- Research domain literature and optimal modeling strategies.
- Formulate and iteratively explore diverse, orthogonal hypotheses (data transformations, feature representations, algorithmic families, neural architectures, loss functions, hyperparameter spaces, and ensemble strategies).
- Reject overfit or ungeneralizable candidates via objective validation checks.
- Produce production-ready, verified model bundles containing native weights, frozen preprocessors, deterministic inference entrypoints, and full provenance documentation.

### 1.2 Desired Operational Policy vs. Legacy Micro-Ceilings
A core design friction exists between the user's desired operational model and the legacy implementation:
- **User's Desired Autonomy Policy:** The user desires **configurable, initially generous operational ceilings** that give the agent freedom to explore without premature artificial halts:
  1. **Total Model Tokens:** e.g., 5,000,000 tokens across prompt and completion calls.
  2. **Total Kaggle Compute:** e.g., 30.00 accelerator hours for heavy training.
  3. **Total Web Search Queries:** e.g., 500 queries to credible external research sources.
- **Legacy Implementation Reality:** The existing execution harness retains a rigid, low-headroom micro-spending ceiling ($0.28 total campaign cap, $0.05 per-run cap in `docs/v2/POLICY_AMAZON_LIVE_20261102.json` and `governance.py`) and a turn limit (`--max-turns 4` or `15`). Even when the user configured 5,000,000 tokens in `config/operational_budgets.json`, Run 6 halted after 9 turns because the uncoordinated $0.05 per-run financial ceiling was tripped.

### 1.3 Campaign Headroom Depletion Crisis
In `docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json`:
- **Campaign Ceiling:** $0.280494020
- **Total ledger `billed_usd` (may include estimates):** $0.084074020
- **Total Uncertain USD:** $0.186040000
- **Remaining Campaign Headroom:** **$0.010380000** (~1 cent!)
The settlement path retains unused reservations for unverified charges as `uncertain_usd`; the inspected authority exposes no automatic release of terminal entries. Over 66% of the ceiling is unresolved liability, not proven phantom charges. Remaining headroom may severely limit another run, depending on reservations and policy. Reconciliation must be evidence-based; do not reset or silently release the ledger.

The system must harmonize these budget authorities: preserving strict accounting and safety controls while granting the autonomous agent sufficient turns and resources to conduct meaningful exploration.

---

## 2. Architecture & Data-Flow Map

The platform is structured as a dual-tier system separating governance/authority (Rust) from scientific execution (Python):

```
+---------------------------------------------------------------------------------------------------+
|                                  RUST HARNESS (Authority Tier)                                    |
|  - Role-based Gateway Dispatcher & Outbound Data Redaction (gateway.rs)                          |
|  - Cryptographic Transaction Ledger & Price Caps (campaign_budget_ledger.json)                   |
|  - Custom Protocol Split Management & Lockbox Carving (custom_protocol.rs)                       |
|  - Trusted Independent Evaluator (trusted_evaluation.rs: SMAPE, ROC-AUC, LogLoss, RMSLE)         |
|  - Cryptographic Bundle Packaging & Integrity Verifier (bundle.rs)                                |
+---------------------------------------------------------------------------------------------------+
                                         |  JSON-RPC / Subprocess / Staged Files
                                         v
+---------------------------------------------------------------------------------------------------+
|                            PYTHON WORKER (Scientific Execution Tier)                              |
|                                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   | AutonomousAgentLoop (autonomous_loop.py)                                                  |   |
|   |  - State Machine: turn-based LLM prompt / action execution loop                           |   |
|   |  - Actions: ANALYZE, HYPOTHESIS_TRANSFORM, EVALUATE_PORTFOLIO, BLEND_AND_AUDIT,           |   |
|   |             RESEARCH_TOPIC, WRITE_PYTHON, RUN_PYTHON, FINALIZE_AND_STOP, HANDOFF          |   |
|   +-------------------------------------------------------------------------------------------+   |
|               |                                                   |                               |
|               v                                                   v                               |
|   +---------------------------------------+   +-----------------------------------------------+   |
|   | Candidate Pipeline (candidate_pipeline.py)|   | Governed Research (governed_research.py)      |   |
|   |  - Transforms: TF-IDF SVD, Skew,      |   |  - Providers: Tavily, SerpApi, Brave, Mock    |   |
|   |    Interactions, MNAR, Frequency      |   |  - Privacy Sanitizer: pattern-based checks,  |   |
|   |  - Estimators: CatBoost, LightGBM,    |   |    not a guarantee against data leakage          |   |
|   |    XGBoost, HistGBDT, Ridge, TabularNN|   |  - Context Firewall: markdown < 2048 bytes    |   |
|   +---------------------------------------+   +-----------------------------------------------+   |
|               |                                                                                   |
|               v                                                                                   |
|   +-------------------------------------------------------------------------------------------+   |
|   | Critic Engine (critic_engine.py)                                                          |   |
|   |  - Checks train-validation generalization gap (vetoes candidates exceeding gap threshold)  |   |
|   |  - Evaluates feature dominance and multi-family diversity                                 |   |
|   +-------------------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
```

### End-to-End Execution Trace
1. **Intake & Staging:** Master dataset staged at `.mlrescurix-state/<project>/staged/development.csv`. Protocol freezes development folds (`development_folds.json`).
2. **Turn Dispatch:** `AutonomousAgentLoop` compiles sliding window of the last 8 turns and sends request through Rust Gateway to the LLM (`deepseek/deepseek-v4.1-flash-1017`).
3. **Outbound Redaction:** Gateway applies outbound checks and redaction; these should not be treated as proof that all sensitive content is blocked.
4. **Action Execution:**
   - `HYPOTHESIS_TRANSFORM`: Evaluates feature engineering on paired folds; updates active pipeline if CV score improves.
   - `EVALUATE_PORTFOLIO`: Fits declared model families on fold train splits, generates out-of-fold predictions (`.oof.npy`), computes validation metrics.
   - `BLEND_AND_AUDIT`: Optimizes or evaluates weighted combinations across distinct model families.
5. **Independent Controller Verification:** The controller independently loads raw `.oof.npy` predictions and computes primary metrics against locked ground-truth labels.
6. **Critic Audit:** `CriticEngine` audits train vs. validation score difference. If overfit gap exceeds threshold, it issues a `VETO`.
7. **Apex Promotion:** If `APPROVED` and CV score strictly beats previous best eligible score, candidate is promoted to Apex.
8. **Delivery:** `verify-bundle` verifies bundle integrity, schema conformance, and deterministic inference bitwise parity.

---

## 3. Stage-by-Stage Workflow: Intended vs. Implemented Reality

| Lifecycle Stage | Intended Autonomous Design | Actual Implemented Code Reality | Gap / Vulnerability |
|:---|:---|:---|:---|
| **1. Planning** | Dynamic DAG / hierarchical task graph, branching search, backtracking | Linear turn loop (`autonomous_loop.py`) with 8-turn sliding history window; no persistent graph or branching | No explicit persistent search graph; the effect on long-horizon planning needs further evaluation |
| **2. Research** | Autonomous literature search, loss formulation, model ideas from papers | `GovernedResearchClient` exists with Tavily/SerpApi/Brave/Mock, sanitizers, and Context Firewall (<2KB) | **0 recorded queries in Runs 6–8.** Omitted from `ACTIONS` on line 51 of `autonomous_loop.py` (crashes if invoked). Failed queries unbilled |
| **3. Hypothesis Selection** | Unconstrained hypothesis formulation across features, models, losses, and architectures | Hardcoded 6 transforms in `candidate_pipeline.py` (lines 80–135); 6 estimators in `model_zoo.py` | Any novel transform throws `ValueError: Unsupported transform`. Text locked to 16-component SVD |
| **4. Data Processing** | Adaptive preprocessing, deep text tokenization, embeddings, learned representations | Fixed pipeline: skew transforms, interactions, frequency encoding, 16-component TF-IDF TruncatedSVD | No subword tokenizers, pretrained embeddings, transformers, or audio/image feature pipelines |
| **5. Model Selection** | Multi-family zoo, modern tabular neural networks (FT-Transformer, TabNet), custom architectures | 4 GBDT libraries, Ridge regression, and a single 2-block Residual MLP (`_TabularResNetModule`) | No FT-Transformers, no TabNet, no custom PyTorch neural architectures; prompt enumerates strict allowed parameters |
| **6. Training** | Remote Kaggle GPU/TPU offloading or sandboxed local container compute | Local CPU subprocesses (`candidate_worker.py`) using single thread (`OMP_NUM_THREADS="1"`) | Kaggle never used (0.00h). 30h budget counter disconnected from Kaggle account quota |
| **7. Evaluation** | Robust nested CV or train/validation/holdout partition with leak-free scoring | 2 pinned folds (37,500 rows each) repeatedly evaluated across all turns; 0 holdout | Risk of adaptive selection bias; 525,209 source rows were not used in these runs |
| **8. Experiment Tracking** | Cryptographic ledgers, candidate hashes, out-of-fold receipts, audit logs | SHA-256 candidate identities, `.oof.npy` predictions, controller recomputation event | In Run 6, resume logic dropped turns 1–8 tokens (1,537 vs 60,302); CatBoost score misreported as winner |
| **9. Retries & Recovery** | Diagnostic error diagnosis, hyperparameter repair, anti-plateau recovery | Unhandled exceptions caught as `rejected` turns; no autonomous steering | In Run 8, agent wasted 50% of turns on `C` and `penalty` crashes. In Run 6, stuck in 7-turn veto loop |
| **10. Stopping** | Convergence detection, Pareto optimality, operational budget exhaustion | Hardcoded turn limits (`--max-turns 4`), micro-spending ceiling ($0.05 per run) | Run 6 choked after 9 turns by $0.05 cap. Run 8 stopped at turn 4 by CLI arg |
| **11. Handoff & Packaging** | Self-contained, portable, reproducible bundle verified on held-out data | `verify-bundle` verifies manifest, hashes, and inference bitwise parity on sample rows | Verifies packaging integrity only; explicitly records `production_certified: false` |

---

## 4. Timeline of Live Amazon Runs & Empirical Comparison

Three live autonomous runs were conducted on the 75,000-row Amazon ML Challenge catalog dataset using DeepSeek V4.1 Flash via OpenRouter `wafer/us`.

### Verified Comparative Run Metrics Table

| Metric / Dimension | Amazon Run 6 (`live-20261102-run6`) | Amazon Run 7 (`live-20261102-run7`) | Amazon Run 8 (`live-20261102-run8`) |
|:---|:---|:---|:---|
| **First–last trace event (UTC)** | 2026-11-02 06:32–06:46 UTC | 2026-11-02 09:51–09:58 UTC | 2026-11-02 10:30–10:35 UTC |
| **Model & Route** | DeepSeek V4.1 Flash via `wafer/us` | DeepSeek V4.1 Flash via `wafer/us` | DeepSeek V4.1 Flash via `wafer/us` |
| **Turns Planned / Configured** | `--max-turns 15` (current loop default: no turn limit) | `--max-turns 4` | `--max-turns 4` |
| **Turns Executed / Completed** | 9 turns (Turn 0 to Turn 8) | 4 turns (Turn 0 to Turn 3) | 4 turns (Turn 0 to Turn 3) |
| **Stopping Reason** | `budget_exhaustion` (Per-run $0.05 cap) | `max_turns_reached` | `max_turns_reached` |
| **Model Tokens Used (Actual Receipts)** | **60,302 tokens** (9 receipts) | **21,392 tokens** (4 receipts) | **20,413 tokens** (4 receipts) |
| **Model Tokens (Reported in Handoff)** | **1,537 tokens** *(Defect: restored only Turn 0)* | 21,392 tokens *(Correct)* | 20,413 tokens *(Correct)* |
| **Configured Token Limit** | 5,000,000 tokens | 5,000,000 tokens | 5,000,000 tokens |
| **Kaggle Accelerator Compute Used** | 0.00h (Local Apple Silicon CPU) | 0.00h (Local Apple Silicon CPU) | 0.00h (Local Apple Silicon CPU) |
| **Kaggle Compute Enforcement** | `enforced: False` (Not connected) | `enforced: False` (Not connected) | `enforced: False` (Not connected) |
| **Web Search Queries Used** | 0 queries (Provider unconfigured) | 0 queries (Provider unconfigured) | 0 queries (Provider unconfigured) |
| **Web Search Enforcement** | `enforced: False` (Not connected) | `enforced: False` (Not connected) | `enforced: False` (Not connected) |
| **Session ledger cost (estimated/unverified USD)** | $0.00619 | $0.00174 | $0.00200 |
| **Session Uncertain / Reserved USD** | $0.03872 | $0.01034 | $0.00710 |
| **Campaign Remaining Headroom** | $0.031560000 | $0.019480000 | **$0.010380000** |
| **Selected Winning Candidate ID** | `cand_4dd30caf643a04af52e06c86` | `cand_accf1e986a6d4ee90a3fcfb0` | `cand_33a0c4a2d20123c09e3d7020` |
| **Winning Model Family** | `neural_tabular` (Tabular ResNet) | `neural_tabular` (Tabular ResNet) | `neural_tabular` (Tabular ResNet) |
| **Claimed Score in Run Handoff** | 62.48337930596179 *(Wrong candidate)* | 62.46631892131646 *(Exact match)* | 62.77886850142577 *(Exact match)* |
| **True Selected Recomputed SMAPE** | **62.46631892131646** | **62.46631892131646** | **62.77886850142577** |
| **Score Discrepancy** | **+0.0170603846** (Turn 8 CatBoost) | **0.0000000000** (Resolved) | **0.0000000000** (Resolved) |
| **Bundle Verification Status** | `integrity_verified` (896 KB, 19 files) | `integrity_verified` (895 KB, 19 files) | `integrity_verified` (895 KB, 19 files) |
| **Inference Bitwise Parity** | Passed (100% exact match) | Passed (100% exact match) | Passed (100% exact match) |
| **Holdout Evaluation** | **None** (75k development rows only) | **None** (75k development rows only) | **None** (75k development rows only) |

---

## 5. Action Trace Analysis: Run 6 vs. Run 8

### 5.1 Amazon Run 6 Turn-by-Turn Trace
- **Turn 0 (HYPOTHESIS_TRANSFORM):** Extracted TF-IDF SVD features (16 components) on `catalog_content`. CV SMAPE improved from 80.5682 to 74.9141 (confirmed, pipeline updated).
- **Turn 1 (EVALUATE_PORTFOLIO):** Evaluated portfolio across 5 families. `neural_tabular` achieved CV SMAPE **62.4663189213** with train-validation gap 2.91%. Approved by CriticEngine and promoted to Apex #1.
- **Turns 2–8 (Repetitive Overfitting Trap):** Across 7 consecutive turns, the agent repeatedly submitted minor hyperparameter tweaks of `neural_tabular` (varying learning rate from 0.001 to 0.0005, dropout, etc.):
  - Turn 2 (Iteration 3): CV 61.7489, Gap 5.21% -> **VETO**
  - Turn 3 (Iteration 4): CV 61.7489, Gap 5.21% -> **VETO**
  - Turn 4 (Iteration 5): CV 61.7296, Gap 7.18% -> **VETO**
  - Turn 5 (Iteration 6): CV 61.7489, Gap 5.21% -> **VETO**
  - Turn 6 (Iteration 7): CV 61.7296, Gap 7.18% -> **VETO**
  - Turn 7 (Iteration 8): CV 61.7296, Gap 7.18% -> **VETO**
  - Turn 8 (Iteration 9): CV 61.7489, Gap 5.21% -> **VETO**; also evaluated CatBoost scoring 62.4834.
- **Turn 9 (Dispatch Attempt):** Halted by `BudgetExhaustionError: Per-run spending ceiling exhausted` because committed spend ($0.04491) plus reservation ($0.00788) exceeded $0.05.

### 5.2 Amazon Run 8 Turn-by-Turn Trace
- **Turn 0 (HYPOTHESIS_TRANSFORM):** Applied TF-IDF SVD on `catalog_content` (CV 74.9141, confirmed).
- **Turn 1 (EVALUATE_PORTFOLIO):** LLM attempted to pass linear classification hyperparameter `C` to `penalized_linear`. In `candidate_pipeline.py`, Ridge regression threw `ValueError: Hyperparameter C is unsupported for this penalized_linear task`. Disposition: **rejected**. Turn and tokens wasted.
- **Turn 2 (EVALUATE_PORTFOLIO):** LLM attempted to pass `penalty` to `penalized_linear`. Threw `ValueError: Hyperparameter penalty is unsupported for this penalized_linear task`. Disposition: **rejected**. Second consecutive turn wasted.
- **Turn 3 (EVALUATE_PORTFOLIO):** Evaluated portfolio without invalid linear parameters. `neural_tabular` achieved CV SMAPE **62.77886850142577**. Approved by CriticEngine and promoted to Apex #1.
- **Turn 4 (Premature Stop):** Stopped immediately with `stop_reason: max_turns_reached` because `--max-turns 4` was passed in `run_live_amazon.sh`.
- **Master Plan False Assertion:** `FINAL_AUTONOMOUS_MODEL_CREATOR_MASTER_PLAN.md` asserted that Run 8 was executed *following repairs to linear model hyperparameter tolerance*. This is false: the patch in `candidate_pipeline.py` (lines 179–188) was added *after* Run 8, and has **never been tested in a live Amazon run**.

---

## 6. Key Evidence & Root Cause Analysis of Discovered Defects

### 6.1 Defect 1: The Score Provenance Mismatch in Run 6
- **Evidence Files:**
  - `.mlrescurix-state/amazon-2025/live-20261102-run6/HANDOFF.md`: `Controller Recomputed Score: 62.48337930596179`
  - `.mlrescurix-state/amazon-2025/live-20261102-run6/bundle/scorecard.json`: `primary_metric: 62.46631892131646`
  - `.mlrescurix-state/amazon-2025/live-20261102-run6/evaluations/4ef0d33237e0ed5264230986016811b9e26d663c531d8516c5e4c6044747eb6e.result.json`: `cv_score: 62.48337930596179`
- **Root Cause in Code:** In `autonomous_loop.py`, the old `export_handoff()` wrote:
  ```python
  f"Controller Recomputed Score: {self.state.get('latest_controller_verified_score')}"
  ```
  This recorded whatever candidate happened to be evaluated last on Turn 8 (a CatBoost candidate scoring 62.4834), rather than binding to the accepted candidate `cand_4dd30caf...` (neural tabular scoring 62.4663).
- **Resolution:** Patched via `_selected_controller_verified_score()`. Verified in Run 7 and Run 8 (0.00000 discrepancy).

### 6.2 Defect 2: Token Accounting Truncation on Checkpoint Reload
- **Evidence Files:**
  - `HANDOFF.md` in Run 6 reported: `1,537 used / 5,000,000 limit`.
  - Actual receipts in `autonomous_trace.jsonl` totaled **60,302 tokens**.
- **Root Cause in Code:** `OperationalBudgets.restore_token_usage()` originally processed only `turns[0]`, discarding turns 1–8.
- **Resolution:** Rebuilt in `operational_budgets.py` (`restore_operational_budgets()`) iterating every turn. Verified in Run 7 and 8.

### 6.3 Defect 3: Fragile Parameter Handling Crippled Run 8
- **Evidence Files:**
  - `.mlrescurix-state/amazon-2025/live-20261102-run8/autonomous_trace.jsonl`:
    `turn_index: 1, disposition: rejected, measurements: {"error": "ValueError: Hyperparameter C is unsupported for this penalized_linear task"}`
  - `.mlrescurix-state/amazon-2025/live-20261102-run8/autonomous_trace.jsonl`:
    `turn_index: 2, disposition: rejected, measurements: {"error": "ValueError: Hyperparameter penalty is unsupported for this penalized_linear task"}`
- **Root Cause in Code:** In `candidate_pipeline.py`:
  ```python
  key = f"model__{name}" if family == "penalized_linear" else name
  if key not in available:
      raise ValueError(f"Hyperparameter {name} is unsupported for this {family} task")
  ```
  Regression uses `Ridge` (which takes `alpha`), whereas classification uses `LogisticRegression` (which takes `C` and `penalty`). The pipeline threw an unhandled `ValueError`, immediately aborting candidate evaluation for the turn.
- **Untested Working-Tree Patch:** Lines 179–188 were added to map `C` to `model__alpha = 1.0 / max(float(value), 1e-6)` and ignore `penalty`. **No live run has verified this fix.**

### 6.4 Defect 4: Missing Action Dispatch for Governed Web Research
- **Evidence Files:**
  - `python/src/mlrescurix_worker/scientific/autonomous_loop.py`:
    ```python
    ACTIONS = {"HYPOTHESIS_TRANSFORM", "EVALUATE_PORTFOLIO", "BLEND_AND_AUDIT", "FINALIZE_AND_STOP",
        "STOP_EARLY", "ANALYZE", "WRITE_PYTHON", "RUN_PYTHON", "HANDOFF"}
    ```
  - `crates/harness/src/agents/gateway.rs`:
    Rust `ALLOWED_ACTIONS` contains `"RESEARCH_TOPIC"`.
  - `python/src/mlrescurix_worker/scientific/autonomous_loop.py`:
    ```python
    if parsed["action_type"] not in ACTIONS:
        raise ValueError(f"Unknown action type '{parsed['action_type']}'")
    ```
- **Finding:** While `RESEARCH_TOPIC` was documented in the prompt and allowed in Rust's `gateway.rs`, it was **omitted from the Python `ACTIONS` set** at line 51! Any agent returning `RESEARCH_TOPIC` would be immediately rejected with `ValueError: Unknown action type 'RESEARCH_TOPIC'`.
- **Search Accounting Gap:** In `governed_research.py` line 345, search query budget is recorded *only after* a provider returns successfully:
  ```python
  # Line 334: sources = self.provider.search(sanitized_query, max_results=max_results)
  # Line 346: budgets.record_search_query(1)
  ```
  If a search request times out, fails with HTTP 500, or encounters a network error, it is never recorded in the budget ledger.

### 6.5 Defect 5: Unreconciled Reservation Ratchet in Budget Governance
- **Evidence Files:**
  - `python/src/mlrescurix_worker/scientific/governance.py`:
    ```python
    # Line 118: Estimates cannot release unused reservation; provider charges remain unknown.
    retained = max(amount, reserved) if cost_kind != "provider_verified" else amount
    ...
    if operation == "settle":
        change("total_billed_usd", "billed_usd", amount)
        change("total_uncertain_usd", "uncertain_usd", retained - amount)
    ```
- **Finding:** For estimated providers (such as OpenRouter), `cost_kind != "provider_verified"`.
  When a turn settles, `retained` is forced to `reserved`. The unused reservation `(reserved - amount)` is transferred to `uncertain_usd` and **is not automatically released by the inspected settlement path**.
  Because `run_committed = billed_usd + reserved_usd + uncertain_usd`, `run_committed` grows by the full reservation of every turn regardless of how cheap the call was.
  - In Run 6: Recorded estimated cost was **$0.00619**, but `uncertain_usd` reached **$0.03872**. Total committed reached **$0.04491**, tripping the $0.05 per-run cap on Turn 9!
  - In Campaign Ledger: `total_billed_usd` is **$0.08407**, but `total_uncertain_usd` is **$0.18604**! Out of the $0.28049 campaign ceiling, only **$0.01038** remains!

### 6.6 Defect 6: Master Source Data Inode Sharing via Hard Links
- **Evidence Files:**
  - `python/src/mlrescurix_worker/scientific/autonomous_loop.py`:
    ```python
    try:
        os.link(dataset_path, copied)
    except OSError:
        copied.write_bytes(input_bytes)
    ```
- **Finding:** `os.link` creates a filesystem hard link between the staged master dataset `.mlrescurix-state/amazon-2025/staged/development.csv` and the session scratchpad `inputs/development.csv`.
  Both paths share the **exact same inode** (verified: link count is 16, permissions `-rw-r--r--`).
  An in-place write to `inputs/development.csv` can also alter the master staged dataset; a replace-by-rename behaves differently. This is a potential integrity risk, not evidence that corruption occurred. Verify when and where hash checks run.

---

## 7. Why Did the Agent Not Explore Other Paradigms?

Run 6 included an initial multi-family portfolio, followed by repeated neural-model variations; Run 8 had little room for exploration after two rejected turns. The following are plausible contributing factors, not independently established causes:

| Factor | Mechanism / Ground Truth Evidence |
|:---|:---|
| **Rigid Model Zoo** | `model_zoo.py` includes linear models, gradient-boosted trees and a 2-block Residual MLP (`_TabularResNetModule`). No HuggingFace models, no FT-Transformers, no TabNet, no CNNs, no recurrent nets. |
| **Closed Feature Transforms** | `candidate_pipeline.py` (lines 80–135) allows only 6 specific transform names (`ADD_MNAR_INDICATOR`, `DELETE_FEATURE`, `APPLY_SKEW_TRANSFORM`, `ADD_INTERACTION`, `ENCODE_HIGH_CARD_CAT`, `TEXT_TFIDF_SVD`). Any novel feature engineering code raises `ValueError: Unsupported transform`. |
| **Text Processing Bottleneck** | `candidate_pipeline.py` lines 111–134 hardcodes text representation to `TfidfVectorizer(max_features=5000, ngram_range=(1,2))` + `TruncatedSVD(n_components=16)`. No word2vec, fasttext, byte-pair tokenizers, or sentence transformers exist. |
| **System Prompt Anchoring** | The prompt (`autonomous_loop.py` lines 370–384) specifies fixed JSON actions and enumerates explicit allowed parameters. The enumerated template may anchor the model; the trace alone cannot establish why it made each choice. |
| **Critic Veto Trap** | In Run 6, `neural_tabular` achieved 61.72–61.74 SMAPE with a 5.2%–7.2% train-validation gap. The critic vetoed these candidates for overfitting. The current prompt contains advice to pivot after stalled progress; its effectiveness was not established in a subsequent live run. In Run 6, the agent submitted near-identical learning rates for 7 consecutive turns. |
| **Turn and Budget Stranding** | `--max-turns 4` in Run 8 allowed only 1 turn of model evaluation after two syntax crashes. In Run 6, the $0.05 per-run cap killed the process at turn 8, right before the planned ensemble blend. |

---

## 8. Web Research & Kaggle Compute Reality

| Capability | What Exists in Code | Reality in Live Runs | Gaps & Integrity Risks |
|:---|:---|:---|:---|
| **Governed Web Research** | Pluggable providers (`Tavily`, `SerpApi`, `Brave`, `Mock`), regex sanitizers blocking dataset extensions/delimiters, Context Firewall markdown compression (<2 KB). | **0 queries used.** `search_connected: False`. Fails closed cleanly when no API key is present in environment. | - Omitted from `ACTIONS` set in `autonomous_loop.py` (crashes if invoked).<br>- Failed queries not billed to budget.<br>- Regex sanitizers cannot catch semantic rephrasing of proprietary data. |
| **Kaggle Execution** | `kaggle_pool.py`, `kaggle_output_transport.py`: notebook kernel pushing, polling, account leasing. | **0.00 hours used.** `kaggle_connected: False`. Runs executed locally on Apple Silicon CPU. | - 30.00h budget counter in `operational_budgets.py` is not wired to Kaggle API quotas.<br>- Privacy: uploading private datasets to Kaggle exposes data to third-party cloud. |

---

## 9. Validation Strategy & Scientific Integrity Risks

1. **Repeated Selection on Development Folds (Adaptive Overfitting):**
   All hyperparameter tuning and model selection in Runs 6, 7, and 8 were performed on the exact same two pinned folds (37,500 train / 37,500 val). Repeatedly querying the same validation set can produce optimistic selection bias. The apex score of 62.4663 is an in-development metric, not proof of generalization.
2. **Zero Untouched Holdout Evaluation:**
   The source dataset `train.csv` contains **600,209 rows**. `split.json` carved out rows 0–74,999 (75,000 rows) as `development.csv`. The remaining **525,209 rows were not used for evaluation in these runs**, and no test split was reserved from the 75,000 rows. No final blind test was performed. `production_certified: false` is accurately recorded in all handoffs.
3. **Critic Gap Heuristic Limitations:**
   The `CriticEngine` vetoes candidates when $| \text{train\_score} - \text{val\_score} | > \text{threshold}$. While this detects extreme memorization, a small gap on two folds does not guarantee generalization, nor does a larger gap prove that a model with superior validation error is inferior.

---

## 10. Summary of Repository Implementation Status

| Component | Status | Empirical Verification |
|:---|:---|:---|
| **Rust Gateway Routing & Accounting** | Fully Implemented & Tested | Unit tested (`gateway.rs`), live verified in Runs 6, 7, 8 |
| **Trusted Evaluator (Rust)** | Fully Implemented & Tested | Unit tested (`trusted_evaluation.rs`: SMAPE, ROC-AUC, LogLoss) |
| **Bundle Packaging & Verification** | Fully Implemented & Tested | `verify-bundle` CLI verified across Runs 6, 7, 8 |
| **Candidate-Bound Score Handoff** | Fully Implemented & Tested | Verified in Run 7 & 8 (`0.00000` discrepancy) |
| **Sklearn Parameter Tolerance** | Implemented (Working Tree) | Unit tested (`test_candidate_pipeline.py`); **Untested in Live Run** |
| **Governed Web Research Client** | Implemented (Working Tree) | Unit tested (`test_governed_research.py`); **Bug: Missing in `ACTIONS`**; **Untested in Live Run** |
| **Kaggle Remote Execution Broker** | Partially Implemented | Offline tests only; **Untested in Live Run**; **Quota Unenforced** |
| **General Hypothesis Generator** | Not Implemented | System relies on rigid JSON templates in `autonomous_loop.py` |
| **Multi-Stage Validation (Nested CV / Holdout)** | Not Implemented | Fixed 2-fold cross-validation only; 0 holdout |

---

## 11. Key Questions for the Independent Reviewer

An external architectural reviewer should evaluate and address the following questions:

1. **Autonomous Hypothesis Engine:** How should the agent be structured so that it formulates truly orthogonal, creative scientific hypotheses (e.g., custom PyTorch neural architectures, learned tokenizers, contrastive pretraining, domain-specific feature engineering) rather than selecting from a hardcoded enum of 5 scikit-learn/GBDT models?
2. **Loop Recovery & Plateau Breaking:** How can the agent recognize unproductive loops (such as the 7 consecutive vetoed neural tabular runs in Run 6 or repeated parameter syntax crashes in Run 8) and autonomously pivot to a completely different paradigm?
3. **Budget Harmonization & Liability Ratchet:** How should the system reconcile the user's desire for generous operational budgets (5M tokens, 30h Kaggle, 500 search queries) with Rust-level cryptographic financial controls, and specifically how should it prevent the `uncertain_usd` ratchet from exhausting the campaign ceiling on estimated provider calls?
4. **Governed Web Research Integration:** How should search be integrated into the agent's reasoning loop so that it actively seeks solutions to training plateaus, verifies literature, accounts for failed/retried requests, and prevents prompt injection without leaking proprietary dataset schemas?
5. **Generalization & Validation Integrity:** How should the validation protocol be structured to reduce and measure adaptive selection bias across multi-turn exploration without leaking data from an untouched lockbox holdout?
6. **Master Source Data Protection:** How should scratchpad filesystems and worker processes be isolated (e.g. Copy-on-Write, read-only loopback mounts) to guarantee that master source data cannot be mutated via shared inodes or permissions?
7. **Extensibility Beyond Tabular Regression:** How can the framework generalize to computer vision, NLP, time-series forecasting, and multimodal tasks without hardcoding dataset-specific logic?

---

## 12. Ready-to-Copy Prompt for External LLM Review

```text
You are an expert Principal AI Systems Architect and Lead Machine Learning Engineer conducting an independent, critical architectural review of MLRescurix.

MLRescurix is an autonomous, cost-aware dual-tier machine learning framework designed to independently explore, evaluate, and produce apex machine learning models under strict governance.

Please read the ground-truth technical handoff below. Do NOT merely praise or endorse the current design. You are expected to challenge its limitations, diagnose its architectural weaknesses, and propose a concrete, prioritized, implementation plan with verifiable milestones and explicit limitations.

### Review Ground-Truth Summary:
1. Dual-Tier Architecture: Rust owns authority, gateway routing, ledger accounting, protocol freezing, and trusted evaluation. Python implements the scientific execution loop (AutonomousAgentLoop, CandidatePipeline, ModelZoo, CriticEngine).
2. Empirical Live Runs (Amazon Catalog Pricing SMAPE):
   - Run 6: Reached apex CV SMAPE 62.4663 with a Tabular Neural Network (Residual MLP). However, the agent was trapped in a repetitive loop for 7 consecutive turns (turns 2–8) trying minor hyperparameter tweaks that were repeatedly vetoed by the critic for overfitting, until halted by an uncoordinated $0.05 per-run budget ceiling. The handoff also misreported tokens (1,537 vs 60,302 actual) and printed the score of a different candidate (62.4834 CatBoost vs 62.4663 Neural).
   - Run 7: Fixed candidate score binding (0.00000 discrepancy) and cumulative token restoration.
   - Run 8: Reached apex CV SMAPE 62.7789. However, 50% of the turns were completely wasted because the LLM passed linear classification parameters ('C', 'penalty') to Ridge regression, causing unhandled ValueErrors. The run halted at turn 4 due to a hardcoded CLI turn limit.
3. Budget Governance Ratchet:
   - In governance.py, when OpenRouter returns estimated charges, the unused reservation is transferred to uncertain_usd; no automatic release of terminal entries was found in the inspected authority path. Out of a $0.28049 campaign cap, $0.18604 is recorded as uncertain liability, leaving only $0.01038 in headroom.
4. The Search & Remote Compute Reality:
   - Runs 6–8 record zero web searches; their counters were not connected, so the counters alone do not prove absence of all external calls. Recent code created GovernedResearchClient with pattern-based query filtering and Context Firewall compression (<2 KB), but it was accidentally omitted from the loop's ACTIONS set, preventing live dispatch.
   - Runs 6–8 record 0.00h Kaggle compute, with local execution reported; the counter was not connected to Kaggle telemetry. Code exists, but the 30.00h operational budget is not wired to real-time Kaggle account quotas.
5. User Policy Preference: The user desires configurable, initially generous limits (5,000,000 model tokens, 30.00 Kaggle hours, 500 web search queries) that give the agent freedom to explore, while preserving existing financial spend ledgers and cryptographic safety controls.

### Required Deliverables in Your Review:
1. Prioritized Diagnosis & Architectural Redesign:
   - Identify the top structural bottlenecks preventing true autonomy and propose an elegant, decoupled architecture.
2. Unconstrained, Orthogonal Hypothesis Exploration:
   - Propose an extensible mechanism allowing the agent to formulate and execute diverse hypotheses across model families, custom PyTorch neural architectures (e.g., TabNet, FT-Transformers, deep MLPs), tokenization/embeddings, and feature representations, using an extensible, validated plugin or execution contract rather than relying only on fixed enums, with appropriate isolation and reproducibility.
3. Adaptive Autonomy & Plateau Breaking:
   - Design an autonomous steering and recovery strategy that detects repetitive veto loops or unhandled parameter crashes and forces the agent to pivot to orthogonal strategies (feature engineering, ensembling, alternative losses).
4. Governed Research Integration:
   - Specify a complete, privacy-preserving research workflow that allows the agent to consult web literature when stalled, sanitized against data leakage, with resilient accounting for failed/uncertain queries and prompt injection defenses.
5. Unified Resource & Budget Governance:
   - Design a unified budget manager that harmonizes the user's generous operational budgets (tokens, Kaggle hours, search queries) with Rust's transaction ledgers, eliminating surprise halts from low micro-ceilings and resolving the uncertain_usd ratchet.
6. Robust Scientific Validation Strategy:
   - Design a validation framework that prevents adaptive overfitting on development folds during long exploration campaigns while reserving an untouched lockbox holdout for final release assessment; do not assume a holdout alone certifies production readiness.
7. Concrete Acceptance Criteria & Live Test Plan:
   - Provide concrete, phased milestones with specific file changes, pseudocode/contracts, tradeoffs, and what should be verified in a fresh live test.
```
