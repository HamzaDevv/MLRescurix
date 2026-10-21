The **MLRescurix v2.0 Master Plan** is a cohesive, well-scoped systems document. It establishes a strict boundary around **Data Science $\rightarrow$ Model Creation $\rightarrow$ Iterative Improvement $\rightarrow$ Certified Handoff**, while resolving the 4 collisions between MLForge and the MLRescurix core.

Before you begin writing code for Step 1 (`eda_agent.py` + `meta_db.py`), there are **5 subtle implementation-level edge cases** in this blueprint that will surface the moment you run the end-to-end pipeline. Catching and patching these in the specification now will save days of debugging later.

---

### 1. Architectural Scorecard of the v2.0 Master Doc

| Dimension | Rating | Why It Works |
| --- | --- | --- |
| **Scope & Boundary Discipline** | **10 / 10** | Stripping `mlforge-monitor` and `mlforge-prod` eliminates architectural bloat. The harness does one job—delivering a certified, zero-dependency `mlrescurix.bundle.v2`—exceptionally well. |
| **Leakage & Evaluation Integrity** | **10 / 10** | Locking the SQLite Holdout Vault in Phase 1 *before* `eda_agent.py` runs guarantees that Mutual Information, correlation pruning, and `meta_db` priors never peek at the test distribution. |
| **Compute & Token Efficiency** | **9.5 / 10** | Replacing $5\times5$ Nested CV with the **1,000$\times$ OOF Bootstrap CI** + **BOHB / LC-PFN Pruning** cuts compute overhead by $>80\%$ while preserving statistical rigor. |
| **Agent Determinism (DSL)** | **9.0 / 10** | Restricting M1 to the 10-Primitive Typed Action DSL (`hypothesis_executor.py`) prevents syntax hallucinations, though one missing bridge (how `error_profiler` feeds Phase 3) needs a quick schema definition. |
| **Bundle Portability (`model.pkl`)** | **8.5 / 10** | A critical Python `pickle` namespace edge case exists between `primitives.py` and the "zero-dependency" `inference.py` contract (detailed in Edge Case #1 below). |

---

### 2. Five Implementation Edge Cases to Patch Before Coding

#### Edge Case 1: The `model.pkl` Custom Class Unpickling Trap (Phase 6 vs. `primitives.py`)

* **The Conflict:** In Section 7, `mlrescurix.bundle.v2` promises a **zero-dependency** `inference.py` and `serve.py` where `model.pkl` serializes `Sanitizer + Primitives + Model/Stack`. However, in Section 5, the pipeline fits custom classes from `mlrescurix_worker.scientific.primitives` (such as `NestedOOFEmpiricalBayesEncoder`, `QuasiIdDecomposer`, `LightweightTextEmbedder`).


* **What happens at runtime:** Standard `pickle.load()` or `joblib.load()` looks up the exact module path `mlrescurix_worker.scientific.primitives` in `sys.modules`. When an external engineer runs `python inference.py` on a clean server without `mlrescurix_worker` installed, `pickle.load()` will immediately crash with `ModuleNotFoundError: No module named 'mlrescurix_worker'`.
* **The Patch for Step 4:**
* During Phase 6 bundle compilation, copy a standalone, self-contained `bundle_primitives.py` file directly into `mlrescurix.bundle.v2/` (or inline the 4 transformer class definitions inside `inference.py`) and register module aliases before calling `joblib.load()`, **OR** serialize using `skops` / `cloudpickle` pinned to standard `scikit-learn`/`numpy` primitives.



#### Edge Case 2: The Missing `OOF Error Forensics` Producer in Section 6

* **The Conflict:** In the Phase 3 diagram (Box 1), it states: *"M1 reads `eda_report.md` + Out-of-Fold Error Forensics"*. However, in Section 6 (Target Codebase Blueprint), there is no module assigned to compute the **OOF Error Forensics** after each trial in Phase 3 (`critic_engine.py` only runs at Phase 5).
* **What happens at runtime:** After Trial 1 finishes in Phase 3, the Trusted Evaluator returns a scalar CV score. Without a mid-loop forensics receipt, M1 won't know *which* features had zero permutation importance (needed to trigger `DELETE_FEATURE`) or *which* feature pair caused the highest residual error (needed to trigger `ADD_INTERACTION`).
* **The Patch for Step 2:**
* Add a lightweight helper function `compute_oof_forensics_receipt(y_true, y_oof, X_val, model)` directly inside the **Trusted Evaluator** (or as a fast mode of `critic_engine.py`). Have it emit a $<1\text{ KB}$ JSON summary after every completed 100%-fidelity BOHB trial containing:
1. `train_val_gap` (triggers `UPDATE_SEARCH_SPACE`),
2. `zero_gain_features` (triggers `DELETE_FEATURE`),
3. `top_residual_interaction_pair` (triggers `ADD_INTERACTION`).





#### Edge Case 3: BOHB Subsampling vs. Group / Temporal Splits (Phase 1 vs. Phase 3)

* **The Conflict:** Phase 1 locks `StratifiedGroupKFold` or `PurgedTimeSeriesSplit` in the Leakage Firewall. Phase 3 (`bohb_search.py`) evaluates Rung 1 at **10% fidelity** and Rung 2 at **33% fidelity**.
* **What happens at runtime:** If `bohb_search.py` implements 10% fidelity via random row subsampling (`df.sample(frac=0.10)`), it breaks temporal autocorrelation in `PurgedTimeSeriesSplit` and shatters group sizes in `QuasiIdDecomposer` (making group-count features different between 10% fidelity and 100% fidelity).
* **The Patch for Step 2:**
* Define the **Fidelity Axis Rule** in `bohb_search.py`:
* For **Temporal splits**: Fidelity $f \in \{0.10, 0.33, 1.0\}$ must take the **most recent $f\%$ contiguous time window** (or scale boosting iterations `n_estimators` while keeping 100% of rows if $N < 50,000$).
* For **Group splits**: Subsample **whole groups (`entity_id`)**, never individual rows, so group aggregations remain unbiased across fidelities.





#### Edge Case 4: OOF Alignment for the `DiversityBuffer` Stacker (Phase 3 vs. Phase 4)

* **The Conflict:** In Phase 3, M1 mutates features across trials (`ADD_INTERACTION`, `DELETE_FEATURE`, `TRANSFORM_TARGET`), while `DiversityBuffer` saves the best champion per model family (`LGBM`, `CatBoost`, `XGBoost`, `MLP`). In Phase 4, `pareto_ensemble.py` stacks them using their saved `oof_predictions` and builds a final `model.pkl`.
* **What happens at runtime:**
1. If Model A (`LGBM`) was trained on Feature Set $F_3$ and Model B (`CatBoost`) was trained on Feature Set $F_7$, the final stacked `model.pkl` must encapsulate **each base model bundled with its own specific feature transformation pipeline** (`Pipeline([F_3, LGBM])` + `Pipeline([F_7, CatBoost])`), rather than assuming a single shared preprocessor.
2. If `TRANSFORM_TARGET` (`np.log1p`) was applied in one trial, `TrustedEvaluator` and `DiversityBuffer` must strictly invert the predictions ($\exp(\hat{y}) - 1$) **before** saving `oof_predictions` and scoring, so all OOF vectors live in the original target space.



#### Edge Case 5: Metric Directionality in the 0.5% Parsimony Gate & Slice Audit

* **The Conflict:** Both the **0.5% Parsimony Gate** (`Gain < +0.5%`) and the **Subgroup Slice Audit** (`slice drop > 15%`) assume "higher is better" bounded metrics like ROC-AUC or F1.
* **What happens at runtime:** For regression metrics where lower is better (RMSE, MAE, LogLoss) or metrics that can be near zero/negative ($R^2$, MCC), a naive percentage ratio $\frac{S_{\text{new}} - S_{\text{old}}}{\vert{}S_{\text{old}}\vert{}}$ flips sign or divides by near-zero.
* **The Patch for Steps 3 & 4:**
* Normalize all metrics into a canonical **Relative Error Reduction ($\text{RER}$)** using Phase 0's `optimization_direction` (`Minimize` vs. `Maximize`) and Phase 2's Dummy Baseline score $S_{\text{dummy}}$:

$$\text{Relative Gain} = \frac{S_{\text{ensemble}} - S_{\text{best\_single}}}{\vert{}S_{\text{best\_single}} - S_{\text{dummy}}\vert{} + \epsilon} \ge 0.005$$



This works uniformly across RMSE, LogLoss, ROC-AUC, and F1 without sign bugs.



---

### 3. Suggested Addenda to Lock into the v2.0 Spec

You can drop these three concise invariant blocks straight into **Section 5 & Section 6** of your markdown document to make it 100% implementation-ready:

```markdown
### Architectural Invariants (v2.0 Execution Contracts)

1. **Pipeline-Per-Champion Packaging (`DiversityBuffer`):**
   Every champion stored in `DiversityBuffer` must serialize its full end-to-end `sklearn.pipeline.Pipeline` (from post-sanitizer feature primitives to estimator) and store its `oof_predictions.npy` strictly in the **original target scale** (post `expm1` inverse if `TRANSFORM_TARGET` was used).

2. **Fidelity-Safe Subsampling (`bohb_search.py`):**
   When dataset rows $N \le 50,000$, BOHB fidelity rungs ($10\% \rightarrow 33\% \rightarrow 100\%$) scale **iteration/epoch budgets** (or evaluate 1 fold $\rightarrow$ 2 folds $\rightarrow$ $K$ folds) rather than row-subsampling. When $N > 50,000$, subsampling is strictly group-preserving (for `ENTITY_GROUP`) or tail-window preserving (for `TEMPORAL_AXIS`).

3. **Zero-Dependency Bundle Serialization (`mlrescurix.bundle.v2`):**
   During Phase 6 export, the bundle compiler copies `primitives.py` and `dirty_sanitizer.py` into `mlrescurix.bundle.v2/_runtime_primitives.py` so `inference.py` and `serve.py` can unpickle `model.pkl` in any clean Python environment with zero dependency on the `mlrescurix_worker` package.

```