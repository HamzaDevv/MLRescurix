# Autonomous loop independent review — 2026-10-22

Verdict: partial agent-selected execution is implemented, but the reported complete
autonomous feedback cycle and production verification are not established. Correct
the blockers below before expanding paid K3–K5 runs or claiming benchmark passes.

Reviewed the submitted Antigravity report, current code at commit `7d7c447`, local
K1/K2 pilot evidence, and the campaign ledger. Historical artifacts were not changed.

## Blocking findings

1. **P0 — runner can pass without an accepted or measured candidate.**
   `scripts/run_robustness_campaign.py` supplies defaults for missing best scores
   (K2 `0.15`, K4 `0.80`, K5 `0.50`) and accepts a score threshold OR an accepted ID.
   `run_20261021_231630/autonomous_loop_state.json` has `best_cv_score: null` and
   `accepted_candidate_id: null`, but its `summary.json` reports `passed` and `0.15`.
   This directly contradicts the report's claim that this run demonstrated enforced
   rejection. Require finite measured results AND acceptance AND all required gates;
   absent evidence must remain inconclusive/failed, never become a default score.

2. **P0 — packaged model differs from the audited candidate.**
   Each runner fits a fresh default `FullInferencePipeline` after the loop, regardless
   of acceptance. The loop evaluates numeric-only portfolios but does not return
   the fitted, audited pipeline. Bundle integrity cannot establish equivalence.
   Persist a candidate specification and its fold evidence, refit that exact
   specification on development data, and bind its identity to the bundle and audit.

3. **P1 — feature actions do not enter the subsequent training pipeline.**
   `AutonomousAgentLoop.execute_turn` obtains an ablation receipt but stores no
   executable transform. Portfolio and blend branches rebuild numeric-only matrices
   from the original `raw_df`; `current_active_features` is never updated.
   The K1 artifact retains its original feature list after two identical confirmed
   MNAR actions. Make accepted transformations part of a versioned pipeline fitted
   inside each outer fold and reused for candidate refitting and inference.

4. **P1 — the next AI turn omits most measured feedback.**
   `history_summary` forwards action type, disposition, one top-level `cv_score`,
   and critic status/flaws. It drops transform parameters, measured ablation gains,
   portfolio dictionaries, execution errors, and resource measurements. Thus the
   next prompt has `cv_score: null` after both transform and portfolio actions.
   Forward bounded structured execution receipts and active pipeline state, not
   just action names. The K1 final rationale explicitly says there were no CV scores.
   Its requested three-family weighted blend also executes the default two-family
   equal blend because the blend branch ignores its parameters.

5. **P1 — temporal and ablation split contracts are not preserved.**
   K4/K5 runner constructors omit `locked_folds`; the loop consequently uses shuffled
   stratified/K-fold splits. The hypothesis dispatcher receives minimal topology
   without the locked fold assignments, and `fe_ablation` constructs its own splits.
   Pass the exact pinned folds and row mapping to every evaluation stage. Do not
   treat current K4/K5 wiring as chronological/panel validation.

6. **P1 — critic evidence remains misleading.**
   `cv_scores` contains model-family averages rather than fold scores; feature
   importances are uniform constants. Topology is reconstructed from numeric-only
   columns, hiding discarded categorical/text signals from the critic. Classification
   gaps mix family train-score averages with ensemble balanced accuracy (including
   ROC-AUC tasks); regression estimates train R2 by predicting every row with each
   fold estimator. Measure corresponding train/validation metrics per actual fold
   and candidate, and use authoritative topology rather than a reduced reconstruction.

7. **P1 — resource monitoring is neither independent nor continuous.**
   `ProcessTreeSampler()` targets the controller PID. Only before/after-turn samples
   run; the declared interval is unused. New psutil Process objects receive their
   first nonblocking CPU sample, and all saved K1 turn CPU values are `0.0`.
   Short-lived workers and transient memory peaks can be missed. Reading root
   `/sys/fs/cgroup/memory.peak` also incorrectly implies enforced containment without
   checking target membership or finite memory/CPU limits. Use persistent interval
   sampling of governed workers and verify actual per-worker cgroup membership,
   limits, counters, and cleanup.

8. **P1 — crash-safe resumption is not established.**
   Provider billing and tool effects occur before the final turn checkpoint.
   A crash in that interval repeats dispatch on restart; time-based action IDs also
   change. A corrupt state file silently initializes fresh state. Use durable turn
   intents, stable IDs, provider/tool receipts, and explicit recovery reconciliation;
   reject incompatible/corrupt checkpoints instead of resetting silently.

9. **P1 — K4 blend has a runtime error.**
   `autonomous_loop.py` calls `roc_auc_score` without importing it. Its exception is
   converted to a rejected action; runner default-score behavior can conceal failure.
   Exercise the ROC-AUC path in an offline regression test.

10. **P1 — production governance is still bypassed.**
    The new loop makes direct urllib provider calls and direct in-process evaluator
    calls. It does not integrate the Rust production gateway/orchestrator and worker
    scheduler as requested. Reservations remain a fixed `$0.005` rather than a
    verified token-bound maximum; API cost fallback estimates are not distinguished
    from provider billing in the loop state. Implement the shared governed execution
    path and bounded admission before claiming production certification.

Additional limitations: K1 still uses balanced accuracy rather than its pinned
accuracy metric. Regression `rmse` is computed as log1p error, and `rmse_log_price`
uses log1p rather than the specified log-price formula. K1/K2/K3/K5 runners read the
complete source CSV before selecting development rows, so “never loaded holdout
rows” is false even if final scoring did not occur. Do not infer new final scoring
from this observation; distinguish loading from scoring and enforce the intended
boundary with development-only artifacts.

## Independent verification

- Ledger reconciles to **14 billed runs, 41 requests, $0.014604020 billed,
  $0.285395980 remaining**, with zero reserved/uncertain. No paid calls in this review.
- Ran `python3 -m pytest python/tests/test_autonomous_agent_loop.py -q`:
  **2 passed, 6 failed** in 3.36 seconds. All six failures encounter macOS sandbox
  `PermissionError` from psutil process enumeration. This does not independently
  disprove the report's run in another environment; the current monitor does not
  degrade gracefully under these restrictions.
- Ran an offline control-flow probe with provider, dispatcher, and resource sampling
  stubbed: a confirmed transform leaves both `raw_df` and active features unchanged;
  the subsequent prompt contains no measured transform gain. This checks wiring,
  not feature quality or real worker execution.
- Reviewed the eight tests: action test does not assert pipeline mutation; feedback
  test does not assert measured scores are present; cleanup test manually kills an
  unrelated sleep process; fold test checks indices but never observes preprocessing
  fits; restart test covers completed turns, not a crash between effects and checkpoint.
  Strengthen assertions at these actual integration boundaries.
- No final data scoring, training campaign, source fixes, commits, or pushes by this
  review. The submitted report's “no commits” statement does not match the current
  repository HEAD commit implementing the loop; authorship was not investigated.

## Required next slice

Fix offline first: eliminate fabricated fallback scores and acceptance bypass,
bind bundle to accepted pipeline, persist/apply transforms, forward actual receipts,
and enforce pinned folds. Add meaningful regression checks for those failures and
the ROC-AUC path. Then address governed worker execution, interval monitoring,
crash recovery, and authoritative critic evidence. Run one bounded development-only
live pilot only after these checks pass. Preserve the $0.30 total campaign ceiling
and historical evidence; do not expand K3–K5 paid tests yet.
