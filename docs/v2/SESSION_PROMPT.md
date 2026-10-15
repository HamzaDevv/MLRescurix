# Paste-ready continuation prompt

Use this in a fresh local coding session for Phase P9 (Separately Gated Expansion & Optimizations: P9.1 Out-of-Core Arrow Data Adapter & P9.2 Time-Series Forecasting Adapter).
Phases P1 through P8 are complete, evidenced, and their acceptance gates are closed on `main`.

```text
Continue MLRescurix in /Users/hamza/Developer/Imp_Projects/MLRescurix on branch 'main'.
Authorized scope: implement Phase P9.1 (Out-of-Core / Arrow Data Adapter) and P9.2 (Time-Series Forecasting Adapter) and satisfy their respective adapter acceptance gates as defined in docs/v2/ROADMAP.md, ARCHITECTURE.md, DECISIONS.md, and PROGRESS.md. Do not begin P9.3 or later slices without explicit authorization.

Start by inspecting git status on 'main' and reading:
- docs/v2/README.md
- docs/v2/PROGRESS.md
- docs/v2/ROADMAP.md (Phase P9 section)
- docs/v2/DECISIONS.md (Decisions D1–D38)
- docs/v2/ARCHITECTURE.md
- docs/evidence/v2/p8/20261015-p8-gate-closure/summary.md

Retain `PRAGMA user_version = 10` and keep `docs/evidence/m6/` completely untouched.

--------------------------------------------------------------------------------
1. CORE DELIVERABLES (P9.1 & P9.2)
--------------------------------------------------------------------------------
1. P9.1 Out-of-Core / Apache Arrow Data Adapter:
   - Typed schema and contract for Arrow IPC / Feather / Parquet ingestion fingerprints, split views, and byte limits.
   - True memory-mapped column access and chunked batch iterators without inflating memory into dense NumPy arrays.
   - Empirical memory accounting: measure actual preprocessing and model copies; do not make unfounded zero-copy claims.
   - Out-of-core baseline models (SGDClassifier/Regressor partial_fit, incremental feature scaling).

2. P9.2 Time-Series Forecasting Adapter:
   - Strict temporal split contracts: entity-aware, rolling-window, and expanding-window backtesting without future lookahead leakage.
   - Lagged feature generation, rolling statistical aggregations, and horizon alignment.
   - Naive and seasonal baselines (LastValue, SeasonalMean) against which ML models must compete.
   - Time-series delivery inference contract: multi-step horizon prediction with timestamp input validation.

3. P9.1 & P9.2 Acceptance Gates:
   - Complete fit/predict/evaluate/package vertical slice for both out-of-core and time-series models.
   - Negative scientific cases: reject future lookahead leakage, timestamps out of order, and out-of-core allocations exceeding memory caps.
   - Full locked regression suites re-verified: feasibility_p8, release_p7, dashboard_p6, collaboration_p5, candidate_tools_p4, resource_broker_p3, artifact_registry_p2, foundation, agent_m4, recovery_cycles, release_m6, and Python tests.
   - Save fresh evidence logs under docs/evidence/v2/p9/<session>/.
   - Record Decision D39 in docs/v2/DECISIONS.md, update PROGRESS.md, ROADMAP.md, and rebuild docs/v2/ARGON_REVIEW_PACKET.txt.
```
