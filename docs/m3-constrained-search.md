# M3 constrained, resumable search

This document describes MLRescurix's M3 milestone: constrained, resumable search over parameterized trusted recipes, backed by a managed Optuna sidecar, bounded multi-fidelity evaluation, three-seed confirmation, deterministic tie-breaking, budget protection, and honest development Apex selection.

## 1. Milestone assertion & product flow

MLRescurix explores finite allowlisted recipe configurations on development data, recovers interrupted search without losing budgets or duplicating work, and selects a feasible development champion (Apex) from comparable official evidence. Improvement on every dataset is not required.

Product flow:
```text
register → audit → baselines → bounded search → confirm finalists → select development Apex → explicit finalize → bundle
```

Search operates exclusively on development data (train/validation partitions). The final test partition remains sealed until an explicit finalization request.

## 2. Parameterized trusted recipes

Only existing baseline families are parameterized:
- **Dummy**: Fixed reference baseline; accepts no hyperparameters.
- **Linear**:
  - Binary/Multiclass Classification: regularization `C` in `[0.001, 1000.0]` (log-scaled).
  - Regression: regularization `alpha` in `[0.001, 1000.0]` (log-scaled).
- **Random Forest (tree)**:
  - `n_estimators`: integer in `[16, 128]`
  - `max_depth`: integer in `[2, 20]`
  - `min_samples_leaf`: integer in `[1, 20]`
  - `max_features`: float in `[0.2, 1.0]`

Unknown or invalid parameters are deterministically rejected. Preprocessing remains train-partition only, and single-thread estimator execution (`n_jobs=1`, `OMP_NUM_THREADS=1`) is strictly enforced. No arbitrary or generated executable code is allowed.

## 3. Rust-owned search sessions & experiment DAG

Rust maintains the authoritative session ledger in SQLite (`PRAGMA user_version = 4`):
- `search_sessions`: records session status (`created`, `screening`, `evaluating`, `confirming`, `completed`, `provisional`, `failed`, `cancelled`), dataset reference, spec, created/updated timestamps, Apex candidate ID, and locked `SelectionRecord`.
- `experiment_dag`: persists immutable experiment specs with canonical hashes (`dataset_sha | split | recipe | params | seed | fidelity`), parent lineage, node types (`baseline`, `screening`, `full_fidelity`, `repeat`).
- `optimizer_trials`: maps Optuna study trials to experiment DAG nodes.
- `trial_results`: stores immutable per-trial results, primary validation metrics, constraint measurements, feasibility verdicts, train durations, and model/prediction fingerprints.

Retries are tracked as attempts of the same experiment. Overwriting the shared M2 `results.json` as the search ledger is forbidden; trials write to immutable `trials/{exp_id}/trial.json` files.

## 4. Managed Optuna sidecar

Optuna operates strictly behind a replaceable ask/tell adapter:
- Separate studies and SQLite tables per recipe and fidelity protocol (e.g. `{session_id}_{recipe}_f50` vs `{session_id}_{recipe}_f100`).
- Optuna proposes parameters (`ask`) and receives committed outcomes (`tell`).
- Rust alone admits execution, tracks budget reservations and charges, publishes official results, and selects Apex.
- Pinned dependency profile in `python/search-requirements.txt` (`optuna==5.0.0` and exact transitive locks).
- `DeterministicFallbackOptimizer` guarantees offline and local operation without external network access or an Optuna installation.
- Ready, leased environments cannot be mutated during execution.

## 5. Bounded multi-fidelity search

Evaluation proceeds through bounded stages:
- **Screening (Fidelity = 0.5)**: Trains on 50% deterministic subset of training rows. Respects split semantics (`stratified`, `chronological`, `grouped`, `random`), keeping the validation partition identical. Screening results never qualify directly for Apex.
- **Full Fidelity Promotion (Fidelity = 1.0)**: Up to `max_full_trials` top screening configurations are promoted to 100% training rows.
- **Finalist Confirmation**: Up to two top full-fidelity configurations plus the strongest feasible baseline are confirmed across identical declared estimator seeds (`[2026, 2027, 2028]`).

## 6. Feasible Apex selection & tie-breaking

Apex selection evaluates contenders using the mean primary validation metric across all three seeds:
- Respects metric direction (`maximize` for accuracy/AUC; `minimize` for log-loss/MSE/MAE).
- Applies a practical tie tolerance (`default = 0.005`). Contenders within this tolerance break ties on:
  1. Lower warm p95 inference latency.
  2. Smaller complete model size.
  3. Stable alphabetical candidate identity.
- Infeasible contenders (failing latency or size limits) are disqualified.
- Insufficient budget during confirmation yields an explicit `provisional_budget_exhausted` status.

## 7. Budgets, admission, and recovery

- Enforces strict limits: proposal caps, search wall budget (`search_wall_ms`), per-job timeout (`job_timeout_ms`), repeat allowances, and concurrency bound (default 1 active search trial).
- **Wall budget guard**: Protects at least 20% of the declared project wall budget (or `finalization_allowance_ms`, whichever is larger). If remaining budget drops below the guard allowance, search admission terminates to ensure finalization can complete safely.
- Crash recovery reconciles uncommitted trials across `ask`, mapping, dispatch, result commit, and `tell` boundaries without duplicating executions or resetting budgets.

## 8. Staged bundle size measurement

The previous M2 check measured only `frozen_model.joblib`. M3 fixes this by staging the complete deployment bundle in `.bundle_stage`:
- Complete runtime bundle files (model, `inference.py`, lockfile, schemas, example inputs/expected outputs, README, and scorecard) are staged and measured against `spec.max_bundle_bytes` prior to accessing the single-use test gate.
- If the bundle exceeds the limit, finalization returns `provisional_infeasible` without consuming the test gate.

## 9. Dashboard integration

The dashboard at `/` presents:
- Committed search sessions, lifecycle state, and locked Apex candidate.
- Real-time multi-fidelity experiment DAG table with node types, fidelity levels, seeds, primary metrics, feasibility verdicts, and constraint measurements.
- Budget ledger (charged, held, remaining).
- One-click verified bundle download when finalization is certified.

## 10. CLI commands

```sh
# Initialize search session with immutable spec
mlrescurix-harness search-init --state /path/to/state --spec /path/to/search-spec.json

# Inspect committed search session and experiment DAG
mlrescurix-harness search-inspect --state /path/to/state --project project-1 --session session-1

# Execute search pipeline (screening, full evaluation, confirmation, Apex selection)
mlrescurix-harness search-run --state /path/to/state --project project-1 --session session-1 --task task-1 --generation 1

# Install managed Optuna search profile inside ready owned environment
mlrescurix-harness install-search-profile --state /path/to/state --id env-search
```
