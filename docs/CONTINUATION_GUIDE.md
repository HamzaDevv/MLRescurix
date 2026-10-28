# Continuation Guide for AI Agents and Developers

This guide provides instructions and operating rules for any AI coding agent or human engineer continuing work in the MLRescurix repository.

For the current general ML/deep-learning infrastructure build, read [IMPLEMENTATION_REVIEW_20261028.md](IMPLEMENTATION_REVIEW_20261028.md) first for corrected implementation status, then [ARCHITECTURE_BUILD_HANDOFF.md](ARCHITECTURE_BUILD_HANDOFF.md) for scope and plans. [ANTIGRAVITY_IMPLEMENTATION_PROMPT.md](ANTIGRAVITY_IMPLEMENTATION_PROMPT.md) provides the reusable implementation prompt. These documents do not change the historical campaign constraints below.

---

## 1. Prime Directives (Non-Negotiable Rules)

Before editing code or executing commands, commit these non-negotiable rules:

1. **Strictly Offline by Default**: No paid LLM API calls, live network requests, or benchmark holdout scoring are authorized unless the user explicitly grants approval in the prompt.
2. **Never Reset the Campaign Budget Ledger**: The shared campaign ceiling is **USD 0.30 across all tasks, sessions, and retries**. Do not create new campaign directories, mint fresh allowances, or delete [`docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json).
3. **Never Regenerate Frozen Split Manifests**: Frozen split commitments in `data/official/` and `docs/v2/` must remain untouched. Do not resplit to eliminate imbalances or bypass constraints.
4. **Never Invent Missing Quality Thresholds or Temporal Gaps**:
   - Do not invent absolute quality thresholds for K1 or K2 (keep them `null`).
   - Do not invent temporal index spaces or gaps for K4.
   - Incomplete contracts must deterministically fail closed before execution.
5. **Preserve K5 Protocol Invariants**: Do not substitute the old June-only historical subsets or backfilled oil features into K5 Store Sales. Keep the complete 1,782-entity observation calendar intact with deterministic Gregorian calendar predictors only.
6. **Reconcile Checkpoints Explicitly**: Checkpoints lacking new source, configuration, or receipt bindings require **explicit operator reconciliation**, not silent automated migration, redispatch, or re-certification. Never delete pending provider intents to force a session to resume.
7. **Maintain OS Identity Separation**: Local actor strings (e.g. `--actor`) are local telemetry, not adversarial authentication. Never grant the worker or controller execution authority over the operator's approval commands.

---

## 2. Inspecting the Current State

When starting a new session in this repository, run these initial inspection steps:

```sh
# 1. Verify working branch and clean worktree
git status
git log -n 5 --oneline

# 2. Check campaign ledger integrity (SHA-256 must match exactly)
shasum -a 256 docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json
# Expected: 827abd984dd960f91cd39725d7dcbdb78d47c114322a0358cbbb08a5b30d0b52

# 3. Check accounting reconciliation across historical runs
python3 scripts/reconcile_campaign_accounting.py

# 4. Check Docker daemon availability
docker info 2>&1 | head -n 10
```

---

## 3. Standard Operating Workflows

### 3.1 Running Offline Verification Suites

Verify repository integrity without spending tokens or altering state:

```sh
# 1. Run final affected Python regression suite (105 tests, ~50s)
PYTHONPATH=python/src python3 -m pytest \
  python/tests/test_architecture_consolidation.py \
  python/tests/test_autonomous_agent_loop.py \
  python/tests/test_autonomy_regressions.py \
  python/tests/test_rust_gateway_adapter.py \
  python/tests/test_prerequisite_repairs.py \
  python/tests/test_panel_evaluator_and_critic.py \
  python/tests/test_robustness_runner_gates.py \
  python/tests/test_s1_followup_guards.py -q

# 2. Run core Rust unit and integration tests (63 passed, 2 ignored, 1 filtered)
cargo test --locked --offline --lib --test agent_m4 --test reflexive_loop_gate \
  -- --skip p1_scheduler_job_submission_crash_reconciliation_and_fault_regressions

# 3. Run installed-runtime recheck (1 passed)
cargo test --locked --offline --lib \
  assets::tests::installed_runtime_supports_common_controller_and_human_interface_without_checkout

# 4. Run static compilation checks
python3 -m compileall python/src python/tests scripts
cargo build --locked --offline -p mlrescurix-harness
```

### 3.2 Interacting with Running Sessions via `AgentInterface`

Interact with active sessions at turn boundaries:

```sh
# Inspect session status
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace status

# Submit guidance
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace steer \
  --message "Focus on gradient boosting; prune high-correlation features." \
  --command-id steer-101

# Pause / Resume / Stop
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace pause --command-id pause-101
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace resume --command-id resume-101
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace stop --command-id stop-101
```

### 3.3 Launching Governed Development Sessions

To run a development session on staged data:

```sh
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_cli \
  --development-csv /path/to/development.csv \
  --folds /path/to/folds.json \
  --workspace /path/to/workspace \
  --project my_project \
  --session sess_001 \
  --target target_col \
  --task binary_classification \
  --metric balanced_accuracy \
  --direction maximize \
  --max-turns 3 \
  --fixture-actions /path/to/actions.json
```

---

## 4. Preparing for Future Live Execution

If the user explicitly authorizes live execution in a future prompt, satisfy all of the following gates prior to dispatch:

1. **Verify Endpoint & Route**: Verify that OpenRouter model `deepseek/deepseek-v4.1-flash-1017` is available on authorized route `inference-net/fp8` or `deepinfra/fp8/flex` with no fallback.
2. **Revalidate Pricing**: Update and explicitly revalidate input/output price ceilings in the campaign policy before execution.
3. **Inspect & Approve Outbound Views**:
   ```sh
   mlrescurix-harness inspect-outbound-view \
     --state /path/to/state \
     --project <PROJECT> \
     --request-sha256 <DIGEST>

   mlrescurix-harness approve-outbound-view \
     --state /path/to/state \
     --project <PROJECT> \
     --request-sha256 <DIGEST> \
     --actor <OPERATOR_IDENTITY> \
     --ttl-ms 3600000
   ```
4. **Ensure Sufficient Budget**: Confirm that remaining campaign authority is $\ge \text{USD } 0.05$ (the per-run cap).
5. **Never Commit Secrets**: Never commit `.config/openrouter.key`, `.config/kaggle.token`, runtime SQLite databases, raw dataset CSVs, or generated model binaries (`.joblib`, `.safetensors`) to Git.
