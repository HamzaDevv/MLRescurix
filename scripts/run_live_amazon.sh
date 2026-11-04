#!/bin/bash
set -euo pipefail

# Configurable defaults via environment variables
WORKSPACE="${WORKSPACE:-.mlrescurix-state/amazon-2025/live-$(date +%Y%m%d-%H%M%S)}"
PROJECT="${PROJECT:-amazon-live-$(date +%Y%m%d)}"
SESSION="${SESSION:-amazon-live-session-$(date +%Y%m%d-%H%M%S)}"
POLICY="${POLICY:-docs/v2/POLICY_AMAZON_LIVE_20261102.json}"
DEV_CSV="${DEV_CSV:-.mlrescurix-state/amazon-2025/staged/development.csv}"
FOLDS="${FOLDS:-.mlrescurix-state/amazon-2025/development_folds.json}"
MODEL="${MODEL:-deepseek/deepseek-v4.1-flash-1017}"
ROUTE="${ROUTE:-relace/fp4}"
MAX_TURNS="${MAX_TURNS:-}"

# Parse CLI options if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --project) PROJECT="$2"; shift 2 ;;
    --session) SESSION="$2"; shift 2 ;;
    --policy) POLICY="$2"; shift 2 ;;
    --dev-csv) DEV_CSV="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --route) ROUTE="$2"; shift 2 ;;
    --max-turns) MAX_TURNS="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--workspace PATH] [--project NAME] [--session ID] [--policy PATH] [--model MODEL] [--route ROUTE] [--max-turns N]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Preflight checks
echo "=== Running Preflight Checks ==="
if [[ ! -f "$POLICY" ]]; then
  echo "Preflight failed: policy file not found: $POLICY" >&2
  exit 1
fi
if [[ ! -f "config/operational_budgets.json" ]]; then
  echo "Preflight failed: config/operational_budgets.json not found" >&2
  exit 1
fi
# Validate JSON syntax of policy and budget config
python3 -c "import json; json.load(open('$POLICY')); json.load(open('config/operational_budgets.json'))" || {
  echo "Preflight failed: policy or operational budget configuration is not valid JSON" >&2
  exit 1
}

PROBLEM="Autonomous ML engineering for Amazon catalog pricing: progressively hypothesize, extract text and tabular features, evaluate diverse model families (hist_gbdt, penalized_linear, lgbm, xgboost, catboost, neural_tabular) with log1p target transform, explore hyperparameters, and form an ensemble blend (BLEND_AND_AUDIT) to beat the prior apex CV scores before finalizing."

mkdir -p "$WORKSPACE"

echo "=== Launching MLRescurix Amazon 2025 Live Run ==="
echo "Workspace: $WORKSPACE"
echo "Project: $PROJECT"
echo "Session: $SESSION"
echo "Model: $MODEL via $ROUTE"
echo "Policy: $POLICY"
if [[ -n "$MAX_TURNS" ]]; then
  echo "Max turns: $MAX_TURNS"
else
  echo "Max turns: unbounded (operational budget ceiling enforced)"
fi

CLI_ARGS=(
  --development-csv "$DEV_CSV"
  --folds "$FOLDS"
  --workspace "$WORKSPACE"
  --project "$PROJECT"
  --session "$SESSION"
  --target price
  --task regression
  --metric smape
  --direction minimize
  --problem "$PROBLEM"
  --model "$MODEL"
  --provider-route "$ROUTE"
  --policy "$POLICY"
  --approval-wait-seconds 120
  --development-live
)

if [[ -n "$MAX_TURNS" ]]; then
  CLI_ARGS+=(--max-turns "$MAX_TURNS")
fi

# Start the agent CLI
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_cli \
  "${CLI_ARGS[@]}" \
  > "$WORKSPACE/agent.log" 2>&1 &
AGENT_PID=$!
echo "Agent CLI running in background with PID $AGENT_PID"

# Start the operator supervisor watching AGENT_PID
python3 scripts/operator_supervisor.py \
  --state "$WORKSPACE/rust_harness_state" \
  --project "$PROJECT" \
  --model "$MODEL" \
  --route "$ROUTE" \
  --log-file "$WORKSPACE/operator_audit.log" \
  --watch-pid "$AGENT_PID" \
  > "$WORKSPACE/supervisor.log" 2>&1 &
SUPERVISOR_PID=$!
echo "Operator supervisor running in background with PID $SUPERVISOR_PID"

echo "Waiting for agent to complete..."
AGENT_EXIT=0
wait "$AGENT_PID" || AGENT_EXIT=$?
echo "Agent CLI exited with code $AGENT_EXIT"

echo "Waiting for operator supervisor to finish..."
wait "$SUPERVISOR_PID" || true

echo "=== Live Run Process Terminated ==="
exit "$AGENT_EXIT"
