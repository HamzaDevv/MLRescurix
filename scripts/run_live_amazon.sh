#!/bin/bash
set -euo pipefail

WORKSPACE=".mlrescurix-state/amazon-2025/live-20261102-run2"
PROJECT="amazon-live-20261102"
SESSION="amazon-live-20261102-run2"
POLICY="docs/v2/POLICY_AMAZON_LIVE_20261102.json"
DEV_CSV=".mlrescurix-state/amazon-2025/staged/development.csv"
FOLDS=".mlrescurix-state/amazon-2025/development_folds.json"
MODEL="deepseek/deepseek-v4.1-flash-1017"
ROUTE="wafer/us"

mkdir -p "$WORKSPACE"

echo "=== Launching MLRescurix Amazon 2025 Live Run ==="
echo "Workspace: $WORKSPACE"
echo "Project: $PROJECT"
echo "Session: $SESSION"
echo "Model: $MODEL via $ROUTE"

# Start the agent CLI
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_cli \
  --development-csv "$DEV_CSV" \
  --folds "$FOLDS" \
  --workspace "$WORKSPACE" \
  --project "$PROJECT" \
  --session "$SESSION" \
  --target price \
  --task regression \
  --metric smape \
  --direction minimize \
  --model "$MODEL" \
  --provider-route "$ROUTE" \
  --policy "$POLICY" \
  --approval-wait-seconds 120 \
  --max-turns 6 \
  --development-live \
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
