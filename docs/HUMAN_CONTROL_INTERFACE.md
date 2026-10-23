# Common Human Control Interface

This document specifies the human interaction and session steering interface for MLRescurix. The interface is implemented by `AgentInterface` in [`python/src/mlrescurix_worker/scientific/agent_interface.py`](file:///Users/hamza/Developer/Imp_Projects/MLRescurix/python/src/mlrescurix_worker/scientific/agent_interface.py).

Every task (development agent or benchmark runner) uses the same control point for status inspection, steering, pause, resume, and termination.

---

## 1. Quickstart & CLI Usage

From the repository root with `PYTHONPATH=python/src`:

```sh
# 1. Inspect session status (zero LLM tokens, non-mutating)
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace status

# 2. Inject human steering directive
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace steer \
  --message "Prioritize interpretable models; inspect feature residuals before ensembling." \
  --command-id steer-001

# 3. Pause session at next safe turn boundary
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace pause \
  --command-id pause-001

# 4. Resume paused session
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace resume \
  --command-id resume-001

# 5. Stop session cleanly (preserves checkpoints and receipts)
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface \
  --workspace /path/to/session/workspace stop \
  --command-id stop-001
```

---

## 2. Python API

Callers interact via the `AgentInterface` class:

```python
from mlrescurix_worker.scientific.agent_interface import AgentInterface

# Initialize interface bound to the session workspace
agent = AgentInterface('/path/to/session/workspace')

# Inspect current status without provider calls
status = agent.status()
print(f"Session {status['session_id']}, Completed turns: {status['turns_completed']}")
print(f"Current stop reason: {status['stop_reason']}")
print(f"Accepted candidate: {status['accepted_candidate_id']}, Best CV: {status['best_cv_score']}")

# Submit steering guidance
cmd_id = agent.submit(
    'steer',
    'Focus on tree-based models and eliminate high-cardinality features.',
    command_id='guidance-101'
)
```

---

## 3. Supported Operations & Turn-Boundary Semantics

| Command | Behavior | When Consumed |
|---|---|---|
| `status()` | Reads session binding, completed turns, stop reason, pending provider intent, human guidance, pending commands, accepted candidate ID, and best CV score. Makes **zero provider calls**. | Immediate (read-only) |
| `submit('steer', message)` | Appends guidance for the controller's subsequent turns. The last 8 guidance messages are injected into the governed LLM prompt. Guidance represents preference, **not authority to bypass scientific gates**. | Next turn boundary |
| `submit('pause')` | Sets `stop_reason = 'human_pause'` at the next safe turn boundary. Does not kill in-flight workers or discard pending provider responses. | Next turn boundary |
| `submit('resume')` | Clears `human_pause` on the next run. Does not automatically launch background processes; the operator or runner must reinvoke the session. | On controller reinvocation |
| `submit('stop')` | Permanently halts adaptive search (`stop_reason = 'human_stop'`). The campaign runner will not proceed to candidate packaging or refitting. Checkpoints and receipts are preserved. | Next turn boundary |

> [!IMPORTANT]
> **Turn-Boundary Execution**: Commands submitted via `AgentInterface` are returned as **queued**, not immediately executed. The controller drains and applies queued commands at safe turn boundaries between AI turns. If a controller has already exited with `human_pause`, queuing `resume` prepares the state; the operator must reinvoke the runner to resume execution.

---

## 4. Concurrency, Invariants & Inbox Protocol

- **Durable Inbox (`human_control.json`)**: All submitted commands are serialized as an append-only JSON list in the session workspace.
- **Independent Lock (`.human_control.lock`)**: Synchronized via `fcntl.flock(fcntl.LOCK_EX)`. This lock is strictly separate from the controller's main workspace lock, enabling human steering and status inspection while the controller is actively training models or querying LLMs.
- **Idempotency & Replay**:
  - Each command requires a stable `command_id` (auto-generated UUID4 if omitted).
  - Resubmitting an identical `command_id` with identical content is idempotent and succeeds without duplicate entries.
  - Reusing a `command_id` with altered content raises a `ValueError` to prevent conflicting directive replays.
- **Bounds & Caps**:
  - Maximum message size: **6,000 UTF-8 bytes**.
  - Maximum inbox size: **1,000 commands**. A full inbox must be preserved and reconciled, never silently purged.
  - Retained guidance window: The controller checkpoints the **last 8 guidance messages** into `autonomous_loop_state.json`.

---

## 5. Strict Approval Boundaries

The human control interface is a **local session-steering mechanism**, not a security or authorization bypass. To preserve scientific integrity and financial safety, the following actions are **strictly prohibited** via human steering:

1. **CANNOT Authorize Provider Uploads**: Human steering messages cannot authorize live network dispatches to frontier LLM APIs. Live dispatches require explicit, separate operator approval via the Rust control plane:
   ```sh
   mlrescurix-harness inspect-outbound-view --state <STATE> --project <PROJECT> --request-sha256 <DIGEST>
   mlrescurix-harness approve-outbound-view --state <STATE> --project <PROJECT> --request-sha256 <DIGEST> --actor <OPERATOR> --ttl-ms 3600000
   ```
2. **CANNOT Approve Models or Veto Audits**: Human steering cannot promote an unverified model to Apex, override the `CriticEngine` veto, or lower benchmark quality thresholds.
3. **CANNOT Alter Frozen Contracts**: Human steering cannot alter frozen split manifests, add/remove dataset rows, or modify task metrics and horizons.
4. **CANNOT Reset Campaign Budgets**: Human steering cannot reset the USD 0.30 campaign budget ceiling, release uncertain spend, or mint new allowances.
5. **CANNOT Clear Ambiguous Dispatches**: Human steering cannot overwrite pending or uncertain provider responses. Ambiguous dispatches require authoritative reconciliation with the gateway database.

> [!CAUTION]
> **Operating Identity Separation**: `AgentInterface` executes under the local session owner's OS identity. Actor strings (e.g. `--actor`) are local telemetry and do not provide cryptographic authentication against an adversarial process sharing the same OS user. In production environments, operator approval commands must be executed under a distinct, non-shared OS identity.
