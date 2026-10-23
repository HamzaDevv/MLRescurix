# Common human control interface

The common entry point for interacting with any running task is `AgentInterface(workspace)`, implemented in `python/src/mlrescurix_worker/scientific/agent_interface.py`. A future UI or conversational frontend can use this interface without knowing the task adapter, training backend or provider transport.

Start a development session using `scientific.agent_cli` with its development CSV, frozen fold file, target/task/metric and workspace arguments. Benchmark sessions use `run_robustness_campaign.py --mode live_dev --task ... --run-id ...`; their workspace is the printed evidence run directory. Live execution still requires all scientific, pricing, shared-budget and independent upload approvals. Consult `--help` before configuring a session; no live execution is needed to inspect or submit local control commands.

From the repository root, with the existing Python environment:

```sh
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /absolute/session/workspace status
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /absolute/session/workspace steer --message "Prioritize interpretable models; explain measured failures before another experiment." --command-id guidance-1
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /absolute/session/workspace pause --command-id pause-1
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /absolute/session/workspace resume --command-id resume-1
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.agent_interface --workspace /absolute/session/workspace stop --command-id stop-1
```

Python callers use the same contract:

```python
from mlrescurix_worker.scientific.agent_interface import AgentInterface

agent = AgentInterface('/absolute/session/workspace')
status = agent.status()
command_id = agent.submit('steer', 'Inspect the last measured residuals before choosing the next model.',
                          command_id='guidance-2')
```

| Operation | Behavior |
| --- | --- |
| `status()` | Reads session/binding identity, completed turns, stop reason, pending provider intent, human guidance, queued commands and accepted candidate. Makes no provider call. |
| `submit('steer', message)` | Records guidance for subsequent controller decisions; recent guidance enters the governed next prompt. It is a preference, not permission to bypass gates. |
| `submit('pause')` | Stops further turns at the next safe boundary. Does not kill an in-flight worker or discard an uncertain provider outcome. |
| `submit('resume')` | Clears only a human pause when the controller next runs. It cannot reopen a human stop, agent finalization, handoff or ambiguous dispatch. |
| `submit('stop')` | Ends adaptive work at the next boundary, retaining checkpoint/evidence. The campaign runner does not start packaging/refit after this human stop. |

Submission returns **queued**, not executed. If the controller is currently running, it consumes commands before a subsequent turn. If it has exited paused, queue `resume` and invoke the original controller/runner again with the same workspace/session/input configuration. Commands do not start background workers. A final decision, exhausted budget, missing approval or interrupted provider request still requires its original resolution. Guidance is included in the exact outbound content, so changed guidance requires matching independent outbound approval before a live upload.

`human_control.json` is the durable inbox; `.human_control.lock` serializes readers/writers independently of the controller workspace lock. Commands carry session/config binding, immutable content digest, timestamp and stable ID. Retrying the same ID/content is idempotent; changing content under that ID fails. The controller records applied IDs and the last eight guidance messages in its checkpoint before dispatch. Crash/restart replays unapplied commands safely. Corrupt/foreign inbox receipts fail closed. The inbox is bounded to 1,000 commands and each message to 6,000 UTF-8 bytes; preserve/reconcile a full inbox instead of silently deleting its history.

This is local session control under the session owner's OS identity. It is not remote authentication, an upload-operator identity, or candidate approval. Never grant the controller/worker authority to execute the independent operator's upload approval commands. Status and raw receipts are local evidence, not automatically authorized outbound content.
