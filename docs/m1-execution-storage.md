# M1 safe execution and managed storage

M1 adds a durable local queue, one shared-device supervisor, a typed worker process, environment ownership, and managed artifact cleanup. It proves process and storage boundaries with deterministic probes. There is no model training, scoring, HPO, generated-code execution, provider, dashboard, GPU, or remote runner in this phase.

## Runner boundary

The initial runner is validated on macOS with Rust 1.94 and Python 3.13 available for optional standard-library virtual environments. Jobs select one of three built-in probes: bounded sleep, an intentional nonzero exit, or writing one bounded text file. The request cannot name an executable, shell command, or arbitrary source file. A private worker waits for the supervisor's durable dispatch gate before it starts.

The supervisor holds an exclusive OS lock for the state directory. Run it in a foreground terminal and stop it with Ctrl+C to terminate and reap its active workers. It polls durable queue and cancellation state, uses round-robin project rotation with configurable project weights and waiting-time aging, and records job events. Queue bounds are 128 pending jobs per project and 1024 across the device. Jobs also obey M0 task leases, project/device worker allocations, a `wall_ms` budget reservation, and the 64 KiB command frame bound.

Resource claims are deliberately specific. Worker count and declared thread allocations are admitted against policy. Common native thread-pool environment settings are passed to the process. The probes do not exercise scientific thread pools, and those settings are not OS CPU quotas. The supervisor stores one best-effort `ps` sample per running job, with CPU percent, resident bytes, and peak resident bytes; these measurements are not hard limits or certification evidence. macOS memory quotas are unsupported, outputs are capped, and a free-space check preserves 64 MiB of state-storage headroom. A policy that needs unsupported hard enforcement cannot be certified by this runner.

## State policy

For M1 execution, device and project policies must declare matching budget dimensions including `wall_ms`. The supervisor reserves the job's full timeout before dispatch and charges measured elapsed time after a terminal outcome. A worker that outlives its timeout is terminated as a process group. If the supervisor dies after dispatch, the worker detects that its control pipe closed and stops. The job remains capacity-reserved until its bounded deadline; the supervisor then verifies any published/staged output or records an interrupted outcome and charges the reserved timeout conservatively.

Use only a new, explicitly owned state directory. M1 migrates that database from schema version 1 to version 2 inside a SQLite transaction. User files are never placed in managed paths. Environment creation uses `python3 -m venv --without-pip` in the owned state directory, checks Python 3.13, and does not install packages or modify the selected interpreter. A job can lease a ready environment; removal is refused while a lease exists. Artifact manifests record each file's size and SHA-256. Pins block garbage collection, and GC removes only terminal job outputs/logs recorded under the managed state root.

## Minimal workflow

Initialize an empty state directory with complete device budget dimensions. This example is for disposable local work:

```json
{"budget":{"calls":100,"tokens":50000,"wall_ms":3600000},"capacity":{"workers":2},"max_tasks":100}
```

Create a project with matching budget and capacity dimensions, then create a branch and task and claim it with a lease that lasts at least as long as the job timeout. Start the supervisor:

```sh
cargo run -p mlrescurix-harness -- supervise --state /absolute/path/to/new-state --workers 2
```

In another terminal, submit a typed probe after replacing the IDs and generation with the current claimed task values:

```json
{"job_id":"probe-1","project_id":"project-1","task_id":"task-1","generation":2,"timeout_ms":5000,"threads":1,"max_output_bytes":1024,"environment_id":null,"probe":{"kind":"write","name":"result.txt","content":"synthetic output"}}
```

```sh
cargo run -p mlrescurix-harness -- submit --state /absolute/path/to/new-state --spec /absolute/path/to/job.json
cargo run -p mlrescurix-harness -- inspect-job --state /absolute/path/to/new-state --project project-1 --job probe-1
```

Use `cancel-job` for one job. The existing project lifecycle command pauses admission and drains active work, or cancels a project and terminates its workers. `set-project-weight --weight 1..=10` adjusts its scheduling share. Create an optional standard-library environment with `create-environment --id py313 --python /absolute/path/to/python3.13`; a job requests it with `environment_id`. Use `pin-artifact` to protect a completed output and `collect-artifacts` to remove unpinned terminal outputs and logs. The metadata ledger and receipts remain available after artifact cleanup.

## Limitations

This is an execution kernel and storage lifecycle demonstration. The worker does not train a model, evaluate predictions, or create official scores. It does not enforce a memory quota, provide a generated-code sandbox, sample continuous CPU telemetry, or retain an environment cache policy. A supervisor crash can leave a probe running until its own bounded timeout; recovery does not kill by PID because that could target a reused process identity. M2 separately adds scientific workers and evaluation.
