# M2 first tabular workflow

This document describes M2's bounded CSV audit,
fixed baseline recipes, development metrics, one candidate refit, a test-access
record, and portable trusted inference bundle. Rust M1 routes the fixed
scientific operations through supervised Python workers. The live local
dashboard is available through the Rust CLI and displays committed snapshots,
events, jobs, receipts, candidate metrics, finalization status, and bundle
download.

## Supported input and policies

Use Python 3.13 and the exact dependency versions in `python/pyproject.toml`.
Do not install them into a global interpreter. The CSV reader caps files at
32 MiB, rows at 50,000, columns at 128, and one-hot categories at 64. It rejects
symbolic links, missing targets, duplicate/empty column names, unsupported task
and metric combinations, and infeasible splits. Input row order defines time
for chronological splits. Group splitting excludes the group column from model
features and keeps groups disjoint across all three partitions.

The deterministic split has 60% train, 20% validation, and 20% test for the
random and stratified policies. Grouped and chronological partitions use the
same approximate proportions. Seeds are recorded in the dataset manifest.
Recipe parameters are fixed in `scientific/workflow.py`; this is not a tuning
interface.

## Local use

Run from the repository root. To exercise the Python primitives directly on
small disposable fixtures:

```sh
PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.cli audit \
  --csv /absolute/path/to/input.csv \
  --spec /absolute/path/to/project-spec.json \
  --output /absolute/path/to/new-project-workspace

PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.cli baselines \
  --output /absolute/path/to/new-project-workspace

PYTHONPATH=python/src python3 -m mlrescurix_worker.scientific.cli finalize \
  --output /absolute/path/to/new-project-workspace \
  --candidate linear --test-access-id final-assessment-1
```

The spec requires `project_id`, `target`, `task`, and `metric`. Task values are
`binary_classification`, `multiclass_classification`, or `regression`; metric
values are `balanced_accuracy`/`log_loss` or `mae`/`rmse` respectively. Optional
fields select `split` (`stratified`, `random`, `grouped`, or `chronological`),
`group_column`, `positive_label`, `inference_batch_size`,
`inference_repetitions`, `max_latency_ms`, and `max_bundle_bytes`.

The output directory must be new and owned by the project. Keep it outside Git.
The snapshot is marked read-only. Do not run `finalize` more than once: the
test-access record binds one key, dataset, split, candidate, and frozen model.
After a completed assessment, inspect `final.json` and `scorecard.json`.

For supervised use, create a managed Python 3.13 environment with
`create-environment`, then run `install-scientific-profile`. The installer
bootstraps pip only inside that owned venv and installs the exact pins in
`python/scientific-requirements.txt`, binary wheels only, with a 20-minute
deadline and no pip cache. `register-dataset` copies the explicitly selected
CSV (maximum 32 MiB) into the owned project store, verifies its hash, marks it
read-only, and records a committed event. Each scientific `JobSpec` uses the
typed `scientific` probe with operation `audit`, `baselines`, or `finalize`, a
registered `dataset_id`, and the required task generation, environment lease,
deadline, thread request, and output bound. Audit additionally carries the
strict project spec; finalization carries the candidate and unique test access
IDs. The supervisor invokes only the packaged scientific module operations.

An audit request looks like this; replace IDs and generation with the values
from the active M0 task claim. Give it a timeout that fits the task lease and
the configured device/project `wall_ms` budget:

```json
{
  "job_id": "audit-1",
  "project_id": "project-1",
  "task_id": "task-1",
  "generation": 2,
  "timeout_ms": 300000,
  "threads": 1,
  "max_output_bytes": 1048576,
  "environment_id": "scientific",
  "probe": {
    "kind": "scientific",
    "operation": "audit",
    "dataset_id": "dataset-1",
    "spec": {
      "project_id": "project-1",
      "target": "target",
      "task": "binary_classification",
      "metric": "balanced_accuracy",
      "split": "stratified",
      "positive_label": "yes"
    },
    "candidate_id": null,
    "test_access_id": null
  }
}
```

Write it to a request file and submit with
`cargo run -p mlrescurix-harness -- submit --state /absolute/state --spec
/absolute/audit.json`. Then inspect the durable job with `inspect-job`. Submit
the same shape for `baselines` with `spec` set to null, or `finalize` with
candidate and access IDs populated. M1's supervisor handles fairness, deadline,
pause, cancellation, and cleanup for each of these operations.

Start `supervise --state ...` and `dashboard --state ...` in separate terminals.
The dashboard binds to loopback and prints a one-hour session token once. Enter
that token in the page. Its controls issue revision-checked pause, resume, and
cancel commands with durable receipts.

## Evidence and limits

Each baseline fits preprocessing only on its training partition and scores the
validation partition. Final refit uses train plus validation. Inference
constraints are measured on validation features before test access. The test
gate is written before the test partition is scored. A retry can resume only
the same frozen artifact and access key. A successful bundle contains the
fitted pipeline, raw-row inference function, input schema, dependency pins,
scorecard, verification record, training recipe/source, examples, and hashes.

Job dispatch, deadlines, worker leases, wall-time charges, task fencing,
artifact hashes, committed job events, and project commands remain under M0/M1.
M2 records a durable per-project test reservation in SQLite before admitting a
finalization job, then commits its candidate/artifact/result identity after
successful assessment. The file gate permits retry only with that same frozen
artifact and key. The trusted evaluator shares the managed scientific
environment; generated code cannot run in this milestone. The local gate
remains a workflow guard, not a boundary against a person who can edit the
owned state directory.

Process memory is not quota-enforced, and inference measurement omits cold-load
time. A required hard memory limit is unsupported. The first scope has no HPO,
provider or autonomous agent, custom recipes, GPU, Parquet, repeated-seed
uncertainty, or remote runner. Its verified label means the declared local
recipe, inference, and quality checks passed; it does not certify unsupported
host limits or another machine.

Joblib artifacts can execute code when loaded. Load only bundles produced by
this trusted workflow and preserve their dependency lock.
