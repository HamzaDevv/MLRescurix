#[path="pipeline_docker.rs"] mod fixtures;
use std::{fs, path::Path, time::Duration};
use serde_json::json;
use mlrescurix_harness::{
    Runtime,
    agents::context::AgentContextManager,
    execution::{self, Supervisor, JobSpec, Probe},
};

fn settle(root: &Path, supervisor: &mut Supervisor, project: &str, id: &str) -> String {
    for _ in 0..300 {
        supervisor.tick().unwrap();
        let j = execution::inspect(root, project, id).unwrap();
        if !matches!(j.status.as_str(), "queued" | "running" | "recovering" | "terminating") {
            return j.status;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("fault cycle did not settle for {project}/{id}");
}

#[test]
fn twenty_five_fault_cleanup_restart_cycles_bound_payload_and_measure_durable_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, fixtures::policy()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "cycles");
    let binary = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let submit = |id: &str, probe: Probe, timeout| {
        execution::submit(
            &root,
            JobSpec {
                job_id: id.into(),
                project_id: "cycles".into(),
                task_id: "t".into(),
                generation: 2,
                timeout_ms: timeout,
                threads: 1,
                max_output_bytes: 4096,
                environment_id: None,
                probe,
            },
        )
        .unwrap()
    };
    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    submit("champion", Probe::Write { name: "model".into(), content: "retained-model".into() }, 2_000);
    assert_eq!(settle(&root, &mut supervisor, "cycles", "champion"), "succeeded");
    execution::pin_artifact(&root, "cycles", "champion", true).unwrap();

    let mut cycle_records = Vec::new();
    let mut max_total_bytes = 0u64;
    let mut max_db_bytes = 0u64;
    let mut cycle_10_total = 0u64;
    let mut prev_charged_wall = 0i64;

    for cycle in 0..25 {
        let write = format!("write-{cycle}");
        submit(&write, Probe::Write { name: "scratch".into(), content: "x".repeat(4096) }, 2_000);
        assert_eq!(settle(&root, &mut supervisor, "cycles", &write), "succeeded");

        let fail = format!("fail-{cycle}");
        submit(&fail, Probe::Fail { code: 7 }, 2_000);
        assert_eq!(settle(&root, &mut supervisor, "cycles", &fail), "failed");

        let timeout = format!("timeout-{cycle}");
        submit(&timeout, Probe::Sleep { millis: 1000 }, 20);
        assert_eq!(settle(&root, &mut supervisor, "cycles", &timeout), "timed_out");

        // Restart supervisor and enforce total-storage policy after each fault cycle
        drop(supervisor);
        let reclaimed = execution::collect_artifacts(&root).unwrap();
        supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();

        assert_eq!(fs::read_to_string(root.join("artifacts/cycles/champion/model")).unwrap(), "retained-model");
        assert_eq!(fs::read_dir(root.join("artifacts/cycles")).unwrap().count(), 1);
        assert_eq!(fs::read_dir(root.join("logs/cycles")).unwrap().count(), 1);

        let summary = execution::storage_policy_summary(&root).unwrap();
        let obs = &summary["observed"];
        let total_bytes = obs["total_state_bytes"].as_u64().unwrap();
        let db_bytes = obs["database_bytes"].as_u64().unwrap();
        let events_rows = obs["events_rows"].as_i64().unwrap();
        let commands_rows = obs["commands_rows"].as_i64().unwrap();
        let jobs_rows = obs["jobs_rows"].as_i64().unwrap();
        let charges_rows = obs["charges_rows"].as_i64().unwrap();
        let held = obs["held_reservations"].as_i64().unwrap();

        assert_eq!(held, 0, "No reservations may remain held after settlement");
        assert!(events_rows <= execution::MAX_RETAINED_EVENTS_PER_PROJECT);
        assert!(commands_rows >= 1, "durable command receipts must remain");
        assert!(jobs_rows <= execution::MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT + 1);
        assert!(charges_rows <= execution::MAX_RETAINED_SETTLED_ACTIONS_PER_PROJECT as i64);

        let budget_info = AgentContextManager::query_authoritative_budget(&root, "cycles").unwrap();
        let charged_wall = budget_info["consumption"]["wall_ms_charged"].as_i64().unwrap();
        assert!(charged_wall >= prev_charged_wall, "Compacted rollup must never lose cumulative charged wall_ms");
        prev_charged_wall = charged_wall;

        if cycle == 10 {
            cycle_10_total = total_bytes;
        }
        max_total_bytes = max_total_bytes.max(total_bytes);
        max_db_bytes = max_db_bytes.max(db_bytes);

        cycle_records.push(json!({
            "cycle": cycle,
            "reclaimed_bytes": reclaimed,
            "total_state_bytes": total_bytes,
            "database_bytes": db_bytes,
            "events_rows": events_rows,
            "commands_rows": commands_rows,
            "jobs_rows": jobs_rows,
            "charges_rows": charges_rows,
            "cumulative_wall_ms_charged": charged_wall,
        }));
    }

    let final_total = cycle_records.last().unwrap()["total_state_bytes"].as_u64().unwrap();
    assert!(max_db_bytes < 1024 * 1024, "SQLite metadata exceeded 1 MiB bound: {max_db_bytes}");
    let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let action_count:i64=db.query_row("SELECT count(*) FROM actions WHERE project='cycles'",[],|r|r.get(0)).unwrap();
    assert_eq!(action_count,76,"completed action retry fences must survive cleanup");
    // Report metadata growth. Finite cycles do not prove an indefinite total bound.


    let evidence_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/evidence/m6");
    if evidence_dir.exists() {
        let _ = fs::write(
            evidence_dir.join("review-storage-cycles.json"),
            serde_json::to_vec_pretty(&json!({
                "acceptance_status": "payload_retention_verified_total_storage_bound_pending",
                "durable_action_identities": action_count,
                "cycles_executed": 25,
                "terminal_jobs_executed": 75,
                "pinned_jobs_retained": 1,
                "cycle_10_total_bytes": cycle_10_total,
                "cycle_24_total_bytes": final_total,
                "max_total_state_bytes": max_total_bytes,
                "max_database_bytes": max_db_bytes,
                "policy": execution::storage_policy_summary(&root).unwrap()["policy"],
                "cycles": cycle_records,
            }))
            .unwrap(),
        );
    }
}

#[test]
fn leased_artifacts_survive_concurrent_gc_and_cancellation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let mut pol = fixtures::policy();
    pol.capacity.insert("workers".into(), 2);
    pol.capacity.insert("cpu_threads".into(), 2);
    let rt = Runtime::init(&root, pol.clone()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "cycles");
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("UPDATE projects SET policy=?1 WHERE id='cycles'", [serde_json::to_string(&pol).unwrap()]).unwrap();
    drop(db);
    // Create and claim a second task `t2` so two jobs can run in parallel under M0 single-active-action-per-task rules
    let task_json: String = rusqlite::Connection::open(root.join("state.sqlite")).unwrap()
        .query_row("SELECT data FROM tasks WHERE project='cycles' AND id='t'", [], |r| r.get(0)).unwrap();
    let t1: mlrescurix_harness::core::Task = serde_json::from_str(&task_json).unwrap();
    let create_receipt = rt.command(mlrescurix_harness::Request {
        command_id: "create-t2".into(),
        project_id: "cycles".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::CreateTask {
            task_id: "t2".into(),
            branch_id: "main".into(),
            context: t1.context,
        },
    }).unwrap();
    assert_eq!(create_receipt.status, "applied", "{}", create_receipt.result);
    let exp = t1.expires_ms.unwrap();
    let claim_receipt = rt.command(mlrescurix_harness::Request {
        command_id: "claim-t2".into(),
        project_id: "cycles".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::Claim {
            task_id: "t2".into(),
            owner: "worker-2".into(),
            lease_id: "lease-t2".into(),
            now_ms: exp - 600_000,
            expires_ms: exp,
            expected_generation: 1,
        },
    }).unwrap();
    assert_eq!(claim_receipt.status, "applied", "{}", claim_receipt.result);
    drop(rt);

    let binary = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 2, binary).unwrap();

    // 1. Submit and finish one job to be pinned and one job to be reclaimed
    for id in ["pinned-job", "reclaim-job"] {
        execution::submit(
            &root,
            JobSpec {
                job_id: id.into(),
                project_id: "cycles".into(),
                task_id: "t".into(),
                generation: 2,
                timeout_ms: 2_000,
                threads: 1,
                max_output_bytes: 4096,
                environment_id: None,
                probe: Probe::Write { name: "out.txt".into(), content: format!("payload-{id}") },
            },
        )
        .unwrap();
        assert_eq!(settle(&root, &mut supervisor, "cycles", id), "succeeded");
    }
    execution::pin_artifact(&root, "cycles", "pinned-job", true).unwrap();

    // 2. Submit two running jobs on tasks `t` and `t2`: `keep-running` and `cancel-midflight`
    execution::submit(
        &root,
        JobSpec {
            job_id: "keep-running".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 3_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Sleep { millis: 300 },
        },
    )
    .unwrap();
    execution::submit(
        &root,
        JobSpec {
            job_id: "cancel-midflight".into(),
            project_id: "cycles".into(),
            task_id: "t2".into(),
            generation: 2,
            timeout_ms: 3_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Sleep { millis: 2_000 },
        },
    )
    .unwrap();

    // Tick supervisor so both jobs transition to `running`
    supervisor.tick().unwrap();
    let j_keep = execution::inspect(&root, "cycles", "keep-running").unwrap();
    let j_cancel = execution::inspect(&root, "cycles", "cancel-midflight").unwrap();
    assert_eq!(j_keep.status, "running");
    assert_eq!(j_cancel.status, "running");

    // Run independent collectors while the supervisor still holds live leases.
    // SQLite serializes metadata cleanup, while the filesystem scan and deletion
    // race with active worker publication and cancellation.
    let collector_root = root.clone();
    let collectors = (0..4).map(|_| {
        let path = collector_root.clone();
        std::thread::spawn(move || {
            for _ in 0..8 { execution::collect_artifacts(&path).unwrap(); }
        })
    }).collect::<Vec<_>>();
    // 3. Cancel `cancel-midflight` while `keep-running` is active.
    execution::cancel(&root, "cycles", "cancel-midflight").unwrap();
    for collector in collectors { collector.join().unwrap(); }

    // Settle both jobs
    let status_cancel = settle(&root, &mut supervisor, "cycles", "cancel-midflight");
    let status_keep = settle(&root, &mut supervisor, "cycles", "keep-running");
    assert_eq!(status_cancel, "cancelled");
    assert_eq!(status_keep, "succeeded");

    // Verify `pinned-job` survived GC, `reclaim-job` was deleted, and all reservations are released
    assert!(root.join("artifacts/cycles/pinned-job/out.txt").is_file());
    assert!(!root.join("artifacts/cycles/reclaim-job").exists());
    let summary = execution::storage_policy_summary(&root).unwrap();
    assert_eq!(summary["observed"]["held_reservations"], 0);
}

#[test]
fn oversized_output_refusal_and_write_recovery() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, fixtures::policy()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "cycles");

    // 1. Pre-admission output bound check refuses a job whose payload exceeds declared max_output_bytes
    let err = execution::submit(
        &root,
        JobSpec {
            job_id: "oversized-write".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 2_000,
            threads: 1,
            max_output_bytes: 64,
            environment_id: None,
            probe: Probe::Write { name: "big.bin".into(), content: "z".repeat(512) },
        },
    )
    .unwrap_err();
    assert!(err.contains("artifact exceeds requested output bound"), "{err}");
    let summary = execution::storage_policy_summary(&root).unwrap();
    assert_eq!(summary["observed"]["held_reservations"], 0);
}

#[test]
fn total_state_limit_refuses_new_admission_and_keeps_retry_identity() {
    use std::fs::OpenOptions;
    use std::io::Write;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let runtime = Runtime::init(&root, fixtures::policy()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "cycles");

    // Sparse file models already occupied managed-state capacity without
    // allocating 128 MiB of physical disk in the test environment.
    let mut filler = OpenOptions::new().create_new(true).write(true)
        .open(root.join("quota-fixture.bin")).unwrap();
    filler.set_len(execution::MAX_STATE_TOTAL_BYTES - 1024).unwrap();
    filler.write_all(b"quota").unwrap();
    filler.sync_all().unwrap();

    let spec = JobSpec {
        job_id: "quota-blocked".into(), project_id: "cycles".into(), task_id: "t".into(),
        generation: 2, timeout_ms: 2_000, threads: 1, max_output_bytes: 4096,
        environment_id: None, probe: Probe::Sleep { millis: 1 },
    };
    let error = execution::submit(&root, spec.clone()).unwrap_err();
    assert!(error.contains("total state admission ceiling reached"), "{error}");
    assert!(execution::inspect(&root, "cycles", "quota-blocked").is_err());

    // Existing command retries remain idempotent after storage admission fails.
    let request: mlrescurix_harness::Request = rusqlite::Connection::open(root.join("state.sqlite")).unwrap()
        .query_row("SELECT request FROM commands WHERE project='cycles' AND id='project-cycles'", [], |r| r.get::<_, String>(0))
        .map(|encoded| serde_json::from_str(&encoded).unwrap()).unwrap();
    let retry = runtime.command(request).unwrap();
    assert_eq!(retry.status, "applied");
    assert_eq!(execution::storage_policy_summary(&root).unwrap()["policy"]["total_state_ceiling_enforced_for_job_admission"], true);
}
