#[path="pipeline_docker.rs"] mod fixtures;
use std::{fs, path::Path, time::Duration};
use serde_json::json;
use mlrescurix_harness::{
    Runtime,
    agents::{context::AgentContextManager, workspace::BranchWorkspaceManager},
    execution::{self, Supervisor, JobSpec, Probe, StorageBudgetConfig, StorageWriteKind},
};

fn settle(root: &Path, supervisor: &mut Supervisor, project: &str, id: &str) -> String {
    for _ in 0..600 {
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
            JobSpec { resources: Default::default(),
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
    // Accommodate additive v2 and P9.1-P9.4 schema metadata growth (tables and indexes) while retaining tight bound
    assert!(max_db_bytes <= 1152 * 1024, "SQLite metadata exceeded bound: {max_db_bytes}");
    let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let action_count:i64=db.query_row("SELECT count(*) FROM actions WHERE project='cycles'",[],|r|r.get(0)).unwrap();
    assert_eq!(action_count,76,"completed action retry fences must survive cleanup");
    // Report metadata growth. Finite cycles do not prove an indefinite total bound.


    let evidence_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/evidence/m6");
    if evidence_dir.exists() && (!evidence_dir.join("review-storage-cycles.json").exists() || std::env::var("UPDATE_M6_EVIDENCE").is_ok()) {
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
            JobSpec { resources: Default::default(),
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
        JobSpec { resources: Default::default(),
            job_id: "keep-running".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 15_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Sleep { millis: 2_000 },
        },
    )
    .unwrap();
    execution::submit(
        &root,
        JobSpec { resources: Default::default(),
            job_id: "cancel-midflight".into(),
            project_id: "cycles".into(),
            task_id: "t2".into(),
            generation: 2,
            timeout_ms: 15_000,
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
        JobSpec { resources: Default::default(),
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

    let spec = JobSpec { resources: Default::default(),
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

#[test]
fn p1_storage_exhaustion_enospc_and_reservation_fault_regressions() {
    use std::fs::OpenOptions;
    use std::io::Write;

    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let _rt = Runtime::init(&root, fixtures::policy()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "cycles");
    let binary = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));

    // -------------------------------------------------------------------------
    // 1. Initial storage policy inspection: verify all reservation headroom & budgets
    // -------------------------------------------------------------------------
    let summary = execution::storage_policy_summary(&root).unwrap();
    let policy = &summary["policy"];
    assert_eq!(policy["max_state_total_bytes"], execution::MAX_STATE_TOTAL_BYTES);
    assert_eq!(policy["emergency_recovery_headroom_bytes"], execution::EMERGENCY_RECOVERY_HEADROOM_BYTES);
    assert_eq!(policy["sqlite_wal_reserve_bytes"], execution::SQLITE_WAL_RESERVE_BYTES);
    assert_eq!(policy["max_artifact_budget_bytes"], execution::DEFAULT_MAX_ARTIFACT_BUDGET_BYTES);
    assert_eq!(policy["max_environment_budget_bytes"], execution::DEFAULT_MAX_ENVIRONMENT_BUDGET_BYTES);
    assert_eq!(policy["max_export_budget_bytes"], execution::DEFAULT_MAX_EXPORT_BUDGET_BYTES);
    assert_eq!(policy["max_workflow_budget_bytes"], execution::DEFAULT_MAX_WORKFLOW_BUDGET_BYTES);
    assert_eq!(policy["max_checkpoint_budget_bytes"], execution::DEFAULT_MAX_CHECKPOINT_BUDGET_BYTES);
    assert_eq!(policy["max_provider_receipt_bytes"], execution::DEFAULT_MAX_PROVIDER_RECEIPT_BYTES);

    // -------------------------------------------------------------------------
    // 2. Establish champion artifact & verified action fence
    // -------------------------------------------------------------------------
    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    execution::submit(
        &root,
        JobSpec { resources: Default::default(),
            job_id: "champ-1".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 2_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Write { name: "champion.bin".into(), content: "champion-payload-bytes".into() },
        },
    ).unwrap();
    assert_eq!(settle(&root, &mut supervisor, "cycles", "champ-1"), "succeeded");
    execution::pin_artifact(&root, "cycles", "champ-1", true).unwrap();
    let champ_path = root.join("artifacts/cycles/champ-1/champion.bin");
    assert!(champ_path.is_file());

    // -------------------------------------------------------------------------
    // 3. Scheduler ENOSPC recovery on disposable bounded storage
    // Worker probe exits with ENOSPC (code 28). Supervisor must clean up staging,
    // release held reservations and leases, and transition job to failed without
    // corrupting state or losing the pinned champion artifact.
    // -------------------------------------------------------------------------
    execution::submit(
        &root,
        JobSpec { resources: Default::default(),
            job_id: "enospc-probe-1".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 2_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Fail { code: 28 }, // POSIX ENOSPC exit status
        },
    ).unwrap();
    let status = settle(&root, &mut supervisor, "cycles", "enospc-probe-1");
    assert_eq!(status, "failed");
    let view = execution::inspect(&root, "cycles", "enospc-probe-1").unwrap();
    assert!(view.reason.as_deref().unwrap_or("").contains("28"), "expected exit code 28 in reason: {:?}", view.reason);
    // Staging directory must be cleaned up
    assert!(!root.join("artifacts/cycles/.enospc-probe-1.staging").exists());
    // Reservations must be completely settled and released
    let post_enospc_summary = execution::storage_policy_summary(&root).unwrap();
    assert_eq!(post_enospc_summary["observed"]["held_reservations"], 0);
    // Champion artifact remains intact
    assert!(champ_path.is_file());

    // Initialize branch workspace for cycles before saturation
    BranchWorkspaceManager::init_branch(&root, "cycles", "main", "initial-commit").unwrap();

    // -------------------------------------------------------------------------
    // 4. Non-silent reservation accounting across write paths under state saturation
    // Squeeze managed state into the emergency recovery headroom window
    // (e.g. 126 MiB, between usable 122 MiB and hard 128 MiB ceiling).
    // -------------------------------------------------------------------------
    let current_bytes = execution::total_state_bytes(&root).unwrap();
    let target_headroom_bytes = execution::MAX_STATE_TOTAL_BYTES.saturating_sub(execution::EMERGENCY_RECOVERY_HEADROOM_BYTES / 2);
    let mut filler = OpenOptions::new().create_new(true).write(true)
        .open(root.join("quota-saturation-fixture.bin")).unwrap();
    if target_headroom_bytes > current_bytes {
        filler.set_len(target_headroom_bytes - current_bytes).unwrap();
        filler.write_all(b"headroom").unwrap();
        filler.sync_all().unwrap();
    }

    // Verify all write paths are rejected non-silently with total state admission ceiling reached
    let err_job = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::JobAdmission, Some("cycles")).unwrap_err();
    assert!(err_job.contains("total state admission ceiling reached"), "{err_job}");

    let err_chk = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::Checkpoint, Some("cycles")).unwrap_err();
    assert!(err_chk.contains("total state admission ceiling reached"), "{err_chk}");

    let err_pr = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::ProviderReceipt, Some("cycles")).unwrap_err();
    assert!(err_pr.contains("total state admission ceiling reached"), "{err_pr}");

    let err_blob = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::ManagedBlob, Some("cycles")).unwrap_err();
    assert!(err_blob.contains("total state admission ceiling reached"), "{err_blob}");

    let err_asset = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::RuntimeAsset, Some("cycles")).unwrap_err();
    assert!(err_asset.contains("total state admission ceiling reached"), "{err_asset}");

    let err_bundle = execution::enforce_state_write_admission(&root, None, 4096, StorageWriteKind::BundleStaging, Some("cycles")).unwrap_err();
    assert!(err_bundle.contains("total state admission ceiling reached"), "{err_bundle}");

    // Agent checkpointing routed through reservation accounting must be rejected under saturation
    let mut agent_ctx = AgentContextManager::new("cycles", "t", "researcher", "system prompt");
    let err_agent_chk = agent_ctx.record_checkpoint(&root).unwrap_err();
    assert!(err_agent_chk.contains("total state admission ceiling reached"), "{err_agent_chk}");

    // Branch workspace manager file writes must be rejected under saturation
    let err_workspace = BranchWorkspaceManager::write_file(&root, "cycles", "main", "saturated_file.py", "x = 1").unwrap_err();
    assert!(err_workspace.contains("total state admission ceiling reached"), "{err_workspace}");

    // Dataset registration must be rejected under saturation
    let ds_temp = tempfile::NamedTempFile::new().unwrap();
    fs::write(ds_temp.path(), "a,b\n1,2\n").unwrap();
    let err_dataset = execution::register_dataset(&root, "cycles", "ds_sat", ds_temp.path()).unwrap_err();
    assert!(err_dataset.contains("total state admission ceiling reached"), "{err_dataset}");

    // Emergency recovery writes MUST succeed using the emergency headroom
    execution::enforce_state_write_admission(&root, None, 1024, StorageWriteKind::EmergencyRecovery, Some("cycles")).unwrap();

    // Verify emergency recovery hard ceiling when state exceeds 128 MiB
    let current_plus = execution::total_state_bytes(&root).unwrap();
    if execution::MAX_STATE_TOTAL_BYTES >= current_plus {
        let hard_filler = OpenOptions::new().create_new(true).write(true)
            .open(root.join("hard-ceiling-fixture.bin")).unwrap();
        hard_filler.set_len(execution::MAX_STATE_TOTAL_BYTES - current_plus + 1024).unwrap();
        hard_filler.sync_all().unwrap();
        let err_emerg = execution::enforce_state_write_admission(&root, None, 1024, StorageWriteKind::EmergencyRecovery, Some("cycles")).unwrap_err();
        assert!(err_emerg.contains("emergency recovery hard ceiling reached"), "{err_emerg}");
        drop(hard_filler);
        let _ = fs::remove_file(root.join("hard-ceiling-fixture.bin"));
    }

    // -------------------------------------------------------------------------
    // 5. Job cancellation under storage saturation using emergency recovery headroom
    // -------------------------------------------------------------------------
    // First, submit a long-running job and allow it to enter running state
    drop(filler);
    fs::remove_file(root.join("quota-saturation-fixture.bin")).unwrap();

    execution::submit(
        &root,
        JobSpec { resources: Default::default(),
            job_id: "running-to-cancel".into(),
            project_id: "cycles".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 5_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Sleep { millis: 4_000 },
        },
    ).unwrap();
    supervisor.tick().unwrap();
    let j_run = execution::inspect(&root, "cycles", "running-to-cancel").unwrap();
    assert_eq!(j_run.status, "running");

    // Squeeze the storage into the emergency headroom window while job is running
    let run_current = execution::total_state_bytes(&root).unwrap();
    let mut filler2 = OpenOptions::new().create_new(true).write(true)
        .open(root.join("quota-saturation-fixture2.bin")).unwrap();
    if target_headroom_bytes > run_current {
        filler2.set_len(target_headroom_bytes - run_current).unwrap();
        filler2.write_all(b"saturated-again").unwrap();
        filler2.sync_all().unwrap();
    }

    // Cancellation must succeed despite total state saturation due to emergency recovery headroom
    let cancel_view = execution::cancel(&root, "cycles", "running-to-cancel").unwrap();
    assert!(matches!(cancel_view.status.as_str(), "terminating" | "cancelled"));

    let settled_cancel = settle(&root, &mut supervisor, "cycles", "running-to-cancel");
    assert_eq!(settled_cancel, "cancelled");
    assert!(!root.join("artifacts/cycles/.running-to-cancel.staging").exists());

    // Champion artifact remains protected
    assert!(champ_path.is_file());

    // Clean up second filler
    drop(filler2);
    fs::remove_file(root.join("quota-saturation-fixture2.bin")).unwrap();
    drop(supervisor);

    // -------------------------------------------------------------------------
    // 6. Repeated restart and cleanup cycles under storage constraints
    // -------------------------------------------------------------------------
    for cycle in 0..5 {
        // Create an orphaned staging directory to simulate crash leftovers
        let orphan_stage = root.join("artifacts/cycles").join(format!(".orphan-{cycle}.staging"));
        fs::create_dir_all(&orphan_stage).unwrap();
        fs::write(orphan_stage.join("trash.bin"), "orphan-garbage").unwrap();

        // Restart supervisor
        let mut sup = Supervisor::start_with_executable(&root, 1, binary).unwrap();
        sup.tick().unwrap();

        // Run garbage collection
        let _ = execution::collect_artifacts(&root);
        drop(sup);

        // Verify orphaned staging was reclaimed
        assert!(!orphan_stage.exists(), "cycle {cycle}: orphan staging was not cleaned up");
        // Verify champion artifact survived all restart/cleanup cycles
        assert!(champ_path.is_file(), "cycle {cycle}: champion artifact lost");
    }

    // Verify SQLite integrity after 5 restart and cleanup cycles
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let check: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0)).unwrap();
    assert_eq!(check, "ok", "SQLite quick_check failed after restart cycles: {check}");
    drop(conn);

    // -------------------------------------------------------------------------
    // 7. Configurable category budgets via storage_budgets.json
    // -------------------------------------------------------------------------
    let custom_temp = tempfile::tempdir().unwrap();
    let custom_root = custom_temp.path().join("state");
    Runtime::init(&custom_root, fixtures::policy()).unwrap();
    execution::migrate(&custom_root).unwrap();

    let custom_cfg = json!({
        "max_artifact_budget_bytes": 1048576,
        "max_environment_budget_bytes": 2097152,
        "max_export_budget_bytes": 1048576,
        "max_workflow_budget_bytes": 4194304,
        "max_checkpoint_budget_bytes": 524288,
        "max_provider_receipt_bytes": 262144
    });
    fs::write(custom_root.join("storage_budgets.json"), serde_json::to_vec_pretty(&custom_cfg).unwrap()).unwrap();

    let loaded_cfg = StorageBudgetConfig::load(&custom_root);
    assert_eq!(loaded_cfg.max_artifact_budget_bytes, 1048576);
    assert_eq!(loaded_cfg.max_checkpoint_budget_bytes, 524288);
    assert_eq!(loaded_cfg.max_provider_receipt_bytes, 262144);
    assert_eq!(loaded_cfg.max_environment_budget_bytes, 2097152);
    assert_eq!(loaded_cfg.max_export_budget_bytes, 1048576);
    assert_eq!(loaded_cfg.max_workflow_budget_bytes, 4194304);

    let custom_summary = execution::storage_policy_summary(&custom_root).unwrap();
    assert_eq!(custom_summary["policy"]["max_checkpoint_budget_bytes"], 524288);
    assert_eq!(custom_summary["policy"]["max_provider_receipt_bytes"], 262144);

    // Category budget enforcement: 600_000 exceeds max_checkpoint_budget_bytes (524_288)
    let cat_err = execution::enforce_state_write_admission(&custom_root, None, 600_000, StorageWriteKind::Checkpoint, Some("cycles")).unwrap_err();
    assert!(cat_err.contains("checkpoint storage budget reached"), "{cat_err}");

    // But a write within category budget succeeds
    execution::enforce_state_write_admission(&custom_root, None, 100_000, StorageWriteKind::Checkpoint, Some("cycles")).unwrap();
}

#[test]
fn p1_archival_tombstone_fencing_and_publication_recovery_fault_regressions() {
    use sha2::Digest;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let rt = Runtime::init(&root, fixtures::policy()).unwrap();
    execution::migrate(&root).unwrap();
    fixtures::setup(&root, "fencing");
    let binary = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));

    // -------------------------------------------------------------------------
    // 1. Unpinned terminal job artifact collection creates artifact tombstones
    // -------------------------------------------------------------------------
    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    let spec1 = JobSpec { resources: Default::default(),
        job_id: "unpinned-job".into(),
        project_id: "fencing".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 2_000,
        threads: 1,
        max_output_bytes: 4096,
        environment_id: None,
        probe: Probe::Write { name: "metric.txt".into(), content: "loss=0.042".into() },
    };
    execution::submit(&root, spec1).unwrap();
    assert_eq!(settle(&root, &mut supervisor, "fencing", "unpinned-job"), "succeeded");

    let art_path = root.join("artifacts/fencing/unpinned-job/metric.txt");
    assert!(art_path.is_file());

    // Pin another job to verify pinned artifacts survive collect_artifacts
    let spec2 = JobSpec { resources: Default::default(),
        job_id: "pinned-job".into(),
        project_id: "fencing".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 2_000,
        threads: 1,
        max_output_bytes: 4096,
        environment_id: None,
        probe: Probe::Write { name: "model.bin".into(), content: "weights-data".into() },
    };
    execution::submit(&root, spec2).unwrap();
    assert_eq!(settle(&root, &mut supervisor, "fencing", "pinned-job"), "succeeded");
    execution::pin_artifact(&root, "fencing", "pinned-job", true).unwrap();

    drop(supervisor);

    // Run collect_artifacts
    let reclaimed = execution::collect_artifacts(&root).unwrap();
    assert!(reclaimed > 0, "Reclaimed unpinned artifact bytes");

    // Unpinned artifact was removed from disk
    assert!(!root.join("artifacts/fencing/unpinned-job").exists());
    // Pinned artifact was preserved on disk
    assert!(root.join("artifacts/fencing/pinned-job/model.bin").is_file());

    // Verify artifact tombstone was recorded with SHA-256
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let (entity_kind, entity_id, parent_id, identity_hash, reclaimed_bytes): (String, String, String, String, i64) = db.query_row(
        "SELECT entity_kind, entity_id, parent_id, identity_hash, reclaimed_bytes FROM tombstones WHERE project='fencing' AND entity_kind='artifact' AND parent_id='unpinned-job'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    ).unwrap();
    assert_eq!(entity_kind, "artifact");
    assert_eq!(entity_id, "unpinned-job:metric.txt");
    assert_eq!(parent_id, "unpinned-job");
    assert_eq!(identity_hash, format!("{:x}", sha2::Sha256::digest(b"loss=0.042")));
    assert_eq!(reclaimed_bytes, 10);

    // -------------------------------------------------------------------------
    // 2. Terminal job pruning beyond MAX_RETAINED_TERMINAL_JOBS creates job tombstones
    // -------------------------------------------------------------------------
    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    for i in 0..25 {
        let j_id = format!("prune-target-{i}");
        let spec = JobSpec { resources: Default::default(),
            job_id: j_id.clone(),
            project_id: "fencing".into(),
            task_id: "t".into(),
            generation: 2,
            timeout_ms: 2_000,
            threads: 1,
            max_output_bytes: 4096,
            environment_id: None,
            probe: Probe::Fail { code: 1 },
        };
        execution::submit(&root, spec).unwrap();
        assert_eq!(settle(&root, &mut supervisor, "fencing", &j_id), "failed");
    }
    drop(supervisor);

    // Collect artifacts and prune excess terminal jobs
    execution::collect_artifacts(&root).unwrap();

    let job_tombstones: i64 = db.query_row(
        "SELECT count(*) FROM tombstones WHERE project='fencing' AND entity_kind='job'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(job_tombstones >= 1, "At least 1 older terminal job must be tombstoned after exceeding MAX_RETAINED_TERMINAL_JOBS_PER_PROJECT");

    // Verify pinned job is NEVER pruned from jobs table
    let pinned_exists: bool = db.query_row(
        "SELECT EXISTS(SELECT 1 FROM jobs WHERE project='fencing' AND id='pinned-job')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(pinned_exists, "Pinned job must strictly survive job table pruning");

    // -------------------------------------------------------------------------
    // 3. Compact command retry fences (body hashed, exact retry succeeds, altered rejects)
    // -------------------------------------------------------------------------
    // Submit 30 steer commands (exceeding MAX_ACTIVE_COMMAND_RAW_BODIES = 24)
    let steer_0 = mlrescurix_harness::Request {
        command_id: "cmd-steer-0".into(),
        project_id: "fencing".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::Steer { intent: "intent-payload-0".into() },
    };
    let orig_receipt = rt.command(steer_0.clone()).unwrap();
    assert_eq!(orig_receipt.status, "applied");

    for i in 1..30 {
        let req = mlrescurix_harness::Request {
            command_id: format!("cmd-steer-{i}"),
            project_id: "fencing".into(),
            expected_revision: None,
            operation: mlrescurix_harness::Operation::Steer { intent: format!("intent-payload-{i}") },
        };
        assert_eq!(rt.command(req).unwrap().status, "applied");
    }

    // Run collect_artifacts to trigger command body compaction
    execution::collect_artifacts(&root).unwrap();

    // Verify older command body was compacted to hash:
    let stored_req: String = db.query_row(
        "SELECT request FROM commands WHERE project='fencing' AND id='cmd-steer-0'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(stored_req.starts_with("hash:"), "Older command body must be compacted to hash, got: {stored_req}");

    // Retry with EXACT payload: succeeds idempotently and returns original receipt
    let retry_receipt = rt.command(steer_0.clone()).unwrap();
    assert_eq!(retry_receipt.status, "applied");
    assert_eq!(retry_receipt.revision, orig_receipt.revision);

    // Retry with ALTERED payload: rejected with payload mismatch
    let altered_req = mlrescurix_harness::Request {
        command_id: "cmd-steer-0".into(),
        project_id: "fencing".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::Steer { intent: "altered-intent-conflict".into() },
    };
    let err = rt.command(altered_req).unwrap_err();
    assert!(err.contains("command identity reused with different payload"), "Altered retry must fail: {err}");

    // -------------------------------------------------------------------------
    // 4. Action compaction & action_fences retry durability
    // -------------------------------------------------------------------------
    // Check that action_fences and tombstones table exists and summary reports them
    let summary = execution::storage_policy_summary(&root).unwrap();
    assert!(summary["observed"]["tombstones_rows"].as_i64().unwrap() > 0);

    // Create fresh task t2 for fenced job admission after earlier steering invalidated task t
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    rt.command(mlrescurix_harness::Request {
        command_id: "create-t2".into(),
        project_id: "fencing".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::CreateTask {
            task_id: "t2".into(),
            branch_id: "main".into(),
            context: mlrescurix_harness::contracts::Context {
                role_id: "lead".into(), role_version: "v1".into(), prompt_hash: "prompt".into(),
                model: None, phase: "search".into(), hypothesis: None, base_hash: "base".into(),
                code_hash: "code".into(), evidence: vec![], artifacts: vec![], pending_questions: vec![],
                calls: vec![], jobs: vec![], next_action: "search".into(), stop_conditions: vec![]
            },
        },
    }).unwrap();
    rt.command(mlrescurix_harness::Request {
        command_id: "claim-t2".into(),
        project_id: "fencing".into(),
        expected_revision: None,
        operation: mlrescurix_harness::Operation::Claim {
            task_id: "t2".into(), owner: "fixture".into(), lease_id: "l2".into(),
            now_ms: now, expires_ms: now + 600_000, expected_generation: 1
        },
    }).unwrap();

    let current_intent_rev: u64 = db.query_row("SELECT intent_revision FROM projects WHERE id='fencing'", [], |r| r.get(0)).unwrap();

    // Verify Supervisor::admit blocks admission if job action ID exists in action_fences
    db.execute(
        "INSERT OR REPLACE INTO action_fences(project, action_id, task_id, generation, intent_revision, lease_id, input_hash, status, receipt_json, tombstoned_ms)
         VALUES('fencing', 'job.fenced-job-id', 't2', 2, ?1, 'l2', 'hash:dummy', 'succeeded', NULL, 1000)",
        [current_intent_rev],
    ).unwrap();

    let mut supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    let fenced_spec = JobSpec { resources: Default::default(),
        job_id: "fenced-job-id".into(),
        project_id: "fencing".into(),
        task_id: "t2".into(),
        generation: 2,
        timeout_ms: 2_000,
        threads: 1,
        max_output_bytes: 4096,
        environment_id: None,
        probe: Probe::Sleep { millis: 1 },
    };
    execution::submit(&root, fenced_spec).unwrap();
    supervisor.tick().unwrap(); // admit tick
    let j_view = execution::inspect(&root, "fencing", "fenced-job-id").unwrap();
    assert_eq!(j_view.status, "rejected");
    assert_eq!(j_view.reason.as_deref(), Some("job action identity already exists"));
    drop(supervisor);

    // -------------------------------------------------------------------------
    // 5. Interrupted publication recovery: crash right after directory rename
    // -------------------------------------------------------------------------
    // Simulate worker that produced output and supervisor that crashed right after
    // fs::rename(&stage, &final_dir) but before finish()
    let recovery_job = "interrupted-pub-job";
    let final_dir = root.join("artifacts/fencing").join(recovery_job);
    fs::create_dir_all(&final_dir).unwrap();
    fs::write(final_dir.join("result.csv"), "col1,col2\n1,2\n").unwrap();

    // Call Supervisor::publish directly via a simulated recovering JobView
    let supervisor = Supervisor::start_with_executable(&root, 1, binary).unwrap();
    let recovery_spec = JobSpec { resources: Default::default(),
        job_id: recovery_job.into(),
        project_id: "fencing".into(),
        task_id: "t2".into(),
        generation: 2,
        timeout_ms: 2_000,
        threads: 1,
        max_output_bytes: 4096,
        environment_id: None,
        probe: Probe::Write { name: "result.csv".into(), content: "col1,col2\n1,2\n".into() },
    };
    let recovery_view = execution::JobView { resource_receipt:None,
        spec: recovery_spec,
        status: "running".into(),
        reason: None,
        pid: None,
        submitted_ms: 0,
        started_ms: Some(100),
        ended_ms: None,
        artifacts: vec![],
        telemetry: None,
    };
    // publish() must adopt the existing valid final_dir without failing
    let recovered_arts = supervisor.publish(&recovery_view).unwrap();
    assert_eq!(recovered_arts.len(), 1);
    assert_eq!(recovered_arts[0]["name"], "result.csv");
    drop(supervisor);

    // -------------------------------------------------------------------------
    // 6. Leaked bundle staging clean up by collect_artifacts
    // -------------------------------------------------------------------------
    let leaked_bundle_stage = root.join("staging/bundle_verification_leaked_test");
    fs::create_dir_all(&leaked_bundle_stage).unwrap();
    fs::write(leaked_bundle_stage.join("temp.txt"), "leftover").unwrap();

    let leaked_wf_stage = root.join("workflows/fencing/.bundle_stage");
    fs::create_dir_all(&leaked_wf_stage).unwrap();
    fs::write(leaked_wf_stage.join("staged.bin"), "leftover").unwrap();

    let reclaimed_staging = execution::collect_artifacts(&root).unwrap();
    assert!(reclaimed_staging > 0);
    assert!(!leaked_bundle_stage.exists(), "Leaked bundle verification staging must be purged");
    assert!(!leaked_wf_stage.exists(), "Leaked workflow bundle staging must be purged");

    // -------------------------------------------------------------------------
    // 7. Interrupted delivery publication crash recovery in recovery.rs
    // -------------------------------------------------------------------------
    // Set project to 'finalizing' to test unconfirmed publication recovery
    db.execute("UPDATE projects SET state='finalizing' WHERE id='fencing'", []).unwrap();
    let tool_id = mlrescurix_harness::agents::recovery::ToolIdentity {
        project: "fencing",
        task: "t2",
        turn: 1,
        call_id: "call-finalize-crash",
        name: "request_finalization",
        arguments: "{\"candidate_id\":\"cand-test\",\"finalize_job_id\":\"job-missing-crash\"}",
        branch: "main",
    };
    let canon_str = serde_json::to_string(&json!({"candidate_id":"cand-test","finalize_job_id":"job-missing-crash"})).unwrap();
    let canon_hash = format!("{:x}", sha2::Sha256::digest(canon_str.as_bytes()));
    let intent = json!({
        "kind": "delivery_publication",
        "operation": "request_finalization",
        "project": "fencing",
        "task": "t2",
        "turn": 1,
        "call_id": "call-finalize-crash",
        "finalize_job_id": "job-missing-crash",
        "candidate_id": "cand-test",
        "generation": 2,
        "intent_revision": current_intent_rev,
        "canonical_arguments": {"candidate_id":"cand-test","finalize_job_id":"job-missing-crash"},
        "canonical_arguments_sha256": canon_hash,
    });

    let recon_result = mlrescurix_harness::agents::recovery::reconcile_delivery_publication(
        &root,
        &db,
        &tool_id,
        true, // existed = true: process crashed after intent was written
        &intent,
        || Ok("never called".into()),
    ).unwrap();

    let parsed_res: serde_json::Value = serde_json::from_str(&recon_result).unwrap();
    assert_eq!(parsed_res["status"], "rejected");
    assert_eq!(parsed_res["reason"], "unconfirmed_publication_crash");
    assert_eq!(parsed_res["effect_outcome"], "uncertain");

    // Crucial: project state must be safely recovered back to 'created'
    let proj_state: String = db.query_row("SELECT state FROM projects WHERE id='fencing'", [], |r| r.get(0)).unwrap();
    assert_eq!(proj_state, "created", "Project stuck in 'finalizing' must be reset to 'created' upon crash recovery");
}

#[test]
#[ignore="explicit disposable mounted bounded volume via MLRESCURIX_ENOSPC_VOLUME required"]
fn p1_real_enospc_on_disposable_volume_preserves_fences_across_restarts() {
    use std::{io::Write,os::unix::fs::PermissionsExt};
    let volume=std::path::PathBuf::from(std::env::var("MLRESCURIX_ENOSPC_VOLUME").unwrap());
    assert!(volume.starts_with("/private/tmp/"));
    assert!(volume.join("MLRESCURIX_DISPOSABLE").is_file(),"explicit disposable marker required");
    let host=tempfile::tempdir().unwrap();let root=volume.join("state");
    let rt=Runtime::init(&root,fixtures::policy()).unwrap();
    fs::write(root.join("storage_budgets.json"),json!({"disk_headroom_bytes":0}).to_string()).unwrap();
    execution::migrate(&root).unwrap();fixtures::setup(&root,"cycles");
    let binary=Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let champion=JobSpec{resources:Default::default(),job_id:"champion".into(),project_id:"cycles".into(),task_id:"t".into(),generation:2,timeout_ms:2000,threads:1,max_output_bytes:4096,environment_id:None,probe:Probe::Write{name:"model".into(),content:"protected".into()}};
    execution::submit(&root,champion).unwrap();
    let mut supervisor=Supervisor::start_with_executable(&root,1,binary).unwrap();assert_eq!(settle(&root,&mut supervisor,"cycles","champion"),"succeeded");execution::pin_artifact(&root,"cycles","champion",true).unwrap();drop(supervisor);
    let script=host.path().join("delayed-worker");let release=host.path().join("release");let exit_receipt=host.path().join("exit");
    // Trusted control-plane fault fixture only; no host scientific execution.
    fs::write(&script,format!("#!/bin/sh\nwhile [ ! -e '{}' ]; do sleep 0.01; done\n'{}' \"$@\"\ncode=$?\necho \"$code\" > '{}'\nexit \"$code\"\n",release.display(),binary.display(),exit_receipt.display())).unwrap();fs::set_permissions(&script,fs::Permissions::from_mode(0o700)).unwrap();
    for cycle in 0..3 {
        if release.exists(){fs::remove_file(&release).unwrap();}
        let id=format!("real-enospc-{cycle}");
        execution::submit(&root,JobSpec{resources:Default::default(),job_id:id.clone(),project_id:"cycles".into(),task_id:"t".into(),generation:2,timeout_ms:3000,threads:1,max_output_bytes:32*1024,environment_id:None,probe:Probe::Write{name:"payload".into(),content:"x".repeat(32*1024)}}).unwrap();
        let mut supervisor=Supervisor::start_with_executable(&root,1,&script).unwrap();supervisor.tick().unwrap();
        let filler=volume.join("filler");let mut file=fs::File::create(&filler).unwrap();let block=vec![0u8;64*1024];
        loop {match file.write_all(&block){Ok(())=>{},Err(e)=>{assert_eq!(e.raw_os_error(),Some(28));break;}}}
        // Fill the final allocation units too; 32 KiB probe cannot fit.
        while file.write_all(&[0u8;512]).is_ok() {}
        drop(file);if exit_receipt.exists(){fs::remove_file(&exit_receipt).unwrap();}fs::write(&release,"go").unwrap();
        let log=root.join("logs/cycles").join(format!("{id}.log"));
        let mut observed=false;
        for _ in 0..200 {if exit_receipt.is_file(){observed=true;break;}std::thread::sleep(Duration::from_millis(10));}
        assert!(observed,"actual worker must fail on bounded volume: {}",fs::read_to_string(&log).unwrap_or_default());
        assert_ne!(fs::read_to_string(&exit_receipt).unwrap().trim(),"0");
        assert!(fs::metadata(root.join("artifacts/cycles").join(format!(".{id}.staging/payload"))).map(|m|m.len()<32*1024).unwrap_or(true));
        // Free disposable filler before durable recovery writes, retaining state.
        fs::remove_file(&filler).unwrap();
        if cycle%2==0 {drop(supervisor);supervisor=Supervisor::start_with_executable(&root,1,binary).unwrap();}
        let status=settle(&root,&mut supervisor,"cycles",&id);assert!(matches!(status.as_str(),"failed"|"interrupted"));drop(supervisor);
        assert_eq!(fs::read_to_string(root.join("artifacts/cycles/champion/model")).unwrap(),"protected");
        let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
        assert_eq!(db.query_row("SELECT count(*) FROM reservations WHERE held=1",[],|r|r.get::<_,i64>(0)).unwrap(),0);
        assert_eq!(db.query_row("SELECT count(*) FROM charges WHERE action=?1",[format!("job.{id}")],|r|r.get::<_,i64>(0)).unwrap(),1);
        assert_eq!(db.query_row("PRAGMA integrity_check",[],|r|r.get::<_,String>(0)).unwrap(),"ok");
        assert_eq!(db.query_row("PRAGMA user_version",[],|r|r.get::<_,i64>(0)).unwrap(),10);
        execution::collect_artifacts(&root).unwrap();
    }
    let retry=rt.command(mlrescurix_harness::core::Request{command_id:"project-cycles".into(),project_id:"cycles".into(),expected_revision:None,operation:mlrescurix_harness::core::Operation::CreateProject{intent:"Docker pipeline integration fixture".into(),finalization_id:"f-cycles".into(),policy:fixtures::policy()}}).unwrap();assert_eq!(retry.status,"applied");
}

#[test]
fn p1_concurrent_workspace_writers_cannot_share_the_same_remaining_room() {
    let temp=tempfile::tempdir().unwrap();let root=temp.path().join("state");
    Runtime::init(&root,fixtures::policy()).unwrap();execution::migrate(&root).unwrap();fixtures::setup(&root,"writers");
    for branch in ["left","right"] {BranchWorkspaceManager::init_branch(&root,"writers",branch,"base").unwrap();}
    let limit=execution::total_state_bytes(&root).unwrap()+700*1024;
    fs::write(root.join("storage_budgets.json"),json!({"max_state_total_bytes":limit,"disk_headroom_bytes":0,"emergency_recovery_headroom_bytes":0,"sqlite_wal_reserve_bytes":0}).to_string()).unwrap();
    let barrier=std::sync::Arc::new(std::sync::Barrier::new(2));
    let mut threads=vec![];
    for branch in ["left","right"] {let root=root.clone();let barrier=barrier.clone();threads.push(std::thread::spawn(move||{barrier.wait();BranchWorkspaceManager::write_file(&root,"writers",branch,"payload",&"x".repeat(512*1024))}));}
    let successes=threads.into_iter().map(|t|t.join().unwrap().is_ok()).filter(|ok|*ok).count();assert_eq!(successes,1);
    assert!(execution::total_state_bytes(&root).unwrap()<=limit);
    fs::write(root.join("storage_budgets.json"),json!({"max_state_total_bytes":512*1024*1024}).to_string()).unwrap();
    assert_eq!(execution::StorageBudgetConfig::load(&root).max_state_total_bytes,128*1024*1024);
}

#[test]
fn p1_supervisor_drains_worker_output_without_exceeding_shared_log_cap() {
    use std::os::unix::fs::PermissionsExt;
    let temp=tempfile::tempdir().unwrap();let root=temp.path().join("state");Runtime::init(&root,fixtures::policy()).unwrap();execution::migrate(&root).unwrap();fixtures::setup(&root,"logs");
    // Fixed trusted control probe fixture, never candidate/scientific host code.
    let script=temp.path().join("log-probe");fs::write(&script,"#!/bin/sh\nhead -c 2097152 /dev/zero\nhead -c 2097152 /dev/zero >&2\n").unwrap();fs::set_permissions(&script,fs::Permissions::from_mode(0o700)).unwrap();
    execution::submit(&root,JobSpec{resources:Default::default(),job_id:"bounded-log".into(),project_id:"logs".into(),task_id:"t".into(),generation:2,timeout_ms:5000,threads:1,max_output_bytes:4096,environment_id:None,probe:Probe::Sleep{millis:1}}).unwrap();
    let mut supervisor=Supervisor::start_with_executable(&root,1,&script).unwrap();assert_eq!(settle(&root,&mut supervisor,"logs","bounded-log"),"succeeded");
    assert_eq!(fs::metadata(root.join("logs/logs/bounded-log.log")).unwrap().len(),1024*1024);
    assert_eq!(execution::storage_policy_summary(&root).unwrap()["observed"]["held_reservations"],0);
}
