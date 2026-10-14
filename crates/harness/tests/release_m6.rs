use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::Duration,
};
use serde_json::json;
use mlrescurix_harness::{
    bundle::verify_bundle,
    contracts::*,
    core::*,
    execution::{self, JobSpec, Probe, Supervisor},
    search::{self, SearchSpec},
    storage::Runtime,
};

fn m1policy(tokens: u32, workers: u32) -> Policy {
    let mut budget = BTreeMap::new();
    budget.insert("tokens".into(), tokens as i64 * 1000);
    budget.insert("calls".into(), tokens as i64);
    budget.insert("wall_ms".into(), tokens as i64 * 10000);
    let mut capacity = BTreeMap::new();
    capacity.insert("workers".into(), workers as i64);
    Policy {
        budget,
        capacity,
        max_tasks: 1024,
    }
}

fn create_m1(r: &Runtime, p: &str, budget: i64) {
    let mut b = BTreeMap::new();
    b.insert("tokens".into(), budget * 1000);
    b.insert("calls".into(), budget);
    b.insert("wall_ms".into(), budget * 10000);
    let mut cap = BTreeMap::new();
    cap.insert("workers".into(), 4);
    let pol = Policy {
        budget: b,
        capacity: cap,
        max_tasks: 1024,
    };
    r.command(Request {
        command_id: format!("init-{p}"), project_id: p.into(), expected_revision: None,
        operation: Operation::CreateProject {
            intent: "M6 release fixture".into(),
            finalization_id: format!("final-{p}"),
            policy: pol,
        },
    }).unwrap();
}

fn chrono_free_now() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

fn execution_task(r: &Runtime, p: &str, t: &str) {
    let _ = r.command(Request {
        command_id: format!("branch-{p}-{t}"), project_id: p.into(), expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "main".into(), base_hash: "base".into(), code_hash: "code".into(),
        },
    });
    r.command(Request {
        command_id: format!("task-{p}-{t}"), project_id: p.into(), expected_revision: None,
        operation: Operation::CreateTask {
            task_id: t.into(), branch_id: "main".into(),
            context: Context {
                role_id: "lead".into(), role_version: "v1".into(), prompt_hash: "prompt".into(),
                model: None, phase: "search".into(), hypothesis: None, base_hash: "base".into(),
                code_hash: "code".into(), evidence: vec![], artifacts: vec![], pending_questions: vec![],
                calls: vec![], jobs: vec![], next_action: "search".into(), stop_conditions: vec![],
            },
        },
    }).unwrap();
    let now = chrono_free_now();
    r.command(Request {
        command_id: format!("claim-{p}-{t}"), project_id: p.into(), expected_revision: None,
        operation: Operation::Claim {
            task_id: t.into(), owner: "local-worker".into(), lease_id: format!("lease{t}"),
            now_ms: now, expires_ms: now + 3_600_000, expected_generation: 1,
        },
    }).unwrap();
}

fn setup_test_env(root: &Path, id: &str) {
    let image=std::env::var("MLRESCURIX_BUNDLE_IMAGE").expect("legacy scientific fixture requires pinned Docker; host execution prohibited");
    execution::register_container_environment(root,id,&image).unwrap();
}

fn run_job_to_completion(sup: &mut Supervisor, root: &Path, project: &str, job_id: &str) {
    for _ in 0..400 {
        sup.tick().unwrap();
        let j = execution::inspect(root, project, job_id).unwrap();
        if !["queued", "running", "terminating", "recovering"].contains(&j.status.as_str()) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("job {} did not complete: {}", job_id, execution::inspect(root, project, job_id).unwrap().status);
}

#[test]
#[ignore="historical host scientific fixture; replaced by explicit digest-pinned Docker pipeline tests; does not satisfy content-bound finalization"]
fn m6_bundle_verification_valid_bundle_and_tampering_detection() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&r, "p-bundle", 100);
    execution_task(&r, "p-bundle", "t-bundle");
    setup_test_env(&root, "scientific");

    let csv = d.path().join("dataset.csv");
    let mut data = "feature1,feature2,target\n".to_string();
    for i in 0..60 {
        let t = if i % 2 == 0 { 0 } else { 1 };
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * 0.2;
        data.push_str(&format!("{f1:.2},{f2:.2},{t}\n"));
    }
    fs::write(&csv, &data).unwrap();
    execution::register_dataset(&root, "p-bundle", "data-b", &csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    // 1. Audit
    let audit_job = JobSpec { resources: Default::default(),
        job_id: "job-audit-b".into(), project_id: "p-bundle".into(), task_id: "t-bundle".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "audit".into(), dataset_id: "data-b".into(),
            spec: Some(json!({
                "project_id": "p-bundle", "target": "target", "task": "binary_classification",
                "metric": "balanced_accuracy", "split": "stratified", "inference_batch_size": 16,
                "inference_repetitions": 3,
            })),
            candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, audit_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-bundle", "job-audit-b");

    // 2. Baselines
    let baselines_job = JobSpec { resources: Default::default(),
        job_id: "job-baselines-b".into(), project_id: "p-bundle".into(), task_id: "t-bundle".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "baselines".into(), dataset_id: "data-b".into(),
            spec: None, candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, baselines_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-bundle", "job-baselines-b");

    // 3. Search
    let spec = SearchSpec {
        session_id: "sess-b".into(), project_id: "p-bundle".into(), dataset_id: "data-b".into(),
        target: "target".into(), task: "binary_classification".into(), metric: "balanced_accuracy".into(),
        direction: "maximize".into(), split: "stratified".into(), fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()], max_trials: 8, max_screen_trials: 2, max_full_trials: 1,
        search_wall_ms: 300_000, job_timeout_ms: 30_000, finalization_allowance_ms: 30_000,
        max_repeats: 3, tie_tolerance: 0.005, concurrency: 1, max_retries: 2, constraints: None,
    };
    search::init_search(&root, spec).unwrap();
    let selection = search::execute_search(&root, "p-bundle", "sess-b", "t-bundle", 2, &mut sup, Some("scientific")).unwrap();
    assert_eq!(selection.status, "apex_selected");
    let apex_id = selection.selected_candidate_id.expect("Apex candidate must be selected");

    // 4. Finalize
    let finalize_job = JobSpec { resources: Default::default(),
        job_id: "job-finalize-b".into(), project_id: "p-bundle".into(), task_id: "t-bundle".into(),
        generation: 2, timeout_ms: 60_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "finalize".into(), dataset_id: "data-b".into(), spec: None,
            candidate_id: Some(apex_id.clone()), test_access_id: Some("test-gate-bundle".into()),
        },
    };
    execution::submit(&root, finalize_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-bundle", "job-finalize-b");

    let bundle_path = root.join("workflows/p-bundle/bundle");
    assert!(bundle_path.is_dir());

    // 5. Positive verification test
    let report = verify_bundle(&bundle_path, false).unwrap();
    assert_eq!(report.status, "integrity_verified");
    assert_eq!(report.format, "mlrescurix.bundle.v1");
    assert_eq!(report.candidate_id, apex_id);
    assert_eq!(report.parity_check, "skipped_by_request");
    assert!(report.manifest_verified);
    assert!(report.constraints_verified);
    assert!(report.total_bytes > 0);

    // 6. Anti-tampering check: modified file content fails explicitly
    let model_file = bundle_path.join("frozen_model.joblib");
    let orig_model_bytes = fs::read(&model_file).unwrap();
    let mut tampered = orig_model_bytes.clone();
    tampered[0] ^= 0xFF;
    fs::write(&model_file, &tampered).unwrap();

    let tampered_res = verify_bundle(&bundle_path, false);
    assert!(tampered_res.is_err(), "Modified file must fail verification");
    let err = tampered_res.unwrap_err();
    assert!(err.contains("tampered bundle") || err.contains("mismatch"), "Error must indicate tampering: {err}");

    // Restore original model bytes
    fs::write(&model_file, &orig_model_bytes).unwrap();

    // 7. Anti-tampering check: extra untracked file fails explicitly
    let untracked_file = bundle_path.join("untracked_trojan.py");
    fs::write(&untracked_file, "print('trojan')").unwrap();
    let untracked_res = verify_bundle(&bundle_path, false);
    assert!(untracked_res.is_err(), "Untracked file must fail verification");
    let err = untracked_res.unwrap_err();
    assert!(err.contains("untracked file") && err.contains("found in bundle"), "Expected untracked file error: {err}");
    let _ = fs::remove_file(&untracked_file);

    // 8. Positive check passes again after removing untracked file
    let post_restore = verify_bundle(&bundle_path, false);
    assert!(post_restore.is_ok(), "Post restore verification failed: {:?}", post_restore.err());
}

#[test]
#[ignore="historical delivery fixture lacks content-bound approval; real bundle integrity/parity now covered in bundle_scientific_docker"]
fn m6_exported_bundle_integrity_without_executing_candidate() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&r, "p-inf", 100);
    execution_task(&r, "p-inf", "t-inf");
    setup_test_env(&root, "scientific");

    let csv = d.path().join("dataset.csv");
    let mut data = "feature1,feature2,target\n".to_string();
    for i in 0..60 {
        let t = if i % 2 == 0 { 0 } else { 1 };
        data.push_str(&format!("{:.2},{:.2},{t}\n", (i as f64) * 0.1, (i as f64) * 0.2));
    }
    fs::write(&csv, &data).unwrap();
    execution::register_dataset(&root, "p-inf", "data-inf", &csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let audit_job = JobSpec { resources: Default::default(),
        job_id: "j-aud".into(), project_id: "p-inf".into(), task_id: "t-inf".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "audit".into(), dataset_id: "data-inf".into(),
            spec: Some(json!({
                "project_id": "p-inf", "target": "target", "task": "binary_classification",
                "metric": "balanced_accuracy", "split": "stratified", "inference_batch_size": 16,
                "inference_repetitions": 3,
            })),
            candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, audit_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-inf", "j-aud");

    let baselines_job = JobSpec { resources: Default::default(),
        job_id: "j-base".into(), project_id: "p-inf".into(), task_id: "t-inf".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "baselines".into(), dataset_id: "data-inf".into(), spec: None, candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, baselines_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-inf", "j-base");

    let finalize_job = JobSpec { resources: Default::default(),
        job_id: "j-fin".into(), project_id: "p-inf".into(), task_id: "t-inf".into(),
        generation: 2, timeout_ms: 60_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "finalize".into(), dataset_id: "data-inf".into(), spec: None,
            candidate_id: Some("linear".into()), test_access_id: Some("test-gate-inf".into()),
        },
    };
    execution::submit(&root, finalize_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-inf", "j-fin");

    let bundle_path = root.join("workflows/p-inf/bundle");
    let rep = verify_bundle(&bundle_path, false).unwrap();
    assert_eq!(rep.parity_check, "skipped_by_request");
}

#[test]
fn m6_scientific_lineage_registration_and_fencing() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(50, 2)).unwrap();
    create_m1(&r, "p-lineage", 50);
    execution_task(&r, "p-lineage", "t-lineage");

    let csv = d.path().join("dataset.csv");
    fs::write(&csv, "col1,col2,target\n1,2,0\n3,4,1\n").unwrap();
    execution::register_dataset(&root, "p-lineage", "data-lin", &csv).unwrap();

    // 1. Register Feature Transform
    let tf_spec = json!({
        "numeric_scaler": "robust",
        "missing_indicators": true,
        "interactions": true,
    });
    let tf_res = execution::register_feature_transform(&root, "p-lineage", "tf-1", &tf_spec, "data-lin").unwrap();
    assert_eq!(tf_res["status"], "registered");
    assert_eq!(tf_res["transform_id"], "tf-1");

    // Negative gate: Unknown dataset rejected
    let tf_err = execution::register_feature_transform(&root, "p-lineage", "tf-2", &tf_spec, "unknown-dataset");
    assert!(tf_err.is_err());
    assert!(tf_err.unwrap_err().contains("does not exist in scientific_datasets"));

    // 2. Initialize search session and insert evaluated trial in trial_results to act as valid parents
    let spec = SearchSpec {
        session_id: "sess-1".into(), project_id: "p-lineage".into(), dataset_id: "data-lin".into(),
        target: "target".into(), task: "binary_classification".into(), metric: "balanced_accuracy".into(),
        direction: "maximize".into(), split: "stratified".into(), fidelities: vec![1.0],
        recipes: vec!["linear".into(), "tree".into()], max_trials: 4, max_screen_trials: 2, max_full_trials: 2,
        search_wall_ms: 60_000, job_timeout_ms: 10_000, finalization_allowance_ms: 10_000,
        max_repeats: 1, tie_tolerance: 0.01, concurrency: 1, max_retries: 1, constraints: None,
    };
    search::init_search(&root, spec).unwrap();

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    db.execute(
        "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
         VALUES('p-lineage', 'sess-1', 'parent-1', 'linear', '{}', 2026, 1.0, 'full_fidelity', NULL, ?1),
               ('p-lineage', 'sess-1', 'parent-2', 'tree', '{}', 2026, 1.0, 'full_fidelity', NULL, ?1)",
        [now],
    ).unwrap();
    db.execute(
        "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
         VALUES('p-lineage', 'parent-1', 'evaluated', 0.85, '{}', '{}', 1, 10.0, 'sha_parent1', 'sha_pred1', 'art1', ?1),
               ('p-lineage', 'parent-2', 'evaluated', 0.88, '{}', '{}', 1, 12.0, 'sha_parent2', 'sha_pred2', 'art2', ?1)",
        [now],
    ).unwrap();

    // 3. Register Calibration Record
    let cal_res = execution::register_calibration_record(&root, "p-lineage", "cal-1", "parent-1", "isotonic", Some(0.05), Some(0.20)).unwrap();
    assert_eq!(cal_res["status"], "registered");
    assert_eq!(cal_res["method"], "isotonic");

    // Negative gate: Unknown base candidate rejected
    let cal_err = execution::register_calibration_record(&root, "p-lineage", "cal-2", "non-existent-candidate", "isotonic", None, None);
    assert!(cal_err.is_err());
    assert!(cal_err.unwrap_err().contains("not found in experiment_dag or trial_results"));

    // 4. Register Scientific Ensemble (Voting)
    let parent_ids = vec!["parent-1".to_string(), "parent-2".to_string()];
    let ens_res = execution::register_scientific_ensemble(&root, "p-lineage", "ens-vote-1", "voting", &parent_ids, Some(&[0.4, 0.6]), None, false).unwrap();
    assert_eq!(ens_res["status"], "registered");
    assert_eq!(ens_res["ensemble_id"], "ens-vote-1");

    // Negative gate: Un-evaluated or non-existent parent candidate rejected
    let fake_parents = vec!["parent-1".to_string(), "ghost-parent".to_string()];
    let ens_err = execution::register_scientific_ensemble(&root, "p-lineage", "ens-fake", "voting", &fake_parents, None, None, true);
    assert!(ens_err.is_err());
    assert!(ens_err.unwrap_err().contains("does not exist or has not been authoritatively evaluated"));

    // 5. Inspect complete scientific lineage
    let lineage = execution::inspect_scientific_lineage(&root, "p-lineage").unwrap();
    assert_eq!(lineage["project"], "p-lineage");
    assert_eq!(lineage["ensembles"].as_array().unwrap().len(), 1);
    assert_eq!(lineage["feature_transforms"].as_array().unwrap().len(), 1);
    assert_eq!(lineage["calibrations"].as_array().unwrap().len(), 1);
}

#[test]
fn m6_project_cancellation_isolation() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&r, "p-cancel", 100);
    create_m1(&r, "p-stay", 100);
    execution_task(&r, "p-cancel", "t-cancel");
    execution_task(&r, "p-stay", "t-stay");

    let mut cap = BTreeMap::new();
    cap.insert("workers".into(), 1);

    // Reserve budget in both projects (already claimed in execution_task with generation 2)
    let res1 = r.command(Request {
        command_id: "res-cancel".into(), project_id: "p-cancel".into(), expected_revision: None,
        operation: Operation::Prepare {
            task_id: "t-cancel".into(), generation: 2, lease_id: "leaset-cancel".into(), action_id: "act-c".into(),
            input: "in".into(), budget: {
                let mut b = BTreeMap::new();
                b.insert("tokens".into(), 500);
                b.insert("calls".into(), 1);
                b.insert("wall_ms".into(), 5000);
                b
            }, capacity: cap.clone(), deadline_ms: 100000, now_ms: 1000,
        },
    }).unwrap();
    assert_eq!(res1.status, "applied");

    let res2 = r.command(Request {
        command_id: "res-stay".into(), project_id: "p-stay".into(), expected_revision: None,
        operation: Operation::Prepare {
            task_id: "t-stay".into(), generation: 2, lease_id: "leaset-stay".into(), action_id: "act-s".into(),
            input: "in".into(), budget: {
                let mut b = BTreeMap::new();
                b.insert("tokens".into(), 500);
                b.insert("calls".into(), 1);
                b.insert("wall_ms".into(), 5000);
                b
            }, capacity: cap, deadline_ms: 100000, now_ms: 1000,
        },
    }).unwrap();
    assert_eq!(res2.status, "applied");

    // Cancel p-cancel
    let cancel_res = r.command(Request {
        command_id: "cancel-cmd".into(), project_id: "p-cancel".into(), expected_revision: None,
        operation: Operation::Lifecycle { state: "cancelled".into() },
    }).unwrap();
    assert_eq!(cancel_res.status, "applied");

    // Verify p-cancel is cancelled and reservations released
    let snap_c = r.snapshot("p-cancel").unwrap();
    assert_eq!(snap_c["state"], "cancelled");
    assert_eq!(snap_c["project_ledger"]["capacity_held"]["workers"].as_i64().unwrap_or(0), 0);

    // Verify p-stay is completely intact with active reservations!
    let snap_s = r.snapshot("p-stay").unwrap();
    assert_eq!(snap_s["state"], "created");
    assert!(snap_s["project_ledger"]["reserved"]["tokens"].as_i64().unwrap_or(0) > 0, "Project p-stay reservations must remain untouched");
}

#[test]
fn m6_lease_safe_storage_gc_and_protected_artifact_retention() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(20, 2)).unwrap();
    create_m1(&r, "p-gc", 20);
    execution_task(&r, "p-gc", "t-gc");
    let spec = JobSpec { resources: Default::default(),
        job_id: "job-1".into(),
        project_id: "p-gc".into(),
        task_id: "t-gc".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        probe: Probe::Write { name: "model.bin".into(), content: "important-weights".into() },
    };
    execution::submit(&root, spec).unwrap();
    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();
    run_job_to_completion(&mut sup, &root, "p-gc", "job-1");

    let art1 = root.join("artifacts/p-gc/job-1/model.bin");
    assert!(art1.exists(), "Created artifact must exist on disk");

    // 1. When artifact is pinned, collect_artifacts must NOT remove it
    execution::pin_artifact(&root, "p-gc", "job-1", true).unwrap();
    let reclaimed_pinned = execution::collect_artifacts(&root).unwrap();
    assert_eq!(reclaimed_pinned, 0, "Pinned artifact must not be reclaimed");
    assert!(art1.exists(), "Pinned artifact file must still exist");

    // 2. When unpinned, collect_artifacts reclaims it
    execution::pin_artifact(&root, "p-gc", "job-1", false).unwrap();
    let reclaimed_unpinned = execution::collect_artifacts(&root).unwrap();
    assert!(reclaimed_unpinned >= 17, "Unpinned artifact must be reclaimed");
    assert!(!art1.exists(), "Reclaimed artifact file must be deleted");
}

#[test]
#[ignore="historical host scientific fixture; no longer valid autonomous delivery evidence; composite finalization remains pending"]
fn m6_end_to_end_autonomous_task_no_manual_edits() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&r, "p-auto", 100);
    execution_task(&r, "p-auto", "t-auto");
    setup_test_env(&root, "scientific");

    let csv = d.path().join("dataset.csv");
    let mut data = "feature1,feature2,target\n".to_string();
    for i in 0..60 {
        let t = if i % 2 == 0 { 0 } else { 1 };
        data.push_str(&format!("{:.2},{:.2},{t}\n", (i as f64) * 0.1, (i as f64) * 0.2));
    }
    fs::write(&csv, &data).unwrap();
    execution::register_dataset(&root, "p-auto", "data-auto", &csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    // Step 1: Audit
    let audit_job = JobSpec { resources: Default::default(),
        job_id: "auto-audit".into(), project_id: "p-auto".into(), task_id: "t-auto".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "audit".into(), dataset_id: "data-auto".into(),
            spec: Some(json!({
                "project_id": "p-auto", "target": "target", "task": "binary_classification",
                "metric": "balanced_accuracy", "split": "stratified", "inference_batch_size": 16,
                "inference_repetitions": 3,
            })),
            candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, audit_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-auto", "auto-audit");

    // Step 2: Baselines
    let baselines_job = JobSpec { resources: Default::default(),
        job_id: "auto-base".into(), project_id: "p-auto".into(), task_id: "t-auto".into(),
        generation: 2, timeout_ms: 30_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "baselines".into(), dataset_id: "data-auto".into(), spec: None, candidate_id: None, test_access_id: None,
        },
    };
    execution::submit(&root, baselines_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-auto", "auto-base");

    // Step 3: Search
    let spec = SearchSpec {
        session_id: "sess-auto".into(), project_id: "p-auto".into(), dataset_id: "data-auto".into(),
        target: "target".into(), task: "binary_classification".into(), metric: "balanced_accuracy".into(),
        direction: "maximize".into(), split: "stratified".into(), fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()], max_trials: 8, max_screen_trials: 2, max_full_trials: 1,
        search_wall_ms: 300_000, job_timeout_ms: 30_000, finalization_allowance_ms: 30_000,
        max_repeats: 3, tie_tolerance: 0.005, concurrency: 1, max_retries: 2, constraints: None,
    };
    search::init_search(&root, spec).unwrap();
    let selection = search::execute_search(&root, "p-auto", "sess-auto", "t-auto", 2, &mut sup, Some("scientific")).unwrap();
    assert_eq!(selection.status, "apex_selected");
    let apex_id = selection.selected_candidate_id.expect("Apex candidate must be selected");

    // Step 4: Finalize
    let finalize_job = JobSpec { resources: Default::default(),
        job_id: "auto-fin".into(), project_id: "p-auto".into(), task_id: "t-auto".into(),
        generation: 2, timeout_ms: 60_000, threads: 1, max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "finalize".into(), dataset_id: "data-auto".into(), spec: None,
            candidate_id: Some(apex_id.clone()), test_access_id: Some("test-gate-auto".into()),
        },
    };
    execution::submit(&root, finalize_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-auto", "auto-fin");

    // Step 5: Verify Bundle
    let bundle_path = root.join("workflows/p-auto/bundle");
    let report = verify_bundle(&bundle_path, false).unwrap();
    assert_eq!(report.status, "integrity_verified");
    assert_eq!(report.candidate_id, apex_id);
    assert_eq!(report.parity_check, "skipped_by_request");
}

#[test]
fn m6_existing_version_six_database_receives_boosting_migration() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let runtime = Runtime::init(&root, m1policy(100000, 4)).unwrap();
    drop(runtime);
    execution::migrate(&root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let old_schema = include_str!("../src/schema_m3.sql");
    let start = old_schema.find("CREATE TABLE experiment_dag(").unwrap();
    let end = old_schema[start..].find("CREATE TABLE optimizer_trials").unwrap() + start;
    db.execute_batch("PRAGMA foreign_keys=OFF; DROP TABLE experiment_dag;").unwrap();
    db.execute_batch(&old_schema[start..end]).unwrap();
    db.execute_batch("DROP TABLE tool_outcomes; DROP TABLE pipeline_receipts; DROP TABLE content_decisions; DROP TABLE provider_turn_receipts; DROP TABLE outbound_views; PRAGMA user_version=6;").unwrap();
    drop(db);
    execution::migrate(&root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let version: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 9);
    let ddl: String = db.query_row("SELECT sql FROM sqlite_master WHERE name='experiment_dag'", [], |r| r.get(0)).unwrap();
    assert!(ddl.contains("hist_boosting"));
    let foreign_keys: String = db.query_row("SELECT sql FROM sqlite_master WHERE name='trial_results'", [], |r| r.get(0)).unwrap();
    assert!(foreign_keys.contains("REFERENCES experiment_dag"));
    execution::migrate(&root).unwrap(); // Repeated migration is harmless.
}

#[test]
fn m6_standalone_executable_runs_without_source_checkout() {
    let temp = tempfile::tempdir().unwrap();
    let bin_dir = temp.path().join("bin");
    let work_dir = temp.path().join("workdir");
    fs::create_dir_all(&bin_dir).unwrap();
    fs::create_dir_all(&work_dir).unwrap();
    let installed_bin = bin_dir.join("mlrescurix-harness");
    fs::copy(env!("CARGO_BIN_EXE_mlrescurix-harness"), &installed_bin).unwrap();
    let state_dir = work_dir.join("state");

    // Materialize embedded Python runtime assets directly into state without referencing repo checkout
    let py_root = mlrescurix_harness::assets::ensure_python_root(&state_dir).unwrap();
    assert!(py_root.join("src/mlrescurix_worker/scientific/workflow.py").is_file());
    fs::remove_dir_all(&state_dir).unwrap();

    let run_cli = |args: &[&str]| -> serde_json::Value {
        let out = std::process::Command::new(&installed_bin)
            .args(args)
            .current_dir(&work_dir)
            .env_clear()
            .env("PATH", format!("{}:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin", bin_dir.display()))
            .env("HOME", &work_dir)
            .output()
            .unwrap();
        assert!(out.status.success(), "CLI {:?} failed: {}", args, String::from_utf8_lossy(&out.stderr));
        serde_json::from_slice(&out.stdout).unwrap()
    };

    let caps = run_cli(&["capabilities"]);
    assert_eq!(caps["standalone_install_without_checkout"], true);

    let policy_path = work_dir.join("policy.json");
    fs::write(&policy_path, serde_json::to_vec(&m1policy(50, 2)).unwrap()).unwrap();
    let init_res = run_cli(&["init", "--state", state_dir.to_str().unwrap(), "--policy", policy_path.to_str().unwrap()]);
    assert_eq!(init_res["initialized"], true);

    let r = Runtime::open(&state_dir).unwrap();
    create_m1(&r, "p-standalone", 50);
    execution_task(&r, "p-standalone", "t-standalone");
    drop(r);

    let job_spec_path = work_dir.join("job.json");
    let spec = JobSpec { resources: Default::default(),
        job_id: "standalone-job-1".into(),
        project_id: "p-standalone".into(),
        task_id: "t-standalone".into(),
        generation: 2,
        timeout_ms: 5_000,
        threads: 1,
        max_output_bytes: 4096,
        environment_id: None,
        probe: Probe::Write { name: "result.txt".into(), content: "standalone-ok".into() },
    };
    fs::write(&job_spec_path, serde_json::to_vec(&spec).unwrap()).unwrap();
    run_cli(&["submit", "--state", state_dir.to_str().unwrap(), "--spec", job_spec_path.to_str().unwrap()]);
    let drain_res = run_cli(&["supervise", "--state", state_dir.to_str().unwrap(), "--workers", "1", "--drain"]);
    assert_eq!(drain_res["drained"], true);

    let inspect_res = run_cli(&["inspect-job", "--state", state_dir.to_str().unwrap(), "--project", "p-standalone", "--job", "standalone-job-1"]);
    assert_eq!(inspect_res["status"], "succeeded");
    assert_eq!(fs::read_to_string(state_dir.join("artifacts/p-standalone/standalone-job-1/result.txt")).unwrap(), "standalone-ok");

    let storage_res = run_cli(&["storage-policy-summary", "--state", state_dir.to_str().unwrap()]);
    assert_eq!(storage_res["observed"]["held_reservations"], 0);
}
