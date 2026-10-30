use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use mlrescurix_harness::{
    bundle,
    contracts::{Amounts, Context, CustomJobResources, CustomJobSpec, TrainingCheckpoint, TrainingResult},
    core::Operation,
    execution::{self, JobSpec, Probe, Supervisor},
    Policy,
    Request, Runtime,
};

fn hash(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn resources(v: i64) -> Amounts {
    BTreeMap::from([("workers".into(), v)])
}
fn m1policy(limit: i64, workers: i64) -> Policy {
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), limit),
            ("tokens".into(), limit * 1000),
            ("wall_ms".into(), limit * 10000),
        ]),
        capacity: resources(workers),
        max_tasks: 1024,
    }
}
fn context(base: &str, code: &str) -> Context {
    Context {
        role_id: "researcher".into(),
        role_version: "1".into(),
        prompt_hash: "prompt-hash".into(),
        model: None,
        phase: "research".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "inspect".into(),
        stop_conditions: vec!["budget".into()],
    }
}
fn chrono_free_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
fn req(project: &str, cmd: &str, op: Operation) -> Request {
    Request {
        command_id: cmd.into(),
        project_id: project.into(),
        expected_revision: None,
        operation: op,
    }
}
fn send(r: &Runtime, p: &str, c: &str, o: Operation) -> mlrescurix_harness::Receipt {
    r.command(req(p, c, o)).unwrap()
}
fn create_m1(r: &Runtime, p: &str) {
    assert_eq!(
        send(
            r,
            p,
            "project",
            Operation::CreateProject {
                intent: "M1 execution fixture".into(),
                finalization_id: format!("final-{p}"),
                policy: m1policy(100, 8),
            }
        )
        .status,
        "applied"
    );
}
fn branch_task(r: &Runtime, p: &str, branch: &str, task: &str) {
    send(
        r,
        p,
        &format!("b{branch}"),
        Operation::CreateBranch {
            branch_id: branch.into(),
            base_hash: format!("base-{branch}"),
            code_hash: format!("code-{branch}"),
        },
    );
    send(
        r,
        p,
        &format!("t{task}"),
        Operation::CreateTask {
            task_id: task.into(),
            branch_id: branch.into(),
            context: context(&format!("base-{branch}"), &format!("code-{branch}")),
        },
    );
}
fn execution_task(r: &Runtime, p: &str, t: &str) {
    branch_task(r, p, &format!("b{t}"), t);
    let now = chrono_free_now();
    send(
        r,
        p,
        &format!("claim{t}"),
        Operation::Claim {
            task_id: t.into(),
            owner: "local-worker".into(),
            lease_id: format!("lease{t}"),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        },
    );
}

fn get_task_generation_and_intent(root: &Path, project: &str, task: &str) -> (u64, u64) {
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let data_str: String = db.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        rusqlite::params![project, task],
        |r| r.get(0),
    ).unwrap();
    let task_val: serde_json::Value = serde_json::from_str(&data_str).unwrap();
    (
        task_val["generation"].as_u64().unwrap(),
        task_val["intent_revision"].as_u64().unwrap(),
    )
}

fn find_python() -> PathBuf {
    if let Ok(p) = std::env::var("MLRRESCURIX_PYTHON") {
        if Path::new(&p).is_file() {
            return PathBuf::from(p);
        }
    }
    let candidates = [
        "/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13",
        "/usr/local/bin/python3",
        "/opt/homebrew/bin/python3",
        "/usr/bin/python3",
    ];
    for c in &candidates {
        let p = Path::new(c);
        if p.is_file() {
            return p.to_path_buf();
        }
    }
    PathBuf::from("python3")
}

fn setup_ready_environment(root: &Path, env_id: &str) {
    execution::migrate(root).unwrap();
    let py = find_python();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO environments(id, python, identity, path, status) VALUES(?1, ?2, 'ident-custom', ?3, 'ready')
         ON CONFLICT(id) DO UPDATE SET python=excluded.python, status='ready'",
        rusqlite::params![env_id, py.to_string_lossy(), py.to_string_lossy()],
    ).unwrap();
}

#[allow(dead_code)]
fn setup_container_environment(root: &Path, env_id: &str, image: &str) {
    execution::migrate(root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute(
        "INSERT INTO environments(id, python, identity, path, status) VALUES(?1, ?2, 'ident-custom', '/managed/path', 'ready')
         ON CONFLICT(id) DO UPDATE SET python=excluded.python, status='ready'",
        rusqlite::params![env_id, format!("docker:{image}")],
    ).unwrap();
}

fn create_synthetic_dataset(path: &Path) {
    let mut csv = String::from("x1,x2,x3,target\n");
    for i in 0..100 {
        let x1 = (i as f64) * 0.1;
        let x2 = (i as f64) * -0.05;
        let x3 = ((i % 5) as f64) * 0.5;
        let target = if i % 2 == 0 { 1 } else { 0 };
        csv.push_str(&format!("{x1:.3},{x2:.3},{x3:.3},{target}\n"));
    }
    fs::write(path, csv).unwrap();
}

fn setup_synthetic_dataset_manifest(root: &Path, project: &str, ds_csv: &Path) {
    let workflow_dir = root.join("workflows").join(project);
    fs::create_dir_all(&workflow_dir).unwrap();
    let csv_bytes = fs::read(ds_csv).unwrap();
    let h = hash(&csv_bytes);
    let train: Vec<usize> = (0..70).collect();
    let val: Vec<usize> = (70..90).collect();
    let test: Vec<usize> = (90..100).collect();
    let manifest = json!({
        "format": "mlrescurix.dataset.v1",
        "project_id": project,
        "sha256": h,
        "dataset_sha256": h,
        "target": "target",
        "target_column": "target",
        "features": ["x1", "x2", "x3"],
        "feature_columns": ["x1", "x2", "x3"],
        "metric": "accuracy",
        "primary_metric": "accuracy",
        "metric_direction": "maximize",
        "split_policy": "random",
        "splits": {
            "train": train,
            "validation": val,
            "test": test,
        }
    });
    fs::write(workflow_dir.join("dataset_manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
}

fn run_until_terminal(s: &mut Supervisor, root: &Path, project: &str, job_id: &str) -> String {
    for _ in 0..500 {
        s.tick().unwrap();
        let j = execution::inspect(root, project, job_id).unwrap();
        if !["queued", "running", "terminating", "recovering"].contains(&j.status.as_str()) {
            return j.status;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("job {} did not reach terminal state within limit", job_id);
}

#[test]
#[ignore = "custom admission is disabled pending isolation and authoritative evaluation"]
fn test_custom_model_train_resume_and_export_governed_slice() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();

    let project = "p-custom";
    let task = "t-custom";
    let env_id = "env-custom";
    let dataset_id = "ds-custom";

    create_m1(&r, project);
    execution_task(&r, project, task);
    setup_container_environment(&root, env_id, mlrescurix_harness::agents::sandbox::SCIENTIFIC_IMAGE_REF.trim());

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();
    setup_synthetic_dataset_manifest(&root, project, &ds_csv);

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    // 1. Submit custom_train
    let train_spec = CustomJobSpec {
        job_id: "job-train-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-torch-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: Some(CustomJobResources {
            memory_bytes: 1024 * 1024 * 1024,
            cpu_threads: 2,
            timeout_ms: 60_000,
        }),
    };
    train_spec.validate().unwrap();

    let train_probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&train_spec).unwrap()),
        candidate_id: Some("cand-torch-1".into()),
        test_access_id: None,
    };

    let train_job_spec = JobSpec {
        job_id: "job-train-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe: train_probe,
    };

    assert_eq!(execution::submit(&root, train_job_spec).unwrap().status, "queued");
    let status = run_until_terminal(&mut supervisor, &root, project, "job-train-01");
    let log_path = root.join("logs").join(project).join("job-train-01.log");
    let log_content = fs::read_to_string(&log_path).unwrap_or_else(|_| "NO LOG FILE".into());
    eprintln!("=== WORKER LOG ===\n{}\n==================", log_content);
    let train_job_view = execution::inspect(&root, project, "job-train-01").unwrap();
    assert_eq!((status.as_str(), train_job_view.reason.as_deref()), ("succeeded", None));
    assert_eq!(train_job_view.status, "succeeded");

    let art_dir = root.join("artifacts").join(project).join("job-train-01");
    assert!(art_dir.join("result.json").is_file());
    assert!(art_dir.join("checkpoint.json").is_file());

    let result_raw = fs::read(art_dir.join("result.json")).unwrap();
    let tr: TrainingResult = serde_json::from_slice(&result_raw).unwrap();
    tr.validate().unwrap();
    assert_eq!(tr.status, "succeeded");
    assert_eq!(tr.step_reached, 10);
    assert!(tr.primary_metric_score.unwrap() >= 0.0);

    let ckpt_raw = fs::read(art_dir.join("checkpoint.json")).unwrap();
    let tc: TrainingCheckpoint = serde_json::from_slice(&ckpt_raw).unwrap();
    tc.validate().unwrap();
    assert_eq!(tc.step, 10);

    // 2. Submit custom_resume from step 10 to step 20
    let resume_spec = CustomJobSpec {
        job_id: "job-resume-02".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-torch-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_resume".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 20,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: Some("job-train-01".into()),
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: Some(CustomJobResources {
            memory_bytes: 1024 * 1024 * 1024,
            cpu_threads: 2,
            timeout_ms: 60_000,
        }),
    };
    resume_spec.validate().unwrap();

    let resume_probe = Probe::Scientific {
        operation: "custom_resume".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&resume_spec).unwrap()),
        candidate_id: Some("cand-torch-1".into()),
        test_access_id: None,
    };

    let resume_job_spec = JobSpec {
        job_id: "job-resume-02".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe: resume_probe,
    };

    assert_eq!(execution::submit(&root, resume_job_spec).unwrap().status, "queued");
    let status_resume = run_until_terminal(&mut supervisor, &root, project, "job-resume-02");
    assert_eq!(status_resume, "succeeded");

    let resume_art_dir = root.join("artifacts").join(project).join("job-resume-02");
    let resume_result_raw = fs::read(resume_art_dir.join("result.json")).unwrap();
    let tr_resumed: TrainingResult = serde_json::from_slice(&resume_result_raw).unwrap();
    assert_eq!(tr_resumed.status, "succeeded");
    assert_eq!(tr_resumed.step_reached, 20);

    // 3. Submit custom_evaluate to evaluate candidate on immutable development validation split
    let eval_spec = CustomJobSpec {
        job_id: "job-eval-03".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-torch-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_evaluate".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 20,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: Some("job-resume-02_resumed".into()),
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: Some(CustomJobResources {
            memory_bytes: 1024 * 1024 * 1024,
            cpu_threads: 2,
            timeout_ms: 60_000,
        }),
    };
    eval_spec.validate().unwrap();

    let eval_probe = Probe::Scientific {
        operation: "custom_evaluate".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&eval_spec).unwrap()),
        candidate_id: Some("cand-torch-1".into()),
        test_access_id: None,
    };

    let eval_job_spec = JobSpec {
        job_id: "job-eval-03".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe: eval_probe,
    };

    assert_eq!(execution::submit(&root, eval_job_spec).unwrap().status, "queued");
    let status_eval = run_until_terminal(&mut supervisor, &root, project, "job-eval-03");
    assert_eq!(status_eval, "succeeded");

    let eval_art_dir = root.join("artifacts").join(project).join("job-eval-03");
    assert!(eval_art_dir.join("result.json").is_file());
    assert!(eval_art_dir.join("evaluation.json").is_file());

    // 4. Submit custom_export to generate clean-room delivery bundle
    let export_spec = CustomJobSpec {
        job_id: "job-export-04".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-torch-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_export".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 20,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: Some("job-resume-02_resumed".into()),
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: Some(CustomJobResources {
            memory_bytes: 1024 * 1024 * 1024,
            cpu_threads: 2,
            timeout_ms: 60_000,
        }),
    };
    export_spec.validate().unwrap();

    let export_probe = Probe::Scientific {
        operation: "custom_export".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&export_spec).unwrap()),
        candidate_id: Some("cand-torch-1".into()),
        test_access_id: None,
    };

    let export_job_spec = JobSpec {
        job_id: "job-export-04".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe: export_probe,
    };

    assert_eq!(execution::submit(&root, export_job_spec).unwrap().status, "queued");
    let status_export = run_until_terminal(&mut supervisor, &root, project, "job-export-04");
    assert_eq!(status_export, "succeeded");

    let export_art_dir = root.join("artifacts").join(project).join("job-export-04");
    assert!(export_art_dir.join("bundle_manifest.json").is_file());
    assert!(export_art_dir.join("scorecard.json").is_file());
    assert!(export_art_dir.join("final.json").is_file());
    assert!(export_art_dir.join("verification.json").is_file());

    // 4. Verify delivery bundle using Rust authoritative bundle verifier
    let bundle_dir = root.join("workflows").join(project).join("bundle");
    assert!(bundle_dir.is_dir());
    let report = bundle::verify_bundle(&bundle_dir, false).unwrap();
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
    assert_eq!(report.format, "mlrescurix.bundle.v2");
    assert_eq!(report.candidate_id, "cand-torch-1");
}

#[test]
#[ignore = "custom admission is disabled pending isolation and authoritative evaluation"]
fn test_custom_model_timeout_boundary_enforcement() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();

    let project = "p-timeout";
    let task = "t-timeout";
    let env_id = "env-timeout";
    let dataset_id = "ds-timeout";

    create_m1(&r, project);
    execution_task(&r, project, task);
    setup_container_environment(&root, env_id, mlrescurix_harness::agents::sandbox::SCIENTIFIC_IMAGE_REF.trim());

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();
    setup_synthetic_dataset_manifest(&root, project, &ds_csv);

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    let slow_spec = CustomJobSpec {
        job_id: "job-slow-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-slow-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 1000,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(64),
        seed: Some(42),
        resources: None,
    };

    let probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&slow_spec).unwrap()),
        candidate_id: Some("cand-slow-1".into()),
        test_access_id: None,
    };

    // 1ms timeout guaranteed to exceed
    let job_spec = JobSpec {
        job_id: "job-slow-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 1,
        threads: 1,
        max_output_bytes: 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe,
    };

    execution::submit(&root, job_spec).unwrap();
    let status = run_until_terminal(&mut supervisor, &root, project, "job-slow-01");
    assert_eq!(status, "timed_out");

    let view = execution::inspect(&root, project, "job-slow-01").unwrap();
    assert_eq!(view.status, "timed_out");
    assert_eq!(view.reason, Some("deadline exceeded".into()));

    // Verify supervisor survives timeout kill and executes subsequent jobs cleanly
    let survivor_spec = JobSpec {
        job_id: "job-survivor-02".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024 * 1024,
        environment_id: None,
        resources: Default::default(),
        probe: Probe::Write {
            name: "survivor.txt".into(),
            content: "supervisor healthy".into(),
        },
    };
    execution::submit(&root, survivor_spec).unwrap();
    let status2 = run_until_terminal(&mut supervisor, &root, project, "job-survivor-02");
    assert_eq!(status2, "succeeded");
    let view2 = execution::inspect(&root, project, "job-survivor-02").unwrap();
    assert_eq!(view2.status, "succeeded");
}

#[test]
#[ignore = "custom admission is disabled pending isolation and authoritative evaluation"]
fn test_custom_model_memory_limit_enforcement() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();

    let project = "p-oom";
    let task = "t-oom";
    let env_id = "env-oom";
    let dataset_id = "ds-oom";

    create_m1(&r, project);
    execution_task(&r, project, task);
    setup_container_environment(&root, env_id, mlrescurix_harness::agents::sandbox::SCIENTIFIC_IMAGE_REF.trim());

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();
    setup_synthetic_dataset_manifest(&root, project, &ds_csv);

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    // Run enough steps so supervisor reap() samples RSS and detects exceeded memory limit
    let oom_spec = CustomJobSpec {
        job_id: "job-oom-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-oom-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10_000,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(64),
        seed: Some(42),
        resources: None,
    };

    let probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: dataset_id.into(),
        spec: Some(serde_json::to_value(&oom_spec).unwrap()),
        candidate_id: Some("cand-oom-1".into()),
        test_access_id: None,
    };

    // 64 MiB memory limit; PyTorch RSS will exceed ~100+ MiB
    let job_spec = JobSpec {
        job_id: "job-oom-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 1,
        max_output_bytes: 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: execution::resources::ResourceRequirements {
            memory_bytes: 64 * 1024 * 1024,
            ..Default::default()
        },
        probe,
    };

    execution::submit(&root, job_spec).unwrap();
    let status = run_until_terminal(&mut supervisor, &root, project, "job-oom-01");
    assert_eq!(status, "failed");

    let view = execution::inspect(&root, project, "job-oom-01").unwrap();
    assert_eq!(view.status, "failed");
    assert_eq!(view.reason, Some("worker terminated by SIGKILL (exit 137); cause unverified".into()));

    // Verify supervisor survives memory kill and executes subsequent jobs cleanly
    let survivor_spec = JobSpec {
        job_id: "job-survivor-oom".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024 * 1024,
        environment_id: None,
        resources: Default::default(),
        probe: Probe::Write {
            name: "survivor_oom.txt".into(),
            content: "supervisor healthy after OOM".into(),
        },
    };
    execution::submit(&root, survivor_spec).unwrap();
    let status2 = run_until_terminal(&mut supervisor, &root, project, "job-survivor-oom");
    assert_eq!(status2, "succeeded");
    let view2 = execution::inspect(&root, project, "job-survivor-oom").unwrap();
    assert_eq!(view2.status, "succeeded");
}

#[test]
fn test_custom_model_v2_contracts_serialization_and_validation() {
    let spec = CustomJobSpec {
        job_id: "job-schema-01".into(),
        project_id: "p-schema".into(),
        task_id: "t-schema".into(),
        candidate_id: Some("cand-schema-01".into()),
        generation: 1,
        intent_revision: 1,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: "ds-schema".into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: None,
    };
    let spec_json = serde_json::to_string(&spec).unwrap();
    assert!(spec_json.contains(r#""schema_version":"v2""#));
    assert!(spec_json.contains(r#""candidate_id":"cand-schema-01""#));
    let parsed: CustomJobSpec = serde_json::from_str(&spec_json).unwrap();
    assert_eq!(parsed.schema_version.as_deref(), Some("v2"));
    assert_eq!(parsed.candidate_id.as_deref(), Some("cand-schema-01"));
    parsed.validate().unwrap();

    let mut bad_traverse = parsed.clone();
    bad_traverse.resume_checkpoint = Some("../evil".into());
    assert!(bad_traverse.validate().unwrap_err().contains("directory traversal"));

    let mut bad_abs = parsed.clone();
    bad_abs.resume_checkpoint = Some("/etc/passwd".into());
    assert!(bad_abs.validate().unwrap_err().contains("directory traversal"));

    let job = JobSpec {
        job_id: "job-schema-01".into(),
        project_id: "p-schema".into(),
        task_id: "t-schema".into(),
        generation: 1,
        timeout_ms: 30_000,
        threads: 1,
        max_output_bytes: 1024 * 1024,
        environment_id: Some("env-schema".into()),
        resources: Default::default(),
        probe: Probe::Scientific {
            operation: "custom_train".into(),
            dataset_id: "ds-schema".into(),
            spec: Some(serde_json::to_value(&parsed).unwrap()),
            candidate_id: Some("cand-schema-01".into()),
            test_access_id: None,
        },
    };
    assert!(execution::validate_spec(&job).unwrap_err().contains("custom scientific execution is experimental"));

    // Identity mismatches rejected by authoritative Rust control plane
    let mut bad_job = job.clone();
    bad_job.job_id = "job-different".into();
    assert!(execution::validate_spec(&bad_job).unwrap_err().contains("job identity mismatch"));

    let mut bad_proj = job.clone();
    bad_proj.project_id = "p-different".into();
    assert!(execution::validate_spec(&bad_proj).unwrap_err().contains("project identity mismatch"));

    let mut bad_task = job.clone();
    bad_task.task_id = "t-different".into();
    assert!(execution::validate_spec(&bad_task).unwrap_err().contains("task identity mismatch"));

    let mut bad_op = job.clone();
    bad_op.probe = Probe::Scientific {
        operation: "custom_resume".into(),
        dataset_id: "ds-schema".into(),
        spec: Some(serde_json::to_value(&parsed).unwrap()),
        candidate_id: Some("cand-schema-01".into()),
        test_access_id: None,
    };
    assert!(execution::validate_spec(&bad_op).unwrap_err().contains("operation mismatch"));

    let mut bad_cand = job.clone();
    bad_cand.probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: "ds-schema".into(),
        spec: Some(serde_json::to_value(&parsed).unwrap()),
        candidate_id: Some("cand-different".into()),
        test_access_id: None,
    };
    assert!(execution::validate_spec(&bad_cand).unwrap_err().contains("candidate identity mismatch"));

    let mut bad_cand_probe_none = job.clone();
    bad_cand_probe_none.probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: "ds-schema".into(),
        spec: Some(serde_json::to_value(&parsed).unwrap()),
        candidate_id: None,
        test_access_id: None,
    };
    assert!(execution::validate_spec(&bad_cand_probe_none).unwrap_err().contains("candidate identity"));

    let mut bad_cand_spec_none = job.clone();
    let mut parsed_spec_none = parsed.clone();
    parsed_spec_none.candidate_id = None;
    bad_cand_spec_none.probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: "ds-schema".into(),
        spec: Some(serde_json::to_value(&parsed_spec_none).unwrap()),
        candidate_id: Some("cand-schema-01".into()),
        test_access_id: None,
    };
    assert!(execution::validate_spec(&bad_cand_spec_none).unwrap_err().contains("candidate identity mismatch"));

    let mut export_no_cand = job.clone();
    let mut export_spec_val = parsed.clone();
    export_spec_val.operation = "custom_export".into();
    export_spec_val.candidate_id = None;
    export_no_cand.probe = Probe::Scientific {
        operation: "custom_export".into(),
        dataset_id: "ds-schema".into(),
        spec: Some(serde_json::to_value(&export_spec_val).unwrap()),
        candidate_id: None,
        test_access_id: None,
    };
    assert!(execution::validate_spec(&export_no_cand).unwrap_err().contains("candidate identity"));

    // Verify deserialization default when schema_version is omitted in JSON
    let omitted_json = r#"{"job_id":"job-omitted","project_id":"p1","task_id":"t1","generation":1,"intent_revision":1,"operation":"custom_train","model_plugin":"tiny_torch","dataset_id":"ds1","metric_name":"accuracy","metric_direction":"maximize","max_steps":10,"batch_size":16}"#;
    let parsed_omitted: CustomJobSpec = serde_json::from_str(omitted_json).unwrap();
    assert_eq!(parsed_omitted.schema_version.as_deref(), Some("v2"));
    parsed_omitted.validate().unwrap();

    let mut files = std::collections::BTreeMap::new();
    files.insert("weights.pt".into(), "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into());
    let ckpt = TrainingCheckpoint {
        checkpoint_id: "ckpt-schema-01".into(),
        job_id: "job-schema-01".into(),
        schema_version: Some("v2".into()),
        project_id: Some("p1".into()),
        candidate_id: Some("cand-schema-01".into()),
        weights_sha256: Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()),
        dataset_sha256: Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()),
        split_sha256: Some("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into()),
        source_sha256: Some("1122334455667788112233445566778811223344556677881122334455667788".into()),
        environment_id: Some("env-1".into()),
        training_config: Some(serde_json::json!({"lr": 0.01})),
        replay_guarantee: Some("exact_replay".into()),
        step: 10,
        epoch: 1,
        batch_offset: 2,
        sampler_position: 160,
        loss: Some(0.123),
        files,
        training_state: None,
        timestamp_unix: 1700000000.0,
    };
    let ckpt_json = serde_json::to_string(&ckpt).unwrap();
    assert!(ckpt_json.contains(r#""schema_version":"v2""#));
    assert!(ckpt_json.contains(r#""candidate_id":"cand-schema-01""#));
    assert!(ckpt_json.contains(r#""replay_guarantee":"exact_replay""#));
    let parsed_ckpt: TrainingCheckpoint = serde_json::from_str(&ckpt_json).unwrap();
    assert_eq!(parsed_ckpt.schema_version.as_deref(), Some("v2"));
    assert_eq!(parsed_ckpt.replay_guarantee.as_deref(), Some("exact_replay"));
    parsed_ckpt.validate().unwrap();

    let mut bad_hash_ckpt = parsed_ckpt.clone();
    bad_hash_ckpt.dataset_sha256 = Some("not-64-hex".into());
    assert!(bad_hash_ckpt.validate().unwrap_err().contains("invalid dataset_sha256"));

    let result = TrainingResult {
        job_id: "job-schema-01".into(),
        schema_version: Some("v2".into()),
        project_id: Some("p1".into()),
        candidate_id: Some("cand-schema-01".into()),
        weights_sha256: Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()),
        dataset_sha256: Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()),
        split_sha256: Some("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into()),
        source_sha256: Some("1122334455667788112233445566778811223344556677881122334455667788".into()),
        environment_id: Some("env-1".into()),
        training_config: Some(serde_json::json!({"lr": 0.01})),
        replay_guarantee: Some("exact_replay".into()),
        status: "succeeded".into(),
        step_reached: 10,
        epoch_reached: 1,
        primary_metric_name: Some("accuracy".into()),
        primary_metric_score: Some(0.95),
        validation_loss: Some(0.123),
        peak_memory_bytes: 1024 * 1024,
        duration_seconds: 2.5,
        artifacts: vec!["result.json".into()],
        limitations: vec![],
        reason: None,
    };
    let result_json = serde_json::to_string(&result).unwrap();
    assert!(result_json.contains(r#""schema_version":"v2""#));
    assert!(result_json.contains(r#""candidate_id":"cand-schema-01""#));
    assert!(result_json.contains(r#""replay_guarantee":"exact_replay""#));
    let parsed_result: TrainingResult = serde_json::from_str(&result_json).unwrap();
    assert_eq!(parsed_result.schema_version.as_deref(), Some("v2"));
    parsed_result.validate().unwrap();

    let mut bad_res = parsed_result.clone();
    bad_res.dataset_sha256 = Some("invalid".into());
    assert!(bad_res.validate().unwrap_err().contains("invalid dataset_sha256"));
}

#[test]
fn test_custom_model_admission_rejects_unverified_host_environment() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();

    let project = "p-no-docker";
    let task = "t-no-docker";
    let env_id = "env-host-only";
    let dataset_id = "ds-no-docker";

    create_m1(&r, project);
    execution_task(&r, project, task);
    setup_ready_environment(&root, env_id); // registers host python, NOT docker:...

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    let spec = CustomJobSpec {
        job_id: "job-nodocker-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        candidate_id: Some("cand-1".into()),
        generation: task_gen,
        intent_revision: intent,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: dataset_id.into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: None,
    };
    spec.validate().unwrap();

    let job_spec = JobSpec {
        job_id: "job-nodocker-01".into(),
        project_id: project.into(),
        task_id: task.into(),
        generation: task_gen,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(env_id.into()),
        resources: Default::default(),
        probe: Probe::Scientific {
            operation: "custom_train".into(),
            dataset_id: dataset_id.into(),
            spec: Some(serde_json::to_value(&spec).unwrap()),
            candidate_id: Some("cand-1".into()),
            test_access_id: None,
        },
    };

    let err = execution::submit(&root, job_spec).unwrap_err();
    assert!(err.contains("custom scientific execution is experimental and unavailable pending governed isolation and evaluation"), "Unexpected err: {err}");
}

#[test]
fn test_custom_model_resume_checkpoint_path_traversal_rejected() {
    let mut spec = CustomJobSpec {
        job_id: "job-trav-01".into(),
        project_id: "p1".into(),
        task_id: "t1".into(),
        candidate_id: Some("cand-1".into()),
        generation: 1,
        intent_revision: 1,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: "ds1".into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: Some("../escaped".into()),
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: None,
    };
    let err = spec.validate().unwrap_err();
    assert!(err.contains("directory traversal"));

    spec.resume_checkpoint = Some("/etc/passwd".into());
    let err2 = spec.validate().unwrap_err();
    assert!(err2.contains("directory traversal"));

    spec.resume_checkpoint = Some("\\windows\\system32".into());
    let err3 = spec.validate().unwrap_err();
    assert!(err3.contains("directory traversal"));

    spec.resume_checkpoint = Some("".into());
    let err4 = spec.validate().unwrap_err();
    assert!(err4.contains("directory traversal"));

    spec.resume_checkpoint = Some("a".repeat(300));
    let err5 = spec.validate().unwrap_err();
    assert!(err5.contains("directory traversal"));

    spec.resume_checkpoint = Some("foo/../../bar".into());
    let err6 = spec.validate().unwrap_err();
    assert!(err6.contains("directory traversal"));

    spec.resume_checkpoint = Some("C:checkpoint".into());
    let err7 = spec.validate().unwrap_err();
    assert!(err7.contains("directory traversal"));

    spec.resume_checkpoint = Some("null\0byte".into());
    let err8 = spec.validate().unwrap_err();
    assert!(err8.contains("directory traversal"));
}

#[test]
fn test_custom_model_governance_candidate_evaluations_schema_and_weights_binding() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "p-eval-gov";
    create_m1(&r, project);
    execution::migrate(&root).unwrap();

    // Connect to SQLite DB
    let conn = execution::connect(&root).unwrap();

    // Verify table exists
    let table_exists: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='candidate_evaluations')",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(table_exists);

    // Insert an evaluated record with exact weights_sha256 as scorecard_hash
    let weights_sha = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    conn.execute(
        "INSERT INTO candidate_evaluations(project, evaluation_id, candidate_id, dataset_id, environment_id, source_hash, status, folds, seed, primary_metric, metrics_json, slices_json, scorecard_hash, created_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, 'evaluated', 1, 42, 0.92, '{}', NULL, ?7, 1000)",
        rusqlite::params![project, "eval-01", "cand-01", "ds-01", "env-01", "src-hash", weights_sha],
    ).unwrap();

    // Query it back and verify scorecard_hash binding
    let row_sha: String = conn.query_row(
        "SELECT scorecard_hash FROM candidate_evaluations WHERE project=?1 AND candidate_id=?2 AND status='evaluated'",
        rusqlite::params![project, "cand-01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(row_sha, weights_sha);
}

#[test]
fn test_custom_model_bundle_manifest_and_contract_verification() {
    let temp = tempfile::tempdir().unwrap();
    let bundle_dir = temp.path().join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // 1. Model weights
    fs::write(bundle_dir.join("frozen_model.joblib"), "dummy_model_bytes").unwrap();
    let frozen_sha = format!("{:x}", Sha256::digest(b"dummy_model_bytes"));

    fs::write(bundle_dir.join("weights.pt"), "dummy_pt_bytes").unwrap();
    let weights_sha = format!("{:x}", Sha256::digest(b"dummy_pt_bytes"));

    // 2. Schemas and JSON contracts
    fs::write(bundle_dir.join("input_schema.json"), r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#).unwrap();
    fs::write(bundle_dir.join("output_schema.json"), r#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#).unwrap();
    fs::write(
        bundle_dir.join("scorecard.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "candidate_id": "cand1",
            "bundle_version": "2.0.0",
            "audit_status": "UNVERIFIED_DEVELOPMENT"
        })).unwrap(),
    ).unwrap();
    fs::write(
        bundle_dir.join("verification.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "candidate_id": "cand1",
            "frozen_model_sha256": frozen_sha,
            "weights_sha256": weights_sha,
            "all_passed": false
        })).unwrap(),
    ).unwrap();
    fs::write(
        bundle_dir.join("final.json"),
        serde_json::to_string_pretty(&serde_json::json!({
            "candidate_id": "cand1",
            "status": "unverified_development_only"
        })).unwrap(),
    ).unwrap();
    fs::write(bundle_dir.join("examples_input.json"), r#"[{"x": 1.0}]"#).unwrap();
    fs::write(bundle_dir.join("examples_expected.json"), r#"{"prediction": [0]}"#).unwrap();

    // 3. Scripts and locks
    fs::write(bundle_dir.join("inference.py"), "#!/usr/bin/env python3\nprint('inference')\n").unwrap();
    fs::write(bundle_dir.join("serve.py"), "#!/usr/bin/env python3\nprint('serve')\n").unwrap();
    fs::write(bundle_dir.join("inference.lock"), "numpy==1.26.0\ntorch==2.0.0\n").unwrap();
    fs::write(bundle_dir.join("training.lock"), "numpy==1.26.0\ntorch==2.0.0\n").unwrap();
    fs::write(bundle_dir.join("README.md"), "# Candidate Bundle\n").unwrap();

    // 4. Create manifest
    let mut files = std::collections::HashMap::new();
    for name in [
        "input_schema.json",
        "output_schema.json",
        "scorecard.json",
        "verification.json",
        "final.json",
        "inference.py",
        "inference.lock",
        "training.lock",
        "examples_input.json",
        "examples_expected.json",
        "README.md",
        "serve.py",
        "frozen_model.joblib",
        "weights.pt",
    ] {
        let bytes = fs::read(bundle_dir.join(name)).unwrap();
        files.insert(name.to_string(), bundle::BundleManifestFileEntry {
            bytes: bytes.len() as u64,
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        });
    }
    let manifest = bundle::BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: "p1".into(),
        candidate_id: "cand1".into(),
        files,
        model: None,
        pipeline: None,
        contracts: None,
        dependencies: None,
        scorecard: None,
        lineage: None,
        resource_measurements: None,
    };
    fs::write(bundle_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let rep = bundle::verify_bundle(&bundle_dir, false).unwrap();
    assert!(rep.manifest_verified);
    assert!(rep.contracts_verified);
}

#[test]
fn test_custom_model_spec_validation_rejects_test_access_id() {
    let custom_spec = CustomJobSpec {
        job_id: "job-leak-01".into(),
        project_id: "p1".into(),
        task_id: "t1".into(),
        candidate_id: Some("cand-01".into()),
        generation: 1,
        intent_revision: 1,
        operation: "custom_train".into(),
        model_plugin: "tiny_torch".into(),
        dataset_id: "ds-01".into(),
        schema_version: Some("v2".into()),
        metric_name: "accuracy".into(),
        metric_direction: "maximize".into(),
        max_steps: 10,
        batch_size: 16,
        target_column: Some("target".into()),
        resume_checkpoint: None,
        learning_rate: Some(0.01),
        hidden_dim: Some(16),
        seed: Some(42),
        resources: None,
    };

    let probe = Probe::Scientific {
        operation: "custom_train".into(),
        dataset_id: "ds-01".into(),
        spec: Some(serde_json::to_value(&custom_spec).unwrap()),
        candidate_id: Some("cand-01".into()),
        test_access_id: Some("gate-unauthorized-001".into()), // Attempting to consume sealed test partition!
    };

    let job_spec = JobSpec {
        job_id: "job-leak-01".into(),
        project_id: "p1".into(),
        task_id: "t1".into(),
        generation: 1,
        timeout_ms: 60_000,
        threads: 2,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("env-01".into()),
        resources: Default::default(),
        probe,
    };

    let err = execution::validate_spec(&job_spec).unwrap_err();
    assert!(
        err.contains("custom development operations cannot access sealed test partition"),
        "Expected holdout rejection error, got: {err}"
    );
}

fn make_job_view(spec: JobSpec) -> execution::JobView {
    execution::JobView {
        spec,
        status: "succeeded".into(),
        reason: None,
        pid: None,
        submitted_ms: 1000,
        started_ms: Some(1010),
        ended_ms: Some(1100),
        artifacts: vec![],
        telemetry: None,
        resource_receipt: None,
    }
}

fn make_custom_job_spec(
    project_id: &str,
    job_id: &str,
    operation: &str,
    dataset_id: &str,
    candidate_id: Option<&str>,
    custom_spec: Option<serde_json::Value>,
) -> JobSpec {
    JobSpec {
        job_id: job_id.into(),
        project_id: project_id.into(),
        task_id: "task_001".into(),
        generation: 1,
        timeout_ms: 10000,
        threads: 1,
        max_output_bytes: 10 * 1024 * 1024,
        environment_id: Some("managed_env".into()),
        resources: Default::default(),
        probe: Probe::Scientific {
            operation: operation.into(),
            dataset_id: dataset_id.into(),
            candidate_id: candidate_id.map(|s| s.to_string()),
            test_access_id: None,
            spec: custom_spec,
        },
    }
}

fn insert_job_row(conn: &rusqlite::Connection, spec: &JobSpec) {
    conn.execute(
        "INSERT OR IGNORE INTO branches(project, id, base_hash, code_hash) VALUES(?1, 'b_001', 'base', 'code')",
        rusqlite::params![spec.project_id],
    ).unwrap();
    conn.execute(
        "INSERT OR IGNORE INTO tasks(project, id, branch, data) VALUES(?1, ?2, 'b_001', '{}')",
        rusqlite::params![spec.project_id, spec.task_id],
    ).unwrap();
    let encoded = serde_json::to_string(spec).unwrap();
    conn.execute(
        "INSERT OR REPLACE INTO jobs(id, project, task, generation, spec, status, submitted_ms) VALUES(?1, ?2, ?3, ?4, ?5, 'running', 1000)",
        rusqlite::params![spec.job_id, spec.project_id, spec.task_id, spec.generation, encoded],
    ).unwrap();
}

#[test]
fn test_supervisor_finish_train_verifies_all_provenance_identities() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_train_prov";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let job_id = "job_train_001";
    let candidate = "cand_001";
    let dataset_id = "ds_001";

    let workflow = root.join("workflows").join(project);
    fs::create_dir_all(&workflow).unwrap();

    let manifest_content = json!({
        "format": "mlrescurix.dataset.v1",
        "sha256": "d".repeat(64),
        "dataset_sha256": "d".repeat(64),
        "splits": {
            "train": [0, 1, 2],
            "validation": [3, 4],
            "test": [5]
        }
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest_content).unwrap();
    let split_sha = hash(&manifest_bytes);
    fs::write(workflow.join("dataset_manifest.json"), &manifest_bytes).unwrap();

    let art_dir = root.join("artifacts").join(project).join(job_id);
    fs::create_dir_all(&art_dir).unwrap();

    let weights_data = b"torch_weights_binary_state_dict_12345";
    let weights_sha = hash(weights_data);
    let weights_file = art_dir.join("weights.pt");
    fs::write(&weights_file, weights_data).unwrap();

    let src_sha = mlrescurix_harness::assets::embedded_python_src_hash();
    let mut checkpoint_files = BTreeMap::new();
    checkpoint_files.insert("weights.pt".into(), weights_sha.clone());
    let ckpt = TrainingCheckpoint {
        checkpoint_id: job_id.into(),
        job_id: job_id.into(),
        schema_version: Some("v2".into()),
        project_id: Some(project.into()),
        candidate_id: Some(candidate.into()),
        weights_sha256: Some(weights_sha.clone()),
        step: 10,
        epoch: 1,
        batch_offset: 0,
        sampler_position: 10,
        loss: Some(0.25),
        files: checkpoint_files,
        dataset_sha256: Some("d".repeat(64)),
        split_sha256: Some(split_sha.clone()),
        source_sha256: Some(src_sha.clone()),
        environment_id: Some("managed_env".into()),
        training_config: Some(json!({"lr": 0.01})),
        replay_guarantee: Some("unverified".into()),
        training_state: None,
        timestamp_unix: 1000.0,
    };
    fs::write(art_dir.join("checkpoint.json"), serde_json::to_vec_pretty(&ckpt).unwrap()).unwrap();

    let res = TrainingResult {
        job_id: job_id.into(),
        status: "succeeded".into(),
        schema_version: Some("v2".into()),
        project_id: Some(project.into()),
        candidate_id: Some(candidate.into()),
        weights_sha256: Some(weights_sha.clone()),
        step_reached: 10,
        epoch_reached: 1,
        primary_metric_name: Some("accuracy".into()),
        primary_metric_score: Some(0.92),
        validation_loss: Some(0.25),
        peak_memory_bytes: 1024 * 1024,
        duration_seconds: 5.0,
        artifacts: vec!["weights.pt".into()],
        dataset_sha256: Some("d".repeat(64)),
        split_sha256: Some(split_sha.clone()),
        source_sha256: Some(src_sha),
        environment_id: Some("managed_env".into()),
        training_config: Some(json!({"lr": 0.01})),
        replay_guarantee: Some("unverified".into()),
        limitations: vec![],
        reason: None,
    };
    fs::write(art_dir.join("result.json"), serde_json::to_vec_pretty(&res).unwrap()).unwrap();

    let spec = make_custom_job_spec(project, job_id, "custom_train", dataset_id, Some(candidate), None);
    let view = make_job_view(spec.clone());
    let conn = execution::connect(&root).unwrap();
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![project, dataset_id, "/data.csv", 100, "d".repeat(64), 1000],
    ).unwrap();
    insert_job_row(&conn, &spec);

    // 1. Valid training pass
    supervisor.finish(&view, "succeeded", None, vec![], 100).unwrap();

    // 2. Failure mode: missing weights.pt
    insert_job_row(&conn, &spec);
    fs::remove_file(&weights_file).unwrap();
    let err = supervisor.finish(&view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(err.contains("weights.pt missing"), "Expected weights.pt missing error, got: {err}");

    // 3. Failure mode: altered physical weights
    insert_job_row(&conn, &spec);
    fs::write(&weights_file, b"tampered_weights_binary").unwrap();
    let err2 = supervisor.finish(&view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(err2.contains("does not match physical weights hash"), "Expected weights mismatch error, got: {err2}");

    // Restore correct weights
    fs::write(&weights_file, weights_data).unwrap();

    // 4. Failure mode: missing split_sha256 in result
    insert_job_row(&conn, &spec);
    let mut bad_res = res.clone();
    bad_res.split_sha256 = None;
    fs::write(art_dir.join("result.json"), serde_json::to_vec_pretty(&bad_res).unwrap()).unwrap();
    let err3 = supervisor.finish(&view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(err3.contains("missing split_sha256"), "Expected missing split_sha256 error, got: {err3}");

    // Restore valid result.json
    fs::write(art_dir.join("result.json"), serde_json::to_vec_pretty(&res).unwrap()).unwrap();

    // 5. Failure mode: candidate_id mismatch
    let spec_cand_mismatch = make_custom_job_spec(project, job_id, "custom_train", dataset_id, Some("cand_impostor"), None);
    let view_cand_mismatch = make_job_view(spec_cand_mismatch.clone());
    insert_job_row(&conn, &spec_cand_mismatch);
    let err4 = supervisor.finish(&view_cand_mismatch, "succeeded", None, vec![], 100).unwrap_err();
    assert!(err4.contains("candidate_id in training artifacts does not match"), "Expected candidate mismatch error, got: {err4}");
}

#[test]
fn test_supervisor_finish_evaluate_rejects_worker_reported_metrics() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_eval_gate";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let eval_job_id = "job_eval_001";
    let candidate = "cand_eval_001";
    let dataset_id = "ds_eval_001";

    let workflow = root.join("workflows").join(project);
    let trials_dir = workflow.join("trials").join(candidate);
    fs::create_dir_all(&trials_dir).unwrap();

    let ckpt_dir = workflow.join("checkpoints").join("job_train_ref");
    fs::create_dir_all(&ckpt_dir).unwrap();
    let weights_data = b"evaluated_neural_network_weights_bytes";
    let weights_sha = hash(weights_data);
    fs::write(ckpt_dir.join("weights.pt"), weights_data).unwrap();

    let manifest_content = json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": "d".repeat(64),
        "primary_metric": "accuracy",
        "splits": {
            "train": [0, 1],
            "validation": [2]
        }
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest_content).unwrap();
    let split_sha = hash(&manifest_bytes);
    fs::write(workflow.join("dataset_manifest.json"), &manifest_bytes).unwrap();

    let scorecard = json!({
        "candidate_id": candidate,
        "weights_sha256": weights_sha,
        "primary_metric": "accuracy",
        "primary_metric_score": 0.94,
        "primary_metric_direction": "maximize",
        "dataset_sha256": "d".repeat(64),
        "split_sha256": split_sha,
        "audit_status": "UNVERIFIED_DEVELOPMENT"
    });
    let scorecard_bytes = serde_json::to_vec_pretty(&scorecard).unwrap();
    let expected_sc_hash = hash(&scorecard_bytes);
    fs::write(trials_dir.join("scorecard.json"), &scorecard_bytes).unwrap();

    let mut ckpt_files = BTreeMap::new();
    ckpt_files.insert("weights.pt".into(), weights_sha.clone());
    let ckpt = TrainingCheckpoint {
        checkpoint_id: "job_train_ref".into(),
        job_id: "job_train_ref".into(),
        schema_version: Some("v2".into()),
        project_id: Some(project.into()),
        candidate_id: Some(candidate.into()),
        weights_sha256: Some(weights_sha.clone()),
        step: 10,
        epoch: 1,
        batch_offset: 0,
        sampler_position: 10,
        loss: Some(0.25),
        files: ckpt_files,
        dataset_sha256: Some("d".repeat(64)),
        split_sha256: Some(split_sha),
        source_sha256: Some(mlrescurix_harness::assets::embedded_python_src_hash()),
        environment_id: Some("managed_env".into()),
        training_config: Some(json!({"lr": 0.01})),
        replay_guarantee: Some("unverified".into()),
        training_state: None,
        timestamp_unix: 1000.0,
    };
    fs::write(ckpt_dir.join("checkpoint.json"), serde_json::to_vec_pretty(&ckpt).unwrap()).unwrap();

    let trial_record = json!({
        "project_id": project,
        "candidate_id": candidate,
        "status": "evaluated_development",
        "metric_name": "accuracy",
        "metric_score": 0.94,
        "weights_sha256": weights_sha,
        "source_sha256": mlrescurix_harness::assets::embedded_python_src_hash(),
        "environment_id": "managed_env",
        "seed": 777,
        "scorecard_hash": expected_sc_hash,
        "checkpoint_ref": "job_train_ref"
    });
    fs::write(trials_dir.join("trial.json"), serde_json::to_vec_pretty(&trial_record).unwrap()).unwrap();

    let spec = make_custom_job_spec(project, eval_job_id, "custom_evaluate", dataset_id, Some(candidate), None);
    let view = make_job_view(spec.clone());
    let conn = execution::connect(&root).unwrap();
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![project, dataset_id, "/data.csv", 100, "d".repeat(64), 1000],
    ).unwrap();
    insert_job_row(&conn, &spec);

    let err = supervisor.finish(&view, "succeeded", None, vec![], 150).unwrap_err();
    assert!(err.contains("controller-owned inference and metric verification"));
    let count: i64 = conn.query_row(
        "SELECT count(*) FROM candidate_evaluations WHERE project=?1 AND candidate_id=?2",
        rusqlite::params![project, candidate], |r| r.get(0),
    ).unwrap();
    assert_eq!(count, 0, "a consistent worker claim is not an authoritative evaluation");
}

#[test]
fn test_supervisor_finish_export_rejects_altered_checkpoint_weights_after_evaluation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_export_tamper";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let candidate = "cand_tamper_001";
    let dataset_id = "ds_tamper_001";
    let conn = execution::connect(&root).unwrap();

    let true_weights_data = b"authorized_checkpoint_weights";
    let true_weights_sha = hash(true_weights_data);
    let true_sc_data = b"authorized_scorecard_json";
    let true_sc_sha = hash(true_sc_data);

    // Record approved evaluation
    let eval_id = "eval_tamper_001";
    let metrics_json = json!({"weights_sha256": true_weights_sha, "accuracy": 0.95}).to_string();
    conn.execute(
        "INSERT INTO candidate_evaluations(project, evaluation_id, candidate_id, dataset_id, environment_id, source_hash, status, folds, seed, primary_metric, metrics_json, scorecard_hash, created_ms)
         VALUES(?1, ?2, ?3, ?4, 'env_01', 'src_01', 'evaluated', 1, 123, 0.95, ?5, ?6, 1000)",
        rusqlite::params![project, eval_id, candidate, dataset_id, metrics_json, true_sc_sha],
    ).unwrap();

    // Prepare workflow trials with true weights
    let workflow = root.join("workflows").join(project);
    let trials_dir = workflow.join("trials").join(candidate);
    fs::create_dir_all(&trials_dir).unwrap();
    let trial_record = json!({
        "status": "evaluated_development",
        "weights_sha256": true_weights_sha,
        "checkpoint_ref": "ckpt_orig"
    });
    fs::write(trials_dir.join("trial.json"), serde_json::to_vec(&trial_record).unwrap()).unwrap();

    // Prepare bundle directory with ALTERED physical weights
    let bundle_dir = workflow.join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();
    let altered_weights_data = b"tampered_export_weights_different";
    let _altered_weights_sha = hash(altered_weights_data);
    fs::write(bundle_dir.join("weights.pt"), altered_weights_data).unwrap();
    fs::write(bundle_dir.join("scorecard.json"), true_sc_data).unwrap();

    let export_spec = make_custom_job_spec(project, "job_export_001", "custom_export", dataset_id, Some(candidate), None);
    let export_view = make_job_view(export_spec.clone());
    insert_job_row(&conn, &export_spec);

    let err = supervisor.finish(&export_view, "succeeded", None, vec![], 200).unwrap_err();
    assert!(
        err.contains("does not match approved evaluation weights"),
        "Expected export weights tampering rejection, got: {err}"
    );
}

#[test]
fn test_supervisor_finish_export_rejects_cross_dataset() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_cross_ds";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let candidate = "cand_cross_001";
    let conn = execution::connect(&root).unwrap();

    // Record evaluation on dataset_A
    let weights_data = b"cross_ds_weights";
    let weights_sha = hash(weights_data);
    let sc_data = b"cross_ds_scorecard";
    let sc_sha = hash(sc_data);
    let metrics_json = json!({"weights_sha256": weights_sha}).to_string();
    conn.execute(
        "INSERT INTO candidate_evaluations(project, evaluation_id, candidate_id, dataset_id, environment_id, source_hash, status, folds, seed, primary_metric, metrics_json, scorecard_hash, created_ms)
         VALUES(?1, 'eval_cross', ?2, 'dataset_A', 'env_01', 'src_01', 'evaluated', 1, 123, 0.90, ?3, ?4, 1000)",
        rusqlite::params![project, candidate, metrics_json, sc_sha],
    ).unwrap();

    let workflow = root.join("workflows").join(project);
    let bundle_dir = workflow.join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Attempt export on dataset_B
    let export_spec = make_custom_job_spec(project, "job_export_cross", "custom_export", "dataset_B", Some(candidate), None);
    let export_view = make_job_view(export_spec.clone());
    insert_job_row(&conn, &export_spec);

    let err = supervisor.finish(&export_view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(
        err.contains("no approved record in candidate_evaluations table for dataset 'dataset_B'"),
        "Expected cross-dataset export rejection, got: {err}"
    );
}

#[test]
fn test_supervisor_finish_export_respects_operator_user_pin() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_user_pin";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let candidate = "cand_unpinned";
    let dataset_id = "ds_pin";
    let conn = execution::connect(&root).unwrap();

    // Record evaluation
    let weights_data = b"pin_weights";
    let weights_sha = hash(weights_data);
    let sc_data = b"pin_scorecard";
    let sc_sha = hash(sc_data);
    let metrics_json = json!({"weights_sha256": weights_sha}).to_string();
    conn.execute(
        "INSERT INTO candidate_evaluations(project, evaluation_id, candidate_id, dataset_id, environment_id, source_hash, status, folds, seed, primary_metric, metrics_json, scorecard_hash, created_ms)
         VALUES(?1, 'eval_pin', ?2, ?3, 'env_01', 'src_01', 'evaluated', 1, 123, 0.90, ?4, ?5, 1000)",
        rusqlite::params![project, candidate, dataset_id, metrics_json, sc_sha],
    ).unwrap();

    let workflow = root.join("workflows").join(project);
    let bundle_dir = workflow.join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Set operator pin in results.json to a different candidate
    let results = json!({
        "pinned_candidate": "cand_operator_selected_golden"
    });
    fs::write(workflow.join("results.json"), serde_json::to_vec(&results).unwrap()).unwrap();

    let export_spec = make_custom_job_spec(project, "job_export_pin", "custom_export", dataset_id, Some(candidate), None);
    let export_view = make_job_view(export_spec.clone());
    insert_job_row(&conn, &export_spec);

    let err = supervisor.finish(&export_view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(
        err.contains("results.json operator pin is set to 'cand_operator_selected_golden'"),
        "Expected operator pin preservation rejection, got: {err}"
    );
}

#[test]
fn test_supervisor_finish_export_rejects_overwriting_pinned_bundle() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    let project = "proj_pinned_artifact";
    create_m1(&r, project);
    let supervisor = Supervisor::start(&root, 1).unwrap();
    let candidate = "cand_bundle_001";
    let dataset_id = "ds_bundle";
    let conn = execution::connect(&root).unwrap();

    let workflow = root.join("workflows").join(project);
    let bundle_dir = workflow.join("bundle");
    fs::create_dir_all(&bundle_dir).unwrap();

    // Insert an existing pinned artifact in artifacts table (using valid kind apex_bundle)
    conn.execute(
        "INSERT INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
         VALUES(?1, 'bundle/manifest.json', 'apex_bundle', 'evaluator:finalizer', 'v1', 1, 'project', 100, 100)",
        rusqlite::params![project],
    ).unwrap();

    let export_spec = make_custom_job_spec(project, "job_export_over", "custom_export", dataset_id, Some(candidate), None);
    let export_view = make_job_view(export_spec.clone());
    insert_job_row(&conn, &export_spec);

    let err = supervisor.finish(&export_view, "succeeded", None, vec![], 100).unwrap_err();
    assert!(
        err.contains("pinned delivery bundle already exists in artifacts; overwriting pinned artifact is prohibited"),
        "Expected pinned bundle overwrite rejection, got: {err}"
    );
}

#[test]
fn test_dev_splitter_streaming_record_aware_multiline_and_quoted_csv() {
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let manifest = tmp.path().join("manifest.json");
    let out = tmp.path().join("dev.csv");

    let raw_csv_content = concat!(
        "feature,details,target\n",
        "\"test_val\",\"super\nsecret\r\ntest data with \"\"quotes\"\" and, commas\",1\n",
        "\"train_val\",\"regular text\",0\n",
        "\"train_multi\",\"first line\nsecond line with λ and 🚀\",1\n",
        "\"val_val\",\"escaped \"\"quote\"\" text\",0\n"
    );
    fs::write(&raw, raw_csv_content).unwrap();
    let raw_bytes = fs::read(&raw).unwrap();
    let raw_sha = hash(&raw_bytes);

    let manifest_obj = json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": raw_sha,
        "splits": {
            "train": [1, 2],
            "validation": [3],
            "test": [0]
        }
    });
    let manifest_bytes = serde_json::to_vec_pretty(&manifest_obj).unwrap();
    let original_manifest_hash = hash(&manifest_bytes);
    fs::write(&manifest, &manifest_bytes).unwrap();

    let dev_sha = execution::prepare_dev_dataset(&raw, &manifest, &out).unwrap();
    assert!(!dev_sha.is_empty());
    assert!(out.exists());

    // Verify manifest immutability: manifest on disk was strictly NOT modified
    let disk_manifest_bytes = fs::read(&manifest).unwrap();
    assert_eq!(hash(&disk_manifest_bytes), original_manifest_hash, "Manifest must remain strictly immutable");

    // Read dev.csv and verify isolation and content
    let dev_content = fs::read_to_string(&out).unwrap();
    assert!(!dev_content.contains("super"), "Sealed test partition leaked into dev dataset!");
    assert!(!dev_content.contains("secret"), "Sealed test partition leaked into dev dataset!");
    assert!(dev_content.starts_with("__mlr_row_id,feature,details,target\n"));

    // Verify dev rows have correct prepended IDs
    assert!(dev_content.contains("1,train_val"));
    assert!(dev_content.contains("2,train_multi"));
    assert!(dev_content.contains("3,val_val"));
}

#[test]
fn test_dev_splitter_adversarial_rejections() {
    let tmp = tempfile::tempdir().unwrap();

    // 1. Unclosed quote in raw CSV
    let raw_unclosed = tmp.path().join("unclosed.csv");
    let manifest_unclosed = tmp.path().join("man_unclosed.json");
    let out_unclosed = tmp.path().join("dev_unclosed.csv");
    fs::write(&raw_unclosed, "a,b\n\"unclosed,1\n2,0\n3,1\n").unwrap();
    let sha1 = hash(&fs::read(&raw_unclosed).unwrap());
    fs::write(&manifest_unclosed, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": sha1,
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    let err = execution::prepare_dev_dataset(&raw_unclosed, &manifest_unclosed, &out_unclosed).unwrap_err();
    assert!(err.contains("unclosed quote") || err.contains("malformed CSV"), "Expected unclosed quote error, got: {err}");
    assert!(!out_unclosed.exists(), "failed parsing must not publish partial output");

    // 2. Reserved __mlr_row_id column in raw dataset
    let raw_reserved = tmp.path().join("reserved.csv");
    let manifest_reserved = tmp.path().join("man_reserved.json");
    let out_reserved = tmp.path().join("dev_reserved.csv");
    fs::write(&raw_reserved, "__mlr_row_id,feature,target\n0,val,1\n1,v2,0\n2,v3,1\n").unwrap();
    let sha2 = hash(&fs::read(&raw_reserved).unwrap());
    fs::write(&manifest_reserved, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": sha2,
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    let err2 = execution::prepare_dev_dataset(&raw_reserved, &manifest_reserved, &out_reserved).unwrap_err();
    assert!(err2.contains("__mlr_row_id") && err2.contains("not permitted"), "Expected reserved column error, got: {err2}");

    // 3. Duplicate / overlapping split indices
    let raw_dup = tmp.path().join("dup.csv");
    let manifest_dup = tmp.path().join("man_dup.json");
    let out_dup = tmp.path().join("dev_dup.csv");
    fs::write(&raw_dup, "f,t\n1,0\n2,1\n3,0\n").unwrap();
    let sha3 = hash(&fs::read(&raw_dup).unwrap());
    fs::write(&manifest_dup, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": sha3,
        "splits": {"train": [0], "validation": [0], "test": [1]}
    })).unwrap()).unwrap();
    let err3 = execution::prepare_dev_dataset(&raw_dup, &manifest_dup, &out_dup).unwrap_err();
    assert!(err3.contains("data leakage") && err3.contains("overlap"), "Expected partition overlap error, got: {err3}");

    // 4. Raw dataset SHA-256 mismatch against manifest
    let raw_mismatch = tmp.path().join("mismatch.csv");
    let manifest_mismatch = tmp.path().join("man_mismatch.json");
    let out_mismatch = tmp.path().join("dev_mismatch.csv");
    fs::write(&raw_mismatch, "f,t\n1,0\n2,1\n3,0\n").unwrap();
    fs::write(&manifest_mismatch, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1",
        "dataset_sha256": "f".repeat(64),
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    let err4 = execution::prepare_dev_dataset(&raw_mismatch, &manifest_mismatch, &out_mismatch).unwrap_err();
    assert!(err4.contains("does not match registered manifest"), "Expected dataset SHA mismatch, got: {err4}");
    assert!(!out_mismatch.exists(), "hash mismatch must not publish output");
}

#[test]
fn test_dev_splitter_never_overwrites_and_fences_concurrent_publication() {
    use std::sync::{Arc, Barrier};
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let manifest = tmp.path().join("manifest.json");
    let out = tmp.path().join("dev.csv");
    let bytes = b"f,t\n1,0\n2,1\n3,0\n";
    fs::write(&raw, bytes).unwrap();
    fs::write(&manifest, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1", "dataset_sha256": hash(bytes),
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let threads: Vec<_> = (0..2).map(|_| {
        let (raw, manifest, out, barrier) = (raw.clone(), manifest.clone(), out.clone(), barrier.clone());
        std::thread::spawn(move || {
            barrier.wait();
            execution::prepare_dev_dataset(&raw, &manifest, &out)
        })
    }).collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    let published = fs::read(&out).unwrap();
    assert!(!String::from_utf8_lossy(&published).contains("3,0"));
    assert!(execution::prepare_dev_dataset(&raw, &manifest, &out).is_err());
    assert_eq!(fs::read(&out).unwrap(), published);
    assert!(!fs::read_dir(tmp.path()).unwrap().any(|e| e.unwrap().file_name().to_string_lossy().starts_with(".mlr-dev-")));
}

#[test]
fn test_dev_splitter_rejects_raw_and_destination_symlinks() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let manifest = tmp.path().join("manifest.json");
    let target = tmp.path().join("target");
    let out = tmp.path().join("dev.csv");
    let bytes = b"f,t\n1,0\n2,1\n3,0\n";
    fs::write(&raw, bytes).unwrap();
    fs::write(&manifest, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1", "dataset_sha256": hash(bytes),
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    let raw_link = tmp.path().join("raw-link");
    symlink(&raw, &raw_link).unwrap();
    assert!(execution::prepare_dev_dataset(&raw_link, &manifest, &out).is_err());
    fs::write(&target, b"unchanged").unwrap();
    symlink(&target, &out).unwrap();
    assert!(execution::prepare_dev_dataset(&raw, &manifest, &out).is_err());
    assert_eq!(fs::read(&target).unwrap(), b"unchanged");
}

#[test]
fn test_dev_splitter_distinguishes_blank_lines_from_quoted_empty_records() {
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let manifest = tmp.path().join("manifest.json");
    let out = tmp.path().join("dev.csv");
    let bytes = b"value\n\n\"\"\nvalidation\nsecret\n";
    fs::write(&raw, bytes).unwrap();
    fs::write(&manifest, serde_json::to_vec(&json!({
        "format": "mlrescurix.dataset.v1", "dataset_sha256": hash(bytes),
        "splits": {"train": [0], "validation": [1], "test": [2]}
    })).unwrap()).unwrap();
    execution::prepare_dev_dataset(&raw, &manifest, &out).unwrap();
    assert_eq!(fs::read_to_string(out).unwrap(), "__mlr_row_id,value\n0,\n1,validation\n");
}
