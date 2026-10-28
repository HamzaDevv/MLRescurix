use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use mlrescurix_harness::{
    bundle,
    contracts::{Amounts, Context, CustomJobResources, CustomJobSpec, TrainingCheckpoint, TrainingResult},
    core::Operation,
    execution::{self, JobSpec, Probe, Supervisor},
    Policy,
    Request, Runtime,
};

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
#[ignore = "experimental custom execution is disabled pending governed isolation and evaluation"]
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
    setup_ready_environment(&root, env_id);

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    // 1. Submit custom_train
    let train_spec = CustomJobSpec {
        job_id: "job-train-01".into(),
        project_id: project.into(),
        task_id: task.into(),
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
        resume_checkpoint: Some(root.join("workflows").join(project).join("checkpoints").join("job-train-01").to_string_lossy().to_string()),
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

    // 3. Submit custom_export to generate clean-room delivery bundle
    let export_spec = CustomJobSpec {
        job_id: "job-export-03".into(),
        project_id: project.into(),
        task_id: task.into(),
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
        resume_checkpoint: Some(root.join("workflows").join(project).join("checkpoints").join("job-resume-02_resumed").to_string_lossy().to_string()),
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
        job_id: "job-export-03".into(),
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
    let status_export = run_until_terminal(&mut supervisor, &root, project, "job-export-03");
    assert_eq!(status_export, "succeeded");

    let export_art_dir = root.join("artifacts").join(project).join("job-export-03");
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
#[ignore = "experimental custom execution is disabled pending governed isolation and evaluation"]
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
    setup_ready_environment(&root, env_id);

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    let slow_spec = CustomJobSpec {
        job_id: "job-slow-01".into(),
        project_id: project.into(),
        task_id: task.into(),
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
}

#[test]
#[ignore = "experimental custom execution is disabled pending governed isolation and evaluation"]
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
    setup_ready_environment(&root, env_id);

    let ds_csv = temp.path().join("dataset.csv");
    create_synthetic_dataset(&ds_csv);
    execution::register_dataset(&root, project, dataset_id, &ds_csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut supervisor = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let (task_gen, intent) = get_task_generation_and_intent(&root, project, task);

    // Run enough steps so supervisor reap() samples RSS and detects exceeded memory limit
    let oom_spec = CustomJobSpec {
        job_id: "job-oom-01".into(),
        project_id: project.into(),
        task_id: task.into(),
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
    assert_eq!(view.reason, Some("memory limit exceeded".into()));
}

#[test]
fn test_custom_model_v2_contracts_serialization_and_validation() {
    let spec = CustomJobSpec {
        job_id: "job-schema-01".into(),
        project_id: "p-schema".into(),
        task_id: "t-schema".into(),
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
    let parsed: CustomJobSpec = serde_json::from_str(&spec_json).unwrap();
    assert_eq!(parsed.schema_version.as_deref(), Some("v2"));
    parsed.validate().unwrap();

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
            candidate_id: None,
            test_access_id: None,
        },
    };
    assert!(execution::validate_spec(&job).unwrap_err().contains("experimental"));

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
    let parsed_ckpt: TrainingCheckpoint = serde_json::from_str(&ckpt_json).unwrap();
    assert_eq!(parsed_ckpt.schema_version.as_deref(), Some("v2"));
    parsed_ckpt.validate().unwrap();

    let result = TrainingResult {
        job_id: "job-schema-01".into(),
        status: "succeeded".into(),
        schema_version: Some("v2".into()),
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
    let parsed_result: TrainingResult = serde_json::from_str(&result_json).unwrap();
    assert_eq!(parsed_result.schema_version.as_deref(), Some("v2"));
    parsed_result.validate().unwrap();
}
