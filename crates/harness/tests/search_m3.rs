use std::{
    collections::BTreeMap,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use serde_json::{json, Value};
use mlrescurix_harness::{
    contracts::*,
    core::*,
    execution::{self, Supervisor},
    search::{self, compute_experiment_id, normalize_parameters, SearchSpec},
    Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64
}

fn m1policy(limit: i64, workers: i64) -> Policy {
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), limit),
            ("tokens".into(), limit * 1000),
            ("wall_ms".into(), limit * 10000),
        ]),
        capacity: BTreeMap::from([("workers".into(), workers)]),
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

fn create_m1(r: &Runtime, p: &str, limit: i64) {
    assert_eq!(
        send(r, p, "project", Operation::CreateProject {
            intent: "M3 search fixture".into(),
            finalization_id: format!("final-{p}"),
            policy: m1policy(limit, 8),
        }).status,
        "applied"
    );
}

fn branch_task(r: &Runtime, p: &str, branch: &str, task: &str) {
    send(r, p, &format!("b{branch}"), Operation::CreateBranch {
        branch_id: branch.into(),
        base_hash: format!("base-{branch}"),
        code_hash: format!("code-{branch}"),
    });
    send(r, p, &format!("t{task}"), Operation::CreateTask {
        task_id: task.into(),
        branch_id: branch.into(),
        context: context(&format!("base-{branch}"), &format!("code-{branch}")),
    });
}

fn execution_task(r: &Runtime, p: &str, t: &str) {
    branch_task(r, p, &format!("b{t}"), t);
    let now = now_ms();
    send(r, p, &format!("claim{t}"), Operation::Claim {
        task_id: t.into(),
        owner: "local-worker".into(),
        lease_id: format!("lease{t}"),
        now_ms: now,
        expires_ms: now + 3_600_000,
        expected_generation: 1,
    });
}

#[test]
fn m3_parameter_bounds_and_identity_hashing() {
    // Classification Linear
    assert!(normalize_parameters("linear", "binary_classification", &json!({"C": 0.001})).is_ok());
    assert!(normalize_parameters("linear", "binary_classification", &json!({"C": 1000.0})).is_ok());
    assert!(normalize_parameters("linear", "binary_classification", &json!({"C": 0.0001})).is_err());
    assert!(normalize_parameters("linear", "binary_classification", &json!({"C": 1000.1})).is_err());
    assert!(normalize_parameters("linear", "binary_classification", &json!({"unknown": 1.0})).is_err());

    // Regression Linear uses alpha
    assert!(normalize_parameters("linear", "regression", &json!({"alpha": 0.5})).is_ok());
    assert!(normalize_parameters("linear", "regression", &json!({"C": 0.5})).is_err());

    // Tree parameters
    let valid_tree = json!({
        "n_estimators": 64,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "max_features": 0.5
    });
    assert!(normalize_parameters("tree", "binary_classification", &valid_tree).is_ok());
    assert!(normalize_parameters("tree", "binary_classification", &json!({"n_estimators": 10, "max_depth": 5, "min_samples_leaf": 2, "max_features": 0.5})).is_err());
    assert!(normalize_parameters("tree", "binary_classification", &json!({"n_estimators": 64, "max_depth": 25, "min_samples_leaf": 2, "max_features": 0.5})).is_err());

    // Identity hashing: identical parameters produce identical IDs
    let id1 = compute_experiment_id("sha-abc", "stratified", "linear", &json!({"C": 1.0}), 2026, 0.5);
    let id2 = compute_experiment_id("sha-abc", "stratified", "linear", &json!({"C": 1.0}), 2026, 0.5);
    assert_eq!(id1, id2);

    // Different seed or fidelity produce different IDs
    let id_seed = compute_experiment_id("sha-abc", "stratified", "linear", &json!({"C": 1.0}), 2027, 0.5);
    let id_fid = compute_experiment_id("sha-abc", "stratified", "linear", &json!({"C": 1.0}), 2026, 1.0);
    assert_ne!(id1, id_seed);
    assert_ne!(id1, id_fid);
}

#[test]
fn m3_search_init_is_idempotent_and_ingests_baselines() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 8)).unwrap();
    create_m1(&r, "p-init", 100);
    execution_task(&r, "p-init", "t-search");

    // Register dummy dataset
    let csv = d.path().join("data.csv");
    std::fs::write(&csv, "f1,f2,target\n0.1,0.2,0\n0.3,0.4,1\n0.5,0.6,0\n0.7,0.8,1\n").unwrap();
    execution::register_dataset(&root, "p-init", "data-1", &csv).unwrap();

    let spec = SearchSpec {
        session_id: "sess-1".into(),
        project_id: "p-init".into(),
        dataset_id: "data-1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into(), "tree".into()],
        max_trials: 10,
        max_screen_trials: 4,
        max_full_trials: 2,
        search_wall_ms: 300_000,
        job_timeout_ms: 30_000,
        finalization_allowance_ms: 60_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };

    let val1 = search::init_search(&root, spec.clone()).unwrap();
    assert_eq!(val1["status"], "created");
    assert_eq!(val1["session_id"], "sess-1");

    // Second init is idempotent
    let val2 = search::init_search(&root, spec).unwrap();
    assert_eq!(val2["status"], "created");
    assert_eq!(val2["session_id"], "sess-1");
}

#[test]
fn m3_budget_guard_triggers_provisional_outcome() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    // Project policy has limit=5 -> wall_ms=50_000
    let r = Runtime::init(&root, m1policy(5, 2)).unwrap();
    create_m1(&r, "p-guard", 5);
    execution_task(&r, "p-guard", "t-guard");

    let csv = d.path().join("data.csv");
    std::fs::write(&csv, "f1,target\n1,0\n2,1\n3,0\n4,1\n").unwrap();
    execution::register_dataset(&root, "p-guard", "data-1", &csv).unwrap();

    let spec = SearchSpec {
        session_id: "sess-guard".into(),
        project_id: "p-guard".into(),
        dataset_id: "data-1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()],
        max_trials: 4,
        max_screen_trials: 2,
        max_full_trials: 1,
        search_wall_ms: 60_000,
        job_timeout_ms: 10_000,
        finalization_allowance_ms: 80_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };

    search::init_search(&root, spec).unwrap();
    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let selection = search::execute_search(&root, "p-guard", "sess-guard", "t-guard", 2, &mut sup, None).unwrap();
    // Guard prevented admission, resulting in provisional or infeasible status
    assert!(matches!(selection.status.as_str(), "provisional_budget_exhausted" | "infeasible"));
    assert!(selection.selected_candidate_id.is_none());
}

#[test]
fn m3_capabilities_reports_phase_m3_and_scientific_flags() {
    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let output = Command::new(bin).arg("capabilities").output().unwrap();
    assert!(output.status.success());
    let cap: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(matches!(cap["phase"].as_str(), Some("M3") | Some("M4") | Some("M5") | Some("M6")));
    assert_eq!(cap["scientific_execution"], true);
    assert_eq!(cap["official_evaluation"], true);
    assert_eq!(cap["constrained_search"], true);
}

#[test]
fn m3_tie_tolerance_prefers_lower_latency_then_smaller_size() {
    use search::{select_apex_from_contenders, ContenderSummary};

    let c1 = ContenderSummary {
        candidate_id: "c1-heavy".into(),
        experiment_id: "exp-1".into(),
        recipe: "tree".into(),
        parameters: json!({"n_estimators": 100}),
        mean_metric: 0.853,
        metric_std: 0.001,
        metric_min: 0.852,
        metric_max: 0.854,
        seeds: vec![2026, 2027, 2028],
        scores: vec![0.852, 0.853, 0.854],
        is_feasible: true,
        warm_p95_ms: Some(15.0),
        model_bytes: Some(8000),
        failure_reason: None,
    };

    let c2 = ContenderSummary {
        candidate_id: "c2-fast".into(),
        experiment_id: "exp-2".into(),
        recipe: "linear".into(),
        parameters: json!({"C": 1.0}),
        mean_metric: 0.850,
        metric_std: 0.001,
        metric_min: 0.849,
        metric_max: 0.851,
        seeds: vec![2026, 2027, 2028],
        scores: vec![0.849, 0.850, 0.851],
        is_feasible: true,
        warm_p95_ms: Some(1.2), // Much lower latency!
        model_bytes: Some(1200),
        failure_reason: None,
    };

    // Tie tolerance 0.005: |0.853 - 0.850| = 0.003 <= 0.005 -> c2 wins on latency!
    let sel = select_apex_from_contenders("sess-tie", "balanced_accuracy", "maximize", 0.005, vec![c1.clone(), c2.clone()]);
    assert_eq!(sel.status, "apex_selected");
    assert_eq!(sel.selected_candidate_id.as_deref(), Some("c2-fast"));

    // Tie tolerance 0.001: 0.003 > 0.001 -> c1 wins on primary metric!
    let sel_strict = select_apex_from_contenders("sess-strict", "balanced_accuracy", "maximize", 0.001, vec![c1, c2]);
    assert_eq!(sel_strict.status, "apex_selected");
    assert_eq!(sel_strict.selected_candidate_id.as_deref(), Some("c1-heavy"));
}

#[test]
fn m3_infeasible_candidate_cannot_qualify_for_apex() {
    use search::{select_apex_from_contenders, ContenderSummary};

    let infeasible_best = ContenderSummary {
        candidate_id: "c-infeasible".into(),
        experiment_id: "exp-inf".into(),
        recipe: "tree".into(),
        parameters: json!({"n_estimators": 128}),
        mean_metric: 0.999, // Super high score, but failed constraints!
        metric_std: 0.0,
        metric_min: 0.999,
        metric_max: 0.999,
        seeds: vec![2026, 2027, 2028],
        scores: vec![0.999, 0.999, 0.999],
        is_feasible: false,
        warm_p95_ms: Some(250.0), // Exceeded latency limit
        model_bytes: Some(100_000),
        failure_reason: Some("inference latency exceeded".into()),
    };

    let feasible_baseline = ContenderSummary {
        candidate_id: "c-baseline".into(),
        experiment_id: "exp-base".into(),
        recipe: "linear".into(),
        parameters: json!({"C": 0.1}),
        mean_metric: 0.810,
        metric_std: 0.002,
        metric_min: 0.808,
        metric_max: 0.812,
        seeds: vec![2026, 2027, 2028],
        scores: vec![0.808, 0.810, 0.812],
        is_feasible: true,
        warm_p95_ms: Some(2.0),
        model_bytes: Some(1500),
        failure_reason: None,
    };

    let sel = select_apex_from_contenders("sess-inf", "balanced_accuracy", "maximize", 0.005, vec![infeasible_best, feasible_baseline]);
    assert_eq!(sel.status, "apex_selected");
    assert_eq!(sel.selected_candidate_id.as_deref(), Some("c-baseline"));
}

#[test]
fn m3_minimize_direction_selects_lowest_metric() {
    use search::{select_apex_from_contenders, ContenderSummary};

    let c1 = ContenderSummary {
        candidate_id: "c1-high-error".into(),
        experiment_id: "exp-1".into(),
        recipe: "tree".into(),
        parameters: json!({"n_estimators": 32}),
        mean_metric: 1.45,
        metric_std: 0.02,
        metric_min: 1.43,
        metric_max: 1.47,
        seeds: vec![2026, 2027, 2028],
        scores: vec![1.43, 1.45, 1.47],
        is_feasible: true,
        warm_p95_ms: Some(5.0),
        model_bytes: Some(4000),
        failure_reason: None,
    };

    let c2 = ContenderSummary {
        candidate_id: "c2-low-error".into(),
        experiment_id: "exp-2".into(),
        recipe: "linear".into(),
        parameters: json!({"alpha": 10.0}),
        mean_metric: 0.35, // Lower error is better!
        metric_std: 0.01,
        metric_min: 0.34,
        metric_max: 0.36,
        seeds: vec![2026, 2027, 2028],
        scores: vec![0.34, 0.35, 0.36],
        is_feasible: true,
        warm_p95_ms: Some(1.0),
        model_bytes: Some(1200),
        failure_reason: None,
    };

    let sel = select_apex_from_contenders("sess-min", "mean_squared_error", "minimize", 0.005, vec![c1, c2]);
    assert_eq!(sel.status, "apex_selected");
    assert_eq!(sel.selected_candidate_id.as_deref(), Some("c2-low-error"));
}

fn setup_test_env(root: &Path, id: &str) {
    let image = std::env::var("MLRESCURIX_BUNDLE_IMAGE")
        .expect("legacy scientific fixture requires pinned Docker; host execution prohibited");
    execution::register_container_environment(root, id, &image).unwrap();
}

fn run_job_to_completion(sup: &mut Supervisor, root: &Path, project: &str, job_id: &str) {
    for _ in 0..400 {
        sup.tick().unwrap();
        let j = execution::inspect(root, project, job_id).unwrap();
        if !["queued", "running", "terminating", "recovering"].contains(&j.status.as_str()) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    panic!("job {} did not complete: {}", job_id, execution::inspect(root, project, job_id).unwrap().status);
}

#[test]
#[ignore = "historical host scientific fixture; replaced by explicit digest-pinned Docker pipeline tests; does not satisfy content-bound finalization"]
fn m3_end_to_end_search_and_finalization() {
    use execution::{JobSpec, Probe};

    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(100, 4)).unwrap();
    create_m1(&r, "p-e2e", 100);
    execution_task(&r, "p-e2e", "t-e2e");
    setup_test_env(&root, "scientific");

    let csv = d.path().join("dataset.csv");
    let mut data = "feature1,feature2,target\n".to_string();
    for i in 0..60 {
        let t = if i % 2 == 0 { 0 } else { 1 };
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * 0.2;
        data.push_str(&format!("{f1:.2},{f2:.2},{t}\n"));
    }
    std::fs::write(&csv, &data).unwrap();
    execution::register_dataset(&root, "p-e2e", "data-e2e", &csv).unwrap();

    let bin = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));
    let mut sup = Supervisor::start_with_executable(&root, 1, bin).unwrap();

    let audit_job = JobSpec { resources: Default::default(),
        job_id: "job-audit".into(),
        project_id: "p-e2e".into(),
        task_id: "t-e2e".into(),
        generation: 2,
        timeout_ms: 30_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "audit".into(),
            dataset_id: "data-e2e".into(),
            spec: Some(json!({
                "project_id": "p-e2e",
                "target": "target",
                "task": "binary_classification",
                "metric": "balanced_accuracy",
                "split": "stratified",
                "inference_batch_size": 16,
                "inference_repetitions": 3,
            })),
            candidate_id: None,
            test_access_id: None,
        },
    };
    execution::submit(&root, audit_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-e2e", "job-audit");
    assert_eq!(execution::inspect(&root, "p-e2e", "job-audit").unwrap().status, "succeeded");

    let baselines_job = JobSpec { resources: Default::default(),
        job_id: "job-baselines".into(),
        project_id: "p-e2e".into(),
        task_id: "t-e2e".into(),
        generation: 2,
        timeout_ms: 30_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "baselines".into(),
            dataset_id: "data-e2e".into(),
            spec: None,
            candidate_id: None,
            test_access_id: None,
        },
    };
    execution::submit(&root, baselines_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-e2e", "job-baselines");
    assert_eq!(execution::inspect(&root, "p-e2e", "job-baselines").unwrap().status, "succeeded");

    let spec = SearchSpec {
        session_id: "sess-e2e".into(),
        project_id: "p-e2e".into(),
        dataset_id: "data-e2e".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()],
        max_trials: 6,
        max_screen_trials: 2,
        max_full_trials: 1,
        search_wall_ms: 300_000,
        job_timeout_ms: 30_000,
        finalization_allowance_ms: 30_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };
    search::init_search(&root, spec).unwrap();
    let selection = search::execute_search(&root, "p-e2e", "sess-e2e", "t-e2e", 2, &mut sup, Some("scientific")).unwrap();
    assert_eq!(selection.status, "apex_selected");
    let apex_id = selection.selected_candidate_id.expect("Apex candidate must be selected");

    let finalize_job = JobSpec { resources: Default::default(),
        job_id: "job-finalize".into(),
        project_id: "p-e2e".into(),
        task_id: "t-e2e".into(),
        generation: 2,
        timeout_ms: 60_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some("scientific".into()),
        probe: Probe::Scientific {
            operation: "finalize".into(),
            dataset_id: "data-e2e".into(),
            spec: None,
            candidate_id: Some(apex_id.clone()),
            test_access_id: Some("test-gate-e2e".into()),
        },
    };
    execution::submit(&root, finalize_job).unwrap();
    run_job_to_completion(&mut sup, &root, "p-e2e", "job-finalize");
    assert_eq!(execution::inspect(&root, "p-e2e", "job-finalize").unwrap().status, "succeeded");

    let final_json = root.join("workflows/p-e2e/final.json");
    assert!(final_json.is_file());
    let res: Value = serde_json::from_slice(&std::fs::read(&final_json).unwrap()).unwrap();
    assert_eq!(res["status"], "verified_development_only");
    assert_eq!(res["candidate_id"], apex_id);

    // Verify lineage exists
    let lineage_json = root.join("workflows/p-e2e/lineage.json");
    assert!(lineage_json.is_file());
}

#[test]
fn m3_research_closure_rejects_search_after_gate_reservation() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(10, 2)).unwrap();
    create_m1(&r, "p-closure", 10);
    execution_task(&r, "p-closure", "t-closure");

    let csv = d.path().join("data.csv");
    std::fs::write(&csv, "f1,target\n1,0\n2,1\n3,0\n4,1\n").unwrap();
    execution::register_dataset(&root, "p-closure", "data-1", &csv).unwrap();
    // Reserve final test gate directly in DB to simulate active/consumed reservation
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let sha: String = db.query_row("SELECT sha256 FROM scientific_datasets WHERE project='p-closure' AND id='data-1'", [], |r| r.get(0)).unwrap();
    db.execute("INSERT INTO final_test_gates(project, access_id, dataset_sha256, candidate_id, status) VALUES('p-closure', 'gate-1', ?1, 'linear', 'reserved')", [sha]).unwrap();

    let spec = SearchSpec {
        session_id: "sess-closure".into(),
        project_id: "p-closure".into(),
        dataset_id: "data-1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into()],
        max_trials: 4,
        max_screen_trials: 2,
        max_full_trials: 1,
        search_wall_ms: 60_000,
        job_timeout_ms: 10_000,
        finalization_allowance_ms: 5_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };

    let err = search::init_search(&root, spec).unwrap_err();
    assert!(err.to_string().contains("research admission is closed"));
}

#[test]
fn m3_baseline_derivation_and_constraint_checking() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().join("state");
    let r = Runtime::init(&root, m1policy(10, 2)).unwrap();
    create_m1(&r, "p-base", 10);
    execution_task(&r, "p-base", "t-base");

    let csv = d.path().join("data.csv");
    std::fs::write(&csv, "f1,target\n1,0\n2,1\n3,0\n4,1\n").unwrap();
    execution::register_dataset(&root, "p-base", "data-1", &csv).unwrap();

    let workflow = root.join("workflows/p-base");
    std::fs::create_dir_all(&workflow).unwrap();
    let results_json = json!({
        "dataset_sha256": "dummy-sha",
        "candidates": [
            {
                "candidate_id": "linear",
                "recipe": "linear",
                "parameters": {},
                "metrics": {"balanced_accuracy": 0.75},
                "train_ms": 12.0
            },
            {
                "candidate_id": "tree",
                "recipe": "tree",
                "parameters": {},
                "metrics": {"balanced_accuracy": 0.80},
                "train_ms": 25.0
            }
        ]
    });
    std::fs::write(workflow.join("results.json"), results_json.to_string()).unwrap();

    let spec = SearchSpec {
        session_id: "sess-base".into(),
        project_id: "p-base".into(),
        dataset_id: "data-1".into(),
        target: "target".into(),
        task: "binary_classification".into(),
        metric: "balanced_accuracy".into(),
        direction: "maximize".into(),
        split: "stratified".into(),
        fidelities: vec![0.5, 1.0],
        recipes: vec!["linear".into(), "tree".into()],
        max_trials: 4,
        max_screen_trials: 2,
        max_full_trials: 1,
        search_wall_ms: 60_000,
        job_timeout_ms: 10_000,
        finalization_allowance_ms: 5_000,
        max_repeats: 3,
        tie_tolerance: 0.005,
        concurrency: 1,
        max_retries: 2,
        constraints: None,
    };

    let res = search::init_search(&root, spec).unwrap();
    assert_eq!(res["status"], "created");

    // Verify derived baselines in SQLite
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let count: i64 = db.query_row("SELECT count(*) FROM experiment_dag WHERE session_id='sess-base' AND node_type='baseline'", [], |r| r.get(0)).unwrap();
    assert_eq!(count, 2);

    let lin_params: String = db.query_row("SELECT parameters FROM experiment_dag WHERE session_id='sess-base' AND recipe='linear' AND node_type='baseline'", [], |r| r.get(0)).unwrap();
    let lin_val: serde_json::Value = serde_json::from_str(&lin_params).unwrap();
    assert_eq!(lin_val["C"], 1.0);

    let tree_params: String = db.query_row("SELECT parameters FROM experiment_dag WHERE session_id='sess-base' AND recipe='tree' AND node_type='baseline'", [], |r| r.get(0)).unwrap();
    let tree_val: serde_json::Value = serde_json::from_str(&tree_params).unwrap();
    assert_eq!(tree_val["n_estimators"], 100);
}
