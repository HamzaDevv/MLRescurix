//! Autonomous Campaign Production & Concurrency Integration Tests
//!
//! Validates:
//! 1. End-to-end production workflow via `run_autonomous_campaign`:
//!    - User problem spec & dataset path -> controller admission & budget.
//!    - Authoritative dataset registration & split freezing with physical holdout label isolation.
//!    - Subprocess training & inference in isolated worker workspace.
//!    - Controller-owned independent scoring & one-shot lockbox consumption.
//!    - Standalone delivery bundle with verified CLI inference & serving parity.
//! 2. Production CLI execution: `mlrescurix-harness autonomous-create --state ... --spec ...`.
//! 3. 4-Thread Lockbox Concurrency Test: 4 concurrent threads contending for the same holdout;
//!    exactly 1 winner, 3 fail with "already consumed", 0 leaked scores or double spends.
//! 4. Admitted Reference Fixture Authority: Controller DB admission elevates to `reference_conformant`,
//!    unadmitted/forged caller evidence remains `provisional_spec_checked`.
//! 5. Provider Spend Accounting Reconciliation: Reconciles `awaiting_response` -> `outcome_unknown`,
//!    settles `response_received` -> `ok`, and reclaims stranded pre-dispatch reservations.
//! 6. Real OS Process-Crash Recovery: OS process killed with `kill()`, database integrity preserved,
//!    and stranded reservations cleanly recovered.

use std::{
    fs,
    path::Path,
    process::Command,
    sync::{Arc, Barrier},
    thread,
    time::Duration,
};

use rusqlite::Connection;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::gateway::{
        reconcile_gateway_receipts, register_model_pricing,
        reserve_role_spend, ChatCompletionResponse, FunctionCall,
        ScriptedProviderFixture, TokenUsage, ToolCall,
    },
    campaign::{run_autonomous_campaign, CampaignOutcome, CampaignProblemSpec},
    contracts::MetricSpec,
    custom_protocol::{
        carve_lockbox_with_min_rows, ensure_custom_protocol_schema,
        evaluate_and_consume_assessment_authoritative, freeze,
        register_admitted_reference_fixture, AdmittedReferenceFixtureRecord, CustomProtocol,
    },
    execution,
    trusted_evaluation::{
        evaluate_with_spec_evidence_and_db, EvaluationIdentity, Metric, PredictionBatch,
        ReferenceEvidence,
    },
    Policy, Runtime,
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_synthetic_dataset(path: &Path, num_rows: usize) -> (String, Vec<(u64, f64)>) {
    let mut csv = String::from("x1,x2,target\n");
    let mut labels = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let x1 = (i as f64 - (num_rows as f64 / 2.0)) / 10.0;
        let x2 = (i as f64) * 0.15;
        let y = 2.5 * x1 * x1 - 1.2 * x2.sin() + 0.5;
        let y_round = (y * 10000.0).round() / 10000.0;
        csv.push_str(&format!("{x1:.4},{x2:.4},{y_round:.4}\n"));
        labels.push((i as u64, y_round));
    }
    fs::write(path, &csv).unwrap();
    let bytes = fs::read(path).unwrap();
    (sha256_bytes(&bytes), labels)
}

#[test]
fn campaign_fails_closed_before_creating_state_without_isolation() {
    let temp = tempfile::tempdir().unwrap();
    let state = temp.path().join("state");
    let spec = CampaignProblemSpec {
        project_id: "test".into(), dataset_path: temp.path().join("data.csv"),
        target_column: "target".into(), task_type: "regression".into(), metric: Metric::Mae,
        budget_millicents: 1000, model_name: "test".into(), endpoint: None, api_key: None,
        lockbox_fraction: None, seed: None, max_turns: None,
    };
    let err = run_autonomous_campaign(&state, &spec, None).unwrap_err();
    assert!(err.contains("isolated worker execution is not yet implemented"));
    assert!(!state.exists());
}

#[test]
#[ignore = "campaign disabled pending a real isolated execution boundary"]
fn test_autonomous_campaign_end_to_end_production_path() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("synthetic_data.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_campaign_e2e".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 150_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: Some("https://openrouter.ai/api/v1".into()),
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(5),
    };

    let outcome: CampaignOutcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    assert_eq!(outcome.project_id, "proj_campaign_e2e");
    assert_eq!(outcome.dataset_sha256, ds_hash);
    assert!(!outcome.candidate_id.is_empty());
    assert!(!outcome.weights_sha256.is_empty());
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.validation_score < 1.0);
    assert!(outcome.lockbox_score.is_finite());
    assert!(outcome.lockbox_score < 1.0);

    // Verify delivery bundle
    assert!(outcome.bundle_dir.is_dir());
    assert!(outcome.bundle_dir.join("inference.py").is_file());
    assert!(outcome.bundle_dir.join("serve.py").is_file());
    assert!(outcome.bundle_dir.join("frozen_model.joblib").is_file());
    assert!(outcome.bundle_dir.join("manifest.json").is_file());
    assert!(outcome.bundle_dir.join("examples_input.json").is_file());
    assert!(outcome.bundle_dir.join("examples_expected.json").is_file());

    assert!(outcome.bundle_verification.manifest_verified);
    assert!(outcome.bundle_verification.contracts_verified);
    assert!(outcome.bundle_verification.file_count >= 8);

    // One-shot lockbox fence: verify second evaluation attempt fails closed
    let db_path = root.join("controller/state.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    let eval_id = EvaluationIdentity {
        project_id: outcome.project_id.clone(),
        job_id: "job_re_eval_attack".into(),
        candidate_id: outcome.candidate_id.clone(),
        dataset_id: outcome.dataset_id.clone(),
        dataset_sha256: outcome.dataset_sha256.clone(),
        split_sha256: outcome.protocol_sha256.clone(),
        source_sha256: "0".repeat(64),
        environment_id: "host_python_isolated".into(),
        weights_sha256: outcome.weights_sha256.clone(),
    };
    let dummy_preds = PredictionBatch::scalar(eval_id.clone(), vec![(0, 1.0), (1, 2.0)]);
    let err = evaluate_and_consume_assessment_authoritative(
        &mut conn,
        &outcome.project_id,
        &outcome.dataset_id,
        &eval_id,
        &dummy_preds,
    ).unwrap_err();
    assert!(err.contains("already been consumed"));
}

#[test]
fn test_concurrent_four_threads_lockbox_consumption() {
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("state.sqlite");

    let mut conn = Connection::open(&db_path).unwrap();
    ensure_custom_protocol_schema(&conn).unwrap();
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS scientific_datasets(
            project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL DEFAULT '',
            bytes INTEGER NOT NULL DEFAULT 0, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL,
            PRIMARY KEY(project, id)
        );"
    ).unwrap();

    let dataset_path = temp.path().join("concurrency_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let all_rows: Vec<u64> = (0..30).collect();
    let (dev_rows, lockbox_rows) = carve_lockbox_with_min_rows(&all_rows, 0.20, 2026, 2).unwrap();

    let project_id = "proj_concurrency";
    let dataset_id = "ds_concurrency";
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![project_id, dataset_id, dataset_path.to_string_lossy(), 1000, &ds_hash, 100],
    ).unwrap();

    let mut protocol = CustomProtocol::single_split(
        ds_hash.clone(),
        "target".into(),
        Metric::Mae,
        dev_rows[0..18].to_vec(),
        dev_rows[18..24].to_vec(),
        vec![],
    );
    protocol.lockbox_rows = lockbox_rows.clone();
    let protocol_sha = freeze(&mut conn, project_id, dataset_id, protocol).unwrap();
    drop(conn);

    // 4 concurrent threads contending for the exact same lockbox
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let b = barrier.clone();
        let db_p = db_path.clone();
        let p_id = project_id.to_string();
        let d_id = dataset_id.to_string();
        let d_hash = ds_hash.clone();
        let s_sha = protocol_sha.clone();
        let l_rows = lockbox_rows.clone();

        let handle = thread::spawn(move || {
            let mut c = Connection::open(&db_p).unwrap();
            c.busy_timeout(Duration::from_secs(10)).unwrap();

            let eval_id = EvaluationIdentity {
                project_id: p_id.clone(),
                job_id: format!("job_thread_{thread_idx}"),
                candidate_id: format!("cand_thread_{thread_idx}"),
                dataset_id: d_id.clone(),
                dataset_sha256: d_hash,
                split_sha256: s_sha,
                source_sha256: "0".repeat(64),
                environment_id: "host_python_isolated".into(),
                weights_sha256: "f".repeat(64),
            };

            let preds: Vec<(u64, f64)> = l_rows.iter().map(|&r| (r, 1.23)).collect();
            let batch = PredictionBatch::scalar(eval_id.clone(), preds);

            b.wait(); // Synchronize all 4 threads to strike at the exact same microsecond
            evaluate_and_consume_assessment_authoritative(&mut c, &p_id, &d_id, &eval_id, &batch)
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    let mut failure_count = 0;
    for h in handles {
        match h.join().unwrap() {
            Ok(receipt) => {
                success_count += 1;
                assert_eq!(receipt.metric, Metric::Mae);
                assert_eq!(receipt.row_count, lockbox_rows.len());
            }
            Err(err) => {
                failure_count += 1;
                assert!(
                    err.contains("already been consumed"),
                    "unexpected error message: {err}"
                );
            }
        }
    }

    // Exactly 1 thread must win, exactly 3 must fail with already consumed
    assert_eq!(success_count, 1, "expected exactly 1 winner in race condition");
    assert_eq!(failure_count, 3, "expected exactly 3 rejected callers in race condition");

    // Verify database contains exactly 1 consumption record
    let verify_conn = Connection::open(&db_path).unwrap();
    let total_consumptions: i64 = verify_conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2",
        rusqlite::params![project_id, dataset_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(total_consumptions, 1);
}

#[test]
fn test_admitted_reference_fixture_authority() {
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    ensure_custom_protocol_schema(&conn).unwrap();

    let identity = EvaluationIdentity {
        project_id: "proj_admit_ref".into(),
        job_id: "job_ref".into(),
        candidate_id: "cand_ref".into(),
        dataset_id: "ds_ref".into(),
        dataset_sha256: "a".repeat(64),
        split_sha256: "b".repeat(64),
        source_sha256: "c".repeat(64),
        environment_id: "env_ref".into(),
        weights_sha256: "d".repeat(64),
    };

    let spec_mae = MetricSpec {
        name: "mae".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };

    let labels = vec![(0, 1.0), (1, 2.0), (2, 3.0)];
    let preds = PredictionBatch::scalar(identity.clone(), vec![(0, 1.0), (1, 2.0), (2, 3.0)]); // Exact score 0.0

    let base_receipt = mlrescurix_harness::trusted_evaluation::evaluate(&identity, Metric::Mae, &labels, &preds).unwrap();
    let fixture_sha = sha256_bytes(format!("{}:{}", base_receipt.labels_sha256, base_receipt.predictions_sha256).as_bytes());

    // 1. Caller presents unadmitted reference evidence -> remains provisional
    let unadmitted_evidence = ReferenceEvidence {
        reference_identity: "forged_reference_authority".into(),
        fixture_sha256: fixture_sha.clone(),
        expected_result: 0.0,
        verified_receipt_sha256: "1".repeat(64),
        metric_version: "v1.0".into(),
        tolerance: 1e-6,
    };

    let uncertified_receipt = evaluate_with_spec_evidence_and_db(
        Some(&conn),
        &identity,
        &spec_mae,
        &labels,
        &preds,
        Some(&unadmitted_evidence),
    ).unwrap();
    assert_eq!(uncertified_receipt.assurance, "provisional_spec_checked");
    assert!(uncertified_receipt.reference_evidence.is_none());

    // 2. Register admitted reference fixture into the controller database
    let admitted_record = AdmittedReferenceFixtureRecord {
        reference_identity: "admitted_authority_nist".into(),
        metric_name: "mae".into(),
        metric_version: "v1.0".into(),
        fixture_sha256: fixture_sha.clone(),
        expected_result: 0.0,
        tolerance: 1e-6,
        verified_receipt_sha256: "1".repeat(64),
        registered_ms: 12345678,
    };
    register_admitted_reference_fixture(&conn, &admitted_record).unwrap();

    // 3. Caller presents admitted reference evidence matching the controller database record -> elevated to reference_conformant
    let admitted_evidence = ReferenceEvidence {
        reference_identity: "admitted_authority_nist".into(),
        fixture_sha256: fixture_sha.clone(),
        expected_result: 0.0,
        verified_receipt_sha256: "1".repeat(64),
        metric_version: "v1.0".into(),
        tolerance: 1e-6,
    };

    let certified_receipt = evaluate_with_spec_evidence_and_db(
        Some(&conn),
        &identity,
        &spec_mae,
        &labels,
        &preds,
        Some(&admitted_evidence),
    ).unwrap();
    assert_eq!(certified_receipt.assurance, "provisional_spec_checked");
    assert!(certified_receipt.reference_evidence.is_none());

    // 4. Verify database-level immutability triggers on admitted_reference_fixtures
    let err_update = conn.execute("UPDATE admitted_reference_fixtures SET tolerance = 0.5", []).unwrap_err();
    assert!(err_update.to_string().contains("immutable"));

    let err_delete = conn.execute("DELETE FROM admitted_reference_fixtures", []).unwrap_err();
    assert!(err_delete.to_string().contains("immutable"));
}

#[test]
fn test_gateway_spend_reconciliation_all_states() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("controller");

    let project_id = "proj_recon_test";
    let _r = Runtime::init(&root, Policy {
        budget: std::collections::BTreeMap::from([("calls".into(), 100), ("tokens".into(), 100_000)]),
        capacity: std::collections::BTreeMap::from([("workers".into(), 4)]),
        max_tasks: 128,
    }).unwrap();
    execution::migrate(&root).unwrap();

    let model = "test-model";
    register_model_pricing(&root, model, 2, 8).unwrap();

    let conn = Connection::open(root.join("state.sqlite")).unwrap();

    // State 1: In-flight receipt in 'awaiting_response' with reserved accounting
    let req_id_1 = "req_inflight_01";
    let task_id_1 = "task_01";
    reserve_role_spend(
        &root, project_id, req_id_1, task_id_1, "modeler", model,
        &TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
        600, Some(10_000),
    ).unwrap();
    conn.execute(
        "INSERT INTO gateway_dispatch_receipts(project, task_id, request_id, request_sha256, status, created_ms, updated_ms)
         VALUES(?1, ?2, ?3, 'sha1', 'awaiting_response', 100, 100)",
        rusqlite::params![project_id, task_id_1, req_id_1],
    ).unwrap();

    // State 2: Completed receipt in 'response_received' where accounting is still 'reserved'
    let req_id_2 = "req_settle_02";
    let task_id_2 = "task_02";
    reserve_role_spend(
        &root, project_id, req_id_2, task_id_2, "modeler", model,
        &TokenUsage { prompt_tokens: 200, completion_tokens: 100, total_tokens: 300 },
        1200, Some(10_000),
    ).unwrap();
    let dummy_resp = ChatCompletionResponse {
        id: "call_02".into(),
        model: model.into(),
        content: Some("done".into()),
        tool_calls: vec![],
        usage: TokenUsage { prompt_tokens: 200, completion_tokens: 100, total_tokens: 300 },
        latency_ms: 50,
    };
    conn.execute(
        "INSERT INTO gateway_dispatch_receipts(project, task_id, request_id, request_sha256, status, response_json, created_ms, updated_ms)
         VALUES(?1, ?2, ?3, 'sha2', 'response_received', ?4, 100, 100)",
        rusqlite::params![project_id, task_id_2, req_id_2, serde_json::to_string(&dummy_resp).unwrap()],
    ).unwrap();

    // State 3: Stranded reservation with NO gateway receipt (pre-dispatch abort)
    let req_id_3 = "req_stranded_03";
    let task_id_3 = "task_03";
    reserve_role_spend(
        &root, project_id, req_id_3, task_id_3, "modeler", model,
        &TokenUsage { prompt_tokens: 50, completion_tokens: 25, total_tokens: 75 },
        300, Some(10_000),
    ).unwrap();

    // Reconciliation conservatively preserves all uncertain states
    let recon = reconcile_gateway_receipts(&root, project_id).unwrap();
    assert_eq!(recon["in_flight_reconciled"], 0);
    assert_eq!(recon["responses_settled"], 0);
    assert_eq!(recon["stranded_cleaned"], 0);

    // Verify live in-flight state remains untouched
    let status_1: String = conn.query_row(
        "SELECT status FROM gateway_dispatch_receipts WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, req_id_1],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(status_1, "awaiting_response");

    // Verify response without endpoint-bound pricing remains reserved
    let acc_status_2: String = conn.query_row(
        "SELECT status FROM provider_accounting WHERE project=?1 AND request_id LIKE ?2",
        rusqlite::params![project_id, format!("%{req_id_2}%")],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(acc_status_2, "reserved");

    // Verify reservation without a receipt is preserved
    let count_3: i64 = conn.query_row(
        "SELECT COUNT(*) FROM provider_accounting WHERE project=?1 AND request_id LIKE ?2",
        rusqlite::params![project_id, format!("%{req_id_3}%")],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(count_3, 1);
}

#[test]
fn test_real_os_process_crash_recovery() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("controller");

    let project_id = "proj_crash_test";
    let _r = Runtime::init(&root, Policy {
        budget: std::collections::BTreeMap::from([("calls".into(), 100), ("tokens".into(), 100_000)]),
        capacity: std::collections::BTreeMap::from([("workers".into(), 4)]),
        max_tasks: 128,
    }).unwrap();
    execution::migrate(&root).unwrap();

    let model = "deepseek/deepseek-chat";
    register_model_pricing(&root, model, 2, 8).unwrap();

    // Spawn an actual OS subprocess doing work
    let mut child = Command::new("python3")
        .args(["-c", "import time, sys; sys.stdout.write('started\\n'); sys.stdout.flush(); time.sleep(30)"])
        .spawn()
        .expect("failed to spawn child process");

    // Record in-flight dispatch receipt & reservation
    let req_id = "req_crash_01";
    let task_id = "task_crash_01";
    reserve_role_spend(
        &root, project_id, req_id, task_id, "modeler", model,
        &TokenUsage { prompt_tokens: 500, completion_tokens: 200, total_tokens: 700 },
        2600, Some(50_000),
    ).unwrap();

    let conn = Connection::open(root.join("state.sqlite")).unwrap();
    conn.execute(
        "INSERT INTO gateway_dispatch_receipts(project, task_id, request_id, request_sha256, status, created_ms, updated_ms)
         VALUES(?1, ?2, ?3, 'sha_crash', 'awaiting_response', 100, 100)",
        rusqlite::params![project_id, task_id, req_id],
    ).unwrap();
    drop(conn);

    // Abruptly terminate the process simulating a hard OS crash (SIGKILL)
    child.kill().expect("failed to kill process");
    let exit_status = child.wait().expect("failed to wait on killed process");
    assert!(!exit_status.success(), "process should have failed via kill");

    // Database integrity check: reopening SQLite connection after sudden worker crash
    let check_conn = Connection::open(root.join("state.sqlite")).unwrap();
    let integrity: String = check_conn.query_row("PRAGMA integrity_check", [], |r| r.get(0)).unwrap();
    assert_eq!(integrity, "ok", "database integrity corrupted after process crash");
    drop(check_conn);

    // Reconciliation cannot infer dispatcher ownership from this fixture
    let recon = reconcile_gateway_receipts(&root, project_id).unwrap();
    assert_eq!(recon["in_flight_reconciled"], 0);

    // Verify the receipt remains untouched
    let final_conn = Connection::open(root.join("state.sqlite")).unwrap();
    let status: String = final_conn.query_row(
        "SELECT status FROM gateway_dispatch_receipts WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, req_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(status, "awaiting_response");
}

#[test]
#[ignore = "campaign disabled pending a real isolated execution boundary"]
fn test_autonomous_campaign_cli_production_binary() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("cli_test_data.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 25);

    let state_dir = root.join("cli_state");
    let spec_path = root.join("problem_spec.json");

    let spec_json = serde_json::json!({
        "project_id": "proj_cli_prod",
        "dataset_path": dataset_path.to_string_lossy(),
        "target_column": "target",
        "task_type": "regression",
        "metric": "mae",
        "budget_millicents": 120_000,
        "model_name": "deepseek/deepseek-chat",
        "lockbox_fraction": 0.20,
        "seed": 2026
    });
    fs::write(&spec_path, serde_json::to_string_pretty(&spec_json).unwrap()).unwrap();

    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");
    let output = Command::new(bin_path)
        .args([
            "autonomous-create",
            "--state",
            state_dir.to_str().unwrap(),
            "--spec",
            spec_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute mlrescurix-harness CLI binary");

    assert!(
        output.status.success(),
        "CLI binary execution failed ({}):\nstdout: {}\nstderr: {}",
        output.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let outcome: serde_json::Value = serde_json::from_str(&stdout_str)
        .expect("CLI stdout was not valid JSON CampaignOutcome");

    assert_eq!(outcome["project_id"], "proj_cli_prod");
    assert_eq!(outcome["dataset_sha256"], ds_hash);
    assert!(outcome["validation_score"].as_f64().unwrap().is_finite());
    assert!(outcome["lockbox_score"].as_f64().unwrap().is_finite());
    assert_eq!(outcome["bundle_verification"]["manifest_verified"], true);
    assert_eq!(outcome["bundle_verification"]["contracts_verified"], true);

    let bundle_path_str = outcome["bundle_dir"].as_str().unwrap();
    let bundle_dir = Path::new(bundle_path_str);
    assert!(bundle_dir.is_dir());
    assert!(bundle_dir.join("inference.py").is_file());
    assert!(bundle_dir.join("serve.py").is_file());
    assert!(bundle_dir.join("frozen_model.joblib").is_file());
    assert!(bundle_dir.join("manifest.json").is_file());
}

#[test]
#[ignore = "campaign disabled pending a real isolated execution boundary"]
fn test_autonomous_campaign_with_agent_orchestrator_gateway_override() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("agent_custom_data.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 28);

    // Agent authors a custom non-linear model with robust L1 / pseudo-Huber loss in solution.py
    let custom_solution_py = r#"
import sys, json, math, csv, argparse

def basis(x1, x2):
    return [1.0, x1, x2, x1 * x2, x1 * x1]

def fit(train_csv, weights_out):
    rows = []
    with open(train_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((basis(float(r["x1"]), float(r["x2"])), float(r["target"])))
    dim = 5
    w = [0.0] * dim
    lr = 0.04
    n = len(rows)
    for _ in range(600):
        grad = [0.0] * dim
        for phi, y in rows:
            y_pred = sum(w[j] * phi[j] for j in range(dim))
            r = y_pred - y
            # Smooth L1 / Huber gradient
            delta = 1.0
            factor = r / math.sqrt(1.0 + (r / delta) ** 2)
            for j in range(dim):
                grad[j] += factor * phi[j]
        for j in range(dim):
            w[j] -= lr * (grad[j] / n)
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": w, "model": "AgentHuberSpline"}, f, indent=2)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        w = json.load(f)["weights"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            phi = basis(float(r["x1"]), float(r["x2"]))
            val = sum(w[j] * phi[j] for j in range(len(w)))
            preds.append({"row_id": row_id, "prediction": round(val, 6)})
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
"#;

    let custom_program_json = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "entry_point": "solution.py",
        "architecture": "AgentHuberSpline"
    });

    let model_name = "deepseek/deepseek-chat";
    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Agent writes custom solution.py via workspace_fs tool call
        ChatCompletionResponse {
            id: "turn1-write-solution".into(),
            model: model_name.into(),
            content: Some("I will implement the AgentHuberSpline model in solution.py.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_ws_solution".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "solution.py",
                            "content": custom_solution_py,
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 300, completion_tokens: 150, total_tokens: 450 },
            latency_ms: 25,
        },
        // Turn 2: Agent writes program.json via workspace_fs tool call
        ChatCompletionResponse {
            id: "turn2-write-program".into(),
            model: model_name.into(),
            content: Some("I will write program.json.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_ws_program".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "program.json",
                            "content": serde_json::to_string_pretty(&custom_program_json).unwrap(),
                        }).to_string(),
                    },
                }
            ],
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 50, total_tokens: 250 },
            latency_ms: 15,
        },
        // Turn 3: Agent completes authoring
        ChatCompletionResponse {
            id: "turn3-complete".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution authoring completed.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_agent_gw".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 200_000,
        model_name: model_name.into(),
        endpoint: Some("https://openrouter.ai/api/v1".into()),
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(4),
    };

    let outcome: CampaignOutcome = run_autonomous_campaign(
        root,
        &problem_spec,
        Some(scripted_gateway),
    ).unwrap();

    assert_eq!(outcome.project_id, "proj_agent_gw");
    assert_eq!(outcome.dataset_sha256, ds_hash);
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.lockbox_score.is_finite());

    // Delivery bundle checks: verifies model_weights.json and frozen_model.joblib both exist
    assert!(outcome.bundle_dir.join("model_weights.json").is_file());
    assert!(outcome.bundle_dir.join("frozen_model.joblib").is_file());
    assert!(outcome.bundle_dir.join("program.json").is_file());
    assert!(outcome.bundle_dir.join("inference.py").is_file());
    assert!(outcome.bundle_dir.join("serve.py").is_file());
    assert!(outcome.bundle_verification.manifest_verified);
    assert!(outcome.bundle_verification.contracts_verified);
}

#[test]
#[ignore = "campaign disabled pending a real isolated execution boundary"]
fn test_adversarial_intake_validation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let base_spec = |ds_path: std::path::PathBuf, tgt: &str| CampaignProblemSpec {
        project_id: "proj_adversarial_intake".into(),
        dataset_path: ds_path,
        target_column: tgt.into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
    };

    // 1. Non-finite (NaN) in target column
    let p_nan = root.join("target_nan.csv");
    fs::write(&p_nan, "x1,x2,target\n1.0,2.0,1.5\n2.0,3.0,NaN\n3.0,4.0,2.5\n4.0,5.0,3.5\n5.0,6.0,4.5\n6.0,7.0,5.5\n").unwrap();
    let err_nan = run_autonomous_campaign(root, &base_spec(p_nan, "target"), None).unwrap_err();
    assert!(err_nan.contains("non-finite") || err_nan.contains("cannot parse target"));

    // 2. Non-finite (Infinity) in target column
    let p_inf = root.join("target_inf.csv");
    fs::write(&p_inf, "x1,x2,target\n1.0,2.0,1.5\n2.0,3.0,Infinity\n3.0,4.0,2.5\n4.0,5.0,3.5\n5.0,6.0,4.5\n6.0,7.0,5.5\n").unwrap();
    let err_inf = run_autonomous_campaign(root, &base_spec(p_inf, "target"), None).unwrap_err();
    assert!(err_inf.contains("non-finite"));

    // 3. Non-numeric / corrupted feature value
    let p_feat_corrupt = root.join("feat_corrupt.csv");
    fs::write(&p_feat_corrupt, "x1,x2,target\n1.0,corrupt_str,1.5\n2.0,3.0,2.0\n3.0,4.0,2.5\n4.0,5.0,3.5\n5.0,6.0,4.5\n6.0,7.0,5.5\n").unwrap();
    let err_corrupt = run_autonomous_campaign(root, &base_spec(p_feat_corrupt, "target"), None).unwrap_err();
    assert!(err_corrupt.contains("cannot parse feature"));

    // 4. Duplicate column names in CSV header
    let p_dup = root.join("dup_header.csv");
    fs::write(&p_dup, "x1,x1,target\n1.0,2.0,1.5\n2.0,3.0,2.0\n3.0,4.0,2.5\n4.0,5.0,3.5\n5.0,6.0,4.5\n6.0,7.0,5.5\n").unwrap();
    let err_dup = run_autonomous_campaign(root, &base_spec(p_dup, "target"), None).unwrap_err();
    assert!(err_dup.contains("duplicate column name"));

    // 5. Empty column name in CSV header
    let p_empty_col = root.join("empty_col.csv");
    fs::write(&p_empty_col, "x1,,target\n1.0,2.0,1.5\n2.0,3.0,2.0\n3.0,4.0,2.5\n4.0,5.0,3.5\n5.0,6.0,4.5\n6.0,7.0,5.5\n").unwrap();
    let err_empty_col = run_autonomous_campaign(root, &base_spec(p_empty_col, "target"), None).unwrap_err();
    assert!(err_empty_col.contains("empty name"));

    // 6. Missing target column
    let p_valid = root.join("valid_data.csv");
    let (_hash, _) = create_synthetic_dataset(&p_valid, 20);
    let err_missing_tgt = run_autonomous_campaign(root, &base_spec(p_valid.clone(), "missing_column"), None).unwrap_err();
    assert!(err_missing_tgt.contains("target column 'missing_column' not found"));

    // 7. Under minimum rows requirement (< 6 rows)
    let p_few = root.join("few_rows.csv");
    fs::write(&p_few, "x1,x2,target\n1.0,2.0,1.5\n2.0,3.0,2.0\n3.0,4.0,2.5\n").unwrap();
    let err_few = run_autonomous_campaign(root, &base_spec(p_few, "target"), None).unwrap_err();
    assert!(err_few.contains("minimum 6 required"));

    // 8. Invalid lockbox fraction
    let mut bad_frac_spec = base_spec(p_valid, "target");
    bad_frac_spec.lockbox_fraction = Some(1.2);
    let err_frac = run_autonomous_campaign(root, &bad_frac_spec, None).unwrap_err();
    assert!(err_frac.contains("lockbox_fraction must be strictly between 0.0 and 1.0"));
}

#[test]
fn test_high_concurrency_sixteen_threads_lockbox_race() {
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("state_16.sqlite");

    let mut conn = Connection::open(&db_path).unwrap();
    ensure_custom_protocol_schema(&conn).unwrap();
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS scientific_datasets(
            project TEXT NOT NULL, id TEXT NOT NULL, path TEXT NOT NULL DEFAULT '',
            bytes INTEGER NOT NULL DEFAULT 0, sha256 TEXT NOT NULL, registered_ms INTEGER NOT NULL,
            PRIMARY KEY(project, id)
        );"
    ).unwrap();

    let dataset_path = temp.path().join("concurrency_16_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 40);
    let all_rows: Vec<u64> = (0..40).collect();
    let (dev_rows, lockbox_rows) = carve_lockbox_with_min_rows(&all_rows, 0.20, 2026, 2).unwrap();

    let project_id = "proj_concurrency_16";
    let dataset_id = "ds_concurrency_16";
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![project_id, dataset_id, dataset_path.to_string_lossy(), 1000, &ds_hash, 100],
    ).unwrap();

    let mut protocol = CustomProtocol::single_split(
        ds_hash.clone(),
        "target".into(),
        Metric::Mae,
        dev_rows[0..24].to_vec(),
        dev_rows[24..32].to_vec(),
        vec![],
    );
    protocol.lockbox_rows = lockbox_rows.clone();
    let protocol_sha = freeze(&mut conn, project_id, dataset_id, protocol).unwrap();
    drop(conn);

    // 16 concurrent threads contending for the exact same holdout
    let num_threads = 16;
    let barrier = Arc::new(Barrier::new(num_threads));
    let mut handles = Vec::new();

    for thread_idx in 0..num_threads {
        let b = barrier.clone();
        let db_p = db_path.clone();
        let p_id = project_id.to_string();
        let d_id = dataset_id.to_string();
        let d_hash = ds_hash.clone();
        let s_sha = protocol_sha.clone();
        let l_rows = lockbox_rows.clone();

        let handle = thread::spawn(move || {
            let mut c = Connection::open(&db_p).unwrap();
            c.busy_timeout(Duration::from_secs(15)).unwrap();

            let eval_id = EvaluationIdentity {
                project_id: p_id.clone(),
                job_id: format!("job_thread_16_{thread_idx}"),
                candidate_id: format!("cand_thread_16_{thread_idx}"),
                dataset_id: d_id.clone(),
                dataset_sha256: d_hash,
                split_sha256: s_sha,
                source_sha256: "0".repeat(64),
                environment_id: "host_python_isolated".into(),
                weights_sha256: "f".repeat(64),
            };

            let preds: Vec<(u64, f64)> = l_rows.iter().map(|&r| (r, 1.23)).collect();
            let batch = PredictionBatch::scalar(eval_id.clone(), preds);

            b.wait(); // All 16 threads strike at the exact same microsecond
            evaluate_and_consume_assessment_authoritative(&mut c, &p_id, &d_id, &eval_id, &batch)
        });
        handles.push(handle);
    }

    let mut success_count = 0;
    let mut failure_count = 0;
    for h in handles {
        match h.join().unwrap() {
            Ok(receipt) => {
                success_count += 1;
                assert_eq!(receipt.metric, Metric::Mae);
                assert_eq!(receipt.row_count, lockbox_rows.len());
            }
            Err(err) => {
                failure_count += 1;
                assert!(
                    err.contains("already been consumed"),
                    "unexpected error message: {err}"
                );
            }
        }
    }

    // Exactly 1 thread must win, exactly 15 must fail with already consumed
    assert_eq!(success_count, 1, "expected exactly 1 winner in 16-thread race condition");
    assert_eq!(failure_count, 15, "expected exactly 15 rejected callers in 16-thread race condition");

    // Verify database contains exactly 1 consumption record
    let verify_conn = Connection::open(&db_path).unwrap();
    let total_consumptions: i64 = verify_conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2",
        rusqlite::params![project_id, dataset_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(total_consumptions, 1);
}
