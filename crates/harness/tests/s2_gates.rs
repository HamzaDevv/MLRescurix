//! S2 Verification Gates A, B, D, E Integration Tests
//!
//! Validates:
//! Gate A: Real owner ReAct trajectories (Trajectory 1: Boss approves -> Challenger -> Finalize;
//!         Trajectory 2: Boss observes poor validation -> Early Stop -> No lockbox consumed);
//!         Structured challenger audit result bound to candidate, protocol, and evaluation.
//! Gate B: Concurrent owner admission conflict; SIGKILL recovery past 30s TTL;
//!         Stale completion write rejected by epoch mismatch; resumption without duplicate lockbox consumption.
//! Gate D: Worker label isolation (val_features.csv has no target column);
//!         Honest backend availability (campaign_isolation_available remains false).
//! Gate E: Resumption rejection on changed metric (MAE vs RMSE);
//!         Repackaging rejection on tampered model weights.

use std::{
    fs,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::Connection;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::gateway::{
        ChatCompletionResponse, ScriptedProviderFixture, TokenUsage,
    },
    campaign::{
        acquire_campaign_lease, campaign_isolation_available, ensure_campaign_leases_schema,
        parse_structured_challenger_result, release_campaign_lease,
        run_offline_fixture_campaign as run_autonomous_campaign, verify_active_lease,
        CampaignProblemSpec, LEASE_TTL_MS,
    },
    trusted_evaluation::Metric,
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_synthetic_dataset(path: &Path, num_rows: usize) -> String {
    let mut csv = String::from("x1,x2,target\n");
    for i in 0..num_rows {
        let x1 = (i as f64 - (num_rows as f64 / 2.0)) / 10.0;
        let x2 = (i as f64) * 0.15;
        let y = 2.5 * x1 * x1 - 1.2 * x2.sin() + 0.5;
        let y_round = (y * 10000.0).round() / 10000.0;
        csv.push_str(&format!("{x1:.4},{x2:.4},{y_round:.4}\n"));
    }
    fs::write(path, &csv).unwrap();
    let bytes = fs::read(path).unwrap();
    sha256_bytes(&bytes)
}

#[test]
fn test_gate_a_trajectory_1_boss_delegates_challenger_and_finalizes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("traj1_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 30);
    let model_name = "deepseek/deepseek-chat";

    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist finalizes authoring
        ChatCompletionResponse {
            id: "turn1-spec".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution ready.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
        // Turn 2: Boss ReAct Turn 1 observes evidence and delegates to challenger
        ChatCompletionResponse {
            id: "boss_turn1_delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Validation score is competitive; audit for variance and protocol leak.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 25, total_tokens: 175 },
            latency_ms: 12,
        },
        // Turn 3: Challenger performs audit and outputs structured approval
        ChatCompletionResponse {
            id: "challenger_audit_step".into(),
            model: model_name.into(),
            content: Some(serde_json::to_string(&serde_json::json!({
                "verdict": "APPROVED",
                "rationale": "Validation split boundaries and metric strictly conform to protocol.",
                "findings": ["No target leakage detected", "Residuals well-bounded"]
            })).unwrap()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 250, completion_tokens: 45, total_tokens: 295 },
            latency_ms: 20,
        },
        // Turn 4: Boss ReAct Turn 2 observes challenger audit and approves finalization
        ChatCompletionResponse {
            id: "boss_turn2_finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 20, total_tokens: 200 },
            latency_ms: 10,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_a_traj1".into(),
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

    let outcome = run_autonomous_campaign(root, &problem_spec, Some(scripted_gateway.clone())).unwrap();
    assert_eq!(outcome.project_id, "proj_gate_a_traj1");
    assert!(outcome.lockbox_score > 0.0);

    // Verify lockbox was consumed exactly once
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_a_traj1'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 1);

    // Verify structured challenger audit stored in task completions
    let challenger_task_id = format!("task_challenger_{}", problem_spec.project_id);
    let delta_json: String = conn.query_row(
        "SELECT metric_delta FROM task_completions WHERE project='proj_gate_a_traj1' AND task_id=?1",
        rusqlite::params![&challenger_task_id],
        |r| r.get(0),
    ).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&delta_json).unwrap();
    assert_eq!(parsed["verdict"], "APPROVED");
    assert_eq!(parsed["schema_version"], "mlrescurix.challenger.v1");
}

#[test]
fn test_gate_a_trajectory_2_boss_observes_poor_score_and_stops_early() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("traj2_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 30);
    let model_name = "deepseek/deepseek-chat";

    let scripted_gateway = Arc::new(ScriptedProviderFixture::new(vec![
        // Turn 1: Specialist finalizes
        ChatCompletionResponse {
            id: "turn1-spec".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution ready.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
        // Turn 2: Boss observes validation evidence and decides to STOP EARLY
        ChatCompletionResponse {
            id: "boss_turn1_early_stop".into(),
            model: model_name.into(),
            content: Some("STOP_EARLY: Validation error exceeds acceptable scientific tolerance; halting without candidate audit.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 30, total_tokens: 180 },
            latency_ms: 12,
        },
    ]));

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_gate_a_traj2".into(),
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

    let err = run_autonomous_campaign(root, &problem_spec, Some(scripted_gateway)).unwrap_err();
    assert!(err.contains("campaign stopped early by owner"), "expected early stop error, got: {err}");

    // Verify lockbox was NEVER consumed
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project='proj_gate_a_traj2'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 0, "lockbox must NEVER be consumed on early stop");
}

#[test]
fn test_gate_a_structured_challenger_audit_result_parser() {
    let task_id = "task_challenger_test";
    let candidate_id = "cand_test_001";
    let proto_sha = "abcdef123456";
    let score = 0.042;

    // Structured JSON approval
    let json_approved = serde_json::to_string(&serde_json::json!({
        "verdict": "APPROVED",
        "rationale": "All protocols respected",
        "findings": ["No leakage", "Valid split"]
    })).unwrap();
    let res = parse_structured_challenger_result(&json_approved, task_id, candidate_id, proto_sha, score).unwrap();
    assert_eq!(res.verdict, "APPROVED");
    assert_eq!(res.findings.len(), 2);

    // Structured JSON challenge
    let json_challenged = serde_json::to_string(&serde_json::json!({
        "verdict": "CHALLENGED",
        "rationale": "High variance detected",
        "findings": ["Variance outside 3-sigma"]
    })).unwrap();
    let res = parse_structured_challenger_result(&json_challenged, task_id, candidate_id, proto_sha, score).unwrap();
    assert_eq!(res.verdict, "CHALLENGED");

    // Prose format fallback
    let prose_approved = "APPROVED: Independent audit passed cleanly.";
    let res = parse_structured_challenger_result(prose_approved, task_id, candidate_id, proto_sha, score).unwrap();
    assert_eq!(res.verdict, "APPROVED");

    // Ambiguous and missing verdicts must fail closed
    assert!(parse_structured_challenger_result("NOT APPROVED", task_id, candidate_id, proto_sha, score).is_err());
    assert!(parse_structured_challenger_result("Maybe it looks okay?", task_id, candidate_id, proto_sha, score).is_err());
    assert!(parse_structured_challenger_result("", task_id, candidate_id, proto_sha, score).is_err());
}

#[test]
fn test_gate_b_concurrent_owner_admission_race_and_epoch_fencing() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let db_path = root.join("test_leases.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    ensure_campaign_leases_schema(&conn).unwrap();

    let project = "proj_race_test";
    let owner1 = "controller_proc_1";
    let owner2 = "controller_proc_2";

    // 1. Owner 1 acquires lease
    let (lease1, epoch1) = acquire_campaign_lease(&mut conn, project, owner1).unwrap();
    assert_eq!(epoch1, 1);

    // 2. Owner 2 tries to acquire lease concurrently -> fails closed with conflict
    let err = acquire_campaign_lease(&mut conn, project, owner2).unwrap_err();
    assert!(err.contains("campaign lease conflict"), "expected conflict error, got: {err}");

    // 3. Stale completion write: Owner with wrong epoch is rejected
    let stale_err = verify_active_lease(&conn, project, &lease1, epoch1 + 99).unwrap_err();
    assert!(stale_err.contains("lease epoch mismatch"), "expected epoch mismatch, got: {stale_err}");

    // 4. Owner 1 releases lease
    release_campaign_lease(&conn, project, &lease1, epoch1).unwrap();

    // 5. Owner 2 can now acquire lease with incremented epoch
    let (lease2, epoch2) = acquire_campaign_lease(&mut conn, project, owner2).unwrap();
    assert_ne!(lease1, lease2);
    assert_eq!(epoch2, 2);
}

#[test]
fn test_gate_b_sigkill_recovery_past_lease_ttl() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let db_path = root.join("test_sigkill.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    ensure_campaign_leases_schema(&conn).unwrap();

    let project = "proj_sigkill_test";
    let dead_owner = "dead_process_pid_99999";
    let surviving_owner = "surviving_controller";

    // Insert an active lease with heartbeat older than LEASE_TTL_MS (e.g. 45s ago)
    let past_time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64 - (LEASE_TTL_MS + 15_000);

    conn.execute(
        "INSERT INTO campaign_leases(project, lease_id, owner_id, epoch, status, acquired_ms, heartbeat_ms)
         VALUES(?1, 'dead_lease_abc', ?2, 1, 'active', ?3, ?3)",
        rusqlite::params![project, dead_owner, past_time],
    ).unwrap();

    // Surviving controller attempts to acquire lease -> recovers dead lease with incremented epoch
    let (recovered_lease, epoch) = acquire_campaign_lease(&mut conn, project, surviving_owner).unwrap();
    assert_eq!(epoch, 2);
    assert_ne!(recovered_lease, "dead_lease_abc");

    // Verify state in DB
    let status: String = conn.query_row(
        "SELECT status FROM campaign_leases WHERE project=?1",
        rusqlite::params![project],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(status, "active");
}

#[test]
fn test_gate_d_physical_label_isolation_and_honest_reporting() {
    // 1. Honest isolation capability: Must remain false on host
    assert!(!campaign_isolation_available(), "campaign_isolation_available must be false without verified container runtime");

    // 2. Physical label isolation: Worker val_features.csv has NO target column
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("iso_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_iso_check".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "test/model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: None,
    };

    let outcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();
    let _candidate_id = &outcome.candidate_id;

    // Check worker val_features.csv
    let worker_val_csv = root.join("worker_workspace/val_features.csv");
    assert!(worker_val_csv.is_file(), "worker val_features.csv must exist");
    let val_csv_content = fs::read_to_string(&worker_val_csv).unwrap();
    let first_line = val_csv_content.lines().next().unwrap();
    assert!(!first_line.split(',').any(|c| c.trim() == "target"), "target column must NEVER be present in worker val_features.csv");
    assert!(first_line.split(',').any(|c| c.trim() == "x1"));
    assert!(first_line.split(',').any(|c| c.trim() == "x2"));

    // Check worker train.csv DOES have target column
    let worker_train_csv = root.join("worker_workspace/train.csv");
    assert!(worker_train_csv.is_file());
    let train_csv_content = fs::read_to_string(&worker_train_csv).unwrap();
    let train_first_line = train_csv_content.lines().next().unwrap();
    assert!(train_first_line.split(',').any(|c| c.trim() == "target"), "target column must be in worker train.csv");
}

#[test]
fn test_gate_e_resumption_rejects_changed_metric() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("metric_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);

    let problem_spec_mae = CampaignProblemSpec {
        project_id: "proj_metric_resumption".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "test/model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: None,
    };

    // First run with MAE succeeds
    let _outcome1 = run_autonomous_campaign(root, &problem_spec_mae, None).unwrap();

    // Second run with RMSE on same project must fail closed
    let mut problem_spec_rmse = problem_spec_mae.clone();
    problem_spec_rmse.metric = Metric::Rmse;

    let err = run_autonomous_campaign(root, &problem_spec_rmse, None).unwrap_err();
    assert!(
        err.contains("campaign resumption rejected: requested metric 'Rmse' does not match frozen protocol metric 'Mae'"),
        "expected metric resumption rejection, got: {err}"
    );
}

#[test]
fn test_gate_e_repackaging_rejects_tampered_model_weights() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("tamper_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_tamper_test".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "test/model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: None,
    };

    let outcome = run_autonomous_campaign(root, &problem_spec, None).unwrap();
    let candidate_id = &outcome.candidate_id;

    // Vector 1: Tamper with model_weights.json in bundle
    let bundle_weights = root.join(format!("delivery_bundle_{}/model_weights.json", candidate_id));
    fs::write(&bundle_weights, b"{\"weights\": [999.999], \"tampered\": true}").unwrap();

    let err = run_autonomous_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        err.contains("does not match authoritative lockbox evaluation artifact SHA-256") || err.contains("tampered bundle"),
        "expected weights tampering rejection, got: {err}"
    );

    // Vector 2: Delete manifest.json and tamper with worker_workspace/weights.json
    let bundle_manifest = root.join(format!("delivery_bundle_{}/manifest.json", candidate_id));
    let _ = fs::remove_file(bundle_manifest);
    let worker_weights = root.join("worker_workspace/weights.json");
    fs::write(&worker_weights, b"{\"weights\": [888.888], \"tampered\": true}").unwrap();

    let err2 = run_autonomous_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        err2.contains("cannot repackage modified weights with old scores") || err2.contains("does not match evaluated artifact SHA-256"),
        "expected lockbox consumption mismatch rejection, got: {err2}"
    );
}

#[test]
fn test_gate_e_resumption_rejects_changed_seed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("seed_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);

    let problem_spec_2026 = CampaignProblemSpec {
        project_id: "proj_seed_resumption".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "test/model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: None,
    };

    // First run with seed 2026 succeeds
    let _outcome1 = run_autonomous_campaign(root, &problem_spec_2026, None).unwrap();

    // Second run with seed 7 on same project must fail closed
    let mut problem_spec_7 = problem_spec_2026.clone();
    problem_spec_7.seed = Some(7);

    let err = run_autonomous_campaign(root, &problem_spec_7, None).unwrap_err();
    assert!(
        err.contains("campaign resumption rejected: requested seed"),
        "expected seed resumption rejection, got: {err}"
    );
}

#[test]
fn test_gate_e_resumption_rejects_changed_dataset_bytes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let dataset_path = root.join("dataset_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_dataset_resumption".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "test/model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: None,
    };

    // First run succeeds
    let _outcome1 = run_autonomous_campaign(root, &problem_spec, None).unwrap();

    // Now overwrite dataset file with different content (35 rows instead of 25)
    let _new_hash = create_synthetic_dataset(&dataset_path, 35);

    let err = run_autonomous_campaign(root, &problem_spec, None).unwrap_err();
    assert!(
        err.contains("dataset bytes changed") || err.contains("does not match frozen protocol dataset SHA-256"),
        "expected dataset change rejection, got: {err}"
    );
}

#[test]
fn test_gate_b_immediate_killed_process_recovery() {
    let temp = tempfile::tempdir().unwrap();
    let db_path = temp.path().join("state.sqlite");
    let mut conn = Connection::open(&db_path).unwrap();
    ensure_campaign_leases_schema(&conn).unwrap();

    let project = "proj_immediate_kill_recovery";
    // Spawn a short-lived process that exits immediately
    let mut child = std::process::Command::new("true")
        .spawn()
        .expect("failed to spawn child process");
    let dead_pid = child.id();
    let _ = child.wait(); // child is now terminated and dead

    let dead_owner = format!("pid_{}_{}", dead_pid, project);

    // Insert active lease for the dead process with current time (0 seconds elapsed, well under 30s TTL!)
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as i64;
    conn.execute(
        "INSERT INTO campaign_leases (project, lease_id, owner_id, epoch, acquired_ms, heartbeat_ms, status)
         VALUES (?1, ?2, ?3, 1, ?4, ?4, 'active')",
        rusqlite::params![project, "dead_lease_xyz", &dead_owner, now],
    ).unwrap();

    // Surviving controller acquires lease: should detect dead process and recover IMMEDIATELY (< 500ms) with epoch 2
    let start = std::time::Instant::now();
    let surviving_owner = format!("pid_{}_{}", std::process::id(), project);
    let (recovered_lease, epoch) = acquire_campaign_lease(&mut conn, project, &surviving_owner).unwrap();
    let elapsed = start.elapsed();

    assert_eq!(epoch, 2);
    assert_ne!(recovered_lease, "dead_lease_xyz");
    assert!(elapsed < std::time::Duration::from_millis(500), "recovery must be immediate without waiting 30s TTL!");
}

#[test]
fn test_gate_a_cli_binary_with_mock_fixture() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("cli_data.csv");
    let _ds_hash = create_synthetic_dataset(&dataset_path, 25);
    let model_name = "deepseek/deepseek-chat";

    let spec = serde_json::json!({
        "project_id": "proj_cli_mock_test",
        "dataset_path": dataset_path.to_str().unwrap(),
        "target_column": "target",
        "task_type": "regression",
        "metric": "mae",
        "budget_millicents": 200000,
        "model_name": model_name,
        "lockbox_fraction": 0.20,
        "seed": 2026,
        "max_turns": 4
    });
    let spec_path = root.join("spec.json");
    fs::write(&spec_path, serde_json::to_string_pretty(&spec).unwrap()).unwrap();

    let responses = vec![
        ChatCompletionResponse {
            id: "turn1-spec".into(),
            model: model_name.into(),
            content: Some("[FINAL] Solution authored.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 20, total_tokens: 120 },
            latency_ms: 10,
        },
        ChatCompletionResponse {
            id: "turn2-boss-delegate".into(),
            model: model_name.into(),
            content: Some("DELEGATE_CHALLENGER: Validation score acceptable; audit required.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 25, total_tokens: 175 },
            latency_ms: 12,
        },
        ChatCompletionResponse {
            id: "turn3-challenger".into(),
            model: model_name.into(),
            content: Some("APPROVED: Validation score and protocol independently verified.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 200, completion_tokens: 30, total_tokens: 230 },
            latency_ms: 15,
        },
        ChatCompletionResponse {
            id: "turn4-boss-finalize".into(),
            model: model_name.into(),
            content: Some("FINALIZE_AND_STOP: Candidate approved.".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 180, completion_tokens: 20, total_tokens: 200 },
            latency_ms: 10,
        },
    ];
    let mock_path = root.join("mock_responses.json");
    fs::write(&mock_path, serde_json::to_string_pretty(&responses).unwrap()).unwrap();

    let state_dir = root.join("cli_state");
    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");

    let output = std::process::Command::new(bin_path)
        .args([
            "autonomous-create",
            "--offline-fixture",
            "--state", state_dir.to_str().unwrap(),
            "--spec", spec_path.to_str().unwrap(),
            "--mock", mock_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to execute cli binary");

    assert!(output.status.success(), "CLI failed: {}", String::from_utf8_lossy(&output.stderr));
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("proj_cli_mock_test"));
    assert!(stdout.contains("bundle_verification"));
}
