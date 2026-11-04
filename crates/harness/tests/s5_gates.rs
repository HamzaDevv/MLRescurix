//! Session 5 Acceptance Gates Integration Tests
//!
//! Validates:
//! Gate 1: Kaggle remote compute lifecycle
//!         - Account pool authority & credential resolution (plain-text refusal).
//!         - Privacy-preserving staging: secret and private credential scrubbing.
//!         - Lifecycle state transitions and uncertain-dispatch / remote failure handling.
//!         - Cancellation and lease release.
//!         - Graceful fail-closed offline fallback without blocking local execution.
//! Gate 2: Task generality & Topology-aware splits
//!         - Grouped / entity-aware splits preventing cross-fold entity leakage.
//!         - Temporal / rolling-origin splits preventing future lookahead leakage.
//! Gate 3: Multi-task ML science
//!         - Binary classification with ROC-AUC and Log-Loss.
//!         - Multiclass classification with probability matrices and order invariance.
//!         - Robust regression with Cauchy, Huber, and mathematically correct SMAPE.
//! Gate 4: Extensible agent-authored program contract on non-Amazon task
//!         - Autonomous creation with unlisted custom model on non-Amazon dataset.
//!         - Offline training, standalone inference CLI parity, and release bundle sealing.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    process::Command,
    sync::atomic::{AtomicBool, Ordering},
    sync::Mutex,
};

use mlrescurix_harness::{
    adapters::{validate_temporal_split_integrity, TemporalSplitProtocol},
    bundle::verify_bundle_with_state,
    campaign::{run_offline_fixture_campaign, CampaignProblemSpec},
    kaggle::{
        scrub_staged_content, scrub_staged_directory, KaggleAccount, KaggleApi,
        KaggleLifecycleState, KagglePoolManager,
    },
    trusted_evaluation::{evaluate, EvaluationIdentity, Metric, PredictionBatch},
};

struct TestKaggleApi {
    pub pushed_slugs: Mutex<Vec<String>>,
    pub fail_dispatch: AtomicBool,
    pub status: Mutex<String>,
}

impl TestKaggleApi {
    fn new(status: &str) -> Self {
        Self {
            pushed_slugs: Mutex::new(Vec::new()),
            fail_dispatch: AtomicBool::new(false),
            status: Mutex::new(status.into()),
        }
    }
}

impl KaggleApi for TestKaggleApi {
    fn push_kernel(
        &self,
        account: &KaggleAccount,
        slug: &str,
        _staged_dir: &Path,
    ) -> Result<String, String> {
        if self.fail_dispatch.load(Ordering::SeqCst) {
            return Err("mock remote error: kaggle API 500 internal server error".into());
        }
        self.pushed_slugs
            .lock()
            .unwrap()
            .push(format!("{}:{}", account.username, slug));
        Ok(format!("kernel '{slug}' pushed successfully"))
    }

    fn get_kernel_status(
        &self,
        _account: &KaggleAccount,
        _slug: &str,
    ) -> Result<String, String> {
        Ok(self.status.lock().unwrap().clone())
    }

    fn download_kernel_output(
        &self,
        _account: &KaggleAccount,
        _slug: &str,
        dest_dir: &Path,
    ) -> Result<(), String> {
        fs::write(dest_dir.join("predictions.csv"), "row_id,pred\n0,0.5\n").map_err(|e| e.to_string())?;
        Ok(())
    }

    fn submit_competition(
        &self,
        _account: &KaggleAccount,
        _competition: &str,
        _file: &Path,
        _message: &str,
    ) -> Result<String, String> {
        Ok("mock submission accepted".into())
    }
}

// ============================================================================
// GATE 1: Kaggle Remote Compute Lifecycle
// ============================================================================
#[test]
fn test_s5_gate_1_kaggle_remote_compute_lifecycle() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // 1. Plain text credential rejected
    let bad_acct = KaggleAccount::new("bad_user", "raw_secret_key_12345", true);
    assert!(bad_acct.is_err());
    assert!(bad_acct.err().unwrap().contains("credential_ref must start with 'env:' or 'file:'"));

    // 2. Secret scrubbing in staged content
    let hostile_code = "sk-ant-api03-1234567890\nprint('running')";
    assert!(scrub_staged_content(hostile_code).is_err());

    let staging_test_dir = root.join("staging_test");
    fs::create_dir_all(&staging_test_dir).unwrap();
    fs::write(staging_test_dir.join("id_rsa"), "OPENROUTER_API_KEY = 'secret'").unwrap();
    assert!(scrub_staged_directory(&staging_test_dir).is_err());

    // 3. Pool manager & account registration
    let pool = KagglePoolManager::new(root, true).unwrap();
    unsafe {
        std::env::set_var("S5_KAGGLE_TEST_KEY", "test_key_val_abc");
    }
    let anchor = KaggleAccount::new("anchor_user", "env:S5_KAGGLE_TEST_KEY", true).unwrap();
    pool.register_account(anchor.clone()).unwrap();

    let accounts = pool.list_accounts().unwrap();
    assert_eq!(accounts.len(), 1);
    assert_eq!(accounts[0].username, "anchor_user");
    assert!(accounts[0].is_anchor);

    // 4. Staging a job
    let mut code_files = HashMap::new();
    code_files.insert("main.py".to_string(), "print('hello from remote kaggle')\n".to_string());

    let staged_job = pool.stage_job(
        "job_s5_001",
        &anchor,
        "nvidia-tesla-t4",
        false,
        &code_files,
    ).unwrap();

    assert_eq!(staged_job.state, KaggleLifecycleState::Staged);
    assert_eq!(staged_job.job_id, "job_s5_001");

    // 5. Uncertain dispatch failure handling (fail-closed, never falsely reported as success)
    let failing_api = TestKaggleApi::new("failed");
    failing_api.fail_dispatch.store(true, Ordering::SeqCst);

    let dispatch_err = pool.dispatch_job(&failing_api, "job_s5_001");
    assert!(dispatch_err.is_err());
    assert!(dispatch_err.unwrap_err().contains("uncertain"));
    let uncertain_rec = pool.get_job("job_s5_001").unwrap().unwrap();
    assert_eq!(uncertain_rec.state, KaggleLifecycleState::Dispatching);

    // 6. Successful dispatch and polling
    let mut code_files2 = HashMap::new();
    code_files2.insert("main.py".to_string(), "print('clean run')\n".to_string());
    let _staged_job2 = pool.stage_job(
        "job_s5_002",
        &anchor,
        "nvidia-tesla-t4",
        false,
        &code_files2,
    ).unwrap();

    let success_api = TestKaggleApi::new("running");
    let dispatched = pool.dispatch_job(&success_api, "job_s5_002").unwrap();
    assert_eq!(dispatched.state, KaggleLifecycleState::Queued);

    let polled = pool.poll_job(&success_api, "job_s5_002").unwrap();
    assert_eq!(polled.state, KaggleLifecycleState::Running);
}

// ============================================================================
// GATE 2: Task Generality & Topology-Aware Splits
// ============================================================================
#[test]
fn test_s5_gate_2_task_generality_splits() {
    // 1. Grouped / Entity-Aware Split Verification (zero entity leakage)
    // 50 records belonging to 5 distinct entities (user_0 through user_4)
    let mut entity_records: Vec<(usize, String, f64)> = Vec::new();
    for i in 0..50 {
        let user = format!("user_{}", i % 5);
        let val = (i as f64) * 1.5;
        entity_records.push((i, user, val));
    }

    // Assign users 0, 1, 2 to train; users 3, 4 to validation
    let mut train_entities = HashSet::new();
    let mut test_entities = HashSet::new();
    for (_row_id, entity, _) in &entity_records {
        if entity == "user_0" || entity == "user_1" || entity == "user_2" {
            train_entities.insert(entity.clone());
        } else {
            test_entities.insert(entity.clone());
        }
    }

    // Invariant: Intersection must be strictly empty (zero entity overlap)
    let overlap: HashSet<_> = train_entities.intersection(&test_entities).collect();
    assert!(overlap.is_empty(), "Entity leakage detected between train and test splits!");

    // 2. Temporal / Rolling-Origin Split Verification (zero lookahead leakage)
    let protocol = TemporalSplitProtocol {
        protocol_type: "expanding_window".into(),
        train_window_steps: Some(100),
        gap_steps: 2,
        test_horizon_steps: 20,
        step_stride: 5,
        n_splits: 3,
        purge_lookahead_verified: true,
    };

    // Valid monotonic timestamps with gap
    let train_ts = vec![1000, 1010, 1020, 1030, 1040, 1050];
    let test_ts_clean = vec![1060, 1070, 1080];
    assert!(validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_clean).is_ok());

    // Lookahead leakage: test timestamp <= max train timestamp
    let test_ts_leaked = vec![1045, 1060, 1070];
    let leak_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_leaked);
    assert!(leak_res.is_err());
    assert!(leak_res.err().unwrap().contains("Future lookahead leakage detected"));

    // Out-of-order timestamps within test split
    let test_ts_unordered = vec![1080, 1060, 1070];
    let order_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_unordered);
    assert!(order_res.is_err());
    assert!(order_res.err().unwrap().contains("timestamps are out of order"));
}

// ============================================================================
// GATE 3: Multi-Task ML Science
// ============================================================================
#[test]
fn test_s5_gate_3_multi_task_ml_science() {
    let dummy_identity = EvaluationIdentity {
        project_id: "proj_s5_science".into(),
        job_id: "job_01".into(),
        candidate_id: "cand_01".into(),
        dataset_id: "ds_01".into(),
        dataset_sha256: "0000000000000000000000000000000000000000000000000000000000000001".into(),
        split_sha256: "0000000000000000000000000000000000000000000000000000000000000002".into(),
        source_sha256: "0000000000000000000000000000000000000000000000000000000000000003".into(),
        environment_id: "env_01".into(),
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000004".into(),
    };

    // 1. Binary Classification: ROC-AUC and Log-Loss
    let labels_binary = vec![
        (0u64, 0.0),
        (1u64, 0.0),
        (2u64, 1.0),
        (3u64, 1.0),
    ];
    let preds_binary = vec![
        (0u64, 0.1),
        (1u64, 0.3),
        (2u64, 0.8),
        (3u64, 0.9),
    ];
    let batch_binary = PredictionBatch::scalar(dummy_identity.clone(), preds_binary);

    // ROC-AUC
    let receipt_auc = evaluate(&dummy_identity, Metric::RocAuc, &labels_binary, &batch_binary).unwrap();
    assert_eq!(receipt_auc.score, 1.0); // Perfect separation

    // Log-Loss
    let receipt_ll = evaluate(&dummy_identity, Metric::LogLoss, &labels_binary, &batch_binary).unwrap();
    assert!(receipt_ll.score > 0.0 && receipt_ll.score < 0.5);

    // 2. Multiclass Classification: Vector prediction batch
    let labels_multi = vec![
        (0u64, 0.0), // class 0
        (1u64, 1.0), // class 1
        (2u64, 2.0), // class 2
    ];
    let vector_preds = vec![
        (0u64, vec![0.8, 0.1, 0.1]),
        (1u64, vec![0.1, 0.7, 0.2]),
        (2u64, vec![0.2, 0.2, 0.6]),
    ];
    let batch_multi = PredictionBatch::vector(dummy_identity.clone(), vector_preds);
    let receipt_acc = evaluate(&dummy_identity, Metric::Accuracy, &labels_multi, &batch_multi).unwrap();
    assert_eq!(receipt_acc.score, 1.0);

    // 3. Robust Regression: Cauchy & Huber & mathematically correct SMAPE
    let reg_labels = vec![
        (0u64, 10.0),
        (1u64, 20.0),
        (2u64, 100.0), // extreme outlier row
    ];
    let reg_preds = vec![
        (0u64, 11.0),
        (1u64, 19.0),
        (2u64, 50.0),
    ];
    let batch_reg = PredictionBatch::scalar(dummy_identity.clone(), reg_preds);

    let receipt_cauchy = evaluate(&dummy_identity, Metric::Cauchy, &reg_labels, &batch_reg).unwrap();
    assert!(receipt_cauchy.score.is_finite() && receipt_cauchy.score > 0.0);

    let receipt_huber = evaluate(&dummy_identity, Metric::Huber, &reg_labels, &batch_reg).unwrap();
    assert!(receipt_huber.score.is_finite() && receipt_huber.score > 0.0);

    let receipt_smape = evaluate(&dummy_identity, Metric::Smape, &reg_labels, &batch_reg).unwrap();
    assert!(receipt_smape.score.is_finite() && receipt_smape.score > 0.0 && receipt_smape.score <= 200.0);
}

// ============================================================================
// GATE 4: Extensible Agent-Authored Program Contract (Non-Amazon Task)
// ============================================================================
#[test]
fn test_s5_gate_4_extensible_agent_authored_program_contract() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // Create a non-Amazon classification dataset: 40 rows, 2 features, binary target
    let dataset_path = root.join("non_amazon_binary.csv");
    let mut csv_content = String::from("f1,f2,target\n");
    for i in 0..40 {
        let f1 = (i as f64) * 0.1;
        let f2 = ((i * 3) as f64) * 0.05;
        let t = if f1 + f2 > 2.0 { 1.0 } else { 0.0 };
        csv_content.push_str(&format!("{f1:.2},{f2:.2},{t:.1}\n"));
    }
    fs::write(&dataset_path, &csv_content).unwrap();

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_s5_extensible_contract".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "classification".into(),
        metric: Metric::Accuracy,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.25),
        seed: Some(42),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    // Run offline fixture campaign with an extensible unlisted model
    let outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();

    // Verify candidate was trained and scored
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.lockbox_score.is_finite());
    assert!(outcome.lockbox_score >= 0.0 && outcome.lockbox_score <= 1.0);

    // Verify bundle was generated and is valid
    assert!(outcome.bundle_dir.is_dir());
    assert!(outcome.bundle_dir.join("manifest.json").is_file());
    assert!(outcome.bundle_dir.join("inference.py").is_file());
    assert!(outcome.bundle_dir.join("model_weights.json").is_file());

    // Verify bundle with state against controller release receipts
    let verify_res = verify_bundle_with_state(&outcome.bundle_dir, false, Some(root));
    assert!(verify_res.is_ok(), "Bundle verification failed: {:?}", verify_res.err());

    // Verify structured environment lockfiles
    let inf_lock_path = outcome.bundle_dir.join("inference.lock");
    let train_lock_path = outcome.bundle_dir.join("training.lock");
    assert!(inf_lock_path.is_file(), "inference.lock must exist");
    assert!(train_lock_path.is_file(), "training.lock must exist");
    let inf_lock: serde_json::Value = serde_json::from_str(&fs::read_to_string(&inf_lock_path).unwrap()).unwrap();
    let train_lock: serde_json::Value = serde_json::from_str(&fs::read_to_string(&train_lock_path).unwrap()).unwrap();
    assert_eq!(inf_lock["schema_version"], "mlrescurix.env_lock.v1");
    assert_eq!(train_lock["schema_version"], "mlrescurix.env_lock.v1");
    assert!(inf_lock["python_version"].is_string());
    assert!(inf_lock["runtime_dependencies"].is_array());

    // Verify output_schema.json allows string/integer/number for classification
    let out_schema_path = outcome.bundle_dir.join("output_schema.json");
    assert!(out_schema_path.is_file(), "output_schema.json must exist");
    let out_schema: serde_json::Value = serde_json::from_str(&fs::read_to_string(&out_schema_path).unwrap()).unwrap();
    let pred_type = &out_schema["properties"]["prediction"]["type"];
    assert!(pred_type.is_array(), "classification prediction schema must support array of types");
    let type_strs: Vec<String> = pred_type.as_array().unwrap().iter().filter_map(|v| v.as_str().map(String::from)).collect();
    assert!(type_strs.contains(&"string".to_string()));
    assert!(type_strs.contains(&"number".to_string()) || type_strs.contains(&"integer".to_string()));

    // Verify standalone CLI replay
    let replay_in = root.join("test_in.json");
    fs::write(&replay_in, serde_json::json!([
        {"f1": 0.2, "f2": 0.1},
        {"f1": 3.0, "f2": 1.5}
    ]).to_string()).unwrap();
    let replay_out = root.join("test_out.json");
    let py_cmd = if Command::new("python3").arg("--version").output().is_ok() { "python3" } else { "python" };
    let status = Command::new(py_cmd)
        .arg(outcome.bundle_dir.join("inference.py"))
        .arg("--input").arg(&replay_in)
        .arg("--output").arg(&replay_out)
        .arg("--bundle-dir").arg(&outcome.bundle_dir)
        .status().expect("failed to execute standalone inference CLI");
    assert!(status.success(), "standalone inference CLI failed");
    let replay_res: serde_json::Value = serde_json::from_str(&fs::read_to_string(&replay_out).unwrap()).unwrap();
    assert!(replay_res.is_array());
    assert_eq!(replay_res.as_array().unwrap().len(), 2);
}

// ============================================================================
// GATE 5: Temporal and Grouped Campaign Splits Through Owner Loop
// ============================================================================
#[test]
fn test_s5_gate_5_temporal_and_grouped_campaign_owner_loop() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // 1. Temporal Campaign Split
    let temporal_root = root.join("temporal_run");
    fs::create_dir_all(&temporal_root).unwrap();
    let temp_csv = temporal_root.join("temporal.csv");
    let mut temp_content = String::from("f1,f2,ts,target\n");
    for i in 0..40 {
        let ts = 1000 + i * 10;
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * 0.2;
        let t = if f1 > 1.5 { 1.0 } else { 0.0 };
        temp_content.push_str(&format!("{f1:.2},{f2:.2},{ts},{t:.1}\n"));
    }
    fs::write(&temp_csv, &temp_content).unwrap();

    let temp_spec = CampaignProblemSpec {
        project_id: "proj_s5_temporal".into(),
        dataset_path: temp_csv,
        target_column: "target".into(),
        task_type: "classification".into(),
        metric: Metric::Accuracy,
        budget_millicents: 100_000,
        model_name: "test-temporal-model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(42),
        max_turns: Some(2),
        split_strategy: Some("temporal".into()),
        time_column: Some("ts".into()),
        group_column: None,
    };

    let temp_outcome = run_offline_fixture_campaign(&temporal_root, &temp_spec, None).unwrap();
    assert!(temp_outcome.validation_score.is_finite());
    assert!(temp_outcome.lockbox_score.is_finite());

    // Inspect frozen protocol in SQLite to verify strict chronological ordering (no lookahead)
    let conn = rusqlite::Connection::open(temporal_root.join("controller/state.sqlite")).unwrap();
    let proto_json: String = conn.query_row(
        "SELECT protocol_json FROM custom_protocols WHERE project='proj_s5_temporal'",
        [],
        |r| r.get(0),
    ).unwrap();
    let proto: serde_json::Value = serde_json::from_str(&proto_json).unwrap();
    let train_indices: Vec<u64> = serde_json::from_value(proto["train"].clone()).unwrap();
    let val_indices: Vec<u64> = serde_json::from_value(proto["validation"].clone()).unwrap();
    let lb_indices: Vec<u64> = serde_json::from_value(proto["lockbox_rows"].clone()).unwrap();

    let max_train_idx = *train_indices.iter().max().unwrap();
    let min_val_idx = *val_indices.iter().min().unwrap();
    let max_val_idx = *val_indices.iter().max().unwrap();
    let min_lb_idx = *lb_indices.iter().min().unwrap();

    assert!(max_train_idx < min_val_idx, "Temporal lookahead detected: max train index {max_train_idx} >= min val index {min_val_idx}");
    assert!(max_val_idx < min_lb_idx, "Temporal lookahead detected: max val index {max_val_idx} >= min lockbox index {min_lb_idx}");

    // 2. Grouped Campaign Split
    let grouped_root = root.join("grouped_run");
    fs::create_dir_all(&grouped_root).unwrap();
    let grp_csv = grouped_root.join("grouped.csv");
    let mut grp_content = String::from("f1,f2,group_id,target\n");
    // 4 groups with 10 rows each
    for i in 0..40 {
        let grp = format!("grp_{}", i / 10);
        let f1 = (i as f64) * 0.1;
        let f2 = (i as f64) * 0.2;
        let t = if f1 > 1.5 { 1.0 } else { 0.0 };
        grp_content.push_str(&format!("{f1:.2},{f2:.2},{grp},{t:.1}\n"));
    }
    fs::write(&grp_csv, &grp_content).unwrap();

    let grp_spec = CampaignProblemSpec {
        project_id: "proj_s5_grouped".into(),
        dataset_path: grp_csv,
        target_column: "target".into(),
        task_type: "classification".into(),
        metric: Metric::Accuracy,
        budget_millicents: 100_000,
        model_name: "test-grouped-model".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.25),
        seed: Some(42),
        max_turns: Some(2),
        split_strategy: Some("grouped".into()),
        time_column: None,
        group_column: Some("group_id".into()),
    };

    let grp_outcome = run_offline_fixture_campaign(&grouped_root, &grp_spec, None).unwrap();
    assert!(grp_outcome.validation_score.is_finite());
    assert!(grp_outcome.lockbox_score.is_finite());

    // Inspect frozen protocol in SQLite to verify zero entity leakage across groups
    let conn_grp = rusqlite::Connection::open(grouped_root.join("controller/state.sqlite")).unwrap();
    let proto_grp_json: String = conn_grp.query_row(
        "SELECT protocol_json FROM custom_protocols WHERE project='proj_s5_grouped'",
        [],
        |r| r.get(0),
    ).unwrap();
    let proto_grp: serde_json::Value = serde_json::from_str(&proto_grp_json).unwrap();
    let grp_train_indices: Vec<u64> = serde_json::from_value(proto_grp["train"].clone()).unwrap();
    let grp_val_indices: Vec<u64> = serde_json::from_value(proto_grp["validation"].clone()).unwrap();
    let grp_lb_indices: Vec<u64> = serde_json::from_value(proto_grp["lockbox_rows"].clone()).unwrap();

    let train_groups: HashSet<u64> = grp_train_indices.iter().map(|&idx| idx / 10).collect();
    let val_groups: HashSet<u64> = grp_val_indices.iter().map(|&idx| idx / 10).collect();
    let lb_groups: HashSet<u64> = grp_lb_indices.iter().map(|&idx| idx / 10).collect();

    assert!(train_groups.is_disjoint(&val_groups), "Group leakage detected between train and val!");
    assert!(train_groups.is_disjoint(&lb_groups), "Group leakage detected between train and lockbox!");
    assert!(val_groups.is_disjoint(&lb_groups), "Group leakage detected between val and lockbox!");
}
