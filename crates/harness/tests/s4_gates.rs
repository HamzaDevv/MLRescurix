//! Session 4 Hardening Gates Integration Tests
//!
//! Validates:
//! Gate 1: Real process crash recovery window.
//!         - Process is killed with exit code 137 immediately after authoritative lockbox consumption.
//!         - Database verified: lockbox consumption = 1, controller release receipts = 0.
//!         - Second invocation cleanly recovers the candidate, packages the delivery bundle,
//!           records the release receipt, and preserves strictly once-only holdout evaluation (count = 1).
//! Gate 2: Tamper detection during the crash window fails closed.
//!         - Tampering with candidate source or weights after crash but before recovery is rejected.
//!         - Holdout is never re-evaluated or double-consumed on rejection.
//! Gate 3: Single Rust resource authority concurrency.
//!         - Concurrent multi-threaded search query admission strictly enforces limits without races.
//!         - Idempotent settlement succeeds.
//!         - Conflicting terminal state settlement fails closed.
//! Gate 4: Kaggle account pool registration and listing through harness authority.
//! Gate 5: Unified resource status queries.

use std::{
    fs,
    path::Path,
    thread,
};

use rusqlite::Connection;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    bundle::{verify_bundle, verify_bundle_with_state},
    campaign::CampaignProblemSpec,
    kaggle::{KaggleAccount, KagglePoolManager},
    resource_authority::{
        admit_kaggle_job, admit_search_attempt, admit_token_reservation,
        get_unified_resource_status, query_search_attempt_count, settle_kaggle_job,
        settle_search_attempt, settle_token_reservation,
    },
    trusted_evaluation::Metric,
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

fn spawn_and_sigkill_at_crash_window(bin_path: &str, root: &Path, spec_path: &Path) {
    let marker_path = root.join(".lockbox_consumed");
    if marker_path.exists() {
        let _ = fs::remove_file(&marker_path);
    }

    let mut child = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(spec_path)
        .arg("--offline-fixture")
        .env("MLRESCURIX_CRASH_AFTER_LOCKBOX_CONSUMPTION", "1")
        .spawn()
        .expect("must spawn child harness binary");

    let mut consumed = false;
    for _ in 0..300 {
        if marker_path.is_file() {
            consumed = true;
            break;
        }
        thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(consumed, "child process must reach lockbox consumption and write .lockbox_consumed marker");

    // Deliver actual external SIGKILL (Signal 9) via libc::kill
    let pid = child.id() as libc::pid_t;
    let ret = unsafe { libc::kill(pid, libc::SIGKILL) };
    assert_eq!(ret, 0, "libc::kill must successfully deliver SIGKILL to child PID");

    let status = child.wait().expect("child wait must succeed");
    use std::os::unix::process::ExitStatusExt;
    assert_eq!(status.signal(), Some(9), "child process must be terminated by external Signal 9 (SIGKILL)");
    assert_eq!(status.code(), None, "externally killed child must have no exit code");
}

// ============================================================================
// GATE 1: Real Process Termination Crash Recovery Window (External SIGKILL)
// ============================================================================
#[test]
fn test_s4_gate_real_process_termination_crash_recovery_window() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("s4_crash_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let project_id = "proj_s4_crash_recovery";

    let problem_spec = CampaignProblemSpec {
        project_id: project_id.into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let spec_path = root.join("problem_spec.json");
    fs::write(&spec_path, serde_json::to_string_pretty(&problem_spec).unwrap()).unwrap();

    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");

    // Phase 1: Child process runs with MLRESCURIX_CRASH_AFTER_LOCKBOX_CONSUMPTION=1.
    // It commits authoritative lockbox consumption and signals readiness; external supervisor delivers SIGKILL.
    spawn_and_sigkill_at_crash_window(bin_path, root, &spec_path);

    // Verify DB state immediately after crash:
    // Lockbox consumption record MUST exist (1).
    // Controller release receipt MUST NOT exist (0).
    let db_path = root.join("controller/state.sqlite");
    assert!(db_path.is_file(), "database must exist after first execution");
    let conn = Connection::open(&db_path).unwrap();

    let consumptions_post_crash: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(
        consumptions_post_crash, 1,
        "lockbox consumption must be committed before crash"
    );

    let release_receipts_post_crash: i64 = conn.query_row(
        "SELECT COUNT(*) FROM controller_release_receipts WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(
        release_receipts_post_crash, 0,
        "release receipt must not exist yet when crash occurs"
    );

    // Phase 2: Resume campaign without crash flag.
    // It MUST safely recover without evaluating lockbox a second time!
    let child2 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must execute child resumption");

    let stdout_str = String::from_utf8_lossy(&child2.stdout);
    let stderr_str = String::from_utf8_lossy(&child2.stderr);
    assert!(
        child2.status.success(),
        "resumption campaign must succeed after crash! stdout: {}\nstderr: {}",
        stdout_str,
        stderr_str
    );

    // Verify DB state after recovery:
    // Lockbox consumption count MUST STILL BE EXACTLY 1 (zero second evaluations).
    let consumptions_post_recovery: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(
        consumptions_post_recovery, 1,
        "lockbox holdout MUST NOT be double-consumed during crash recovery"
    );

    // Controller release receipts MUST now be exactly 1.
    let release_receipts_post_recovery: i64 = conn.query_row(
        "SELECT COUNT(*) FROM controller_release_receipts WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(
        release_receipts_post_recovery, 1,
        "controller release receipt must be recorded upon recovery completion"
    );

    // Verify environment_id is properly bound in release receipt
    let env_id: String = conn.query_row(
        "SELECT json_extract(receipt_json, '$.environment_id') FROM controller_release_receipts WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(env_id, "host_python_uncontained_fixture");

    // Phase 3: Verify the packaged bundle is verified and complete
    let cand_id = format!("cand_{}_{}", project_id, &ds_hash[..8]);
    let bundle_dir = root.join(format!("delivery_bundle_{}", cand_id));
    assert!(bundle_dir.join("manifest.json").is_file());
    assert!(bundle_dir.join("scorecard.json").is_file());
    assert!(bundle_dir.join("solution.py").is_file());
    assert!(bundle_dir.join("model_weights.json").is_file());

    let report = verify_bundle(&bundle_dir, false).unwrap();
    assert!(report.manifest_verified);
    assert!(report.contracts_verified);
}

// ============================================================================
// GATE 2: Tamper Rejection During Crash Recovery Window
// ============================================================================
#[test]
fn test_s4_gate_crash_window_tamper_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("s4_tamper_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let project_id = "proj_s4_crash_tamper";

    let problem_spec = CampaignProblemSpec {
        project_id: project_id.into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let spec_path = root.join("problem_spec.json");
    fs::write(&spec_path, serde_json::to_string_pretty(&problem_spec).unwrap()).unwrap();

    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");

    // 1. Crash after lockbox consumption via external SIGKILL
    spawn_and_sigkill_at_crash_window(bin_path, root, &spec_path);

    let cand_id = format!("cand_{}_{}", project_id, &ds_hash[..8]);
    let branch_sol = root.join(format!("controller/workflows/{}/branches/branch_{}/solution.py", project_id, cand_id));
    assert!(branch_sol.is_file(), "candidate solution must exist in branch");

    // 2. Tamper with candidate source in branch workspace during the crash window
    let original_sol = fs::read_to_string(&branch_sol).unwrap();
    let tampered_sol = original_sol.replace("val = sum(w[j] * phi[j] for j in range(dim))", "val = 999999.0");
    assert_ne!(original_sol, tampered_sol);
    fs::write(&branch_sol, tampered_sol).unwrap();

    // 3. Attempt resumption - MUST fail closed because candidate source changed after lockbox consumption!
    let child2 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must run resumption");

    assert!(!child2.status.success(), "tampered resumption must fail closed");
    let err_str = String::from_utf8_lossy(&child2.stderr);
    assert!(
        err_str.contains("cannot repackage modified source") || err_str.contains("does not match evaluated source SHA-256") || err_str.contains("lockbox consumption mismatch"),
        "expected tamper error, got: {}",
        err_str
    );

    // Verify lockbox is STILL consumed strictly once
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 1, "lockbox must never be double-consumed on tamper");

    // 4. Restore original source and verify clean recovery succeeds
    fs::write(&branch_sol, original_sol).unwrap();
    let child3 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must run resumption");

    assert!(child3.status.success(), "clean restored candidate must succeed recovery: {}", String::from_utf8_lossy(&child3.stderr));
}

#[test]
fn test_s4_gate_crash_window_weights_tamper_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("s4_weights_tamper_dataset.csv");
    let (ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let project_id = "proj_s4_crash_weights_tamper";

    let problem_spec = CampaignProblemSpec {
        project_id: project_id.into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let spec_path = root.join("problem_spec.json");
    fs::write(&spec_path, serde_json::to_string_pretty(&problem_spec).unwrap()).unwrap();

    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");

    // 1. Crash after lockbox consumption via external SIGKILL
    spawn_and_sigkill_at_crash_window(bin_path, root, &spec_path);

    let cand_id = format!("cand_{}_{}", project_id, &ds_hash[..8]);
    let branch_weights = root.join(format!("controller/workflows/{}/branches/branch_{}/weights.json", project_id, cand_id));
    assert!(branch_weights.is_file(), "candidate weights must exist in branch");

    // 2. Tamper with candidate weights in branch workspace during the crash window
    let original_weights = fs::read_to_string(&branch_weights).unwrap();
    let tampered_weights = r#"{"weights": [9999.0, 9999.0, 9999.0, 9999.0]}"#;
    fs::write(&branch_weights, tampered_weights).unwrap();

    // Also tamper in worker_workspace if present
    let worker_weights = root.join("worker_workspace/weights.json");
    if worker_weights.is_file() {
        fs::write(&worker_weights, tampered_weights).unwrap();
    }

    // 3. Attempt resumption - MUST fail closed because candidate weights changed after lockbox consumption!
    let child2 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must run resumption");

    assert!(!child2.status.success(), "tampered weights resumption must fail closed");
    let err_str = String::from_utf8_lossy(&child2.stderr);
    assert!(
        err_str.contains("cannot repackage modified weights") || err_str.contains("does not match evaluated artifact SHA-256"),
        "expected weights tamper error, got: {}",
        err_str
    );

    // Verify lockbox is STILL consumed strictly once
    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();
    let consumptions: i64 = conn.query_row(
        "SELECT COUNT(*) FROM assessment_lockbox_consumptions WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(consumptions, 1, "lockbox must never be double-consumed on weights tamper");

    // 4. Restore original weights and verify clean recovery succeeds
    fs::write(&branch_weights, &original_weights).unwrap();
    if worker_weights.is_file() {
        fs::write(&worker_weights, &original_weights).unwrap();
    }
    let child3 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must run resumption");

    assert!(child3.status.success(), "clean restored candidate must succeed recovery: {}", String::from_utf8_lossy(&child3.stderr));
}

// ============================================================================
// GATE 3: Concurrent Rust Resource Authority (Search & Token Admissions)
// ============================================================================
#[test]
fn test_s4_gate_rust_resource_authority_concurrent_search() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let db_path = root.join("state.sqlite");

    let project = "proj_resource_authority_concurrency";
    let search_limit = 10;
    let total_attempts = 25;

    // Spawn 25 concurrent threads attempting to admit search queries against a limit of 10
    let mut handles = Vec::new();
    for i in 0..total_attempts {
        let db_p = db_path.clone();
        let proj = project.to_string();
        let query = format!("query about tabular models turn {i}");
        let sanitized = format!("sanitized query {i}");
        handles.push(thread::spawn(move || {
            admit_search_attempt(&db_p, &proj, &query, &sanitized, "mock_provider", search_limit)
        }));
    }

    let mut admitted_count = 0;
    let mut rejected_count = 0;
    let mut admitted_ids = Vec::new();

    for h in handles {
        let res = h.join().unwrap().unwrap();
        match res {
            Some(id) => {
                admitted_count += 1;
                admitted_ids.push(id);
            }
            None => {
                rejected_count += 1;
            }
        }
    }

    assert_eq!(admitted_count, search_limit, "strictly {search_limit} attempts admitted");
    assert_eq!(rejected_count, total_attempts - search_limit, "excess attempts rejected");

    let count = query_search_attempt_count(&db_path, project).unwrap();
    assert_eq!(count, search_limit);

    // Idempotent settlement test: settling the first attempt twice with "success"
    let first_id = &admitted_ids[0];
    let settle1 = settle_search_attempt(&db_path, project, first_id, "success", None, 0);
    assert!(settle1.is_ok(), "first settlement must succeed");

    let settle2 = settle_search_attempt(&db_path, project, first_id, "success", None, 0);
    assert!(settle2.is_ok(), "idempotent repeated settlement must succeed");
    assert!(settle_search_attempt(&db_path, project, first_id, "success", None, 100).is_err(),
        "a duplicate status with a conflicting charge must not be silently accepted");

    // Conflicting terminal settlement test: settling an already successful attempt as failed
    let settle_conflict = settle_search_attempt(&db_path, project, first_id, "failed", Some("fake error"), 0);
    assert!(settle_conflict.is_err(), "conflicting terminal settlement must be rejected");
    let conflict_err = settle_conflict.unwrap_err();
    assert!(
        conflict_err.contains("already terminal"),
        "expected terminal rejection error, got: {conflict_err}"
    );

    // Settle all remaining admitted attempts
    for id in &admitted_ids[1..] {
        settle_search_attempt(&db_path, project, id, "success", None, 0).unwrap();
    }

    // Unified resource status test
    let status = get_unified_resource_status(root, project).unwrap();
    assert_eq!(status["search_attempts"], search_limit);
}

#[test]
fn test_s4_gate_token_and_kaggle_resource_authority_lifecycle() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let db_path = root.join("state.sqlite");
    let project = "proj_token_kaggle_lifecycle";

    // 1. Kaggle job admission and settlement lifecycle
    let k_limit = 5.0; // 5.0 hours limit
    let job1 = admit_kaggle_job(&db_path, project, "kjob_1", "user1", "notebook_1", "gpu", 3.0, k_limit).unwrap();
    assert_eq!(job1, Some("kjob_1".into()));

    // Settling job1 as failed with 2.5 actual hours
    settle_kaggle_job(&db_path, project, "kjob_1", "failed", 2.5, Some("out of memory")).unwrap();

    // Idempotent settlement succeeds
    settle_kaggle_job(&db_path, project, "kjob_1", "failed", 2.5, Some("out of memory")).unwrap();

    // Conflicting settlement with different actual hours is rejected
    assert!(settle_kaggle_job(&db_path, project, "kjob_1", "failed", 3.0, Some("out of memory")).is_err());

    // Conflicting settlement with different status is rejected
    assert!(settle_kaggle_job(&db_path, project, "kjob_1", "success", 2.5, None).is_err());

    // Job 2 requests 3.0 hours: 2.5 used + 3.0 requested = 5.5 > 5.0 limit -> MUST BE REJECTED!
    let job2 = admit_kaggle_job(&db_path, project, "kjob_2", "user1", "notebook_2", "gpu", 3.0, k_limit).unwrap();
    assert_eq!(job2, None, "job requesting hours exceeding quota taking failed hours into account must be rejected");

    // Job 3 requests 2.0 hours: 2.5 used + 2.0 requested = 4.5 <= 5.0 limit -> MUST BE ADMITTED!
    let job3 = admit_kaggle_job(&db_path, project, "kjob_3", "user1", "notebook_3", "gpu", 2.0, k_limit).unwrap();
    assert_eq!(job3, Some("kjob_3".into()));

    // Settle job 3 as success with 1.5 actual hours
    settle_kaggle_job(&db_path, project, "kjob_3", "success", 1.5, None).unwrap();

    // 2. Token reservation lifecycle and conflicting settlement rejection
    let tok_limit = 12_000;
    let spend_limit = 100_000; // millicents
    let res1 = admit_token_reservation(&db_path, project, "res_1", "worker", "deepseek-v4.1", 1000, 500, 29, tok_limit, spend_limit).unwrap();
    assert_eq!(res1, Some("res_1".into()));

    // Settle res1
    settle_token_reservation(&db_path, project, "res_1", "ok", 950, 480, 28, None).unwrap();

    // Idempotent settlement with exact same parameters succeeds
    settle_token_reservation(&db_path, project, "res_1", "ok", 950, 480, 28, None).unwrap();

    // Conflicting settlement with altered prompt tokens is rejected
    assert!(settle_token_reservation(&db_path, project, "res_1", "ok", 1200, 480, 28, None).is_err());

    // Conflicting settlement with altered cost is rejected
    assert!(settle_token_reservation(&db_path, project, "res_1", "ok", 950, 480, 50, None).is_err());

    // Conflicting settlement with altered status is rejected
    assert!(settle_token_reservation(&db_path, project, "res_1", "failed", 950, 480, 28, Some("err")).is_err());

    // Outcome unknown reservation must count towards budget ceiling
    // res2 requests 8000 + 1000 = 9000 tokens (1430 used + 9000 requested = 10430 <= 12000 -> admitted)
    let res2 = admit_token_reservation(&db_path, project, "res_2", "worker", "deepseek-v4.1", 8000, 1000, 200, tok_limit, spend_limit).unwrap();
    assert_eq!(res2, Some("res_2".into()));

    // Mark res2 as outcome_unknown via settlement
    settle_token_reservation(&db_path, project, "res_2", "outcome_unknown", 8000, 1000, 200, None).unwrap();

    // Idempotent re-settlement of outcome_unknown succeeds
    settle_token_reservation(&db_path, project, "res_2", "outcome_unknown", 8000, 1000, 200, None).unwrap();

    // Conflicting settlement of outcome_unknown reservation must be strictly rejected
    assert!(
        settle_token_reservation(&db_path, project, "res_2", "ok", 8000, 1000, 200, None).is_err(),
        "conflicting status transition on outcome_unknown reservation must be rejected"
    );
    assert!(
        settle_token_reservation(&db_path, project, "res_2", "outcome_unknown", 8000, 1000, 300, None).is_err(),
        "conflicting charge on outcome_unknown reservation must be rejected"
    );

    // Now tokens used: 1430 (ok) + 9000 (outcome_unknown) = 10430.
    // res3 requests 2000 tokens: 10430 + 2000 = 12430 > 12000 limit -> res3 MUST BE REJECTED!
    let res3 = admit_token_reservation(&db_path, project, "res_3", "worker", "deepseek-v4.1", 1000, 1000, 50, tok_limit, spend_limit).unwrap();
    assert_eq!(res3, None, "outcome_unknown tokens must be conservatively counted towards limit");

    // 3. Search query defensive sanitization inside Rust authority
    assert!(
        admit_search_attempt(&db_path, project, "sk-secret1234567890", "sanitized", "mock", 10).is_err(),
        "search query with secret token must be rejected by Rust authority"
    );
    assert!(
        admit_search_attempt(&db_path, project, "col1,col2,col3,col4", "sanitized", "mock", 10).is_err(),
        "search query with raw row delimiters must be rejected by Rust authority"
    );

    // 4. Unified resource status reflects actual Kaggle hours and tokens accurately
    let status = get_unified_resource_status(root, project).unwrap();
    assert_eq!(status["kaggle_jobs_count"], 2);
    // 2.5 (from failed job1) + 1.5 (from successful job3) = 4.0 hours
    assert_eq!(status["kaggle_hours_used"], 4.0);
    assert_eq!(status["total_tokens"], 10430);
    assert_eq!(status["total_spend_millicents"], 228);
}

// ============================================================================
// GATE 4: Kaggle Pool Registration and Listing Authority
// ============================================================================
#[test]
fn test_s4_gate_kaggle_pool_authority() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let pool = KagglePoolManager::new(root, true).unwrap();

    let anchor = KaggleAccount::new("anchor_user", "env:KAGGLE_ANCHOR_KEY", true).unwrap();
    let worker1 = KaggleAccount::new("worker_1", "env:KAGGLE_WORKER1_KEY", false).unwrap();

    pool.register_account(anchor).unwrap();
    pool.register_account(worker1).unwrap();

    let accounts = pool.list_accounts().unwrap();
    assert_eq!(accounts.len(), 2);

    let anchor_acct = accounts.iter().find(|a| a.is_anchor).unwrap();
    assert_eq!(anchor_acct.username, "anchor_user");
    assert_eq!(anchor_acct.gpu_hours_remaining, 30.0);

    let worker_acct = accounts.iter().find(|a| !a.is_anchor).unwrap();
    assert_eq!(worker_acct.username, "worker_1");

    // Plain text credential leak rejection
    let plain_cred_err = KaggleAccount::new("bad_user", "plain_password_123", false);
    assert!(plain_cred_err.is_err(), "plain text credential must be rejected");
}

// ============================================================================
// GATE 6: Complete Execution-to-Release Provenance Boundary Matrix
// ============================================================================
#[test]
fn test_s4_gate_execution_to_release_provenance_boundaries() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("dataset.csv");
    create_synthetic_dataset(&dataset_path, 30);
    let project_id = "proj_provenance_boundaries";

    let problem_spec = CampaignProblemSpec {
        project_id: project_id.into(),
        dataset_path,
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "fixture".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome = mlrescurix_harness::campaign::run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    let cand_id = outcome.candidate_id;

    let bundle_dir = root.join(format!("delivery_bundle_{cand_id}"));
    assert!(bundle_dir.join("manifest.json").is_file());

    // 1. Baseline verification against authoritative state MUST succeed.
    let baseline_report = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap();
    assert!(baseline_report.manifest_verified);
    assert!(baseline_report.contracts_verified);

    let db_path = root.join("controller/state.sqlite");
    let conn = Connection::open(&db_path).unwrap();

    // 2. Boundary A: Tampered protocol in custom_protocols fails closed.
    let original_proto: String = conn.query_row(
        "SELECT protocol_sha256 FROM custom_protocols WHERE project=?1",
        [project_id],
        |r| r.get(0),
    ).unwrap();

    // Drop immutable trigger to test tamper detection in verification layer
    conn.execute("DROP TRIGGER custom_protocol_no_update", []).unwrap();
    conn.execute(
        "UPDATE custom_protocols SET protocol_sha256='tampered_proto_sha' WHERE project=?1",
        [project_id],
    ).unwrap();
    let err_proto = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_proto.contains("does not match authoritative custom_protocols"),
        "expected protocol mismatch error, got: {err_proto}"
    );

    // Restore protocol and verify ok
    conn.execute(
        "UPDATE custom_protocols SET protocol_sha256=?1 WHERE project=?2",
        rusqlite::params![original_proto, project_id],
    ).unwrap();
    assert!(verify_bundle_with_state(&bundle_dir, false, Some(root)).is_ok());

    // 3. Boundary B: Tampered lockbox predictions in assessment_lockbox_consumptions fails closed.
    let (orig_pred_sha, orig_art_sha, orig_score): (String, String, f64) = conn.query_row(
        "SELECT predictions_sha256, artifact_sha256, score FROM assessment_lockbox_consumptions WHERE project=?1 AND candidate_id=?2",
        rusqlite::params![project_id, cand_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).unwrap();

    conn.execute("DROP TRIGGER assessment_consumption_no_update", []).unwrap();
    conn.execute(
        "UPDATE assessment_lockbox_consumptions SET predictions_sha256='forged_lb_pred_sha' WHERE project=?1",
        [project_id],
    ).unwrap();
    let err_rcpt = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_rcpt.contains("does not match authoritative lockbox consumption predictions"),
        "expected lockbox receipt mismatch error, got: {err_rcpt}"
    );

    // 4. Boundary C: Tampered artifact weights sha in assessment_lockbox_consumptions fails closed.
    conn.execute(
        "UPDATE assessment_lockbox_consumptions SET predictions_sha256=?1, artifact_sha256='tampered_weights_sha' WHERE project=?2",
        rusqlite::params![orig_pred_sha, project_id],
    ).unwrap();
    let err_weights = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_weights.contains("does not match evaluated lockbox artifact sha256"),
        "expected artifact weights mismatch error, got: {err_weights}"
    );

    // 5. Boundary D: Tampered lockbox score in assessment_lockbox_consumptions fails closed.
    conn.execute(
        "UPDATE assessment_lockbox_consumptions SET artifact_sha256=?1, score=999.99 WHERE project=?2",
        rusqlite::params![orig_art_sha, project_id],
    ).unwrap();
    let err_score = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_score.contains("does not match authoritative lockbox consumption score"),
        "expected score mismatch error, got: {err_score}"
    );

    // Restore lockbox consumption record
    conn.execute(
        "UPDATE assessment_lockbox_consumptions SET score=?1 WHERE project=?2",
        rusqlite::params![orig_score, project_id],
    ).unwrap();
    assert!(verify_bundle_with_state(&bundle_dir, false, Some(root)).is_ok());

    // 6. Boundary E: Missing assessment_lockbox_consumptions record fails closed.
    conn.execute("DROP TRIGGER assessment_consumption_no_delete", []).unwrap();
    conn.execute("DELETE FROM assessment_lockbox_consumptions WHERE project=?1", [project_id]).unwrap();
    let err_missing_lb = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_missing_lb.contains("has no authoritative assessment_lockbox_consumptions record"),
        "expected missing lockbox record error, got: {err_missing_lb}"
    );

    // 7. Boundary F: Untrusted self-owned manifest where controller release receipt is missing fails closed.
    conn.execute("DROP TRIGGER IF EXISTS controller_release_receipts_no_delete", []).unwrap();
    conn.execute("DELETE FROM controller_release_receipts WHERE project=?1", [project_id]).unwrap();
    let err_no_receipt = verify_bundle_with_state(&bundle_dir, false, Some(root)).unwrap_err();
    assert!(
        err_no_receipt.contains("lacks verified user approval"),
        "unauthenticated self-owned bundle manifest must be rejected, got: {err_no_receipt}"
    );
}

// ============================================================================
// GATE 7: Crash Recovery Fails Closed on Corrupt/Missing Validation Score
// ============================================================================
#[test]
fn test_s4_gate_crash_window_missing_or_corrupt_val_score_fails_closed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("s4_crash_val_dataset.csv");
    let (_ds_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let project_id = "proj_s4_val_corruption";

    let problem_spec = CampaignProblemSpec {
        project_id: project_id.into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let spec_path = root.join("problem_spec.json");
    fs::write(&spec_path, serde_json::to_string_pretty(&problem_spec).unwrap()).unwrap();

    let bin_path = env!("CARGO_BIN_EXE_mlrescurix-harness");

    // Phase 1: Real external SIGKILL at crash window
    spawn_and_sigkill_at_crash_window(bin_path, root, &spec_path);

    // Tamper: Delete candidate_evaluations row
    let db_path = root.join("controller/state.sqlite");
    assert!(db_path.is_file());
    let conn = Connection::open(&db_path).unwrap();
    conn.execute("DELETE FROM candidate_evaluations WHERE project=?1", [project_id]).unwrap();

    // Phase 2: Resumption must fail closed
    let child2 = std::process::Command::new(bin_path)
        .arg("autonomous-create")
        .arg("--state")
        .arg(root)
        .arg("--spec")
        .arg(&spec_path)
        .arg("--offline-fixture")
        .output()
        .expect("must run child");
    assert!(!child2.status.success(), "resumption must fail closed when validation score is missing");
    let err_str = String::from_utf8_lossy(&child2.stderr);
    assert!(
        err_str.contains("cannot recover validation score") || err_str.contains("QueryReturnedNoRows"),
        "expected error regarding missing validation score, got: {err_str}"
    );
}

// ============================================================================
// GATE 8: Unified Resource Status Missing DB & Unverified Remote Quota
// ============================================================================
#[test]
fn test_s4_gate_resource_status_missing_db_and_unverified_kaggle() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    // 1. Missing state DB: must return null and accounting_verified=false, NOT fabricated 0
    let status_empty = get_unified_resource_status(root, "nonexistent_proj").unwrap();
    assert!(status_empty["total_spend_millicents"].is_null());
    assert!(status_empty["prompt_tokens"].is_null());
    assert!(status_empty["completion_tokens"].is_null());
    assert!(status_empty["total_tokens"].is_null());
    assert!(status_empty["search_attempts"].is_null());
    assert_eq!(status_empty["accounting_verified"], false);
    assert_eq!(status_empty["accounting_status"], "missing_state_db");
    assert_eq!(status_empty["kaggle_status"], "unconnected_or_missing_pool");
    assert_eq!(status_empty["kaggle_remote_usage_verified"], false);

    // 2. Kaggle pool with accounts: kaggle_hours_used must be null (never fabricated as 30h allowance)
    let pool = KagglePoolManager::new(root, true).unwrap();
    let anchor = KaggleAccount::new("pool_user", "env:KAGGLE_POOL_KEY", true).unwrap();
    pool.register_account(anchor).unwrap();

    let status_with_pool = get_unified_resource_status(root, "pool_proj").unwrap();
    assert_eq!(status_with_pool["kaggle_accounts_count"], 1);
    assert!(status_with_pool["kaggle_hours_used"].is_null(), "must NOT fabricate usage from 30h allowance");
    assert_eq!(status_with_pool["kaggle_remote_usage_verified"], false);
    assert_eq!(status_with_pool["kaggle_status"], "registered_accounts_present_remote_unverified");
}

#[test]
fn uncertain_resources_and_search_evidence_remain_accountable() {
    use mlrescurix_harness::resource_authority::settle_search_attempt_with_evidence;
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("authority.sqlite");
    let project = "liability";
    assert!(admit_kaggle_job(&db, project, "job1", "account", "slug", "gpu", 8.0, 10.0).unwrap().is_some());
    assert!(settle_kaggle_job(&db, project, "job1", "success", -1.0, None).is_err());
    settle_kaggle_job(&db, project, "job1", "outcome_unknown", 0.0, None).unwrap();
    assert!(admit_kaggle_job(&db, project, "job2", "account", "slug", "gpu", 3.0, 10.0).unwrap().is_none());
    assert!(admit_token_reservation(&db, project, "bad", "worker", "model", 1, 1, -1, 100, 100).is_err());
    assert!(admit_token_reservation(&db, project, "r1", "worker", "model", 40, 20, 10, 100, 100).unwrap().is_some());
    assert!(settle_token_reservation(&db, project, "r1", "outcome_unknown", 0, 0, 0, None).is_err());
    settle_token_reservation(&db, project, "r1", "outcome_unknown", 40, 20, 10, None).unwrap();
    assert!(admit_token_reservation(&db, project, "r2", "worker", "model", 30, 20, 1, 100, 100).unwrap().is_none());
    let attempt = admit_search_attempt(&db, project, "topic", "topic", "provider", 10).unwrap().unwrap();
    settle_search_attempt_with_evidence(&db, project, &attempt, "success", None, 0, Some("first")).unwrap();
    assert!(settle_search_attempt_with_evidence(&db, project, &attempt, "success", None, 0, Some("changed")).is_err());
}
