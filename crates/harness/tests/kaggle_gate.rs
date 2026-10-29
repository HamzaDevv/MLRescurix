use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use mlrescurix_harness::kaggle::{
    scrub_staged_content, scrub_staged_directory, KaggleAccount, KaggleApi, KaggleLifecycleState,
    KagglePoolManager,
};
use sha2::{Digest, Sha256};

fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

struct MockKaggleApi {
    pub pushed_slugs: Mutex<Vec<String>>,
    pub status_sequence: Mutex<Vec<String>>,
    pub fail_push: AtomicBool,
    pub fail_download: AtomicBool,
    pub output_files: Mutex<HashMap<String, Vec<u8>>>,
    pub competition_submissions: Mutex<Vec<(String, String)>>,
}

impl MockKaggleApi {
    fn new() -> Self {
        Self {
            pushed_slugs: Mutex::new(Vec::new()),
            status_sequence: Mutex::new(vec!["queued".into(), "running".into(), "complete".into()]),
            fail_push: AtomicBool::new(false),
            fail_download: AtomicBool::new(false),
            output_files: Mutex::new(HashMap::new()),
            competition_submissions: Mutex::new(Vec::new()),
        }
    }
}

impl KaggleApi for MockKaggleApi {
    fn push_kernel(&self, account: &KaggleAccount, slug: &str, _staged_dir: &Path) -> Result<String, String> {
        if self.fail_push.load(Ordering::SeqCst) {
            return Err("mock remote error: push kernel failed (network timeout)".into());
        }
        self.pushed_slugs.lock().unwrap().push(format!("{}:{}", account.username, slug));
        Ok(format!("kernel '{slug}' pushed successfully"))
    }

    fn get_kernel_status(&self, _account: &KaggleAccount, _slug: &str) -> Result<String, String> {
        let mut seq = self.status_sequence.lock().unwrap();
        if seq.is_empty() {
            Ok("complete".into())
        } else {
            Ok(seq.remove(0))
        }
    }

    fn download_kernel_output(&self, _account: &KaggleAccount, _slug: &str, dest_dir: &Path) -> Result<(), String> {
        if self.fail_download.load(Ordering::SeqCst) {
            return Err("mock download error: output extraction failed".into());
        }
        let files = self.output_files.lock().unwrap();
        for (rel_name, bytes) in files.iter() {
            let target = dest_dir.join(rel_name);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(&target, bytes).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    fn submit_competition(&self, account: &KaggleAccount, competition: &str, file: &Path, message: &str) -> Result<String, String> {
        self.competition_submissions.lock().unwrap().push((account.username.clone(), competition.to_string()));
        Ok(format!("mock submission accepted for competition '{competition}' with file '{}' message '{message}'", file.display()))
    }
}

#[test]
fn test_kaggle_account_credential_ref_and_resolution() {
    let temp = tempfile::tempdir().unwrap();

    // 1. Plain text credential rejected
    let err = KaggleAccount::new("user1", "plain_api_key_12345", true).unwrap_err();
    assert!(err.contains("credential_ref must start with 'env:' or 'file:'"));

    // 2. Env credential resolution
    unsafe {
        std::env::set_var("TEST_KAGGLE_SECRET_KEY", "env_secret_token_abc");
    }
    let acct_env = KaggleAccount::new("anchor_user", "env:TEST_KAGGLE_SECRET_KEY", true).unwrap();
    let resolved_env = acct_env.resolve_credential(None).unwrap();
    assert_eq!(resolved_env, "env_secret_token_abc");

    // 3. File credential resolution
    let key_file = temp.path().join("kaggle.key");
    fs::write(&key_file, "file_secret_token_xyz\n").unwrap();
    let acct_file = KaggleAccount::new("secondary_user", &format!("file:{}", key_file.display()), false).unwrap();
    let resolved_file = acct_file.resolve_credential(None).unwrap();
    assert_eq!(resolved_file, "file_secret_token_xyz");

    // 4. Relative file resolution with storage_dir
    let rel_file = temp.path().join("sub").join("secret.token");
    fs::create_dir_all(rel_file.parent().unwrap()).unwrap();
    fs::write(&rel_file, "nested_token_123").unwrap();
    let acct_rel = KaggleAccount::new("worker_rel", "file:sub/secret.token", false).unwrap();
    let resolved_rel = acct_rel.resolve_credential(Some(temp.path())).unwrap();
    assert_eq!(resolved_rel, "nested_token_123");
}

#[test]
fn test_anti_ban_single_anchor_and_secondary_isolation() {
    let temp = tempfile::tempdir().unwrap();
    let pool = KagglePoolManager::new(temp.path(), true).unwrap();

    let anchor = KaggleAccount::new("anchor_main", "env:DUMMY_KEY", true).unwrap();
    pool.register_account(anchor.clone()).unwrap();

    // Enforce single anchor: registering another anchor must fail
    let second_anchor = KaggleAccount::new("anchor_impostor", "env:DUMMY_KEY2", true).unwrap();
    let err = pool.register_account(second_anchor).unwrap_err();
    assert!(err.contains("anti-ban violation: another anchor account 'anchor_main' is already registered"));

    // Register secondary compute worker
    let secondary = KaggleAccount::new("compute_worker_01", "env:DUMMY_KEY3", false).unwrap();
    pool.register_account(secondary.clone()).unwrap();

    // Verify secondary account CANNOT submit to competition
    let mock_api = MockKaggleApi::new();
    let sub_file = temp.path().join("sub.csv");
    fs::write(&sub_file, "id,target\n1,0\n").unwrap();

    let sub_err = pool.submit_competition(&mock_api, &secondary, "titanic", &sub_file, "test submit").unwrap_err();
    assert!(sub_err.contains("anti-ban violation: secondary account cannot submit to competitions"));

    // Verify secondary account CANNOT stage a competition job
    let mut code_files = HashMap::new();
    code_files.insert("train.py".into(), "print('training model')\n".into());
    let stage_err = pool.stage_job("job-comp-01", &secondary, "gpu", true, &code_files).unwrap_err();
    assert!(stage_err.contains("anti-ban violation: secondary account cannot submit to competitions"));

    // Verify Anchor CAN submit to competition
    let sub_ok = pool.submit_competition(&mock_api, &anchor, "titanic", &sub_file, "valid anchor submit").unwrap();
    assert!(sub_ok.contains("mock submission accepted"));
}

#[test]
fn test_staging_scrubber_credential_leak_rejection() {
    let temp = tempfile::tempdir().unwrap();

    // 1. Direct content scrubbing
    assert!(scrub_staged_content("import os\napi_key = 'sk-1234567890abcdef'\n").is_err());
    assert!(scrub_staged_content("DEEPSEEK_API_KEY = 'secret'\n").is_err());
    assert!(scrub_staged_content("OPENROUTER_API_KEY = 'secret'\n").is_err());
    assert!(scrub_staged_content("import torch\nmodel = torch.nn.Linear(10, 2)\n").is_ok());

    // 2. Directory scrubbing
    let stage_dir = temp.path().join("stage_scrub");
    fs::create_dir_all(&stage_dir).unwrap();
    fs::write(stage_dir.join("model.py"), "import torch\n").unwrap();
    fs::write(stage_dir.join("config.py"), "sk-live-secret-token\n").unwrap();

    let err = scrub_staged_directory(&stage_dir).unwrap_err();
    assert!(err.contains("security violation: potential API key (sk-...) detected"));
}

#[test]
fn test_fenced_atomic_leases_rejects_stale_collection() {
    let temp = tempfile::tempdir().unwrap();
    let pool = KagglePoolManager::new(temp.path(), true).unwrap();
    let mock_api = MockKaggleApi::new();

    let anchor = KaggleAccount::new("anchor_user", "env:DUMMY_KEY", true).unwrap();
    pool.register_account(anchor.clone()).unwrap();

    let mut code_files = HashMap::new();
    code_files.insert("run.py".into(), "print('job execution')\n".into());
    let job = pool.stage_job("job-fence-01", &anchor, "gpu", false, &code_files).unwrap();
    assert_eq!(job.state, KaggleLifecycleState::Staged);

    // Prepare mock output
    let weights_bytes = b"model_neural_weights_data";
    let weights_sha = compute_sha256(weights_bytes);
    let manifest_bytes = serde_json::to_vec(&serde_json::json!({
        "files": {
            "weights.pt": weights_sha
        }
    })).unwrap();

    mock_api.output_files.lock().unwrap().insert("weights.pt".into(), weights_bytes.to_vec());
    mock_api.output_files.lock().unwrap().insert("manifest.json".into(), manifest_bytes);

    // 1. Attempt collection with stale/invalid lease token
    let dest_dir = temp.path().join("out_stale");
    let err = pool.collect_and_verify_job(&mock_api, "job-fence-01", "invalid-superseded-lease-token", &dest_dir).unwrap_err();
    assert!(err.contains("stale lease"));

    // 2. Attempt collection with valid active lease token -> succeeds
    let valid_dest = temp.path().join("out_valid");
    let verified_job = pool.collect_and_verify_job(&mock_api, "job-fence-01", &job.lease_token, &valid_dest).unwrap();
    assert_eq!(verified_job.state, KaggleLifecycleState::Verified);
    assert!(valid_dest.join("weights.pt").is_file());
}

#[test]
fn test_checksum_verification_against_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let pool = KagglePoolManager::new(temp.path(), true).unwrap();
    let mock_api = MockKaggleApi::new();

    let anchor = KaggleAccount::new("anchor_user", "env:DUMMY_KEY", true).unwrap();
    pool.register_account(anchor.clone()).unwrap();

    let mut code_files = HashMap::new();
    code_files.insert("run.py".into(), "print('job')\n".into());
    let job = pool.stage_job("job-tamper-01", &anchor, "gpu", false, &code_files).unwrap();

    // Mock output with tampered file (actual hash != expected manifest hash)
    let weights_bytes = b"real_weights_data";
    let manifest_bytes = serde_json::to_vec(&serde_json::json!({
        "files": {
            "weights.pt": "0000000000000000000000000000000000000000000000000000000000000000"
        }
    })).unwrap();

    mock_api.output_files.lock().unwrap().insert("weights.pt".into(), weights_bytes.to_vec());
    mock_api.output_files.lock().unwrap().insert("manifest.json".into(), manifest_bytes);

    let dest_dir = temp.path().join("out_tamper");
    let err = pool.collect_and_verify_job(&mock_api, "job-tamper-01", &job.lease_token, &dest_dir).unwrap_err();
    assert!(err.contains("checksum mismatch"));
}

#[test]
fn test_quota_pool_scheduling_and_exhaustion() {
    let temp = tempfile::tempdir().unwrap();
    let pool = KagglePoolManager::new(temp.path(), true).unwrap();

    let mut acct1 = KaggleAccount::new("acct_low_quota", "env:DUMMY_1", true).unwrap();
    acct1.gpu_hours_remaining = 2.0;
    pool.register_account(acct1).unwrap();

    let mut acct2 = KaggleAccount::new("acct_high_quota", "env:DUMMY_2", false).unwrap();
    acct2.gpu_hours_remaining = 28.0;
    pool.register_account(acct2).unwrap();

    // 1. Lease for compute job (non-competition) chooses account with highest quota (acct2)
    let leased = pool.lease_worker_account("gpu", 5.0, false).unwrap();
    assert_eq!(leased.username, "acct_high_quota");
    assert_eq!(leased.gpu_hours_remaining, 23.0);

    // 2. Requesting 30 hours exceeds remaining quota of all accounts -> quota exhausted
    let err = pool.lease_worker_account("gpu", 30.0, false).unwrap_err();
    assert!(err.contains("quota exhausted"));

    // 3. Requesting competition job requires anchor account
    let comp_leased = pool.lease_worker_account("gpu", 1.5, true).unwrap();
    assert_eq!(comp_leased.username, "acct_low_quota");
    assert!(comp_leased.is_anchor);
}

#[test]
fn test_end_to_end_mocked_lifecycle_and_failure_injection() {
    let temp = tempfile::tempdir().unwrap();
    let pool = KagglePoolManager::new(temp.path(), true).unwrap();
    let mock_api = MockKaggleApi::new();

    let anchor = KaggleAccount::new("anchor_user", "env:DUMMY_KEY", true).unwrap();
    pool.register_account(anchor.clone()).unwrap();

    let mut code_files = HashMap::new();
    code_files.insert("train.py".into(), "print('stage')\n".into());

    // 1. Failure injection: remote push failure
    mock_api.fail_push.store(true, Ordering::SeqCst);
    let _job_fail = pool.stage_job("job-push-fail", &anchor, "gpu", false, &code_files).unwrap();
    let push_err = pool.dispatch_job(&mock_api, "job-push-fail").unwrap_err();
    assert!(push_err.contains("push kernel failed"));

    // Restore push
    mock_api.fail_push.store(false, Ordering::SeqCst);

    *mock_api.status_sequence.lock().unwrap() = vec!["running".into(), "complete".into()];
    let job = pool.stage_job("job-e2e-01", &anchor, "gpu", false, &code_files).unwrap();
    assert_eq!(job.state, KaggleLifecycleState::Staged);

    let dispatched = pool.dispatch_job(&mock_api, "job-e2e-01").unwrap();
    assert_eq!(dispatched.state, KaggleLifecycleState::Queued);

    // Poll 1: running
    let polled1 = pool.poll_job(&mock_api, "job-e2e-01").unwrap();
    assert_eq!(polled1.state, KaggleLifecycleState::Running);

    // Poll 2: complete -> transitions to collecting
    let polled2 = pool.poll_job(&mock_api, "job-e2e-01").unwrap();
    assert_eq!(polled2.state, KaggleLifecycleState::Collecting);

    // Prepare mock artifacts
    let weights_bytes = b"final_verified_checkpoint_weights";
    let weights_sha = compute_sha256(weights_bytes);
    let manifest_bytes = serde_json::to_vec(&serde_json::json!({
        "files": {
            "weights.pt": weights_sha
        }
    })).unwrap();

    mock_api.output_files.lock().unwrap().insert("weights.pt".into(), weights_bytes.to_vec());
    mock_api.output_files.lock().unwrap().insert("manifest.json".into(), manifest_bytes);

    let out_dir = temp.path().join("e2e_out");
    let verified = pool.collect_and_verify_job(&mock_api, "job-e2e-01", &job.lease_token, &out_dir).unwrap();
    assert_eq!(verified.state, KaggleLifecycleState::Verified);

    let disk_weights = fs::read(out_dir.join("weights.pt")).unwrap();
    assert_eq!(disk_weights, weights_bytes);
}
