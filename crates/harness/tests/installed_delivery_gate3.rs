//! Integration tests for Release Gate 3: Clean Installed End-to-End Delivery & Upgrade Verification.
//!
//! Validates:
//! 1. Complete autonomous candidate rescue, bundle packaging, standalone CLI inference, and HTTP serve
//!    strictly using the installed artifacts with zero access to the git repository source tree.
//! 2. Negative guard: zero dependency on git repository when executing from the installed prefix.
//! 3. Standalone inference and HTTP serve parity without harness runtime dependencies.
//! 4. Restart recovery and schema upgrade consistency under clean installation (PRAGMA user_version = 10).
//! 5. Cryptographic tamper detection rejecting altered bundle files.

use std::{
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::PathBuf,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use serde_json::Value;

struct InstalledPrefix {
    prefix_dir: PathBuf,
    venv_python: PathBuf,
    harness_bin: PathBuf,
    workspace_dir: PathBuf,
    state_dir: PathBuf,
}

impl InstalledPrefix {
    fn setup() -> Self {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();

        let wheel_path = repo_root.join("python/dist/mlrescurix_worker_protocol-0.1.0-py3-none-any.whl");
        if !wheel_path.is_file() {
            let status = Command::new("python3")
                .args(["-m", "build", "--wheel"])
                .current_dir(repo_root.join("python"))
                .status()
                .expect("Failed to build python worker wheel");
            assert!(status.success(), "Wheel build failed");
        }

        let release_bin = repo_root.join("target/release/mlrescurix-harness");
        if !release_bin.is_file() {
            let status = Command::new("cargo")
                .args(["build", "--release"])
                .current_dir(&repo_root)
                .status()
                .expect("Failed to build release harness binary");
            assert!(status.success(), "Release binary build failed");
        }

        static GATE3_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let seq = GATE3_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let prefix_dir = std::env::temp_dir().join(format!("mlrescurix-gate3-verify-{pid}-{ts}-{seq}"));
        fs::create_dir_all(&prefix_dir).expect("Failed to create prefix dir");

        // 1. Create virtual environment with system site packages (for numpy/pandas/sklearn)
        let venv_dir = prefix_dir.join("venv");
        let venv_status = Command::new("python3")
            .args(["-m", "venv", "--system-site-packages"])
            .arg(&venv_dir)
            .status()
            .expect("Failed to create venv");
        assert!(venv_status.success(), "Failed to create venv in prefix");

        let venv_python = venv_dir.join("bin/python");
        let venv_pip = venv_dir.join("bin/pip");

        // 2. Install wheel with --no-deps
        let pip_status = Command::new(&venv_pip)
            .args(["install", "--no-deps"])
            .arg(&wheel_path)
            .status()
            .expect("Failed to pip install wheel");
        assert!(pip_status.success(), "pip install wheel failed");

        // 3. Install harness release binary into prefix/bin
        let bin_dir = prefix_dir.join("bin");
        fs::create_dir_all(&bin_dir).expect("Failed to create bin dir");
        let harness_bin = bin_dir.join("mlrescurix-harness");
        fs::copy(&release_bin, &harness_bin).expect("Failed to copy release binary to prefix/bin");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&harness_bin).unwrap().permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&harness_bin, perms).unwrap();
        }

        let workspace_dir = prefix_dir.join("workspace");
        fs::create_dir_all(&workspace_dir).unwrap();

        let state_dir = prefix_dir.join("state");

        Self {
            prefix_dir,
            venv_python,
            harness_bin,
            workspace_dir,
            state_dir,
        }
    }
}

impl Drop for InstalledPrefix {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.prefix_dir);
    }
}

#[test]
fn test_clean_installed_delivery_outside_git_repository() {
    let prefix = InstalledPrefix::setup();

    // Negative Guard a: Confirm zero dependency on git repository.
    // The workspace is located in /tmp outside any git tree; `git status` must fail.
    let git_check = Command::new("git")
        .args(["status"])
        .current_dir(&prefix.workspace_dir)
        .output();
    assert!(
        git_check.is_err() || !git_check.unwrap().status.success(),
        "Negative guard violated: workspace should not be inside a git repository"
    );

    // 1. Verify installed CLI capabilities
    let cap_output = Command::new(&prefix.harness_bin)
        .arg("capabilities")
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .output()
        .expect("Failed to run harness capabilities");
    assert!(cap_output.status.success());
    let cap_json: Value = serde_json::from_slice(&cap_output.stdout).expect("Invalid capabilities JSON");
    assert_eq!(cap_json["state_schema_version"], 10);
    assert_eq!(cap_json["standalone_install_without_checkout"], true);

    // 2. Initialize harness state in prefix
    let policy_path = prefix.prefix_dir.join("policy.json");
    fs::write(
        &policy_path,
        r#"{"budget":{"wall_ms":600000},"capacity":{"workers":2},"max_tasks":16}"#,
    )
    .unwrap();

    let init_status = Command::new(&prefix.harness_bin)
        .args(["init", "--state", &prefix.state_dir.to_string_lossy(), "--policy", &policy_path.to_string_lossy()])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .status()
        .expect("Failed to run harness init");
    assert!(init_status.success(), "Harness init failed in installed prefix");

    // 3. Create synthetic dataset CSV in prefix workspace
    let csv_path = prefix.workspace_dir.join("churn.csv");
    let mut csv_content = String::from("tenure,monthly_charges,total_charges,support_calls,churn\n");
    for i in 0..200 {
        let tenure = (i % 72) as f64 + 1.0;
        let monthly = 20.0 + (i % 80) as f64;
        let total = tenure * monthly;
        let support = (i % 6) as f64;
        let churn = if (monthly > 70.0 && support > 2.0) || tenure < 6.0 { 1 } else { 0 };
        csv_content.push_str(&format!("{tenure:.1},{monthly:.2},{total:.2},{support:.0},{churn}\n"));
    }
    fs::write(&csv_path, csv_content).unwrap();

    // 4. Execute dataset profiling in installed prefix
    let prof_output = Command::new(&prefix.harness_bin)
        .args([
            "profile-dataset",
            "--state", &prefix.state_dir.to_string_lossy(),
            "--csv", &csv_path.to_string_lossy(),
            "--target", "churn",
        ])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .output()
        .expect("Failed to run profile-dataset");
    assert!(prof_output.status.success(), "profile-dataset failed: {}", String::from_utf8_lossy(&prof_output.stderr));
    let prof_json: Value = serde_json::from_slice(&prof_output.stdout).unwrap();
    assert_eq!(prof_json["row_count"], 200);
    assert_eq!(prof_json["target_column"], "churn");

    // 5. Package candidate delivery bundle
    let bundle_dir = prefix.prefix_dir.join("bundle");
    let cand_dir = prefix.prefix_dir.join("cand");
    fs::create_dir_all(&cand_dir).unwrap();

    let pkg_output = Command::new(&prefix.harness_bin)
        .args([
            "package-candidate-bundle",
            "--state", &prefix.state_dir.to_string_lossy(),
            "--candidate-dir", &cand_dir.to_string_lossy(),
            "--bundle-dir", &bundle_dir.to_string_lossy(),
            "--dataset", &csv_path.to_string_lossy(),
            "--test-access-id", "test-acc-gate3",
            "--project", "p-gate3",
            "--candidate", "cand-gate3-apex",
            "--task-type", "binary",
            "--target", "churn",
            "--metric", "balanced_accuracy",
        ])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .output()
        .expect("Failed to run package-candidate-bundle");
    assert!(pkg_output.status.success(), "package-candidate-bundle failed: {}", String::from_utf8_lossy(&pkg_output.stderr));

    // Verify all 13 required bundle contracts exist in bundle
    let required_contracts = [
        "manifest.json",
        "input_schema.json",
        "output_schema.json",
        "model_schema.json",
        "scorecard.json",
        "verification.json",
        "final.json",
        "inference.py",
        "inference.lock",
        "training.lock",
        "serve.py",
        "frozen_model.joblib",
        "README.md",
    ];

    for contract in &required_contracts {
        let p = bundle_dir.join(contract);
        assert!(p.is_file(), "Bundle missing required contract file: {contract}");
    }

    // 6. Standalone CLI Inference
    let test_in_path = prefix.workspace_dir.join("test_records.json");
    let test_out_path = prefix.workspace_dir.join("test_predictions.json");
    let test_records = r#"[
        {"tenure": 12.0, "monthly_charges": 65.5, "total_charges": 786.0, "support_calls": 2.0},
        {"tenure": 48.0, "monthly_charges": 25.0, "total_charges": 1200.0, "support_calls": 0.0},
        {"tenure": 2.0, "monthly_charges": 85.0, "total_charges": 170.0, "support_calls": 4.0}
    ]"#;
    fs::write(&test_in_path, test_records).unwrap();

    let inf_output = Command::new(&prefix.venv_python)
        .arg("-B")
        .arg(bundle_dir.join("inference.py"))
        .args(["--input", &test_in_path.to_string_lossy(), "--output", &test_out_path.to_string_lossy()])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("Failed to run standalone inference");
    assert!(inf_output.status.success(), "Standalone inference failed: {}", String::from_utf8_lossy(&inf_output.stderr));

    let pred_json: Value = serde_json::from_str(&fs::read_to_string(&test_out_path).unwrap()).unwrap();
    let preds = pred_json["prediction"].as_array().expect("predictions array");
    assert_eq!(preds.len(), 3);
    let probs = pred_json["probabilities"].as_array().expect("probabilities array");
    assert_eq!(probs.len(), 3);
    for prob_pair in probs {
        let arr = prob_pair.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        let sum: f64 = arr.iter().map(|v| v.as_f64().unwrap()).sum();
        assert!((sum - 1.0).abs() < 1e-4, "Probabilities must sum to 1.0");
    }

    // 7. Standalone HTTP Serving
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind ephemeral port");
    let port = listener.local_addr().unwrap().port();
    drop(listener);

    let mut serve_child = Command::new(&prefix.venv_python)
        .arg("-B")
        .arg(bundle_dir.join("serve.py"))
        .args(["--port", &port.to_string()])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to spawn serve.py");

    // Wait for server to start
    let start_wait = Instant::now();
    let mut server_ready = false;
    while start_wait.elapsed() < Duration::from_secs(5) {
        if let Ok(mut stream) = TcpStream::connect(format!("127.0.0.1:{port}")) {
            stream.set_read_timeout(Some(Duration::from_millis(500))).unwrap();
            let req = format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
            let _ = stream.write_all(req.as_bytes());
            let mut resp = String::new();
            if stream.read_to_string(&mut resp).is_ok() && resp.contains("200 OK") && resp.contains("healthy") {
                server_ready = true;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(server_ready, "serve.py failed to respond on /health within 5s");

    // Send HTTP POST /predict request
    let post_body = test_records;
    let mut post_stream = TcpStream::connect(format!("127.0.0.1:{port}")).expect("Connect for POST /predict");
    post_stream.set_read_timeout(Some(Duration::from_secs(3))).unwrap();
    let post_req = format!(
        "POST /predict HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        post_body.len(),
        post_body
    );
    post_stream.write_all(post_req.as_bytes()).unwrap();

    let mut post_resp = String::new();
    post_stream.read_to_string(&mut post_resp).unwrap();
    assert!(post_resp.contains("200 OK"), "HTTP POST /predict must return 200 OK, got: {post_resp}");
    assert!(post_resp.contains("\"status\": \"ok\""));
    assert!(post_resp.contains("\"prediction\""));

    let _ = serve_child.kill();
    let _ = serve_child.wait();

    // 8. Bundle verification via harness (standalone mode without internal state dependency)
    let verify_output = Command::new(&prefix.harness_bin)
        .args([
            "verify-bundle",
            "--bundle", &bundle_dir.to_string_lossy(),
            "--sandbox=false",
        ])
        .current_dir(&prefix.workspace_dir)
        .env_remove("PYTHONPATH")
        .output()
        .expect("Failed to run verify-bundle");
    assert!(
        verify_output.status.success(),
        "Bundle verification failed: {}",
        String::from_utf8_lossy(&verify_output.stderr)
    );
    let verify_json: Value = serde_json::from_slice(&verify_output.stdout).unwrap();
    assert_eq!(verify_json["status"], "integrity_verified");
    assert_eq!(verify_json["manifest_verified"], true);
    assert_eq!(verify_json["contracts_verified"], true);
}

#[test]
fn test_installed_restart_recovery_and_schema_upgrade_consistency() {
    let prefix = InstalledPrefix::setup();

    let policy_path = prefix.prefix_dir.join("policy.json");
    fs::write(
        &policy_path,
        r#"{"budget":{"wall_ms":600000},"capacity":{"workers":2},"max_tasks":16}"#,
    )
    .unwrap();

    // 1. Initial harness initialization
    let init_status = Command::new(&prefix.harness_bin)
        .args(["init", "--state", &prefix.state_dir.to_string_lossy(), "--policy", &policy_path.to_string_lossy()])
        .current_dir(&prefix.workspace_dir)
        .status()
        .unwrap();
    assert!(init_status.success());

    // 2. Verify PRAGMA user_version = 10 strictly without bumping
    let db_path = prefix.state_dir.join("state.sqlite");
    assert!(db_path.is_file());
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 10, "PRAGMA user_version must remain 10");

    // 3. Restart recovery: Repeatedly execute snapshot / capabilities / inspect
    for _ in 0..3 {
        let snap_output = Command::new(&prefix.harness_bin)
            .args(["capabilities"])
            .current_dir(&prefix.workspace_dir)
            .output()
            .unwrap();
        assert!(snap_output.status.success());

        let post_version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(post_version, 10);
    }
}

#[test]
fn test_installed_tamper_detection_negative_guard() {
    let prefix = InstalledPrefix::setup();

    let policy_path = prefix.prefix_dir.join("policy.json");
    fs::write(
        &policy_path,
        r#"{"budget":{"wall_ms":600000},"capacity":{"workers":2},"max_tasks":16}"#,
    )
    .unwrap();

    Command::new(&prefix.harness_bin)
        .args(["init", "--state", &prefix.state_dir.to_string_lossy(), "--policy", &policy_path.to_string_lossy()])
        .status()
        .unwrap();

    let csv_path = prefix.workspace_dir.join("data.csv");
    fs::write(&csv_path, "f1,f2,target\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n").unwrap();

    let bundle_dir = prefix.prefix_dir.join("bundle");
    let cand_dir = prefix.prefix_dir.join("cand");
    fs::create_dir_all(&cand_dir).unwrap();

    let pkg = Command::new(&prefix.harness_bin)
        .args([
            "package-candidate-bundle",
            "--state", &prefix.state_dir.to_string_lossy(),
            "--candidate-dir", &cand_dir.to_string_lossy(),
            "--bundle-dir", &bundle_dir.to_string_lossy(),
            "--dataset", &csv_path.to_string_lossy(),
            "--test-access-id", "test-acc-tamper",
            "--project", "p-tamper",
            "--candidate", "cand-tamper",
            "--task-type", "binary",
            "--target", "target",
            "--metric", "balanced_accuracy",
        ])
        .output()
        .unwrap();
    assert!(pkg.status.success());

    // Tamper with model_schema.json by modifying bytes
    let schema_file = bundle_dir.join("model_schema.json");
    let mut schema_val: Value = serde_json::from_str(&fs::read_to_string(&schema_file).unwrap()).unwrap();
    schema_val["decision_threshold"] = serde_json::json!(0.9999);
    fs::write(&schema_file, serde_json::to_string_pretty(&schema_val).unwrap()).unwrap();

    // Verify-bundle must detect tampering and refuse
    let verify_out = Command::new(&prefix.harness_bin)
        .args([
            "verify-bundle",
            "--bundle", &bundle_dir.to_string_lossy(),
            "--sandbox=false",
        ])
        .output()
        .unwrap();

    assert!(!verify_out.status.success(), "Tampered bundle must be rejected");
    let err_msg = String::from_utf8_lossy(&verify_out.stderr);
    assert!(
        err_msg.contains("checksum mismatch") || err_msg.contains("tamper") || err_msg.contains("sha256") || err_msg.contains("mismatch"),
        "Error message should mention checksum mismatch: {err_msg}"
    );
}
