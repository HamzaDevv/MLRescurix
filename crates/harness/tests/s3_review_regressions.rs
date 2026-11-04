use mlrescurix_harness::campaign::compute_workspace_source_sha256;
use std::fs;

#[test]
fn subprocess_result_is_rejected_after_lease_loss() {
    use mlrescurix_harness::campaign::{acquire_campaign_lease, run_cmd_with_heartbeats};
    use rusqlite::Connection;
    use std::{process::Command, thread, time::Duration};
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("state.sqlite");
    let mut conn = Connection::open(&db).unwrap();
    let (lease, epoch) = acquire_campaign_lease(&mut conn, "p", "owner").unwrap();
    let db2 = db.clone();
    let change = thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        Connection::open(db2).unwrap().execute("UPDATE campaign_leases SET status='released' WHERE project='p'", []).unwrap();
    });
    let result = run_cmd_with_heartbeats(
        Command::new("python3").args(["-c", "import time; time.sleep(1.0)"]),
        &db, "p", &lease, epoch,
    );
    change.join().unwrap();
    assert!(result.is_err(), "a stale subprocess result must not be accepted");
}

#[test]
fn nested_inference_and_serve_modules_are_bound_to_source_identity() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("solution.py"), "import sub.inference\n").unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    for name in ["inference.py", "serve.py"] {
        let path = dir.path().join("sub").join(name);
        fs::write(&path, "value = 1\n").unwrap();
        let before = compute_workspace_source_sha256(dir.path()).unwrap();
        fs::write(&path, "value = 2\n").unwrap();
        let after = compute_workspace_source_sha256(dir.path()).unwrap();
        assert_ne!(before, after, "{name} must be included in the source identity");
    }
}

#[test]
fn program_json_and_asset_modifications_are_bound_to_source_identity() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("solution.py"), "print('hello')\n").unwrap();
    let initial_hash = compute_workspace_source_sha256(dir.path()).unwrap();

    // Adding program.json must change source identity
    let prog_path = dir.path().join("program.json");
    fs::write(&prog_path, r#"{"schema_version": "mlrescurix.program.v1", "entrypoint": "solution.py"}"#).unwrap();
    let with_prog = compute_workspace_source_sha256(dir.path()).unwrap();
    assert_ne!(initial_hash, with_prog, "program.json must be bound to source identity");

    // Modifying program.json must change source identity
    fs::write(&prog_path, r#"{"schema_version": "mlrescurix.program.v1", "entrypoint": "other.py"}"#).unwrap();
    let modified_prog = compute_workspace_source_sha256(dir.path()).unwrap();
    assert_ne!(with_prog, modified_prog, "modified program.json must alter source identity");

    // Adding declared non-code asset must change source identity
    let asset_path = dir.path().join("model_config.json");
    fs::write(&asset_path, r#"{"learning_rate": 0.01}"#).unwrap();
    let with_asset = compute_workspace_source_sha256(dir.path()).unwrap();
    assert_ne!(modified_prog, with_asset, "declared assets must be bound to source identity");
}
