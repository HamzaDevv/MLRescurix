use std::{
    fs,
    path::Path,
};
use sha2::{Digest, Sha256};
use mlrescurix_harness::{
    artifacts::ArtifactRegistry,
    cas::ContentAddressableStorage,
    core::{Policy, Request, Operation},
    execution,
    Runtime,
};

fn policy() -> Policy {
    let mut budget = std::collections::BTreeMap::new();
    budget.insert("calls".into(), 1000);
    budget.insert("tokens".into(), 100_000);
    budget.insert("wall_ms".into(), 60_000);
    let mut capacity = std::collections::BTreeMap::new();
    capacity.insert("workers".into(), 4);
    Policy { max_tasks: 16, budget, capacity }
}

fn setup_project(root: &Path, project: &str) {
    let rt = Runtime::open(root).unwrap();
    let req = Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P2 artifact registry test".into(),
            finalization_id: format!("f-{project}"),
            policy: policy(),
        },
    };
    let receipt = rt.command(req).unwrap();
    assert_eq!(receipt.status, "applied");
}

#[test]
fn p2_test_migration_v9_to_v10_with_populated_state() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");

    // Initialize state and migrate all migrations
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-mig");
    execution::migrate(&root).unwrap();

    // Roll back to version 9 by dropping P2 tables to simulate a pre-P2 (v9) database
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute_batch(
        "PRAGMA foreign_keys=OFF;
         DROP TABLE IF EXISTS durable_events;
         DROP TABLE IF EXISTS project_selections;
         DROP TABLE IF EXISTS artifact_annotations;
         DROP TABLE IF EXISTS artifact_lineage;
         DROP TABLE IF EXISTS artifact_versions;
         DROP TABLE IF EXISTS artifacts;
         DROP TABLE IF EXISTS blobs;
         PRAGMA user_version=9;
         PRAGMA foreign_keys=ON;"
    ).unwrap();

    // Populate legacy tables before inspecting
    let hash_m = "a".repeat(64);
    let hash_d = "b".repeat(64);
    db.execute(
        "INSERT INTO branches(project, id, base_hash, code_hash) VALUES('p-mig', 'b1', 'h1', 'h2')",
        [],
    ).unwrap();
    db.execute(
        "INSERT INTO tasks(project, id, branch, data) VALUES('p-mig', 't1', 'b1', '{}')",
        [],
    ).unwrap();
    db.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms)
         VALUES('job_pre', 'p-mig', 't1', 1, '{}', 'succeeded', 100)",
        [],
    ).unwrap();
    db.execute(
        "INSERT OR IGNORE INTO managed_artifacts(project, job, name, bytes, sha256, pinned)
         VALUES('p-mig', 'job_pre', 'legacy_model.joblib', 2048, ?1, 1)",
        [&hash_m],
    ).unwrap();
    db.execute(
        "INSERT OR IGNORE INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES('p-mig', 'dataset_pre', '/data/pre.csv', 4096, ?1, 123456789)",
        [&hash_d],
    ).unwrap();
    drop(db);

    // Run migration (which executes schema_m7_artifacts.sql from v9 to v10)
    execution::migrate(&root).unwrap();

    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let version: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 10);

    let fk_violations: Vec<String> = db.prepare("PRAGMA foreign_key_check").unwrap()
        .query_map([], |r| Ok(format!("{}:{}", r.get::<_, String>(0)?, r.get::<_, String>(2)?)))
        .unwrap()
        .map(|r| r.unwrap())
        .collect();
    assert!(fk_violations.is_empty(), "Foreign key violations found: {:?}", fk_violations);

    // Verify backfill from managed_artifacts
    let blob_count: i64 = db.query_row("SELECT count(*) FROM blobs WHERE sha256=?1", [&hash_m], |r| r.get(0)).unwrap();
    assert_eq!(blob_count, 1);
    let art_count: i64 = db.query_row("SELECT count(*) FROM artifacts WHERE project='p-mig' AND artifact_id='legacy_model.joblib'", [], |r| r.get(0)).unwrap();
    assert_eq!(art_count, 1);
    let ver_count: i64 = db.query_row("SELECT count(*) FROM artifact_versions WHERE project='p-mig' AND artifact_id='legacy_model.joblib'", [], |r| r.get(0)).unwrap();
    assert_eq!(ver_count, 1);

    // Verify backfill from scientific_datasets
    let ds_blob: i64 = db.query_row("SELECT count(*) FROM blobs WHERE sha256=?1", [&hash_d], |r| r.get(0)).unwrap();
    assert_eq!(ds_blob, 1);
    let ds_art: i64 = db.query_row("SELECT count(*) FROM artifacts WHERE project='p-mig' AND artifact_id='dataset_pre' AND kind='dataset'", [], |r| r.get(0)).unwrap();
    assert_eq!(ds_art, 1);

    // Re-running migration must be idempotent and harmless
    drop(db);
    execution::migrate(&root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let version_after: i64 = db.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version_after, 10);
}

#[test]
fn p2_test_cas_blob_storage_deduplication_and_permissions() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-cas");

    let cas = ContentAddressableStorage::new(&root);
    let payload = b"critical candidate model weights bytes";
    let sha = format!("{:x}", Sha256::digest(payload));

    let receipt1 = cas.store_blob(payload, "application/octet-stream", Some("p-cas")).unwrap();
    assert_eq!(receipt1.sha256, sha);
    assert_eq!(receipt1.bytes, payload.len() as u64);
    assert!(receipt1.path.is_file());

    // Check permissions are read-only (0o444)
    use std::os::unix::fs::PermissionsExt;
    let perms = fs::metadata(&receipt1.path).unwrap().permissions();
    assert_eq!(perms.mode() & 0o777, 0o444);

    // Check deduplication
    let receipt2 = cas.store_blob(payload, "application/octet-stream", Some("p-cas")).unwrap();
    assert_eq!(receipt2.sha256, sha);
    assert_eq!(receipt2.path, receipt1.path);

    // Read back and verify hash integrity
    let bytes = cas.read_blob(&sha).unwrap();
    assert_eq!(bytes, payload);
}

#[test]
fn p2_test_two_tier_summary_bounds_and_authority() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    execution::migrate(&root).unwrap();
    setup_project(&root, "p-bounds");

    let registry = ArtifactRegistry::new(&root);

    // 1. Register candidate artifact
    registry.register_artifact("p-bounds", "cand_log", "code", "agent:worker", None).unwrap();

    let dummy_sha = "c".repeat(64);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT OR IGNORE INTO blobs VALUES(?1, 100, 'text/plain', 1000)", [&dummy_sha]).unwrap();
    drop(db);

    // 2. Reject index_digest > 512 bytes
    let huge_digest = "x".repeat(513);
    let err_digest = registry.publish_version(
        "p-bounds", "cand_log", &dummy_sha, &huge_digest, None, 1, None, None,
    ).unwrap_err();
    assert!(err_digest.contains("index_digest exceeds bound of 512"));

    // 3. Reject summary_text > 4096 bytes
    let huge_summary = "y".repeat(4097);
    let err_summary = registry.publish_version(
        "p-bounds", "cand_log", &dummy_sha, "valid digest", Some(&huge_summary), 1, None, None,
    ).unwrap_err();
    assert!(err_summary.contains("summary_text exceeds bound of 4096"));

    // 4. Valid publish succeeds
    let ver_id = registry.publish_version(
        "p-bounds", "cand_log", &dummy_sha, "[cand_log] valid digest", Some("valid summary"), 1, None, None,
    ).unwrap();
    assert!(!ver_id.is_empty());

    // 5. Official scorecard authority: agent cannot publish scorecard
    registry.register_artifact("p-bounds", "sc_fraud", "scorecard", "agent:hacker", None).unwrap();
    let err_auth = registry.publish_version(
        "p-bounds", "sc_fraud", &dummy_sha, "fraud score", None, 1, None, None,
    ).unwrap_err();
    assert!(err_auth.contains("unauthorized attempt to publish scorecard artifact: requires evaluator:trusted"));

    // 6. Trusted evaluator can publish scorecard
    registry.register_artifact("p-bounds", "sc_legit", "scorecard", "evaluator:trusted", None).unwrap();
    let legit_ver = registry.publish_version(
        "p-bounds", "sc_legit", &dummy_sha, "legit score 0.85", None, 1, None, None,
    ).unwrap();
    assert!(!legit_ver.is_empty());
}

#[test]
fn p2_test_lineage_annotations_and_apex_selection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    execution::migrate(&root).unwrap();
    setup_project(&root, "p-lin");

    let registry = ArtifactRegistry::new(&root);

    let dummy_sha = "d".repeat(64);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT OR IGNORE INTO blobs VALUES(?1, 200, 'text/plain', 1000)", [&dummy_sha]).unwrap();
    drop(db);

    registry.register_artifact("p-lin", "ds1", "data_profile", "system:dataset", None).unwrap();
    let v_ds = registry.publish_version("p-lin", "ds1", &dummy_sha, "dataset v1", None, 1, None, None).unwrap();

    registry.register_artifact("p-lin", "c1", "code", "agent:engineer", None).unwrap();
    let v_c1 = registry.publish_version("p-lin", "c1", &dummy_sha, "candidate c1", None, 1, None, None).unwrap();

    registry.register_artifact("p-lin", "s1", "scorecard", "evaluator:trusted", None).unwrap();
    let v_s1 = registry.publish_version("p-lin", "s1", &dummy_sha, "scorecard s1", None, 1, None, None).unwrap();

    // Lineage recording
    registry.record_lineage("p-lin", &v_c1, &v_ds, "derived_from", None).unwrap();
    registry.record_lineage("p-lin", &v_s1, &v_c1, "evaluates", None).unwrap();

    let lineage_c1 = registry.get_lineage("p-lin", &v_c1).unwrap();
    assert_eq!(lineage_c1.len(), 2);

    // Annotations
    let note_id = registry.annotate("p-lin", &v_c1, "critic", "Verified holdout isolation").unwrap();
    assert!(!note_id.is_empty());
    let notes = registry.get_annotations("p-lin", &v_c1).unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].author, "critic");
    assert_eq!(notes[0].note_text, "Verified holdout isolation");

    // Apex Crown
    let apex_id = registry.set_apex_selection("p-lin", &v_c1, &v_s1, "prot_123", "Highest verified F1").unwrap();
    assert!(!apex_id.is_empty());
    let current_apex = registry.get_current_apex("p-lin").unwrap().unwrap();
    assert_eq!(current_apex.candidate_version_id, v_c1);
    assert_eq!(current_apex.scorecard_version_id, v_s1);
    assert!(current_apex.is_apex);

    // Durable events journal
    let events = registry.get_events("p-lin", 0, 100).unwrap();
    assert!(events.len() >= 6);
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    assert!(event_types.contains(&"artifact.version_published"));
    assert!(event_types.contains(&"lineage.edge_created"));
    assert!(event_types.contains(&"artifact.annotated"));
    assert!(event_types.contains(&"project.apex_crowned"));
}

#[test]
fn p2_test_directory_manifest_canonicalization_and_symlink_rejection() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-dir");

    let dir = temp.path().join("model_dir");
    fs::create_dir_all(dir.join("sub")).unwrap();
    fs::write(dir.join("zebra.txt"), b"z").unwrap();
    fs::write(dir.join("apple.txt"), b"a").unwrap();
    fs::write(dir.join("sub").join("nested.txt"), b"n").unwrap();

    let cas = ContentAddressableStorage::new(&root);
    let (receipt, manifest) = cas.create_directory_manifest(&dir, 10000, Some("p-dir")).unwrap();

    assert_eq!(manifest.file_count, 3);
    assert_eq!(manifest.files[0].path, "apple.txt");
    assert_eq!(manifest.files[1].path, "sub/nested.txt");
    assert_eq!(manifest.files[2].path, "zebra.txt");
    assert_eq!(receipt.media_type, "application/x-directory-manifest+json");

    // Check symlink rejection
    #[cfg(unix)]
    {
        let sym_dir = temp.path().join("sym_model_dir");
        fs::create_dir_all(&sym_dir).unwrap();
        fs::write(sym_dir.join("real.txt"), b"real").unwrap();
        std::os::unix::fs::symlink(sym_dir.join("real.txt"), sym_dir.join("link.txt")).unwrap();

        let err = cas.create_directory_manifest(&sym_dir, 10000, Some("p-dir")).unwrap_err();
        assert!(err.contains("symlink rejected in directory artifact"));
    }
}
