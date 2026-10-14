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
    assert!(err_auth.contains("scorecard publication is restricted to the trusted evaluator API"));

    // A caller cannot self-assert evaluator authority through the general API.
    let forged = registry.register_artifact("p-bounds", "sc_legit", "scorecard", "evaluator:trusted", None).unwrap_err();
    assert!(forged.contains("reserved for the trusted evaluator API"));
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

    registry.register_artifact("p-lin", "s1", "scorecard", "agent:critic", None).unwrap();
    assert!(registry.publish_version("p-lin", "s1", &dummy_sha, "claimed scorecard", None, 1, None, None).unwrap_err().contains("trusted evaluator API"));
    // Fixture a scorecard row as if committed by the private evaluator adapter.
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT INTO artifacts(project,artifact_id,kind,producer_authority,created_ms,updated_ms) VALUES('p-lin','trusted_score','scorecard','evaluator:trusted',1,1)", []).unwrap();
    db.execute("INSERT INTO artifact_versions(project,version_id,artifact_id,blob_sha256,index_digest,created_ms) VALUES('p-lin','v_score','trusted_score',?1,'trusted evaluator receipt',1)", [&dummy_sha]).unwrap();
    db.execute("UPDATE artifacts SET current_version_id='v_score' WHERE project='p-lin' AND artifact_id='trusted_score'", []).unwrap();
    let v_s1 = "v_score".to_string();

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

#[test]
fn p23_queries_use_bounded_read_only_pool_and_pin_events() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-query");
    execution::migrate(&root).unwrap();
    let registry = ArtifactRegistry::new(&root);
    registry.register_artifact("p-query", "a1", "code", "agent:worker", None).unwrap();
    let digest = "e".repeat(64);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT OR IGNORE INTO blobs VALUES(?1, 1, 'text/plain', 1)", [&digest]).unwrap();
    drop(db);
    let version = registry.publish_version("p-query", "a1", &digest, "bounded index", Some("bounded summary"), 1, None, None).unwrap();
    assert_eq!(registry.get_artifact("p-query", "a1").unwrap().unwrap().current_version_id.as_deref(), Some(version.as_str()));
    assert_eq!(registry.get_version("p-query", &version).unwrap().unwrap().summary_text.as_deref(), Some("bounded summary"));
    assert_eq!(registry.list_artifacts("p-query", None, 1).unwrap().len(), 1);
    assert_eq!(registry.list_versions("p-query", "a1").unwrap().len(), 1);
    assert!(registry.pin("p-query", "a1").is_ok());
    assert!(registry.unpin("p-query", "a1").is_ok());

    let subscription = registry.subscribe_events("p-query").unwrap();
    assert_eq!(subscription.artifacts.len(), 1);
    assert!(!subscription.snapshot_truncated);
    let replay = registry.replay_events("p-query", 0, 128).unwrap();
    assert!(!replay.reset_required);
    assert_eq!(replay.head_cursor, subscription.snapshot_cursor);
    assert!(replay.events.iter().any(|e| e.event_type == "artifact.version_published"));
    assert_eq!(registry.replay_events("p-query", 0, 0).unwrap_err(), "invalid replay cursor/page bound");

    let pool = registry.read_pool().unwrap();
    let other_registry = ArtifactRegistry::new(&root);
    assert!(std::ptr::eq(pool, other_registry.read_pool().unwrap()));
    let leases: Vec<_> = (0..mlrescurix_harness::read_pool::READ_CONNECTIONS).map(|_| pool.try_read().unwrap()).collect();
    assert!(pool.try_read().err().unwrap().contains("pool"));
    let query_only: i64 = leases[0].query_row("PRAGMA query_only", [], |r| r.get(0)).unwrap();
    assert_eq!(query_only, 1);
    assert!(leases[0].execute("INSERT INTO durable_events(project,event_type,payload_json,committed_ms) VALUES('p-query','test','{}',1)", []).is_err());
    let writer = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    assert_eq!(writer.execute("UPDATE projects SET revision=revision+1 WHERE id='p-query'", []).unwrap(), 1);
}

#[test]
fn p24_artifact_event_rolls_back_with_registry_mutation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-atomic");
    execution::migrate(&root).unwrap();
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    db.execute("INSERT INTO artifacts(project,artifact_id,kind,producer_authority,created_ms,updated_ms) VALUES('p-atomic','rollback_artifact','code','agent:test',1,1)", []).unwrap();
    let in_tx: i64 = db.query_row("SELECT count(*) FROM durable_events WHERE project='p-atomic' AND event_type='artifact.created'", [], |r| r.get(0)).unwrap();
    assert_eq!(in_tx, 1);
    db.execute_batch("ROLLBACK").unwrap();
    let after: i64 = db.query_row("SELECT count(*) FROM durable_events WHERE project='p-atomic' AND event_type='artifact.created'", [], |r| r.get(0)).unwrap();
    assert_eq!(after, 0);
}

#[test]
fn p24_event_retention_tracks_expired_cursor_per_project() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    setup_project(&root, "p-other");
    setup_project(&root, "p-retain");
    execution::migrate(&root).unwrap();
    let registry=ArtifactRegistry::new(&root);
    registry.register_artifact("p-other","other","code","agent:test",None).unwrap();
    registry.register_artifact("p-retain","many","code","agent:test",None).unwrap();
    let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let sha="f".repeat(64);
    db.execute("INSERT OR IGNORE INTO blobs VALUES(?1,1,'text/plain',1)",[&sha]).unwrap();
    let tx=db.unchecked_transaction().unwrap();
    for n in 0..10_005 {
        tx.execute("INSERT INTO artifact_versions(project,version_id,artifact_id,blob_sha256,index_digest,created_ms) VALUES('p-retain',?1,'many',?2,'bounded',?3)",rusqlite::params![format!("v{n}"),sha,n]).unwrap();
    }
    tx.commit().unwrap();
    let initial=registry.replay_events("p-retain",0,10).unwrap();
    assert!(initial.reset_required);
    assert!(initial.events.is_empty());
    assert!(initial.head_cursor>0);
    let other=registry.replay_events("p-other",0,10).unwrap();
    assert!(!other.reset_required, "other project's global cursor gaps are not expiry");
    let db=rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let retained:i64=db.query_row("SELECT count(*) FROM durable_events WHERE project='p-retain'",[],|r|r.get(0)).unwrap();
    assert_eq!(retained,10_000);
}

#[test]
fn p26_successor_and_tombstone_commands_are_revisioned_and_idempotent() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    execution::migrate(&root).unwrap();
    setup_project(&root, "p-p26");
    // Model an already-v10 database that predates the additive tombstone event trigger.
    rusqlite::Connection::open(root.join("state.sqlite")).unwrap()
        .execute_batch("DROP TRIGGER IF EXISTS p2_event_artifact_tombstone").unwrap();
    execution::migrate(&root).unwrap();
    let registry = ArtifactRegistry::new(&root);
    let digest = "e".repeat(64);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT INTO blobs VALUES(?1,1,'text/plain',1)", [&digest]).unwrap();
    drop(db);

    registry.register_artifact("p-p26", "candidate", "code", "agent:worker", None).unwrap();
    let first = registry.publish_version("p-p26", "candidate", &digest, "first", None, 1, None, None).unwrap();
    let created = registry.publish_successor_command("p-p26", "succ-1", 1, "candidate", &digest, "second", Some("summary"), 1, None).unwrap();
    assert_eq!(created.status, "applied");
    assert_eq!(created.revision, 2);
    let second = created.result["version_id"].as_str().unwrap().to_string();
    assert_eq!(created.result["supersedes"], first);
    let replay = registry.publish_successor_command("p-p26", "succ-1", 1, "candidate", &digest, "second", Some("summary"), 1, None).unwrap();
    assert_eq!(replay, created);
    assert!(registry.publish_successor_command("p-p26", "succ-1", 1, "candidate", &digest, "different", None, 1, None).unwrap_err().contains("reused with different payload"));
    let stale = registry.publish_successor_command("p-p26", "succ-stale", 1, "candidate", &digest, "third", None, 1, None).unwrap();
    assert_eq!(stale.status, "rejected");
    assert_eq!(stale.result["reason"], "revision conflict");
    assert_eq!(registry.get_artifact("p-p26", "candidate").unwrap().unwrap().current_version_id.as_deref(), Some(second.as_str()));
    let lineage = registry.query_lineage("p-p26", &second, 8).unwrap();
    assert!(lineage.iter().any(|edge| edge.source_version_id == second && edge.target_version_id == first && edge.relation == "supersedes"));

    registry.register_artifact("p-p26", "retire-me", "run_log", "agent:worker", None).unwrap();
    let tombstone = registry.tombstone_artifact_command("p-p26", "tomb-1", 2, "retire-me").unwrap();
    assert_eq!(tombstone.status, "applied");
    assert_eq!(tombstone.revision, 3);
    assert_eq!(registry.tombstone_artifact_command("p-p26", "tomb-1", 2, "retire-me").unwrap(), tombstone);
    assert!(registry.list_artifacts("p-p26", None, 128).unwrap().iter().all(|(a,_)| a.artifact_id != "retire-me"));
    assert!(registry.get_artifact("p-p26", "retire-me").unwrap().is_some(), "tombstone preserves audit metadata");
    let events = registry.get_events("p-p26", 0, 128).unwrap();
    assert!(events.iter().any(|event| event.event_type == "artifact.tombstoned"));
    assert!(events.iter().any(|event| event.event_type == "artifact.version_published"));
}

#[test]
fn p26_tombstone_respects_pins_and_selection_history() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, policy()).unwrap();
    execution::migrate(&root).unwrap();
    setup_project(&root, "p-p26-refs");
    let registry = ArtifactRegistry::new(&root);
    let digest = "f".repeat(64);
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    db.execute("INSERT INTO blobs VALUES(?1,1,'text/plain',1)", [&digest]).unwrap();
    db.execute("INSERT INTO artifacts(project,artifact_id,kind,producer_authority,created_ms,updated_ms) VALUES('p-p26-refs','score','scorecard','evaluator:trusted',1,1)", []).unwrap();
    db.execute("INSERT INTO artifact_versions(project,version_id,artifact_id,blob_sha256,index_digest,created_ms) VALUES('p-p26-refs','score-v1','score',?1,'trusted',1)", [&digest]).unwrap();
    db.execute("UPDATE artifacts SET current_version_id='score-v1' WHERE project='p-p26-refs' AND artifact_id='score'", []).unwrap();
    drop(db);
    registry.register_artifact("p-p26-refs", "candidate", "code", "agent:worker", None).unwrap();
    let candidate = registry.publish_version("p-p26-refs", "candidate", &digest, "candidate", None, 1, None, None).unwrap();
    registry.set_apex_selection("p-p26-refs", &candidate, "score-v1", "protocol", "selected").unwrap();
    registry.register_artifact("p-p26-refs", "pinned", "run_log", "agent:worker", None).unwrap();
    registry.pin("p-p26-refs", "pinned").unwrap();
    let pinned = registry.tombstone_artifact_command("p-p26-refs", "tomb-pinned", 1, "pinned").unwrap();
    assert_eq!(pinned.status, "rejected");
    assert!(pinned.result["reason"].as_str().unwrap().contains("pinned"));
    let selected = registry.tombstone_artifact_command("p-p26-refs", "tomb-selected", 1, "candidate").unwrap();
    assert_eq!(selected.status, "rejected");
    assert!(selected.result["reason"].as_str().unwrap().contains("selection history"));
}
