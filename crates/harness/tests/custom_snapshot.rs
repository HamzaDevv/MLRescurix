use mlrescurix_harness::{custom_snapshot::{create, verify, SnapshotIdentity}, development_inputs::DevelopmentInputs};

fn identity() -> SnapshotIdentity {
    SnapshotIdentity { project_id: "project".into(), job_id: "job".into(), candidate_id: "candidate".into(), dataset_id: "ds".into(), dataset_sha256: "a".repeat(64), protocol_sha256: "b".repeat(64), source_sha256: "c".repeat(64), environment_id: "env".into() }
}
fn inputs() -> DevelopmentInputs {
    DevelopmentInputs { training_csv: b"id,y\n0,1\n".to_vec(), validation_features_csv: b"id,x\n1,2\n".to_vec(), labels: vec![(1, 5.0)] }
}

#[test]
fn snapshot_is_job_bound_and_rejects_tampering_and_symlinks() {
    use std::{fs, os::unix::fs::symlink};
    let tmp = tempfile::tempdir().unwrap();
    let id = identity();
    let dir = create(tmp.path(), &id, &inputs()).unwrap();
    verify(tmp.path(), &id).unwrap();
    let mut wrong = id.clone(); wrong.candidate_id = "other".into();
    assert!(verify(tmp.path(), &wrong).is_err());
    assert!(create(tmp.path(), &id, &inputs()).is_err());
    fs::write(dir.join("training.csv"), b"tampered").unwrap();
    assert!(verify(tmp.path(), &id).is_err());
    fs::remove_file(dir.join("training.csv")).unwrap();
    symlink(dir.join("validation_features.csv"), dir.join("training.csv")).unwrap();
    assert!(verify(tmp.path(), &id).is_err());
}

#[test]
fn incomplete_and_concurrent_snapshots_cannot_be_reused() {
    let tmp = tempfile::tempdir().unwrap();
    let id = identity();
    let mut too_large = inputs(); too_large.validation_features_csv = vec![0; 64 * 1024 * 1024 + 1];
    assert!(create(tmp.path(), &id, &too_large).is_err());
    assert!(verify(tmp.path(), &id).is_err());
    assert!(create(tmp.path(), &id, &inputs()).is_err());
    let mut other = id; other.job_id = "another".into();
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let threads: Vec<_> = (0..2).map(|_| {
        let (root, id, barrier) = (tmp.path().to_path_buf(), other.clone(), barrier.clone());
        std::thread::spawn(move || { barrier.wait(); create(&root, &id, &inputs()) })
    }).collect();
    let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    verify(tmp.path(), &other).unwrap();
}

#[test]
fn registered_protocol_to_snapshot_keeps_holdout_and_labels_private() {
    use mlrescurix_harness::{custom_protocol::{freeze, CustomProtocol}, custom_snapshot::{prepare_registered, evaluate_predictions}, trusted_evaluation::{EvaluationIdentity, Metric, PredictionBatch}};
    use sha2::{Digest, Sha256};
    use std::fs;
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let root = tmp.path().join("snapshots"); fs::create_dir(&root).unwrap();
    let bytes = b"x,target\ntrain,1\nvalidation,83749\nsealed,9\n";
    fs::write(&raw, bytes).unwrap();
    let hash = format!("{:x}", Sha256::digest(bytes));
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets VALUES('project','ds',?1,?2)", rusqlite::params![raw.to_string_lossy(), hash]).unwrap();
    let protocol = CustomProtocol::single_split(hash.clone(), "target".into(), Metric::Accuracy, vec![0], vec![1], vec![2]);
    let protocol_hash = freeze(&mut c, "project", "ds", protocol).unwrap();
    let mut id = identity(); id.dataset_sha256 = hash; id.protocol_sha256 = protocol_hash;
    let dir = prepare_registered(&c, &root, "ds", &id).unwrap();
    verify(&root, &id).unwrap();
    assert_eq!(fs::read_to_string(dir.join("training.csv")).unwrap(), "__mlr_row_id,x,target\n0,train,1\n");
    assert_eq!(fs::read_to_string(dir.join("validation_features.csv")).unwrap(), "__mlr_row_id,x\n1,validation\n");
    assert_eq!(fs::read_to_string(dir.join("labels.json")).unwrap(), "[[1,83749.0]]");
    assert!(!fs::read_to_string(dir.join("training.csv")).unwrap().contains("sealed"));
    let evaluation = EvaluationIdentity { project_id: id.project_id.clone(), job_id: id.job_id.clone(), candidate_id: id.candidate_id.clone(), dataset_id: id.dataset_id.clone(), dataset_sha256: id.dataset_sha256.clone(), split_sha256: id.protocol_sha256.clone(), source_sha256: id.source_sha256.clone(), environment_id: id.environment_id.clone(), weights_sha256: "d".repeat(64) };
    let prediction = PredictionBatch::scalar(evaluation.clone(), vec![(1, 0.0)]);
    assert_eq!(evaluate_predictions(&c, &root, "ds", &id, &evaluation, &prediction).unwrap().score, 0.0);
    let mut mismatched = evaluation.clone(); mismatched.job_id = "other".into();
    assert!(evaluate_predictions(&c, &root, "ds", &id, &mismatched, &prediction).is_err());
    assert_eq!(prepare_registered(&c, &root, "ds", &id).unwrap(), dir);
    let mut fabricated_id = id.clone(); fabricated_id.job_id = "fabricated".into();
    create(&root, &fabricated_id, &DevelopmentInputs { training_csv: b"x\n1\n".to_vec(), validation_features_csv: b"x\n2\n".to_vec(), labels: vec![(999, 0.0)] }).unwrap();
    let mut fabricated_eval = evaluation.clone(); fabricated_eval.job_id = fabricated_id.job_id.clone();
    let fabricated_prediction = PredictionBatch::scalar(fabricated_eval.clone(), vec![(999, 0.0)]);
    assert!(evaluate_predictions(&c, &root, "ds", &fabricated_id, &fabricated_eval, &fabricated_prediction).is_err());
    let mut another = id.clone(); another.job_id = "job2".into();
    fs::write(raw, b"x,target\nmodified,1\nvalidation,83749\nsealed,9\n").unwrap();
    assert!(prepare_registered(&c, &root, "ds", &another).is_err());
    assert!(verify(&root, &another).is_err());
}
