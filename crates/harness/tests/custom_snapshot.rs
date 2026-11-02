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
fn snapshot_rejects_fifo_without_waiting_for_a_writer() {
    use std::{ffi::CString, fs, os::unix::ffi::OsStrExt};
    let tmp = tempfile::tempdir().unwrap();
    let id = identity();
    let dir = create(tmp.path(), &id, &inputs()).unwrap();
    let path = dir.join("training.csv");
    fs::remove_file(&path).unwrap();
    let path = CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
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

#[test]
fn registered_snapshot_streams_large_development_data() {
    use mlrescurix_harness::{custom_protocol::{freeze, CustomProtocol}, custom_snapshot::prepare_registered, trusted_evaluation::Metric};
    use sha2::{Digest, Sha256};
    use std::{fs, io::{BufWriter, Write}};
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("large.csv");
    let root = tmp.path().join("snapshots");
    fs::create_dir(&root).unwrap();
    let mut out = BufWriter::new(fs::File::create(&raw).unwrap());
    let mut hasher = Sha256::new();
    let header = b"x,target\n";
    out.write_all(header).unwrap(); hasher.update(header);
    let row = format!("{},1\n", "a".repeat(32768));
    for _ in 0..2400 { out.write_all(row.as_bytes()).unwrap(); hasher.update(row.as_bytes()); }
    out.flush().unwrap();
    assert!(fs::metadata(&raw).unwrap().len() > 64 * 1024 * 1024);
    let hash = format!("{:x}", hasher.finalize());
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets VALUES('project','ds',?1,?2)", rusqlite::params![raw.to_string_lossy(), hash]).unwrap();
    let protocol = CustomProtocol::single_split(hash.clone(), "target".into(), Metric::Mae,
        (0..2200).collect(), (2200..2300).collect(), (2300..2400).collect());
    let protocol_hash = freeze(&mut c, "project", "ds", protocol).unwrap();
    let mut id = identity(); id.dataset_sha256 = hash; id.protocol_sha256 = protocol_hash;
    let dir = prepare_registered(&c, &root, "ds", &id).unwrap();
    verify(&root, &id).unwrap();
    assert!(fs::metadata(dir.join("training.csv")).unwrap().len() > 64 * 1024 * 1024);
    let labels: Vec<(u64, f64)> = serde_json::from_slice(&fs::read(dir.join("labels.json")).unwrap()).unwrap();
    assert_eq!(labels.len(), 100);
    assert_eq!(labels[0], (2200, 1.0));
    assert!(!dir.join("dev.csv").exists());
}

#[test]
#[ignore = "requires the pinned local Docker image"]
fn isolated_inference_scores_controller_labels_and_binds_inputs() {
    use mlrescurix_harness::{agents::sandbox::SandboxLimits, custom_protocol::{freeze, CustomProtocol}, custom_snapshot::prepare_registered, isolated_inference, trusted_evaluation::Metric};
    use sha2::{Digest, Sha256};
    use std::{fs, os::unix::fs::PermissionsExt};
    let tmp = tempfile::tempdir().unwrap();
    let raw = tmp.path().join("raw.csv");
    let root = tmp.path().join("snapshots"); fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let bytes = b"x,target\n1,10\n2,25\n3,999\n";
    fs::write(&raw, bytes).unwrap();
    let data_hash = format!("{:x}", Sha256::digest(bytes));
    let source = tmp.path().join("candidate.py");
    let program = b"import argparse,csv,json,os\np=argparse.ArgumentParser();p.add_argument('--input');p.add_argument('--weights');a=p.parse_args()\nassert not os.path.exists('/workspace/labels.json')\nwith open(a.weights) as f: weight=float(f.read())\nwith open(a.input) as f: rows=list(csv.DictReader(f))\nprint(json.dumps([[int(r['__mlr_row_id']),float(r['x'])*weight] for r in rows]))\n";
    fs::write(&source, program).unwrap();
    let weights = tmp.path().join("weights"); fs::write(&weights, b"10").unwrap();
    let weight_hash = format!("{:x}", Sha256::digest(b"10"));
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets VALUES('project','ds',?1,?2)", rusqlite::params![raw.to_string_lossy(), data_hash]).unwrap();
    let protocol = CustomProtocol::single_split(data_hash.clone(), "target".into(), Metric::Mae, vec![0], vec![1], vec![2]);
    let protocol_hash = freeze(&mut c, "project", "ds", protocol).unwrap();
    let mut id = identity(); id.dataset_sha256 = data_hash; id.protocol_sha256 = protocol_hash;
    id.source_sha256 = format!("{:x}", Sha256::digest(program));
    id.environment_id = "sha256:897f14c33bb58c525ccb6fcefc8e46e7de6138faf73b36c57eabdba71946559c".into();
    prepare_registered(&c, &root, "ds", &id).unwrap();
    let result = isolated_inference::evaluate(&c, &root, "ds", &id, &source, &weights, &weight_hash, &SandboxLimits::default()).unwrap();
    assert_eq!(result.score, 5.0); assert_eq!(result.row_count, 1);
    assert_eq!(result.identity.weights_sha256, weight_hash);
    fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(isolated_inference::evaluate(&c, &root, "ds", &id, &source, &weights, &weight_hash, &SandboxLimits::default()).unwrap_err().contains("private"));
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let bad_program = b"print('[[1, 0], [1, 0]]')\n";
    fs::write(&source, bad_program).unwrap();
    let mut bad_id = id.clone(); bad_id.job_id = "bad-predictions".into();
    bad_id.source_sha256 = format!("{:x}", Sha256::digest(bad_program));
    prepare_registered(&c, &root, "ds", &bad_id).unwrap();
    assert!(isolated_inference::evaluate(&c, &root, "ds", &bad_id, &source, &weights, &weight_hash, &SandboxLimits::default()).unwrap_err().contains("duplicate"));
    fs::write(&source, program).unwrap();
    fs::write(&weights, b"100").unwrap();
    assert!(isolated_inference::evaluate(&c, &root, "ds", &id, &source, &weights, &weight_hash, &SandboxLimits::default()).unwrap_err().contains("hash mismatch"));
    fs::write(&source, b"print('forged')").unwrap();
    assert!(isolated_inference::evaluate(&c, &root, "ds", &id, &source, &weights, &weight_hash, &SandboxLimits::default()).unwrap_err().contains("hash mismatch"));
}

#[test]
#[ignore = "requires the pinned local Docker image"]
fn isolated_fit_then_inference_keeps_validation_private_and_fences_replay() {
    use mlrescurix_harness::{agents::sandbox::SandboxLimits, custom_protocol::{freeze, CustomProtocol}, custom_snapshot::prepare_registered, isolated_inference, trusted_evaluation::Metric};
    use sha2::{Digest, Sha256};
    use std::{fs, os::unix::fs::PermissionsExt};
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("snapshots"); fs::create_dir(&root).unwrap();
    fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();
    let raw = tmp.path().join("raw.csv"); let data = b"x,target\n1,10\n2,25\n3,999\n"; fs::write(&raw, data).unwrap();
    let data_hash = format!("{:x}", Sha256::digest(data));
    let program = b"import argparse,csv,json,os\np=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--input');p.add_argument('--weights');a=p.parse_args()\nassert not os.path.exists('/workspace/labels.json')\nwith open(a.input) as f: rows=list(csv.DictReader(f))\nif a.fit:\n assert 'target' in rows[0] and not os.path.exists('/workspace/validation.csv')\n print(json.dumps(str(float(rows[0]['target'])/float(rows[0]['x'])).encode().hex()))\nelse:\n assert 'target' not in rows[0] and not os.path.exists('/workspace/training.csv')\n with open(a.weights) as f: weight=float(f.read())\n print(json.dumps([[int(r['__mlr_row_id']),float(r['x'])*weight] for r in rows]))\n";
    let source=tmp.path().join("candidate.py"); fs::write(&source,program).unwrap();
    let mut c = rusqlite::Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets VALUES('project','ds',?1,?2)", rusqlite::params![raw.to_string_lossy(), data_hash]).unwrap();
    let protocol = CustomProtocol::single_split(data_hash.clone(), "target".into(), Metric::Mae, vec![0], vec![1], vec![2]);
    let protocol_hash = freeze(&mut c, "project", "ds", protocol).unwrap();
    let mut id = identity(); id.dataset_sha256 = data_hash; id.protocol_sha256 = protocol_hash;
    id.source_sha256 = format!("{:x}", Sha256::digest(program));
    id.environment_id = "sha256:897f14c33bb58c525ccb6fcefc8e46e7de6138faf73b36c57eabdba71946559c".into();
    prepare_registered(&c, &root, "ds", &id).unwrap();
    let (weights, receipt) = isolated_inference::fit(&c, &root, "ds", &id, &source, &SandboxLimits::default()).unwrap();
    assert_eq!(fs::read(&weights).unwrap(), b"10.0");
    assert_eq!(receipt.weights_sha256, format!("{:x}", Sha256::digest(b"10.0")));
    assert!(weights.parent().unwrap().join("intent.json").is_file());
    assert!(weights.parent().unwrap().join("receipt.json").is_file());
    assert_eq!(fs::read(weights.parent().unwrap().join("candidate.py")).unwrap(), program);
    fs::write(&source, b"changed after fitting").unwrap();
    let evaluation = isolated_inference::evaluate_fitted(&c, &root, "ds", &id, &SandboxLimits::default()).unwrap();
    assert_eq!(evaluation.score, 5.0);
    let durable: mlrescurix_harness::trusted_evaluation::EvaluationReceipt = serde_json::from_slice(&fs::read(weights.parent().unwrap().join("evaluation.json")).unwrap()).unwrap();
    assert_eq!(durable.score, evaluation.score);
    assert_eq!(durable.identity.weights_sha256, receipt.weights_sha256);
    assert!(weights.parent().unwrap().join("evaluation-intent.json").is_file());
    fs::write(&weights, b"11.0").unwrap();
    assert!(isolated_inference::evaluate_fitted(&c, &root, "ds", &id, &SandboxLimits::default()).unwrap_err().contains("already exists"));
    assert!(isolated_inference::fit(&c, &root, "ds", &id, &source, &SandboxLimits::default()).unwrap_err().contains("already exists"));
    fs::write(&source, program).unwrap();
    let mut failed_evaluation_id = id.clone(); failed_evaluation_id.job_id = "failed-evaluation".into();
    prepare_registered(&c, &root, "ds", &failed_evaluation_id).unwrap();
    let (failed_weights, _) = isolated_inference::fit(&c, &root, "ds", &failed_evaluation_id, &source, &SandboxLimits::default()).unwrap();
    fs::write(&failed_weights, b"11.0").unwrap();
    assert!(isolated_inference::evaluate_fitted(&c, &root, "ds", &failed_evaluation_id, &SandboxLimits::default()).unwrap_err().contains("hash mismatch"));
    fs::write(&failed_weights, b"10.0").unwrap();
    assert!(isolated_inference::evaluate_fitted(&c, &root, "ds", &failed_evaluation_id, &SandboxLimits::default()).unwrap_err().contains("already exists"));
    let failing_program = b"raise RuntimeError('fit failed')\n";
    fs::write(&source, failing_program).unwrap();
    let mut failed_id = id.clone(); failed_id.job_id = "failed-fit".into();
    failed_id.source_sha256 = format!("{:x}", Sha256::digest(failing_program));
    prepare_registered(&c, &root, "ds", &failed_id).unwrap();
    assert!(isolated_inference::fit(&c, &root, "ds", &failed_id, &source, &SandboxLimits::default()).unwrap_err().contains("isolated fit failed"));
    assert!(isolated_inference::evaluate_fitted(&c, &root, "ds", &failed_id, &SandboxLimits::default()).is_err());
    assert!(isolated_inference::fit(&c, &root, "ds", &failed_id, &source, &SandboxLimits::default()).unwrap_err().contains("already exists"));
}
