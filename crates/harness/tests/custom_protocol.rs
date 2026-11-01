use mlrescurix_harness::{
    custom_protocol::{
        carve_lockbox, consume_assessment, freeze, generate_k_fold_splits, load, CustomProtocol,
    },
    trusted_evaluation::Metric,
};
use rusqlite::Connection;
use std::time::Duration;

fn setup(c: &Connection) {
    c.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,path TEXT NOT NULL DEFAULT '',sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets(project,id,path,sha256) VALUES('p','d','',?1)", [&"a".repeat(64)]).unwrap();
}

fn protocol() -> CustomProtocol {
    CustomProtocol::single_split(
        "a".repeat(64),
        "target".into(),
        Metric::Accuracy,
        vec![2, 0],
        vec![1],
        vec![3],
    )
}

#[test]
fn frozen_protocol_is_bound_idempotent_and_immutable() {
    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);
    let digest = freeze(&mut c, "p", "d", protocol()).unwrap();
    let (stored, loaded_digest) = load(&c, "p", "d").unwrap();
    assert_eq!(digest, loaded_digest);
    assert_eq!(stored.train, vec![0, 2]);
    assert_eq!(freeze(&mut c, "p", "d", stored.clone()).unwrap(), digest);
    let mut different = stored.clone();
    different.metric = Metric::Mse;
    assert!(freeze(&mut c, "p", "d", different).is_err());
    let mut wrong_dataset = stored.clone();
    wrong_dataset.dataset_sha256 = "b".repeat(64);
    assert!(freeze(&mut c, "p", "d", wrong_dataset).is_err());
    assert!(freeze(&mut c, "p", "other", stored).is_err());
    assert!(c.execute("UPDATE custom_protocols SET protocol_json='{}'", []).is_err());
    assert!(c.execute("DELETE FROM custom_protocols", []).is_err());
    let mut overlap = protocol();
    overlap.test.push(0);
    assert!(freeze(&mut c, "p", "d", overlap).is_err());
}

#[test]
fn concurrent_conflicting_freezes_have_one_winner_and_survive_reopen() {
    let temp = tempfile::tempdir().unwrap();
    let db = temp.path().join("state.sqlite");
    let c = Connection::open(&db).unwrap();
    setup(&c);
    drop(c);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let workers: Vec<_> = (0..2)
        .map(|i| {
            let (db, barrier) = (db.clone(), barrier.clone());
            std::thread::spawn(move || {
                let mut c = Connection::open(db).unwrap();
                c.busy_timeout(Duration::from_secs(10)).unwrap();
                let mut p = protocol();
                if i == 1 {
                    p.metric = Metric::Mae;
                }
                barrier.wait();
                freeze(&mut c, "p", "d", p)
            })
        })
        .collect();
    let results: Vec<_> = workers.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
}

#[test]
fn multi_fold_splits_and_lockbox_carve_out_and_consumption() {
    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    // 1. Create a 100-row synthetic dataset: rows 0..100
    let all_rows: Vec<u64> = (0..100).collect();

    // 2. Deterministically carve out 10% lockbox before search
    let (dev_rows, lockbox_rows) = carve_lockbox(&all_rows, 0.10, 2026).unwrap();
    assert_eq!(lockbox_rows.len(), 10);
    assert_eq!(dev_rows.len(), 90);

    // Disjointness of lockbox and dev
    for &id in &lockbox_rows {
        assert!(!dev_rows.contains(&id));
    }

    // 3. Generate 5-fold cross validation over dev rows
    let folds = generate_k_fold_splits(&dev_rows, 5, 2026).unwrap();
    assert_eq!(folds.len(), 5);
    for fold in &folds {
        assert_eq!(fold.train.len(), 72);
        assert_eq!(fold.validation.len(), 18);
        for &v in &fold.validation {
            assert!(!fold.train.contains(&v));
            assert!(!lockbox_rows.contains(&v));
        }
        for &t in &fold.train {
            assert!(!lockbox_rows.contains(&t));
        }
    }

    // 4. Construct protocol with folds and lockbox
    let mut multi_protocol = CustomProtocol::single_split(
        "a".repeat(64),
        "target".into(),
        Metric::RocAuc,
        folds[0].train.clone(),
        folds[0].validation.clone(),
        vec![9999], // dummy test
    );
    multi_protocol.folds = folds;
    multi_protocol.lockbox_rows = lockbox_rows.clone();
    multi_protocol.strategy = Some("stratified_k_fold".into());

    let digest = freeze(&mut c, "p", "d", multi_protocol).unwrap();
    let (loaded, loaded_digest) = load(&c, "p", "d").unwrap();
    assert_eq!(digest, loaded_digest);
    assert_eq!(loaded.folds.len(), 5);
    assert_eq!(loaded.lockbox_rows.len(), 10);

    // 5. Test one-shot assessment consumption
    let binding = mlrescurix_harness::custom_protocol::AssessmentBinding {
        candidate_id: "cand_01".into(),
        artifact_sha256: "0".repeat(64),
        predictions_sha256: "1".repeat(64),
        metric: Metric::RocAuc,
        score: 0.85,
        receipt_identity: "receipt_01".into(),
        receipt_json: "{}".into(),
    };
    let consumed = consume_assessment(&mut c, "p", "d", &binding).unwrap();
    assert_eq!(consumed, lockbox_rows);

    // Verify record retrieval
    let record = mlrescurix_harness::custom_protocol::get_assessment_consumption(&c, "p", "d").unwrap().unwrap();
    assert_eq!(record.candidate_id, "cand_01");
    assert_eq!(record.score, 0.85);
    assert_eq!(record.metric, Metric::RocAuc);

    // Verify immutability triggers on assessment_lockbox_consumptions
    assert!(c.execute("UPDATE assessment_lockbox_consumptions SET score=0.99", []).is_err());
    assert!(c.execute("DELETE FROM assessment_lockbox_consumptions", []).is_err());

    // 6. Second consumption MUST fail (fencing second untouched evaluation)
    let err = consume_assessment(&mut c, "p", "d", &binding).unwrap_err();
    assert!(err.contains("already been consumed"));
}

#[test]
fn rejects_overlapping_lockbox_and_dev_rows() {
    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    let mut bad_protocol = protocol();
    // Make lockbox overlap with train row 0
    bad_protocol.lockbox_rows = vec![0, 10, 11];
    let err = freeze(&mut c, "p", "d", bad_protocol).unwrap_err();
    assert!(err.contains("duplicate or overlapping protocol rows"));
}

#[test]
fn fold_adversarial_validation_checks() {
    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    // Test 1: Empty test split is allowed when folds are present
    let mut valid_empty_test = protocol();
    valid_empty_test.test = vec![];
    valid_empty_test.folds = vec![mlrescurix_harness::custom_protocol::FoldSplit {
        split_id: "fold_0".into(),
        train: vec![0],
        validation: vec![2],
    }];
    assert!(freeze(&mut c, "p", "d", valid_empty_test).is_ok());

    // Reopen DB for fresh tests
    let mut c2 = Connection::open_in_memory().unwrap();
    setup(&c2);

    // Test 2: Duplicate fold split IDs
    let mut dup_split_id = protocol();
    dup_split_id.folds = vec![
        mlrescurix_harness::custom_protocol::FoldSplit {
            split_id: "fold_0".into(),
            train: vec![0],
            validation: vec![2],
        },
        mlrescurix_harness::custom_protocol::FoldSplit {
            split_id: "fold_0".into(),
            train: vec![2],
            validation: vec![0],
        },
    ];
    let err = freeze(&mut c2, "p", "d", dup_split_id).unwrap_err();
    assert!(err.contains("duplicate or empty fold split_id"));

    // Test 3: Duplicate rows within fold train
    let mut dup_train_row = protocol();
    dup_train_row.folds = vec![mlrescurix_harness::custom_protocol::FoldSplit {
        split_id: "fold_0".into(),
        train: vec![0, 0],
        validation: vec![2],
    }];
    let err = freeze(&mut c2, "p", "d", dup_train_row).unwrap_err();
    assert!(err.contains("has duplicate row in train"));

    // Test 4: Unknown row in fold train (not in dev train/validation)
    let mut unknown_row = protocol();
    unknown_row.folds = vec![mlrescurix_harness::custom_protocol::FoldSplit {
        split_id: "fold_0".into(),
        train: vec![0, 999], // 999 not in train [0,2] or val [1]
        validation: vec![2],
    }];
    let err = freeze(&mut c2, "p", "d", unknown_row).unwrap_err();
    assert!(err.contains("contains unknown row 999"));

    // Test 5: Test row leaked into fold train
    let mut test_row_leak = protocol(); // test has [3]
    test_row_leak.folds = vec![mlrescurix_harness::custom_protocol::FoldSplit {
        split_id: "fold_0".into(),
        train: vec![0, 3], // 3 is in test
        validation: vec![2],
    }];
    let err = freeze(&mut c2, "p", "d", test_row_leak).unwrap_err();
    assert!(err.contains("contains unknown row 3") || err.contains("contains test row 3"));

    // Test 6: Strategy semantics - group strategy requires group_column
    let mut bad_strat = protocol();
    bad_strat.strategy = Some("group_k_fold".into());
    let err = freeze(&mut c2, "p", "d", bad_strat).unwrap_err();
    assert!(err.contains("group-based validation strategy requires group_column"));

    // Test 7: Feasibility-aware lockbox carve-out on small dataset (10 rows)
    let small_rows: Vec<u64> = (0..10).collect();
    let (dev, lockbox) = mlrescurix_harness::custom_protocol::carve_lockbox(&small_rows, 0.20, 42).unwrap();
    assert_eq!(lockbox.len(), 2);
    assert_eq!(dev.len(), 8);
}

#[test]
fn end_to_end_assessment_evaluation_and_consumption() {
    use mlrescurix_harness::trusted_evaluation::{EvaluationIdentity, PredictionBatch};

    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    let mut p = protocol();
    p.lockbox_rows = vec![10, 11];
    p.metric = Metric::Accuracy;
    let protocol_sha = freeze(&mut c, "p", "d", p).unwrap();

    let identity = EvaluationIdentity {
        project_id: "p".into(),
        job_id: "job_01".into(),
        candidate_id: "cand_01".into(),
        dataset_id: "d".into(),
        dataset_sha256: "a".repeat(64),
        split_sha256: protocol_sha,
        source_sha256: "0".repeat(64),
        environment_id: "env_01".into(),
        weights_sha256: "b".repeat(64),
    };

    let labels = vec![(10, 1.0), (11, 0.0)];
    let preds = PredictionBatch::scalar(identity.clone(), vec![(10, 1.0), (11, 0.0)]);

    for field in ["project", "dataset", "hash"] {
        let mut wrong = identity.clone();
        match field {
            "project" => wrong.project_id = "other".into(),
            "dataset" => wrong.dataset_id = "other".into(),
            _ => wrong.dataset_sha256 = "c".repeat(64),
        }
        let wrong_preds = PredictionBatch::scalar(wrong.clone(), vec![(10, 1.0), (11, 0.0)]);
        let err = mlrescurix_harness::custom_protocol::evaluate_and_consume_assessment(
            &mut c, "p", "d", &wrong, &labels, &wrong_preds,
        ).unwrap_err();
        assert!(err.contains("assessment identity does not match"));
    }

    // 1. Successful authoritative evaluation and atomic lockbox consumption
    let receipt = mlrescurix_harness::custom_protocol::evaluate_and_consume_assessment(
        &mut c, "p", "d", &identity, &labels, &preds,
    ).unwrap();
    assert_eq!(receipt.score, 1.0);
    assert_eq!(receipt.metric, Metric::Accuracy);

    // 2. Consumption record exists and matches
    let record = mlrescurix_harness::custom_protocol::get_assessment_consumption(&c, "p", "d").unwrap().unwrap();
    assert_eq!(record.candidate_id, "cand_01");
    assert_eq!(record.artifact_sha256, "b".repeat(64));
    assert_eq!(record.score, 1.0);
    assert!(!record.receipt_json.is_empty());

    // 2b. Crash recovery: get_assessment_consumption_receipt returns identical evaluation receipt without re-evaluation
    let recovered_receipt = mlrescurix_harness::custom_protocol::get_assessment_consumption_receipt(&c, "p", "d").unwrap().unwrap();
    assert_eq!(recovered_receipt.score, receipt.score);
    assert_eq!(recovered_receipt.metric, receipt.metric);
    assert_eq!(recovered_receipt.predictions_sha256, receipt.predictions_sha256);
    assert_eq!(recovered_receipt.identity, receipt.identity);

    // 3. Second evaluation on the same lockbox MUST fail
    let err = mlrescurix_harness::custom_protocol::evaluate_and_consume_assessment(
        &mut c, "p", "d", &identity, &labels, &preds,
    ).unwrap_err();
    assert!(err.contains("already been consumed"));

    // 4. Mismatched labels for a new protocol fails before consumption
    let mut c2 = Connection::open_in_memory().unwrap();
    setup(&c2);
    let mut p2 = protocol();
    p2.lockbox_rows = vec![20, 21];
    let protocol2_sha = freeze(&mut c2, "p", "d", p2).unwrap();
    let mut identity2 = identity.clone();
    identity2.split_sha256 = protocol2_sha;
    let bad_labels = vec![(99, 1.0), (100, 0.0)];
    let err2 = mlrescurix_harness::custom_protocol::evaluate_and_consume_assessment(
        &mut c2, "p", "d", &identity2, &bad_labels, &preds,
    ).unwrap_err();
    assert!(err2.contains("trusted lockbox labels do not match registered lockbox rows"));
}

#[test]
fn group_separation_and_leakage_rejection() {
    use std::collections::HashMap;

    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    // Protocol: train [0, 2], val [1], test [3], lockbox [4, 5]
    let mut p = protocol();
    p.lockbox_rows = vec![4, 5];

    let mut valid_groups = HashMap::new();
    valid_groups.insert(0, "grp_A".into());
    valid_groups.insert(2, "grp_A".into());
    valid_groups.insert(1, "grp_B".into());
    valid_groups.insert(3, "grp_C".into());
    valid_groups.insert(4, "grp_D".into());
    valid_groups.insert(5, "grp_D".into());

    let mut valid_p = p.clone();
    valid_p.row_groups = Some(valid_groups.clone());
    assert!(freeze(&mut c, "p", "d", valid_p).is_ok());

    // Leaking group between train (0) and val (1)
    let mut c_leak1 = Connection::open_in_memory().unwrap();
    setup(&c_leak1);
    let mut leak_tv = valid_groups.clone();
    leak_tv.insert(1, "grp_A".into()); // Leaks grp_A into val!
    let mut leak_p1 = p.clone();
    leak_p1.row_groups = Some(leak_tv);
    let err1 = freeze(&mut c_leak1, "p", "d", leak_p1).unwrap_err();
    assert!(err1.contains("group leakage detected: group 'grp_A' present in both train and validation"));

    // Leaking group between train and lockbox
    let mut c_leak2 = Connection::open_in_memory().unwrap();
    setup(&c_leak2);
    let mut leak_t_lock = valid_groups.clone();
    leak_t_lock.insert(4, "grp_A".into()); // Leaks grp_A into lockbox!
    let mut leak_p2 = p.clone();
    leak_p2.row_groups = Some(leak_t_lock);
    let err2 = freeze(&mut c_leak2, "p", "d", leak_p2).unwrap_err();
    assert!(err2.contains("group leakage detected: group 'grp_A' present in both train and lockbox"));

    // Leaking group across fold splits
    let mut c_leak3 = Connection::open_in_memory().unwrap();
    setup(&c_leak3);
    let mut fold_p = p.clone();
    fold_p.folds = vec![mlrescurix_harness::custom_protocol::FoldSplit {
        split_id: "fold_0".into(),
        train: vec![0],
        validation: vec![2],
    }];
    let fold_groups = valid_groups.clone();
    // 0 and 2 are both grp_A, but fold_0 splits 0 into train and 2 into validation!
    fold_p.row_groups = Some(fold_groups);
    let err3 = freeze(&mut c_leak3, "p", "d", fold_p).unwrap_err();
    assert!(err3.contains("group leakage in fold 'fold_0': group 'grp_A' present in both train and validation"));
}

#[test]
fn temporal_ordering_and_lookahead_rejection() {
    use std::collections::HashMap;

    let mut c = Connection::open_in_memory().unwrap();
    setup(&c);

    // Protocol: train [0, 2], val [1], test [3], lockbox [4, 5]
    let mut p = protocol();
    p.lockbox_rows = vec![4, 5];

    // Valid chronological timestamps: train (10, 20) <= val (30) <= test (40) <= lockbox (50, 60)
    let mut valid_timestamps = HashMap::new();
    valid_timestamps.insert(0, 10);
    valid_timestamps.insert(2, 20);
    valid_timestamps.insert(1, 30);
    valid_timestamps.insert(3, 40);
    valid_timestamps.insert(4, 50);
    valid_timestamps.insert(5, 60);

    let mut valid_p = p.clone();
    valid_p.row_timestamps = Some(valid_timestamps.clone());
    assert!(freeze(&mut c, "p", "d", valid_p).is_ok());

    // Lookahead: validation (timestamp 15) is earlier than train max (timestamp 20)
    let mut c_leak1 = Connection::open_in_memory().unwrap();
    setup(&c_leak1);
    let mut lookahead_val = valid_timestamps.clone();
    lookahead_val.insert(1, 15); // val < train max (20)
    let mut leak_p1 = p.clone();
    leak_p1.row_timestamps = Some(lookahead_val);
    let err1 = freeze(&mut c_leak1, "p", "d", leak_p1).unwrap_err();
    assert!(err1.contains("temporal lookahead detected: validation min timestamp (15) < train max timestamp (20)"));

    // Lookahead: lockbox (timestamp 35) is earlier than test max (timestamp 40)
    let mut c_leak2 = Connection::open_in_memory().unwrap();
    setup(&c_leak2);
    let mut lookahead_lock = valid_timestamps.clone();
    lookahead_lock.insert(4, 35); // lockbox < test (40)
    let mut leak_p2 = p.clone();
    leak_p2.row_timestamps = Some(lookahead_lock);
    let err2 = freeze(&mut c_leak2, "p", "d", leak_p2).unwrap_err();
    assert!(err2.contains("temporal lookahead detected: lockbox min timestamp (35) < dev max timestamp (40)"));
}
