use mlrescurix_harness::{custom_protocol::{freeze, load, CustomProtocol}, trusted_evaluation::Metric};
use rusqlite::{params, Connection};
use std::time::Duration;

fn setup(c: &Connection) {
    c.execute_batch("PRAGMA foreign_keys=ON; CREATE TABLE scientific_datasets(project TEXT NOT NULL,id TEXT NOT NULL,sha256 TEXT NOT NULL,PRIMARY KEY(project,id));").unwrap();
    c.execute_batch(include_str!("../src/schema_custom_protocol.sql")).unwrap();
    c.execute("INSERT INTO scientific_datasets(project,id,sha256) VALUES('p','d',?1)", [&"a".repeat(64)]).unwrap();
}
fn protocol() -> CustomProtocol {
    CustomProtocol { dataset_sha256: "a".repeat(64), target_column: "target".into(), metric: Metric::Accuracy, train: vec![2, 0], validation: vec![1], test: vec![3] }
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
    let mut different = stored.clone(); different.metric = Metric::Mse;
    assert!(freeze(&mut c, "p", "d", different).is_err());
    let mut wrong_dataset = stored.clone(); wrong_dataset.dataset_sha256 = "b".repeat(64);
    assert!(freeze(&mut c, "p", "d", wrong_dataset).is_err());
    assert!(freeze(&mut c, "p", "other", stored).is_err());
    assert!(c.execute("UPDATE custom_protocols SET protocol_json='{}'", []).is_err());
    assert!(c.execute("DELETE FROM custom_protocols", []).is_err());
    let mut overlap = protocol(); overlap.test.push(0);
    assert!(freeze(&mut c, "p", "d", overlap).is_err());
}

#[test]
fn concurrent_conflicting_freezes_have_one_winner_and_survive_reopen() {
    let temp = tempfile::tempdir().unwrap();
    let db = temp.path().join("state.sqlite");
    let c = Connection::open(&db).unwrap(); setup(&c); drop(c);
    let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
    let workers: Vec<_> = (0..2).map(|i| {
        let (db, barrier) = (db.clone(), barrier.clone());
        std::thread::spawn(move || {
            let mut c = Connection::open(db).unwrap();
            c.busy_timeout(Duration::from_secs(10)).unwrap();
            let mut p = protocol(); if i == 1 { p.metric = Metric::Mae; }
            barrier.wait(); freeze(&mut c, "p", "d", p)
        })
    }).collect();
    let results: Vec<_> = workers.into_iter().map(|t| t.join().unwrap()).collect();
    assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
    let c = Connection::open(db).unwrap();
    let (_, hash) = load(&c, "p", "d").unwrap();
    assert!(results.iter().any(|r| r.as_ref().is_ok_and(|value| value == &hash)));
    c.execute("UPDATE scientific_datasets SET sha256=?1 WHERE project='p' AND id='d'", params!["b".repeat(64)]).unwrap();
    assert!(load(&c, "p", "d").is_err());
}
