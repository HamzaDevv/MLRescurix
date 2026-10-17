use std::{
    collections::BTreeMap,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::json;

use mlrescurix_harness::{
    core::{Operation, Policy, Request},
    execution,
    Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy { max_tasks: 1024, budget: b, capacity: c }
}

#[test]
fn test_schema_v11_migration_and_tables() {
    let td = tempfile::tempdir().unwrap();
    let root = td.path().join("state");
    let policy = test_policy(100);

    // 1. Initialize Runtime (starts at v1 in schema.sql)
    let rt = Runtime::init(&root, policy.clone()).unwrap();

    // 2. Run execution::migrate to bump to user_version = 11
    execution::migrate(&root).unwrap();

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let version: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(version, 11, "PRAGMA user_version must be migrated to 11");

    // 3. Verify all 4 Phase 1 tables exist
    for table in [
        "dataset_topology_contracts",
        "research_ledger_epochs",
        "feature_ablations",
        "critique_receipts",
    ] {
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
                [table],
                |r| r.get(0),
            )
            .unwrap();
        assert!(exists, "Table '{table}' must exist in schema v11");
    }

    // 4. Create project via Runtime to satisfy FK constraint
    let create_res = rt.command(Request {
        command_id: "cmd-p11-init".into(),
        project_id: "p11-test".into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P11 reflexive test".into(),
            finalization_id: "fin-p11".into(),
            policy,
        },
    }).unwrap();
    assert_eq!(create_res.status, "applied");

    let now = now_ms();

    // 5. Test dataset_topology_contracts insertion & query
    conn.execute(
        "INSERT INTO dataset_topology_contracts (
            project, dataset_id, target_column, task_type, id_columns_json,
            entity_group_col, temporal_col, cv_scheme, cv_splits,
            column_classifications_json, clean_feature_columns_json,
            row_count, column_count, contract_json, locked_at_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        params![
            "p11-test",
            "titanic_raw",
            "Survived",
            "binary_classification",
            "[\"PassengerId\"]",
            "Ticket",
            Option::<String>::None,
            "stratified_group_kfold",
            5,
            "{\"PassengerId\":\"DROP_SURROGATE_ID\",\"Name\":\"SHORT_TEXT_EMBED\",\"Ticket\":\"QUASI_ID_DECOMPOSE\"}",
            "[\"Pclass\",\"Sex\",\"Age\",\"SibSp\",\"Parch\",\"Fare\",\"Embarked\"]",
            891,
            12,
            json!({"schema_version": "v2", "locked": true}).to_string(),
            now,
        ],
    ).unwrap();

    let target_col: String = conn.query_row(
        "SELECT target_column FROM dataset_topology_contracts WHERE project=?1 AND dataset_id=?2",
        params!["p11-test", "titanic_raw"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(target_col, "Survived");

    // 6. Test research_ledger_epochs insertion & query
    conn.execute(
        "INSERT INTO research_ledger_epochs (
            project, session_id, epoch, hypotheses_json, accepted_features_json,
            rejected_features_json, best_cv_score, apex_candidate_id, status,
            epoch_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            "p11-test",
            "sess_01",
            0,
            "[\"baseline_anchor\"]",
            "[\"Pclass\",\"Sex\"]",
            "[]",
            0.785,
            "cand_baseline",
            "active",
            json!({"epoch": 0, "status": "active"}).to_string(),
            now,
        ],
    ).unwrap();

    let epoch_status: String = conn.query_row(
        "SELECT status FROM research_ledger_epochs WHERE project=?1 AND session_id=?2 AND epoch=?3",
        params!["p11-test", "sess_01", 0],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(epoch_status, "active");

    // 7. Test feature_ablations insertion & query
    conn.execute(
        "INSERT INTO feature_ablations (
            project, ablation_id, session_id, epoch, hypothesis_id,
            feature_names_json, baseline_cv_score, ablated_cv_score,
            cv_score_delta, train_val_gap_delta, fold_win_count,
            total_folds, status, evaluated_at_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
        params![
            "p11-test",
            "abl_01",
            "sess_01",
            1,
            "hyp_honorifics",
            "[\"Name__honorific\"]",
            0.785,
            0.812,
            0.027,
            0.015,
            4,
            5,
            "ACCEPTED",
            now,
        ],
    ).unwrap();

    let abl_status: String = conn.query_row(
        "SELECT status FROM feature_ablations WHERE project=?1 AND ablation_id=?2",
        params!["p11-test", "abl_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(abl_status, "ACCEPTED");

    // 8. Test critique_receipts insertion & query
    conn.execute(
        "INSERT INTO critique_receipts (
            project, critique_id, candidate_id, session_id, epoch,
            status, blocking_flaws_json, audit_checks_json,
            receipt_json, evaluated_at_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            "p11-test",
            "crit_01",
            "cand_overfit",
            "sess_01",
            1,
            "VETO",
            "[\"Train-val gap exceeded: 0.15 > 0.12\"]",
            json!({"target_memorization": {"passed": false}}).to_string(),
            json!({"status": "VETO", "epoch": 1}).to_string(),
            now,
        ],
    ).unwrap();

    let crit_status: String = conn.query_row(
        "SELECT status FROM critique_receipts WHERE project=?1 AND critique_id=?2",
        params!["p11-test", "crit_01"],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(crit_status, "VETO");
}
