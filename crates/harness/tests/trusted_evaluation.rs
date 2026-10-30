use mlrescurix_harness::trusted_evaluation::{evaluate, EvaluationIdentity, Metric, PredictionBatch};

fn identity() -> EvaluationIdentity {
    EvaluationIdentity {
        project_id: "project".into(), job_id: "job".into(), candidate_id: "candidate".into(), dataset_id: "dataset".into(),
        dataset_sha256: "a".repeat(64), split_sha256: "b".repeat(64),
        source_sha256: "c".repeat(64), environment_id: "environment".into(), weights_sha256: "d".repeat(64),
    }
}

#[test]
fn computes_metrics_from_labels_and_canonicalizes_row_order() {
    let id = identity();
    let labels = [(2, 1.0), (1, 0.0)];
    let prediction = PredictionBatch { identity: id.clone(), rows: vec![(1, 0.0), (2, 0.5)] };
    let receipt = evaluate(&id, Metric::Accuracy, &labels, &prediction).unwrap();
    assert_eq!(receipt.score, 0.5);
    assert_eq!(evaluate(&id, Metric::Mae, &labels, &prediction).unwrap().score, 0.25);
    assert_eq!(evaluate(&id, Metric::Mse, &labels, &prediction).unwrap().score, 0.125);
    assert!((evaluate(&id, Metric::Rmse, &labels, &prediction).unwrap().score - 0.125_f64.sqrt()).abs() < 1e-12);
    let reordered = evaluate(&id, Metric::Accuracy, &[(1, 0.0), (2, 1.0)], &prediction).unwrap();
    assert_eq!(receipt.labels_sha256, reordered.labels_sha256);
}

#[test]
fn rejects_cross_scope_and_incomplete_or_fabricated_rows() {
    let id = identity();
    let labels = [(1, 0.0), (2, 1.0)];
    let mut batch = PredictionBatch { identity: id.clone(), rows: vec![(1, 0.0), (2, 1.0)] };
    for change in 0..9 {
        let mut other = id.clone();
        match change {
            0 => other.project_id.push('x'), 1 => other.job_id.push('x'), 2 => other.candidate_id.push('x'),
            3 => other.dataset_sha256 = "1".repeat(64), 4 => other.split_sha256 = "2".repeat(64),
            5 => other.source_sha256 = "3".repeat(64), 6 => other.environment_id.push('x'),
            7 => other.weights_sha256 = "4".repeat(64), _ => other.dataset_id.push('x'),
        }
        batch.identity = other;
        assert!(evaluate(&id, Metric::Accuracy, &labels, &batch).is_err());
    }
    batch.identity = id.clone();
    for rows in [vec![(1, 0.0)], vec![(1, 0.0), (1, 1.0)], vec![(1, 0.0), (3, 1.0)], vec![(1, f64::NAN), (2, 1.0)], vec![(1, f64::INFINITY), (2, 1.0)]] {
        batch.rows = rows;
        assert!(evaluate(&id, Metric::Accuracy, &labels, &batch).is_err());
    }
    batch.rows = vec![(1, f64::MAX), (2, 1.0)];
    assert!(evaluate(&id, Metric::Mse, &labels, &batch).is_err());
    assert!(evaluate(&id, Metric::Accuracy, &[(1, 0.0), (1, 1.0)], &batch).is_err());
}
