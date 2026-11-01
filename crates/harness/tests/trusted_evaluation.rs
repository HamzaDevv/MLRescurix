use mlrescurix_harness::{
    contracts::MetricSpec,
    trusted_evaluation::{
        check_identity_property, check_permutation_invariance_property,
        check_spec_identity_property, check_spec_permutation_invariance_property, evaluate,
        evaluate_with_spec, evaluate_with_spec_and_evidence, EvaluationIdentity, Metric,
        PredictionBatch, ReferenceEvidence,
    },
};
use sha2::{Digest, Sha256};

fn identity() -> EvaluationIdentity {
    EvaluationIdentity {
        project_id: "project".into(),
        job_id: "job".into(),
        candidate_id: "candidate".into(),
        dataset_id: "dataset".into(),
        dataset_sha256: "a".repeat(64),
        split_sha256: "b".repeat(64),
        source_sha256: "c".repeat(64),
        environment_id: "environment".into(),
        weights_sha256: "d".repeat(64),
    }
}

#[test]
fn computes_metrics_from_labels_and_canonicalizes_row_order() {
    let id = identity();
    let labels = [(2, 1.0), (1, 0.0)];
    let prediction = PredictionBatch {
        identity: id.clone(),
        rows: vec![(1, 0.0), (2, 0.5)],
        vector_rows: vec![],
    };
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
    let mut batch = PredictionBatch {
        identity: id.clone(),
        rows: vec![(1, 0.0), (2, 1.0)],
        vector_rows: vec![],
    };
    for change in 0..9 {
        let mut other = id.clone();
        match change {
            0 => other.project_id.push('x'),
            1 => other.job_id.push('x'),
            2 => other.candidate_id.push('x'),
            3 => other.dataset_sha256 = "1".repeat(64),
            4 => other.split_sha256 = "2".repeat(64),
            5 => other.source_sha256 = "3".repeat(64),
            6 => other.environment_id.push('x'),
            7 => other.weights_sha256 = "4".repeat(64),
            _ => other.dataset_id.push('x'),
        }
        batch.identity = other;
        assert!(evaluate(&id, Metric::Accuracy, &labels, &batch).is_err());
    }
    batch.identity = id.clone();
    for rows in [
        vec![(1, 0.0)],
        vec![(1, 0.0), (1, 1.0)],
        vec![(1, 0.0), (3, 1.0)],
        vec![(1, f64::NAN), (2, 1.0)],
        vec![(1, f64::INFINITY), (2, 1.0)],
    ] {
        batch.rows = rows;
        assert!(evaluate(&id, Metric::Accuracy, &labels, &batch).is_err());
    }
    batch.rows = vec![(1, f64::MAX), (2, 1.0)];
    assert!(evaluate(&id, Metric::Mse, &labels, &batch).is_err());
    assert!(evaluate(&id, Metric::Accuracy, &[(1, 0.0), (1, 1.0)], &batch).is_err());
}

#[test]
fn computes_exact_smape_with_zero_zero_convention() {
    let id = identity();
    // Case 1: perfect predictions -> 0.0
    let labels = [(1, 10.0), (2, 20.0), (3, 0.0)];
    let preds = PredictionBatch::scalar(id.clone(), vec![(1, 10.0), (2, 20.0), (3, 0.0)]);
    let receipt = evaluate(&id, Metric::Smape, &labels, &preds).unwrap();
    assert_eq!(receipt.score, 0.0);

    // Case 2: Hand-calculated SMAPE
    // row 1: y=10, p=15 -> 2*5 / (10+15) = 10/25 = 0.4
    // row 2: y=20, p=20 -> 0.0
    // row 3: y=0,  p=0  -> 0.0 (zero/zero convention)
    // mean = 0.4 / 3 = 0.13333333333333333
    // percentage: * 100 = 13.333333333333334
    let preds2 = PredictionBatch::scalar(id.clone(), vec![(1, 15.0), (2, 20.0), (3, 0.0)]);
    let r2 = evaluate(&id, Metric::Smape, &labels, &preds2).unwrap();
    assert!((r2.score - (100.0 * 0.4 / 3.0)).abs() < 1e-12);

    // Negative values must be rejected
    let bad_preds = PredictionBatch::scalar(id.clone(), vec![(1, -1.0), (2, 20.0), (3, 0.0)]);
    assert!(evaluate(&id, Metric::Smape, &labels, &bad_preds).is_err());
}

#[test]
fn computes_exact_rmsle() {
    let id = identity();
    let labels = [(1, 0.0), (2, 1.0), (3, 3.0)];
    let preds = PredictionBatch::scalar(id.clone(), vec![(1, 0.0), (2, 2.0), (3, 3.0)]);
    // row 1: diff = ln(1) - ln(1) = 0
    // row 2: diff = ln(3) - ln(2) = ln(1.5)
    // row 3: diff = ln(4) - ln(4) = 0
    // mean sq = (ln(1.5))^2 / 3
    // rmsle = ln(1.5) / sqrt(3)
    let expected = (1.5_f64.ln()) / 3.0_f64.sqrt();
    let r = evaluate(&id, Metric::Rmsle, &labels, &preds).unwrap();
    assert!((r.score - expected).abs() < 1e-12);
}

#[test]
fn computes_roc_auc_and_f1() {
    let id = identity();
    let labels = [(1, 0.0), (2, 0.0), (3, 1.0), (4, 1.0)];

    // Perfect ranking: 0.1, 0.2, 0.8, 0.9 -> AUC = 1.0
    let perfect = PredictionBatch::scalar(id.clone(), vec![(1, 0.1), (2, 0.2), (3, 0.8), (4, 0.9)]);
    let r_auc = evaluate(&id, Metric::RocAuc, &labels, &perfect).unwrap();
    assert_eq!(r_auc.score, 1.0);

    // Inverted ranking -> AUC = 0.0
    let inverted = PredictionBatch::scalar(id.clone(), vec![(1, 0.9), (2, 0.8), (3, 0.2), (4, 0.1)]);
    let r_inv = evaluate(&id, Metric::RocAuc, &labels, &inverted).unwrap();
    assert_eq!(r_inv.score, 0.0);

    // Tied ranking -> AUC = 0.5
    let tied = PredictionBatch::scalar(id.clone(), vec![(1, 0.5), (2, 0.5), (3, 0.5), (4, 0.5)]);
    let r_tied = evaluate(&id, Metric::RocAuc, &labels, &tied).unwrap();
    assert_eq!(r_tied.score, 0.5);

    // F1 test: threshold 0.5
    // labels: 0, 0, 1, 1
    // preds:  0.1, 0.6, 0.4, 0.8 -> pred labels: 0, 1, 0, 1
    // row 1: y=0, p=0 (TN)
    // row 2: y=0, p=1 (FP)
    // row 3: y=1, p=0 (FN)
    // row 4: y=1, p=1 (TP)
    // TP=1, FP=1, FN=1 -> F1 = 2*1 / (2*1 + 1 + 1) = 2/4 = 0.5
    let f1_batch = PredictionBatch::scalar(id.clone(), vec![(1, 0.1), (2, 0.6), (3, 0.4), (4, 0.8)]);
    let r_f1 = evaluate(&id, Metric::F1, &labels, &f1_batch).unwrap();
    assert_eq!(r_f1.score, 0.5);
}

#[test]
fn computes_multiclass_log_loss_vector_rows() {
    let id = identity();
    // 3 classes: labels are 0, 1, 2
    let labels = [(1, 0.0), (2, 1.0), (3, 2.0)];
    let vector_batch = PredictionBatch::vector(
        id.clone(),
        vec![
            (1, vec![0.8, 0.1, 0.1]),
            (2, vec![0.1, 0.7, 0.2]),
            (3, vec![0.05, 0.05, 0.9]),
        ],
    );
    let r = evaluate(&id, Metric::LogLoss, &labels, &vector_batch).unwrap();
    let expected = (-(0.8_f64.ln()) - (0.7_f64.ln()) - (0.9_f64.ln())) / 3.0;
    assert!((r.score - expected).abs() < 1e-12);
}

#[test]
fn log_loss_rejects_invalid_probabilities_and_class_labels() {
    let id = identity();
    for p in [-0.1, 1.1] {
        let batch = PredictionBatch::scalar(id.clone(), vec![(1, p)]);
        assert!(evaluate(&id, Metric::LogLoss, &[(1, 1.0)], &batch).is_err());
    }
    for probabilities in [vec![0.8, 0.8], vec![-0.1, 1.1]] {
        let batch = PredictionBatch::vector(id.clone(), vec![(1, probabilities)]);
        assert!(evaluate(&id, Metric::LogLoss, &[(1, 1.0)], &batch).is_err());
    }
    let batch = PredictionBatch::vector(id.clone(), vec![(1, vec![0.5, 0.5])]);
    for label in [-1.0, 0.5, 2.0] {
        assert!(evaluate(&id, Metric::LogLoss, &[(1, label)], &batch).is_err());
    }
}

#[test]
fn metamorphic_property_checks_pass() {
    let labels = [(1, 0.0), (2, 1.0), (3, 1.0), (4, 0.0)];
    let id = identity();

    // Check identity for Accuracy
    let id_acc = check_identity_property(Metric::Accuracy, &labels).unwrap();
    assert!(id_acc.applicable);
    assert!(id_acc.passed);

    // Check identity for SMAPE
    let reg_labels = [(1, 5.0), (2, 10.0), (3, 25.0)];
    let id_smape = check_identity_property(Metric::Smape, &reg_labels).unwrap();
    assert!(id_smape.applicable);
    assert!(id_smape.passed);

    // Check permutation invariance
    let preds = PredictionBatch::scalar(id.clone(), vec![(1, 0.2), (2, 0.8), (3, 0.9), (4, 0.1)]);
    let perm = check_permutation_invariance_property(Metric::RocAuc, &labels, &preds).unwrap();
    assert!(perm.applicable);
    assert!(perm.passed);
}

#[test]
fn spec_based_evaluation_f1_threshold_and_signed_smape() {
    let id = identity();

    // 1. Dynamic F1 threshold: prediction of 0.6 is positive at threshold 0.5, but negative at threshold 0.8
    let f1_labels = [(1, 1.0), (2, 0.0)];
    let f1_preds = PredictionBatch::scalar(id.clone(), vec![(1, 0.6), (2, 0.1)]);

    let spec_f1_05 = MetricSpec {
        name: "f1".into(),
        direction: "maximize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: Some(0.5),
        variant: None,
    };
    let r_f1_05 = evaluate_with_spec(&id, &spec_f1_05, &f1_labels, &f1_preds).unwrap();
    assert_eq!(r_f1_05.score, 1.0); // TP=1, FP=0, FN=0 -> F1=1.0

    let spec_f1_08 = MetricSpec {
        name: "f1".into(),
        direction: "maximize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: Some(0.8),
        variant: None,
    };
    let r_f1_08 = evaluate_with_spec(&id, &spec_f1_08, &f1_labels, &f1_preds).unwrap();
    assert_eq!(r_f1_08.score, 0.0); // TP=0, FP=0, FN=1 -> F1=0.0

    // 2. Signed SMAPE: supports negative values when not restricted
    let signed_labels = [(1, -10.0), (2, 20.0)];
    let signed_preds = PredictionBatch::scalar(id.clone(), vec![(1, -15.0), (2, 20.0)]);
    let spec_signed_smape = MetricSpec {
        name: "smape".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: Some("signed".into()),
    };
    // row 1: 2*|-15 - (-10)| / (|-10| + |-15|) = 2*5 / 25 = 0.4
    // row 2: 0.0
    // mean = 0.4 / 2 = 0.2 -> 20.0%
    let r_signed = evaluate_with_spec(&id, &spec_signed_smape, &signed_labels, &signed_preds).unwrap();
    assert!((r_signed.score - 20.0).abs() < 1e-12);

    // 3. SMAPE non_negative_only variant: rejects negative values
    let spec_non_neg = MetricSpec {
        name: "smape".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: Some("non_negative_only".into()),
    };
    assert!(evaluate_with_spec(&id, &spec_non_neg, &signed_labels, &signed_preds).is_err());

    // 4. Evidence-bound assurance tiers
    // Plain unverified evaluation
    let plain_receipt = evaluate(&id, Metric::Accuracy, &[(1, 1.0)], &PredictionBatch::scalar(id.clone(), vec![(1, 1.0)])).unwrap();
    assert_eq!(plain_receipt.assurance, "unverified_reference");

    // Spec evaluation passing properties gets provisional_spec_checked
    assert_eq!(r_f1_05.assurance, "provisional_spec_checked");

    // Caller-supplied evidence cannot certify an independent reference.
    let fixture_sha = format!(
        "{:x}",
        Sha256::digest(format!("{}:{}", r_f1_05.labels_sha256, r_f1_05.predictions_sha256).as_bytes())
    );
    let valid_evidence = ReferenceEvidence {
        reference_identity: "sklearn_v1_f1_reference".into(),
        fixture_sha256: fixture_sha,
        expected_result: 1.0,
        verified_receipt_sha256: "1".repeat(64),
        metric_version: "v1.0".into(),
        tolerance: 1e-6,
    };
    let conf_receipt = evaluate_with_spec_and_evidence(&id, &spec_f1_05, &f1_labels, &f1_preds, Some(&valid_evidence)).unwrap();
    assert_eq!(conf_receipt.assurance, "provisional_spec_checked");
    assert!(conf_receipt.reference_evidence.is_none());

    // Invalid evidence fixture hash mismatch fails closed
    let mut bad_evidence = valid_evidence.clone();
    bad_evidence.fixture_sha256 = "f".repeat(64);
    let err_bad_ev = evaluate_with_spec_and_evidence(&id, &spec_f1_05, &f1_labels, &f1_preds, Some(&bad_evidence)).unwrap_err();
    assert!(err_bad_ev.contains("does not match evaluated fixture digest"));

    // Diverging score exceeding tolerance fails closed
    let mut bad_score_evidence = valid_evidence.clone();
    bad_score_evidence.expected_result = 0.5;
    let err_bad_score = evaluate_with_spec_and_evidence(&id, &spec_f1_05, &f1_labels, &f1_preds, Some(&bad_score_evidence)).unwrap_err();
    assert!(err_bad_score.contains("diverges from evaluated score"));

    // Invalid format fails validation
    let mut bad_format_evidence = valid_evidence.clone();
    bad_format_evidence.fixture_sha256 = "short_hash".into();
    let err_bad_fmt = evaluate_with_spec_and_evidence(&id, &spec_f1_05, &f1_labels, &f1_preds, Some(&bad_format_evidence)).unwrap_err();
    assert!(err_bad_fmt.contains("invalid fixture_sha256 in ReferenceEvidence"));

    // Spec-aware property checks
    let spec_id_acc = check_spec_identity_property(&spec_f1_05, &f1_labels).unwrap();
    assert!(spec_id_acc.applicable);
    assert!(spec_id_acc.passed);

    let spec_perm = check_spec_permutation_invariance_property(&spec_f1_05, &f1_labels, &f1_preds).unwrap();
    assert!(spec_perm.applicable);
    assert!(spec_perm.passed);
}

#[test]
fn roc_auc_exact_float_ranking_without_tie_collapsing() {
    let id = identity();
    // Two close but strictly distinct predictions
    let labels = [(1, 1.0), (2, 0.0)];
    // Row 1 score is slightly higher than Row 2 (by 1e-16), so positive > negative
    let preds_distinct = PredictionBatch::scalar(id.clone(), vec![(1, 0.5000000000000002), (2, 0.5)]);
    let receipt = evaluate(&id, Metric::RocAuc, &labels, &preds_distinct).unwrap();
    // Exact ranking ranks 0.5 < 0.5000000000000002, so pos is rank 2, neg is rank 1 -> AUC = 1.0
    assert_eq!(receipt.score, 1.0);

    // If genuinely identical (tied), rank is averaged -> AUC = 0.5
    let preds_tied = PredictionBatch::scalar(id.clone(), vec![(1, 0.5), (2, 0.5)]);
    let receipt_tied = evaluate(&id, Metric::RocAuc, &labels, &preds_tied).unwrap();
    assert_eq!(receipt_tied.score, 0.5);
}

#[test]
fn custom_and_robust_loss_metrics_cauchy_huber_and_custom() {
    let id = identity();
    let labels = [(1, 2.0), (2, 5.0), (3, 10.0)];
    let preds_perfect = PredictionBatch::scalar(id.clone(), vec![(1, 2.0), (2, 5.0), (3, 10.0)]);
    let preds_noisy = PredictionBatch::scalar(id.clone(), vec![(1, 2.5), (2, 4.0), (3, 12.0)]);

    // 1. Cauchy Loss
    let receipt_cauchy_perf = evaluate(&id, Metric::Cauchy, &labels, &preds_perfect).unwrap();
    assert_eq!(receipt_cauchy_perf.score, 0.0);
    let receipt_cauchy_noisy = evaluate(&id, Metric::Cauchy, &labels, &preds_noisy).unwrap();
    assert!(receipt_cauchy_noisy.score > 0.0);

    let spec_cauchy = MetricSpec {
        name: "cauchy".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let prop_cauchy = check_spec_identity_property(&spec_cauchy, &labels).unwrap();
    assert!(prop_cauchy.passed);

    // 2. Pseudo-Huber Loss
    let receipt_huber_perf = evaluate(&id, Metric::Huber, &labels, &preds_perfect).unwrap();
    assert_eq!(receipt_huber_perf.score, 0.0);
    let receipt_huber_noisy = evaluate(&id, Metric::Huber, &labels, &preds_noisy).unwrap();
    assert!(receipt_huber_noisy.score > 0.0);

    let spec_huber = MetricSpec {
        name: "huber".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let prop_huber = check_spec_identity_property(&spec_huber, &labels).unwrap();
    assert!(prop_huber.passed);

    // 3. Custom Metric name (unclosed enum)
    let spec_custom = MetricSpec {
        name: "custom_loss_fn".into(),
        direction: "minimize".into(),
        class_order: None,
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let err = evaluate_with_spec(&id, &spec_custom, &labels, &preds_noisy).unwrap_err();
    assert!(err.contains("requires an admitted evaluator"));
}

#[test]
fn multiclass_class_order_validation() {
    let id = identity();
    let labels = [(1, 0.0), (2, 1.0)];

    // Vector predictions with dimension 2
    let preds_vec = PredictionBatch::vector(
        id.clone(),
        vec![(1, vec![0.8, 0.2]), (2, vec![0.3, 0.7])],
    );

    // Class order matching dimension passes
    let spec_matching = MetricSpec {
        name: "log_loss".into(),
        direction: "minimize".into(),
        class_order: Some(vec!["cat".into(), "dog".into()]),
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let receipt = evaluate_with_spec(&id, &spec_matching, &labels, &preds_vec).unwrap();
    assert_eq!(receipt.class_order, Some(vec!["cat".into(), "dog".into()]));

    // Class order mismatching dimension fails closed
    let spec_mismatch = MetricSpec {
        name: "log_loss".into(),
        direction: "minimize".into(),
        class_order: Some(vec!["cat".into(), "dog".into(), "fish".into()]),
        epsilon: None,
        decision_threshold: None,
        variant: None,
    };
    let err = evaluate_with_spec(&id, &spec_mismatch, &labels, &preds_vec).unwrap_err();
    assert!(err.contains("does not match frozen class_order length"));
}
