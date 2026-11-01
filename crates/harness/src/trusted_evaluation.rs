//! Metric computation for a trusted controller with separately held labels.
//!
//! This primitive is not an admission or promotion authority. The caller must
//! supply controller-owned labels and an independently bound inference result;
//! worker-provided labels or identity claims are not authoritative.
use crate::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

const MAX_ROWS: usize = 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationIdentity {
    pub project_id: String,
    pub job_id: String,
    pub candidate_id: String,
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub split_sha256: String,
    pub source_sha256: String,
    pub environment_id: String,
    pub weights_sha256: String,
}

impl EvaluationIdentity {
    pub fn validate(&self) -> Result<()> {
        for id in [&self.project_id, &self.job_id, &self.candidate_id, &self.dataset_id, &self.environment_id] {
            if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
                return Err("invalid evaluation identity".into());
            }
        }
        for digest in [&self.dataset_sha256, &self.split_sha256, &self.source_sha256, &self.weights_sha256] {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err("invalid evaluation sha256".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Accuracy,
    Mae,
    Mse,
    Rmse,
    Smape,
    Rmsle,
    LogLoss,
    RocAuc,
    F1,
    Cauchy,
    Huber,
    #[serde(untagged)]
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBatch {
    pub identity: EvaluationIdentity,
    #[serde(default)]
    pub rows: Vec<(u64, f64)>,
    #[serde(default)]
    pub vector_rows: Vec<(u64, Vec<f64>)>,
}

impl PredictionBatch {
    pub fn scalar(identity: EvaluationIdentity, rows: Vec<(u64, f64)>) -> Self {
        Self {
            identity,
            rows,
            vector_rows: Vec::new(),
        }
    }

    pub fn vector(identity: EvaluationIdentity, vector_rows: Vec<(u64, Vec<f64>)>) -> Self {
        Self {
            identity,
            rows: Vec::new(),
            vector_rows,
        }
    }
}

fn default_metric_version() -> String {
    "v1".into()
}

fn default_tolerance() -> f64 {
    1e-6
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReferenceEvidence {
    pub reference_identity: String,
    pub fixture_sha256: String,
    pub expected_result: f64,
    pub verified_receipt_sha256: String,
    #[serde(default = "default_metric_version")]
    pub metric_version: String,
    #[serde(default = "default_tolerance")]
    pub tolerance: f64,
}

impl ReferenceEvidence {
    pub fn validate(&self) -> Result<()> {
        if self.reference_identity.is_empty() || self.reference_identity.len() > 256 {
            return Err("invalid reference_identity in ReferenceEvidence".into());
        }
        for (name, sha) in [("fixture_sha256", &self.fixture_sha256), ("verified_receipt_sha256", &self.verified_receipt_sha256)] {
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
                return Err(format!("invalid {name} in ReferenceEvidence; expected 64 hex chars"));
            }
        }
        if !self.expected_result.is_finite() {
            return Err("expected_result in ReferenceEvidence must be finite".into());
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err("tolerance in ReferenceEvidence must be finite and non-negative".into());
        }
        Ok(())
    }
}

fn default_assurance_str() -> String {
    "unverified_reference".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationReceipt {
    pub identity: EvaluationIdentity,
    pub metric: Metric,
    pub score: f64,
    pub row_count: usize,
    pub labels_sha256: String,
    pub predictions_sha256: String,
    #[serde(default = "default_assurance_str")]
    pub assurance: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_order: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_evidence: Option<ReferenceEvidence>,
}

fn canonical_rows(rows: &[(u64, f64)], name: &str) -> Result<BTreeMap<u64, f64>> {
    if rows.is_empty() || rows.len() > MAX_ROWS {
        return Err(format!("{name} row count outside bounds"));
    }
    let mut map = BTreeMap::new();
    for &(id, value) in rows {
        if !value.is_finite() {
            return Err(format!("non-finite {name}"));
        }
        if map.insert(id, value).is_some() {
            return Err(format!("duplicate {name} row id"));
        }
    }
    Ok(map)
}

fn canonical_vector_rows(rows: &[(u64, Vec<f64>)], name: &str) -> Result<BTreeMap<u64, Vec<f64>>> {
    if rows.is_empty() || rows.len() > MAX_ROWS {
        return Err(format!("{name} row count outside bounds"));
    }
    let mut map = BTreeMap::new();
    let mut expected_dim: Option<usize> = None;
    for (id, values) in rows {
        if values.is_empty() {
            return Err(format!("empty vector in {name}"));
        }
        if let Some(dim) = expected_dim {
            if values.len() != dim {
                return Err(format!("inconsistent vector dimension in {name}"));
            }
        } else {
            expected_dim = Some(values.len());
        }
        for &v in values {
            if !v.is_finite() {
                return Err(format!("non-finite value in {name}"));
            }
        }
        if map.insert(*id, values.clone()).is_some() {
            return Err(format!("duplicate {name} row id"));
        }
    }
    Ok(map)
}

fn digest_rows(rows: &BTreeMap<u64, f64>) -> String {
    let mut digest = Sha256::new();
    for (&id, &value) in rows {
        digest.update(id.to_be_bytes());
        digest.update((if value == 0.0 { 0.0_f64 } else { value }).to_bits().to_be_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn digest_vector_rows(rows: &BTreeMap<u64, Vec<f64>>) -> String {
    let mut digest = Sha256::new();
    for (&id, values) in rows {
        digest.update(id.to_be_bytes());
        digest.update((values.len() as u64).to_be_bytes());
        for &value in values {
            digest.update((if value == 0.0 { 0.0_f64 } else { value }).to_bits().to_be_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

fn compute_roc_auc(labels: &BTreeMap<u64, f64>, values: &BTreeMap<u64, f64>) -> Result<f64> {
    // Collect (score, label) pairs
    let mut pairs = Vec::with_capacity(labels.len());
    let mut pos_count = 0usize;
    let mut neg_count = 0usize;
    for (id, &label) in labels {
        if label == 1.0 {
            pos_count += 1;
        } else if label == 0.0 {
            neg_count += 1;
        } else {
            return Err("ROC-AUC requires binary labels {0.0, 1.0}".into());
        }
        let score = values[id];
        pairs.push((score, label));
    }
    if pos_count == 0 || neg_count == 0 {
        return Err("ROC-AUC requires both positive and negative examples".into());
    }

    // Sort by score ascending
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Handle ties with average ranks (1-based ranking)
    let n = pairs.len();
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        // Average rank for indices i..j is: (sum of r from i+1 to j) / (j - i) = (i + 1 + j) / 2.0
        let avg_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            if pairs[k].1 == 1.0 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }

    let n1 = pos_count as f64;
    let n0 = neg_count as f64;
    let u1 = rank_sum_pos - (n1 * (n1 + 1.0)) / 2.0;
    let auc = u1 / (n0 * n1);
    Ok(auc.clamp(0.0, 1.0))
}

fn compute_f1(labels: &BTreeMap<u64, f64>, values: &BTreeMap<u64, f64>, threshold: f64) -> Result<f64> {
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_count = 0.0;
    for (id, &label) in labels {
        let is_true_pos = if label == 1.0 {
            true
        } else if label == 0.0 {
            false
        } else {
            return Err("F1 metric requires binary labels {0.0, 1.0}".into());
        };
        let is_pred_pos = values[id] >= threshold;
        match (is_true_pos, is_pred_pos) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fn_count += 1.0,
            (false, false) => {}
        }
    }
    let denom = 2.0 * tp + fp + fn_count;
    if denom == 0.0 {
        Ok(0.0)
    } else {
        Ok((2.0 * tp) / denom)
    }
}

pub trait CustomMetricEvaluator: Send + Sync {
    fn evaluate(&self, labels: &BTreeMap<u64, f64>, predictions: &BTreeMap<u64, f64>, epsilon: Option<f64>) -> Result<f64>;
}

static ADMITTED_CUSTOM_EVALUATORS: std::sync::RwLock<Option<std::collections::HashMap<String, std::sync::Arc<dyn CustomMetricEvaluator>>>> = std::sync::RwLock::new(None);

pub fn register_admitted_custom_evaluator(name: &str, evaluator: std::sync::Arc<dyn CustomMetricEvaluator>) {
    let mut reg = ADMITTED_CUSTOM_EVALUATORS.write().unwrap();
    let map = reg.get_or_insert_with(std::collections::HashMap::new);
    map.insert(name.to_lowercase(), evaluator);
}

/// Compute a score from controller-owned labels and predictions. Neither the
/// identity equality check nor a receipt proves that a worker ran the bound
/// model; the supervisor must independently enforce that inference binding.
pub fn evaluate(
    expected: &EvaluationIdentity,
    metric: Metric,
    trusted_labels: &[(u64, f64)],
    predictions: &PredictionBatch,
) -> Result<EvaluationReceipt> {
    evaluate_internal(expected, metric, trusted_labels, predictions, 0.5, true, &default_assurance_str(), None, None)
}

pub fn parse_metric_name(name: &str) -> Result<Metric> {
    match name.to_lowercase().as_str() {
        "accuracy" => Ok(Metric::Accuracy),
        "mae" => Ok(Metric::Mae),
        "mse" => Ok(Metric::Mse),
        "rmse" => Ok(Metric::Rmse),
        "smape" => Ok(Metric::Smape),
        "rmsle" => Ok(Metric::Rmsle),
        "log_loss" | "logloss" => Ok(Metric::LogLoss),
        "roc_auc" | "roc-auc" | "auc" => Ok(Metric::RocAuc),
        "f1" | "f1_score" => Ok(Metric::F1),
        "cauchy" | "cauchy_loss" => Ok(Metric::Cauchy),
        "huber" | "pseudo_huber" => Ok(Metric::Huber),
        other => Ok(Metric::Custom(other.to_string())),
    }
}

/// Compute a score adhering to an authoritative MetricSpec (direction, variant, decision_threshold).
pub fn evaluate_with_spec(
    expected: &EvaluationIdentity,
    spec: &crate::contracts::MetricSpec,
    trusted_labels: &[(u64, f64)],
    predictions: &PredictionBatch,
) -> Result<EvaluationReceipt> {
    evaluate_with_spec_and_evidence(expected, spec, trusted_labels, predictions, None)
}

/// Evaluate a spec and check supplied fixture consistency without certifying its source.
pub fn evaluate_with_spec_and_evidence(
    expected: &EvaluationIdentity,
    spec: &crate::contracts::MetricSpec,
    trusted_labels: &[(u64, f64)],
    predictions: &PredictionBatch,
    evidence: Option<&ReferenceEvidence>,
) -> Result<EvaluationReceipt> {
    spec.validate()?;
    let metric = parse_metric_name(&spec.name)?;

    let threshold = spec.decision_threshold.unwrap_or(0.5);
    let non_negative_only = spec.variant.as_deref() == Some("non_negative_only");

    let mut receipt = evaluate_internal(
        expected,
        metric,
        trusted_labels,
        predictions,
        threshold,
        non_negative_only,
        "unverified_reference",
        spec.class_order.as_deref(),
        spec.epsilon,
    )?;

    if let Some(ev) = evidence {
        ev.validate()?;
        // Cryptographically verify fixture evidence against evaluated inputs
        let fixture_sha = format!(
            "{:x}",
            Sha256::digest(format!("{}:{}", receipt.labels_sha256, receipt.predictions_sha256).as_bytes())
        );
        if ev.fixture_sha256 != fixture_sha && ev.fixture_sha256 != receipt.labels_sha256 {
            return Err(format!(
                "ReferenceEvidence fixture_sha256 '{}' does not match evaluated fixture digest '{}'",
                ev.fixture_sha256, fixture_sha
            ));
        }
        let diff = (receipt.score - ev.expected_result).abs();
        if diff > ev.tolerance {
            return Err(format!(
                "ReferenceEvidence expected_result {} diverges from evaluated score {} (diff {} > tolerance {})",
                ev.expected_result, receipt.score, diff, ev.tolerance
            ));
        }

        // Matching caller-provided evidence cannot establish an independent reference.

    }
    if receipt.assurance != "reference_conformant" {
        let perm_ok = check_spec_permutation_invariance_property(spec, trusted_labels, predictions)
            .map(|r| r.passed)
            .unwrap_or(false);
        if perm_ok {
            receipt.assurance = "provisional_spec_checked".into();
        }
    }

    Ok(receipt)
}

fn evaluate_internal(
    expected: &EvaluationIdentity,
    metric: Metric,
    trusted_labels: &[(u64, f64)],
    predictions: &PredictionBatch,
    f1_threshold: f64,
    smape_non_negative_only: bool,
    assurance: &str,
    class_order: Option<&[String]>,
    epsilon: Option<f64>,
) -> Result<EvaluationReceipt> {
    expected.validate()?;
    if &predictions.identity != expected {
        return Err("evaluation identity mismatch".into());
    }
    let labels = canonical_rows(trusted_labels, "label")?;

    let is_vector = !predictions.vector_rows.is_empty();
    if is_vector && !predictions.rows.is_empty() {
        return Err("prediction batch contains both scalar and vector rows; choose one".into());
    }

    let (score, pred_sha256) = if is_vector {
        let vec_values = canonical_vector_rows(&predictions.vector_rows, "prediction vector")?;
        if labels.keys().ne(vec_values.keys()) {
            return Err("prediction row ids do not match validation rows".into());
        }
        if let Some(classes) = class_order {
            if classes.len() < 2 {
                return Err("class_order must contain at least 2 classes".into());
            }
            let class_set: std::collections::HashSet<_> = classes.iter().collect();
            if class_set.len() != classes.len() {
                return Err("class_order contains duplicate class names".into());
            }
            for (_id, vec) in &vec_values {
                if vec.len() != classes.len() {
                    return Err(format!(
                        "prediction vector dimension {} does not match frozen class_order length {}",
                        vec.len(), classes.len()
                    ));
                }
            }
        }
        let pred_sha = digest_vector_rows(&vec_values);

        let mut sum = 0.0;
        match metric {
            Metric::Accuracy => {
                for (id, label) in &labels {
                    let probs = &vec_values[id];
                    let mut max_idx = 0usize;
                    let mut max_val = probs[0];
                    for (idx, &v) in probs.iter().enumerate().skip(1) {
                        if v > max_val {
                            max_val = v;
                            max_idx = idx;
                        }
                    }
                    if *label == max_idx as f64 {
                        sum += 1.0;
                    }
                }
                (sum / labels.len() as f64, pred_sha)
            }
            Metric::LogLoss => {
                const EPS: f64 = 1e-15;
                for (id, label) in &labels {
                    let probs = &vec_values[id];
                    if *label < 0.0 || label.fract() != 0.0 {
                        return Err("vector LogLoss requires non-negative integer class labels".into());
                    }
                    if probs.iter().any(|p| !(0.0..=1.0).contains(p))
                        || (probs.iter().sum::<f64>() - 1.0).abs() > 1e-9
                    {
                        return Err("vector LogLoss requires normalized probabilities in [0, 1]".into());
                    }
                    let class_idx = *label as usize;
                    if class_idx >= probs.len() {
                        return Err(format!("label class index {class_idx} out of range for vector dim {}", probs.len()));
                    }
                    let p = probs[class_idx].clamp(EPS, 1.0);
                    sum += -p.ln();
                }
                (sum / labels.len() as f64, pred_sha)
            }
            _ => {
                return Err(format!("metric '{metric:?}' does not support vector predictions; use scalar rows"));
            }
        }
    } else {
        let values = canonical_rows(&predictions.rows, "prediction")?;
        if labels.keys().ne(values.keys()) {
            return Err("prediction row ids do not match validation rows".into());
        }
        let pred_sha = digest_rows(&values);

        let score = match metric {
            Metric::Accuracy => {
                let mut sum = 0.0;
                for (id, label) in &labels {
                    if *label == values[id] {
                        sum += 1.0;
                    }
                }
                sum / labels.len() as f64
            }
            Metric::Mae => {
                let mut sum = 0.0;
                for (id, label) in &labels {
                    sum += (values[id] - label).abs();
                }
                sum / labels.len() as f64
            }
            Metric::Mse => {
                let mut sum = 0.0;
                for (id, label) in &labels {
                    let diff = values[id] - label;
                    sum += diff * diff;
                    if !sum.is_finite() {
                        return Err("metric overflow".into());
                    }
                }
                sum / labels.len() as f64
            }
            Metric::Rmse => {
                let mut sum = 0.0;
                for (id, label) in &labels {
                    let diff = values[id] - label;
                    sum += diff * diff;
                    if !sum.is_finite() {
                        return Err("metric overflow".into());
                    }
                }
                (sum / labels.len() as f64).sqrt()
            }
            Metric::Smape => {
                // Section 7.2 formula:
                // SMAPE(y, p) = (100 / N) * sum( 2 |p_i - y_i| / (|y_i| + |p_i|) )
                // Zero/zero contributes zero. Supports signed values unless non_negative_only is specified.
                let mut sum = 0.0;
                for (id, &label) in &labels {
                    let p = values[id];
                    if smape_non_negative_only && (label < 0.0 || p < 0.0) {
                        return Err("SMAPE metric requires non-negative labels and predictions".into());
                    }
                    if label == 0.0 && p == 0.0 {
                        continue;
                    }
                    let denom = label.abs() + p.abs();
                    sum += (2.0 * (p - label).abs()) / denom;
                }
                (100.0 / labels.len() as f64) * sum
            }
            Metric::Rmsle => {
                // RMSLE = sqrt( 1/N * sum( (ln(p + 1) - ln(y + 1))^2 ) )
                let mut sum = 0.0;
                for (id, &label) in &labels {
                    let p = values[id];
                    if label < 0.0 || p < 0.0 {
                        return Err("RMSLE metric requires non-negative labels and predictions".into());
                    }
                    let diff = (p + 1.0).ln() - (label + 1.0).ln();
                    sum += diff * diff;
                }
                (sum / labels.len() as f64).sqrt()
            }
            Metric::LogLoss => {
                const EPS: f64 = 1e-15;
                let mut sum = 0.0;
                for (id, &label) in &labels {
                    if label != 0.0 && label != 1.0 {
                        return Err("scalar LogLoss requires binary labels {0.0, 1.0}".into());
                    }
                    if !(0.0..=1.0).contains(&values[id]) {
                        return Err("scalar LogLoss requires probabilities in [0, 1]".into());
                    }
                    let p = values[id].clamp(EPS, 1.0 - EPS);
                    sum += -(label * p.ln() + (1.0 - label) * (1.0 - p).ln());
                }
                sum / labels.len() as f64
            }
            Metric::RocAuc => compute_roc_auc(&labels, &values)?,
            Metric::F1 => compute_f1(&labels, &values, f1_threshold)?,
            Metric::Cauchy => {
                let gamma = epsilon.unwrap_or(1.0);
                if gamma <= 0.0 || !gamma.is_finite() {
                    return Err("Cauchy loss epsilon (gamma) must be strictly positive and finite".into());
                }
                let mut sum = 0.0;
                for (id, &label) in &labels {
                    let p = values[id];
                    let r = p - label;
                    sum += (gamma * gamma / 2.0) * (1.0 + (r / gamma).powi(2)).ln();
                }
                sum / labels.len() as f64
            }
            Metric::Huber => {
                let delta = epsilon.unwrap_or(1.0);
                if delta <= 0.0 || !delta.is_finite() {
                    return Err("Huber loss epsilon (delta) must be strictly positive and finite".into());
                }
                let mut sum = 0.0;
                for (id, &label) in &labels {
                    let p = values[id];
                    let r = p - label;
                    sum += delta * delta * ((1.0 + (r / delta).powi(2)).sqrt() - 1.0);
                }
                sum / labels.len() as f64
            }
            Metric::Custom(ref name) => {
                let reg = ADMITTED_CUSTOM_EVALUATORS.read().unwrap();
                if let Some(evaluator) = reg.as_ref().and_then(|m| m.get(&name.to_lowercase())) {
                    evaluator.evaluate(&labels, &values, epsilon)?
                } else {
                    return Err(format!("custom metric '{name}' requires an admitted evaluator"));
                }
            }
        };
        (score, pred_sha)
    };

    if !score.is_finite() {
        return Err("metric produced non-finite score".into());
    }

    Ok(EvaluationReceipt {
        identity: expected.clone(),
        metric,
        score,
        row_count: labels.len(),
        labels_sha256: digest_rows(&labels),
        predictions_sha256: pred_sha256,
        assurance: assurance.to_string(),
        class_order: class_order.map(|c| c.to_vec()),
        reference_evidence: None,
    })
}

// ==================== Metamorphic Property Checks ====================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvaluatorPropertyCheckResult {
    pub property_name: String,
    pub applicable: bool,
    pub passed: bool,
    pub tolerance: f64,
    pub detail: String,
}

/// Test identity property: when predictions match true labels, metric attains optimum.
pub fn check_identity_property(metric: Metric, labels: &[(u64, f64)]) -> Result<EvaluatorPropertyCheckResult> {
    let dummy_id = EvaluationIdentity {
        project_id: "prop_test".into(),
        job_id: "prop_job".into(),
        candidate_id: "prop_cand".into(),
        dataset_id: "prop_data".into(),
        dataset_sha256: "0".repeat(64),
        split_sha256: "0".repeat(64),
        source_sha256: "0".repeat(64),
        environment_id: "prop_env".into(),
        weights_sha256: "0".repeat(64),
    };

    let (applicable, expected_score, tolerance) = match metric {
        Metric::Accuracy | Metric::F1 => (true, 1.0, 1e-9),
        Metric::Mae | Metric::Mse | Metric::Rmse | Metric::Smape | Metric::Rmsle | Metric::Cauchy | Metric::Huber => (true, 0.0, 1e-9),
        Metric::Custom(ref name) => {
            let reg = ADMITTED_CUSTOM_EVALUATORS.read().unwrap();
            if reg.as_ref().and_then(|m| m.get(&name.to_lowercase())).is_some() {
                (true, 0.0, 1e-9)
            } else {
                return Ok(EvaluatorPropertyCheckResult {
                    property_name: "identity_at_optimum".into(),
                    applicable: false,
                    passed: false,
                    tolerance: 1e-9,
                    detail: format!("custom metric '{name}' requires an admitted evaluator"),
                });
            }
        }
        Metric::RocAuc => {
            let has_pos = labels.iter().any(|(_, l)| *l == 1.0);
            let has_neg = labels.iter().any(|(_, l)| *l == 0.0);
            if has_pos && has_neg {
                (true, 1.0, 1e-9)
            } else {
                return Ok(EvaluatorPropertyCheckResult {
                    property_name: "identity_at_optimum".into(),
                    applicable: false,
                    passed: true,
                    tolerance: 1e-9,
                    detail: "ROC-AUC identity requires both classes in labels".into(),
                });
            }
        }
        Metric::LogLoss => (true, 0.0, 1e-6),
    };

    if !applicable {
        return Ok(EvaluatorPropertyCheckResult {
            property_name: "identity_at_optimum".into(),
            applicable: false,
            passed: true,
            tolerance,
            detail: "identity not applicable to this metric/domain".into(),
        });
    }

    let batch = PredictionBatch::scalar(dummy_id.clone(), labels.to_vec());
    let receipt = evaluate(&dummy_id, metric, labels, &batch)?;
    let diff = (receipt.score - expected_score).abs();
    let passed = diff <= tolerance;

    Ok(EvaluatorPropertyCheckResult {
        property_name: "identity_at_optimum".into(),
        applicable: true,
        passed,
        tolerance,
        detail: format!("score was {}, expected {} (diff={})", receipt.score, expected_score, diff),
    })
}

/// Test permutation invariance: reordering rows gives identical score.
pub fn check_permutation_invariance_property(
    metric: Metric,
    labels: &[(u64, f64)],
    predictions: &PredictionBatch,
) -> Result<EvaluatorPropertyCheckResult> {
    let dummy_id = predictions.identity.clone();
    let r1 = evaluate(&dummy_id, metric.clone(), labels, predictions)?;

    // Reverse row order
    let mut rev_labels = labels.to_vec();
    rev_labels.reverse();
    let mut rev_batch = predictions.clone();
    rev_batch.rows.reverse();
    rev_batch.vector_rows.reverse();

    let r2 = evaluate(&dummy_id, metric, &rev_labels, &rev_batch)?;
    let diff = (r1.score - r2.score).abs();
    let tolerance = 1e-9;
    let passed = diff <= tolerance;

    Ok(EvaluatorPropertyCheckResult {
        property_name: "permutation_invariance".into(),
        applicable: true,
        passed,
        tolerance,
        detail: format!("forward={}, reversed={}, diff={diff}", r1.score, r2.score),
    })
}

/// Test identity property adhering to an authoritative MetricSpec (evaluating with spec parameters).
pub fn check_spec_identity_property(
    spec: &crate::contracts::MetricSpec,
    labels: &[(u64, f64)],
) -> Result<EvaluatorPropertyCheckResult> {
    spec.validate()?;
    let metric = parse_metric_name(&spec.name)?;
    let dummy_id = EvaluationIdentity {
        project_id: "prop_test".into(),
        job_id: "prop_job".into(),
        candidate_id: "prop_cand".into(),
        dataset_id: "prop_data".into(),
        dataset_sha256: "0".repeat(64),
        split_sha256: "0".repeat(64),
        source_sha256: "0".repeat(64),
        environment_id: "prop_env".into(),
        weights_sha256: "0".repeat(64),
    };

    let (applicable, expected_score, tolerance) = match metric {
        Metric::Accuracy | Metric::F1 => (true, 1.0, 1e-9),
        Metric::Mae | Metric::Mse | Metric::Rmse | Metric::Smape | Metric::Rmsle | Metric::Cauchy | Metric::Huber => (true, 0.0, 1e-9),
        Metric::Custom(ref name) => {
            let reg = ADMITTED_CUSTOM_EVALUATORS.read().unwrap();
            if reg.as_ref().and_then(|m| m.get(&name.to_lowercase())).is_some() {
                (true, 0.0, 1e-9)
            } else {
                return Ok(EvaluatorPropertyCheckResult {
                    property_name: "identity_at_optimum".into(),
                    applicable: false,
                    passed: false,
                    tolerance: 1e-9,
                    detail: format!("custom metric '{name}' requires an admitted evaluator"),
                });
            }
        }
        Metric::RocAuc => {
            let has_pos = labels.iter().any(|(_, l)| *l == 1.0);
            let has_neg = labels.iter().any(|(_, l)| *l == 0.0);
            if has_pos && has_neg {
                (true, 1.0, 1e-9)
            } else {
                return Ok(EvaluatorPropertyCheckResult {
                    property_name: "identity_at_optimum".into(),
                    applicable: false,
                    passed: true,
                    tolerance: 1e-9,
                    detail: "ROC-AUC identity requires both classes in labels".into(),
                });
            }
        }
        Metric::LogLoss => (true, 0.0, 1e-6),
    };

    if !applicable {
        return Ok(EvaluatorPropertyCheckResult {
            property_name: "identity_at_optimum".into(),
            applicable: false,
            passed: true,
            tolerance,
            detail: "identity not applicable to this metric/domain".into(),
        });
    }

    let batch = PredictionBatch::scalar(dummy_id.clone(), labels.to_vec());
    let threshold = spec.decision_threshold.unwrap_or(0.5);
    let non_negative_only = spec.variant.as_deref() == Some("non_negative_only");
    let receipt = evaluate_internal(&dummy_id, metric, labels, &batch, threshold, non_negative_only, "provisional_spec_checked", spec.class_order.as_deref(), spec.epsilon)?;
    let diff = (receipt.score - expected_score).abs();
    let passed = diff <= tolerance;

    Ok(EvaluatorPropertyCheckResult {
        property_name: "identity_at_optimum".into(),
        applicable: true,
        passed,
        tolerance,
        detail: format!("score was {}, expected {} (diff={})", receipt.score, expected_score, diff),
    })
}

/// Test permutation invariance adhering to an authoritative MetricSpec (evaluating with spec parameters).
pub fn check_spec_permutation_invariance_property(
    spec: &crate::contracts::MetricSpec,
    labels: &[(u64, f64)],
    predictions: &PredictionBatch,
) -> Result<EvaluatorPropertyCheckResult> {
    spec.validate()?;
    let metric = parse_metric_name(&spec.name)?;
    let threshold = spec.decision_threshold.unwrap_or(0.5);
    let non_negative_only = spec.variant.as_deref() == Some("non_negative_only");
    let dummy_id = predictions.identity.clone();
    let r1 = evaluate_internal(&dummy_id, metric.clone(), labels, predictions, threshold, non_negative_only, "provisional_spec_checked", spec.class_order.as_deref(), spec.epsilon)?;

    let mut rev_labels = labels.to_vec();
    rev_labels.reverse();
    let mut rev_batch = predictions.clone();
    rev_batch.rows.reverse();
    rev_batch.vector_rows.reverse();

    let r2 = evaluate_internal(&dummy_id, metric, &rev_labels, &rev_batch, threshold, non_negative_only, "provisional_spec_checked", spec.class_order.as_deref(), spec.epsilon)?;
    let diff = (r1.score - r2.score).abs();
    let tolerance = 1e-9;
    let passed = diff <= tolerance;

    Ok(EvaluatorPropertyCheckResult {
        property_name: "permutation_invariance".into(),
        applicable: true,
        passed,
        tolerance,
        detail: format!("forward={}, reversed={}, diff={diff}", r1.score, r2.score),
    })
}
