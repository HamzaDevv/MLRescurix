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
    fn validate(&self) -> Result<()> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric { Accuracy, Mae, Mse, Rmse }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionBatch {
    pub identity: EvaluationIdentity,
    pub rows: Vec<(u64, f64)>,
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
}

fn canonical_rows(rows: &[(u64, f64)], name: &str) -> Result<BTreeMap<u64, f64>> {
    if rows.is_empty() || rows.len() > MAX_ROWS { return Err(format!("{name} row count outside bounds")); }
    let mut map = BTreeMap::new();
    for &(id, value) in rows {
        if !value.is_finite() { return Err(format!("non-finite {name}")); }
        if map.insert(id, value).is_some() { return Err(format!("duplicate {name} row id")); }
    }
    Ok(map)
}

fn digest_rows(rows: &BTreeMap<u64, f64>) -> String {
    // Fixed-width, sorted and endian-defined encoding avoids JSON float and
    // map-order ambiguities. Normalize signed zero, which compares equal.
    let mut digest = Sha256::new();
    for (&id, &value) in rows {
        digest.update(id.to_be_bytes());
        digest.update((if value == 0.0 { 0.0_f64 } else { value }).to_bits().to_be_bytes());
    }
    format!("{:x}", digest.finalize())
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
    expected.validate()?;
    if &predictions.identity != expected { return Err("evaluation identity mismatch".into()); }
    let labels = canonical_rows(trusted_labels, "label")?;
    let values = canonical_rows(&predictions.rows, "prediction")?;
    if labels.keys().ne(values.keys()) { return Err("prediction row ids do not match validation rows".into()); }
    let mut sum = 0.0;
    for (id, label) in &labels {
        let prediction = values[id];
        let contribution = match metric {
            Metric::Accuracy => if *label == prediction { 1.0 } else { 0.0 },
            Metric::Mae => (prediction - label).abs(),
            Metric::Mse | Metric::Rmse => (prediction - label).powi(2),
        };
        sum += contribution;
        if !sum.is_finite() { return Err("metric overflow".into()); }
    }
    let mean = sum / labels.len() as f64;
    let score = if metric == Metric::Rmse { mean.sqrt() } else { mean };
    Ok(EvaluationReceipt { identity: expected.clone(), metric, score, row_count: labels.len(), labels_sha256: digest_rows(&labels), predictions_sha256: digest_rows(&values) })
}
