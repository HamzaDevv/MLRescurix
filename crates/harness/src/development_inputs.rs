//! Bounded preparation of separate training and validation-inference inputs.
//! Callers must keep `labels` in trusted state and must not mount the original
//! development CSV into either worker. This module does not itself mount files.
use crate::{execution::StreamingCsvReader, Result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

const MAX_INPUT_BYTES: usize = 32 * 1024 * 1024;
const MAX_ROWS: usize = 1_000_000;

pub struct DevelopmentInputs {
    pub training_csv: Vec<u8>,
    pub validation_features_csv: Vec<u8>,
    pub labels: Vec<(u64, f64)>,
}

fn append_record(out: &mut Vec<u8>, fields: &[String]) {
    for (i, field) in fields.iter().enumerate() {
        if i != 0 { out.push(b','); }
        if field.bytes().any(|b| matches!(b, b',' | b'"' | b'\r' | b'\n')) {
            out.push(b'"');
            for b in field.bytes() { if b == b'"' { out.push(b'"'); } out.push(b); }
            out.push(b'"');
        } else { out.extend_from_slice(field.as_bytes()); }
    }
    out.push(b'\n');
}

/// `dev_csv` must be a controller-owned extraction bound to a frozen protocol.
/// The caller supplies its previously recorded digest and exact split row IDs.
pub fn separate(
    dev_csv: &[u8], expected_sha256: &str, target_column: &str,
    train_ids: &[u64], validation_ids: &[u64],
) -> Result<DevelopmentInputs> {
    if dev_csv.len() > MAX_INPUT_BYTES || dev_csv.is_empty() { return Err("development CSV size outside bounds".into()); }
    if format!("{:x}", Sha256::digest(dev_csv)) != expected_sha256 { return Err("development CSV hash mismatch".into()); }
    if train_ids.is_empty() || validation_ids.is_empty() || train_ids.len().saturating_add(validation_ids.len()) > MAX_ROWS {
        return Err("split size outside bounds".into());
    }
    let train: HashSet<_> = train_ids.iter().copied().collect();
    let validation: HashSet<_> = validation_ids.iter().copied().collect();
    if train.len() != train_ids.len() || validation.len() != validation_ids.len() || !train.is_disjoint(&validation) {
        return Err("duplicate or overlapping split row ids".into());
    }
    let mut parser = StreamingCsvReader::new(dev_csv);
    let header = parser.read_record()?.ok_or("missing development header")?;
    if header.first().map(String::as_str) != Some("__mlr_row_id") { return Err("development row id column missing".into()); }
    let unique: HashSet<_> = header.iter().collect();
    if unique.len() != header.len() || header.iter().any(String::is_empty) { return Err("duplicate or empty column name".into()); }
    let target = header.iter().position(|name| name == target_column).ok_or("target column missing")?;
    if target == 0 { return Err("row id cannot be the target".into()); }
    let feature_header: Vec<_> = header.iter().enumerate().filter(|(i, _)| *i != target).map(|(_, x)| x.clone()).collect();
    let mut training_csv = Vec::new();
    let mut validation_features_csv = Vec::new();
    append_record(&mut training_csv, &header);
    append_record(&mut validation_features_csv, &feature_header);
    let mut labels = Vec::new();
    let mut seen = HashSet::new();
    while let Some(row) = parser.read_record()? {
        if row.len() != header.len() { return Err("development CSV field count mismatch".into()); }
        let id: u64 = row[0].parse().map_err(|_| "invalid development row id")?;
        if !seen.insert(id) { return Err("duplicate development row id".into()); }
        if train.contains(&id) {
            append_record(&mut training_csv, &row);
        } else if validation.contains(&id) {
            let label: f64 = row[target].parse().map_err(|_| "invalid numeric validation label")?;
            if !label.is_finite() { return Err("non-finite validation label".into()); }
            labels.push((id, label));
            let features: Vec<_> = row.into_iter().enumerate().filter(|(i, _)| *i != target).map(|(_, x)| x).collect();
            append_record(&mut validation_features_csv, &features);
        } else { return Err("unexpected development row id".into()); }
    }
    if seen.len() != train.len() + validation.len() { return Err("development split rows missing".into()); }
    Ok(DevelopmentInputs { training_csv, validation_features_csv, labels })
}
