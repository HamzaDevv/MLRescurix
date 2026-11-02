//! Bounded preparation of separate training and validation-inference inputs.
//! Callers must keep `labels` in trusted state and must not mount the original
//! development CSV into either worker. This module does not itself mount files.
use crate::{execution::StreamingCsvReader, Result};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::{self, Read, Write};

const MAX_INPUT_BYTES: usize = 32 * 1024 * 1024;
const MAX_ROWS: usize = 1_000_000;
const MAX_STREAM_BYTES: u64 = 256 * 1024 * 1024;

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
    let mut training_csv = Vec::new();
    let mut validation_features_csv = Vec::new();
    let labels = separate_stream(dev_csv, expected_sha256, target_column, train_ids, validation_ids,
        &mut training_csv, &mut validation_features_csv, MAX_INPUT_BYTES as u64)?;
    Ok(DevelopmentInputs { training_csv, validation_features_csv, labels })
}

struct BoundedHashReader<R> { inner: R, bytes: u64, limit: u64, digest: Sha256 }
impl<R: Read> Read for BoundedHashReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.limit.saturating_sub(self.bytes).saturating_add(1);
        let cap = buf.len().min(available as usize);
        let n = self.inner.read(&mut buf[..cap])?;
        self.bytes += n as u64;
        if self.bytes > self.limit { return Err(io::Error::new(io::ErrorKind::InvalidData, "development CSV exceeds byte limit")); }
        self.digest.update(&buf[..n]);
        Ok(n)
    }
}

fn write_record<W: Write>(writer: &mut W, total: &mut u64, limit: u64, row: &[String]) -> Result<()> {
    let mut record = Vec::new();
    append_record(&mut record, row);
    *total = total.checked_add(record.len() as u64).ok_or("output size overflow")?;
    if *total > limit { return Err("development output exceeds byte limit".into()); }
    writer.write_all(&record).map_err(|e| e.to_string())
}

/// Stream a controller-owned development CSV into two private, unpublished
/// outputs. The digest is computed on the same bytes parsed. On any error the
/// outputs are incomplete and the caller must not publish or mount them.
/// `max_bytes` bounds input and each output; rows and individual records have
/// independent limits. The caller owns secure file creation and publication.
pub fn separate_stream<R: Read, W: Write, V: Write>(
    input: R, expected_sha256: &str, target_column: &str,
    train_ids: &[u64], validation_ids: &[u64], mut training_csv: W,
    mut validation_features_csv: V, max_bytes: u64,
) -> Result<Vec<(u64, f64)>> {
    if max_bytes == 0 || max_bytes > MAX_STREAM_BYTES { return Err("development stream size limit outside bounds".into()); }
    if train_ids.is_empty() || validation_ids.is_empty() || train_ids.len().saturating_add(validation_ids.len()) > MAX_ROWS {
        return Err("split size outside bounds".into());
    }
    let train: HashSet<_> = train_ids.iter().copied().collect();
    let validation: HashSet<_> = validation_ids.iter().copied().collect();
    if train.len() != train_ids.len() || validation.len() != validation_ids.len() || !train.is_disjoint(&validation) {
        return Err("duplicate or overlapping split row ids".into());
    }
    let mut reader = BoundedHashReader { inner: input, bytes: 0, limit: max_bytes, digest: Sha256::new() };
    let mut parser = StreamingCsvReader::new(&mut reader);
    let header = parser.read_record()?.ok_or("missing development header")?;
    if header.first().map(String::as_str) != Some("__mlr_row_id") { return Err("development row id column missing".into()); }
    let unique: HashSet<_> = header.iter().collect();
    if unique.len() != header.len() || header.iter().any(String::is_empty) { return Err("duplicate or empty column name".into()); }
    let target = header.iter().position(|name| name == target_column).ok_or("target column missing")?;
    if target == 0 { return Err("row id cannot be the target".into()); }
    let feature_header: Vec<_> = header.iter().enumerate().filter(|(i, _)| *i != target).map(|(_, x)| x.clone()).collect();
    let mut training_bytes = 0;
    let mut validation_bytes = 0;
    write_record(&mut training_csv, &mut training_bytes, max_bytes, &header)?;
    write_record(&mut validation_features_csv, &mut validation_bytes, max_bytes, &feature_header)?;
    let mut labels = Vec::new();
    let mut seen = HashSet::new();
    while let Some(row) = parser.read_record()? {
        if row.len() != header.len() { return Err("development CSV field count mismatch".into()); }
        let id: u64 = row[0].parse().map_err(|_| "invalid development row id")?;
        if !seen.insert(id) { return Err("duplicate development row id".into()); }
        if train.contains(&id) {
            write_record(&mut training_csv, &mut training_bytes, max_bytes, &row)?;
        } else if validation.contains(&id) {
            let label: f64 = row[target].parse().map_err(|_| "invalid numeric validation label")?;
            if !label.is_finite() { return Err("non-finite validation label".into()); }
            labels.push((id, label));
            let features: Vec<_> = row.into_iter().enumerate().filter(|(i, _)| *i != target).map(|(_, x)| x).collect();
            write_record(&mut validation_features_csv, &mut validation_bytes, max_bytes, &features)?;
        } else { return Err("unexpected development row id".into()); }
    }
    if seen.len() != train.len() + validation.len() { return Err("development split rows missing".into()); }
    drop(parser);
    if reader.bytes == 0 || format!("{:x}", reader.digest.finalize()) != expected_sha256 {
        return Err("development CSV hash mismatch".into());
    }
    Ok(labels)
}
