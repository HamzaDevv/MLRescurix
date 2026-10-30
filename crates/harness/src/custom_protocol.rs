//! Persistent controller-owned split and metric contract. Custom execution is
//! still gated until this authority is integrated into dispatch and evaluation.
use crate::{trusted_evaluation::Metric, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomProtocol {
    pub dataset_sha256: String,
    pub target_column: String,
    pub metric: Metric,
    pub train: Vec<u64>,
    pub validation: Vec<u64>,
    pub test: Vec<u64>,
}

fn canonical(mut p: CustomProtocol) -> Result<CustomProtocol> {
    if p.dataset_sha256.len() != 64 || !p.dataset_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("invalid registered dataset hash".into());
    }
    if p.target_column.is_empty() || p.target_column.len() > 256 || p.target_column == "__mlr_row_id" || p.target_column.chars().any(char::is_control) {
        return Err("invalid target column".into());
    }
    let count = p.train.len().saturating_add(p.validation.len()).saturating_add(p.test.len());
    if p.train.is_empty() || p.validation.is_empty() || p.test.is_empty() || count > 1_000_000 { return Err("protocol split size outside bounds".into()); }
    let mut all = HashSet::with_capacity(count);
    if p.train.iter().chain(&p.validation).chain(&p.test).any(|id| !all.insert(*id)) {
        return Err("duplicate or overlapping protocol rows".into());
    }
    p.train.sort_unstable(); p.validation.sort_unstable(); p.test.sort_unstable();
    Ok(p)
}

/// Freeze once for a registered dataset; identical retries are idempotent and
/// different retries fail. The transaction serializes concurrent publishers.
pub fn freeze(connection: &mut Connection, project: &str, dataset_id: &str, protocol: CustomProtocol) -> Result<String> {
    let protocol = canonical(protocol)?;
    let encoded = serde_json::to_string(&protocol).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(encoded.as_bytes()));
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    let dataset_sha: Option<String> = tx.query_row("SELECT sha256 FROM scientific_datasets WHERE project=?1 AND id=?2", params![project, dataset_id], |r| r.get(0)).optional().map_err(|e| e.to_string())?;
    if dataset_sha.as_deref() != Some(protocol.dataset_sha256.as_str()) { return Err("protocol does not match registered dataset".into()); }
    let old: Option<(String, String)> = tx.query_row("SELECT protocol_sha256, protocol_json FROM custom_protocols WHERE project=?1 AND dataset_id=?2", params![project, dataset_id], |r| Ok((r.get(0)?, r.get(1)?))).optional().map_err(|e| e.to_string())?;
    if let Some((old_hash, old_json)) = old {
        if old_hash != digest || old_json != encoded { return Err("dataset protocol already frozen with different contents".into()); }
    } else {
        tx.execute("INSERT INTO custom_protocols(project,dataset_id,dataset_sha256,protocol_sha256,protocol_json) VALUES(?1,?2,?3,?4,?5)", params![project,dataset_id,protocol.dataset_sha256,digest,encoded]).map_err(|e| e.to_string())?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    Ok(digest)
}

/// Reject corrupted persisted records rather than returning a possibly altered
/// split as authority. Database access must remain outside worker mounts.
pub fn load(connection: &Connection, project: &str, dataset_id: &str) -> Result<(CustomProtocol, String)> {
    let (dataset_sha, digest, encoded): (String, String, String) = connection.query_row(
        "SELECT dataset_sha256,protocol_sha256,protocol_json FROM custom_protocols WHERE project=?1 AND dataset_id=?2",
        params![project,dataset_id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?)),
    ).map_err(|e| e.to_string())?;
    let p: CustomProtocol = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
    let p = canonical(p)?;
    let canonical_json = serde_json::to_string(&p).map_err(|e| e.to_string())?;
    let registered: String = connection.query_row("SELECT sha256 FROM scientific_datasets WHERE project=?1 AND id=?2", params![project,dataset_id], |r| r.get(0)).map_err(|e| e.to_string())?;
    if canonical_json != encoded || p.dataset_sha256 != dataset_sha || dataset_sha != registered || format!("{:x}", Sha256::digest(encoded.as_bytes())) != digest {
        return Err("frozen protocol integrity failure".into());
    }
    Ok((p, digest))
}
