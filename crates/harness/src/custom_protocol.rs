//! Persistent controller-owned split and metric contract. Custom execution is
//! still gated until this authority is integrated into dispatch and evaluation.
use crate::{trusted_evaluation::Metric, Result};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FoldSplit {
    pub split_id: String,
    pub train: Vec<u64>,
    pub validation: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CustomProtocol {
    pub dataset_sha256: String,
    pub target_column: String,
    pub metric: Metric,
    pub train: Vec<u64>,
    pub validation: Vec<u64>,
    pub test: Vec<u64>,
    #[serde(default)]
    pub folds: Vec<FoldSplit>,
    #[serde(default)]
    pub lockbox_rows: Vec<u64>,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub group_column: Option<String>,
    #[serde(default)]
    pub time_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_groups: Option<HashMap<u64, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub row_timestamps: Option<HashMap<u64, i64>>,
}

impl CustomProtocol {
    pub fn single_split(
        dataset_sha256: String,
        target_column: String,
        metric: Metric,
        train: Vec<u64>,
        validation: Vec<u64>,
        test: Vec<u64>,
    ) -> Self {
        Self {
            dataset_sha256,
            target_column,
            metric,
            train,
            validation,
            test,
            folds: Vec::new(),
            lockbox_rows: Vec::new(),
            strategy: None,
            group_column: None,
            time_column: None,
            row_groups: None,
            row_timestamps: None,
        }
    }

    pub fn validate_group_separation(&self, row_groups: &HashMap<u64, String>) -> Result<()> {
        let mut train_groups = HashSet::new();
        for &id in &self.train {
            let g = row_groups.get(&id).ok_or_else(|| format!("row {id} missing group mapping"))?;
            train_groups.insert(g);
        }
        let mut val_groups = HashSet::new();
        for &id in &self.validation {
            let g = row_groups.get(&id).ok_or_else(|| format!("row {id} missing group mapping"))?;
            if train_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both train and validation"));
            }
            val_groups.insert(g);
        }
        let mut test_groups = HashSet::new();
        for &id in &self.test {
            let g = row_groups.get(&id).ok_or_else(|| format!("row {id} missing group mapping"))?;
            if train_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both train and test"));
            }
            if val_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both validation and test"));
            }
            test_groups.insert(g);
        }
        for &id in &self.lockbox_rows {
            let g = row_groups.get(&id).ok_or_else(|| format!("row {id} missing group mapping"))?;
            if train_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both train and lockbox"));
            }
            if val_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both validation and lockbox"));
            }
            if test_groups.contains(g) {
                return Err(format!("group leakage detected: group '{g}' present in both test and lockbox"));
            }
        }
        for fold in &self.folds {
            let mut ft_groups = HashSet::new();
            for &id in &fold.train {
                let g = row_groups.get(&id).ok_or_else(|| format!("fold '{}' row {id} missing group mapping", fold.split_id))?;
                ft_groups.insert(g);
            }
            for &id in &fold.validation {
                let g = row_groups.get(&id).ok_or_else(|| format!("fold '{}' row {id} missing group mapping", fold.split_id))?;
                if ft_groups.contains(g) {
                    return Err(format!("group leakage in fold '{}': group '{g}' present in both train and validation", fold.split_id));
                }
            }
        }
        Ok(())
    }

    pub fn validate_temporal_ordering(&self, row_timestamps: &HashMap<u64, i64>) -> Result<()> {
        let max_train = self.train.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().max().copied();

        let min_val = self.validation.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().min().copied();

        if let (Some(max_t), Some(min_v)) = (max_train, min_val) {
            if min_v < max_t {
                return Err(format!("temporal lookahead detected: validation min timestamp ({min_v}) < train max timestamp ({max_t})"));
            }
        }

        let max_val = self.validation.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().max().copied();

        let min_test = self.test.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().min().copied();

        let max_pre_test = max_val.or(max_train);
        if let (Some(max_prev), Some(min_te)) = (max_pre_test, min_test) {
            if min_te < max_prev {
                return Err(format!("temporal lookahead detected: test min timestamp ({min_te}) < dev max timestamp ({max_prev})"));
            }
        }

        let max_dev = self.test.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().max().copied()
            .or(max_val)
            .or(max_train);

        let min_lockbox = self.lockbox_rows.iter()
            .map(|id| row_timestamps.get(id).ok_or_else(|| format!("row {id} missing timestamp mapping")))
            .collect::<Result<Vec<&i64>>>()?
            .into_iter().min().copied();

        if let (Some(max_d), Some(min_l)) = (max_dev, min_lockbox) {
            if min_l < max_d {
                return Err(format!("temporal lookahead detected: lockbox min timestamp ({min_l}) < dev max timestamp ({max_d})"));
            }
        }

        for fold in &self.folds {
            let max_ft = fold.train.iter()
                .map(|id| row_timestamps.get(id).ok_or_else(|| format!("fold '{}' row {id} missing timestamp mapping", fold.split_id)))
                .collect::<Result<Vec<&i64>>>()?
                .into_iter().max().copied();

            let min_fv = fold.validation.iter()
                .map(|id| row_timestamps.get(id).ok_or_else(|| format!("fold '{}' row {id} missing timestamp mapping", fold.split_id)))
                .collect::<Result<Vec<&i64>>>()?
                .into_iter().min().copied();

            if let (Some(max_t), Some(min_v)) = (max_ft, min_fv) {
                if min_v < max_t {
                    return Err(format!("temporal lookahead in fold '{}': validation min timestamp ({min_v}) < train max timestamp ({max_t})", fold.split_id));
                }
            }
        }

        Ok(())
    }
}

fn canonical(mut p: CustomProtocol) -> Result<CustomProtocol> {
    if p.dataset_sha256.len() != 64 || !p.dataset_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("invalid registered dataset hash".into());
    }
    if p.target_column.is_empty() || p.target_column.len() > 256 || p.target_column == "__mlr_row_id" || p.target_column.chars().any(char::is_control) {
        return Err("invalid target column".into());
    }
    if let Some(group) = &p.group_column {
        if group.is_empty() || group.len() > 256 || group == "__mlr_row_id" || group.chars().any(char::is_control) {
            return Err("invalid group column".into());
        }
    }
    if let Some(time) = &p.time_column {
        if time.is_empty() || time.len() > 256 || time == "__mlr_row_id" || time.chars().any(char::is_control) {
            return Err("invalid time column".into());
        }
    }
    if let (Some(g), Some(t)) = (&p.group_column, &p.time_column) {
        if g == t {
            return Err("time_column and group_column cannot be identical".into());
        }
    }
    if let Some(strat) = &p.strategy {
        if strat.starts_with("group") && p.group_column.is_none() {
            return Err("group-based validation strategy requires group_column".into());
        }
        if strat.starts_with("time") && p.time_column.is_none() {
            return Err("time-based validation strategy requires time_column".into());
        }
    }

    let count = p.train.len()
        .saturating_add(p.validation.len())
        .saturating_add(p.test.len())
        .saturating_add(p.lockbox_rows.len());

    if p.train.is_empty() || (p.validation.is_empty() && p.folds.is_empty()) || count > 1_000_000 {
        return Err("protocol split size outside bounds".into());
    }

    let mut all = HashSet::with_capacity(count);
    for &id in p.train.iter().chain(&p.validation).chain(&p.test).chain(&p.lockbox_rows) {
        if !all.insert(id) {
            return Err("duplicate or overlapping protocol rows".into());
        }
    }

    let dev_set: HashSet<u64> = p.train.iter().chain(&p.validation).copied().collect();
    let test_set: HashSet<u64> = p.test.iter().copied().collect();
    let lockbox_set: HashSet<u64> = p.lockbox_rows.iter().copied().collect();
    let mut seen_split_ids = HashSet::new();

    // Verify folds if present
    for fold in &mut p.folds {
        if fold.split_id.is_empty() || !seen_split_ids.insert(fold.split_id.clone()) {
            return Err(format!("duplicate or empty fold split_id: '{}'", fold.split_id));
        }
        if fold.train.is_empty() || fold.validation.is_empty() {
            return Err(format!("fold '{}' train and validation sets cannot be empty", fold.split_id));
        }
        let mut fold_train_set = HashSet::with_capacity(fold.train.len());
        for &t in &fold.train {
            if !fold_train_set.insert(t) {
                return Err(format!("fold '{}' has duplicate row in train: {t}", fold.split_id));
            }
            if !dev_set.contains(&t) {
                return Err(format!("fold '{}' train set contains unknown row {t} not in development rows", fold.split_id));
            }
            if test_set.contains(&t) {
                return Err(format!("fold '{}' train set contains test row {t}", fold.split_id));
            }
            if lockbox_set.contains(&t) {
                return Err(format!("fold '{}' train set contains lockbox row {t}", fold.split_id));
            }
        }
        let mut fold_val_set = HashSet::with_capacity(fold.validation.len());
        for &v in &fold.validation {
            if !fold_val_set.insert(v) {
                return Err(format!("fold '{}' has duplicate row in validation: {v}", fold.split_id));
            }
            if fold_train_set.contains(&v) {
                return Err(format!("fold '{}' has overlapping train and validation row {v}", fold.split_id));
            }
            if !dev_set.contains(&v) {
                return Err(format!("fold '{}' validation set contains unknown row {v} not in development rows", fold.split_id));
            }
            if test_set.contains(&v) {
                return Err(format!("fold '{}' validation set contains test row {v}", fold.split_id));
            }
            if lockbox_set.contains(&v) {
                return Err(format!("fold '{}' validation set contains lockbox row {v}", fold.split_id));
            }
        }
        fold.train.sort_unstable();
        fold.validation.sort_unstable();
    }

    p.train.sort_unstable();
    p.validation.sort_unstable();
    p.test.sort_unstable();
    p.lockbox_rows.sort_unstable();

    if let Some(groups) = &p.row_groups {
        p.validate_group_separation(groups)?;
    }
    if let Some(timestamps) = &p.row_timestamps {
        p.validate_temporal_ordering(timestamps)?;
    }

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

/// Carve out a deterministic assessment lockbox from total rows before search.
pub fn carve_lockbox(
    total_rows: &[u64],
    fraction: f64,
    seed: u64,
) -> Result<(Vec<u64>, Vec<u64>)> {
    carve_lockbox_with_min_rows(total_rows, fraction, seed, 2)
}

/// Carve out a deterministic assessment lockbox with task-specific feasibility threshold.
pub fn carve_lockbox_with_min_rows(
    total_rows: &[u64],
    fraction: f64,
    seed: u64,
    min_rows: usize,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let effective_min = min_rows.max(2);
    if total_rows.len() < effective_min {
        return Err(format!("insufficient rows ({}) to carve a valid lockbox; need at least {effective_min} rows", total_rows.len()));
    }
    if !fraction.is_finite() || fraction <= 0.0 || fraction >= 0.5 {
        return Err("lockbox fraction must be strictly between 0.0 and 0.5".into());
    }
    let mut rows = total_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    if rows.len() != total_rows.len() {
        return Err("total_rows contains duplicates".into());
    }

    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut rng = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };

    let n = rows.len();
    let mut shuffled = rows.clone();
    for i in (1..n).rev() {
        let j = (rng() as usize) % (i + 1);
        shuffled.swap(i, j);
    }

    let lockbox_count = ((n as f64) * fraction).round().max(1.0) as usize;
    if lockbox_count >= n {
        return Err(format!("lockbox fraction {fraction} leaves no development rows"));
    }
    let (lockbox_part, dev_part) = shuffled.split_at(lockbox_count);

    let mut lockbox = lockbox_part.to_vec();
    let mut dev = dev_part.to_vec();
    lockbox.sort_unstable();
    dev.sort_unstable();

    Ok((dev, lockbox))
}

/// Generate deterministic k-fold splits over development rows.
pub fn generate_k_fold_splits(
    dev_rows: &[u64],
    k: usize,
    seed: u64,
) -> Result<Vec<FoldSplit>> {
    if k < 2 || k > 50 {
        return Err("folds k must be 2..=50".into());
    }
    if dev_rows.len() < k {
        return Err(format!("insufficient rows ({}) for {k} folds", dev_rows.len()));
    }
    let mut rows = dev_rows.to_vec();
    rows.sort_unstable();
    rows.dedup();
    if rows.len() != dev_rows.len() {
        return Err("dev_rows contains duplicates".into());
    }

    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    let mut rng = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        state
    };

    let n = rows.len();
    let mut shuffled = rows.clone();
    for i in (1..n).rev() {
        let j = (rng() as usize) % (i + 1);
        shuffled.swap(i, j);
    }

    let mut fold_splits = Vec::with_capacity(k);
    for fold_idx in 0..k {
        let mut train = Vec::new();
        let mut val = Vec::new();
        for (idx, &row_id) in shuffled.iter().enumerate() {
            if idx % k == fold_idx {
                val.push(row_id);
            } else {
                train.push(row_id);
            }
        }
        train.sort_unstable();
        val.sort_unstable();
        fold_splits.push(FoldSplit {
            split_id: format!("fold_{fold_idx}"),
            train,
            validation: val,
        });
    }

    Ok(fold_splits)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentBinding {
    pub candidate_id: String,
    pub artifact_sha256: String,
    pub predictions_sha256: String,
    pub metric: Metric,
    pub score: f64,
    pub receipt_identity: String,
    #[serde(default)]
    pub receipt_json: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssessmentRecord {
    pub project: String,
    pub dataset_id: String,
    pub candidate_id: String,
    pub artifact_sha256: String,
    pub predictions_sha256: String,
    pub metric: Metric,
    pub score: f64,
    pub receipt_identity: String,
    pub receipt_json: String,
    pub consumed_at_ms: i64,
}

/// Record the one-shot consumption of the assessment lockbox bound to an evaluated candidate.
/// Fails if already consumed or if no lockbox was defined.
pub fn consume_assessment(
    connection: &mut Connection,
    project: &str,
    dataset_id: &str,
    binding: &AssessmentBinding,
) -> Result<Vec<u64>> {
    if binding.candidate_id.is_empty() || binding.artifact_sha256.len() != 64 || binding.predictions_sha256.len() != 64 || binding.receipt_identity.is_empty() {
        return Err("invalid assessment binding fields".into());
    }
    if !binding.score.is_finite() {
        return Err("assessment score must be finite".into());
    }
    let tx = connection.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    tx.execute_batch(include_str!("schema_custom_protocol.sql")).map_err(|e| e.to_string())?;

    let is_consumed: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2)",
        params![project, dataset_id],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;

    if is_consumed {
        return Err("assessment lockbox has already been consumed; cannot evaluate holdout a second time".into());
    }

    let encoded: String = tx.query_row(
        "SELECT protocol_json FROM custom_protocols WHERE project=?1 AND dataset_id=?2",
        params![project, dataset_id],
        |r| r.get(0),
    ).map_err(|e| format!("protocol not found for project '{project}', dataset '{dataset_id}': {e}"))?;

    let p: CustomProtocol = serde_json::from_str(&encoded).map_err(|e| e.to_string())?;
    if p.lockbox_rows.is_empty() {
        return Err("no assessment lockbox was registered for this protocol".into());
    }

    let metric_str = serde_json::to_string(&binding.metric).unwrap_or_else(|_| "unknown".into()).trim_matches('"').to_string();
    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
    tx.execute(
        "INSERT INTO assessment_lockbox_consumptions(project, dataset_id, candidate_id, artifact_sha256, predictions_sha256, metric, score, receipt_identity, receipt_json, consumed_at_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![project, dataset_id, binding.candidate_id, binding.artifact_sha256, binding.predictions_sha256, metric_str, binding.score, binding.receipt_identity, binding.receipt_json, now_ms],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(p.lockbox_rows)
}

/// Retrieve the immutable consumption record for a project's dataset lockbox if consumed.
pub fn get_assessment_consumption(
    connection: &Connection,
    project: &str,
    dataset_id: &str,
) -> Result<Option<AssessmentRecord>> {
    let row: Option<(String, String, String, String, String, String, f64, String, String, i64)> = connection.query_row(
        "SELECT project, dataset_id, candidate_id, artifact_sha256, predictions_sha256, metric, score, receipt_identity, receipt_json, consumed_at_ms FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2",
        params![project, dataset_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?)),
    ).optional().map_err(|e| e.to_string())?;

    match row {
        Some((p, d, cand, art, pred, met, score, rec, r_json, t)) => {
            let metric: Metric = serde_json::from_value(serde_json::Value::String(met)).unwrap_or(Metric::Accuracy);
            Ok(Some(AssessmentRecord {
                project: p,
                dataset_id: d,
                candidate_id: cand,
                artifact_sha256: art,
                predictions_sha256: pred,
                metric,
                score,
                receipt_identity: rec,
                receipt_json: r_json,
                consumed_at_ms: t,
            }))
        }
        None => Ok(None),
    }
}

/// Retrieve the exact authoritative evaluation receipt from the consumption record if consumed.
/// Returns Ok(None) if not consumed or no receipt_json was stored.
pub fn get_assessment_consumption_receipt(
    connection: &Connection,
    project: &str,
    dataset_id: &str,
) -> Result<Option<crate::trusted_evaluation::EvaluationReceipt>> {
    let row: Option<String> = connection.query_row(
        "SELECT receipt_json FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2",
        params![project, dataset_id],
        |r| r.get(0),
    ).optional().map_err(|e| e.to_string())?;

    match row {
        Some(json_str) => {
            if json_str.trim().is_empty() {
                return Ok(None);
            }
            let receipt: crate::trusted_evaluation::EvaluationReceipt =
                serde_json::from_str(&json_str).map_err(|e| format!("corrupt assessment receipt json: {e}"))?;
            Ok(Some(receipt))
        }
        None => Ok(None),
    }
}

fn extract_lockbox_ground_truth(
    raw_csv_path: &Path,
    target_column: &str,
    lockbox_rows: &[u64],
    expected_sha256: &str,
) -> Result<Vec<(u64, f64)>> {
    let raw_bytes = fs::read(raw_csv_path).map_err(|e| e.to_string())?;
    let actual_sha = format!("{:x}", Sha256::digest(&raw_bytes));
    if actual_sha != expected_sha256 {
        return Err(format!(
            "raw dataset sha256 '{actual_sha}' does not match protocol expected sha '{expected_sha256}'"
        ));
    }
    let content = String::from_utf8(raw_bytes).map_err(|_| "raw dataset is not valid UTF-8 CSV")?;
    let mut lines = content.lines();
    let header_line = lines.next().ok_or("empty raw dataset CSV")?;
    let headers: Vec<&str> = header_line.split(',').map(|s| s.trim()).collect();
    let target_idx = headers
        .iter()
        .position(|&h| h == target_column)
        .ok_or_else(|| format!("target column '{target_column}' not found in raw dataset headers"))?;

    let lockbox_set: HashSet<u64> = lockbox_rows.iter().copied().collect();
    let mut results = Vec::new();
    let mut row_idx: u64 = 0;

    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if lockbox_set.contains(&row_idx) {
            let cols: Vec<&str> = trimmed.split(',').collect();
            if target_idx >= cols.len() {
                return Err(format!("row {row_idx} missing target column at index {target_idx}"));
            }
            let target_val: f64 = cols[target_idx]
                .trim()
                .parse()
                .map_err(|_| format!("cannot parse target value at row {row_idx}: '{}'", cols[target_idx]))?;
            results.push((row_idx, target_val));
        }
        row_idx += 1;
    }

    Ok(results)
}

/// Authoritative end-to-end evaluation and atomic lockbox consumption.
pub fn evaluate_and_consume_assessment(
    connection: &mut Connection,
    project: &str,
    dataset_id: &str,
    expected: &crate::trusted_evaluation::EvaluationIdentity,
    trusted_lockbox_labels: &[(u64, f64)],
    predictions: &crate::trusted_evaluation::PredictionBatch,
) -> Result<crate::trusted_evaluation::EvaluationReceipt> {
    let (p, _) = load(connection, project, dataset_id)?;
    if expected.project_id != project || expected.dataset_id != dataset_id || expected.dataset_sha256 != p.dataset_sha256 {
        return Err("assessment identity does not match the registered project and dataset".into());
    }
    if p.lockbox_rows.is_empty() {
        return Err("no assessment lockbox was registered for this protocol".into());
    }
    let mut label_row_ids: Vec<u64> = trusted_lockbox_labels.iter().map(|(id, _)| *id).collect();
    label_row_ids.sort_unstable();
    if label_row_ids != p.lockbox_rows {
        return Err("trusted lockbox labels do not match registered lockbox rows".into());
    }

    // Verify labels against authoritative scientific_datasets ground truth if available on disk
    let raw_path: Option<String> = match connection.query_row(
        "SELECT path FROM scientific_datasets WHERE project=?1 AND id=?2",
        params![project, dataset_id],
        |r| r.get(0),
    ) {
        Ok(p) => Some(p),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) if e.to_string().contains("no such column: path") => None,
        Err(e) => return Err(e.to_string()),
    };

    if let Some(path_str) = raw_path {
        let pth = Path::new(&path_str);
        if pth.is_file() {
            let gt_labels = extract_lockbox_ground_truth(pth, &p.target_column, &p.lockbox_rows, &p.dataset_sha256)?;
            let gt_map: BTreeMap<u64, f64> = gt_labels.into_iter().collect();
            for (id, val) in trusted_lockbox_labels {
                if let Some(gt_val) = gt_map.get(id) {
                    if (val - gt_val).abs() > 1e-9 {
                        return Err(format!(
                            "trusted lockbox label value mismatch for row {id}: caller passed {val}, registered dataset has {gt_val}"
                        ));
                    }
                } else {
                    return Err(format!("trusted lockbox row id {id} not found in registered lockbox rows"));
                }
            }
        }
    }

    let receipt = crate::trusted_evaluation::evaluate(expected, p.metric, trusted_lockbox_labels, predictions)?;
    let receipt_json = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
    let binding = AssessmentBinding {
        candidate_id: expected.candidate_id.clone(),
        artifact_sha256: expected.weights_sha256.clone(),
        predictions_sha256: receipt.predictions_sha256.clone(),
        metric: receipt.metric.clone(),
        score: receipt.score,
        receipt_identity: format!("{}:{}:{}", expected.project_id, expected.job_id, expected.candidate_id),
        receipt_json,
    };
    consume_assessment(connection, project, dataset_id, &binding)?;
    Ok(receipt)
}

/// Authoritative assessment evaluation reading ground truth labels directly from the registered dataset file.
pub fn evaluate_and_consume_assessment_authoritative(
    connection: &mut Connection,
    project: &str,
    dataset_id: &str,
    expected: &crate::trusted_evaluation::EvaluationIdentity,
    predictions: &crate::trusted_evaluation::PredictionBatch,
) -> Result<crate::trusted_evaluation::EvaluationReceipt> {
    let (p, _) = load(connection, project, dataset_id)?;
    let raw_path: String = connection
        .query_row(
            "SELECT path FROM scientific_datasets WHERE project=?1 AND id=?2",
            params![project, dataset_id],
            |r| r.get(0),
        )
        .map_err(|_| format!("dataset '{dataset_id}' not found in scientific_datasets"))?;
    let gt_labels = extract_lockbox_ground_truth(Path::new(&raw_path), &p.target_column, &p.lockbox_rows, &p.dataset_sha256)?;
    evaluate_and_consume_assessment(connection, project, dataset_id, expected, &gt_labels, predictions)
}
