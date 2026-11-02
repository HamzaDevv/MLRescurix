//! Controller-owned per-job inputs. Only the specific input file for a phase
//! may be mounted into a worker; labels and receipts stay in trusted storage.
use crate::{development_inputs::DevelopmentInputs, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::{self, OpenOptions}, io::{Read, Write}, os::unix::fs::{DirBuilderExt, OpenOptionsExt}, path::{Path, PathBuf}};

const MAX_FILE: u64 = 64 * 1024 * 1024;
const MAX_DATA_FILE: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotIdentity {
    pub project_id: String,
    pub job_id: String,
    pub candidate_id: String,
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub protocol_sha256: String,
    pub source_sha256: String,
    pub environment_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SnapshotReceipt {
    pub identity: SnapshotIdentity,
    pub training_sha256: String,
    pub validation_features_sha256: String,
    pub labels_sha256: String,
}

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn identity_check(identity: &SnapshotIdentity) -> Result<()> {
    for id in [&identity.project_id, &identity.job_id, &identity.candidate_id, &identity.dataset_id, &identity.environment_id] {
        if id.is_empty() || id.len() > 256 || id.chars().any(char::is_control) { return Err("invalid snapshot identity".into()); }
    }
    for hash in [&identity.dataset_sha256, &identity.protocol_sha256, &identity.source_sha256] {
        if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) { return Err("invalid snapshot hash".into()); }
    }
    Ok(())
}
fn snapshot_path(root: &Path, identity: &SnapshotIdentity) -> PathBuf {
    // Length-prefix the project so arbitrary identifiers cannot collide.
    root.join(sha(format!("{}:{}{}", identity.project_id.len(), identity.project_id, identity.job_id).as_bytes()))
}
fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    if bytes.len() as u64 > MAX_FILE { return Err("snapshot file exceeds bound".into()); }
    let mut file = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).map_err(|e| e.to_string())?;
    file.write_all(bytes).and_then(|_| file.sync_all()).map_err(|e| e.to_string())
}
fn read_file(path: &Path) -> Result<Vec<u8>> {
    // Opening a FIFO for reading can block before metadata can reject it.
    let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path).map_err(|e| e.to_string())?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_FILE { return Err("invalid snapshot file".into()); }
    let mut bytes = Vec::new();
    Read::by_ref(&mut file).take(MAX_FILE + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_FILE { return Err("snapshot file exceeds bound".into()); }
    Ok(bytes)
}
fn hash_data_file(path: &Path) -> Result<String> {
    let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(path).map_err(|e| e.to_string())?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_DATA_FILE { return Err("invalid snapshot data file".into()); }
    let mut digest = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        bytes += n as u64;
        if bytes > MAX_DATA_FILE { return Err("snapshot data file exceeds bound".into()); }
        digest.update(&buffer[..n]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Exclusive job directory and completion marker; interrupted publication is
/// never reused or overwritten. The root and its ancestors must be trusted.
pub fn create(root: &Path, identity: &SnapshotIdentity, inputs: &DevelopmentInputs) -> Result<PathBuf> {
    identity_check(identity)?;
    if inputs.training_csv.is_empty() || inputs.validation_features_csv.is_empty() || inputs.labels.is_empty() || inputs.labels.len() > 1_000_000 {
        return Err("snapshot inputs outside bounds".into());
    }
    let mut ids = std::collections::HashSet::with_capacity(inputs.labels.len());
    if inputs.labels.iter().any(|(id, value)| !value.is_finite() || !ids.insert(*id)) {
        return Err("invalid snapshot labels".into());
    }
    let meta = fs::symlink_metadata(root).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() { return Err("snapshot root must be a trusted directory".into()); }
    let labels = serde_json::to_vec(&inputs.labels).map_err(|e| e.to_string())?;
    let receipt = SnapshotReceipt { identity: identity.clone(), training_sha256: sha(&inputs.training_csv), validation_features_sha256: sha(&inputs.validation_features_csv), labels_sha256: sha(&labels) };
    let receipt_bytes = serde_json::to_vec(&receipt).map_err(|e| e.to_string())?;
    let dir = snapshot_path(root, identity);
    fs::DirBuilder::new().mode(0o700).create(&dir).map_err(|e| format!("snapshot already exists or cannot be created: {e}"))?;
    // If any operation fails, retain the incomplete directory for explicit
    // recovery. Deleting it here would permit a competing publisher to reuse
    // the same job identity after an ambiguous write.
    write_file(&dir.join("training.csv"), &inputs.training_csv)?;
    write_file(&dir.join("validation_features.csv"), &inputs.validation_features_csv)?;
    write_file(&dir.join("labels.json"), &labels)?;
    fs::File::open(&dir).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    write_file(&dir.join("receipt.json"), &receipt_bytes)?;
    fs::File::open(&dir).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    Ok(dir)
}

pub fn verify(root: &Path, expected: &SnapshotIdentity) -> Result<SnapshotReceipt> {
    identity_check(expected)?;
    let dir = snapshot_path(root, expected);
    let meta = fs::symlink_metadata(&dir).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink() { return Err("snapshot directory invalid".into()); }
    let receipt: SnapshotReceipt = serde_json::from_slice(&read_file(&dir.join("receipt.json"))?).map_err(|e| e.to_string())?;
    if &receipt.identity != expected { return Err("snapshot identity mismatch".into()); }
    for (name, digest) in [("training.csv", &receipt.training_sha256), ("validation_features.csv", &receipt.validation_features_sha256), ("labels.json", &receipt.labels_sha256)] {
        if hash_data_file(&dir.join(name))? != *digest { return Err(format!("snapshot integrity mismatch: {name}")); }
    }
    Ok(receipt)
}

/// Prepare a snapshot from the registered file and frozen controller protocol.
/// This helper is not yet connected to job admission or container mounting.
pub fn prepare_registered(connection: &rusqlite::Connection, root: &Path, dataset_id: &str, identity: &SnapshotIdentity) -> Result<PathBuf> {
    identity_check(identity)?;
    if dataset_id != identity.dataset_id { return Err("snapshot dataset identity mismatch".into()); }
    let root_meta = fs::symlink_metadata(root).map_err(|e| e.to_string())?;
    if !root_meta.is_dir() || root_meta.file_type().is_symlink() { return Err("snapshot root must be a trusted directory".into()); }
    let (protocol, protocol_hash) = crate::custom_protocol::load(connection, &identity.project_id, dataset_id)?;
    if protocol_hash != identity.protocol_sha256 || protocol.dataset_sha256 != identity.dataset_sha256 {
        return Err("snapshot identity does not match frozen protocol".into());
    }
    let existing = snapshot_path(root, identity);
    if fs::symlink_metadata(&existing).is_ok() {
        verify(root, identity)?;
        return Ok(existing);
    }
    let raw_path: String = connection.query_row(
        "SELECT path FROM scientific_datasets WHERE project=?1 AND id=?2",
        rusqlite::params![identity.project_id, dataset_id], |r| r.get(0),
    ).map_err(|e| e.to_string())?;
    let stage = root.join(format!(".prepare-{}", snapshot_path(root, identity).file_name().ok_or("invalid snapshot path")?.to_string_lossy()));
    fs::DirBuilder::new().mode(0o700).create(&stage).map_err(|e| format!("snapshot preparation already exists or cannot be created: {e}"))?;
    let manifest = serde_json::json!({
        "format": "mlrescurix.dataset.v1", "dataset_sha256": protocol.dataset_sha256,
        "splits": {"train": protocol.train, "validation": protocol.validation, "test": protocol.test}
    });
    let manifest_bytes = serde_json::to_vec(&manifest).map_err(|e| e.to_string())?;
    write_file(&stage.join("manifest.json"), &manifest_bytes)?;
    let dev_path = stage.join("dev.csv");
    let dev_hash = crate::execution::prepare_dev_dataset(Path::new(&raw_path), &stage.join("manifest.json"), &dev_path)?;
    let dev = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(&dev_path).map_err(|e| e.to_string())?;
    if !dev.metadata().map_err(|e| e.to_string())?.is_file() { return Err("invalid development file".into()); }
    let train_path = stage.join("training.csv");
    let validation_path = stage.join("validation_features.csv");
    let mut train = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&train_path).map_err(|e| e.to_string())?;
    let mut validation = OpenOptions::new().write(true).create_new(true).mode(0o600).open(&validation_path).map_err(|e| e.to_string())?;
    let labels = crate::development_inputs::separate_stream(dev, &dev_hash, &protocol.target_column, &protocol.train, &protocol.validation,
        &mut train, &mut validation, MAX_DATA_FILE)?;
    train.sync_all().and_then(|_| validation.sync_all()).map_err(|e| e.to_string())?;
    let labels = serde_json::to_vec(&labels).map_err(|e| e.to_string())?;
    let receipt = SnapshotReceipt { identity: identity.clone(), training_sha256: hash_data_file(&train_path)?,
        validation_features_sha256: hash_data_file(&validation_path)?, labels_sha256: sha(&labels) };
    let result = snapshot_path(root, identity);
    fs::DirBuilder::new().mode(0o700).create(&result).map_err(|e| format!("snapshot already exists or cannot be created: {e}"))?;
    // An incomplete final directory remains fenced if publication is interrupted.
    fs::rename(&train_path, result.join("training.csv")).map_err(|e| e.to_string())?;
    fs::rename(&validation_path, result.join("validation_features.csv")).map_err(|e| e.to_string())?;
    write_file(&result.join("labels.json"), &labels)?;
    fs::File::open(&result).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    write_file(&result.join("receipt.json"), &serde_json::to_vec(&receipt).map_err(|e| e.to_string())?)?;
    fs::File::open(&result).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    fs::remove_dir_all(&stage).map_err(|e| format!("snapshot published but staging cleanup failed: {e}"))?;
    fs::File::open(root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    Ok(result)
}

/// Prepare under the owned state directory while holding the common SQLite
/// writer fence. The reservation covers three bounded 256 MiB data streams,
/// labels, manifests and publication overhead; interrupted files remain counted
/// by state admission. This deliberately requires an explicit larger state
/// budget rather than bypassing the default 128 MiB ceiling.
pub fn prepare_managed(state_root: &Path, dataset_id: &str, identity: &SnapshotIdentity) -> Result<PathBuf> {
    const PEAK_RESERVE: u64 = 896 * 1024 * 1024;
    crate::execution::with_state_write(state_root, PEAK_RESERVE, crate::execution::StorageWriteKind::JobAdmission,
        Some(&identity.project_id), |connection| {
            let root = state_root.join("custom-snapshots");
            match fs::DirBuilder::new().mode(0o700).create(&root) {
                Ok(()) => fs::File::open(state_root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {},
                Err(e) => return Err(e.to_string()),
            }
            let meta = fs::symlink_metadata(&root).map_err(|e| e.to_string())?;
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            if !meta.is_dir() || meta.file_type().is_symlink() || meta.uid() != unsafe { libc::geteuid() }
                || meta.permissions().mode() & 0o077 != 0 {
                return Err("managed snapshot root must be private and controller-owned".into());
            }
            prepare_registered(connection, &root, dataset_id, identity)
        })
}

/// Compute metrics against the snapshot's private labels. The caller must
/// independently establish that predictions came from the stated weights;
/// passing worker-authored identity fields alone does not do that.
pub fn evaluate_predictions(
    connection: &rusqlite::Connection, root: &Path, dataset_id: &str, snapshot: &SnapshotIdentity,
    expected: &crate::trusted_evaluation::EvaluationIdentity, predictions: &crate::trusted_evaluation::PredictionBatch,
) -> Result<crate::trusted_evaluation::EvaluationReceipt> {
    if dataset_id != snapshot.dataset_id || expected.dataset_id != snapshot.dataset_id || expected.project_id != snapshot.project_id || expected.job_id != snapshot.job_id || expected.candidate_id != snapshot.candidate_id
        || expected.dataset_sha256 != snapshot.dataset_sha256 || expected.split_sha256 != snapshot.protocol_sha256
        || expected.source_sha256 != snapshot.source_sha256 || expected.environment_id != snapshot.environment_id {
        return Err("evaluation does not match snapshot identity".into());
    }
    let (protocol, protocol_hash) = crate::custom_protocol::load(connection, &snapshot.project_id, dataset_id)?;
    if protocol_hash != snapshot.protocol_sha256 || protocol.dataset_sha256 != snapshot.dataset_sha256 {
        return Err("evaluation does not match frozen protocol".into());
    }
    let receipt = verify(root, snapshot)?;
    let label_bytes = read_file(&snapshot_path(root, snapshot).join("labels.json"))?;
    if sha(&label_bytes) != receipt.labels_sha256 { return Err("snapshot labels changed during evaluation".into()); }
    let labels: Vec<(u64, f64)> = serde_json::from_slice(&label_bytes).map_err(|e| e.to_string())?;
    let mut label_ids: Vec<_> = labels.iter().map(|(id, _)| *id).collect();
    label_ids.sort_unstable();
    if label_ids != protocol.validation { return Err("snapshot labels do not match frozen validation rows".into()); }
    crate::trusted_evaluation::evaluate(expected, protocol.metric, &labels, predictions)
}
