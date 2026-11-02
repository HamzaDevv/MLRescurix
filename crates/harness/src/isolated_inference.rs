//! Controller-owned inference for a frozen snapshot. This is a building block,
//! not an admission or promotion authority: callers must govern fit, durable
//! artifacts, storage reservations, and delivery separately.
use crate::{agents::sandbox::{SandboxLimits, SandboxRunner}, custom_snapshot::{self, SnapshotIdentity}, trusted_evaluation::{EvaluationIdentity, EvaluationReceipt, PredictionBatch}, Result};
use sha2::{Digest, Sha256};
use std::{fs::{self, OpenOptions}, io::{Read, Write}, os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt}, path::{Path, PathBuf}};

const SOURCE_LIMIT: u64 = 100_000;
const WEIGHTS_LIMIT: u64 = 64 * 1024 * 1024;
const FEATURES_LIMIT: u64 = 256 * 1024 * 1024;

// Read once from an already-open regular file, and copy the verified bytes into
// a private controller-owned directory. The worker never sees the source path.
fn copy_checked(source: &Path, destination: &Path, expected: &str, limit: u64) -> Result<()> {
    let mut input = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(source).map_err(|e| e.to_string())?;
    let meta = input.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > limit { return Err("inference input is not a bounded regular file".into()); }
    let mut output = OpenOptions::new().write(true).create_new(true).mode(0o600).open(destination).map_err(|e| e.to_string())?;
    let mut hash = Sha256::new(); let mut total = 0u64; let mut buffer = [0u8; 65536];
    loop {
        let n = input.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 { break; }
        total = total.checked_add(n as u64).ok_or("input size overflow")?;
        if total > limit { return Err("inference input exceeds bound".into()); }
        hash.update(&buffer[..n]); output.write_all(&buffer[..n]).map_err(|e| e.to_string())?;
    }
    if format!("{:x}", hash.finalize()) != expected { return Err("inference input hash mismatch".into()); }
    output.sync_all().map_err(|e| e.to_string())?;
    fs::set_permissions(destination, fs::Permissions::from_mode(0o444)).map_err(|e| e.to_string())?;
    Ok(())
}

/// A controller-authored fit receipt. It binds the exact snapshot and image to
/// the bytes collected from the isolated process; it does not certify model
/// quality or authorize promotion.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitReceipt {
    pub snapshot: SnapshotIdentity,
    pub training_sha256: String,
    pub weights_sha256: String,
    pub weights_bytes: u64,
}

fn private_root(root: &Path) -> Result<()> {
    let meta = fs::symlink_metadata(root).map_err(|e| e.to_string())?;
    if !meta.is_dir() || meta.file_type().is_symlink()
        || meta.uid() != unsafe { libc::geteuid() } || meta.mode() & 0o077 != 0 {
        return Err("inference staging root must be controller-owned and private".into());
    }
    Ok(())
}

fn validate_runtime(snapshot: &SnapshotIdentity, limits: &SandboxLimits) -> Result<()> {
    if !(1..=86_400_000).contains(&limits.max_wall_ms)
        || !(64..=65_536).contains(&limits.max_memory_mb)
        || !limits.max_cpus.is_finite() || !(0.1..=64.0).contains(&limits.max_cpus)
        || !(1..=64 * 1024 * 1024).contains(&limits.max_output_bytes)
        || !(1..=4096).contains(&limits.max_scratch_mb) || !(1..=4096).contains(&limits.pids_limit) {
        return Err("isolated inference limits outside bounds".into());
    }
    let image_hash = snapshot.environment_id.strip_prefix("sha256:").ok_or("environment must be a pinned Docker image ID")?;
    if image_hash.len() != 64 || !image_hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("environment must be a pinned Docker image ID".into());
    }
    Ok(())
}

/// Fit with training data only. The program prints a JSON string containing
/// lowercase hexadecimal model bytes. The controller publishes the bytes and
/// receipt under a new exclusive directory; incomplete directories remain
/// fenced after failure. Storage admission must reserve space before this call.
pub fn fit(
    connection: &rusqlite::Connection, snapshot_root: &Path, dataset_id: &str,
    snapshot: &SnapshotIdentity, source: &Path, limits: &SandboxLimits,
) -> Result<(PathBuf, FitReceipt)> {
    private_root(snapshot_root)?;
    validate_runtime(snapshot, limits)?;
    let receipt = custom_snapshot::verify(snapshot_root, snapshot)?;
    let snapshot_dir = custom_snapshot::prepare_registered(connection, snapshot_root, dataset_id, snapshot)?;
    // Fence the job identity before starting an untrusted process. A crash or
    // failed run leaves an incomplete directory that cannot be silently reused.
    let name = format!("fit-{:x}", Sha256::digest(serde_json::to_vec(snapshot).map_err(|e| e.to_string())?));
    let destination = snapshot_root.join(name);
    fs::DirBuilder::new().mode(0o700).create(&destination).map_err(|e| format!("fit identity already exists or cannot be published: {e}"))?;
    let mut intent = OpenOptions::new().write(true).create_new(true).mode(0o600).open(destination.join("intent.json")).map_err(|e| e.to_string())?;
    intent.write_all(&serde_json::to_vec(snapshot).map_err(|e| e.to_string())?).and_then(|_| intent.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(&destination).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(snapshot_root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    let stage = tempfile::Builder::new().prefix(".fit-").tempdir_in(snapshot_root).map_err(|e| e.to_string())?;
    fs::set_permissions(stage.path(), fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
    copy_checked(source, &stage.path().join("candidate.py"), &snapshot.source_sha256, SOURCE_LIMIT)?;
    copy_checked(&stage.path().join("candidate.py"), &destination.join("candidate.py"), &snapshot.source_sha256, SOURCE_LIMIT)?;
    fs::File::open(&destination).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    copy_checked(&snapshot_dir.join("training.csv"), &stage.path().join("training.csv"), &receipt.training_sha256, FEATURES_LIMIT)?;
    let args = ["python3", "candidate.py", "--fit", "--input", "/workspace/training.csv"].into_iter().map(String::from).collect::<Vec<_>>();
    let result = SandboxRunner::execute_argv(stage.path(), &args, limits, Some(&snapshot.environment_id))?;
    if result.exit_code != 0 { return Err("isolated fit failed; worker output withheld".into()); }
    let encoded: String = serde_json::from_str(&result.stdout).map_err(|_| "isolated fit returned invalid model bytes")?;
    if encoded.is_empty() || encoded.len() % 2 != 0 || encoded.len() as u64 > WEIGHTS_LIMIT * 2
        || !encoded.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err("isolated fit returned invalid model bytes".into());
    }
    let decode = |c: u8| if c <= b'9' { c - b'0' } else { c - b'a' + 10 };
    let bytes: Vec<u8> = encoded.as_bytes().chunks_exact(2).map(|pair| decode(pair[0]) * 16 + decode(pair[1])).collect();
    // Verify inputs have not been replaced while the worker ran before binding.
    custom_snapshot::verify(snapshot_root, snapshot)?;
    let fit = FitReceipt { snapshot: snapshot.clone(), training_sha256: receipt.training_sha256,
        weights_sha256: format!("{:x}", Sha256::digest(&bytes)), weights_bytes: bytes.len() as u64 };
    let write = |path: &Path, data: &[u8]| -> Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).mode(0o600).open(path).map_err(|e| e.to_string())?;
        f.write_all(data).and_then(|_| f.sync_all()).map_err(|e| e.to_string())
    };
    let weights_path = destination.join("weights");
    write(&weights_path, &bytes)?;
    write(&destination.join("receipt.json"), &serde_json::to_vec(&fit).map_err(|e| e.to_string())?)?;
    fs::File::open(&destination).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(snapshot_root).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    Ok((weights_path, fit))
}

/// Load only the controller-published fit receipt for this exact snapshot and
/// score its bound weights. A missing or malformed completion receipt is an
/// incomplete fit, never a recoverable success.
pub fn evaluate_fitted(
    connection: &rusqlite::Connection, snapshot_root: &Path, dataset_id: &str,
    snapshot: &SnapshotIdentity, limits: &SandboxLimits,
) -> Result<EvaluationReceipt> {
    private_root(snapshot_root)?;
    let verified = custom_snapshot::verify(snapshot_root, snapshot)?;
    let name = format!("fit-{:x}", Sha256::digest(serde_json::to_vec(snapshot).map_err(|e| e.to_string())?));
    let dir = snapshot_root.join(name);
    let metadata = fs::symlink_metadata(&dir).map_err(|e| e.to_string())?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() || metadata.uid() != unsafe { libc::geteuid() } || metadata.mode() & 0o077 != 0 {
        return Err("invalid fit receipt directory".into());
    }
    let read_bounded = |name: &str| -> Result<Vec<u8>> {
        let mut file = OpenOptions::new().read(true).custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK).open(dir.join(name)).map_err(|e| e.to_string())?;
        let meta = file.metadata().map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() > 16 * 1024 { return Err("invalid fit receipt file".into()); }
        let mut bytes = Vec::new(); Read::by_ref(&mut file).take(16 * 1024 + 1).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() > 16 * 1024 { return Err("invalid fit receipt file".into()); }
        Ok(bytes)
    };
    let intent: SnapshotIdentity = serde_json::from_slice(&read_bounded("intent.json")?).map_err(|e| e.to_string())?;
    let fit: FitReceipt = serde_json::from_slice(&read_bounded("receipt.json")?).map_err(|e| e.to_string())?;
    if intent != *snapshot || fit.snapshot != *snapshot || fit.training_sha256 != verified.training_sha256
        || fit.weights_bytes == 0 || fit.weights_bytes > WEIGHTS_LIMIT {
        return Err("fit receipt does not match snapshot".into());
    }
    let metadata = fs::symlink_metadata(dir.join("weights")).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() != fit.weights_bytes {
        return Err("fit weights do not match receipt".into());
    }
    // Persist the evaluation intent before launching the container. An absent
    // completion receipt after a crash or error is ambiguous and must never be
    // treated as permission to rerun the same identity automatically.
    let mut evaluation_intent = OpenOptions::new().write(true).create_new(true).mode(0o600)
        .open(dir.join("evaluation-intent.json"))
        .map_err(|e| format!("evaluation identity already exists or cannot be fenced: {e}"))?;
    evaluation_intent.write_all(&serde_json::to_vec(&fit).map_err(|e| e.to_string())?)
        .and_then(|_| evaluation_intent.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(&dir).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    let evaluated = evaluate(connection, snapshot_root, dataset_id, snapshot, &dir.join("candidate.py"), &dir.join("weights"), &fit.weights_sha256, limits)?;
    let mut out = OpenOptions::new().write(true).create_new(true).mode(0o600).open(dir.join("evaluation.json"))
        .map_err(|e| format!("evaluation receipt already exists or cannot be published: {e}"))?;
    out.write_all(&serde_json::to_vec(&evaluated).map_err(|e| e.to_string())?).and_then(|_| out.sync_all()).map_err(|e| e.to_string())?;
    fs::File::open(&dir).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    Ok(evaluated)
}

/// Run the exact pinned image and checked source and weights with only
/// validation features mounted. Source must print a JSON array of [row_id,
/// prediction] pairs; metric values or identity claims from the worker are
/// never used. Snapshot root and its ancestors must be controller-owned.
pub fn evaluate(
    connection: &rusqlite::Connection, snapshot_root: &Path, dataset_id: &str,
    snapshot: &SnapshotIdentity, source: &Path, weights: &Path,
    weights_sha256: &str, limits: &SandboxLimits,
) -> Result<EvaluationReceipt> {
    let identity = EvaluationIdentity { project_id: snapshot.project_id.clone(), job_id: snapshot.job_id.clone(),
        candidate_id: snapshot.candidate_id.clone(), dataset_id: snapshot.dataset_id.clone(),
        dataset_sha256: snapshot.dataset_sha256.clone(), split_sha256: snapshot.protocol_sha256.clone(),
        source_sha256: snapshot.source_sha256.clone(), environment_id: snapshot.environment_id.clone(), weights_sha256: weights_sha256.into() };
    identity.validate()?;
    validate_runtime(snapshot, limits)?;
    private_root(snapshot_root)?;
    let receipt = custom_snapshot::verify(snapshot_root, snapshot)?;
    let dir = custom_snapshot::prepare_registered(connection, snapshot_root, dataset_id, snapshot)?;
    // Admission must reserve this staging space before calling this helper.
    let stage = tempfile::Builder::new().prefix(".inference-").tempdir_in(snapshot_root).map_err(|e| e.to_string())?;
    fs::set_permissions(stage.path(), fs::Permissions::from_mode(0o755)).map_err(|e| e.to_string())?;
    copy_checked(source, &stage.path().join("candidate.py"), &snapshot.source_sha256, SOURCE_LIMIT)?;
    copy_checked(weights, &stage.path().join("weights"), weights_sha256, WEIGHTS_LIMIT)?;
    copy_checked(&dir.join("validation_features.csv"), &stage.path().join("validation.csv"), &receipt.validation_features_sha256, FEATURES_LIMIT)?;
    let args = ["python3", "candidate.py", "--input", "/workspace/validation.csv", "--weights", "/workspace/weights"].into_iter().map(String::from).collect::<Vec<_>>();
    let result = SandboxRunner::execute_argv(stage.path(), &args, limits, Some(&snapshot.environment_id))?;
    if result.exit_code != 0 { return Err("isolated inference failed; worker output withheld".into()); }
    let rows: Vec<(u64, f64)> = serde_json::from_str(&result.stdout).map_err(|_| "isolated inference returned invalid predictions")?;
    let predictions = PredictionBatch::scalar(identity.clone(), rows);
    custom_snapshot::evaluate_predictions(connection, snapshot_root, dataset_id, snapshot, &identity, &predictions)
}

/// Managed variants hold the state writer fence while bounded staging and
/// receipt publication occur. They do not grant job admission or promotion.
pub fn managed_fit(
    state_root: &Path, dataset_id: &str, snapshot: &SnapshotIdentity, source: &Path, limits: &SandboxLimits,
) -> Result<(PathBuf, FitReceipt)> {
    crate::execution::with_state_write(state_root, 384 * 1024 * 1024, crate::execution::StorageWriteKind::JobAdmission,
        Some(&snapshot.project_id), |connection| fit(connection, &state_root.join("custom-snapshots"), dataset_id, snapshot, source, limits))
}

pub fn managed_evaluate_fitted(
    state_root: &Path, dataset_id: &str, snapshot: &SnapshotIdentity, limits: &SandboxLimits,
) -> Result<EvaluationReceipt> {
    crate::execution::with_state_write(state_root, 384 * 1024 * 1024, crate::execution::StorageWriteKind::JobAdmission,
        Some(&snapshot.project_id), |connection| evaluate_fitted(connection, &state_root.join("custom-snapshots"), dataset_id, snapshot, limits))
}
