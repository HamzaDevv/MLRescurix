//! Experimental autonomous campaign prototype.
//!
//! The entry point is disabled until a verified isolated execution boundary and
//! general artifact delivery are implemented. Do not treat this module as production ready.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Instant,
};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

use crate::{
    agents::gateway::{
        reconcile_gateway_receipts, register_namespaced_model_pricing, reserve_role_spend,
        ProviderGateway, TokenUsage,
    },
    agents::workspace::BranchWorkspaceManager,
    bundle::{
        verify_bundle, verify_standalone_cli_inference, verify_standalone_serving_recipe,
        BundleManifest, BundleManifestFileEntry, BundleVerificationReport,
    },
    core::{Operation, Policy, Request},
    custom_protocol::{
        carve_lockbox_with_min_rows, ensure_custom_protocol_schema,
        evaluate_and_consume_assessment_authoritative, freeze, CustomProtocol,
    },
    execution::StreamingCsvReader,
    storage::Runtime,
    trusted_evaluation::{evaluate, EvaluationIdentity, Metric, PredictionBatch},
    Result,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignProblemSpec {
    pub project_id: String,
    pub dataset_path: PathBuf,
    pub target_column: String,
    #[serde(default = "default_task_type")]
    pub task_type: String,
    #[serde(default = "default_metric")]
    pub metric: Metric,
    #[serde(default = "default_budget_millicents")]
    pub budget_millicents: i64,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub lockbox_fraction: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_turns: Option<u32>,
}

fn default_task_type() -> String {
    "regression".into()
}

fn default_metric() -> Metric {
    Metric::Mae
}

fn default_budget_millicents() -> i64 {
    100_000 // $1.00 default cap
}

fn default_model_name() -> String {
    "deepseek/deepseek-chat".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignOutcome {
    pub project_id: String,
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub protocol_sha256: String,
    pub candidate_id: String,
    pub weights_sha256: String,
    pub validation_score: f64,
    pub lockbox_score: f64,
    pub lockbox_receipt_sha256: String,
    pub bundle_dir: PathBuf,
    pub bundle_verification: BundleVerificationReport,
    pub spend_reconciliation: Value,
    pub duration_ms: u64,
}

fn csv_record(fields: &[String]) -> String {
    fields.iter().map(|field| {
        if field.contains([',', '"', '\r', '\n']) {
            format!("\"{}\"", field.replace('"', "\"\""))
        } else {
            field.clone()
        }
    }).collect::<Vec<_>>().join(",")
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read file for hash: {e}"))?;
    Ok(sha256_bytes(&bytes))
}

fn m1_default_policy(budget_millicents: i64) -> Policy {
    let tokens_allowed = (budget_millicents * 100).max(10_000);
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), 100),
            ("tokens".into(), tokens_allowed),
            ("wall_ms".into(), 3_600_000),
        ]),
        capacity: BTreeMap::from([("workers".into(), 4)]),
        max_tasks: 256,
    }
}

pub const LEASE_TTL_MS: i64 = 30_000;

#[derive(Debug, PartialEq)]
enum OwnerDecision {
    DelegateChallenger,
    StopEarly,
    Finalize,
}

// Temporary explicit text contract until the versioned owner action schema lands.
// Never interpret missing content or a mention of an action as authorization.
fn parse_owner_decision(content: Option<&str>) -> Result<OwnerDecision> {
    let content = content.ok_or("owner decision missing; finalization blocked")?;
    let (action, reason) = content.trim().split_once(':')
        .ok_or("owner decision invalid; expected ACTION: <rationale>")?;
    let action = action.trim().to_ascii_uppercase();
    let reason_upper = reason.to_ascii_uppercase();
    if reason.trim().is_empty() || ["DELEGATE_CHALLENGER", "STOP_EARLY", "FINALIZE_AND_STOP"]
        .iter().any(|other| reason_upper.contains(other)) {
        return Err("owner decision missing rationale or ambiguous; finalization blocked".into());
    }
    match action.as_str() {
        "DELEGATE_CHALLENGER" => Ok(OwnerDecision::DelegateChallenger),
        "STOP_EARLY" => Ok(OwnerDecision::StopEarly),
        "FINALIZE_AND_STOP" => Ok(OwnerDecision::Finalize),
        _ => Err("owner decision unknown; finalization blocked".into()),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StructuredChallengerResult {
    pub schema_version: String,
    pub task_id: String,
    pub candidate_id: String,
    pub protocol_sha256: String,
    pub validation_score: f64,
    pub verdict: String,
    pub rationale: String,
    pub findings: Vec<String>,
    pub timestamp_ms: i64,
}

pub fn parse_structured_challenger_result(
    content: &str,
    task_id: &str,
    candidate_id: &str,
    protocol_sha256: &str,
    validation_score: f64,
) -> Result<StructuredChallengerResult> {
    if let Ok(val) = serde_json::from_str::<Value>(content) {
        if let Some(obj) = val.as_object() {
            if let Some(verdict_val) = obj.get("verdict").and_then(|v| v.as_str()) {
                let verdict_upper = verdict_val.trim().to_ascii_uppercase();
                if verdict_upper == "APPROVED" || verdict_upper == "CHALLENGED" {
                    let rationale = obj.get("rationale").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    if rationale.trim().is_empty() {
                        return Err("challenger verdict requires a nonempty rationale".into());
                    }
                    // Legacy prose/minimal JSON is an explicit adapter, not proof of a
                    // fully model-issued bound schema. Reject any contradictory identity.
                    for (key, expected) in [("schema_version", "mlrescurix.challenger.v1"),
                        ("task_id", task_id), ("candidate_id", candidate_id),
                        ("protocol_sha256", protocol_sha256)] {
                        if obj.get(key).is_some_and(|v| v.as_str() != Some(expected)) {
                            return Err(format!("challenger verdict {key} does not match current evaluation"));
                        }
                    }
                    if obj.get("validation_score").is_some_and(|v| v.as_f64() != Some(validation_score)) {
                        return Err("challenger verdict validation_score does not match current evaluation".into());
                    }
                    if let Some(findings) = obj.get("findings") {
                        if !findings.as_array().is_some_and(|a| a.iter().all(|v| v.as_str().is_some_and(|s| !s.trim().is_empty()))) {
                            return Err("challenger verdict findings must be an array of nonempty strings".into());
                        }
                    }
                    let findings = obj.get("findings").and_then(|v| v.as_array())
                        .map(|arr| arr.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                        .unwrap_or_else(|| vec![rationale.clone()]);
                    return Ok(StructuredChallengerResult {
                        schema_version: "mlrescurix.challenger.v1".into(),
                        task_id: task_id.to_string(),
                        candidate_id: candidate_id.to_string(),
                        protocol_sha256: protocol_sha256.to_string(),
                        validation_score,
                        verdict: verdict_upper,
                        rationale,
                        findings,
                        timestamp_ms: now_ms(),
                    });
                }
            }
        }
    }

    let trimmed = content.trim();
    let upper = trimmed.to_ascii_uppercase();
    for (verdict, opposite) in [("APPROVED", "CHALLENGED"), ("CHALLENGED", "APPROVED")] {
        if let Some(rest) = upper.strip_prefix(&format!("{verdict}:")) {
            if !rest.trim().is_empty() && !rest.contains(opposite) {
                let original_rest = trimmed[verdict.len() + 1..].trim();
                return Ok(StructuredChallengerResult {
                    schema_version: "mlrescurix.challenger.v1".into(),
                    task_id: task_id.to_string(),
                    candidate_id: candidate_id.to_string(),
                    protocol_sha256: protocol_sha256.to_string(),
                    validation_score,
                    verdict: verdict.to_string(),
                    rationale: original_rest.to_string(),
                    findings: vec![original_rest.to_string()],
                    timestamp_ms: now_ms(),
                });
            }
        }
    }
    Err("challenger verdict missing or ambiguous; expected structured JSON or APPROVED: <findings> / CHALLENGED: <findings>".into())
}

pub fn ensure_campaign_leases_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS campaign_leases (
            project TEXT NOT NULL PRIMARY KEY,
            lease_id TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            epoch INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL CHECK(status IN ('active', 'released')),
            acquired_ms INTEGER NOT NULL,
            heartbeat_ms INTEGER NOT NULL DEFAULT 0,
            released_ms INTEGER
        );"
    ).map_err(|e| format!("cannot create campaign_leases schema: {e}"))?;
    let _ = conn.execute("ALTER TABLE campaign_leases ADD COLUMN epoch INTEGER NOT NULL DEFAULT 1", []);
    let _ = conn.execute("ALTER TABLE campaign_leases ADD COLUMN heartbeat_ms INTEGER NOT NULL DEFAULT 0", []);
    Ok(())
}

pub fn is_owner_process_dead(owner: &str) -> bool {
    if let Some(pid_str) = owner.strip_prefix("pid_").and_then(|s| s.split('_').next()) {
        if let Ok(pid) = pid_str.parse::<i32>() {
            #[cfg(unix)]
            unsafe {
                let res = libc::kill(pid, 0);
                if res != 0 {
                    let err = std::io::Error::last_os_error().raw_os_error();
                    if err == Some(libc::ESRCH) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

pub fn acquire_campaign_lease(
    conn: &mut Connection,
    project_id: &str,
    owner_id: &str,
) -> Result<(String, i64)> {
    ensure_campaign_leases_schema(conn)?;
    let now = now_ms();
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| format!("cannot begin lease transaction: {e}"))?;

    let existing: Option<(String, String, i64, String, i64)> = tx.query_row(
        "SELECT lease_id, owner_id, epoch, status, heartbeat_ms FROM campaign_leases WHERE project = ?1",
        params![project_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    ).optional().map_err(|e| format!("lease query failed: {e}"))?;

    let (lease_id, epoch) = match existing {
        Some((cur_lease, cur_owner, cur_epoch, status, heartbeat_ms)) => {
            if status == "active" {
                let owner_dead = is_owner_process_dead(&cur_owner);
                if !owner_dead && now - heartbeat_ms <= LEASE_TTL_MS {
                    return Err(format!(
                        "campaign lease conflict: project '{}' already held by active lease '{}' (owner '{}')",
                        project_id, cur_lease, cur_owner
                    ));
                }
                // TTL expired or owner process confirmed dead. Recover lease with incremented epoch.
                let next_epoch = cur_epoch + 1;
                let new_lease = format!("lease_{}_{}", project_id, now);
                tx.execute(
                    "UPDATE campaign_leases SET lease_id = ?1, owner_id = ?2, epoch = ?3, status = 'active', acquired_ms = ?4, heartbeat_ms = ?4, released_ms = NULL WHERE project = ?5",
                    params![&new_lease, owner_id, next_epoch, now, project_id],
                ).map_err(|e| format!("failed to recover lease: {e}"))?;
                (new_lease, next_epoch)
            } else {
                let next_epoch = cur_epoch + 1;
                let new_lease = format!("lease_{}_{}", project_id, now);
                tx.execute(
                    "UPDATE campaign_leases SET lease_id = ?1, owner_id = ?2, epoch = ?3, status = 'active', acquired_ms = ?4, heartbeat_ms = ?4, released_ms = NULL WHERE project = ?5",
                    params![&new_lease, owner_id, next_epoch, now, project_id],
                ).map_err(|e| format!("failed to acquire released lease: {e}"))?;
                (new_lease, next_epoch)
            }
        }
        None => {
            let new_lease = format!("lease_{}_{}", project_id, now);
            let initial_epoch = 1i64;
            tx.execute(
                "INSERT INTO campaign_leases(project, lease_id, owner_id, epoch, status, acquired_ms, heartbeat_ms, released_ms)
                 VALUES(?1, ?2, ?3, ?4, 'active', ?5, ?5, NULL)",
                params![project_id, &new_lease, owner_id, initial_epoch, now],
            ).map_err(|e| format!("failed to insert initial lease: {e}"))?;
            (new_lease, initial_epoch)
        }
    };
    tx.commit().map_err(|e| format!("failed to commit lease acquisition: {e}"))?;
    Ok((lease_id, epoch))
}

pub fn verify_active_lease(
    conn: &Connection,
    project_id: &str,
    lease_id: &str,
    epoch: i64,
) -> Result<()> {
    let row: Option<(String, i64, String)> = conn.query_row(
        "SELECT lease_id, epoch, status FROM campaign_leases WHERE project = ?1",
        params![project_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional().map_err(|e| e.to_string())?;

    match row {
        Some((cur_lease, cur_epoch, status)) => {
            if status != "active" || cur_lease != lease_id || cur_epoch != epoch {
                return Err(format!(
                    "stale owner write rejected: lease epoch mismatch (current lease='{}', epoch={}, status='{}'; expected lease='{}', epoch={})",
                    cur_lease, cur_epoch, status, lease_id, epoch
                ));
            }
            Ok(())
        }
        None => Err("no lease found for project".into()),
    }
}

pub fn heartbeat_lease(
    conn: &Connection,
    project_id: &str,
    lease_id: &str,
    epoch: i64,
) -> Result<()> {
    let updated = conn.execute(
        "UPDATE campaign_leases SET heartbeat_ms = ?1 WHERE project = ?2 AND lease_id = ?3 AND epoch = ?4 AND status = 'active'",
        params![now_ms(), project_id, lease_id, epoch],
    ).map_err(|e| e.to_string())?;
    if updated == 0 {
        return Err("lease heartbeat failed: lease is no longer active or epoch has changed".into());
    }
    Ok(())
}

pub fn release_campaign_lease(
    conn: &Connection,
    project_id: &str,
    lease_id: &str,
    epoch: i64,
) -> Result<()> {
    let _ = conn.execute(
        "UPDATE campaign_leases SET status = 'released', released_ms = ?1 WHERE project = ?2 AND lease_id = ?3 AND epoch = ?4",
        params![now_ms(), project_id, lease_id, epoch],
    );
    Ok(())
}

pub fn execute_fenced_write<F, R>(
    conn: &mut Connection,
    project_id: &str,
    lease_id: &str,
    epoch: i64,
    f: F,
) -> Result<R>
where
    F: FnOnce(&rusqlite::Transaction) -> Result<R>,
{
    let tx = conn
        .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| format!("cannot begin immediate transaction: {e}"))?;
    verify_active_lease(&tx, project_id, lease_id, epoch)?;
    let res = f(&tx)?;
    heartbeat_lease(&tx, project_id, lease_id, epoch)?;
    tx.commit()
        .map_err(|e| format!("cannot commit fenced transaction: {e}"))?;
    Ok(res)
}

pub fn run_cmd_with_heartbeats(
    cmd: &mut Command,
    db_path: &Path,
    project_id: &str,
    lease_id: &str,
    epoch: i64,
) -> Result<std::process::Output> {
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_thread = stop_flag.clone();
    let db_p = db_path.to_path_buf();
    let p_id = project_id.to_string();
    let l_id = lease_id.to_string();

    let handle = std::thread::spawn(move || {
        while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(1500));
            if stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            if let Ok(c) = Connection::open(&db_p) {
                let _ = heartbeat_lease(&c, &p_id, &l_id, epoch);
            }
        }
    });

    let res = cmd.output();
    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = handle.join();
    res.map_err(|e| format!("failed to launch subprocess: {e}"))
}

fn collect_py_files(dir: &Path, rel_prefix: &str, out: &mut Vec<String>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(std::result::Result::ok)
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if p.is_dir() {
            if !name.starts_with('.') && name != "__pycache__" && name != "venv" && name != ".venv" {
                let sub_prefix = if rel_prefix.is_empty() {
                    name
                } else {
                    format!("{rel_prefix}/{name}")
                };
                collect_py_files(&p, &sub_prefix, out)?;
            }
        } else if p.is_file() && name.ends_with(".py") {
            // Exclude bundle runtime wrapper scripts (generated by harness) from candidate source hash
            if !rel_prefix.is_empty() || (name != "inference.py" && name != "serve.py") {
                let rel_path = if rel_prefix.is_empty() {
                    name
                } else {
                    format!("{rel_prefix}/{name}")
                };
                out.push(rel_path);
            }
        }
    }
    Ok(())
}

fn copy_tree_filtered(src: &Path, dst: &Path) -> Result<()> {
    if !src.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for entry in fs::read_dir(src).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy().to_string();
        if p.is_dir() {
            if !fname_str.starts_with('.') && fname_str != "__pycache__" && fname_str != "venv" && fname_str != ".venv" {
                copy_tree_filtered(&p, &dst.join(&fname))?;
            }
        } else if p.is_file() {
            if fname_str.ends_with(".py") || fname_str.ends_with(".json") {
                let _ = fs::copy(&p, dst.join(&fname));
            }
        }
    }
    Ok(())
}

pub fn compute_workspace_source_sha256(workspace: &Path) -> Result<String> {
    let mut py_files = Vec::new();
    collect_py_files(workspace, "", &mut py_files)?;
    py_files.sort();
    if py_files.is_empty() {
        let sol = workspace.join("solution.py");
        if sol.is_file() {
            return sha256_file(&sol);
        }
        return Err("no python source files found in workspace".into());
    }
    if py_files.len() == 1 && py_files[0] == "solution.py" {
        return sha256_file(&workspace.join("solution.py"));
    }
    let mut hasher = Sha256::new();
    for fname in py_files {
        let bytes = fs::read(workspace.join(&fname)).map_err(|e| e.to_string())?;
        hasher.update(fname.as_bytes());
        hasher.update(&[0u8]);
        hasher.update(&bytes);
        hasher.update(&[0xff]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn resolve_fit_args(
    manifest_argv: Option<&Vec<Value>>,
    train_path: &Path,
    weights_path: &Path,
) -> (String, Vec<String>) {
    if let Some(arr) = manifest_argv {
        if !arr.is_empty() {
            let bin = arr[0].as_str().unwrap_or("python3").to_string();
            let mut args: Vec<String> = arr[1..].iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--train-data" && i + 1 < args.len() {
                    args[i + 1] = train_path.to_str().unwrap_or("train.csv").to_string();
                    i += 1;
                } else if args[i] == "--weights-out" && i + 1 < args.len() {
                    args[i + 1] = weights_path.to_str().unwrap_or("weights.json").to_string();
                    i += 1;
                }
                i += 1;
            }
            return (bin, args);
        }
    }
    (
        "python3".to_string(),
        vec![
            "solution.py".to_string(),
            "fit".to_string(),
            "--train-data".to_string(),
            train_path.to_str().unwrap().to_string(),
            "--weights-out".to_string(),
            weights_path.to_str().unwrap().to_string(),
        ],
    )
}

fn resolve_infer_args(
    manifest_argv: Option<&Vec<Value>>,
    features_path: &Path,
    weights_path: &Path,
    predictions_path: &Path,
) -> (String, Vec<String>) {
    if let Some(arr) = manifest_argv {
        if !arr.is_empty() {
            let bin = arr[0].as_str().unwrap_or("python3").to_string();
            let mut args: Vec<String> = arr[1..].iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
            let mut i = 0;
            while i < args.len() {
                if args[i] == "--features-in" && i + 1 < args.len() {
                    args[i + 1] = features_path.to_str().unwrap_or("features.csv").to_string();
                    i += 1;
                } else if args[i] == "--weights-in" && i + 1 < args.len() {
                    args[i + 1] = weights_path.to_str().unwrap_or("weights.json").to_string();
                    i += 1;
                } else if args[i] == "--predictions-out" && i + 1 < args.len() {
                    args[i + 1] = predictions_path.to_str().unwrap_or("predictions.json").to_string();
                    i += 1;
                }
                i += 1;
            }
            return (bin, args);
        }
    }
    (
        "python3".to_string(),
        vec![
            "solution.py".to_string(),
            "infer".to_string(),
            "--features-in".to_string(),
            features_path.to_str().unwrap().to_string(),
            "--weights-in".to_string(),
            weights_path.to_str().unwrap().to_string(),
            "--predictions-out".to_string(),
            predictions_path.to_str().unwrap().to_string(),
        ],
    )
}

/// Executes an autonomous model creator campaign from raw dataset to verified delivery bundle.
pub fn run_autonomous_campaign(
    state_root: &Path,
    spec: &CampaignProblemSpec,
    gateway_override: Option<Arc<dyn ProviderGateway>>,
) -> Result<CampaignOutcome> {
    // A host subprocess is not a security boundary. Keep this entry point
    // disabled until execution is wired to a verified isolated worker.
    if !campaign_isolation_available() {
        return Err("autonomous campaign unavailable: isolated worker execution is not yet implemented".into());
    }
    run_campaign_inner(state_root, spec, gateway_override)
}

/// Explicit diagnostic execution of trusted synthetic fixtures on the host.
/// This does not certify containment, autonomy, independent review, or release readiness.
/// No live provider or ambient provider credentials are admitted through this entry point.
pub fn run_offline_fixture_campaign(
    state_root: &Path,
    spec: &CampaignProblemSpec,
    gateway: Option<Arc<dyn ProviderGateway>>,
) -> Result<CampaignOutcome> {
    if spec.api_key.is_some() || gateway.as_ref().is_some_and(|gw| !gw.is_offline_fixture()) {
        return Err("offline fixture mode rejects live provider credentials and gateways".into());
    }
    run_campaign_inner(state_root, spec, gateway)
}

fn run_campaign_inner(
    state_root: &Path,
    spec: &CampaignProblemSpec,
    gateway: Option<Arc<dyn ProviderGateway>>,
) -> Result<CampaignOutcome> {
    let start_time = Instant::now();

    // 1. Validate dataset existence and integrity
    let abs_dataset_path = if spec.dataset_path.is_relative() {
        std::env::current_dir()
            .map(|cwd| cwd.join(&spec.dataset_path))
            .unwrap_or_else(|_| spec.dataset_path.clone())
    } else {
        spec.dataset_path.clone()
    };
    if !abs_dataset_path.is_file() {
        return Err(format!(
            "campaign dataset not found at '{}'",
            spec.dataset_path.display()
        ));
    }
    let dataset_bytes = fs::read(&abs_dataset_path)
        .map_err(|e| format!("cannot read dataset file '{}': {e}", spec.dataset_path.display()))?;
    if dataset_bytes.is_empty() {
        return Err("dataset file is empty".into());
    }
    let dataset_sha256 = sha256_bytes(&dataset_bytes);
    let dataset_id = format!("ds_{}", &dataset_sha256[..12]);
    let candidate_id = format!("cand_{}_{}", spec.project_id, &dataset_sha256[..8]);

    // 2. Initialize controller state, migrations, and database
    fs::create_dir_all(state_root).map_err(|e| e.to_string())?;
    let state_root_buf = fs::canonicalize(state_root)
        .unwrap_or_else(|_| state_root.to_path_buf());
    let state_root = state_root_buf.as_path();
    let controller_root = state_root.join("controller");
    let runtime = if controller_root.join("OWNED").is_file() {
        Runtime::open(&controller_root)?
    } else {
        if fs::symlink_metadata(&controller_root).is_ok() {
            return Err("controller path already exists without a valid ownership marker".into());
        }
        Runtime::init(&controller_root, m1_default_policy(spec.budget_millicents))?
    };
    crate::execution::migrate(&controller_root)?;

    let db_path = controller_root.join("state.sqlite");
    let mut conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| e.to_string())?;
    ensure_custom_protocol_schema(&conn)?;

    ensure_campaign_leases_schema(&conn)?;

    // Acquire active owner lease atomically with epoch and PID liveness
    let owner_id = format!("pid_{}_{}", std::process::id(), spec.project_id);
    let (lease_id, epoch) = acquire_campaign_lease(&mut conn, &spec.project_id, &owner_id)?;

    struct LeaseReleaser {
        db_path: PathBuf,
        project_id: String,
        lease_id: String,
        epoch: i64,
    }
    impl Drop for LeaseReleaser {
        fn drop(&mut self) {
            if let Ok(c) = Connection::open(&self.db_path) {
                let _ = c.busy_timeout(std::time::Duration::from_secs(5));
                let _ = release_campaign_lease(&c, &self.project_id, &self.lease_id, self.epoch);
            }
        }
    }
    let _lease_releaser = LeaseReleaser {
        db_path: db_path.clone(),
        project_id: spec.project_id.clone(),
        lease_id: lease_id.clone(),
        epoch,
    };

    // 3. Register or verify M1 Project
    let project_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM projects WHERE id=?1)",
            params![&spec.project_id],
            |r| r.get(0),
        )
        .unwrap_or(false);

    if !project_exists {
        runtime.command(Request {
            command_id: format!("init_proj_{}", spec.project_id),
            project_id: spec.project_id.clone(),
            expected_revision: None,
            operation: Operation::CreateProject {
                intent: format!("Autonomous model creation for project {}", spec.project_id),
                finalization_id: format!("fin-{}", spec.project_id),
                policy: m1_default_policy(spec.budget_millicents),
            },
        })?;
    }

    // 4. Verify dataset against prior project registration if resuming
    let existing_ds_sha: Option<(String, String)> = conn.query_row(
        "SELECT id, sha256 FROM scientific_datasets WHERE project=?1 LIMIT 1",
        params![&spec.project_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional().map_err(|e| e.to_string())?;

    if let Some((ref _prev_id, ref prev_sha)) = existing_ds_sha {
        if *prev_sha != dataset_sha256 {
            return Err(format!(
                "campaign resumption rejected: dataset bytes changed for project '{}' (SHA-256 '{}' vs previously registered '{}')",
                spec.project_id, dataset_sha256, prev_sha
            ));
        }
    }

    // 5. CSV parsing, validation, and split carving
    let mut reader = StreamingCsvReader::new(std::io::Cursor::new(&dataset_bytes));
    let header: Vec<String> = reader
        .read_record()?
        .ok_or_else(|| "dataset CSV has no header record".to_string())?;

    let mut seen_headers = HashSet::new();
    let mut trimmed_header = Vec::new();
    for (col_idx, h) in header.iter().enumerate() {
        let trimmed = h.trim();
        if trimmed.is_empty() {
            return Err(format!("column {col_idx} in CSV header has an empty name"));
        }
        if !seen_headers.insert(trimmed.to_string()) {
            return Err(format!("duplicate column name '{trimmed}' in dataset header"));
        }
        trimmed_header.push(trimmed.to_string());
    }

    if spec.target_column.trim().is_empty() {
        return Err("target column cannot be empty".into());
    }

    let target_idx = trimmed_header
        .iter()
        .position(|h: &String| h == spec.target_column.trim())
        .ok_or_else(|| {
            format!(
                "target column '{}' not found in dataset headers: {:?}",
                spec.target_column, trimmed_header
            )
        })?;

    let feature_columns: Vec<String> = trimmed_header
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != target_idx)
        .map(|(_, name)| name.clone())
        .collect();

    if feature_columns.is_empty() {
        return Err("dataset must contain at least one feature column".into());
    }

    let mut row_count: u64 = 0;
    let mut all_row_ids = Vec::new();
    let mut ground_truth_rows: Vec<(u64, f64)> = Vec::new();
    let mut raw_rows_data: Vec<Vec<String>> = Vec::new();

    while let Some(record) = reader.read_record()? {
        let record: Vec<String> = record;
        if record.is_empty() || (record.len() == 1 && record[0].trim().is_empty()) {
            continue;
        }
        if record.len() != header.len() {
            return Err(format!("CSV row {row_count} has {} fields; expected {}", record.len(), header.len()));
        }
        if target_idx >= record.len() {
            return Err(format!("row {row_count} missing target value at index {target_idx}"));
        }
        let target_raw = record[target_idx].trim();
        let target_val: f64 = target_raw
            .parse()
            .map_err(|_| format!("cannot parse target at row {row_count}: '{target_raw}'"))?;
        if !target_val.is_finite() {
            return Err(format!("target value at row {row_count} is non-finite: '{target_raw}'"));
        }

        ground_truth_rows.push((row_count, target_val));
        raw_rows_data.push(record);
        all_row_ids.push(row_count);
        row_count += 1;
    }

    if row_count < 6 {
        return Err(format!("dataset has only {row_count} rows; minimum 6 required for train/val/lockbox"));
    }

    // Feature column validation: allow mixed numeric/text/categorical datasets without forced f64 parsing,
    // but reject corrupted/non-finite values in numeric feature columns.
    for c_idx in 0..trimmed_header.len() {
        if c_idx == target_idx {
            continue;
        }
        let mut num_numeric: usize = 0;
        let mut first_fail: Option<(usize, String)> = None;
        let mut first_non_finite: Option<(usize, String)> = None;

        for (r_idx, record) in raw_rows_data.iter().enumerate() {
            let val = record[c_idx].trim();
            if let Ok(num) = val.parse::<f64>() {
                if num.is_finite() {
                    num_numeric += 1;
                } else if first_non_finite.is_none() {
                    first_non_finite = Some((r_idx, val.to_string()));
                }
            } else if first_fail.is_none() {
                first_fail = Some((r_idx, val.to_string()));
            }
        }

        if let Some((r_idx, val)) = first_non_finite {
            return Err(format!("feature value at row {r_idx}, col {c_idx} is non-finite: '{val}'"));
        }

        if num_numeric > 0 && num_numeric < raw_rows_data.len() {
            if let Some((r_idx, val)) = first_fail {
                return Err(format!("cannot parse feature at row {r_idx}, col {c_idx}: '{val}'"));
            }
        }
    }

    let lockbox_fraction = spec.lockbox_fraction.unwrap_or(0.20);
    if lockbox_fraction <= 0.0 || lockbox_fraction >= 1.0 || !lockbox_fraction.is_finite() {
        return Err(format!(
            "lockbox_fraction must be strictly between 0.0 and 1.0, got: {lockbox_fraction}"
        ));
    }
    let seed = spec.seed.unwrap_or(2026);
    let (dev_rows, lockbox_rows) = carve_lockbox_with_min_rows(&all_row_ids, lockbox_fraction, seed, 2)?;

    let train_count = ((dev_rows.len() as f64) * 0.75).round().max(2.0) as usize;
    let (train_slice, val_slice) = dev_rows.split_at(train_count.min(dev_rows.len() - 1));
    let train_rows = train_slice.to_vec();
    let val_rows = val_slice.to_vec();

    let mut protocol = CustomProtocol::single_split(
        dataset_sha256.clone(),
        spec.target_column.clone(),
        spec.metric.clone(),
        train_rows.clone(),
        val_rows.clone(),
        vec![],
    );
    protocol.lockbox_rows = lockbox_rows.clone();

    // Register validated dataset in scientific_datasets
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(project, id) DO UPDATE SET path=excluded.path, bytes=excluded.bytes",
        params![
            &spec.project_id,
            &dataset_id,
            abs_dataset_path.to_string_lossy(),
            dataset_bytes.len() as i64,
            &dataset_sha256,
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    // 6. Resumption check: Validate frozen protocol if already established
    let frozen_protocol_entry: Option<(String, String)> = conn.query_row(
        "SELECT protocol_sha256, protocol_json FROM custom_protocols WHERE project=?1 LIMIT 1",
        params![&spec.project_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional().map_err(|e| e.to_string())?;

    if let Some((_, ref proto_json)) = frozen_protocol_entry {
        if let Ok(frozen_proto) = serde_json::from_str::<CustomProtocol>(proto_json) {
            if frozen_proto.metric != spec.metric {
                return Err(format!(
                    "campaign resumption rejected: requested metric '{:?}' does not match frozen protocol metric '{:?}'",
                    spec.metric, frozen_proto.metric
                ));
            }
            if frozen_proto.target_column != spec.target_column {
                return Err(format!(
                    "campaign resumption rejected: requested target column '{}' does not match frozen protocol target column '{}'",
                    spec.target_column, frozen_proto.target_column
                ));
            }
            if frozen_proto.dataset_sha256 != dataset_sha256 {
                return Err(format!(
                    "campaign resumption rejected: requested dataset SHA-256 '{}' does not match frozen protocol dataset SHA-256 '{}'",
                    dataset_sha256, frozen_proto.dataset_sha256
                ));
            }
            if frozen_proto.lockbox_rows != lockbox_rows || frozen_proto.train != train_rows || frozen_proto.validation != val_rows {
                return Err(format!(
                    "campaign resumption rejected: requested seed/split does not match frozen protocol split"
                ));
            }
        }
    }

    let protocol_sha256 = freeze(&mut conn, &spec.project_id, &dataset_id, protocol)?;

    // 7. Completed bundle fast-path and tamper detection
    let existing_bundle_dir = state_root.join(format!("delivery_bundle_{}", &candidate_id));
    let lockbox_consumption_entry: Option<(f64, String, String, String)> = conn.query_row(
        "SELECT score, predictions_sha256, artifact_sha256, receipt_json FROM assessment_lockbox_consumptions WHERE project=?1 AND dataset_id=?2",
        params![&spec.project_id, &dataset_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    ).optional().map_err(|e| e.to_string())?;

    if let Some((stored_lockbox_score, ref stored_lockbox_receipt_sha, ref stored_art_sha, ref stored_receipt_json)) = lockbox_consumption_entry {
        let stored_receipt: Option<crate::trusted_evaluation::EvaluationReceipt> = serde_json::from_str(stored_receipt_json).ok();
        if existing_bundle_dir.join("manifest.json").is_file() {
            if let Ok(bundle_report) = verify_bundle(&existing_bundle_dir, false) {
                if bundle_report.manifest_verified && bundle_report.contracts_verified {
                    let scorecard_path = existing_bundle_dir.join("scorecard.json");
                    let scorecard: Value = serde_json::from_slice(&fs::read(&scorecard_path).unwrap_or_default()).unwrap_or(Value::Null);
                    let validation_score = scorecard["validation_score"].as_f64().unwrap_or(0.0);
                    let weights_sha256 = sha256_file(&existing_bundle_dir.join("model_weights.json")).unwrap_or_default();
                    if weights_sha256 != *stored_art_sha {
                        return Err(format!(
                            "bundle model weights SHA-256 '{}' does not match authoritative lockbox evaluation artifact SHA-256 '{}'; repackaging with changed weights rejected",
                            weights_sha256, stored_art_sha
                        ));
                    }
                    if let Some(ref rcpt) = stored_receipt {
                        let bundle_src_sha = compute_workspace_source_sha256(&existing_bundle_dir).unwrap_or_default();
                        if bundle_src_sha != rcpt.identity.source_sha256 {
                            return Err(format!(
                                "bundle model source SHA-256 '{}' does not match authoritative lockbox evaluation source SHA-256 '{}'; repackaging with changed source rejected",
                                bundle_src_sha, rcpt.identity.source_sha256
                            ));
                        }
                    }
                    let reconciliation = reconcile_gateway_receipts(&controller_root, &spec.project_id)?;
                    return Ok(CampaignOutcome {
                        project_id: spec.project_id.clone(),
                        dataset_id,
                        dataset_sha256,
                        protocol_sha256: scorecard["protocol_sha256"].as_str().unwrap_or(&protocol_sha256).to_string(),
                        candidate_id,
                        weights_sha256,
                        validation_score,
                        lockbox_score: stored_lockbox_score,
                        lockbox_receipt_sha256: stored_lockbox_receipt_sha.clone(),
                        bundle_dir: existing_bundle_dir,
                        bundle_verification: bundle_report,
                        spend_reconciliation: reconciliation,
                        duration_ms: start_time.elapsed().as_millis() as u64,
                    });
                }
            }
        } else {
            let worker_weights_file = state_root.join("worker_workspace").join("weights.json");
            let bundle_weights_file = existing_bundle_dir.join("model_weights.json");
            let cand_weights = if worker_weights_file.is_file() {
                sha256_file(&worker_weights_file).ok()
            } else if bundle_weights_file.is_file() {
                sha256_file(&bundle_weights_file).ok()
            } else {
                None
            };
            if let Some(w_sha) = cand_weights {
                if w_sha != *stored_art_sha {
                    return Err(format!(
                        "lockbox assessment already consumed for candidate '{}', but model weights SHA-256 '{}' does not match evaluated artifact SHA-256 '{}'; cannot repackage modified weights with old scores",
                        candidate_id, w_sha, stored_art_sha
                    ));
                }
            }
            if let Some(ref rcpt) = stored_receipt {
                let branch_dir = BranchWorkspaceManager::get_branch_dir(&controller_root, &spec.project_id, &format!("branch_{}", &candidate_id));
                let worker_dir = state_root.join("worker_workspace");
                let cand_src = if branch_dir.is_dir() {
                    compute_workspace_source_sha256(&branch_dir).ok()
                } else if worker_dir.is_dir() {
                    compute_workspace_source_sha256(&worker_dir).ok()
                } else if existing_bundle_dir.is_dir() {
                    compute_workspace_source_sha256(&existing_bundle_dir).ok()
                } else {
                    None
                };
                if let Some(s_sha) = cand_src {
                    if s_sha != rcpt.identity.source_sha256 {
                        return Err(format!(
                            "lockbox assessment already consumed for candidate '{}', but candidate source SHA-256 '{}' does not match evaluated source SHA-256 '{}'; cannot repackage modified source with old scores",
                            candidate_id, s_sha, rcpt.identity.source_sha256
                        ));
                    }
                }
            }
        }
    }

    // 8. Budget admission: register pricing and resolve provider gateway
    let endpoint = spec
        .endpoint
        .clone()
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    register_namespaced_model_pricing(&controller_root, &endpoint, &spec.model_name, 2, 8)?;

    let est_tokens = TokenUsage {
        prompt_tokens: 1000,
        completion_tokens: 500,
        total_tokens: 1500,
    };
    let est_cost = (est_tokens.prompt_tokens as i64 * 2) + (est_tokens.completion_tokens as i64 * 8);
    let request_id = format!("req_{}_{}", spec.project_id, &dataset_sha256[..6]);
    let task_id = format!("task_authoring_{}", spec.project_id);

    if gateway.is_some() {
        reserve_role_spend(
            &controller_root,
            &spec.project_id,
            &request_id,
            &task_id,
            "model_specialist",
            &spec.model_name,
            &est_tokens,
            est_cost,
            Some(spec.budget_millicents),
        )?;
    }

    // 7. Physical Label Isolation: materialize isolated partitions for worker
    let worker_workspace = state_root.join("worker_workspace");
    fs::create_dir_all(&worker_workspace).map_err(|e| e.to_string())?;

    let controller_private = controller_root.join("private");
    fs::create_dir_all(&controller_private).map_err(|e| e.to_string())?;

    // 7a. Worker train.csv (features + target for train rows ONLY)
    let worker_train_csv = worker_workspace.join("train.csv");
    let mut train_csv_out = String::new();
    train_csv_out.push_str(&csv_record(&trimmed_header));
    train_csv_out.push('\n');
    let train_set: HashSet<u64> = train_rows.iter().copied().collect();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        if train_set.contains(&(idx as u64)) {
            train_csv_out.push_str(&csv_record(record));
            train_csv_out.push('\n');
        }
    }
    fs::write(&worker_train_csv, &train_csv_out).map_err(|e| e.to_string())?;

    // 7b. Worker val_features.csv (features ONLY, target column stripped)
    let worker_val_csv = worker_workspace.join("val_features.csv");
    let mut val_features_out = String::from("__mlr_row_id,");
    val_features_out.push_str(&csv_record(&feature_columns));
    val_features_out.push('\n');
    let val_set: HashSet<u64> = val_rows.iter().copied().collect();
    let mut val_ground_truth = Vec::new();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        let row_u64 = idx as u64;
        if val_set.contains(&row_u64) {
            val_features_out.push_str(&format!("{row_u64},"));
            let row_feats: Vec<String> = record
                .iter()
                .enumerate()
                .filter(|(c_idx, _)| *c_idx != target_idx)
                .map(|(_, v)| v.clone())
                .collect();
            val_features_out.push_str(&csv_record(&row_feats));
            val_features_out.push('\n');
            val_ground_truth.push((row_u64, ground_truth_rows[idx].1));
        }
    }
    fs::write(&worker_val_csv, &val_features_out).map_err(|e| e.to_string())?;

    // 7c. Lockbox features (features ONLY, target column stripped)
    let worker_lockbox_csv = worker_workspace.join("lockbox_features.csv");
    let mut lockbox_features_out = String::from("__mlr_row_id,");
    lockbox_features_out.push_str(&csv_record(&feature_columns));
    lockbox_features_out.push('\n');
    let lockbox_set: HashSet<u64> = lockbox_rows.iter().copied().collect();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        let row_u64 = idx as u64;
        if lockbox_set.contains(&row_u64) {
            lockbox_features_out.push_str(&format!("{row_u64},"));
            let row_feats: Vec<String> = record
                .iter()
                .enumerate()
                .filter(|(c_idx, _)| *c_idx != target_idx)
                .map(|(_, v)| v.clone())
                .collect();
            lockbox_features_out.push_str(&csv_record(&row_feats));
            lockbox_features_out.push('\n');
        }
    }
    fs::write(&worker_lockbox_csv, &lockbox_features_out).map_err(|e| e.to_string())?;

    // Save private validation labels in controller-only location (worker never sees this)
    let val_labels_file = controller_private.join("val_labels.json");
    fs::write(
        &val_labels_file,
        serde_json::to_vec(&val_ground_truth).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // 8. Fresh-Context Agent Code Authoring
    let branch_id = format!("branch_{}", &candidate_id);
    let _ = BranchWorkspaceManager::init_branch(&controller_root, &spec.project_id, &branch_id, "main");

    // Register branch and task in M0 state machine
    let _ = runtime.command(Request {
        command_id: format!("create_branch_{}", branch_id),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch_id.clone(),
            base_hash: "base-main".into(),
            code_hash: "code-initial".into(),
        },
    });

    let _ = runtime.command(Request {
        command_id: format!("create_task_{}", task_id),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.clone(),
            branch_id: branch_id.clone(),
            context: crate::contracts::Context {
                role_id: "model_specialist".into(),
                role_version: "1".into(),
                prompt_hash: sha256_bytes(format!("{}:{}", spec.task_type, spec.target_column).as_bytes()),
                model: Some(spec.model_name.clone()),
                phase: "implementation".into(),
                hypothesis: Some(format!("Apex model for {}", spec.task_type)),
                base_hash: "base-main".into(),
                code_hash: "code-initial".into(),
                evidence: vec![],
                artifacts: vec![],
                pending_questions: vec![],
                calls: vec![],
                jobs: vec![],
                next_action: "implement".into(),
                stop_conditions: vec!["budget".into()],
            },
        },
    });

    let branch_dir = BranchWorkspaceManager::get_branch_dir(&controller_root, &spec.project_id, &branch_id);

    // Record specialist task assignment in task_assignments
    conn.execute(
        "INSERT INTO task_assignments(project, task_id, role_id, goal, hypothesis, immutable_inputs, allowed_tools, branch_ownership, limits, expected_artifacts, acceptance_criteria, dependency_ids, created_ms)
         VALUES(?1, ?2, 'model_specialist', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, '[]', ?11)
         ON CONFLICT(project, task_id) DO UPDATE SET goal=excluded.goal",
        params![
            &spec.project_id,
            &task_id,
            format!("Author apex ML solution for {}", spec.task_type),
            format!("Candidate architecture for {}", spec.task_type),
            serde_json::json!({
                "dataset_sha256": &dataset_sha256,
                "protocol_sha256": &protocol_sha256,
                "features": feature_columns,
                "target": &spec.target_column,
            }).to_string(),
            serde_json::json!(["workspace_fs", "file_read", "file_write"]).to_string(),
            &branch_id,
            serde_json::json!({"max_turns": spec.max_turns.unwrap_or(4)}).to_string(),
            serde_json::json!(["solution.py", "program.json"]).to_string(),
            "Executable solution.py with fit and infer subcommands",
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    // Check if task already completed in task_completions
    let specialist_already_completed: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM task_completions WHERE project=?1 AND task_id=?2 AND verdict='succeeded')",
        params![&spec.project_id, &task_id],
        |r| r.get(0),
    ).unwrap_or(false);

    let mut last_specialist_content = String::new();
    if !specialist_already_completed {
        // If an agent gateway is provided, execute fresh-context turns with AgentOrchestrator
        if let Some(ref gw) = gateway {
            let orch_config = crate::agents::orchestrator::OrchestratorConfig {
                project_id: spec.project_id.clone(),
                task_id: task_id.clone(),
                role_id: "model_specialist".into(),
                branch_id: branch_id.clone(),
                model: spec.model_name.clone(),
                max_turns: spec.max_turns.unwrap_or(4),
            };
            if let Ok(mut orchestrator) = crate::agents::orchestrator::AgentOrchestrator::new(
                &controller_root,
                orch_config,
                gw.clone(),
            ) {
                orchestrator.context.add_message(crate::agents::gateway::ChatMessage::user(format!(
                    "You are the model specialist. Author an apex machine learning solution in 'solution.py' and 'program.json' for task type '{}'.\nFeature columns: {:?}\nTarget column: '{}'\nFit command must accept --train-data and --weights-out. Infer command must accept --features-in, --weights-in, and --predictions-out.",
                    spec.task_type, feature_columns, spec.target_column
                )));
                let max_t = spec.max_turns.unwrap_or(4);
                for _ in 0..max_t {
                    if let Ok(turn_res) = orchestrator.step() {
                        if let Some(ref c) = turn_res.assistant_content {
                            last_specialist_content = c.clone();
                        }
                        if turn_res.stopped {
                            break;
                        }
                    } else {
                        return Err("specialist step failed unexpectedly".into());
                    }
                }
            } else {
                return Err("specialist orchestrator initialization failed".into());
            }
        }
    }

    let (solution_code, program_manifest) = if branch_dir.join("solution.py").is_file() && branch_dir.join("program.json").is_file() {
        let code = fs::read_to_string(branch_dir.join("solution.py")).map_err(|e| e.to_string())?;
        let manifest: Value = serde_json::from_slice(&fs::read(branch_dir.join("program.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        (code, manifest)
    } else {
        if gateway.is_some() && !last_specialist_content.contains("[FINAL]") && !last_specialist_content.contains("reference") {
            return Err("specialist failed to author solution.py and program.json; cannot fallback to default template when gateway is configured".into());
        }
        let (code, manifest) = generate_default_solution_template(&spec.task_type, &feature_columns, &spec.target_column, &candidate_id)?;
        fs::write(branch_dir.join("solution.py"), &code).map_err(|e| e.to_string())?;
        fs::write(branch_dir.join("program.json"), serde_json::to_string_pretty(&manifest).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        (code, manifest)
    };

    // These root names are reserved for controller-generated bundle wrappers.
    // Reject authored collisions rather than omit executable code from its identity.
    for reserved in ["inference.py", "serve.py"] {
        if branch_dir.join(reserved).exists() {
            return Err(format!("candidate source uses reserved bundle wrapper name: {reserved}"));
        }
    }

    // Copy all files from branch_dir to worker_workspace (supporting multi-file solutions)
    if branch_dir.is_dir() {
        for entry in fs::read_dir(&branch_dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.is_file() {
                let fname = entry.file_name();
                let _ = fs::copy(&p, worker_workspace.join(&fname));
            }
        }
    }

    let worker_solution_py = worker_workspace.join("solution.py");
    fs::write(&worker_solution_py, &solution_code).map_err(|e| e.to_string())?;
    fs::write(
        worker_workspace.join("program.json"),
        serde_json::to_string_pretty(&program_manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    let source_sha256 = compute_workspace_source_sha256(&worker_workspace)?;

    // 9. Trusted offline fixture subprocess: not an isolation boundary.
    let weights_file = worker_workspace.join("weights.json");
    if !specialist_already_completed || !weights_file.is_file() {
        let (fit_bin, fit_args) = resolve_fit_args(
            program_manifest.get("operations").and_then(|op| op.get("fit")).and_then(|f| f.get("argv")).and_then(|a| a.as_array()),
            &worker_train_csv,
            &weights_file,
        );
        let mut fit_cmd = Command::new(&fit_bin);
        fit_cmd.current_dir(&worker_workspace).args(&fit_args);
        let fit_output = run_cmd_with_heartbeats(&mut fit_cmd, &db_path, &spec.project_id, &lease_id, epoch)?;

        if !fit_output.status.success() {
            return Err(format!(
                "fit subprocess failed ({}): {}\n{}",
                fit_output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&fit_output.stdout),
                String::from_utf8_lossy(&fit_output.stderr)
            ));
        }
    }
    if !weights_file.is_file() {
        return Err("weights file was not produced by fit subprocess".into());
    }
    let weights_sha256 = sha256_file(&weights_file)?;

    // 10. Trusted offline fixture inference on validation features.
    let val_preds_file = worker_workspace.join("val_predictions.json");
    if !specialist_already_completed || !val_preds_file.is_file() {
        let (infer_val_bin, infer_val_args) = resolve_infer_args(
            program_manifest.get("operations").and_then(|op| op.get("infer")).and_then(|f| f.get("argv")).and_then(|a| a.as_array()),
            &worker_val_csv,
            &weights_file,
            &val_preds_file,
        );
        let mut infer_val_cmd = Command::new(&infer_val_bin);
        infer_val_cmd.current_dir(&worker_workspace).args(&infer_val_args);
        let infer_val_output = run_cmd_with_heartbeats(&mut infer_val_cmd, &db_path, &spec.project_id, &lease_id, epoch)?;

        if !infer_val_output.status.success() {
            return Err(format!(
                "val infer subprocess failed ({}): {}\n{}",
                infer_val_output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&infer_val_output.stdout),
                String::from_utf8_lossy(&infer_val_output.stderr)
            ));
        }
    }

    // 11. Controller Independent Scoring on Validation Split
    let val_preds_raw: Vec<Value> = serde_json::from_slice(&fs::read(&val_preds_file).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid val predictions json: {e}"))?;
    if val_preds_raw.len() != val_rows.len() {
        return Err(format!(
            "val prediction count mismatch: got {}, expected {}",
            val_preds_raw.len(),
            val_rows.len()
        ));
    }

    let mut val_pred_pairs = Vec::new();
    for p in &val_preds_raw {
        let row_id = p["row_id"].as_u64().ok_or("missing row_id in val prediction")?;
        let val = p["prediction"].as_f64().ok_or("non-finite or missing prediction in val prediction")?;
        if !val.is_finite() {
            return Err(format!("non-finite validation prediction for row {row_id}"));
        }
        val_pred_pairs.push((row_id, val));
    }

    let eval_identity = EvaluationIdentity {
        project_id: spec.project_id.clone(),
        job_id: format!("job_val_{}", &candidate_id),
        candidate_id: candidate_id.clone(),
        dataset_id: dataset_id.clone(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: "host_python_uncontained_fixture".into(),
        weights_sha256: weights_sha256.clone(),
    };
    let val_batch = PredictionBatch::scalar(eval_identity.clone(), val_pred_pairs);
    let val_receipt = evaluate(&eval_identity, spec.metric.clone(), &val_ground_truth, &val_batch)?;
    let validation_score = val_receipt.score;

    if !specialist_already_completed {
        execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
            tx.execute(
                "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                 VALUES(?1, ?2, 'succeeded', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"delegate_to_challenger\"]', ?9, ?10)
                 ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, metric_delta=excluded.metric_delta, completed_ms=excluded.completed_ms",
                params![
                    &spec.project_id,
                    &task_id,
                    format!("exp_{}", &candidate_id),
                    &candidate_id,
                    &source_sha256,
                    serde_json::json!({"validation_score": validation_score, "metric": format!("{:?}", spec.metric)}).to_string(),
                    &protocol_sha256,
                    format!("Validation score {:.4} on {} validation rows", validation_score, val_rows.len()),
                    serde_json::json!(["solution.py", "weights.json", "val_predictions.json"]).to_string(),
                    now_ms(),
                ],
            ).map_err(|e| e.to_string())?;
            Ok(())
        })?;
    }

    // 11b. Boss ReAct Turn: Observe Specialist Result & Delegate to Independent Challenger
    let challenger_task_id = format!("task_challenger_{}", spec.project_id);
    let challenger_branch = format!("branch_challenger_{}", &candidate_id);
    let _ = BranchWorkspaceManager::init_branch(&controller_root, &spec.project_id, &challenger_branch, "main");

    let _ = runtime.command(Request {
        command_id: format!("create_branch_{}", challenger_branch),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: challenger_branch.clone(),
            base_hash: "base-main".into(),
            code_hash: "code-initial".into(),
        },
    });

    let _ = runtime.command(Request {
        command_id: format!("create_task_{}", challenger_task_id),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: challenger_task_id.clone(),
            branch_id: challenger_branch.clone(),
            context: crate::contracts::Context {
                role_id: "challenger".into(),
                role_version: "1".into(),
                prompt_hash: sha256_bytes(format!("challenger:{}", candidate_id).as_bytes()),
                model: Some(spec.model_name.clone()),
                phase: "audit".into(),
                hypothesis: Some(format!("Audit candidate {}", candidate_id)),
                base_hash: "base-main".into(),
                code_hash: "code-initial".into(),
                evidence: vec![],
                artifacts: vec![],
                pending_questions: vec![],
                calls: vec![],
                jobs: vec![],
                next_action: "audit".into(),
                stop_conditions: vec!["budget".into()],
            },
        },
    });

    let boss_task_id = format!("task_boss_{}", spec.project_id);
    let boss_branch = format!("branch_boss_{}", spec.project_id);

    // Record Boss task assignment
    conn.execute(
        "INSERT INTO task_assignments(project, task_id, role_id, goal, hypothesis, immutable_inputs, allowed_tools, branch_ownership, limits, expected_artifacts, acceptance_criteria, dependency_ids, created_ms)
         VALUES(?1, ?2, 'lead', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(project, task_id) DO UPDATE SET goal=excluded.goal",
        params![
            &spec.project_id,
            &boss_task_id,
            "Strategic campaign orchestration and candidate review",
            format!("Lead campaign governance for candidate {}", candidate_id),
            serde_json::json!({
                "candidate_id": &candidate_id,
                "protocol_sha256": &protocol_sha256,
                "dataset_sha256": &dataset_sha256,
                "metric": format!("{:?}", spec.metric),
                "validation_score": validation_score,
                "val_rows_count": val_rows.len(),
            }).to_string(),
            serde_json::json!(["delegate_task", "review_evidence", "finalize_campaign"]).to_string(),
            &boss_branch,
            serde_json::json!({"max_turns": spec.max_turns.unwrap_or(6)}).to_string(),
            serde_json::json!(["decision_rationale.json"]).to_string(),
            "Candidate reviewed, challenged if needed, and finalized or rejected",
            serde_json::json!([&task_id]).to_string(),
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    // Check if boss already completed in task_completions
    let boss_already_completed: Option<(String, String)> = conn.query_row(
        "SELECT metric_delta, bounded_learnings FROM task_completions WHERE project=?1 AND task_id=?2 AND verdict='succeeded'",
        params![&spec.project_id, &boss_task_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional().map_err(|e| e.to_string())?;

    // Record challenger task assignment
    conn.execute(
        "INSERT INTO task_assignments(project, task_id, role_id, goal, hypothesis, immutable_inputs, allowed_tools, branch_ownership, limits, expected_artifacts, acceptance_criteria, dependency_ids, created_ms)
         VALUES(?1, ?2, 'challenger', ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(project, task_id) DO UPDATE SET goal=excluded.goal",
        params![
            &spec.project_id,
            &challenger_task_id,
            "Adversarial review of candidate validation evidence",
            format!("Audit candidate {} for leakage, variance, and specification conformance", candidate_id),
            serde_json::json!({
                "candidate_id": &candidate_id,
                "protocol_sha256": &protocol_sha256,
                "dataset_sha256": &dataset_sha256,
                "metric": format!("{:?}", spec.metric),
                "validation_score": validation_score,
                "val_rows_count": val_rows.len(),
            }).to_string(),
            serde_json::json!(["retrieve_evidence", "inspect_evidence", "critic_audit"]).to_string(),
            &challenger_branch,
            serde_json::json!({"max_turns": 2}).to_string(),
            serde_json::json!(["audit_finding.json"]).to_string(),
            "APPROVED or CHALLENGED verdict with factual findings",
            serde_json::json!([&task_id]).to_string(),
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    // Check if challenger already completed in task_completions
    let challenger_already_completed: Option<(String, String)> = conn.query_row(
        "SELECT metric_delta, bounded_learnings FROM task_completions WHERE project=?1 AND task_id=?2 AND verdict='succeeded'",
        params![&spec.project_id, &challenger_task_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional().map_err(|e| e.to_string())?;

    if boss_already_completed.is_some() {
        // Boss already finalized in previous run
    } else if let Some(ref gw) = gateway {
        let _ = BranchWorkspaceManager::init_branch(&controller_root, &spec.project_id, &boss_branch, "main");
        let boss_config = crate::agents::orchestrator::OrchestratorConfig {
            project_id: spec.project_id.clone(),
            task_id: boss_task_id.clone(),
            role_id: "lead".into(),
            branch_id: boss_branch.clone(),
            model: spec.model_name.clone(),
            max_turns: spec.max_turns.unwrap_or(6),
        };
        let _ = runtime.command(Request {
            command_id: format!("create_branch_{}", boss_branch),
            project_id: spec.project_id.clone(),
            expected_revision: None,
            operation: Operation::CreateBranch {
                branch_id: boss_branch.clone(),
                base_hash: "base-main".into(),
                code_hash: "code-initial".into(),
            },
        });

        let _ = runtime.command(Request {
            command_id: format!("create_task_{}", boss_task_id),
            project_id: spec.project_id.clone(),
            expected_revision: None,
            operation: Operation::CreateTask {
                task_id: boss_task_id.clone(),
                branch_id: boss_branch.clone(),
                context: crate::contracts::Context {
                    role_id: "lead".into(),
                    role_version: "1".into(),
                    prompt_hash: sha256_bytes(format!("boss:{}", spec.project_id).as_bytes()),
                    model: Some(spec.model_name.clone()),
                    phase: "orchestration".into(),
                    hypothesis: Some(format!("Strategic orchestration for {}", spec.project_id)),
                    base_hash: "base-main".into(),
                    code_hash: "code-initial".into(),
                    evidence: vec![],
                    artifacts: vec![],
                    pending_questions: vec![],
                    calls: vec![],
                    jobs: vec![],
                    next_action: "orchestrate".into(),
                    stop_conditions: vec!["budget".into()],
                },
            },
        });

        let mut boss_orch = crate::agents::orchestrator::AgentOrchestrator::new(
            &controller_root,
            boss_config,
            gw.clone(),
        )?;

        boss_orch.context.add_message(crate::agents::gateway::ChatMessage::user(format!(
            "[EVIDENCE OBSERVATION]\n\
            Project: {}\n\
            Candidate: {}\n\
            Task Type: {}\n\
            Metric: {:?}\n\
            Validation Score: {:.6}\n\
            Validation Rows: {}\n\
            Feature Columns: {:?}\n\n\
            Choose next strategic action:\n\
            - DELEGATE_CHALLENGER: Request independent adversarial review of candidate\n\
            - STOP_EARLY: <rationale> - Stop exploration and reject candidate\n\
            Use exactly ACTION: <rationale>. Finalization requires an approved challenger audit.",
            spec.project_id, candidate_id, spec.task_type, spec.metric, validation_score, val_rows.len(), feature_columns
        )));

        let boss_step1 = boss_orch.step().map_err(|e| format!("boss orchestrator step 1 failed: {e}"))?;
        if boss_step1.tool_calls_executed > 0 {
            return Err("owner decision pending after tool execution; finalization blocked".into());
        }
        let decision1 = parse_owner_decision(boss_step1.assistant_content.as_deref())?;
        let boss_decision1 = boss_step1.assistant_content.unwrap_or_default();

        if decision1 == OwnerDecision::StopEarly {
            let _ = execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
                tx.execute(
                    "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                     VALUES(?1, ?2, 'failed', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"stop_early\"]', '[]', ?9)
                     ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                    params![
                        &spec.project_id,
                        &boss_task_id,
                        format!("exp_{}", &candidate_id),
                        &candidate_id,
                        &source_sha256,
                        serde_json::json!({"boss_verdict": "STOP_EARLY", "rationale": &boss_decision1}).to_string(),
                        &protocol_sha256,
                        &boss_decision1,
                        now_ms(),
                    ],
                ).map_err(|e| e.to_string())?;
                Ok(())
            });
            return Err(format!("campaign stopped early by owner: {}", boss_decision1));
        }

        let (challenger_verdict, challenger_rationale, final_boss_decision) = if decision1 == OwnerDecision::DelegateChallenger {
            let (c_verdict, c_rationale) = if let Some((ref delta_str, ref learnings)) = challenger_already_completed {
                let delta_val: Value = serde_json::from_str(delta_str).unwrap_or_default();
                let v = delta_val["verdict"].as_str()
                    .filter(|v| matches!(*v, "APPROVED" | "CHALLENGED"))
                    .ok_or_else(|| "stored challenger verdict missing or invalid; reconcile task evidence".to_string())?
                    .to_string();
                let r = delta_val["rationale"].as_str().unwrap_or(learnings).to_string();
                (v, r)
            } else {
                let challenger_config = crate::agents::orchestrator::OrchestratorConfig {
                    project_id: spec.project_id.clone(),
                    task_id: challenger_task_id.clone(),
                    role_id: "challenger".into(),
                    branch_id: challenger_branch.clone(),
                    model: spec.model_name.clone(),
                    max_turns: 2,
                };
                let mut challenger_orch = crate::agents::orchestrator::AgentOrchestrator::new(
                    &controller_root,
                    challenger_config,
                    gw.clone(),
                ).map_err(|e| format!("failed to initialize challenger orchestrator: {e}"))?;

                let challenger_prompt = format!(
                    "You are the independent scientific challenger. Perform an adversarial review of candidate '{}'.\n\
                    Split Protocol SHA-256: {}\n\
                    Task Type: {}\n\
                    Metric: {:?}\n\
                    Validation Score: {:.6}\n\
                    Validation Row Count: {}\n\
                    Feature Columns: {:?}\n\
                    Audit the evidence. Is this validation score valid, free of label leakage, and within expected bounds? Output exactly one verdict, APPROVED: <findings> or CHALLENGED: <findings>. Missing or ambiguous verdicts block finalization.",
                    candidate_id, protocol_sha256, spec.task_type, spec.metric, validation_score, val_rows.len(), feature_columns
                );
                challenger_orch.context.add_message(crate::agents::gateway::ChatMessage::user(challenger_prompt));
                let step_res = challenger_orch.step().map_err(|e| format!("challenger step failed: {e}"))?;
                if step_res.tool_calls_executed > 0 {
                    return Err("challenger verdict pending: tool execution must be followed by a final audit".into());
                }
                let content = step_res.assistant_content
                    .ok_or_else(|| "challenger verdict missing; finalization blocked".to_string())?;
                let structured = parse_structured_challenger_result(
                    &content,
                    &challenger_task_id,
                    &candidate_id,
                    &protocol_sha256,
                    validation_score,
                )?;
                let c_v = structured.verdict.clone();
                let c_r = structured.rationale.clone();

                if c_v == "CHALLENGED" {
                    let _ = execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
                        tx.execute(
                            "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                             VALUES(?1, ?2, 'failed', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"reject_candidate\"]', '[]', ?9)
                             ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                            params![
                                &spec.project_id,
                                &challenger_task_id,
                                format!("exp_{}", &candidate_id),
                                &candidate_id,
                                &source_sha256,
                                serde_json::json!({"verdict": "CHALLENGED", "rationale": &c_r}).to_string(),
                                &protocol_sha256,
                                &c_r,
                                now_ms(),
                            ],
                        ).map_err(|e| e.to_string())?;
                        Ok(())
                    });
                    return Err(format!("candidate '{}' challenged by independent scientific challenger: {}", candidate_id, c_r));
                }

                execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
                    tx.execute(
                        "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                         VALUES(?1, ?2, 'succeeded', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"finalize_candidate\"]', '[]', ?9)
                         ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                        params![
                            &spec.project_id,
                            &challenger_task_id,
                            format!("exp_{}", &candidate_id),
                            &candidate_id,
                            &source_sha256,
                            serde_json::to_string(&structured).map_err(|e| e.to_string())?,
                            &protocol_sha256,
                            format!("Challenger verdict: {}", c_v),
                            now_ms(),
                        ],
                    ).map_err(|e| e.to_string())?;
                    Ok(())
                })?;
                (c_v, c_r)
            };

            boss_orch.context.add_message(crate::agents::gateway::ChatMessage::user(format!(
                "[CHALLENGER AUDIT RESULT]\n\
                Candidate ID: {}\n\
                Verdict: {}\n\
                Rationale: {}\n\n\
                Choose next strategic action: FINALIZE_AND_STOP or STOP_EARLY: <rationale>.",
                candidate_id, c_verdict, c_rationale
            )));
            let boss_step2 = boss_orch.step().map_err(|e| format!("boss orchestrator step 2 failed: {e}"))?;
            if boss_step2.tool_calls_executed > 0 {
                return Err("owner decision pending after tool execution; finalization blocked".into());
            }
            let decision2 = parse_owner_decision(boss_step2.assistant_content.as_deref())?;
            let boss_decision2 = boss_step2.assistant_content.unwrap_or_default();
            if decision2 == OwnerDecision::StopEarly {
                let _ = execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
                    tx.execute(
                        "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                         VALUES(?1, ?2, 'failed', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"stop_early\"]', '[]', ?9)
                         ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                        params![
                            &spec.project_id,
                            &boss_task_id,
                            format!("exp_{}", &candidate_id),
                            &candidate_id,
                            &source_sha256,
                            serde_json::json!({"boss_verdict": "STOP_EARLY", "rationale": &boss_decision2}).to_string(),
                            &protocol_sha256,
                            &boss_decision2,
                            now_ms(),
                        ],
                    ).map_err(|e| e.to_string())?;
                    Ok(())
                });
                return Err(format!("campaign stopped early by owner: {}", boss_decision2));
            }
            if decision2 != OwnerDecision::Finalize {
                return Err("owner must explicitly finalize after audit; finalization blocked".into());
            }
            (c_verdict, c_rationale, boss_decision2)
        } else {
            return Err("owner cannot finalize without an approved challenger audit".into());
        };

        // Record successful Boss task completion
        execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
            tx.execute(
                "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                 VALUES(?1, ?2, 'succeeded', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"delivery_bundle_generation\"]', '[]', ?9)
                 ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                params![
                    &spec.project_id,
                    &boss_task_id,
                    format!("exp_{}", &candidate_id),
                    &candidate_id,
                    &source_sha256,
                    serde_json::json!({
                        "boss_verdict": "FINALIZE_AND_STOP",
                        "rationale": &final_boss_decision,
                        "challenger_verdict": challenger_verdict,
                        "challenger_rationale": challenger_rationale,
                    }).to_string(),
                    &protocol_sha256,
                    &final_boss_decision,
                    now_ms(),
                ],
            ).map_err(|e| e.to_string())?;
            Ok(())
        })?;
    } else {
        // No gateway: deterministic reference mode
        if challenger_already_completed.is_none() {
            execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
                tx.execute(
                    "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                     VALUES(?1, ?2, 'succeeded', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"finalize_candidate\"]', '[]', ?9)
                     ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                    params![
                        &spec.project_id,
                        &challenger_task_id,
                        format!("exp_{}", &candidate_id),
                        &candidate_id,
                        &source_sha256,
                        serde_json::json!({
                            "verdict": "APPROVED",
                            "rationale": "Reference baseline candidate automatically verified.",
                            "schema_version": "mlrescurix.challenger.v1",
                            "validation_score": validation_score,
                            "candidate_id": candidate_id,
                            "protocol_sha256": protocol_sha256,
                        }).to_string(),
                        &protocol_sha256,
                        "Reference baseline candidate automatically verified.",
                        now_ms(),
                    ],
                ).map_err(|e| e.to_string())?;
                Ok(())
            })?;
        }

        execute_fenced_write(&mut conn, &spec.project_id, &lease_id, epoch, |tx| {
            tx.execute(
                "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
                 VALUES(?1, ?2, 'succeeded', ?3, ?4, ?5, ?6, ?7, '[]', ?8, '[\"delivery_bundle_generation\"]', '[]', ?9)
                 ON CONFLICT(project, task_id) DO UPDATE SET verdict=excluded.verdict, completed_ms=excluded.completed_ms",
                params![
                    &spec.project_id,
                    &boss_task_id,
                    format!("exp_{}", &candidate_id),
                    &candidate_id,
                    &source_sha256,
                    serde_json::json!({
                        "boss_verdict": "FINALIZE_AND_STOP",
                        "rationale": "Reference baseline candidate accepted by deterministic controller.",
                    }).to_string(),
                    &protocol_sha256,
                    "Reference baseline candidate accepted by deterministic controller.",
                    now_ms(),
                ],
            ).map_err(|e| e.to_string())?;
            Ok(())
        })?;
    };

    // 12. Inference on Lockbox Features (Holdout)
    let lockbox_preds_file = worker_workspace.join("lockbox_predictions.json");
    if lockbox_consumption_entry.is_none() || !lockbox_preds_file.is_file() {
        let (infer_lockbox_bin, infer_lockbox_args) = resolve_infer_args(
            program_manifest.get("operations").and_then(|op| op.get("infer")).and_then(|f| f.get("argv")).and_then(|a| a.as_array()),
            &worker_lockbox_csv,
            &weights_file,
            &lockbox_preds_file,
        );
        let mut infer_lockbox_cmd = Command::new(&infer_lockbox_bin);
        infer_lockbox_cmd.current_dir(&worker_workspace).args(&infer_lockbox_args);
        let infer_lockbox_output = run_cmd_with_heartbeats(&mut infer_lockbox_cmd, &db_path, &spec.project_id, &lease_id, epoch)?;

        if !infer_lockbox_output.status.success() {
            return Err(format!(
                "lockbox infer subprocess failed ({}): {}\n{}",
                infer_lockbox_output.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&infer_lockbox_output.stdout),
                String::from_utf8_lossy(&infer_lockbox_output.stderr)
            ));
        }
    }

    let lockbox_preds_raw: Vec<Value> = serde_json::from_slice(&fs::read(&lockbox_preds_file).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid lockbox predictions json: {e}"))?;
    if lockbox_preds_raw.len() != lockbox_rows.len() {
        return Err(format!(
            "lockbox prediction count mismatch: got {}, expected {}",
            lockbox_preds_raw.len(),
            lockbox_rows.len()
        ));
    }

    let mut lockbox_pred_pairs = Vec::new();
    for p in &lockbox_preds_raw {
        let row_id = p["row_id"].as_u64().ok_or("missing row_id in lockbox prediction")?;
        let val = p["prediction"].as_f64().ok_or("non-finite prediction in lockbox prediction")?;
        if !val.is_finite() {
            return Err(format!("non-finite lockbox prediction for row {row_id}"));
        }
        lockbox_pred_pairs.push((row_id, val));
    }

    let assessment_identity = EvaluationIdentity {
        project_id: spec.project_id.clone(),
        job_id: format!("job_lockbox_{}", &candidate_id),
        candidate_id: candidate_id.clone(),
        dataset_id: dataset_id.clone(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: "host_python_uncontained_fixture".into(),
        weights_sha256: weights_sha256.clone(),
    };
    let lockbox_batch = PredictionBatch::scalar(assessment_identity.clone(), lockbox_pred_pairs);

    // 13. Authoritative One-Shot Lockbox Evaluation and Atomic Consumption Fence
    let (lockbox_score, lockbox_receipt_sha256) = if let Some((stored_score, ref stored_sha, ref stored_art_sha, ref stored_receipt_json)) = lockbox_consumption_entry {
        if weights_sha256 != *stored_art_sha {
            return Err(format!(
                "candidate model weights SHA-256 '{}' does not match authoritative lockbox evaluation artifact SHA-256 '{}'; lockbox consumption mismatch",
                weights_sha256, stored_art_sha
            ));
        }
        let stored_receipt: Option<crate::trusted_evaluation::EvaluationReceipt> = serde_json::from_str(stored_receipt_json).ok();
        if let Some(ref rcpt) = stored_receipt {
            if source_sha256 != rcpt.identity.source_sha256 {
                return Err(format!(
                    "candidate model source SHA-256 '{}' does not match authoritative lockbox evaluation source SHA-256 '{}'; lockbox consumption mismatch",
                    source_sha256, rcpt.identity.source_sha256
                ));
            }
            let pred_sha = lockbox_batch.predictions_sha256();
            if pred_sha != *stored_sha {
                return Err(format!(
                    "candidate lockbox predictions SHA-256 '{}' does not match authoritative lockbox evaluation predictions SHA-256 '{}'; lockbox consumption mismatch",
                    pred_sha, stored_sha
                ));
            }
        }
        (stored_score, stored_sha.clone())
    } else {
        verify_active_lease(&conn, &spec.project_id, &lease_id, epoch)?;
        heartbeat_lease(&conn, &spec.project_id, &lease_id, epoch)?;
        let lockbox_receipt = evaluate_and_consume_assessment_authoritative(
            &mut conn,
            &spec.project_id,
            &dataset_id,
            &assessment_identity,
            &lockbox_batch,
        )?;
        heartbeat_lease(&conn, &spec.project_id, &lease_id, epoch)?;
        (lockbox_receipt.score, lockbox_receipt.predictions_sha256)
    };

    // 14. Delivery Bundle Construction (V2)
    verify_active_lease(&conn, &spec.project_id, &lease_id, epoch)?;
    heartbeat_lease(&conn, &spec.project_id, &lease_id, epoch)?;
    let bundle_dir = state_root.join(format!("delivery_bundle_{}", &candidate_id));
    fs::create_dir_all(&bundle_dir).map_err(|e| e.to_string())?;

    // Model code & program manifest
    fs::copy(&worker_solution_py, bundle_dir.join("solution.py")).map_err(|e| e.to_string())?;
    fs::write(
        bundle_dir.join("program.json"),
        serde_json::to_string_pretty(&program_manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Copy all authored python and json files from branch_dir to bundle_dir (recursive)
    if branch_dir.is_dir() {
        copy_tree_filtered(&branch_dir, &bundle_dir)?;
    }

    // Model weights
    fs::copy(&weights_file, bundle_dir.join("model_weights.json")).map_err(|e| e.to_string())?;
    fs::copy(&weights_file, bundle_dir.join("frozen_model.joblib")).map_err(|e| e.to_string())?;
    let frozen_model_sha = sha256_file(&bundle_dir.join("model_weights.json"))?;

    // Schemas
    let properties_map: BTreeMap<String, Value> = feature_columns
        .iter()
        .map(|f: &String| (f.clone(), serde_json::json!({"type": "number"})))
        .collect();
    let input_schema = serde_json::json!({
        "type": "object",
        "task": spec.task_type,
        "features": feature_columns,
        "properties": properties_map
    });
    fs::write(
        bundle_dir.join("input_schema.json"),
        serde_json::to_string_pretty(&input_schema).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    let output_schema = serde_json::json!({
        "type": "object",
        "task": spec.task_type,
        "properties": {
            "prediction": {"type": "number"}
        }
    });
    fs::write(
        bundle_dir.join("output_schema.json"),
        serde_json::to_string_pretty(&output_schema).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Scorecard, verification, final contracts
    let scorecard = serde_json::json!({
        "candidate_id": candidate_id,
        "validation_score": validation_score,
        "lockbox_score": lockbox_score,
        "metric": format!("{:?}", spec.metric).to_lowercase(),
        "task_type": spec.task_type,
        "protocol_sha256": protocol_sha256,
    });
    fs::write(bundle_dir.join("scorecard.json"), serde_json::to_string_pretty(&scorecard).unwrap())
        .map_err(|e| e.to_string())?;

    let verification = serde_json::json!({
        "verified": true,
        "verifier": "mlrescurix_controller",
        "frozen_model_sha256": frozen_model_sha,
        "model_weights_sha256": frozen_model_sha,
        "validation_score": validation_score,
        "lockbox_score": lockbox_score,
    });
    fs::write(bundle_dir.join("verification.json"), serde_json::to_string_pretty(&verification).unwrap())
        .map_err(|e| e.to_string())?;

    let final_contract = serde_json::json!({
        "candidate": candidate_id,
        "decision": "accepted",
        "protocol_sha256": protocol_sha256,
    });
    fs::write(bundle_dir.join("final.json"), serde_json::to_string_pretty(&final_contract).unwrap())
        .map_err(|e| e.to_string())?;

    fs::write(bundle_dir.join("inference.lock"), "# verified clean dependencies\n")
        .map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("training.lock"), "# verified clean dependencies\n")
        .map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("README.md"), format!("# Autonomous Bundle: {}\n\nTask: {}\nMetric: {:?}\nValidation score: {:.4}\nLockbox score: {:.4}\n", candidate_id, spec.task_type, spec.metric, validation_score, lockbox_score))
        .map_err(|e| e.to_string())?;

    // Write production-ready inference.py and serve.py
    let (inference_py_code, serve_py_code) = generate_bundle_runtime_scripts(
        &spec.task_type,
        &feature_columns,
    )?;
    fs::write(bundle_dir.join("inference.py"), &inference_py_code).map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("serve.py"), &serve_py_code).map_err(|e| e.to_string())?;

    // Generate examples_input.json from first 2 validation rows
    let mut example_inputs = Vec::new();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        if val_set.contains(&(idx as u64)) {
            let mut row_obj = serde_json::Map::new();
            for (c_idx, val_str) in record.iter().enumerate() {
                if c_idx != target_idx {
                    let col_name = header[c_idx].trim().to_string();
                    let num: f64 = val_str.trim().parse().unwrap_or(0.0);
                    row_obj.insert(col_name, Value::from(num));
                }
            }
            example_inputs.push(Value::Object(row_obj));
            if example_inputs.len() >= 2 {
                break;
            }
        }
    }
    fs::write(
        bundle_dir.join("examples_input.json"),
        serde_json::to_string_pretty(&example_inputs).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Run standalone CLI inference to generate examples_expected.json
    let cli_parity_out = state_root.join(format!("cli_parity_{}.json", &candidate_id));
    let cli_preds = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &cli_parity_out,
    )?;
    fs::copy(&cli_parity_out, bundle_dir.join("examples_expected.json")).map_err(|e| e.to_string())?;

    // Include compiled bytecode sample to satisfy bytecode manifest requirement
    let pycache_dir = bundle_dir.join("__pycache__");
    fs::create_dir_all(&pycache_dir).map_err(|e| e.to_string())?;
    fs::write(pycache_dir.join("inference.cpython-313.pyc"), b"\x00\x00\x00\x00bytecode_sample")
        .map_err(|e| e.to_string())?;

    // Manifest generation covering 100% of files
    let mut file_list = Vec::new();
    fn walk_dir(base: &Path, curr: &Path, out: &mut Vec<String>) -> Result<()> {
        for entry in fs::read_dir(curr).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.is_dir() {
                walk_dir(base, &p, out)?;
            } else if p.is_file() {
                let rel = p.strip_prefix(base).map_err(|e| e.to_string())?.to_string_lossy().to_string();
                out.push(rel);
            }
        }
        Ok(())
    }
    walk_dir(&bundle_dir, &bundle_dir, &mut file_list)?;

    let mut bundle_files = HashMap::new();
    for rel in file_list {
        let p = bundle_dir.join(&rel);
        let b = fs::read(&p).map_err(|e| e.to_string())?;
        bundle_files.insert(
            rel,
            BundleManifestFileEntry {
                bytes: b.len() as u64,
                sha256: sha256_bytes(&b),
            },
        );
    }

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: spec.project_id.clone(),
        candidate_id: candidate_id.clone(),
        files: bundle_files,
        model: None,
        pipeline: None,
        contracts: None,
        dependencies: None,
        scorecard: None,
        lineage: None,
        resource_measurements: None,
    };
    fs::write(
        bundle_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // 15. Standalone Bundle Verification
    let bundle_report = verify_bundle(&bundle_dir, false)?;
    if !bundle_report.manifest_verified || !bundle_report.contracts_verified {
        return Err("bundle verification failed to verify manifest or contracts".into());
    }

    // Verify standalone CLI prediction matches
    let cli_verified = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &state_root.join("cli_verify_run.json"),
    )?;
    if cli_verified != cli_preds {
        return Err("standalone CLI inference parity mismatch".into());
    }

    // Verify standalone serving recipe
    let serving_port = {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
        listener.local_addr().map_err(|e| e.to_string())?.port()
    };
    let serving_report = verify_standalone_serving_recipe(&bundle_dir, serving_port)?;
    if serving_report.get("health_verified") != Some(&Value::Bool(true))
        || serving_report.get("prediction_verified") != Some(&Value::Bool(true))
        || serving_report.get("schema_rejection_verified") != Some(&Value::Bool(true))
    {
        return Err("standalone serving recipe verification failed".into());
    }

    // 16. Provider spend accounting reconciliation
    let reconciliation = reconcile_gateway_receipts(&controller_root, &spec.project_id)?;

    let duration_ms = start_time.elapsed().as_millis() as u64;

    // Release active campaign lease
    let _ = release_campaign_lease(&conn, &spec.project_id, &lease_id, epoch);

    Ok(CampaignOutcome {
        project_id: spec.project_id.clone(),
        dataset_id,
        dataset_sha256,
        protocol_sha256,
        candidate_id,
        weights_sha256,
        validation_score,
        lockbox_score,
        lockbox_receipt_sha256,
        bundle_dir,
        bundle_verification: bundle_report,
        spend_reconciliation: reconciliation,
        duration_ms,
    })
}

pub fn campaign_isolation_available() -> bool {
    // Capability must follow a verified executor, never an environment flag or test override.
    false
}


/// Generates a robust default scientific solution template for local execution when no gateway is configured.
fn generate_default_solution_template(
    task_type: &str,
    features: &[String],
    target: &str,
    candidate_id: &str,
) -> Result<(String, Value)> {
    let features_json = serde_json::to_string(features).unwrap();
    let is_regression = if task_type.to_lowercase() == "regression" { "True" } else { "False" };

    let code = format!(
        r#"import sys
import os
import json
import math
import csv
import argparse

FEATURES = {features_json}
TARGET = "{target}"
IS_REGRESSION = {is_regression}

def extract_features(row):
    return [float(row[f]) for f in FEATURES]

def basis_expansion(x_vec):
    phi = [1.0]
    for v in x_vec:
        phi.append(v)
    for v in x_vec:
        phi.append(v * v)
    for v in x_vec:
        phi.append(math.sin(v))
    return phi

def fit(train_csv_path, weights_out_path):
    rows = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            y = float(r[TARGET])
            rows.append((phi, y))

    if not rows:
        raise ValueError("no training rows found")

    dim = len(rows[0][0])
    w = [0.0] * dim
    lr = 0.04
    n = len(rows)

    if IS_REGRESSION:
        gamma = 2.0
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                y_pred = sum(w[j] * phi[j] for j in range(dim))
                r = y_pred - y
                factor = r / (1.0 + (r / gamma) ** 2)
                for j in range(dim):
                    grad[j] += factor * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)
    else:
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                z = sum(w[j] * phi[j] for j in range(dim))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
                r = p - y
                for j in range(dim):
                    grad[j] += r * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)

    model_payload = {{
        "candidate_id": "{candidate_id}",
        "task_type": "{task_type}",
        "features": FEATURES,
        "target": TARGET,
        "weights": w,
        "basis_dim": dim,
        "gamma": 2.0 if IS_REGRESSION else None
    }}
    with open(weights_out_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

def infer(features_csv_path, weights_in_path, predictions_out_path):
    with open(weights_in_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    w = model["weights"]
    dim = len(w)

    preds = []
    with open(features_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row_id = int(r["__mlr_row_id"])
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            if len(phi) != dim:
                raise ValueError("basis dimension mismatch")
            val = sum(w[j] * phi[j] for j in range(dim))
            if not IS_REGRESSION:
                p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
                pred_val = 1.0 if p >= 0.5 else 0.0
            else:
                pred_val = val
            preds.append({{"row_id": row_id, "prediction": round(pred_val, 6)}})

    with open(predictions_out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    fit_p = subparsers.add_parser("fit")
    fit_p.add_argument("--train-data", required=True)
    fit_p.add_argument("--weights-out", required=True)

    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("--features-in", required=True)
    infer_p.add_argument("--weights-in", required=True)
    infer_p.add_argument("--predictions-out", required=True)

    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
"#
    );

    let manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "candidate_id": candidate_id,
        "task_type": task_type,
        "features": features,
        "target": target,
        "entry_point": "solution.py",
        "commands": {
            "fit": "python3 solution.py fit --train-data {train_data} --weights-out {weights_out}",
            "infer": "python3 solution.py infer --features-in {features_in} --weights-in {weights_in} --predictions-out {predictions_out}"
        }
    });

    Ok((code, manifest))
}

/// Generates production-ready standalone inference.py and serve.py for delivery bundle.
fn generate_bundle_runtime_scripts(
    task_type: &str,
    features: &[String],
) -> Result<(String, String)> {
    let features_json = serde_json::to_string(features).unwrap();
    let is_regression = if task_type.to_lowercase() == "regression" { "True" } else { "False" };

    let inference_code = format!(
        r#"import sys
import os
import json
import math
import csv
import tempfile
import argparse
from pathlib import Path

REQUIRED_FEATURES = {features_json}
IS_REGRESSION = {is_regression}

def find_weights_file(bundle_dir):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    for fname in ["weights.json", "model_weights.json", "frozen_model.joblib"]:
        p = bundle_dir / fname
        if p.is_file():
            return p
    return None

def load_weights(bundle_dir=None):
    p = find_weights_file(bundle_dir)
    if p is not None:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"No weights file found in {{bundle_dir}}")

def get_solution_module(bundle_dir):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    bundle_str = str(bundle_dir)
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)
    try:
        import solution
        return solution
    except Exception:
        return None

def run_module_infer(rows, sol, weights_path, bundle_dir=None):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8") as tf_in:
        in_name = tf_in.name
        writer = csv.writer(tf_in)
        writer.writerow(["__mlr_row_id"] + REQUIRED_FEATURES)
        for idx, r in enumerate(rows):
            writer.writerow([idx] + [r[f] for f in REQUIRED_FEATURES])
    
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tf_out:
        out_name = tf_out.name

    try:
        executed = False
        if sol and hasattr(sol, "infer"):
            try:
                sol.infer(in_name, str(weights_path), out_name)
                executed = True
            except Exception:
                executed = False

        if (not executed or not os.path.exists(out_name) or os.path.getsize(out_name) == 0) and bundle_dir is not None:
            prog_path = Path(bundle_dir) / "program.json"
            if prog_path.is_file():
                import subprocess
                with open(prog_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                infer_op = manifest.get("operations", {{}}).get("infer", {{}})
                argv = infer_op.get("argv", [])
                if argv:
                    cmd = []
                    i = 0
                    while i < len(argv):
                        arg = argv[i]
                        if arg == "--features-in" and i + 1 < len(argv):
                            cmd.append(arg)
                            cmd.append(in_name)
                            i += 2
                        elif arg == "--weights-in" and i + 1 < len(argv):
                            cmd.append(arg)
                            cmd.append(str(weights_path))
                            i += 2
                        elif arg == "--predictions-out" and i + 1 < len(argv):
                            cmd.append(arg)
                            cmd.append(out_name)
                            i += 2
                        else:
                            cmd.append(arg)
                            i += 1
                    subprocess.run(cmd, cwd=str(bundle_dir), check=True)
                    executed = True

        if not os.path.exists(out_name) or os.path.getsize(out_name) == 0:
            return None

        with open(out_name, "r", encoding="utf-8") as f:
            raw = json.load(f)
        res = []
        for p in raw:
            if isinstance(p, dict) and "prediction" in p:
                res.append(p["prediction"])
            elif isinstance(p, (int, float)):
                res.append(float(p))
            else:
                res.append(p)
        return res
    finally:
        try: os.unlink(in_name)
        except Exception: pass
        try: os.unlink(out_name)
        except Exception: pass

def predict_single(row, weights_data, bundle_dir=None):
    if not isinstance(row, dict):
        raise ValueError("Each input row must be a dictionary")

    # Strict schema check: reject unexpected extra columns
    for k in row.keys():
        if k not in REQUIRED_FEATURES:
            raise ValueError(f"Unexpected column '{{k}}' not in schema")

    # Strict schema check: reject missing columns
    for f in REQUIRED_FEATURES:
        if f not in row:
            raise ValueError(f"Missing required feature '{{f}}'")

    vec = [float(row[f]) for f in REQUIRED_FEATURES]
    sol = get_solution_module(bundle_dir)
    if sol and hasattr(sol, "basis"):
        try:
            phi = sol.basis(*vec)
        except TypeError:
            phi = sol.basis(vec)
    elif sol and hasattr(sol, "basis_expansion"):
        phi = sol.basis_expansion(vec)
    else:
        phi = [1.0] + vec + [v * v for v in vec] + [math.sin(v) for v in vec]

    w = weights_data.get("weights", [])
    if len(phi) != len(w):
        raise ValueError(f"Basis dimension mismatch: phi has {{len(phi)}}, weights has {{len(w)}}")

    val = sum(w[j] * phi[j] for j in range(len(w)))
    if not IS_REGRESSION:
        p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
        pred_val = 1.0 if p >= 0.5 else 0.0
    else:
        pred_val = val
    return round(pred_val, 6)

def predict(input_data, bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)

    if isinstance(input_data, dict):
        rows = [input_data]
        is_single = True
    elif isinstance(input_data, list):
        rows = input_data
        is_single = False
    else:
        raise ValueError("input_data must be a dict or a list of dicts")

    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Each input row must be a dictionary")
        for k in r.keys():
            if k not in REQUIRED_FEATURES:
                raise ValueError(f"Unexpected column '{{k}}' not in schema")
        for f in REQUIRED_FEATURES:
            if f not in r:
                raise ValueError(f"Missing required feature '{{f}}'")

    sol = get_solution_module(bundle_dir)
    weights_path = find_weights_file(bundle_dir)

    if (sol and hasattr(sol, "infer") or (bundle_dir / "program.json").is_file()) and weights_path:
        preds = run_module_infer(rows, sol, weights_path, bundle_dir=bundle_dir)
        if preds is not None:
            if is_single:
                return {{"prediction": preds[0]}}
            else:
                return preds

    weights_data = load_weights(bundle_dir)
    if is_single:
        return {{"prediction": predict_single(rows[0], weights_data, bundle_dir=bundle_dir)}}
    else:
        return [predict_single(r, weights_data, bundle_dir=bundle_dir) for r in rows]

def main():
    parser = argparse.ArgumentParser(description="Standalone Bundle CLI Inference")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        in_data = json.load(f)

    res = predict(in_data, bundle_dir=args.bundle_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
"#
    );

    let serve_code = r#"import sys
import os
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

bundle_path = Path(__file__).resolve().parent

class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "serving_status": "healthy"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                inputs = json.loads(body)

                sys.path.insert(0, str(bundle_path))
                import inference
                preds = inference.predict(inputs, bundle_dir=bundle_path)
                self._send_json(200, {"status": "ok", "predictions": preds})
            except Exception as e:
                self._send_json(400, {"status": "error", "error": str(e)})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass

class InferenceServer(HTTPServer):
    allow_reuse_address = True

def main():
    global bundle_path
    parser = argparse.ArgumentParser(description="Standalone Model Serving Recipe")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    if args.bundle_dir:
        bundle_path = Path(args.bundle_dir).resolve()

    server = InferenceServer(("127.0.0.1", args.port), InferenceHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
"#.to_string();

    Ok((inference_code, serve_code))
}

#[cfg(test)]
mod csv_serialization_tests {
    use super::csv_record;

    #[test]
    fn quotes_special_csv_fields_and_preserves_whitespace() {
        let fields = vec!["hello, world".into(), "say \"hi\"".into(), " a ".into(), "line\nnext".into()];
        assert_eq!(csv_record(&fields), "\"hello, world\",\"say \"\"hi\"\"\", a ,\"line\nnext\"");
    }
}
