//! Durable tool intents precede effects; outcomes precede context checkpoints.
//! A stored outcome may be delivered again, but an unknown effect is never replayed.
use crate::{Result, core::Action, execution::{JobSpec, Probe}};
use super::workspace::BranchWorkspaceManager as Workspace;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::{self, File, OpenOptions}, io::Write, os::fd::AsRawFd,
    os::unix::fs::OpenOptionsExt, path::{Path, PathBuf}, time::Duration};

fn hash(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn sql<T>(r: rusqlite::Result<T>) -> Result<T> { r.map_err(|e| e.to_string()) }

/// Process-scoped serialization; kernel releases the lock on process death.
/// The file remains in state to avoid creating a second lock inode during cleanup.
pub struct BranchLock(File);
impl BranchLock {
    pub fn acquire(root: &Path, project: &str, branch: &str) -> Result<Self> {
        let identity = hash(format!("{project}/{branch}").as_bytes());
        let file = OpenOptions::new().read(true).write(true).create(true)
            .custom_flags(libc::O_NOFOLLOW).open(root.join(format!("tool-{identity}.lock")))
            .map_err(|e| e.to_string())?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err("tool branch executor already active".into());
        }
        Ok(Self(file))
    }
}
impl Drop for BranchLock {
    fn drop(&mut self) { unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN); } }
}

pub struct ToolIdentity<'a> {
    pub project: &'a str, pub task: &'a str, pub branch: &'a str,
    pub turn: u32, pub call_id: &'a str, pub name: &'a str, pub arguments: &'a str,
}

fn file_hash(path: &Path) -> Result<Option<String>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(hash(&bytes))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

pub fn require_settlement(db: &Connection, project: &str, task: &str, action: &str) -> Result<()> {
    let data: String = sql(db.query_row("SELECT data FROM actions WHERE project=?1 AND id=?2",
        params![project,action], |r| r.get(0)))?;
    let a: Action = serde_json::from_str(&data).map_err(|e| e.to_string())?;
    let receipt = a.receipt.as_ref().ok_or("tool execution requires successful settlement receipt")?;
    if a.action_id != action || a.task_id != task || a.status != "succeeded"
        || receipt["action_id"] != action || receipt["status"] != "succeeded" {
        return Err("tool execution requires successful authoritative action settlement".into());
    }
    let held: i64 = sql(db.query_row("SELECT count(*) FROM reservations WHERE project=?1 AND action=?2 AND held=1",
        params![project,action], |r| r.get(0)))?;
    if held != 0 { return Err("tool execution requires released action reservation".into()); }
    Ok(())
}

pub fn canonicalize_value(v: &Value) -> Value {
    match v {
        Value::Object(map) => {
            let mut sorted = std::collections::BTreeMap::new();
            for (k, val) in map {
                sorted.insert(k.clone(), canonicalize_value(val));
            }
            let mut out = serde_json::Map::new();
            for (k, val) in sorted {
                out.insert(k, val);
            }
            Value::Object(out)
        }
        Value::Array(arr) => {
            Value::Array(arr.iter().map(canonicalize_value).collect())
        }
        other => other.clone(),
    }
}

fn build_scheduler_submission_intent(
    tx: &rusqlite::Transaction,
    id: &ToolIdentity<'_>,
    args: &Value,
    action: &str,
) -> Result<Value> {
    let job_id = args["job_id"].as_str().ok_or("job_id missing")?;
    let dataset_id = args["dataset_id"].as_str().ok_or("dataset_id missing")?;
    let environment_id = args["environment_id"].as_str().ok_or("environment_id missing")?;
    let protocol = args.get("protocol").ok_or("protocol missing")?;

    let canonical_args = canonicalize_value(args);
    let canonical_args_str = serde_json::to_string(&canonical_args).map_err(|e| e.to_string())?;
    let canonical_args_sha256 = hash(canonical_args_str.as_bytes());

    let task_data: String = sql(tx.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ))?;
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;
    if task.generation == 0 || task.status != "active" {
        return Err("stale or inactive task fence for scheduler job submission".into());
    }

    let (project_state, project_intent_revision, project_policy): (String, u64, String) = sql(tx.query_row(
        "SELECT state, intent_revision, policy FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ))?;
    if project_state != "created" && project_state != "finalizing" {
        return Err(format!("project does not admit work in state {project_state}"));
    }
    let policy_sha256 = hash(project_policy.as_bytes());

    let dataset_row: Option<(String, i64, String)> = sql(tx.query_row(
        "SELECT path, bytes, sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",
        params![id.project, dataset_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional())?;
    let (_, _, dataset_sha256) = dataset_row.ok_or_else(|| {
        format!("scientific job requires a registered project dataset: {dataset_id}")
    })?;

    let env_row: Option<(String, String)> = sql(tx.query_row(
        "SELECT identity, status FROM environments WHERE id=?1",
        params![environment_id],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).optional())?;
    let (env_identity, env_status) = env_row.ok_or_else(|| {
        format!("requested environment is not registered: {environment_id}")
    })?;
    if env_status != "ready" {
        return Err(format!("requested environment '{environment_id}' is not ready (status: {env_status})"));
    }

    let protocol_str = serde_json::to_string(protocol).map_err(|e| e.to_string())?;
    let protocol_sha256 = hash(protocol_str.as_bytes());

    let planned_spec = JobSpec { resources: Default::default(),
        job_id: job_id.to_string(),
        project_id: id.project.to_string(),
        task_id: id.task.to_string(),
        generation: task.generation,
        timeout_ms: 45_000,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: Some(environment_id.to_string()),
        probe: Probe::Scientific {
            operation: "pipeline".into(),
            dataset_id: dataset_id.to_string(),
            spec: Some(protocol.clone()),
            candidate_id: None,
            test_access_id: None,
        },
    };
    let planned_spec_json = serde_json::to_string(&planned_spec).map_err(|e| e.to_string())?;
    let planned_spec_sha256 = hash(planned_spec_json.as_bytes());

    Ok(json!({
        "kind": "scheduler_job_submission",
        "operation": "schedule_pipeline",
        "project": id.project,
        "task": id.task,
        "turn": id.turn,
        "call_id": id.call_id,
        "action": action,
        "job_id": job_id,
        "generation": task.generation,
        "intent_revision": project_intent_revision,
        "policy_sha256": policy_sha256,
        "dataset_id": dataset_id,
        "dataset_sha256": dataset_sha256,
        "environment_id": environment_id,
        "environment_identity": env_identity,
        "protocol_sha256": protocol_sha256,
        "canonical_arguments": canonical_args,
        "canonical_arguments_sha256": canonical_args_sha256,
        "planned_spec": planned_spec,
        "planned_spec_sha256": planned_spec_sha256,
    }))
}

fn reconcile_scheduler_submission(
    root: &Path,
    db: &Connection,
    id: &ToolIdentity<'_>,
    existed: bool,
    intent: &Value,
    dispatch: impl FnOnce() -> Result<String>,
) -> Result<String> {
    if !existed {
        return match dispatch() {
            Ok(result) => Ok(result),
            Err(error) => Ok(json!({
                "status": "error",
                "error": error,
                "effect_outcome": "unverified"
            }).to_string()),
        };
    }

    // Process crashed after intent persisted. Reconcile against authoritative database state.
    // 1. Verify canonical arguments
    let current_args: Value = serde_json::from_str(id.arguments).map_err(|e| e.to_string())?;
    let current_canonical = canonicalize_value(&current_args);
    let current_canonical_sha256 = hash(serde_json::to_string(&current_canonical).map_err(|e| e.to_string())?.as_bytes());
    if let Some(expected_sha256) = intent.get("canonical_arguments_sha256").and_then(Value::as_str) {
        if current_canonical_sha256 != expected_sha256 {
            return Err("durable tool identity conflict: arguments differ from persisted intent".into());
        }
    }

    // 2. Verify task generation & intent revision fencing
    let task_data: Option<String> = sql(db.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ).optional())?;
    let Some(task_data) = task_data else {
        return Ok(json!({
            "status": "rejected",
            "reason": "task_missing",
            "error": "Task does not exist in authoritative state",
            "effect_outcome": "rejected"
        }).to_string());
    };
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;

    let (project_state, current_intent_revision): (String, u64) = sql(db.query_row(
        "SELECT state, intent_revision FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ))?;

    let expected_gen = intent.get("generation").and_then(Value::as_u64).unwrap_or(0);
    let expected_rev = intent.get("intent_revision").and_then(Value::as_u64).unwrap_or(0);

    if task.status != "active" || task.generation != expected_gen || current_intent_revision != expected_rev {
        return Ok(json!({
            "status": "rejected",
            "reason": "stale_generation_fenced",
            "expected_generation": expected_gen,
            "current_generation": task.generation,
            "expected_intent_revision": expected_rev,
            "current_intent_revision": current_intent_revision,
            "task_status": task.status,
            "effect_outcome": "uncertain_fenced"
        }).to_string());
    }

    if project_state == "cancelled" {
        return Ok(json!({
            "status": "rejected",
            "reason": "project_cancelled_fenced",
            "error": "Project state is cancelled",
            "effect_outcome": "rejected"
        }).to_string());
    }

    // 3. Inspect authoritative `jobs` table
    let job_id = intent.get("job_id").and_then(Value::as_str).ok_or("intent missing job_id")?;
    let job_row: Option<(String, String, Option<String>)> = sql(db.query_row(
        "SELECT spec, status, reason FROM jobs WHERE project=?1 AND id=?2",
        params![id.project, job_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional())?;

    match job_row {
        None => {
            // Process crashed before job was admitted to `jobs` table.
            // Missing authoritative evidence produces explicit uncertain outcome.
            // Never blindly replay unknown effect.
            Ok(json!({
                "status": "rejected",
                "reason": "unconfirmed_submission_crash",
                "error": format!("Process crashed before job '{job_id}' was admitted; missing authoritative job receipt"),
                "effect_outcome": "uncertain"
            }).to_string())
        }
        Some((stored_spec_json, _, _)) => {
            // Job exists in authoritative jobs table. Verify spec matches durable intent.
            let actual_spec_sha256 = hash(stored_spec_json.as_bytes());
            if let Some(expected_spec_sha256) = intent.get("planned_spec_sha256").and_then(Value::as_str) {
                if actual_spec_sha256 != expected_spec_sha256 {
                    return Ok(json!({
                        "status": "rejected",
                        "reason": "contradictory_job_spec",
                        "error": format!("Job '{job_id}' exists with a specification that contradicts durable intent"),
                        "effect_outcome": "rejected"
                    }).to_string());
                }
            }

            // Job matches intent! Recover authoritative result without duplicate submission.
            let job_view = crate::execution::inspect(root, id.project, job_id)?;
            let result_str = serde_json::to_string(&job_view).map_err(|e| e.to_string())?;
            Ok(result_str)
        }
    }
}

/// All calls receive a durable fence, including calls whose effects cannot yet be
/// reconciled. Only file effects have an automatic before/after reconciliation.
pub fn execute(root: &Path, id: ToolIdentity<'_>, allow_new: bool,
    dispatch: impl FnOnce() -> Result<String>) -> Result<String> {
    let _lock = BranchLock::acquire(root, id.project, id.branch)?;
    let mut db = sql(Connection::open(root.join("state.sqlite")))?;
    sql(db.busy_timeout(Duration::from_secs(2)))?;
    sql(db.execute_batch("PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;"))?;
    let action: String = sql(db.query_row(
        "SELECT action FROM provider_turn_receipts WHERE project=?1 AND task=?2 AND turn=?3",
        params![id.project,id.task,id.turn], |r| r.get(0)))?;
    require_settlement(&db,id.project,id.task,&action)?;
    let response: String = sql(db.query_row(
        "SELECT response_json FROM provider_turn_receipts WHERE project=?1 AND task=?2 AND turn=?3",
        params![id.project,id.task,id.turn], |r| r.get(0)))?;
    let response: super::gateway::ChatCompletionResponse = serde_json::from_str(&response).map_err(|e|e.to_string())?;
    if !response.tool_calls.iter().any(|call| call.id == id.call_id && call.function.name == id.name && call.function.arguments == id.arguments) {
        return Err("tool intent is not bound to the authoritative cached response".into());
    }
    let tx = sql(db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate))?;
    let row: Option<(String,String,String,String,String,Option<String>,Option<String>)> = sql(tx.query_row(
        "SELECT action,branch,name,arguments,intent_json,outcome,outcome_sha256 FROM tool_outcomes WHERE project=?1 AND task=?2 AND turn=?3 AND call_id=?4",
        params![id.project,id.task,id.turn,id.call_id],
        |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?,r.get(5)?,r.get(6)?))).optional())?;
    crate::execution::enforce_state_write_admission(root,Some(&tx),
        (id.arguments.len() as u64).saturating_mul(4).saturating_add(32*1024),
        crate::execution::StorageWriteKind::EmergencyRecovery,Some(id.project))?;
    let existed = row.is_some();
    let intent: Value;
    if let Some((a,b,n,args,i,outcome,digest)) = row {
        if a != action || b != id.branch || n != id.name || args != id.arguments {
            return Err("durable tool identity conflict".into());
        }
        if let Some(outcome) = outcome {
            if digest.as_deref() != Some(hash(outcome.as_bytes()).as_str()) {
                return Err("durable tool outcome digest mismatch".into());
            }
            return Ok(outcome);
        }
        intent = serde_json::from_str(&i).map_err(|e| e.to_string())?;
    } else {
        if !allow_new { return Err("tool outcome unknown; legacy call has no durable intent".into()); }
        let args: Value = serde_json::from_str(id.arguments).map_err(|e| e.to_string())?;
        intent = if matches!(id.name, "file_write" | "file_patch") {
            let rel = args["path"].as_str().ok_or("path missing")?;
            let path = Workspace::checked_path(root,id.project,id.branch,rel)?;
            let before = file_hash(&path)?;
            let after = if id.name == "file_write" {
                args["content"].as_str().ok_or("content missing")?.to_string()
            } else {
                let current = fs::read_to_string(&path).map_err(|e| e.to_string())?;
                let target = args["target"].as_str().ok_or("target missing")?;
                if !current.contains(target) { return Err("patch target missing".into()); }
                current.replacen(target,args["replacement"].as_str().ok_or("replacement missing")?,1)
            };
            json!({"path":rel,"before_sha256":before,"after_sha256":hash(after.as_bytes()),"after":after})
        } else if id.name == "schedule_pipeline" {
            build_scheduler_submission_intent(&tx, &id, &args, &action)?
        } else if matches!(id.name, "delegate" | "delegate_task" | "schedule_specialist") {
            build_delegation_intent(&tx, &id, &args, &action)?
        } else if id.name == "register_dataset" {
            build_dataset_registration_intent(root, &tx, &id, &args, &action)?
        } else if matches!(id.name, "request_finalization" | "publish_delivery" | "publish_bundle") {
            build_delivery_publication_intent(&tx, &id, &args, &action)?
        } else { json!({"reconciliation":"requires_authoritative_tool_evidence"}) };
        sql(tx.execute("INSERT INTO tool_outcomes(project,task,turn,call_id,action,branch,name,arguments,intent_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)",
            params![id.project,id.task,id.turn,id.call_id,action,id.branch,id.name,id.arguments,intent.to_string()]))?;
    }
    sql(tx.commit())?; // durable before any external effect
    let result = if matches!(id.name, "file_write" | "file_patch") {
        let rel = intent["path"].as_str().ok_or("invalid file intent")?;
        let path = Workspace::checked_path(root,id.project,id.branch,rel)?;
        let actual = file_hash(&path)?;
        let before = intent["before_sha256"].as_str();
        let after = intent["after_sha256"].as_str().ok_or("invalid file intent hash")?;
        let content = intent["after"].as_str().ok_or("invalid file intent content")?;
        if hash(content.as_bytes()) != after { return Err("file intent content hash mismatch".into()); }
        if actual.as_deref() == Some(after) {
            // Effect already published; repair bookkeeping without repeating mutation.
            File::open(&path).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
        } else if actual.as_deref() == before {
            if let Some(parent) = path.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
            let staging = path.with_file_name(format!(".tool-{}.staging",hash(format!("{}/{}/{}",id.task,id.turn,id.call_id).as_bytes())));
            let mut file = OpenOptions::new().create(true).truncate(true).write(true)
                .custom_flags(libc::O_NOFOLLOW).open(&staging).map_err(|e| e.to_string())?;
            file.write_all(content.as_bytes()).and_then(|_|file.sync_all()).map_err(|e| e.to_string())?;
            fs::rename(&staging,&path).map_err(|e| e.to_string())?;
        } else { return Err("tool outcome unknown; file differs from durable before and after evidence".into()); }
        File::open(path.parent().ok_or("file parent missing")?).and_then(|f|f.sync_all()).map_err(|e|e.to_string())?;
        Workspace::update_branch_head(root,id.project,id.branch)?;
        json!({"status":if id.name=="file_write" {"written"} else {"patched"},"path":rel,"sha256":after}).to_string()
    } else if id.name == "schedule_pipeline" {
        reconcile_scheduler_submission(root, &db, &id, existed, &intent, dispatch)?
    } else if matches!(id.name, "delegate" | "delegate_task" | "schedule_specialist") {
        reconcile_delegation(root, &db, &id, existed, &intent, dispatch)?
    } else if id.name == "register_dataset" {
        reconcile_dataset_registration(root, &db, &id, existed, &intent, dispatch)?
    } else if matches!(id.name, "request_finalization" | "publish_delivery" | "publish_bundle") {
        reconcile_delivery_publication(root, &db, &id, existed, &intent, dispatch)?
    } else {
        if existed { return Err("tool outcome unknown; authoritative tool reconciliation required".into()); }
        match dispatch() { Ok(result) => result, Err(error) => {
            // Record the error for context delivery, never present it as a successful effect.
            json!({"status":"error","error":error,"effect_outcome":"unverified"}).to_string()
        } }
    };
    sql(db.execute("UPDATE tool_outcomes SET outcome=?5,outcome_sha256=?6 WHERE project=?1 AND task=?2 AND turn=?3 AND call_id=?4 AND outcome IS NULL",
        params![id.project,id.task,id.turn,id.call_id,result,hash(result.as_bytes())]))?;
    Ok(result)
}

pub fn build_delegation_intent(
    tx: &rusqlite::Transaction,
    id: &ToolIdentity<'_>,
    args: &Value,
    action: &str,
) -> Result<Value> {
    let target_role = args["role_id"].as_str().ok_or("role_id missing")?;
    if super::roles::get_role(target_role).is_none() {
        return Err(format!("unknown specialist role: {target_role}"));
    }
    let target_task_id = args.get("sub_task_id")
        .or(args.get("child_task_id"))
        .or(args.get("task_id"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("{}-{}-{}", id.task, target_role, id.call_id));

    // Target task identity reuse check before dispatch
    let exists_in_tasks: bool = sql(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM tasks WHERE project=?1 AND id=?2)",
        params![id.project, &target_task_id],
        |r| r.get(0),
    ))?;
    let exists_in_agent_tasks: bool = sql(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM agent_tasks WHERE project=?1 AND task_id=?2)",
        params![id.project, &target_task_id],
        |r| r.get(0),
    ))?;
    if exists_in_tasks || exists_in_agent_tasks {
        return Err("target task identity reuse: child task already exists in authoritative state".into());
    }

    let canonical_args = canonicalize_value(args);
    let canonical_args_str = serde_json::to_string(&canonical_args).map_err(|e| e.to_string())?;
    let canonical_args_sha256 = hash(canonical_args_str.as_bytes());

    let task_data: String = sql(tx.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ))?;
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;
    if task.generation == 0 || task.status != "active" {
        return Err("stale or inactive task fence for specialist delegation".into());
    }

    let (project_state, project_intent_revision, project_policy): (String, u64, String) = sql(tx.query_row(
        "SELECT state, intent_revision, policy FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ))?;
    if project_state != "created" && project_state != "finalizing" {
        return Err(format!("project does not admit work in state {project_state}"));
    }
    let policy_sha256 = hash(project_policy.as_bytes());

    Ok(json!({
        "kind": "specialist_delegation",
        "operation": id.name,
        "project": id.project,
        "task": id.task,
        "turn": id.turn,
        "call_id": id.call_id,
        "action": action,
        "generation": task.generation,
        "intent_revision": project_intent_revision,
        "policy_sha256": policy_sha256,
        "target_role": target_role,
        "target_task_id": target_task_id,
        "canonical_arguments": canonical_args,
        "canonical_arguments_sha256": canonical_args_sha256,
    }))
}

pub fn reconcile_delegation(
    _root: &Path,
    db: &Connection,
    id: &ToolIdentity<'_>,
    existed: bool,
    intent: &Value,
    dispatch: impl FnOnce() -> Result<String>,
) -> Result<String> {
    if !existed {
        return match dispatch() {
            Ok(result) => Ok(result),
            Err(error) => Ok(json!({
                "status": "error",
                "error": error,
                "effect_outcome": "unverified"
            }).to_string()),
        };
    }

    // Process crashed after intent persisted. Reconcile against authoritative database state.
    // 1. Verify canonical arguments
    let current_args: Value = serde_json::from_str(id.arguments).map_err(|e| e.to_string())?;
    let current_canonical = canonicalize_value(&current_args);
    let current_canonical_sha256 = hash(serde_json::to_string(&current_canonical).map_err(|e| e.to_string())?.as_bytes());
    if let Some(expected_sha256) = intent.get("canonical_arguments_sha256").and_then(Value::as_str) {
        if current_canonical_sha256 != expected_sha256 {
            return Err("durable tool identity conflict: arguments differ from persisted intent".into());
        }
    }

    // 2. Verify task generation & intent revision fencing
    let task_data: Option<String> = sql(db.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ).optional())?;
    let Some(task_data) = task_data else {
        return Ok(json!({
            "status": "rejected",
            "reason": "task_missing",
            "error": "Task does not exist in authoritative state",
            "effect_outcome": "rejected"
        }).to_string());
    };
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;

    let (project_state, current_intent_revision): (String, u64) = sql(db.query_row(
        "SELECT state, intent_revision FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ))?;

    let expected_gen = intent.get("generation").and_then(Value::as_u64).unwrap_or(0);
    let expected_rev = intent.get("intent_revision").and_then(Value::as_u64).unwrap_or(0);

    if task.status != "active" || task.generation != expected_gen || current_intent_revision != expected_rev {
        return Ok(json!({
            "status": "rejected",
            "reason": "stale_generation_fenced",
            "expected_generation": expected_gen,
            "current_generation": task.generation,
            "expected_intent_revision": expected_rev,
            "current_intent_revision": current_intent_revision,
            "task_status": task.status,
            "effect_outcome": "uncertain_fenced"
        }).to_string());
    }

    if project_state == "cancelled" {
        return Ok(json!({
            "status": "rejected",
            "reason": "project_cancelled_fenced",
            "error": "Project state is cancelled",
            "effect_outcome": "rejected"
        }).to_string());
    }

    // 3. Inspect authoritative `agent_tasks` table
    let target_task_id = intent.get("target_task_id").and_then(Value::as_str).ok_or("intent missing target_task_id")?;
    let target_role = intent.get("target_role").and_then(Value::as_str).unwrap_or("");
    let agent_task_row: Option<(String, Option<String>, String, Option<String>)> = sql(db.query_row(
        "SELECT role_id, parent_task_id, status, assigned_branch FROM agent_tasks WHERE project=?1 AND task_id=?2",
        params![id.project, target_task_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    ).optional())?;

    match agent_task_row {
        None => {
            // Process crashed before child task was admitted to `agent_tasks`.
            // Missing authoritative evidence produces explicit uncertain outcome.
            Ok(json!({
                "status": "rejected",
                "reason": "unconfirmed_delegation_crash",
                "error": format!("Process crashed before child task '{target_task_id}' was admitted; missing authoritative delegation receipt"),
                "effect_outcome": "uncertain"
            }).to_string())
        }
        Some((stored_role, parent_id, status, assigned_branch)) => {
            // Child task exists! Verify parent and role match durable intent.
            if parent_id.as_deref() != Some(id.task) || stored_role != target_role {
                return Ok(json!({
                    "status": "rejected",
                    "reason": "target_task_identity_reuse",
                    "error": format!("Child task '{target_task_id}' exists with parent '{parent_id:?}' and role '{stored_role}' that contradicts delegation intent"),
                    "effect_outcome": "rejected"
                }).to_string());
            }

            // Delegation matches intent! Recover authoritative result without duplicate submission or billing.
            Ok(json!({
                "status": if status == "paused" { "scheduled" } else { &status },
                "sub_task_id": target_task_id,
                "role": stored_role,
                "assigned_branch": assigned_branch.unwrap_or_default(),
                "reconciled": true
            }).to_string())
        }
    }
}

pub fn build_dataset_registration_intent(
    root: &Path,
    tx: &rusqlite::Transaction,
    id: &ToolIdentity<'_>,
    args: &Value,
    action: &str,
) -> Result<Value> {
    let dataset_id = args.get("dataset_id")
        .or(args.get("id"))
        .and_then(Value::as_str)
        .ok_or("dataset_id missing")?;
    let path_str = args.get("path")
        .or(args.get("source"))
        .and_then(Value::as_str)
        .ok_or("path missing")?;

    let canonical_args = canonicalize_value(args);
    let canonical_args_str = serde_json::to_string(&canonical_args).map_err(|e| e.to_string())?;
    let canonical_args_sha256 = hash(canonical_args_str.as_bytes());

    let task_data: String = sql(tx.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ))?;
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;
    if task.generation == 0 || task.status != "active" {
        return Err("stale or inactive task fence for dataset registration".into());
    }

    let (project_state, project_intent_revision, project_policy): (String, u64, String) = sql(tx.query_row(
        "SELECT state, intent_revision, policy FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ))?;
    if project_state != "created" && project_state != "finalizing" {
        return Err(format!("project does not admit work in state {project_state}"));
    }
    let policy_sha256 = hash(project_policy.as_bytes());

    // Resolve source file to verify existence and compute content hash
    let src_path = if Path::new(path_str).is_absolute() {
        PathBuf::from(path_str)
    } else {
        match Workspace::checked_path(root, id.project, id.branch, path_str) {
            Ok(p) => p,
            Err(_) => root.join(path_str),
        }
    };
    let meta = fs::symlink_metadata(&src_path).map_err(|e| format!("source dataset path error: {e}"))?;
    if meta.file_type().is_symlink() || !meta.is_file() || meta.len() == 0 || meta.len() > 32 * 1024 * 1024 {
        return Err("dataset must be a regular non-symlink CSV between 1 byte and 32 MiB".into());
    }
    let content = fs::read(&src_path).map_err(|e| format!("cannot read source dataset: {e}"))?;
    let source_sha256 = hash(&content);
    let source_bytes = content.len() as u64;

    // Check target dataset identity reuse before dispatch
    let existing_hash: Option<String> = sql(tx.query_row(
        "SELECT sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",
        params![id.project, dataset_id],
        |r| r.get(0),
    ).optional())?;
    if let Some(ref existing) = existing_hash {
        if existing != &source_sha256 {
            return Err("dataset id already exists with different content".into());
        }
    }

    Ok(json!({
        "kind": "dataset_registration",
        "operation": id.name,
        "project": id.project,
        "task": id.task,
        "turn": id.turn,
        "call_id": id.call_id,
        "action": action,
        "generation": task.generation,
        "intent_revision": project_intent_revision,
        "policy_sha256": policy_sha256,
        "target_dataset_id": dataset_id,
        "source_path": path_str,
        "source_sha256": source_sha256,
        "source_bytes": source_bytes,
        "canonical_arguments": canonical_args,
        "canonical_arguments_sha256": canonical_args_sha256,
    }))
}

pub fn reconcile_dataset_registration(
    _root: &Path,
    db: &Connection,
    id: &ToolIdentity<'_>,
    existed: bool,
    intent: &Value,
    dispatch: impl FnOnce() -> Result<String>,
) -> Result<String> {
    if !existed {
        return match dispatch() {
            Ok(result) => Ok(result),
            Err(error) => Ok(json!({
                "status": "error",
                "error": error,
                "effect_outcome": "unverified"
            }).to_string()),
        };
    }

    // Process crashed after intent persisted. Reconcile against authoritative database state.
    // 1. Verify canonical arguments
    let current_args: Value = serde_json::from_str(id.arguments).map_err(|e| e.to_string())?;
    let current_canonical = canonicalize_value(&current_args);
    let current_canonical_sha256 = hash(serde_json::to_string(&current_canonical).map_err(|e| e.to_string())?.as_bytes());
    if let Some(expected_sha256) = intent.get("canonical_arguments_sha256").and_then(Value::as_str) {
        if current_canonical_sha256 != expected_sha256 {
            return Err("durable tool identity conflict: arguments differ from persisted intent".into());
        }
    }

    // 2. Verify task generation & intent revision fencing
    let task_data: Option<String> = sql(db.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ).optional())?;
    let Some(task_data) = task_data else {
        return Ok(json!({
            "status": "rejected",
            "reason": "task_missing",
            "error": "Task does not exist in authoritative state",
            "effect_outcome": "rejected"
        }).to_string());
    };
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;

    let (project_state, current_intent_revision): (String, u64) = sql(db.query_row(
        "SELECT state, intent_revision FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ))?;

    let expected_gen = intent.get("generation").and_then(Value::as_u64).unwrap_or(0);
    let expected_rev = intent.get("intent_revision").and_then(Value::as_u64).unwrap_or(0);

    if task.status != "active" || task.generation != expected_gen || current_intent_revision != expected_rev {
        return Ok(json!({
            "status": "rejected",
            "reason": "stale_generation_fenced",
            "expected_generation": expected_gen,
            "current_generation": task.generation,
            "expected_intent_revision": expected_rev,
            "current_intent_revision": current_intent_revision,
            "task_status": task.status,
            "effect_outcome": "uncertain_fenced"
        }).to_string());
    }

    if project_state == "cancelled" {
        return Ok(json!({
            "status": "rejected",
            "reason": "project_cancelled_fenced",
            "error": "Project state is cancelled",
            "effect_outcome": "rejected"
        }).to_string());
    }

    // 3. Inspect authoritative `scientific_datasets` table
    let target_dataset_id = intent.get("target_dataset_id").and_then(Value::as_str).ok_or("intent missing target_dataset_id")?;
    let row: Option<(String, i64, String)> = sql(db.query_row(
        "SELECT path, bytes, sha256 FROM scientific_datasets WHERE project=?1 AND id=?2",
        params![id.project, target_dataset_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional())?;

    match row {
        None => {
            // Process crashed before dataset was admitted to `scientific_datasets`.
            // Missing authoritative evidence produces explicit uncertain outcome.
            Ok(json!({
                "status": "rejected",
                "reason": "unconfirmed_registration_crash",
                "error": format!("Process crashed before dataset '{target_dataset_id}' was registered; missing authoritative dataset receipt"),
                "effect_outcome": "uncertain"
            }).to_string())
        }
        Some((_stored_path, stored_bytes, stored_sha256)) => {
            let expected_sha256 = intent.get("source_sha256").and_then(Value::as_str).unwrap_or("");
            if stored_sha256 != expected_sha256 {
                return Ok(json!({
                    "status": "rejected",
                    "reason": "contradictory_dataset_content",
                    "error": format!("Dataset '{target_dataset_id}' exists with content hash that contradicts durable intent"),
                    "effect_outcome": "rejected"
                }).to_string());
            }

            Ok(json!({
                "project_id": id.project,
                "dataset_id": target_dataset_id,
                "sha256": stored_sha256,
                "bytes": stored_bytes,
                "reused": true
            }).to_string())
        }
    }
}

pub fn build_delivery_publication_intent(
    tx: &rusqlite::Transaction,
    id: &ToolIdentity<'_>,
    args: &Value,
    action: &str,
) -> Result<Value> {
    let cand_id = args.get("candidate_id").and_then(Value::as_str).ok_or("candidate_id missing")?;
    let target_bundle_id = args.get("bundle_id").and_then(Value::as_str).unwrap_or(cand_id);
    let finalize_job_id = format!("finalize-{cand_id}");

    let canonical_args = canonicalize_value(args);
    let canonical_args_str = serde_json::to_string(&canonical_args).map_err(|e| e.to_string())?;
    let canonical_args_sha256 = hash(canonical_args_str.as_bytes());

    let task_data: String = sql(tx.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ))?;
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;
    if task.generation == 0 || task.status != "active" {
        return Err("stale or inactive task fence for delivery publication".into());
    }

    let (project_state, project_intent_revision, project_policy): (String, u64, String) = sql(tx.query_row(
        "SELECT state, intent_revision, policy FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ))?;
    if project_state == "cancelled" || project_state == "completed" {
        return Err(format!("project does not admit finalization in state {project_state}"));
    }
    let policy_sha256 = hash(project_policy.as_bytes());

    // Check if candidate has received an approved critic audit or content decision
    let has_audit: bool = sql(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM critic_audits WHERE project=?1 AND candidate_id=?2 AND status='approved' AND constraints_verified=1)",
        params![id.project, cand_id],
        |r| r.get(0),
    ))?;
    let has_decision: bool = sql(tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM content_decisions WHERE project=?1 AND candidate=?2 AND approved=1)",
        params![id.project, cand_id],
        |r| r.get(0),
    ))?;
    if !has_audit && !has_decision {
        return Err(format!("delivery publication refused: candidate '{cand_id}' lacks approved critic audit or content decision"));
    }

    Ok(json!({
        "kind": "delivery_publication",
        "operation": id.name,
        "project": id.project,
        "task": id.task,
        "turn": id.turn,
        "call_id": id.call_id,
        "action": action,
        "generation": task.generation,
        "intent_revision": project_intent_revision,
        "policy_sha256": policy_sha256,
        "candidate_id": cand_id,
        "target_bundle_id": target_bundle_id,
        "finalize_job_id": finalize_job_id,
        "canonical_arguments": canonical_args,
        "canonical_arguments_sha256": canonical_args_sha256,
    }))
}

pub fn reconcile_delivery_publication(
    root: &Path,
    db: &Connection,
    id: &ToolIdentity<'_>,
    existed: bool,
    intent: &Value,
    dispatch: impl FnOnce() -> Result<String>,
) -> Result<String> {
    if !existed {
        return match dispatch() {
            Ok(result) => Ok(result),
            Err(error) => Ok(json!({
                "status": "error",
                "error": error,
                "effect_outcome": "unverified"
            }).to_string()),
        };
    }

    // Process crashed after intent persisted. Reconcile against authoritative database state.
    // 1. Verify canonical arguments
    let current_args: Value = serde_json::from_str(id.arguments).map_err(|e| e.to_string())?;
    let current_canonical = canonicalize_value(&current_args);
    let current_canonical_sha256 = hash(serde_json::to_string(&current_canonical).map_err(|e| e.to_string())?.as_bytes());
    if let Some(expected_sha256) = intent.get("canonical_arguments_sha256").and_then(Value::as_str) {
        if current_canonical_sha256 != expected_sha256 {
            return Err("durable tool identity conflict: arguments differ from persisted intent".into());
        }
    }

    // 2. Verify task generation & intent revision fencing
    let task_data: Option<String> = sql(db.query_row(
        "SELECT data FROM tasks WHERE project=?1 AND id=?2",
        params![id.project, id.task],
        |r| r.get(0),
    ).optional())?;
    let Some(task_data) = task_data else {
        return Ok(json!({
            "status": "rejected",
            "reason": "task_missing",
            "error": "Task does not exist in authoritative state",
            "effect_outcome": "rejected"
        }).to_string());
    };
    let task: crate::core::Task = serde_json::from_str(&task_data).map_err(|e| e.to_string())?;

    let (project_state, current_intent_revision): (String, u64) = sql(db.query_row(
        "SELECT state, intent_revision FROM projects WHERE id=?1",
        params![id.project],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ))?;

    let expected_gen = intent.get("generation").and_then(Value::as_u64).unwrap_or(0);
    let expected_rev = intent.get("intent_revision").and_then(Value::as_u64).unwrap_or(0);

    if task.status != "active" || task.generation != expected_gen || current_intent_revision != expected_rev {
        return Ok(json!({
            "status": "rejected",
            "reason": "stale_generation_fenced",
            "expected_generation": expected_gen,
            "current_generation": task.generation,
            "expected_intent_revision": expected_rev,
            "current_intent_revision": current_intent_revision,
            "task_status": task.status,
            "effect_outcome": "uncertain_fenced"
        }).to_string());
    }

    if project_state == "cancelled" {
        return Ok(json!({
            "status": "rejected",
            "reason": "project_cancelled_fenced",
            "error": "Project state is cancelled",
            "effect_outcome": "rejected"
        }).to_string());
    }

    // 3. Inspect authoritative `jobs` table for `finalize_job_id`
    let finalize_job_id = intent.get("finalize_job_id").and_then(Value::as_str).ok_or("intent missing finalize_job_id")?;
    let cand_id = intent.get("candidate_id").and_then(Value::as_str).unwrap_or("");
    let job_row: Option<(String, String, Option<String>)> = sql(db.query_row(
        "SELECT spec, status, reason FROM jobs WHERE project=?1 AND id=?2",
        params![id.project, finalize_job_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional())?;

    match job_row {
        None => {
            // Process crashed before finalization job was admitted.
            // If project is stuck in 'finalizing' and no gate is reserved or consumed, recover to 'created'
            let gate_reserved: bool = db.query_row(
                "SELECT EXISTS(SELECT 1 FROM final_test_gates WHERE project=?1 AND status IN ('reserved','consumed'))",
                params![id.project],
                |r| r.get(0),
            ).unwrap_or(false);
            if !gate_reserved && project_state == "finalizing" {
                let _ = db.execute("UPDATE projects SET state='created' WHERE id=?1 AND state='finalizing'", params![id.project]);
            }
            Ok(json!({
                "status": "rejected",
                "reason": "unconfirmed_publication_crash",
                "error": format!("Process crashed before finalization job '{finalize_job_id}' was admitted; missing authoritative delivery receipt"),
                "effect_outcome": "uncertain"
            }).to_string())
        }
        Some((stored_spec_json, status, reason)) => {
            let stored_spec: Value = serde_json::from_str(&stored_spec_json).unwrap_or(json!({}));
            if let Some(spec_cand) = stored_spec["probe"]["candidate_id"].as_str() {
                if spec_cand != cand_id {
                    return Ok(json!({
                        "status": "rejected",
                        "reason": "contradictory_job_spec",
                        "error": format!("Finalization job '{finalize_job_id}' candidate '{spec_cand}' contradicts durable intent candidate '{cand_id}'"),
                        "effect_outcome": "rejected"
                    }).to_string());
                }
            }

            let bundle_dir = root.join("workflows").join(id.project).join("bundle");
            let manifest_path = bundle_dir.join("manifest.json");
            if status == "succeeded" {
                if !manifest_path.is_file() {
                    return Ok(json!({
                        "status": "rejected",
                        "reason": "incomplete_bundle_publication",
                        "error": format!("Finalization job '{finalize_job_id}' succeeded but bundle manifest at '{:?}' is missing", manifest_path),
                        "effect_outcome": "rejected"
                    }).to_string());
                }
                Ok(json!({
                    "status": "finalized_and_delivered",
                    "candidate_id": cand_id,
                    "finalize_job_id": finalize_job_id,
                    "bundle_dir": bundle_dir,
                    "reconciled": true
                }).to_string())
            } else if status == "failed" || status == "interrupted" {
                let gate_consumed: bool = db.query_row(
                    "SELECT EXISTS(SELECT 1 FROM final_test_gates WHERE project=?1 AND status='consumed')",
                    params![id.project],
                    |r| r.get(0),
                ).unwrap_or(false);
                if !gate_consumed && project_state == "finalizing" {
                    let _ = db.execute("UPDATE projects SET state='created' WHERE id=?1 AND state='finalizing'", params![id.project]);
                }
                Ok(json!({
                    "status": "rejected",
                    "reason": "finalization_job_failed",
                    "error": format!("Finalization job '{finalize_job_id}' status '{status}': {:?}", reason),
                    "effect_outcome": "rejected"
                }).to_string())
            } else {
                Ok(json!({
                    "status": "finalization_queued",
                    "candidate_id": cand_id,
                    "finalize_job_id": finalize_job_id,
                    "reconciled": true
                }).to_string())
            }
        }
    }
}
