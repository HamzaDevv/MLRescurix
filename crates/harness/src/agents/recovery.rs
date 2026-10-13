//! Durable tool intents precede effects; outcomes precede context checkpoints.
//! A stored outcome may be delivered again, but an unknown effect is never replayed.
use crate::{Result, core::Action};
use super::workspace::BranchWorkspaceManager as Workspace;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::{self, File, OpenOptions}, io::Write, os::fd::AsRawFd,
    os::unix::fs::OpenOptionsExt, path::Path, time::Duration};

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
