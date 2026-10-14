use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::{cas::ContentAddressableStorage, Result};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn bounded_output<T: Serialize>(value: T) -> Result<T> {
    let bytes = serde_json::to_vec(&value).map_err(|e| e.to_string())?.len();
    if bytes > 64 * 1024 { return Err("artifact query output exceeds 64 KiB".into()); }
    Ok(value)
}

fn valid_id(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 128 || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b)) {
        return Err(format!("invalid identity: '{s}'"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactView {
    pub project: String,
    pub artifact_id: String,
    pub kind: String,
    pub producer_authority: String,
    pub current_version_id: Option<String>,
    pub pinned: bool,
    pub visibility: String,
    pub created_ms: i64,
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactVersionView {
    pub project: String,
    pub version_id: String,
    pub artifact_id: String,
    pub blob_sha256: String,
    pub bytes: u64,
    pub media_type: String,
    pub index_digest: String,
    pub summary_text: Option<String>,
    pub schema_version: i64,
    pub metadata: Value,
    pub source_job_id: Option<String>,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LineageEdgeView {
    pub project: String,
    pub source_version_id: String,
    pub target_version_id: String,
    pub relation: String,
    pub metadata: Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AnnotationView {
    pub project: String,
    pub annotation_id: String,
    pub version_id: String,
    pub author: String,
    pub revision: i64,
    pub note_text: String,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProjectSelectionView {
    pub project: String,
    pub selection_id: String,
    pub candidate_version_id: String,
    pub scorecard_version_id: String,
    pub protocol_sha256: String,
    pub rationale: String,
    pub is_apex: bool,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventSubscription {
    pub project: String,
    pub snapshot_cursor: i64,
    pub snapshot_truncated: bool,
    pub artifacts: Vec<ArtifactView>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventReplay { pub reset_required: bool, pub head_cursor: i64, pub events: Vec<DurableEventView> }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DurableEventView {
    pub cursor: i64,
    pub project: String,
    pub event_type: String,
    pub version: i64,
    pub payload: Value,
    pub committed_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactCommandReceipt {
    pub command_id: String,
    pub project: String,
    pub status: String,
    pub revision: i64,
    pub result: Value,
}

#[derive(Clone)]
pub struct ArtifactRegistry {
    root: PathBuf,
    cas: ContentAddressableStorage,
    readers: Arc<OnceLock<Result<Arc<crate::read_pool::ReadPool>>>>,
}

impl ArtifactRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let r = root.into();
        let cas = ContentAddressableStorage::new(&r);
        Self { root: r, cas, readers: Arc::new(OnceLock::new()) }
    }

    pub fn read_pool(&self) -> Result<&crate::read_pool::ReadPool> {
        self.readers.get_or_init(|| crate::read_pool::shared(&self.root)).as_deref().map_err(Clone::clone)
    }

    pub fn cas(&self) -> &ContentAddressableStorage {
        &self.cas
    }

    fn conn(&self) -> Result<Connection> {
        crate::execution::connect(&self.root)
    }

    /// Publishes an immutable successor and advances the project's shared
    /// revision in the same transaction as its idempotency receipt.
    pub fn publish_successor_command(
        &self, project: &str, command_id: &str, expected_revision: i64,
        artifact_id: &str, blob_sha256: &str, index_digest: &str,
        summary_text: Option<&str>, schema_version: u32, metadata: Option<&Value>,
    ) -> Result<ArtifactCommandReceipt> {
        valid_id(project)?; valid_id(command_id)?; valid_id(artifact_id)?;
        if index_digest.len() > 512 { return Err("index_digest exceeds bound of 512 UTF-8 bytes".into()); }
        if summary_text.is_some_and(|s| s.len() > 4096) { return Err("summary_text exceeds bound of 4096 UTF-8 bytes".into()); }
        let metadata_json = metadata.map(Value::to_string).unwrap_or_else(|| "{}".into());
        if metadata_json.len() > 4096 { return Err("metadata exceeds 4096 UTF-8 bytes".into()); }
        let request = json!({"op":"publish_successor","project":project,"expected_revision":expected_revision,
            "artifact_id":artifact_id,"blob_sha256":blob_sha256,"index_digest":index_digest,
            "summary_text":summary_text,"schema_version":schema_version,"metadata":metadata_json});
        self.run_artifact_command(project, command_id, expected_revision, request, |tx, revision| {
            let (kind, producer, current, tombstoned): (String,String,Option<String>,Option<i64>) = tx.query_row(
                "SELECT kind,producer_authority,current_version_id,tombstoned_ms FROM artifacts WHERE project=?1 AND artifact_id=?2",
                params![project, artifact_id], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?)),
            ).map_err(|_| format!("artifact '{artifact_id}' not found"))?;
            if tombstoned.is_some() { return Err("cannot publish a successor to a tombstoned artifact".into()); }
            if kind == "scorecard" || producer == "evaluator:trusted" { return Err("official scorecards require the trusted evaluator API".into()); }
            let current = current.ok_or_else(|| "artifact has no current version to supersede".to_string())?;
            let blob_exists: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM blobs WHERE sha256=?1)", [blob_sha256], |r| r.get(0)).map_err(|e|e.to_string())?;
            if !blob_exists { return Err("successor blob is not registered".into()); }
            let seq: i64 = tx.query_row("SELECT COALESCE(MAX(rowid),0)+1 FROM artifact_versions", [], |r| r.get(0)).map_err(|e|e.to_string())?;
            let version_id = format!("{}_{}", artifact_id, seq);
            let now = now_ms();
            tx.execute("INSERT INTO artifact_versions(project,version_id,artifact_id,blob_sha256,index_digest,summary_text,schema_version,metadata_json,source_job_id,created_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,NULL,?9)",
                params![project,version_id,artifact_id,blob_sha256,index_digest,summary_text,schema_version as i64,metadata_json,now]).map_err(|e|e.to_string())?;
            tx.execute("UPDATE artifacts SET current_version_id=?3,updated_ms=?4 WHERE project=?1 AND artifact_id=?2 AND tombstoned_ms IS NULL",
                params![project,artifact_id,version_id,now]).map_err(|e|e.to_string())?;
            // The directed edge points from the new version to the version it supersedes.
            tx.execute("INSERT INTO artifact_lineage(project,source_version_id,target_version_id,relation,metadata_json,created_ms) VALUES(?1,?2,?3,'supersedes','{}',?4)",
                params![project,version_id,current,now]).map_err(|e|e.to_string())?;
            Ok((revision + 1, json!({"version_id":version_id,"supersedes":current})))
        })
    }

    /// Tombstones a logical artifact while retaining versions, lineage and all
    /// bytes as historical evidence. GC remains a separate reference-aware step.
    pub fn tombstone_artifact_command(
        &self, project: &str, command_id: &str, expected_revision: i64, artifact_id: &str,
    ) -> Result<ArtifactCommandReceipt> {
        valid_id(project)?; valid_id(command_id)?; valid_id(artifact_id)?;
        let request = json!({"op":"tombstone_artifact","project":project,"expected_revision":expected_revision,"artifact_id":artifact_id});
        self.run_artifact_command(project, command_id, expected_revision, request, |tx, revision| {
            let (pinned, tombstoned): (i64,Option<i64>) = tx.query_row(
                "SELECT pinned,tombstoned_ms FROM artifacts WHERE project=?1 AND artifact_id=?2",
                params![project,artifact_id], |r| Ok((r.get(0)?,r.get(1)?)),
            ).map_err(|_| format!("artifact '{artifact_id}' not found"))?;
            if tombstoned.is_some() { return Err("artifact is already tombstoned".into()); }
            if pinned != 0 { return Err("pinned artifacts cannot be tombstoned".into()); }
            let selected: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_versions v JOIN project_selections s ON s.project=v.project AND (s.candidate_version_id=v.version_id OR s.scorecard_version_id=v.version_id) WHERE v.project=?1 AND v.artifact_id=?2)",
                params![project,artifact_id], |r| r.get(0),
            ).map_err(|e|e.to_string())?;
            if selected { return Err("artifacts referenced by project selection history cannot be tombstoned".into()); }
            let leased: bool = tx.query_row(
                "SELECT EXISTS(SELECT 1 FROM artifact_versions v JOIN jobs j ON j.project=v.project AND j.id=v.source_job_id WHERE v.project=?1 AND v.artifact_id=?2 AND j.status IN ('queued','running','terminating','recovering'))",
                params![project,artifact_id], |r| r.get(0),
            ).map_err(|e|e.to_string())?;
            if leased { return Err("artifacts produced by active jobs cannot be tombstoned".into()); }
            let now = now_ms();
            tx.execute("UPDATE artifacts SET tombstoned_ms=?3,updated_ms=?3 WHERE project=?1 AND artifact_id=?2 AND tombstoned_ms IS NULL",
                params![project,artifact_id,now]).map_err(|e|e.to_string())?;
            Ok((revision + 1, json!({"artifact_id":artifact_id,"tombstoned_ms":now})))
        })
    }

    fn run_artifact_command<F>(&self, project: &str, command_id: &str, expected_revision: i64, request: Value, apply: F) -> Result<ArtifactCommandReceipt>
    where F: FnOnce(&rusqlite::Transaction<'_>, i64) -> Result<(i64, Value)> {
        let request_text = format!("artifact-v2:{}", request);
        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e|e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, request_text.len() as u64 * 4 + 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        if let Some((prior_request, prior_receipt)) = tx.query_row("SELECT request,receipt FROM commands WHERE project=?1 AND id=?2", params![project,command_id], |r| Ok((r.get::<_,String>(0)?,r.get::<_,String>(1)?))).optional().map_err(|e|e.to_string())? {
            if prior_request != request_text { return Err("artifact command identity reused with different payload or scope".into()); }
            return serde_json::from_str(&prior_receipt).map_err(|e|e.to_string());
        }
        let current_revision: i64 = tx.query_row("SELECT revision FROM projects WHERE id=?1", [project], |r| r.get(0)).map_err(|_| "project not found".to_string())?;
        tx.execute_batch("SAVEPOINT artifact_operation").map_err(|e|e.to_string())?;
        let applied = if expected_revision != current_revision { Err("revision conflict".to_string()) } else { apply(&tx,current_revision) };
        let receipt = match applied {
            Ok((new_revision,result)) => {
                tx.execute_batch("RELEASE artifact_operation").map_err(|e|e.to_string())?;
                tx.execute("UPDATE projects SET revision=?2 WHERE id=?1 AND revision=?3", params![project,new_revision,current_revision]).map_err(|e|e.to_string())?;
                ArtifactCommandReceipt { command_id:command_id.into(), project:project.into(), status:"applied".into(), revision:new_revision, result }
            }
            Err(reason) => {
                tx.execute_batch("ROLLBACK TO artifact_operation; RELEASE artifact_operation").map_err(|e|e.to_string())?;
                ArtifactCommandReceipt { command_id:command_id.into(), project:project.into(), status:"rejected".into(), revision:current_revision, result:json!({"reason":reason}) }
            }
        };
        let receipt_text = serde_json::to_string(&receipt).map_err(|e|e.to_string())?;
        tx.execute("INSERT INTO commands(id,project,request,receipt) VALUES(?1,?2,?3,?4)", params![command_id,project,request_text,receipt_text]).map_err(|e|e.to_string())?;
        tx.commit().map_err(|e|e.to_string())?;
        Ok(receipt)
    }

    pub fn register_blob(&self, sha256: &str, bytes: u64, media_type: &str) -> Result<()> {
        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        tx.execute(
            "INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms) VALUES(?1, ?2, ?3, ?4)",
            params![sha256, bytes as i64, media_type, now_ms()],
        ).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn register_artifact(
        &self, project: &str, artifact_id: &str, kind: &str,
        producer_authority: &str, visibility: Option<&str>,
    ) -> Result<()> {
        if producer_authority == "evaluator:trusted" {
            return Err("evaluator authority is reserved for the trusted evaluator API".into());
        }
        self.register_artifact_inner(project, artifact_id, kind, producer_authority, visibility, false)
    }

    #[allow(dead_code)] // Reserved for the in-crate trusted evaluator adapter.
    pub(crate) fn register_evaluator_scorecard(&self, project:&str, artifact_id:&str, visibility:Option<&str>) -> Result<()> {
        self.register_artifact_inner(project, artifact_id, "scorecard", "evaluator:trusted", visibility, true)
    }

    fn register_artifact_inner(
        &self, project:&str, artifact_id:&str, kind:&str, producer_authority:&str,
        visibility:Option<&str>, trusted_evaluator:bool,
    ) -> Result<()> {
        valid_id(project)?;
        valid_id(artifact_id)?;
        if producer_authority == "evaluator:trusted" && !trusted_evaluator {
            return Err("evaluator authority is reserved for the trusted evaluator API".into());
        }
        if trusted_evaluator && (kind != "scorecard" || producer_authority != "evaluator:trusted") {
            return Err("trusted evaluator API only registers official scorecards".into());
        }
        if !matches!(
            kind,
            "data_profile"
                | "research_note"
                | "code"
                | "run_log"
                | "diagnostic_spec"
                | "prediction"
                | "scorecard"
                | "apex_bundle"
                | "checkpoint"
                | "model_weights"
                | "dataset"
        ) {
            return Err(format!("unsupported artifact kind: '{kind}'"));
        }
        let vis = visibility.unwrap_or("project");
        if !matches!(vis, "public" | "project" | "internal") {
            return Err(format!("invalid visibility: '{vis}'"));
        }

        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let now = now_ms();
        tx.execute(
            "INSERT OR IGNORE INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
             VALUES(?1, ?2, ?3, ?4, NULL, 0, ?5, ?6, ?6)",
            params![project, artifact_id, kind, producer_authority, vis, now],
        ).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn publish_version(
        &self, project:&str, artifact_id:&str, blob_sha256:&str, index_digest:&str,
        summary_text:Option<&str>, schema_version:u32, metadata:Option<&Value>, source_job_id:Option<&str>,
    ) -> Result<String> {
        self.publish_version_inner(project,artifact_id,blob_sha256,index_digest,summary_text,schema_version,metadata,source_job_id,false)
    }

    #[allow(dead_code)] // Reserved for the in-crate trusted evaluator adapter.
    pub(crate) fn publish_evaluator_scorecard_version(
        &self, project:&str, artifact_id:&str, blob_sha256:&str, index_digest:&str,
        summary_text:Option<&str>, schema_version:u32, metadata:Option<&Value>, source_job_id:Option<&str>,
    ) -> Result<String> {
        self.publish_version_inner(project,artifact_id,blob_sha256,index_digest,summary_text,schema_version,metadata,source_job_id,true)
    }

    fn publish_version_inner(
        &self, project:&str, artifact_id:&str, blob_sha256:&str, index_digest:&str,
        summary_text:Option<&str>, schema_version:u32, metadata:Option<&Value>, source_job_id:Option<&str>,
        trusted_evaluator:bool,
    ) -> Result<String> {
        valid_id(project)?;
        valid_id(artifact_id)?;
        if index_digest.len() > 512 {
            return Err("index_digest exceeds bound of 512 UTF-8 bytes".into());
        }
        if let Some(s) = summary_text {
            if s.len() > 4096 {
                return Err("summary_text exceeds bound of 4096 UTF-8 bytes".into());
            }
        }

        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;

        // Verify artifact exists
        let kind: String = tx.query_row(
            "SELECT kind FROM artifacts WHERE project=?1 AND artifact_id=?2",
            params![project, artifact_id],
            |r| r.get(0),
        ).map_err(|_| format!("artifact '{artifact_id}' not found in project '{project}'"))?;

        if trusted_evaluator && kind != "scorecard" {
            return Err("trusted evaluator API only publishes official scorecards".into());
        }
        // Scorecard authority check: only trusted evaluator can publish scorecards
        if kind == "scorecard" {
            if !trusted_evaluator {
                return Err("scorecard publication is restricted to the trusted evaluator API".into());
            }
            let auth: String = tx.query_row(
                "SELECT producer_authority FROM artifacts WHERE project=?1 AND artifact_id=?2",
                params![project, artifact_id],
                |r| r.get(0),
            ).unwrap_or_default();
            if auth != "evaluator:trusted" {
                return Err("unauthorized attempt to publish scorecard artifact: requires evaluator:trusted authority".into());
            }
        }

        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let now = now_ms();
        let version_id = format!("{}_{}", artifact_id, tx.query_row("SELECT COALESCE(MAX(rowid),0)+1 FROM artifact_versions", [], |r| r.get::<_, i64>(0)).map_err(|e| e.to_string())?);
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_else(|| "{}".into());
        if meta_str.len() > 4096 { return Err("metadata exceeds 4096 UTF-8 bytes".into()); }

        if self.cas.has_blob(blob_sha256) {
            if let Ok(p) = self.cas.blob_path(blob_sha256) {
                if let Ok(meta) = std::fs::metadata(&p) {
                    let _ = tx.execute(
                        "INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms) VALUES(?1, ?2, ?3, ?4)",
                        params![blob_sha256, meta.len() as i64, "application/octet-stream", now],
                    );
                }
            }
        }

        tx.execute(
            "INSERT INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, source_job_id, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                project,
                version_id,
                artifact_id,
                blob_sha256,
                index_digest,
                summary_text,
                schema_version as i64,
                meta_str,
                source_job_id,
                now,
            ],
        ).map_err(|e| e.to_string())?;

        tx.execute(
            "UPDATE artifacts SET current_version_id=?2, updated_ms=?3 WHERE project=?1 AND artifact_id=?4",
            params![project, version_id, now, artifact_id],
        ).map_err(|e| e.to_string())?;


        tx.commit().map_err(|e| e.to_string())?;
        Ok(version_id)
    }

    pub fn record_lineage(
        &self,
        project: &str,
        source_version_id: &str,
        target_version_id: &str,
        relation: &str,
        metadata: Option<&Value>,
    ) -> Result<()> {
        valid_id(project)?;
        if !matches!(relation, "derived_from" | "evaluates" | "packages" | "supersedes") {
            return Err(format!("unsupported lineage relation: '{relation}'"));
        }

        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let now = now_ms();
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_else(|| "{}".into());
        if meta_str.len() > 4096 { return Err("metadata exceeds 4096 UTF-8 bytes".into()); }

        tx.execute(
            "INSERT INTO artifact_lineage(project, source_version_id, target_version_id, relation, metadata_json, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![project, source_version_id, target_version_id, relation, meta_str, now],
        ).map_err(|e| e.to_string())?;


        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn annotate(
        &self,
        project: &str,
        version_id: &str,
        author: &str,
        note_text: &str,
    ) -> Result<String> {
        valid_id(project)?;
        if note_text.len() > 4096 {
            return Err("note_text exceeds bound of 4096 UTF-8 bytes".into());
        }

        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let now = now_ms();
        let note_seq:i64=tx.query_row("SELECT COALESCE(MAX(rowid),0)+1 FROM artifact_annotations",[],|r|r.get(0)).map_err(|e|e.to_string())?;
        let annotation_id = format!("note_{}_{}", author, note_seq);

        let next_rev: i64 = tx.query_row(
            "SELECT COALESCE(MAX(revision), 0) + 1 FROM artifact_annotations WHERE project=?1 AND version_id=?2",
            params![project, version_id],
            |r| r.get(0),
        ).unwrap_or(1);

        tx.execute(
            "INSERT INTO artifact_annotations(project, annotation_id, version_id, author, revision, note_text, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![project, annotation_id, version_id, author, next_rev, note_text, now],
        ).map_err(|e| e.to_string())?;


        tx.commit().map_err(|e| e.to_string())?;
        Ok(annotation_id)
    }

    pub fn set_apex_selection(
        &self,
        project: &str,
        candidate_version_id: &str,
        scorecard_version_id: &str,
        protocol_sha256: &str,
        rationale: &str,
    ) -> Result<String> {
        valid_id(project)?;
        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let authorized: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM artifact_versions v JOIN artifacts a ON a.project=v.project AND a.artifact_id=v.artifact_id WHERE v.project=?1 AND v.version_id=?2 AND a.kind='scorecard' AND a.producer_authority='evaluator:trusted')",
            params![project, scorecard_version_id], |r| r.get(0),
        ).map_err(|e| e.to_string())?;
        if !authorized { return Err("Apex selection requires an evaluator-authorized scorecard version".into()); }
        let now = now_ms();
        let selection_seq:i64=tx.query_row("SELECT COALESCE(MAX(rowid),0)+1 FROM project_selections",[],|r|r.get(0)).map_err(|e|e.to_string())?;
        let selection_id = format!("apex_{selection_seq}");

        // Previous selections demoted from Apex
        tx.execute(
            "UPDATE project_selections SET is_apex=0 WHERE project=?1",
            params![project],
        ).map_err(|e| e.to_string())?;

        tx.execute(
            "INSERT INTO project_selections(project, selection_id, candidate_version_id, scorecard_version_id, protocol_sha256, rationale, is_apex, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, 1, ?7)",
            params![project, selection_id, candidate_version_id, scorecard_version_id, protocol_sha256, rationale, now],
        ).map_err(|e| e.to_string())?;


        tx.commit().map_err(|e| e.to_string())?;
        Ok(selection_id)
    }

    pub fn pin_artifact(&self, project: &str, artifact_id: &str, pinned: bool) -> Result<()> {
        valid_id(project)?;
        valid_id(artifact_id)?;
        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        crate::execution::enforce_state_write_admission_conn(&tx, 64 * 1024, crate::execution::StorageWriteKind::Checkpoint, Some(project))?;
        let now = now_ms();

        let changed = tx.execute(
            "UPDATE artifacts SET pinned=?3, updated_ms=?4 WHERE project=?1 AND artifact_id=?2",
            params![project, artifact_id, pinned as i64, now],
        ).map_err(|e| e.to_string())?;

        if changed == 0 { return Err("artifact not found".into()); }

        // Dual-write backward compatibility: update managed_artifacts if matching name exists
        tx.execute(
            "UPDATE managed_artifacts SET pinned=?3 WHERE project=?1 AND name=?2",
            params![project, artifact_id, pinned as i64],
        ).map_err(|e| e.to_string())?;


        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn get_artifact(&self, project: &str, artifact_id: &str) -> Result<Option<ArtifactView>> {
        let c = self.read_pool()?.try_read()?;
        let row = c.query_row(
            "SELECT project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms
             FROM artifacts WHERE project=?1 AND artifact_id=?2",
            params![project, artifact_id],
            |r| Ok(ArtifactView {
                project: r.get(0)?,
                artifact_id: r.get(1)?,
                kind: r.get(2)?,
                producer_authority: r.get(3)?,
                current_version_id: r.get(4)?,
                pinned: r.get::<_, i64>(5)? == 1,
                visibility: r.get(6)?,
                created_ms: r.get(7)?,
                updated_ms: r.get(8)?,
            }),
        ).optional().map_err(|e| e.to_string())?;
        bounded_output(row)
    }

    pub fn get_version(&self, project: &str, version_id: &str) -> Result<Option<ArtifactVersionView>> {
        let c = self.read_pool()?.try_read()?;
        let row = c.query_row(
            "SELECT v.project, v.version_id, v.artifact_id, v.blob_sha256, b.bytes, b.media_type, v.index_digest, v.summary_text, v.schema_version, v.metadata_json, v.source_job_id, v.created_ms
             FROM artifact_versions v
             JOIN blobs b ON b.sha256 = v.blob_sha256
             WHERE v.project=?1 AND v.version_id=?2",
            params![project, version_id],
            |r| Ok(ArtifactVersionView {
                project: r.get(0)?,
                version_id: r.get(1)?,
                artifact_id: r.get(2)?,
                blob_sha256: r.get(3)?,
                bytes: r.get(4)?,
                media_type: r.get(5)?,
                index_digest: r.get(6)?,
                summary_text: r.get(7)?,
                schema_version: r.get(8)?,
                metadata: serde_json::from_str(&r.get::<_, String>(9)?).unwrap_or(json!({})),
                source_job_id: r.get(10)?,
                created_ms: r.get(11)?,
            }),
        ).optional().map_err(|e| e.to_string())?;
        bounded_output(row)
    }

    pub fn list_artifacts(&self, project: &str, after_artifact_id: Option<&str>, limit: usize) -> Result<Vec<(ArtifactView, String)>> {
        if !(1..=128).contains(&limit) { return Err("artifact page limit must be 1..=128".into()); }
        let c = self.read_pool()?.try_read()?;
        let mut stmt = c.prepare("SELECT a.project,a.artifact_id,a.kind,a.producer_authority,a.current_version_id,a.pinned,a.visibility,a.created_ms,a.updated_ms,COALESCE(v.index_digest,'') FROM artifacts a LEFT JOIN artifact_versions v ON v.project=a.project AND v.version_id=a.current_version_id WHERE a.project=?1 AND (?2 IS NULL OR a.artifact_id>?2) AND a.tombstoned_ms IS NULL ORDER BY a.artifact_id LIMIT ?3").map_err(|e|e.to_string())?;
        let rows = stmt.query_map(params![project, after_artifact_id, limit as i64], |r| {
            Ok((ArtifactView {
                project: r.get(0)?, artifact_id: r.get(1)?, kind: r.get(2)?,
                producer_authority: r.get(3)?, current_version_id: r.get(4)?,
                pinned: r.get::<_, i64>(5)? == 1, visibility: r.get(6)?,
                created_ms: r.get(7)?, updated_ms: r.get(8)?,
            }, r.get(9)?))
        }).map_err(|e| e.to_string())?;
        bounded_output(rows.map(|r|r.map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?)
    }

    pub fn query_lineage(&self, project: &str, version_id: &str, limit: usize) -> Result<Vec<LineageEdgeView>> {
        if !(1..=128).contains(&limit) { return Err("lineage page limit must be 1..=128".into()); }
        let c=self.read_pool()?.try_read()?;
        let mut stmt=c.prepare("SELECT project,source_version_id,target_version_id,relation,metadata_json,created_ms FROM artifact_lineage WHERE project=?1 AND (source_version_id=?2 OR target_version_id=?2) ORDER BY created_ms,source_version_id,target_version_id,relation LIMIT ?3").map_err(|e|e.to_string())?;
        let rows=stmt.query_map(params![project,version_id,limit as i64],|r|Ok(LineageEdgeView{project:r.get(0)?,source_version_id:r.get(1)?,target_version_id:r.get(2)?,relation:r.get(3)?,metadata:serde_json::from_str(&r.get::<_,String>(4)?).unwrap_or(json!({})),created_ms:r.get(5)?})).map_err(|e|e.to_string())?;
        bounded_output(rows.map(|r|r.map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?)
    }

    pub fn pin(&self, project:&str, artifact_id:&str)->Result<()> { self.pin_artifact(project,artifact_id,true) }
    pub fn unpin(&self, project:&str, artifact_id:&str)->Result<()> { self.pin_artifact(project,artifact_id,false) }

    pub fn list_versions(&self, project: &str, artifact_id: &str) -> Result<Vec<ArtifactVersionView>> {
        let c = self.read_pool()?.try_read()?;
        let mut stmt = c.prepare(
            "SELECT v.project, v.version_id, v.artifact_id, v.blob_sha256, b.bytes, b.media_type, v.index_digest, v.summary_text, v.schema_version, v.metadata_json, v.source_job_id, v.created_ms
             FROM artifact_versions v
             JOIN blobs b ON b.sha256 = v.blob_sha256
             WHERE v.project=?1 AND v.artifact_id=?2
             ORDER BY v.created_ms DESC, v.version_id DESC LIMIT 128",
        ).map_err(|e| e.to_string())?;

        let rows = stmt.query_map(params![project, artifact_id], |r| {
            Ok(ArtifactVersionView {
                project: r.get(0)?,
                version_id: r.get(1)?,
                artifact_id: r.get(2)?,
                blob_sha256: r.get(3)?,
                bytes: r.get(4)?,
                media_type: r.get(5)?,
                index_digest: r.get(6)?,
                summary_text: r.get(7)?,
                schema_version: r.get(8)?,
                metadata: serde_json::from_str(&r.get::<_, String>(9)?).unwrap_or(json!({})),
                source_job_id: r.get(10)?,
                created_ms: r.get(11)?,
            })
        }).map_err(|e| e.to_string())?;

        let mut res = Vec::new();
        for r in rows {
            res.push(r.map_err(|e| e.to_string())?);
        }
        bounded_output(res)
    }

    pub fn get_lineage(&self, project: &str, version_id: &str) -> Result<Vec<LineageEdgeView>> {
        let c = self.read_pool()?.try_read()?;
        let mut stmt = c.prepare(
            "SELECT project, source_version_id, target_version_id, relation, metadata_json, created_ms
             FROM artifact_lineage
             WHERE project=?1 AND (source_version_id=?2 OR target_version_id=?2)
             ORDER BY created_ms ASC, source_version_id, target_version_id, relation LIMIT 128",
        ).map_err(|e| e.to_string())?;

        let rows = stmt.query_map(params![project, version_id], |r| {
            Ok(LineageEdgeView {
                project: r.get(0)?,
                source_version_id: r.get(1)?,
                target_version_id: r.get(2)?,
                relation: r.get(3)?,
                metadata: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or(json!({})),
                created_ms: r.get(5)?,
            })
        }).map_err(|e| e.to_string())?;

        let mut res = Vec::new();
        for r in rows {
            res.push(r.map_err(|e| e.to_string())?);
        }
        bounded_output(res)
    }

    pub fn get_annotations(&self, project: &str, version_id: &str) -> Result<Vec<AnnotationView>> {
        let c = self.read_pool()?.try_read()?;
        let mut stmt = c.prepare(
            "SELECT project, annotation_id, version_id, author, revision, note_text, created_ms
             FROM artifact_annotations
             WHERE project=?1 AND version_id=?2
             ORDER BY revision ASC LIMIT 128",
        ).map_err(|e| e.to_string())?;

        let rows = stmt.query_map(params![project, version_id], |r| {
            Ok(AnnotationView {
                project: r.get(0)?,
                annotation_id: r.get(1)?,
                version_id: r.get(2)?,
                author: r.get(3)?,
                revision: r.get(4)?,
                note_text: r.get(5)?,
                created_ms: r.get(6)?,
            })
        }).map_err(|e| e.to_string())?;

        let mut res = Vec::new();
        for r in rows {
            res.push(r.map_err(|e| e.to_string())?);
        }
        bounded_output(res)
    }

    pub fn get_current_apex(&self, project: &str) -> Result<Option<ProjectSelectionView>> {
        let c = self.read_pool()?.try_read()?;
        let row = c.query_row(
            "SELECT project, selection_id, candidate_version_id, scorecard_version_id, protocol_sha256, rationale, is_apex, created_ms
             FROM project_selections
             WHERE project=?1 AND is_apex=1
             ORDER BY created_ms DESC LIMIT 1",
            params![project],
            |r| Ok(ProjectSelectionView {
                project: r.get(0)?,
                selection_id: r.get(1)?,
                candidate_version_id: r.get(2)?,
                scorecard_version_id: r.get(3)?,
                protocol_sha256: r.get(4)?,
                rationale: r.get(5)?,
                is_apex: r.get::<_, i64>(6)? == 1,
                created_ms: r.get(7)?,
            }),
        ).optional().map_err(|e| e.to_string())?;
        bounded_output(row)
    }

    /// Snapshot and high-water cursor are captured under one short WAL read transaction.
    pub fn subscribe_events(&self, project: &str) -> Result<EventSubscription> {
        let mut c=self.read_pool()?.try_read()?;
        let tx=c.transaction().map_err(|e|e.to_string())?;
        let cursor:i64=tx.query_row("SELECT COALESCE(MAX(cursor),0) FROM durable_events WHERE project=?1",[project],|r|r.get(0)).map_err(|e|e.to_string())?;
        let truncated: bool = tx.query_row("SELECT EXISTS(SELECT 1 FROM artifacts WHERE project=?1 AND tombstoned_ms IS NULL LIMIT 1 OFFSET 128)",[project],|r|r.get(0)).map_err(|e|e.to_string())?;
        let mut stmt=tx.prepare("SELECT a.project,a.artifact_id,a.kind,a.producer_authority,a.current_version_id,a.pinned,a.visibility,a.created_ms,a.updated_ms FROM artifacts a WHERE a.project=?1 AND a.tombstoned_ms IS NULL ORDER BY a.artifact_id LIMIT 128").map_err(|e|e.to_string())?;
        let rows=stmt.query_map([project],|r|Ok(ArtifactView{project:r.get(0)?,artifact_id:r.get(1)?,kind:r.get(2)?,producer_authority:r.get(3)?,current_version_id:r.get(4)?,pinned:r.get::<_,i64>(5)?==1,visibility:r.get(6)?,created_ms:r.get(7)?,updated_ms:r.get(8)?})).map_err(|e|e.to_string())?;
        let artifacts=rows.map(|r|r.map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?;
        drop(stmt);
        tx.commit().map_err(|e|e.to_string())?;
        bounded_output(EventSubscription{project:project.into(),snapshot_cursor:cursor,snapshot_truncated:truncated,artifacts})
    }

    /// Reconnect from a durable cursor. Expired cursors require a fresh snapshot.
    pub fn replay_events(&self, project:&str, after_cursor:u64, limit:usize)->Result<EventReplay>{
        if after_cursor>i64::MAX as u64 || !(1..=128).contains(&limit){return Err("invalid replay cursor/page bound".into());}
        let mut c=self.read_pool()?.try_read()?;
        let tx=c.transaction().map_err(|e|e.to_string())?;
        let (head,expired):(i64,i64)=tx.query_row("SELECT COALESCE((SELECT MAX(cursor) FROM durable_events WHERE project=?1),0),COALESCE((SELECT expired_through_cursor FROM durable_event_checkpoints WHERE project=?1),0)",[project],|r|Ok((r.get(0)?,r.get(1)?))).map_err(|e|e.to_string())?;
        let reset = after_cursor < expired as u64;
        let events = if reset { Vec::new() } else {
            let mut stmt=tx.prepare("SELECT cursor,project,event_type,version,payload_json,committed_ms FROM durable_events WHERE project=?1 AND cursor>?2 AND cursor<=?3 ORDER BY cursor LIMIT ?4").map_err(|e|e.to_string())?;
            let rows=stmt.query_map(params![project,after_cursor as i64,head,limit as i64],|r|Ok(DurableEventView{cursor:r.get(0)?,project:r.get(1)?,event_type:r.get(2)?,version:r.get(3)?,payload:serde_json::from_str(&r.get::<_,String>(4)?).unwrap_or(json!({})),committed_ms:r.get(5)?})).map_err(|e|e.to_string())?;
            rows.map(|r|r.map_err(|e|e.to_string())).collect::<Result<Vec<_>>>()?
        };
        tx.commit().map_err(|e|e.to_string())?;
        bounded_output(EventReplay{reset_required:reset,head_cursor:head,events})
    }

    pub fn get_events(&self, project: &str, after_cursor: u64, limit: usize) -> Result<Vec<DurableEventView>> {
        if after_cursor > i64::MAX as u64 || !(1..=128).contains(&limit) { return Err("invalid replay cursor/page bound".into()); }
        let c = self.read_pool()?.try_read()?;
        let mut stmt = c.prepare(
            "SELECT cursor, project, event_type, version, payload_json, committed_ms
             FROM durable_events
             WHERE project=?1 AND cursor>?2
             ORDER BY cursor ASC LIMIT ?3",
        ).map_err(|e| e.to_string())?;

        let rows = stmt.query_map(params![project, after_cursor as i64, limit as i64], |r| {
            Ok(DurableEventView {
                cursor: r.get(0)?,
                project: r.get(1)?,
                event_type: r.get(2)?,
                version: r.get(3)?,
                payload: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or(json!({})),
                committed_ms: r.get(5)?,
            })
        }).map_err(|e| e.to_string())?;

        let mut res = Vec::new();
        for r in rows {
            res.push(r.map_err(|e| e.to_string())?);
        }
        bounded_output(res)
    }
}
