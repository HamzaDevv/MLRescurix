use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::{params, Connection, OptionalExtension, TransactionBehavior};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::{cas::ContentAddressableStorage, Result};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
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
pub struct DurableEventView {
    pub cursor: i64,
    pub project: String,
    pub event_type: String,
    pub version: i64,
    pub payload: Value,
    pub committed_ms: i64,
}

pub struct ArtifactRegistry {
    root: PathBuf,
    cas: ContentAddressableStorage,
}

impl ArtifactRegistry {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let r = root.into();
        let cas = ContentAddressableStorage::new(&r);
        Self { root: r, cas }
    }

    pub fn cas(&self) -> &ContentAddressableStorage {
        &self.cas
    }

    fn conn(&self) -> Result<Connection> {
        crate::execution::connect(&self.root)
    }

    pub fn register_artifact(
        &self,
        project: &str,
        artifact_id: &str,
        kind: &str,
        producer_authority: &str,
        visibility: Option<&str>,
    ) -> Result<()> {
        valid_id(project)?;
        valid_id(artifact_id)?;
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
        &self,
        project: &str,
        artifact_id: &str,
        blob_sha256: &str,
        index_digest: &str,
        summary_text: Option<&str>,
        schema_version: u32,
        metadata: Option<&Value>,
        source_job_id: Option<&str>,
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

        // Scorecard authority check: only trusted evaluator can publish scorecards
        if kind == "scorecard" {
            let auth: String = tx.query_row(
                "SELECT producer_authority FROM artifacts WHERE project=?1 AND artifact_id=?2",
                params![project, artifact_id],
                |r| r.get(0),
            ).unwrap_or_default();
            if auth != "evaluator:trusted" {
                return Err("unauthorized attempt to publish scorecard artifact: requires evaluator:trusted authority".into());
            }
        }

        let now = now_ms();
        let version_id = format!("{}_{}", artifact_id, now);
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_else(|| "{}".into());

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

        let event_payload = json!({
            "artifact_id": artifact_id,
            "version_id": version_id,
            "blob_sha256": blob_sha256,
            "kind": kind,
        }).to_string();

        tx.execute(
            "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
             VALUES(?1, 'artifact.version_published', 1, ?2, ?3)",
            params![project, event_payload, now],
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
        let now = now_ms();
        let meta_str = metadata.map(|m| m.to_string()).unwrap_or_else(|| "{}".into());

        tx.execute(
            "INSERT INTO artifact_lineage(project, source_version_id, target_version_id, relation, metadata_json, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
            params![project, source_version_id, target_version_id, relation, meta_str, now],
        ).map_err(|e| e.to_string())?;

        let event_payload = json!({
            "source_version_id": source_version_id,
            "target_version_id": target_version_id,
            "relation": relation,
        }).to_string();

        tx.execute(
            "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
             VALUES(?1, 'lineage.edge_created', 1, ?2, ?3)",
            params![project, event_payload, now],
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
        let now = now_ms();
        let annotation_id = format!("note_{}_{}", author, now);

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

        let event_payload = json!({
            "annotation_id": annotation_id,
            "version_id": version_id,
            "author": author,
            "revision": next_rev,
        }).to_string();

        tx.execute(
            "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
             VALUES(?1, 'artifact.annotated', 1, ?2, ?3)",
            params![project, event_payload, now],
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
        let now = now_ms();
        let selection_id = format!("apex_{now}");

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

        let event_payload = json!({
            "selection_id": selection_id,
            "candidate_version_id": candidate_version_id,
            "scorecard_version_id": scorecard_version_id,
            "protocol_sha256": protocol_sha256,
        }).to_string();

        tx.execute(
            "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
             VALUES(?1, 'project.apex_crowned', 1, ?2, ?3)",
            params![project, event_payload, now],
        ).map_err(|e| e.to_string())?;

        tx.commit().map_err(|e| e.to_string())?;
        Ok(selection_id)
    }

    pub fn pin_artifact(&self, project: &str, artifact_id: &str, pinned: bool) -> Result<()> {
        valid_id(project)?;
        valid_id(artifact_id)?;
        let mut c = self.conn()?;
        let tx = c.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
        let now = now_ms();

        tx.execute(
            "UPDATE artifacts SET pinned=?3, updated_ms=?4 WHERE project=?1 AND artifact_id=?2",
            params![project, artifact_id, pinned as i64, now],
        ).map_err(|e| e.to_string())?;

        // Dual-write backward compatibility: update managed_artifacts if matching name exists
        let _ = tx.execute(
            "UPDATE managed_artifacts SET pinned=?3 WHERE project=?1 AND name=?2",
            params![project, artifact_id, pinned as i64],
        );

        let event_payload = json!({
            "artifact_id": artifact_id,
            "pinned": pinned,
        }).to_string();

        tx.execute(
            "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
             VALUES(?1, 'artifact.pinned_changed', 1, ?2, ?3)",
            params![project, event_payload, now],
        ).map_err(|e| e.to_string())?;

        tx.commit().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn get_artifact(&self, project: &str, artifact_id: &str) -> Result<Option<ArtifactView>> {
        let c = self.conn()?;
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
        Ok(row)
    }

    pub fn get_version(&self, project: &str, version_id: &str) -> Result<Option<ArtifactVersionView>> {
        let c = self.conn()?;
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
        Ok(row)
    }

    pub fn list_versions(&self, project: &str, artifact_id: &str) -> Result<Vec<ArtifactVersionView>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT v.project, v.version_id, v.artifact_id, v.blob_sha256, b.bytes, b.media_type, v.index_digest, v.summary_text, v.schema_version, v.metadata_json, v.source_job_id, v.created_ms
             FROM artifact_versions v
             JOIN blobs b ON b.sha256 = v.blob_sha256
             WHERE v.project=?1 AND v.artifact_id=?2
             ORDER BY v.created_ms DESC, v.version_id DESC",
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
        Ok(res)
    }

    pub fn get_lineage(&self, project: &str, version_id: &str) -> Result<Vec<LineageEdgeView>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT project, source_version_id, target_version_id, relation, metadata_json, created_ms
             FROM artifact_lineage
             WHERE project=?1 AND (source_version_id=?2 OR target_version_id=?2)
             ORDER BY created_ms ASC",
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
        Ok(res)
    }

    pub fn get_annotations(&self, project: &str, version_id: &str) -> Result<Vec<AnnotationView>> {
        let c = self.conn()?;
        let mut stmt = c.prepare(
            "SELECT project, annotation_id, version_id, author, revision, note_text, created_ms
             FROM artifact_annotations
             WHERE project=?1 AND version_id=?2
             ORDER BY revision ASC",
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
        Ok(res)
    }

    pub fn get_current_apex(&self, project: &str) -> Result<Option<ProjectSelectionView>> {
        let c = self.conn()?;
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
        Ok(row)
    }

    pub fn get_events(&self, project: &str, after_cursor: u64, limit: usize) -> Result<Vec<DurableEventView>> {
        let c = self.conn()?;
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
        Ok(res)
    }
}
