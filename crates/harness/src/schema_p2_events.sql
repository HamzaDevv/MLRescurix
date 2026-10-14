CREATE TABLE IF NOT EXISTS durable_event_checkpoints(project TEXT PRIMARY KEY REFERENCES projects(id), expired_through_cursor INTEGER NOT NULL DEFAULT 0);
CREATE TRIGGER IF NOT EXISTS p2_event_cursor_checkpoint BEFORE DELETE ON durable_events BEGIN
 INSERT INTO durable_event_checkpoints(project,expired_through_cursor) VALUES(OLD.project,OLD.cursor)
 ON CONFLICT(project) DO UPDATE SET expired_through_cursor=MAX(expired_through_cursor,excluded.expired_through_cursor);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_artifact AFTER INSERT ON artifacts BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'artifact.created',1,json_object('artifact_id',NEW.artifact_id,'kind',NEW.kind),NEW.created_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_version AFTER INSERT ON artifact_versions BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms)
 SELECT NEW.project,'artifact.version_published',1,json_object('artifact_id',NEW.artifact_id,'version_id',NEW.version_id,'blob_sha256',NEW.blob_sha256),NEW.created_ms;
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_lineage AFTER INSERT ON artifact_lineage BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'lineage.edge_created',1,json_object('source_version_id',NEW.source_version_id,'target_version_id',NEW.target_version_id,'relation',NEW.relation),NEW.created_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_annotation AFTER INSERT ON artifact_annotations BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'artifact.annotated',1,json_object('annotation_id',NEW.annotation_id,'version_id',NEW.version_id,'author',NEW.author,'revision',NEW.revision),NEW.created_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_selection AFTER INSERT ON project_selections BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'project.apex_crowned',1,json_object('selection_id',NEW.selection_id,'candidate_version_id',NEW.candidate_version_id,'scorecard_version_id',NEW.scorecard_version_id,'protocol_sha256',NEW.protocol_sha256),NEW.created_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_pin AFTER UPDATE OF pinned ON artifacts WHEN OLD.pinned != NEW.pinned BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'artifact.pinned_changed',1,json_object('artifact_id',NEW.artifact_id,'pinned',NEW.pinned),NEW.updated_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_event_artifact_tombstone AFTER UPDATE OF tombstoned_ms ON artifacts WHEN OLD.tombstoned_ms IS NULL AND NEW.tombstoned_ms IS NOT NULL BEGIN
 INSERT INTO durable_events(project,event_type,version,payload_json,committed_ms) VALUES(NEW.project,'artifact.tombstoned',1,json_object('artifact_id',NEW.artifact_id,'tombstoned_ms',NEW.tombstoned_ms),NEW.tombstoned_ms);
 DELETE FROM durable_events WHERE project=NEW.project AND cursor <= (SELECT COALESCE(MAX(cursor),0)-10000 FROM durable_events WHERE project=NEW.project);
END;
CREATE TRIGGER IF NOT EXISTS p2_version_metadata_bound BEFORE INSERT ON artifact_versions WHEN length(NEW.metadata_json)>4096 BEGIN SELECT RAISE(ABORT,'artifact metadata exceeds 4096 bytes'); END;
CREATE TRIGGER IF NOT EXISTS p2_lineage_metadata_bound BEFORE INSERT ON artifact_lineage WHEN length(NEW.metadata_json)>4096 BEGIN SELECT RAISE(ABORT,'lineage metadata exceeds 4096 bytes'); END;

CREATE TRIGGER IF NOT EXISTS p2_version_summary_byte_bound BEFORE INSERT ON artifact_versions WHEN length(CAST(NEW.index_digest AS BLOB))>512 OR (NEW.summary_text IS NOT NULL AND length(CAST(NEW.summary_text AS BLOB))>4096) BEGIN SELECT RAISE(ABORT,'artifact summary exceeds UTF-8 byte bound'); END;
CREATE TRIGGER IF NOT EXISTS p2_version_summary_update_byte_bound BEFORE UPDATE OF index_digest,summary_text ON artifact_versions WHEN length(CAST(NEW.index_digest AS BLOB))>512 OR (NEW.summary_text IS NOT NULL AND length(CAST(NEW.summary_text AS BLOB))>4096) BEGIN SELECT RAISE(ABORT,'artifact summary exceeds UTF-8 byte bound'); END;
CREATE TRIGGER IF NOT EXISTS p2_annotation_byte_bound BEFORE INSERT ON artifact_annotations WHEN length(CAST(NEW.note_text AS BLOB))>4096 BEGIN SELECT RAISE(ABORT,'annotation exceeds UTF-8 byte bound'); END;
CREATE TRIGGER IF NOT EXISTS p2_selection_rationale_bound BEFORE INSERT ON project_selections WHEN length(CAST(NEW.rationale AS BLOB))>4096 BEGIN SELECT RAISE(ABORT,'selection rationale exceeds 4096 bytes'); END;
CREATE TRIGGER IF NOT EXISTS p2_event_payload_bound BEFORE INSERT ON durable_events WHEN length(CAST(NEW.payload_json AS BLOB))>4096 BEGIN SELECT RAISE(ABORT,'durable event payload exceeds 4096 bytes'); END;
