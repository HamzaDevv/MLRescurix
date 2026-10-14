-- MLRescurix v2 Artifact Registry Schema Migration: user_version 9 -> 10

-- 1. Immutable Content-Addressable Storage Blobs
CREATE TABLE IF NOT EXISTS blobs(
  sha256 TEXT PRIMARY KEY CHECK(length(sha256) = 64),
  bytes INTEGER NOT NULL CHECK(bytes >= 0),
  media_type TEXT NOT NULL,
  created_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS blobs_created ON blobs(created_ms);

-- 2. Logical Artifact Entities (Project-Scoped)
CREATE TABLE IF NOT EXISTS artifacts(
  project TEXT NOT NULL REFERENCES projects(id),
  artifact_id TEXT NOT NULL,
  kind TEXT NOT NULL CHECK(kind IN (
    'data_profile',
    'research_note',
    'code',
    'run_log',
    'diagnostic_spec',
    'prediction',
    'scorecard',
    'apex_bundle',
    'checkpoint',
    'model_weights',
    'dataset'
  )),
  producer_authority TEXT NOT NULL,
  current_version_id TEXT,
  pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0, 1)),
  visibility TEXT NOT NULL DEFAULT 'project' CHECK(visibility IN ('public', 'project', 'internal')),
  tombstoned_ms INTEGER,
  created_ms INTEGER NOT NULL,
  updated_ms INTEGER NOT NULL,
  PRIMARY KEY(project, artifact_id)
);
CREATE INDEX IF NOT EXISTS artifacts_project_kind ON artifacts(project, kind, pinned);

-- 3. Immutable Artifact Versions
CREATE TABLE IF NOT EXISTS artifact_versions(
  project TEXT NOT NULL REFERENCES projects(id),
  version_id TEXT NOT NULL,
  artifact_id TEXT NOT NULL,
  blob_sha256 TEXT NOT NULL REFERENCES blobs(sha256),
  index_digest TEXT NOT NULL CHECK(length(index_digest) <= 512),
  summary_text TEXT CHECK(summary_text IS NULL OR length(summary_text) <= 4096),
  schema_version INTEGER NOT NULL DEFAULT 1,
  metadata_json TEXT NOT NULL DEFAULT '{}',
  source_job_id TEXT,
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, version_id),
  FOREIGN KEY(project, artifact_id) REFERENCES artifacts(project, artifact_id)
);
CREATE INDEX IF NOT EXISTS artifact_versions_lookup ON artifact_versions(project, artifact_id, created_ms);
CREATE INDEX IF NOT EXISTS artifact_versions_blob ON artifact_versions(blob_sha256);
CREATE INDEX IF NOT EXISTS artifact_versions_source_job ON artifact_versions(project, source_job_id);

-- 4. Directed Lineage Graph
CREATE TABLE IF NOT EXISTS artifact_lineage(
  project TEXT NOT NULL REFERENCES projects(id),
  source_version_id TEXT NOT NULL,
  target_version_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK(relation IN (
    'derived_from',
    'evaluates',
    'packages',
    'supersedes'
  )),
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, source_version_id, target_version_id, relation),
  FOREIGN KEY(project, source_version_id) REFERENCES artifact_versions(project, version_id),
  FOREIGN KEY(project, target_version_id) REFERENCES artifact_versions(project, version_id)
);
CREATE INDEX IF NOT EXISTS artifact_lineage_target ON artifact_lineage(project, target_version_id, relation);

-- 5. Mutable Versioned Annotations (Notes & Critique)
CREATE TABLE IF NOT EXISTS artifact_annotations(
  project TEXT NOT NULL REFERENCES projects(id),
  annotation_id TEXT NOT NULL,
  version_id TEXT NOT NULL,
  author TEXT NOT NULL,
  revision INTEGER NOT NULL DEFAULT 1,
  note_text TEXT NOT NULL CHECK(length(note_text) <= 4096),
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, annotation_id),
  FOREIGN KEY(project, version_id) REFERENCES artifact_versions(project, version_id)
);
CREATE INDEX IF NOT EXISTS artifact_annotations_version ON artifact_annotations(project, version_id);

-- 6. Project Candidate Selection & Apex History
CREATE TABLE IF NOT EXISTS project_selections(
  project TEXT NOT NULL REFERENCES projects(id),
  selection_id TEXT NOT NULL,
  candidate_version_id TEXT NOT NULL,
  scorecard_version_id TEXT NOT NULL,
  protocol_sha256 TEXT NOT NULL,
  rationale TEXT NOT NULL,
  is_apex INTEGER NOT NULL CHECK(is_apex IN (0, 1)),
  created_ms INTEGER NOT NULL,
  PRIMARY KEY(project, selection_id),
  FOREIGN KEY(project, candidate_version_id) REFERENCES artifact_versions(project, version_id),
  FOREIGN KEY(project, scorecard_version_id) REFERENCES artifact_versions(project, version_id)
);
CREATE INDEX IF NOT EXISTS project_selections_apex ON project_selections(project, is_apex, created_ms);

-- 7. Versioned Durable Event Journal
CREATE TABLE IF NOT EXISTS durable_events(
  cursor INTEGER PRIMARY KEY AUTOINCREMENT,
  project TEXT NOT NULL REFERENCES projects(id),
  event_type TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  payload_json TEXT NOT NULL,
  committed_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS durable_events_stream ON durable_events(project, cursor);

-- Backfill existing legacy managed_artifacts into blobs, artifacts, and artifact_versions
INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms)
  SELECT sha256, bytes, 'application/octet-stream', CAST(strftime('%s','now') AS INTEGER) * 1000
  FROM managed_artifacts
  WHERE sha256 IS NOT NULL AND length(sha256) = 64;

INSERT OR IGNORE INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
  SELECT project, name, 'code', 'system:legacy_managed', NULL, pinned, 'project', CAST(strftime('%s','now') AS INTEGER) * 1000, CAST(strftime('%s','now') AS INTEGER) * 1000
  FROM managed_artifacts;

INSERT OR IGNORE INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, source_job_id, created_ms)
  SELECT m.project,
         m.job || '_' || replace(replace(m.name, '/', '_'), '.', '_'),
         m.name,
         m.sha256,
         substr('[' || m.name || '] legacy artifact from job ' || m.job || ' sha:' || substr(m.sha256, 1, 8), 1, 512),
         NULL,
         1,
         '{}',
         m.job,
         CAST(strftime('%s','now') AS INTEGER) * 1000
  FROM managed_artifacts m
  WHERE m.sha256 IS NOT NULL AND length(m.sha256) = 64;

-- Backfill scientific_datasets into blobs, artifacts, and artifact_versions
INSERT OR IGNORE INTO blobs(sha256, bytes, media_type, created_ms)
  SELECT sha256, bytes, 'text/csv', registered_ms
  FROM scientific_datasets
  WHERE sha256 IS NOT NULL AND length(sha256) = 64;

INSERT OR IGNORE INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
  SELECT project, id, 'dataset', 'system:registered_dataset', NULL, 1, 'project', registered_ms, registered_ms
  FROM scientific_datasets;

INSERT OR IGNORE INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, source_job_id, created_ms)
  SELECT project,
         'dataset_' || id || '_v1',
         id,
         sha256,
         substr('[' || id || '] registered scientific dataset (' || bytes || ' B, sha:' || substr(sha256, 1, 8) || ')', 1, 512),
         NULL,
         1,
         '{}',
         NULL,
         registered_ms
  FROM scientific_datasets
  WHERE sha256 IS NOT NULL AND length(sha256) = 64;

UPDATE artifacts SET current_version_id = (
  SELECT version_id FROM artifact_versions v
  WHERE v.project = artifacts.project AND v.artifact_id = artifacts.artifact_id
  ORDER BY created_ms DESC, version_id DESC LIMIT 1
) WHERE current_version_id IS NULL;

PRAGMA user_version = 10;
