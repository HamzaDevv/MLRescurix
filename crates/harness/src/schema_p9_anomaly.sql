CREATE TABLE IF NOT EXISTS anomaly_clustering_adapters(
    project TEXT NOT NULL REFERENCES projects(id),
    adapter_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    adapter_type TEXT NOT NULL CHECK(adapter_type IN ('anomaly_detection', 'clustering', 'combined')),
    contamination REAL NOT NULL CHECK(contamination > 0.0 AND contamination < 0.5),
    clustering_algorithm TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, adapter_id)
);
