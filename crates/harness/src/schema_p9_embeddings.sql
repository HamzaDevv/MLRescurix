CREATE TABLE IF NOT EXISTS frozen_embedding_adapters(
    project TEXT NOT NULL REFERENCES projects(id),
    adapter_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    modality TEXT NOT NULL CHECK(modality IN ('text', 'vision', 'multimodal')),
    model_name TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL CHECK(embedding_dim > 0 AND embedding_dim <= 4096),
    pooling_strategy TEXT NOT NULL CHECK(pooling_strategy IN ('mean', 'cls', 'max', 'flatten_project')),
    metadata_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, adapter_id)
);

CREATE TABLE IF NOT EXISTS embedding_cache_records(
    project TEXT NOT NULL REFERENCES projects(id),
    cache_key TEXT NOT NULL,
    adapter_id TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    byte_size INTEGER NOT NULL CHECK(byte_size > 0),
    access_count INTEGER NOT NULL DEFAULT 1,
    created_ms INTEGER NOT NULL,
    last_accessed_ms INTEGER NOT NULL,
    PRIMARY KEY(project, cache_key)
);
