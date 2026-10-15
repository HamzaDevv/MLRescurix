CREATE TABLE IF NOT EXISTS arrow_dataset_adapters(
    project TEXT NOT NULL REFERENCES projects(id),
    dataset_id TEXT NOT NULL,
    format TEXT NOT NULL CHECK(format IN ('ipc_stream', 'ipc_file', 'feather', 'parquet')),
    path TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    file_bytes INTEGER NOT NULL,
    metadata_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, dataset_id)
);

CREATE TABLE IF NOT EXISTS time_series_backtests(
    project TEXT NOT NULL REFERENCES projects(id),
    backtest_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    timestamp_column TEXT NOT NULL,
    target_column TEXT NOT NULL,
    split_protocol TEXT NOT NULL,
    feature_config_json TEXT NOT NULL,
    baseline_metrics_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    PRIMARY KEY(project, backtest_id)
);
