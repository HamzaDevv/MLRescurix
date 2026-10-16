//! Phase P9 Task Adapters (P9.1 Arrow Out-of-Core & P9.2 Time-Series Forecasting) Integration Tests.
//!
//! Verifies:
//! - P9.1: Strict typed ArrowDatasetAdapter, memory mapping, batch limits, empirical memory bounds.
//! - P9.2: Strict typed TimeSeriesDatasetAdapter, temporal splits (rolling/expanding/entity-aware),
//!   causal lags, naive/seasonal baselines, and delivery contracts.
//! - Scientific negative cases: memory cap overflow, future lookahead temporal leakage, out-of-order timestamps.
//! - Database persistence under schema v10 (`PRAGMA user_version = 10`).
//! - Agent orchestrator tool dispatch (`inspect_arrow_dataset`, `evaluate_temporal_backtest`).
//! - CLI subcommands (`inspect-arrow`, `validate-temporal-split`) and Capabilities reporting.
//! - Full Acceptance Gate closure for P9.1 and P9.2 slices.

use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};

use mlrescurix_harness::{
    adapters::{
        get_anomaly_clustering_adapter, get_arrow_adapter, get_frozen_embedding_adapter,
        get_time_series_backtest, record_anomaly_clustering_adapter, record_arrow_adapter,
        record_embedding_cache_record, record_frozen_embedding_adapter,
        record_time_series_backtest, validate_anomaly_clustering_config,
        validate_arrow_adapter_memory_bounds, validate_frozen_embedding_config,
        validate_temporal_split_integrity, AnomalyClusteringAdapter, ArrowDatasetAdapter,
        EmbeddingCacheRecord, FrozenEmbeddingAdapter, TemporalSplitProtocol,
        TimeSeriesDatasetAdapter,
    },
    agents::{
        gateway::{ProviderGateway, ScriptedProviderFixture},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
    },
    core::{Operation, Policy, Request},
    execution,
    Runtime,
};

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    b.insert("cost_millicents".into(), limit * 50);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy {
        max_tasks: 1024,
        budget: b,
        capacity: c,
    }
}

fn test_context(base: &str, code: &str) -> mlrescurix_harness::contracts::Context {
    mlrescurix_harness::contracts::Context {
        role_id: "lead".into(),
        role_version: "1".into(),
        prompt_hash: "prompt-hash".into(),
        model: None,
        phase: "research".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "inspect".into(),
        stop_conditions: vec!["budget".into()],
    }
}

struct TestHarness {
    root: PathBuf,
}

impl TestHarness {
    fn new(prefix: &str) -> Self {
        let root = std::env::temp_dir().join(format!("mlrescurix_p9_{prefix}_{}", now_nanos()));
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        Runtime::init(&root, test_policy(100)).unwrap();
        execution::migrate(&root).unwrap();
        Self { root }
    }

    fn create_project(&self, project: &str) {
        let r = Runtime::open(&self.root).unwrap();
        let _ = r.command(Request {
            command_id: format!("init-{project}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateProject {
                intent: "P9 Adapter Test".into(),
                finalization_id: format!("final-{project}"),
                policy: test_policy(100),
            },
        }).unwrap();
    }

    fn setup_orchestrator(&self, project: &str, task_id: &str) -> AgentOrchestrator {
        self.create_project(project);
        let r = Runtime::open(&self.root).unwrap();
        let branch = format!("b-{task_id}");

        let _ = r.command(Request {
            command_id: format!("branch-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateBranch {
                branch_id: branch.clone(),
                base_hash: "base-p9".into(),
                code_hash: "code-p9".into(),
            },
        }).unwrap();

        let _ = r.command(Request {
            command_id: format!("task-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateTask {
                task_id: task_id.into(),
                branch_id: branch.clone(),
                context: test_context("base-p9", "code-p9"),
            },
        }).unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(i64::MAX as u128) as i64;
        let _ = r.command(Request {
            command_id: format!("claim-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::Claim {
                task_id: task_id.into(),
                owner: format!("agent-{task_id}"),
                lease_id: format!("lease-{task_id}"),
                now_ms: now,
                expires_ms: now + 300_000,
                expected_generation: 0,
            },
        }).unwrap();

        let config = OrchestratorConfig {
            project_id: project.to_string(),
            task_id: task_id.to_string(),
            role_id: "lead".to_string(),
            branch_id: branch,
            model: "test-model".to_string(),
            max_turns: 5,
        };

        let gateway: Arc<dyn ProviderGateway> = Arc::new(ScriptedProviderFixture::new(vec![]));
        AgentOrchestrator::new(&self.root, config, gateway).unwrap()
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        if self.root.exists() {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
}

#[test]
fn test_p9_1_arrow_adapter_schema_models_and_json_roundtrip() {
    let adapter = ArrowDatasetAdapter::synthetic_parquet("ds_sensor_readings", 20000, 30, "failure");
    assert_eq!(adapter.schema_version, "v2");
    assert_eq!(adapter.format, "parquet");
    assert_eq!(adapter.row_count, 20000);
    assert_eq!(adapter.column_count, 31);
    assert_eq!(adapter.columns.len(), 31);
    assert_eq!(adapter.batch_config.batch_size_rows, 1000);
    assert_eq!(adapter.batch_config.memory_cap_bytes, 64 * 1024 * 1024);

    let serialized = serde_json::to_string_pretty(&adapter).unwrap();
    let deserialized: ArrowDatasetAdapter = serde_json::from_str(&serialized).unwrap();
    assert_eq!(adapter, deserialized);
}

#[test]
fn test_p9_1_arrow_adapter_db_persistence_and_query() {
    let harness = TestHarness::new("arrow_db");
    harness.create_project("proj_arrow");

    let adapter = ArrowDatasetAdapter::synthetic_parquet("ds_flight_data", 50000, 10, "delayed");
    record_arrow_adapter(&harness.root, "proj_arrow", &adapter).unwrap();

    let fetched = get_arrow_adapter(&harness.root, "proj_arrow", "ds_flight_data").unwrap();
    assert_eq!(fetched.dataset_id, "ds_flight_data");
    assert_eq!(fetched.row_count, 50000);
    assert_eq!(fetched.column_count, 11);
    assert_eq!(fetched.format, "parquet");

    // Missing dataset error
    let missing_err = get_arrow_adapter(&harness.root, "proj_arrow", "non_existent_dataset");
    assert!(missing_err.is_err());
    assert!(missing_err.unwrap_err().contains("not found"));
}

#[test]
fn test_p9_1_arrow_adapter_memory_cap_enforcement() {
    let mut adapter = ArrowDatasetAdapter::synthetic_parquet("ds_capped", 10000, 15, "label");
    // Under allowed memory limit
    assert!(validate_arrow_adapter_memory_bounds(&adapter, 128 * 1024 * 1024).is_ok());

    // Case 1: Configured memory_cap_bytes exceeds maximum allowed limit
    let err_cap = validate_arrow_adapter_memory_bounds(&adapter, 32 * 1024 * 1024);
    assert!(err_cap.is_err());
    assert!(err_cap.unwrap_err().contains("exceeds maximum allowed memory limit"));

    // Case 2: Observed peak memory exceeds configured cap
    adapter.memory_accounting.peak_memory_bytes = 128 * 1024 * 1024;
    let err_peak = validate_arrow_adapter_memory_bounds(&adapter, 512 * 1024 * 1024);
    assert!(err_peak.is_err());
    assert!(err_peak.unwrap_err().contains("exceeded configured memory cap"));
}

#[test]
fn test_p9_2_time_series_adapter_schema_models_and_json_roundtrip() {
    let adapter = TimeSeriesDatasetAdapter::synthetic_daily("ds_energy_demand");
    assert_eq!(adapter.schema_version, "v2");
    assert_eq!(adapter.timestamp_column, "timestamp");
    assert_eq!(adapter.target_column, "demand");
    assert_eq!(adapter.split_protocol.protocol_type, "rolling_window");
    assert_eq!(adapter.split_protocol.gap_steps, 1);
    assert_eq!(adapter.split_protocol.test_horizon_steps, 14);
    assert!(adapter.split_protocol.purge_lookahead_verified);
    assert_eq!(adapter.feature_config.lags, vec![1, 2, 7, 14]);
    assert_eq!(adapter.baseline_evaluations.len(), 2);
    assert_eq!(adapter.inference_delivery.horizon_steps, 14);
    assert!(adapter.inference_delivery.strict_future_lookahead_rejection);

    let serialized = serde_json::to_string_pretty(&adapter).unwrap();
    let deserialized: TimeSeriesDatasetAdapter = serde_json::from_str(&serialized).unwrap();
    assert_eq!(adapter, deserialized);
}

#[test]
fn test_p9_2_time_series_db_persistence_and_query() {
    let harness = TestHarness::new("ts_db");
    harness.create_project("proj_ts");

    let adapter = TimeSeriesDatasetAdapter::synthetic_daily("ds_retail_store");
    record_time_series_backtest(&harness.root, "proj_ts", "backtest_01", &adapter).unwrap();

    let fetched = get_time_series_backtest(&harness.root, "proj_ts", "backtest_01").unwrap();
    assert_eq!(fetched.dataset_id, "ds_retail_store");
    assert_eq!(fetched.target_column, "demand");
    assert_eq!(fetched.split_protocol.protocol_type, "rolling_window");
    assert_eq!(fetched.feature_config.lags, vec![1, 2, 7, 14]);

    let missing_err = get_time_series_backtest(&harness.root, "proj_ts", "missing_backtest");
    assert!(missing_err.is_err());
    assert!(missing_err.unwrap_err().contains("not found"));
}

#[test]
fn test_p9_2_temporal_split_integrity_and_leakage_rejection() {
    let protocol = TemporalSplitProtocol {
        protocol_type: "rolling_window".into(),
        train_window_steps: Some(10),
        gap_steps: 2,
        test_horizon_steps: 5,
        step_stride: 5,
        n_splits: 1,
        purge_lookahead_verified: true,
    };

    // Valid causal split with gap
    let train_ts: Vec<i64> = (100..110).collect();
    let test_ts_clean: Vec<i64> = (112..117).collect(); // 110 max train, 112 min test -> gap of 2
    assert!(validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_clean).is_ok());

    // Leakage case: test timestamp strictly inside or before train
    let test_ts_leaked: Vec<i64> = vec![105, 106, 107, 108, 109];
    let leak_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_leaked);
    assert!(leak_res.is_err());
    assert!(leak_res.unwrap_err().contains("Future lookahead leakage detected"));

    // Overlapping boundary: min test == max train
    let test_ts_boundary: Vec<i64> = vec![109, 110, 111];
    let boundary_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_boundary);
    assert!(boundary_res.is_err());
    assert!(boundary_res.unwrap_err().contains("Future lookahead leakage detected"));
}

#[test]
fn test_p9_2_temporal_split_out_of_order_rejection() {
    let protocol = TemporalSplitProtocol {
        protocol_type: "expanding_window".into(),
        train_window_steps: Some(5),
        gap_steps: 1,
        test_horizon_steps: 3,
        step_stride: 3,
        n_splits: 1,
        purge_lookahead_verified: true,
    };

    // Unordered train timestamps
    let train_ts_unordered: Vec<i64> = vec![1, 2, 5, 4, 6];
    let test_ts_clean: Vec<i64> = vec![8, 9, 10];
    let err_train = validate_temporal_split_integrity(&protocol, &train_ts_unordered, &test_ts_clean);
    assert!(err_train.is_err());
    assert!(err_train.unwrap_err().contains("out of order"));

    // Unordered test timestamps
    let train_ts_clean: Vec<i64> = vec![1, 2, 3, 4, 5];
    let test_ts_unordered: Vec<i64> = vec![10, 8, 9];
    let err_test = validate_temporal_split_integrity(&protocol, &train_ts_clean, &test_ts_unordered);
    assert!(err_test.is_err());
    assert!(err_test.unwrap_err().contains("out of order"));
}

#[test]
fn test_p9_orchestrator_tool_dispatch_arrow_and_time_series() {
    let harness = TestHarness::new("orch_tools");
    let mut orchestrator = harness.setup_orchestrator("proj_orch", "task_adapters");

    // 1. Dispatch inspect_arrow_dataset
    let arrow_args = json!({
        "dataset_id": "arrow_orch_ds",
        "path": "/data/telemetry.parquet",
        "format": "parquet",
        "target_column": "status",
        "batch_size_rows": 2000,
        "memory_cap_bytes": 134217728
    });
    let arrow_res = orchestrator.dispatch_tool_call("inspect_arrow_dataset", &arrow_args.to_string()).unwrap();
    let arrow_val: Value = serde_json::from_str(&arrow_res).unwrap();
    assert_eq!(arrow_val["dataset_id"], "arrow_orch_ds");
    assert_eq!(arrow_val["format"], "parquet");
    assert_eq!(arrow_val["batch_config"]["batch_size_rows"], 2000);

    // Confirm stored in SQLite
    let fetched_arrow = get_arrow_adapter(&harness.root, "proj_orch", "arrow_orch_ds").unwrap();
    assert_eq!(fetched_arrow.dataset_id, "arrow_orch_ds");

    // 2. Dispatch evaluate_temporal_backtest
    let ts_args = json!({
        "dataset_id": "ts_orch_ds",
        "timestamp_column": "measured_at",
        "target_column": "reading",
        "horizon_steps": 21,
        "backtest_id": "backtest_orch_01"
    });
    let ts_res = orchestrator.dispatch_tool_call("evaluate_temporal_backtest", &ts_args.to_string()).unwrap();
    let ts_val: Value = serde_json::from_str(&ts_res).unwrap();
    assert_eq!(ts_val["dataset_id"], "ts_orch_ds");
    assert_eq!(ts_val["timestamp_column"], "measured_at");
    assert_eq!(ts_val["target_column"], "reading");
    assert_eq!(ts_val["split_protocol"]["test_horizon_steps"], 21);

    // Confirm stored in SQLite
    let fetched_ts = get_time_series_backtest(&harness.root, "proj_orch", "backtest_orch_01").unwrap();
    assert_eq!(fetched_ts.dataset_id, "ts_orch_ds");

    // 3. Dispatch evaluate_anomaly_clustering
    let ac_args = json!({
        "dataset_id": "ac_orch_ds",
        "adapter_id": "adapter_orch_01",
        "adapter_type": "combined",
        "contamination": 0.08,
        "clustering_algorithm": "kmeans",
        "n_clusters": 4,
        "metric": "euclidean"
    });
    let ac_res = orchestrator.dispatch_tool_call("evaluate_anomaly_clustering", &ac_args.to_string()).unwrap();
    let ac_val: Value = serde_json::from_str(&ac_res).unwrap();
    assert_eq!(ac_val["dataset_id"], "ac_orch_ds");
    assert_eq!(ac_val["adapter_id"], "adapter_orch_01");
    assert_eq!(ac_val["contamination_prior"], 0.08);
    assert_eq!(ac_val["clustering_protocol"]["n_clusters"], 4);

    // Confirm stored in SQLite
    let fetched_ac = get_anomaly_clustering_adapter(&harness.root, "proj_orch", "adapter_orch_01").unwrap();
    assert_eq!(fetched_ac.dataset_id, "ac_orch_ds");
    assert_eq!(fetched_ac.contamination_prior, 0.08);

    // Negative case: reject contamination >= 0.5 in orchestrator dispatch
    let ac_bad_args = json!({
        "dataset_id": "ac_orch_bad",
        "contamination": 0.6
    });
    let ac_err = orchestrator.dispatch_tool_call("evaluate_anomaly_clustering", &ac_bad_args.to_string());
    assert!(ac_err.is_err());
    assert!(ac_err.unwrap_err().contains("Contamination prior must be strictly in (0.0, 0.5)"));

    // 4. Dispatch extract_frozen_embeddings (text modality)
    let emb_args = json!({
        "dataset_id": "text_orch_ds",
        "adapter_id": "adapter_text_emb_01",
        "modality": "text",
        "model_name": "frozen-transformer-text-v1",
        "embedding_dim": 64,
        "pooling_strategy": "cls",
        "cache_enabled": true
    });
    let emb_res = orchestrator.dispatch_tool_call("extract_frozen_embeddings", &emb_args.to_string()).unwrap();
    let emb_val: Value = serde_json::from_str(&emb_res).unwrap();
    assert_eq!(emb_val["dataset_id"], "text_orch_ds");
    assert_eq!(emb_val["adapter_id"], "adapter_text_emb_01");
    assert_eq!(emb_val["dimension_fingerprint"]["embedding_dim"], 64);
    assert_eq!(emb_val["dimension_fingerprint"]["pooling_strategy"], "cls");

    // Confirm stored in SQLite
    let fetched_emb = get_frozen_embedding_adapter(&harness.root, "proj_orch", "adapter_text_emb_01").unwrap();
    assert_eq!(fetched_emb.dataset_id, "text_orch_ds");
    assert_eq!(fetched_emb.dimension_fingerprint.embedding_dim, 64);

    // Negative case: reject uncertified remote download in orchestrator dispatch
    let emb_bad_args = json!({
        "dataset_id": "text_orch_bad",
        "remote_download": true
    });
    let emb_err = orchestrator.dispatch_tool_call("extract_frozen_embeddings", &emb_bad_args.to_string());
    assert!(emb_err.is_err());
    assert!(emb_err.unwrap_err().contains("Uncertified remote model download requests are strictly rejected"));
}

#[test]
fn test_p9_4_frozen_embeddings_adapter_models_and_json_roundtrip() {
    let text_adapter = FrozenEmbeddingAdapter::synthetic_text("ds_txt_roundtrip", 64);
    assert_eq!(text_adapter.schema_version, "v2");
    assert_eq!(text_adapter.modality, "text");
    assert!(validate_frozen_embedding_config(&text_adapter).is_ok());

    let text_json = serde_json::to_string_pretty(&text_adapter).unwrap();
    let text_deser: FrozenEmbeddingAdapter = serde_json::from_str(&text_json).unwrap();
    assert_eq!(text_adapter, text_deser);

    let vis_adapter = FrozenEmbeddingAdapter::synthetic_vision("ds_vis_roundtrip", 128);
    assert_eq!(vis_adapter.modality, "vision");
    assert!(validate_frozen_embedding_config(&vis_adapter).is_ok());

    let vis_json = serde_json::to_string_pretty(&vis_adapter).unwrap();
    let vis_deser: FrozenEmbeddingAdapter = serde_json::from_str(&vis_json).unwrap();
    assert_eq!(vis_adapter, vis_deser);

    // EmbeddingCacheRecord roundtrip
    let cache_rec = EmbeddingCacheRecord {
        project: "proj_rt".into(),
        cache_key: "key_123".into(),
        adapter_id: "adapter_text_emb_01".into(),
        input_sha256: "sha256_deadbeef".into(),
        byte_size: 256,
        access_count: 5,
        created_ms: 1000,
        last_accessed_ms: 1050,
    };
    let rec_json = serde_json::to_string(&cache_rec).unwrap();
    let rec_deser: EmbeddingCacheRecord = serde_json::from_str(&rec_json).unwrap();
    assert_eq!(cache_rec, rec_deser);
}

#[test]
fn test_p9_3_anomaly_clustering_adapter_models_and_json_roundtrip() {
    let adapter = AnomalyClusteringAdapter::synthetic("ds_anomaly_roundtrip");
    assert_eq!(adapter.schema_version, "v2");
    assert_eq!(adapter.contamination_prior, 0.05);
    assert!(validate_anomaly_clustering_config(&adapter).is_ok());

    // JSON round-trip
    let serialized = serde_json::to_string_pretty(&adapter).unwrap();
    let deserialized: AnomalyClusteringAdapter = serde_json::from_str(&serialized).unwrap();
    assert_eq!(adapter, deserialized);
}

#[test]
fn test_p9_cli_commands_and_capabilities() {
    let harness = TestHarness::new("cli_test");
    harness.create_project("proj_cli");

    // Seed an arrow adapter, time series backtest, and anomaly clustering adapter
    let arrow_adapter = ArrowDatasetAdapter::synthetic_parquet("cli_arrow_ds", 15000, 10, "target");
    record_arrow_adapter(&harness.root, "proj_cli", &arrow_adapter).unwrap();

    let ts_adapter = TimeSeriesDatasetAdapter::synthetic_daily("cli_ts_ds");
    record_time_series_backtest(&harness.root, "proj_cli", "cli_bt_01", &ts_adapter).unwrap();

    let ac_adapter = AnomalyClusteringAdapter::synthetic("cli_ac_ds");
    record_anomaly_clustering_adapter(&harness.root, "proj_cli", &ac_adapter).unwrap();

    let emb_adapter = FrozenEmbeddingAdapter::synthetic_text("cli_emb_ds", 64);
    record_frozen_embedding_adapter(&harness.root, "proj_cli", &emb_adapter).unwrap();

    let exe = PathBuf::from(env!("CARGO_BIN_EXE_mlrescurix-harness"));

    // 1. InspectArrow CLI
    let inspect_out = Command::new(&exe)
        .args([
            "inspect-arrow",
            "--state", harness.root.to_str().unwrap(),
            "--project", "proj_cli",
            "--dataset", "cli_arrow_ds",
        ])
        .output()
        .unwrap();
    assert!(inspect_out.status.success());
    let inspect_val: Value = serde_json::from_slice(&inspect_out.stdout).unwrap();
    assert_eq!(inspect_val["dataset_id"], "cli_arrow_ds");

    // 2. ValidateTemporalSplit CLI
    let validate_out = Command::new(&exe)
        .args([
            "validate-temporal-split",
            "--state", harness.root.to_str().unwrap(),
            "--project", "proj_cli",
            "--backtest", "cli_bt_01",
        ])
        .output()
        .unwrap();
    assert!(validate_out.status.success());
    let validate_val: Value = serde_json::from_slice(&validate_out.stdout).unwrap();
    assert_eq!(validate_val["dataset_id"], "cli_ts_ds");

    // 3. InspectAnomalyClustering CLI
    let inspect_ac_out = Command::new(&exe)
        .args([
            "inspect-anomaly-clustering",
            "--state", harness.root.to_str().unwrap(),
            "--project", "proj_cli",
            "--adapter", "adapter_cli_ac_ds",
        ])
        .output()
        .unwrap();
    assert!(inspect_ac_out.status.success());
    let inspect_ac_val: Value = serde_json::from_slice(&inspect_ac_out.stdout).unwrap();
    assert_eq!(inspect_ac_val["dataset_id"], "cli_ac_ds");
    assert_eq!(inspect_ac_val["adapter_id"], "adapter_cli_ac_ds");

    // 4. InspectFrozenEmbeddings CLI
    let inspect_emb_out = Command::new(&exe)
        .args([
            "inspect-frozen-embeddings",
            "--state", harness.root.to_str().unwrap(),
            "--project", "proj_cli",
            "--adapter", "text_emb_cli_emb_ds",
        ])
        .output()
        .unwrap();
    assert!(inspect_emb_out.status.success());
    let inspect_emb_val: Value = serde_json::from_slice(&inspect_emb_out.stdout).unwrap();
    assert_eq!(inspect_emb_val["dataset_id"], "cli_emb_ds");
    assert_eq!(inspect_emb_val["adapter_id"], "text_emb_cli_emb_ds");
    assert_eq!(inspect_emb_val["dimension_fingerprint"]["embedding_dim"], 64);

    // 5. Capabilities CLI
    let caps_out = Command::new(&exe).arg("capabilities").output().unwrap();
    let caps_val: Value = serde_json::from_slice(&caps_out.stdout).unwrap();
    assert_eq!(caps_val["phase"], "P7");
    assert_eq!(caps_val["v2_release_gates"]["p9_adapters"], "passed");
    assert_eq!(caps_val["adapter_capabilities"]["arrow_ipc"], true);
    assert_eq!(caps_val["adapter_capabilities"]["parquet"], true);
    assert_eq!(caps_val["adapter_capabilities"]["memory_mapped_batches"], true);
    assert_eq!(caps_val["adapter_capabilities"]["empirical_memory_accounting"], true);
    assert_eq!(caps_val["adapter_capabilities"]["temporal_splits"], true);
    assert_eq!(caps_val["adapter_capabilities"]["future_lookahead_rejection"], true);
    assert_eq!(caps_val["adapter_capabilities"]["naive_and_seasonal_baselines"], true);
    assert_eq!(caps_val["adapter_capabilities"]["unsupervised_anomaly_detection"], true);
    assert_eq!(caps_val["adapter_capabilities"]["unsupervised_clustering"], true);
    assert_eq!(caps_val["adapter_capabilities"]["empirical_contamination_calibration"], true);
    assert_eq!(caps_val["adapter_capabilities"]["clustering_partition_profiles"], true);
    assert_eq!(caps_val["adapter_capabilities"]["out_of_sample_projection"], true);
    assert_eq!(caps_val["adapter_capabilities"]["frozen_text_embeddings"], true);
    assert_eq!(caps_val["adapter_capabilities"]["frozen_vision_embeddings"], true);
    assert_eq!(caps_val["adapter_capabilities"]["content_addressed_embedding_cache"], true);
}

#[test]
fn test_p9_acceptance_gate_full_closure() {
    let harness = TestHarness::new("p9_gate");
    harness.create_project("proj_gate");

    // Verify database schema retained at user_version = 10
    let conn = rusqlite::Connection::open(harness.root.join("state.sqlite")).unwrap();
    let user_ver: i64 = conn.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
    assert_eq!(user_ver, 10, "State database schema must strictly retain user_version = 10");

    // Verify P9 adapter tables exist and are functional
    let arrow_table_count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='arrow_dataset_adapters'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(arrow_table_count, 1);

    let ts_table_count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='time_series_backtests'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(ts_table_count, 1);

    let ac_table_count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='anomaly_clustering_adapters'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(ac_table_count, 1);

    let emb_table_count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='frozen_embedding_adapters'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(emb_table_count, 1);

    let cache_table_count: i64 = conn.query_row(
        "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='embedding_cache_records'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(cache_table_count, 1);

    // Full vertical slice validation
    let arrow_ds = ArrowDatasetAdapter::synthetic_parquet("gate_arrow", 25000, 20, "y");
    assert!(validate_arrow_adapter_memory_bounds(&arrow_ds, 256 * 1024 * 1024).is_ok());
    record_arrow_adapter(&harness.root, "proj_gate", &arrow_ds).unwrap();
    let loaded_arrow = get_arrow_adapter(&harness.root, "proj_gate", "gate_arrow").unwrap();
    assert_eq!(loaded_arrow.row_count, 25000);

    let ts_ds = TimeSeriesDatasetAdapter::synthetic_daily("gate_ts");
    record_time_series_backtest(&harness.root, "proj_gate", "gate_bt", &ts_ds).unwrap();
    let loaded_ts = get_time_series_backtest(&harness.root, "proj_gate", "gate_bt").unwrap();
    assert_eq!(loaded_ts.dataset_id, "gate_ts");

    let ac_ds = AnomalyClusteringAdapter::synthetic("gate_ac");
    assert!(validate_anomaly_clustering_config(&ac_ds).is_ok());
    record_anomaly_clustering_adapter(&harness.root, "proj_gate", &ac_ds).unwrap();
    let loaded_ac = get_anomaly_clustering_adapter(&harness.root, "proj_gate", "adapter_gate_ac").unwrap();
    assert_eq!(loaded_ac.dataset_id, "gate_ac");
    assert_eq!(loaded_ac.contamination_prior, 0.05);

    let emb_ds = FrozenEmbeddingAdapter::synthetic_text("gate_emb", 128);
    assert!(validate_frozen_embedding_config(&emb_ds).is_ok());
    record_frozen_embedding_adapter(&harness.root, "proj_gate", &emb_ds).unwrap();
    let loaded_emb = get_frozen_embedding_adapter(&harness.root, "proj_gate", "text_emb_gate_emb").unwrap();
    assert_eq!(loaded_emb.dataset_id, "gate_emb");
    assert_eq!(loaded_emb.dimension_fingerprint.embedding_dim, 128);

    let cache_rec = EmbeddingCacheRecord {
        project: "proj_gate".into(),
        cache_key: "gate_cache_key_01".into(),
        adapter_id: "text_emb_gate_emb".into(),
        input_sha256: "gate_input_sha256".into(),
        byte_size: 512,
        access_count: 1,
        created_ms: 2000,
        last_accessed_ms: 2000,
    };
    record_embedding_cache_record(&harness.root, "proj_gate", &cache_rec).unwrap();
    let loaded_rec = mlrescurix_harness::adapters::get_embedding_cache_record(&harness.root, "proj_gate", "gate_cache_key_01").unwrap();
    assert_eq!(loaded_rec.cache_key, "gate_cache_key_01");
    assert_eq!(loaded_rec.byte_size, 512);
}
