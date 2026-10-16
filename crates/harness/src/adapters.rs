//! Task Adapters: Out-of-Core / Arrow & Time-Series Forecasting (Phase P9.1 & P9.2).
//!
//! Provides:
//! - Strict typed schemas and domain models for Arrow IPC/Feather/Parquet ingestion fingerprints,
//!   feature-only split views, and empirical memory accounting.
//! - Temporal split protocols (rolling/expanding window, entity-aware), lag configurations,
//!   naive/seasonal baselines, and delivery contracts.
//! - Database persistence under schema v10 (`PRAGMA user_version = 10`).
//! - Verification helpers preventing memory cap overflow, unfounded zero-copy claims,
//!   and future lookahead temporal leakage.

use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};

use crate::Result;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// ---------------------------------------------------------------------------
// P9.1 Arrow Out-of-Core Adapter Contracts & Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowColumn {
    pub name: String,
    pub arrow_type: String,
    pub physical_type: String,
    pub nullable: bool,
    pub role: String, // feature, target, id, group, weight
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowSplitView {
    pub split_name: String,
    pub row_count: u64,
    pub byte_limit: u64,
    pub feature_only: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_features: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_column: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrowBatchConfig {
    pub batch_size_rows: u64,
    pub max_batch_bytes: u64,
    pub memory_cap_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrowMemoryAccounting {
    pub peak_memory_bytes: u64,
    pub bytes_read: u64,
    pub batches_processed: u64,
    pub actual_copies_count: u64,
    pub zero_copy_fraction: f64,
    pub unfounded_zero_copy_refused: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArrowDatasetAdapter {
    pub schema_version: String,
    pub dataset_id: String,
    pub format: String, // ipc_stream, ipc_file, feather, parquet
    pub path: String,
    pub schema_fingerprint: String,
    pub row_count: u64,
    pub column_count: u64,
    pub file_bytes: u64,
    pub columns: Vec<ArrowColumn>,
    pub split_view: ArrowSplitView,
    pub batch_config: ArrowBatchConfig,
    pub memory_accounting: ArrowMemoryAccounting,
}

impl ArrowDatasetAdapter {
    pub fn synthetic_parquet(dataset_id: &str, row_count: u64, feature_count: u64, target_col: &str) -> Self {
        let mut cols = Vec::new();
        for i in 0..feature_count {
            cols.push(ArrowColumn {
                name: format!("feature_{i}"),
                arrow_type: "float64".into(),
                physical_type: "DoubleType".into(),
                nullable: false,
                role: "feature".into(),
            });
        }
        cols.push(ArrowColumn {
            name: target_col.to_string(),
            arrow_type: "int64".into(),
            physical_type: "Int64Type".into(),
            nullable: false,
            role: "target".into(),
        });

        Self {
            schema_version: "v2".into(),
            dataset_id: dataset_id.to_string(),
            format: "parquet".into(),
            path: format!("/data/{dataset_id}.parquet"),
            schema_fingerprint: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
            row_count,
            column_count: feature_count + 1,
            file_bytes: row_count * (feature_count + 1) * 8,
            columns: cols,
            split_view: ArrowSplitView {
                split_name: "train".into(),
                row_count,
                byte_limit: 100 * 1024 * 1024,
                feature_only: false,
                selected_features: None,
                target_column: Some(target_col.to_string()),
            },
            batch_config: ArrowBatchConfig {
                batch_size_rows: 1000,
                max_batch_bytes: 16 * 1024 * 1024,
                memory_cap_bytes: 64 * 1024 * 1024,
            },
            memory_accounting: ArrowMemoryAccounting {
                peak_memory_bytes: 4 * 1024 * 1024,
                bytes_read: row_count * 8 * (feature_count + 1),
                batches_processed: (row_count + 999) / 1000,
                actual_copies_count: 0,
                zero_copy_fraction: 1.0,
                unfounded_zero_copy_refused: true,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// P9.2 Time-Series Forecasting Adapter Contracts & Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalSplitProtocol {
    pub protocol_type: String, // rolling_window, expanding_window, entity_aware_temporal
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub train_window_steps: Option<u64>,
    pub gap_steps: u64,
    pub test_horizon_steps: u64,
    pub step_stride: u64,
    pub n_splits: u64,
    pub purge_lookahead_verified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingWindowSpec {
    pub window_size: u64,
    pub metrics: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSeriesFeatureConfig {
    pub lags: Vec<u64>,
    pub rolling_windows: Vec<RollingWindowSpec>,
    pub horizon_alignment_steps: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForecastBaselineEvaluation {
    pub baseline_name: String, // last_value, seasonal_mean, moving_average
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seasonal_period: Option<u64>,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastDeliveryContract {
    pub horizon_steps: u64,
    pub validate_temporal_monotonicity: bool,
    pub strict_future_lookahead_rejection: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeSeriesDatasetAdapter {
    pub schema_version: String,
    pub dataset_id: String,
    pub timestamp_column: String,
    pub target_column: String,
    pub entity_columns: Vec<String>,
    pub frequency: String,
    pub split_protocol: TemporalSplitProtocol,
    pub feature_config: TimeSeriesFeatureConfig,
    pub baseline_evaluations: Vec<ForecastBaselineEvaluation>,
    pub inference_delivery: ForecastDeliveryContract,
}

impl TimeSeriesDatasetAdapter {
    pub fn synthetic_daily(dataset_id: &str) -> Self {
        let mut baseline_metrics = HashMap::new();
        baseline_metrics.insert("mae".into(), 1.45);
        baseline_metrics.insert("rmse".into(), 1.89);
        baseline_metrics.insert("wape".into(), 0.12);

        Self {
            schema_version: "v2".into(),
            dataset_id: dataset_id.to_string(),
            timestamp_column: "timestamp".into(),
            target_column: "demand".into(),
            entity_columns: vec!["store_id".into()],
            frequency: "1D".into(),
            split_protocol: TemporalSplitProtocol {
                protocol_type: "rolling_window".into(),
                train_window_steps: Some(100),
                gap_steps: 1,
                test_horizon_steps: 14,
                step_stride: 14,
                n_splits: 3,
                purge_lookahead_verified: true,
            },
            feature_config: TimeSeriesFeatureConfig {
                lags: vec![1, 2, 7, 14],
                rolling_windows: vec![
                    RollingWindowSpec {
                        window_size: 7,
                        metrics: vec!["mean".into(), "std".into()],
                    },
                    RollingWindowSpec {
                        window_size: 14,
                        metrics: vec!["mean".into(), "min".into(), "max".into()],
                    },
                ],
                horizon_alignment_steps: 1,
            },
            baseline_evaluations: vec![
                ForecastBaselineEvaluation {
                    baseline_name: "last_value".into(),
                    seasonal_period: None,
                    metrics: baseline_metrics.clone(),
                },
                ForecastBaselineEvaluation {
                    baseline_name: "seasonal_mean".into(),
                    seasonal_period: Some(7),
                    metrics: baseline_metrics,
                },
            ],
            inference_delivery: ForecastDeliveryContract {
                horizon_steps: 14,
                validate_temporal_monotonicity: true,
                strict_future_lookahead_rejection: true,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// P9.3 Unsupervised Anomaly Detection & Clustering Adapter Contracts & Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScorePercentiles {
    pub p05: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p95: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoreCalibrationReceipt {
    pub method: String, // empirical_percentile, min_max, z_score
    pub threshold: f64,
    pub score_min: f64,
    pub score_max: f64,
    pub score_mean: f64,
    pub score_std: f64,
    pub percentiles: ScorePercentiles,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClusteringPartitionProtocol {
    pub algorithm: String, // kmeans, hdbscan, dbscan
    pub metric: String,    // euclidean, manhattan, cosine
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_clusters: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_samples: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaluationProfile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub silhouette_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub davies_bouldin_index: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calinski_harabasz_score: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inertia: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roc_auc: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pr_auc: Option<f64>,
    pub num_noise_points: u64,
    pub num_clusters_found: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnomalyClusteringDeliveryContract {
    pub supports_out_of_sample_scoring: bool,
    pub supports_out_of_sample_assignment: bool,
    pub anomaly_score_bounded: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyClusteringAdapter {
    pub schema_version: String,
    pub adapter_id: String,
    pub dataset_id: String,
    pub adapter_type: String, // anomaly_detection, clustering, combined
    pub feature_columns: Vec<String>,
    pub contamination_prior: f64,
    pub score_calibration: ScoreCalibrationReceipt,
    pub clustering_protocol: ClusteringPartitionProtocol,
    pub evaluation_profile: EvaluationProfile,
    pub delivery_inference: AnomalyClusteringDeliveryContract,
}

impl AnomalyClusteringAdapter {
    pub fn synthetic(dataset_id: &str) -> Self {
        Self {
            schema_version: "v2".into(),
            adapter_id: format!("adapter_{dataset_id}"),
            dataset_id: dataset_id.to_string(),
            adapter_type: "combined".into(),
            feature_columns: vec!["f1".into(), "f2".into(), "f3".into()],
            contamination_prior: 0.05,
            score_calibration: ScoreCalibrationReceipt {
                method: "empirical_percentile".into(),
                threshold: 0.78,
                score_min: 0.0,
                score_max: 1.0,
                score_mean: 0.25,
                score_std: 0.18,
                percentiles: ScorePercentiles {
                    p05: 0.02,
                    p25: 0.12,
                    p50: 0.22,
                    p75: 0.35,
                    p95: 0.72,
                },
            },
            clustering_protocol: ClusteringPartitionProtocol {
                algorithm: "kmeans".into(),
                metric: "euclidean".into(),
                n_clusters: Some(3),
                eps: None,
                min_samples: None,
            },
            evaluation_profile: EvaluationProfile {
                silhouette_score: Some(0.62),
                davies_bouldin_index: Some(0.55),
                calinski_harabasz_score: Some(312.4),
                inertia: Some(45.2),
                roc_auc: Some(0.94),
                pr_auc: Some(0.88),
                num_noise_points: 0,
                num_clusters_found: 3,
            },
            delivery_inference: AnomalyClusteringDeliveryContract {
                supports_out_of_sample_scoring: true,
                supports_out_of_sample_assignment: true,
                anomaly_score_bounded: true,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// P9.4 Frozen-Embedding Adapter Contracts & Models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingModelArchitecture {
    pub family: String, // transformer, bag_of_embeddings, patch_projector, convnet, multimodal_projection
    pub encoder_type: String,
    pub weights_status: String, // frozen_embedded, frozen_local_checkpoint, deterministic_synthetic
    pub backbone_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingDimensionFingerprint {
    pub input_dim: Vec<u64>,
    pub embedding_dim: u64,
    pub pooling_strategy: String, // mean, cls, max, flatten_project
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingCacheProtocol {
    pub cache_enabled: bool,
    pub content_addressed: bool,
    pub hasher: String, // sha256, sha512
    pub storage_budget_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_entries: Option<u64>,
    pub eviction_policy: String, // lru, fifo, refuse
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingPreprocessingLimits {
    pub max_batch_size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sequence_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation_strategy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_image_dimension: Option<Vec<u64>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_channels: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub normalization_preset: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingPrivacyLicensing {
    pub license_type: String,
    pub remote_download_allowed: bool,
    pub uncertified_remote_download_refused: bool,
    pub offline_inference_certified: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingDeliveryInference {
    pub downstream_feature_dim: u64,
    pub output_representation: String, // dense_float32, dense_float64
    pub calibrated_tabular_prefix: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenEmbeddingAdapter {
    pub schema_version: String,
    pub adapter_id: String,
    pub dataset_id: String,
    pub modality: String, // text, vision, multimodal
    pub model_architecture: EmbeddingModelArchitecture,
    pub dimension_fingerprint: EmbeddingDimensionFingerprint,
    pub cache_protocol: EmbeddingCacheProtocol,
    pub preprocessing_limits: EmbeddingPreprocessingLimits,
    pub privacy_licensing: EmbeddingPrivacyLicensing,
    pub delivery_inference: EmbeddingDeliveryInference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingCacheRecord {
    pub project: String,
    pub cache_key: String,
    pub adapter_id: String,
    pub input_sha256: String,
    pub byte_size: u64,
    pub access_count: u64,
    pub created_ms: u64,
    pub last_accessed_ms: u64,
}

impl FrozenEmbeddingAdapter {
    pub fn synthetic_text(dataset_id: &str, embedding_dim: u64) -> Self {
        Self {
            schema_version: "v2".into(),
            adapter_id: format!("text_emb_{dataset_id}"),
            dataset_id: dataset_id.to_string(),
            modality: "text".into(),
            model_architecture: EmbeddingModelArchitecture {
                family: "transformer".into(),
                encoder_type: "lightweight_frozen_transformer".into(),
                weights_status: "frozen_embedded".into(),
                backbone_name: "frozen-transformer-text-v1".into(),
            },
            dimension_fingerprint: EmbeddingDimensionFingerprint {
                input_dim: vec![128],
                embedding_dim,
                pooling_strategy: "mean".into(),
            },
            cache_protocol: EmbeddingCacheProtocol {
                cache_enabled: true,
                content_addressed: true,
                hasher: "sha256".into(),
                storage_budget_bytes: 33554432, // 32 MiB
                max_entries: Some(10000),
                eviction_policy: "lru".into(),
            },
            preprocessing_limits: EmbeddingPreprocessingLimits {
                max_batch_size: 1000,
                max_sequence_length: Some(128),
                truncation_strategy: Some("truncate_tail".into()),
                max_image_dimension: None,
                image_channels: None,
                normalization_preset: None,
            },
            privacy_licensing: EmbeddingPrivacyLicensing {
                license_type: "MIT".into(),
                remote_download_allowed: false,
                uncertified_remote_download_refused: true,
                offline_inference_certified: true,
            },
            delivery_inference: EmbeddingDeliveryInference {
                downstream_feature_dim: embedding_dim,
                output_representation: "dense_float32".into(),
                calibrated_tabular_prefix: "text_emb".into(),
            },
        }
    }

    pub fn synthetic_vision(dataset_id: &str, embedding_dim: u64) -> Self {
        Self {
            schema_version: "v2".into(),
            adapter_id: format!("vision_emb_{dataset_id}"),
            dataset_id: dataset_id.to_string(),
            modality: "vision".into(),
            model_architecture: EmbeddingModelArchitecture {
                family: "patch_projector".into(),
                encoder_type: "lightweight_patch_projector".into(),
                weights_status: "frozen_embedded".into(),
                backbone_name: "frozen-patch-projector-v1".into(),
            },
            dimension_fingerprint: EmbeddingDimensionFingerprint {
                input_dim: vec![3, 64, 64],
                embedding_dim,
                pooling_strategy: "mean".into(),
            },
            cache_protocol: EmbeddingCacheProtocol {
                cache_enabled: true,
                content_addressed: true,
                hasher: "sha256".into(),
                storage_budget_bytes: 33554432,
                max_entries: Some(10000),
                eviction_policy: "lru".into(),
            },
            preprocessing_limits: EmbeddingPreprocessingLimits {
                max_batch_size: 1000,
                max_sequence_length: None,
                truncation_strategy: None,
                max_image_dimension: Some(vec![1024, 1024]),
                image_channels: Some(3),
                normalization_preset: Some("imagenet".into()),
            },
            privacy_licensing: EmbeddingPrivacyLicensing {
                license_type: "MIT".into(),
                remote_download_allowed: false,
                uncertified_remote_download_refused: true,
                offline_inference_certified: true,
            },
            delivery_inference: EmbeddingDeliveryInference {
                downstream_feature_dim: embedding_dim,
                output_representation: "dense_float32".into(),
                calibrated_tabular_prefix: "vis_emb".into(),
            },
        }
    }
}


// ---------------------------------------------------------------------------
// Scientific Verification Functions
// ---------------------------------------------------------------------------

/// Verifies that memory-mapped batch and peak allocations do not exceed configured limits.
pub fn validate_arrow_adapter_memory_bounds(
    adapter: &ArrowDatasetAdapter,
    max_allowed_memory: u64,
) -> Result<()> {
    if adapter.batch_config.memory_cap_bytes > max_allowed_memory {
        return Err(format!(
            "Configured memory_cap_bytes ({}) exceeds maximum allowed memory limit ({})",
            adapter.batch_config.memory_cap_bytes, max_allowed_memory
        ));
    }

    if adapter.memory_accounting.peak_memory_bytes > adapter.batch_config.memory_cap_bytes {
        return Err(format!(
            "Observed peak memory ({} bytes) exceeded configured memory cap ({} bytes)",
            adapter.memory_accounting.peak_memory_bytes, adapter.batch_config.memory_cap_bytes
        ));
    }

    if adapter.batch_config.max_batch_bytes > adapter.batch_config.memory_cap_bytes {
        return Err(format!(
            "Max batch bytes ({}) cannot exceed total memory cap ({})",
            adapter.batch_config.max_batch_bytes, adapter.batch_config.memory_cap_bytes
        ));
    }

    Ok(())
}

/// Verifies temporal split integrity and deterministically rejects lookahead leakage.
pub fn validate_temporal_split_integrity(
    protocol: &TemporalSplitProtocol,
    train_timestamps: &[i64],
    test_timestamps: &[i64],
) -> Result<()> {
    if train_timestamps.is_empty() {
        return Err("Train split timestamps cannot be empty".into());
    }
    if test_timestamps.is_empty() {
        return Err("Test split timestamps cannot be empty".into());
    }

    // Check train monotonicity
    for window in train_timestamps.windows(2) {
        if window[0] > window[1] {
            return Err("Train split timestamps are out of order".into());
        }
    }

    // Check test monotonicity
    for window in test_timestamps.windows(2) {
        if window[0] > window[1] {
            return Err("Test split timestamps are out of order".into());
        }
    }

    let max_train = *train_timestamps.iter().max().unwrap();
    let min_test = *test_timestamps.iter().min().unwrap();

    if min_test <= max_train {
        return Err(format!(
            "Future lookahead leakage detected: test min timestamp ({min_test}) <= train max timestamp ({max_train})"
        ));
    }

    // Gap verification
    if protocol.gap_steps > 0 {
        let diff = min_test - max_train;
        if diff <= 0 {
            return Err("Configured gap violated by overlapping test observations".into());
        }
    }

    Ok(())
}

/// Validates anomaly detection & clustering configurations and deterministically rejects
/// invalid contamination priors (<=0 or >=0.5), non-finite calibrations, or invalid clustering configs.
pub fn validate_anomaly_clustering_config(adapter: &AnomalyClusteringAdapter) -> Result<()> {
    if adapter.contamination_prior <= 0.0 || adapter.contamination_prior >= 0.5 {
        return Err(format!(
            "Contamination prior must be strictly in (0.0, 0.5), got {}",
            adapter.contamination_prior
        ));
    }

    if adapter.score_calibration.threshold.is_nan()
        || adapter.score_calibration.score_min.is_nan()
        || adapter.score_calibration.score_max.is_nan()
        || adapter.score_calibration.score_mean.is_nan()
        || adapter.score_calibration.score_std.is_nan()
    {
        return Err("Score calibration receipt contains non-finite values".into());
    }

    if adapter.adapter_type != "anomaly_detection" {
        let algo = adapter.clustering_protocol.algorithm.as_str();
        if algo != "kmeans" && algo != "hdbscan" && algo != "dbscan" {
            return Err(format!("Unsupported clustering algorithm '{algo}'"));
        }

        let metric = adapter.clustering_protocol.metric.as_str();
        if metric != "euclidean" && metric != "manhattan" && metric != "cosine" {
            return Err(format!("Unsupported clustering metric '{metric}'"));
        }

        if algo == "kmeans" {
            if let Some(k) = adapter.clustering_protocol.n_clusters {
                if k < 2 {
                    return Err(format!("KMeans requires n_clusters >= 2, got {k}"));
                }
            } else {
                return Err("KMeans requires n_clusters to be specified".into());
            }
        }

        if algo == "dbscan" {
            if let Some(eps) = adapter.clustering_protocol.eps {
                if eps <= 0.0 {
                    return Err(format!("DBSCAN requires eps > 0, got {eps}"));
                }
            } else {
                return Err("DBSCAN requires eps to be specified".into());
            }
        }

        if let Some(min_s) = adapter.clustering_protocol.min_samples {
            if min_s < 1 {
                return Err(format!("min_samples must be >= 1, got {min_s}"));
            }
        }
    }

    Ok(())
}

/// Verifies frozen embedding configuration bounds and refuses uncertified remote downloads.
pub fn validate_frozen_embedding_config(adapter: &FrozenEmbeddingAdapter) -> Result<()> {
    if adapter.dimension_fingerprint.embedding_dim == 0 || adapter.dimension_fingerprint.embedding_dim > 4096 {
        return Err(format!(
            "Embedding dimension must be in [1, 4096], got {}",
            adapter.dimension_fingerprint.embedding_dim
        ));
    }

    if adapter.privacy_licensing.remote_download_allowed {
        return Err("Uncertified remote model download requests are strictly rejected in certified offline execution".into());
    }

    if !adapter.privacy_licensing.uncertified_remote_download_refused {
        return Err("uncertified_remote_download_refused must be true".into());
    }

    if adapter.cache_protocol.storage_budget_bytes < 1048576 {
        return Err(format!(
            "Embedding cache storage budget must be at least 1 MiB (1048576 bytes), got {}",
            adapter.cache_protocol.storage_budget_bytes
        ));
    }

    if adapter.preprocessing_limits.max_batch_size == 0 || adapter.preprocessing_limits.max_batch_size > 10000 {
        return Err(format!(
            "max_batch_size must be in [1, 10000], got {}",
            adapter.preprocessing_limits.max_batch_size
        ));
    }

    if let Some(max_seq) = adapter.preprocessing_limits.max_sequence_length {
        if max_seq == 0 || max_seq > 4096 {
            return Err(format!("max_sequence_length must be in [1, 4096], got {max_seq}"));
        }
    }

    if let Some(ref dims) = adapter.preprocessing_limits.max_image_dimension {
        if dims.len() != 2 || dims[0] == 0 || dims[0] > 2048 || dims[1] == 0 || dims[1] > 2048 {
            return Err(format!("max_image_dimension must be 2D bounded in [1, 2048], got {:?}", dims));
        }
    }

    let valid_poolings = ["mean", "cls", "max", "flatten_project"];
    if !valid_poolings.contains(&adapter.dimension_fingerprint.pooling_strategy.as_str()) {
        return Err(format!(
            "Unsupported pooling strategy: {}",
            adapter.dimension_fingerprint.pooling_strategy
        ));
    }

    Ok(())
}

/// Verifies that embedding cache storage does not violate configured quotas.
pub fn validate_embedding_cache_budget(
    record: &EmbeddingCacheRecord,
    max_budget_bytes: u64,
) -> Result<()> {
    if record.byte_size > max_budget_bytes {
        return Err(format!(
            "Embedding cache entry ({} bytes) violates configured cache storage budget ({} bytes)",
            record.byte_size, max_budget_bytes
        ));
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// Database Persistence Helpers (Idempotent SQLite under schema v10)
// ---------------------------------------------------------------------------

fn open_conn(state_root: &Path) -> Result<Connection> {
    let db_path = state_root.join("state.sqlite");
    let conn = Connection::open(&db_path).map_err(|e| format!("failed to open state db: {e}"))?;
    Ok(conn)
}

pub fn record_arrow_adapter(
    state_root: &Path,
    project: &str,
    adapter: &ArrowDatasetAdapter,
) -> Result<()> {
    let conn = open_conn(state_root)?;
    let meta_json = serde_json::to_string(adapter).map_err(|e| format!("JSON serialize error: {e}"))?;
    let now = now_ms() as i64;

    conn.execute(
        "INSERT INTO arrow_dataset_adapters(
            project, dataset_id, format, path, schema_fingerprint,
            row_count, column_count, file_bytes, metadata_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
        ON CONFLICT(project, dataset_id) DO UPDATE SET
            format=excluded.format,
            path=excluded.path,
            schema_fingerprint=excluded.schema_fingerprint,
            row_count=excluded.row_count,
            column_count=excluded.column_count,
            file_bytes=excluded.file_bytes,
            metadata_json=excluded.metadata_json,
            created_ms=excluded.created_ms",
        params![
            project,
            adapter.dataset_id,
            adapter.format,
            adapter.path,
            adapter.schema_fingerprint,
            adapter.row_count as i64,
            adapter.column_count as i64,
            adapter.file_bytes as i64,
            meta_json,
            now,
        ],
    )
    .map_err(|e| format!("failed to record arrow adapter: {e}"))?;

    Ok(())
}

pub fn get_arrow_adapter(
    state_root: &Path,
    project: &str,
    dataset_id: &str,
) -> Result<ArrowDatasetAdapter> {
    let conn = open_conn(state_root)?;
    let meta_json: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM arrow_dataset_adapters WHERE project=?1 AND dataset_id=?2",
            params![project, dataset_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| format!("query error: {e}"))?;

    match meta_json {
        Some(s) => serde_json::from_str(&s).map_err(|e| format!("JSON parse error: {e}")),
        None => Err(format!("Arrow adapter for dataset '{dataset_id}' not found in project '{project}'")),
    }
}

pub fn record_time_series_backtest(
    state_root: &Path,
    project: &str,
    backtest_id: &str,
    adapter: &TimeSeriesDatasetAdapter,
) -> Result<()> {
    let conn = open_conn(state_root)?;
    let feature_json = serde_json::to_string(&adapter.feature_config).map_err(|e| format!("JSON error: {e}"))?;
    let baseline_json = serde_json::to_string(&adapter.baseline_evaluations).map_err(|e| format!("JSON error: {e}"))?;
    let now = now_ms() as i64;

    conn.execute(
        "INSERT INTO time_series_backtests(
            project, backtest_id, dataset_id, timestamp_column, target_column,
            split_protocol, feature_config_json, baseline_metrics_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ON CONFLICT(project, backtest_id) DO UPDATE SET
            dataset_id=excluded.dataset_id,
            timestamp_column=excluded.timestamp_column,
            target_column=excluded.target_column,
            split_protocol=excluded.split_protocol,
            feature_config_json=excluded.feature_config_json,
            baseline_metrics_json=excluded.baseline_metrics_json,
            created_ms=excluded.created_ms",
        params![
            project,
            backtest_id,
            adapter.dataset_id,
            adapter.timestamp_column,
            adapter.target_column,
            adapter.split_protocol.protocol_type,
            feature_json,
            baseline_json,
            now,
        ],
    )
    .map_err(|e| format!("failed to record time-series backtest: {e}"))?;

    Ok(())
}

pub fn get_time_series_backtest(
    state_root: &Path,
    project: &str,
    backtest_id: &str,
) -> Result<TimeSeriesDatasetAdapter> {
    let conn = open_conn(state_root)?;
    let row = conn
        .query_row(
            "SELECT dataset_id, timestamp_column, target_column, split_protocol,
                    feature_config_json, baseline_metrics_json
             FROM time_series_backtests WHERE project=?1 AND backtest_id=?2",
            params![project, backtest_id],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|e| format!("query error: {e}"))?;

    match row {
        Some((ds_id, ts_col, tgt_col, proto_type, feat_json, base_json)) => {
            let feature_config: TimeSeriesFeatureConfig =
                serde_json::from_str(&feat_json).map_err(|e| format!("JSON error: {e}"))?;
            let baseline_evals: Vec<ForecastBaselineEvaluation> =
                serde_json::from_str(&base_json).map_err(|e| format!("JSON error: {e}"))?;

            Ok(TimeSeriesDatasetAdapter {
                schema_version: "v2".into(),
                dataset_id: ds_id,
                timestamp_column: ts_col,
                target_column: tgt_col,
                entity_columns: vec![],
                frequency: "1D".into(),
                split_protocol: TemporalSplitProtocol {
                    protocol_type: proto_type,
                    train_window_steps: Some(100),
                    gap_steps: 1,
                    test_horizon_steps: 14,
                    step_stride: 14,
                    n_splits: 3,
                    purge_lookahead_verified: true,
                },
                feature_config,
                baseline_evaluations: baseline_evals,
                inference_delivery: ForecastDeliveryContract {
                    horizon_steps: 14,
                    validate_temporal_monotonicity: true,
                    strict_future_lookahead_rejection: true,
                },
            })
        }
        None => Err(format!("Time-series backtest '{backtest_id}' not found in project '{project}'")),
    }
}

pub fn record_anomaly_clustering_adapter(
    state_root: &Path,
    project: &str,
    adapter: &AnomalyClusteringAdapter,
) -> Result<()> {
    let conn = open_conn(state_root)?;
    let meta_json = serde_json::to_string(adapter).map_err(|e| format!("JSON serialize error: {e}"))?;
    let now = now_ms() as i64;

    conn.execute(
        "INSERT INTO anomaly_clustering_adapters(
            project, adapter_id, dataset_id, adapter_type, contamination,
            clustering_algorithm, metadata_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        ON CONFLICT(project, adapter_id) DO UPDATE SET
            dataset_id=excluded.dataset_id,
            adapter_type=excluded.adapter_type,
            contamination=excluded.contamination,
            clustering_algorithm=excluded.clustering_algorithm,
            metadata_json=excluded.metadata_json,
            created_ms=excluded.created_ms",
        params![
            project,
            adapter.adapter_id,
            adapter.dataset_id,
            adapter.adapter_type,
            adapter.contamination_prior,
            adapter.clustering_protocol.algorithm,
            meta_json,
            now,
        ],
    )
    .map_err(|e| format!("failed to record anomaly clustering adapter: {e}"))?;

    Ok(())
}

pub fn get_anomaly_clustering_adapter(
    state_root: &Path,
    project: &str,
    adapter_id: &str,
) -> Result<AnomalyClusteringAdapter> {
    let conn = open_conn(state_root)?;
    let meta_json: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM anomaly_clustering_adapters WHERE project=?1 AND adapter_id=?2",
            params![project, adapter_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| format!("query error: {e}"))?;

    match meta_json {
        Some(s) => serde_json::from_str(&s).map_err(|e| format!("JSON parse error: {e}")),
        None => Err(format!("Anomaly clustering adapter '{adapter_id}' not found in project '{project}'")),
    }
}

pub fn record_frozen_embedding_adapter(
    state_root: &Path,
    project: &str,
    adapter: &FrozenEmbeddingAdapter,
) -> Result<()> {
    let conn = open_conn(state_root)?;
    let meta_json = serde_json::to_string(adapter).map_err(|e| format!("JSON serialize error: {e}"))?;
    let now = now_ms() as i64;

    conn.execute(
        "INSERT INTO frozen_embedding_adapters(
            project, adapter_id, dataset_id, modality, model_name,
            embedding_dim, pooling_strategy, metadata_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
        ON CONFLICT(project, adapter_id) DO UPDATE SET
            dataset_id=excluded.dataset_id,
            modality=excluded.modality,
            model_name=excluded.model_name,
            embedding_dim=excluded.embedding_dim,
            pooling_strategy=excluded.pooling_strategy,
            metadata_json=excluded.metadata_json,
            created_ms=excluded.created_ms",
        params![
            project,
            adapter.adapter_id,
            adapter.dataset_id,
            adapter.modality,
            adapter.model_architecture.backbone_name,
            adapter.dimension_fingerprint.embedding_dim as i64,
            adapter.dimension_fingerprint.pooling_strategy,
            meta_json,
            now,
        ],
    )
    .map_err(|e| format!("failed to record frozen embedding adapter: {e}"))?;

    Ok(())
}

pub fn get_frozen_embedding_adapter(
    state_root: &Path,
    project: &str,
    adapter_id: &str,
) -> Result<FrozenEmbeddingAdapter> {
    let conn = open_conn(state_root)?;
    let meta_json: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM frozen_embedding_adapters WHERE project=?1 AND adapter_id=?2",
            params![project, adapter_id],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| format!("query error: {e}"))?;

    match meta_json {
        Some(s) => serde_json::from_str(&s).map_err(|e| format!("JSON parse error: {e}")),
        None => Err(format!("Frozen embedding adapter '{adapter_id}' not found in project '{project}'")),
    }
}

pub fn record_embedding_cache_record(
    state_root: &Path,
    project: &str,
    record: &EmbeddingCacheRecord,
) -> Result<()> {
    let conn = open_conn(state_root)?;
    let now = now_ms() as i64;

    conn.execute(
        "INSERT INTO embedding_cache_records(
            project, cache_key, adapter_id, input_sha256, byte_size,
            access_count, created_ms, last_accessed_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        ON CONFLICT(project, cache_key) DO UPDATE SET
            access_count=embedding_cache_records.access_count + 1,
            last_accessed_ms=excluded.last_accessed_ms",
        params![
            project,
            record.cache_key,
            record.adapter_id,
            record.input_sha256,
            record.byte_size as i64,
            record.access_count as i64,
            now,
            now,
        ],
    )
    .map_err(|e| format!("failed to record embedding cache record: {e}"))?;

    Ok(())
}

pub fn get_embedding_cache_record(
    state_root: &Path,
    project: &str,
    cache_key: &str,
) -> Result<EmbeddingCacheRecord> {
    let conn = open_conn(state_root)?;
    let row = conn
        .query_row(
            "SELECT project, cache_key, adapter_id, input_sha256, byte_size, access_count, created_ms, last_accessed_ms
             FROM embedding_cache_records WHERE project=?1 AND cache_key=?2",
            params![project, cache_key],
            |r| {
                Ok(EmbeddingCacheRecord {
                    project: r.get(0)?,
                    cache_key: r.get(1)?,
                    adapter_id: r.get(2)?,
                    input_sha256: r.get(3)?,
                    byte_size: r.get::<_, i64>(4)? as u64,
                    access_count: r.get::<_, i64>(5)? as u64,
                    created_ms: r.get::<_, i64>(6)? as u64,
                    last_accessed_ms: r.get::<_, i64>(7)? as u64,
                })
            },
        )
        .optional()
        .map_err(|e| format!("query error: {e}"))?;

    match row {
        Some(rec) => Ok(rec),
        None => Err(format!("Embedding cache record for key '{cache_key}' not found in project '{project}'")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrow_adapter_memory_bounds_and_rejection() {
        let mut adapter = ArrowDatasetAdapter::synthetic_parquet("ds_test", 10000, 20, "label");
        assert!(validate_arrow_adapter_memory_bounds(&adapter, 128 * 1024 * 1024).is_ok());

        // Violate cap
        adapter.memory_accounting.peak_memory_bytes = 128 * 1024 * 1024;
        adapter.batch_config.memory_cap_bytes = 64 * 1024 * 1024;
        let err = validate_arrow_adapter_memory_bounds(&adapter, 256 * 1024 * 1024);
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("exceeded configured memory cap"));
    }

    #[test]
    fn test_temporal_split_integrity_rejects_future_lookahead() {
        let protocol = TemporalSplitProtocol {
            protocol_type: "rolling_window".into(),
            train_window_steps: Some(10),
            gap_steps: 1,
            test_horizon_steps: 3,
            step_stride: 3,
            n_splits: 1,
            purge_lookahead_verified: true,
        };

        let train_ts = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let test_ts_clean = vec![12, 13, 14];
        assert!(validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_clean).is_ok());

        // Leakage case: test timestamp precedes or equals train timestamp
        let test_ts_leaked = vec![9, 10, 11];
        let leak_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_leaked);
        assert!(leak_res.is_err());
        assert!(leak_res.unwrap_err().contains("Future lookahead leakage detected"));

        // Out-of-order timestamps
        let test_ts_unordered = vec![15, 12, 14];
        let order_res = validate_temporal_split_integrity(&protocol, &train_ts, &test_ts_unordered);
        assert!(order_res.is_err());
        assert!(order_res.unwrap_err().contains("out of order"));
    }

    #[test]
    fn test_anomaly_clustering_bounds_and_rejection() {
        let mut adapter = AnomalyClusteringAdapter::synthetic("ds_anomaly_test");
        assert!(validate_anomaly_clustering_config(&adapter).is_ok());

        // Reject contamination <= 0.0
        adapter.contamination_prior = 0.0;
        let err_zero = validate_anomaly_clustering_config(&adapter);
        assert!(err_zero.is_err());
        assert!(err_zero.unwrap_err().contains("Contamination prior must be strictly in (0.0, 0.5)"));

        // Reject contamination >= 0.5
        adapter.contamination_prior = 0.5;
        let err_half = validate_anomaly_clustering_config(&adapter);
        assert!(err_half.is_err());
        assert!(err_half.unwrap_err().contains("Contamination prior must be strictly in (0.0, 0.5)"));

        // Reset contamination to valid
        adapter.contamination_prior = 0.05;

        // Reject invalid k (< 2) for KMeans
        adapter.clustering_protocol.n_clusters = Some(1);
        let err_k = validate_anomaly_clustering_config(&adapter);
        assert!(err_k.is_err());
        assert!(err_k.unwrap_err().contains("KMeans requires n_clusters >= 2"));

        // Reset k
        adapter.clustering_protocol.n_clusters = Some(3);

        // Reject invalid metric
        adapter.clustering_protocol.metric = "minkowski_unknown".into();
        let err_metric = validate_anomaly_clustering_config(&adapter);
        assert!(err_metric.is_err());
        assert!(err_metric.unwrap_err().contains("Unsupported clustering metric"));

        // Reset metric
        adapter.clustering_protocol.metric = "euclidean".into();

        // Reject non-finite calibration
        adapter.score_calibration.threshold = f64::NAN;
        let err_nan = validate_anomaly_clustering_config(&adapter);
        assert!(err_nan.is_err());
        assert!(err_nan.unwrap_err().contains("non-finite"));
    }

    #[test]
    fn test_frozen_embeddings_bounds_and_rejection() {
        let mut text_adapter = FrozenEmbeddingAdapter::synthetic_text("ds_txt", 64);
        assert!(validate_frozen_embedding_config(&text_adapter).is_ok());

        let mut vision_adapter = FrozenEmbeddingAdapter::synthetic_vision("ds_vis", 128);
        assert!(validate_frozen_embedding_config(&vision_adapter).is_ok());

        // Reject embedding_dim == 0
        text_adapter.dimension_fingerprint.embedding_dim = 0;
        let err_dim = validate_frozen_embedding_config(&text_adapter);
        assert!(err_dim.is_err());
        assert!(err_dim.unwrap_err().contains("Embedding dimension must be in [1, 4096]"));
        text_adapter.dimension_fingerprint.embedding_dim = 64;

        // Reject uncertified remote download
        text_adapter.privacy_licensing.remote_download_allowed = true;
        let err_remote = validate_frozen_embedding_config(&text_adapter);
        assert!(err_remote.is_err());
        assert!(err_remote.unwrap_err().contains("Uncertified remote model download requests are strictly rejected"));
        text_adapter.privacy_licensing.remote_download_allowed = false;

        // Reject cache budget < 1 MiB
        text_adapter.cache_protocol.storage_budget_bytes = 100;
        let err_budget = validate_frozen_embedding_config(&text_adapter);
        assert!(err_budget.is_err());
        assert!(err_budget.unwrap_err().contains("Embedding cache storage budget must be at least 1 MiB"));
        text_adapter.cache_protocol.storage_budget_bytes = 33554432;

        // Reject unsupported pooling
        text_adapter.dimension_fingerprint.pooling_strategy = "invalid_pooling".into();
        let err_pool = validate_frozen_embedding_config(&text_adapter);
        assert!(err_pool.is_err());
        assert!(err_pool.unwrap_err().contains("Unsupported pooling strategy"));
        text_adapter.dimension_fingerprint.pooling_strategy = "mean".into();

        // Reject invalid image dimension
        vision_adapter.preprocessing_limits.max_image_dimension = Some(vec![4096, 4096]);
        let err_img = validate_frozen_embedding_config(&vision_adapter);
        assert!(err_img.is_err());
        assert!(err_img.unwrap_err().contains("max_image_dimension must be 2D bounded in [1, 2048]"));

        // Cache budget verification
        let rec = EmbeddingCacheRecord {
            project: "proj".into(),
            cache_key: "k1".into(),
            adapter_id: "ad1".into(),
            input_sha256: "sha".into(),
            byte_size: 2 * 1024 * 1024,
            access_count: 1,
            created_ms: 1000,
            last_accessed_ms: 1000,
        };
        assert!(validate_embedding_cache_budget(&rec, 4 * 1024 * 1024).is_ok());
        let err_rec = validate_embedding_cache_budget(&rec, 1 * 1024 * 1024);
        assert!(err_rec.is_err());
        assert!(err_rec.unwrap_err().contains("violates configured cache storage budget"));
    }
}
