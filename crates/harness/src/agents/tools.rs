use std::{
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agents::{
        gateway::{ToolDefinition, ToolFunction},
        sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus},
        workspace::BranchWorkspaceManager,
    },
    Result,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDescriptor {
    pub name: String,
    pub version: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub entrypoint: String,
    pub source_hash: String,
    pub environment_id: String,
    pub permitted_artifacts: Vec<String>,
    pub max_wall_ms: u64,
    pub provenance: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolExecutionReceipt {
    pub tool_name: String,
    pub version: String,
    pub status: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub output_data: Option<Value>,
    pub elapsed_ms: u64,
}

pub struct ToolRegistry;

impl ToolRegistry {
    /// Return the list of Category A trusted harness tools available to agents.
    pub fn get_trusted_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "workspace_fs".into(),
                    description: "Versioned workspace filesystem operations with boundary enforcement, expected hash checks for mutations, and bounded results.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "action": { "type": "string", "enum": ["read", "write", "patch", "list", "glob", "search"], "description": "Filesystem operation" },
                            "path": { "type": "string", "description": "Relative file path within branch workspace" },
                            "content": { "type": "string", "description": "File content for write action" },
                            "target": { "type": "string", "description": "Target string to replace for patch action" },
                            "replacement": { "type": "string", "description": "Replacement string for patch action" },
                            "expected_hash": { "type": "string", "description": "Optional expected SHA-256 hash before mutation for optimistic concurrency" },
                            "pattern": { "type": "string", "description": "Pattern for glob action, e.g. '*.py' or 'candidate/*'" },
                            "query": { "type": "string", "description": "Substring query for search action" },
                            "max_results": { "type": "integer", "description": "Maximum search results to return (default 50)" }
                        },
                        "required": ["action"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "sandbox_exec".into(),
                    description: "Execute commands or scripts inside an isolated OCI container with /workspace:ro, bounded /scratch:rw, and thread caps.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "argv": { "type": "array", "items": { "type": "string" }, "description": "Structured command line arguments, e.g. ['python3', '-m', 'candidate.entrypoint', 'fit', ...]" },
                            "script": { "type": "string", "description": "Alternative relative path to python script" },
                            "args": { "type": "array", "items": { "type": "string" }, "description": "Arguments for script" },
                            "environment_id": { "type": "string", "description": "Environment identifier (defaults to pinned scientific stack)" },
                            "timeout_ms": { "type": "integer", "description": "Execution timeout in milliseconds (default 30000)" },
                            "max_memory_mb": { "type": "integer", "description": "Memory ceiling in megabytes (default 512)" },
                            "max_scratch_mb": { "type": "integer", "description": "Scratch tmpfs limit in megabytes (default 64)" }
                        }
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "evaluate_submission".into(),
                    description: "Trigger harness-managed fold-isolated cross validation on an executable candidate and generate trusted official scorecards.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Candidate identifier matching workspace candidate manifest" },
                            "candidate_path": { "type": "string", "description": "Relative path to candidate directory in branch workspace (default 'candidate')" },
                            "dataset_id": { "type": "string", "description": "Registered dataset ID" },
                            "folds": { "type": "integer", "description": "Number of cross-validation folds (2 to 5, default 3)" },
                            "seed": { "type": "integer", "description": "Evaluation random seed (default 2026)" },
                            "diagnostic_slices": { "type": "array", "items": { "type": "string" }, "description": "Feature column names for diagnostic slice evaluation" }
                        },
                        "required": ["candidate_id", "dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "artifact_crud".into(),
                    description: "Project-scoped artifact registry operations (publish, get, list, lineage, pin, unpin, tombstone). Scorecards cannot be forged.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "action": { "type": "string", "enum": ["publish", "get", "list", "lineage", "pin", "unpin", "tombstone"], "description": "Registry action" },
                            "artifact_id": { "type": "string", "description": "Artifact identifier" },
                            "kind": { "type": "string", "description": "Artifact kind (model, code, dataset, checkpoint, report)" },
                            "version": { "type": "string", "description": "Semantic or revision string" },
                            "source_path": { "type": "string", "description": "Relative path in branch workspace to publish from" },
                            "summary": { "type": "string", "description": "Human-readable summary text" },
                            "metadata": { "type": "object", "description": "Arbitrary metadata object" },
                            "expected_revision": { "type": "integer", "description": "Expected project revision for optimistic locking" },
                            "edge_type": { "type": "string", "enum": ["derived_from", "evaluates", "packages", "supersedes"], "description": "Lineage relationship edge" },
                            "target_artifact_id": { "type": "string", "description": "Target artifact ID for lineage edge" }
                        },
                        "required": ["action"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "discover_capabilities".into(),
                    description: "Discover harness capabilities, supported tools, sandbox environment profile, resource presets, and validation query limits.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "dataset_query".into(),
                    description: "Bounded query of registered dataset schemas, summary statistics, and missingness without leaking validation/test labels.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Registered dataset identifier" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "experiment_tree".into(),
                    description: "Query experiment DAG nodes, candidates, trials, and parent relationships.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "estimate_feasibility".into(),
                    description: "Estimate analytical compute complexity, run capped micro-pilots (N <= 100), and report transparent multi-tier feasibility verdict before heavy training dispatch.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Registered dataset identifier" },
                            "model_family": { "type": "string", "enum": ["linear", "boosting", "tree", "neural", "custom"], "description": "Model algorithm family" },
                            "cv_folds": { "type": "integer", "description": "Number of cross-validation folds (default 5)" },
                            "include_final_refit": { "type": "boolean", "description": "Whether to include final refit on full training set (default true)" },
                            "concurrent_worker_leases": { "type": "integer", "description": "Number of concurrent worker slots (default 1)" },
                            "time_budget_ms": { "type": "integer", "description": "Time budget in milliseconds (default 3600000)" },
                            "max_memory_mb": { "type": "integer", "description": "Optional custom memory ceiling in megabytes" },
                            "run_pilot": { "type": "boolean", "description": "Whether to run a capped micro-pilot probe (default true)" }
                        },
                        "required": ["dataset_id", "model_family"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "negotiate_execution_plan".into(),
                    description: "Negotiate an execution plan under resource constraints, applying safe deterministic downscaling without quiet semantic degradation or mutating locked splits.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "report_id": { "type": "string", "description": "Feasibility report identifier" },
                            "accepted_adaptations": { "type": "array", "items": { "type": "string" }, "description": "List of accepted adaptation techniques (e.g. streaming, sparse_linear, progressive_fidelity, downsampled_screening, smaller_architecture, quantization)" },
                            "custom_constraints": { "type": "object", "description": "Negotiated constraint values (e.g. latency targets, subsample rows)" },
                            "target_preset": { "type": "string", "enum": ["MICRO", "STANDARD", "HEAVY", "CUSTOM"], "description": "Target resource preset" }
                        },
                        "required": ["report_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "inspect_arrow_dataset".into(),
                    description: "Inspect an Arrow IPC, Feather, or Parquet dataset, extract typed schema fingerprint, column descriptors, and configure chunked memory-mapped batch ingestion.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Identifier for the Arrow dataset" },
                            "path": { "type": "string", "description": "Path to the Arrow/Parquet/Feather file" },
                            "format": { "type": "string", "enum": ["auto", "parquet", "feather", "ipc_file", "ipc_stream"], "description": "Format type (default auto)" },
                            "target_column": { "type": "string", "description": "Optional target column name" },
                            "batch_size_rows": { "type": "integer", "description": "Batch size in rows (default 1000)" },
                            "memory_cap_bytes": { "type": "integer", "description": "Memory ceiling in bytes (default 67108864 = 64 MiB)" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "evaluate_temporal_backtest".into(),
                    description: "Configure and evaluate temporal splits (rolling/expanding window, entity-aware), causal lags, rolling aggregations, and naive/seasonal baselines for time-series forecasting.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Identifier for the time-series dataset" },
                            "timestamp_column": { "type": "string", "description": "Timestamp column name" },
                            "target_column": { "type": "string", "description": "Target forecasting column name" },
                            "horizon_steps": { "type": "integer", "description": "Forecast horizon steps (default 14)" },
                            "backtest_id": { "type": "string", "description": "Optional backtest identifier" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "evaluate_anomaly_clustering".into(),
                    description: "Configure and evaluate unsupervised anomaly detection (IsolationForest, LOF) and clustering (KMeans, DBSCAN/HDBSCAN) with empirical calibrations and evaluation profiles.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "adapter_id": { "type": "string", "description": "Adapter identifier" },
                            "dataset_id": { "type": "string", "description": "Identifier for the dataset" },
                            "adapter_type": { "type": "string", "enum": ["anomaly_detection", "clustering", "combined"], "description": "Adapter type" },
                            "contamination": { "type": "number", "description": "Contamination prior strictly in (0.0, 0.5) (default 0.05)" },
                            "clustering_algorithm": { "type": "string", "enum": ["kmeans", "hdbscan", "dbscan"], "description": "Clustering algorithm (default kmeans)" },
                            "n_clusters": { "type": "integer", "description": "Number of clusters for KMeans (default 3)" },
                            "metric": { "type": "string", "enum": ["euclidean", "manhattan", "cosine"], "description": "Distance metric (default euclidean)" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "extract_frozen_embeddings".into(),
                    description: "Configure and extract frozen text or vision embeddings with content-addressed caching, sequence/image bounds, and calibrated tabular feature delivery.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "adapter_id": { "type": "string", "description": "Adapter identifier" },
                            "dataset_id": { "type": "string", "description": "Identifier for the dataset" },
                            "modality": { "type": "string", "enum": ["text", "vision", "multimodal"], "description": "Modality type (default text)" },
                            "model_name": { "type": "string", "description": "Backbone model name (default frozen-transformer-text-v1 or frozen-patch-projector-v1)" },
                            "embedding_dim": { "type": "integer", "description": "Embedding output dimension in [1, 4096] (default 64)" },
                            "pooling_strategy": { "type": "string", "enum": ["mean", "cls", "max", "flatten_project"], "description": "Pooling strategy (default mean)" },
                            "cache_enabled": { "type": "boolean", "description": "Whether content-addressed embedding cache is enabled (default true)" },
                            "remote_download": { "type": "boolean", "description": "Whether remote download is requested (must be false in certified offline execution)" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "synthesize_candidate".into(),
                    description: "Autonomously synthesize decoupled, protocol-conforming ML candidates (entrypoint.py, model.py, features.py, manifest.json) supporting LightGBM, XGBoost, PyTorch, Trees, Linear, and custom pipelines.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Unique candidate identifier" },
                            "task_type": { "type": "string", "enum": ["binary", "multiclass", "regression"], "description": "Problem task type" },
                            "target_column": { "type": "string", "description": "Target column name in dataset" },
                            "model_family": { "type": "string", "enum": ["boosting", "neural", "tree", "linear", "custom"], "description": "Model architecture family (default boosting)" },
                            "framework": { "type": "string", "enum": ["lightgbm", "xgboost", "pytorch", "scikit-learn", "custom"], "description": "ML execution framework (default lightgbm)" },
                            "class_order": { "type": "array", "description": "Ordered class labels for classification" },
                            "hyperparameters": { "type": "object", "description": "Model hyperparameters" },
                            "feature_pipeline": { "type": "object", "description": "Feature pipeline configuration (scaling, encoding, missing value handling)" },
                            "candidate_path": { "type": "string", "description": "Relative destination directory in branch workspace (default 'candidate')" },
                            "custom_modules": { "type": "object", "description": "Optional dictionary of custom python modules mapping relative path to code string" }
                        },
                        "required": ["candidate_id", "task_type", "target_column"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "inspect_diagnostics".into(),
                    description: "Inspect structured execution diagnostics, root-cause attribution, peak telemetry, and remediation suggestions for candidate failures.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Optional candidate identifier to filter diagnostics" },
                            "failure_type": { "type": "string", "enum": ["cgroup_oom", "non_finite_loss", "shape_mismatch", "class_order_mismatch", "execution_timeout", "missing_dependency", "syntax_error", "runtime_exception"], "description": "Optional failure classification to filter" },
                            "diagnostic_id": { "type": "string", "description": "Specific diagnostic receipt ID to retrieve" },
                            "limit": { "type": "integer", "description": "Maximum receipts to return (default 10)" }
                        }
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "propose_hypothesis".into(),
                    description: "Propose an explicit iteration hypothesis for candidate mutation (feature_engineering, hyperparameter_tuning, architecture_change, ensembling) with expected metric deltas and multi-fidelity budgets.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "hypothesis_id": { "type": "string", "description": "Optional unique hypothesis ID" },
                            "candidate_id": { "type": "string", "description": "Target candidate identifier being mutated/evaluated" },
                            "parent_candidate_id": { "type": "string", "description": "Optional baseline parent candidate identifier" },
                            "mutation_type": { "type": "string", "enum": ["feature_engineering", "hyperparameter_tuning", "architecture_change", "ensembling"], "description": "Type of mutation applied" },
                            "hypothesis_statement": { "type": "string", "description": "Falsifiable hypothesis statement describing expected behavior" },
                            "rationale": { "type": "string", "description": "Engineering or domain rationale" },
                            "metric": { "type": "string", "description": "Evaluation metric to verify (default balanced_accuracy)" },
                            "direction": { "type": "string", "enum": ["maximize", "minimize"], "description": "Metric optimization direction (default maximize)" },
                            "target_threshold": { "type": "number", "description": "Optional target absolute threshold" },
                            "minimum_improvement": { "type": "number", "description": "Minimum expected improvement delta (default 0.005)" },
                            "proxy_max_samples": { "type": "integer", "description": "Max samples for Tier 0 fast screen (default 500)" },
                            "screening_threshold": { "type": "number", "description": "Threshold score for Tier 0 admission" },
                            "full_folds": { "type": "integer", "description": "Number of folds for Tier 1 full CV (default 3)" }
                        },
                        "required": ["candidate_id", "mutation_type", "hypothesis_statement"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "screen_candidate_fidelity".into(),
                    description: "Execute progressive multi-fidelity screening (Tier 0 fast proxy screen N <= 500, Tier 1 full CV) and settle iteration hypothesis with Pareto coordinates.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Candidate identifier" },
                            "dataset_id": { "type": "string", "description": "Registered dataset ID" },
                            "hypothesis_id": { "type": "string", "description": "Optional hypothesis ID to link and settle" },
                            "candidate_path": { "type": "string", "description": "Relative path to candidate in branch workspace (default 'candidate')" },
                            "baseline_score": { "type": "number", "description": "Optional baseline metric score to compare against" },
                            "tier": { "type": "string", "enum": ["proxy", "full", "progressive"], "description": "Screening tier to execute (default progressive)" },
                            "screening_threshold": { "type": "number", "description": "Optional threshold for Tier 0 proxy pass" },
                            "folds": { "type": "integer", "description": "CV folds for Tier 1 (default 3)" },
                            "seed": { "type": "integer", "description": "Random seed (default 2026)" }
                        },
                        "required": ["candidate_id", "dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "query_pareto_frontier".into(),
                    description: "Compute and query non-dominated Pareto frontier balancing validation performance against execution latency ms and peak memory bytes.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "primary_metric": { "type": "string", "description": "Primary metric name (default primary_metric)" },
                            "direction": { "type": "string", "enum": ["maximize", "minimize"], "description": "Optimization direction (default maximize)" },
                            "candidate_ids": { "type": "array", "items": { "type": "string" }, "description": "Optional list of candidate IDs to evaluate on frontier" },
                            "frontier_id": { "type": "string", "description": "Optional frontier snapshot ID" }
                        }
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "profile_dataset".into(),
                    description: "Inspect and profile a registered dataset to discover schema, nulls, cardinalities, dtypes, class balance, and inferred task type.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Registered dataset ID" },
                            "target_column": { "type": "string", "description": "Optional target column name" },
                            "id_column": { "type": "string", "description": "Optional ID column name" }
                        },
                        "required": ["dataset_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "package_candidate_bundle".into(),
                    description: "Package an evaluated candidate into an Apex-certified, self-contained mlrescurix.bundle.v2 delivery bundle with clean dependency isolation.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Candidate identifier" },
                            "dataset_id": { "type": "string", "description": "Registered dataset ID" },
                            "test_access_id": { "type": "string", "description": "Authenticated sealed test access gate identifier" },
                            "candidate_path": { "type": "string", "description": "Relative path to candidate in workspace (default 'candidate')" },
                            "output_bundle_path": { "type": "string", "description": "Relative output bundle path in workspace (default 'bundle')" },
                            "task_type": { "type": "string", "description": "Task type (default binary_classification)" },
                            "target_column": { "type": "string", "description": "Target column name (default target)" }
                        },
                        "required": ["candidate_id", "dataset_id", "test_access_id"]
                    }),
                },
            },
            ToolDefinition {tool_type:"function".into(),function:ToolFunction {
                name:"schedule_pipeline".into(),
                description:"Queue a bounded feature/calibration/ensemble protocol in a verified scientific Docker environment. The Rust supervisor evaluates development data and verifies generated OOF provenance; this does not finalize or certify delivery.".into(),
                parameters:json!({"type":"object","properties":{"job_id":{"type":"string"},"dataset_id":{"type":"string"},"environment_id":{"type":"string"},"protocol":{"type":"object"}},"required":["job_id","dataset_id","environment_id","protocol"],"additionalProperties":false}),
            }},
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_read".into(),
                    description: "Read a file from the current branch workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_write".into(),
                    description: "Write content to a file in the current branch workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" },
                            "content": { "type": "string", "description": "Full file content" }
                        },
                        "required": ["path", "content"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_patch".into(),
                    description: "Apply an exact text replacement edit to a file in the workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" },
                            "target": { "type": "string", "description": "Exact text to find and replace" },
                            "replacement": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["path", "target", "replacement"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_list".into(),
                    description: "List all files present in the current branch workspace.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "sandbox_python".into(),
                    description: "Execute a Python script or test in the isolated sandbox.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "script": { "type": "string", "description": "Relative path to python script" },
                            "args": { "type": "array", "items": { "type": "string" }, "description": "Script arguments" }
                        },
                        "required": ["script"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_project_tool".into(),
                    description: "Register an agent-authored project tool with versioned descriptor and entrypoint.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Tool name" },
                            "version": { "type": "string", "description": "Tool semantic version, e.g. v1" },
                            "description": { "type": "string", "description": "Human-readable description" },
                            "entrypoint": { "type": "string", "description": "Relative script path, e.g. diagnostics/slice_tool.py" },
                            "input_schema": { "type": "object", "description": "JSON Schema for tool input" },
                            "output_schema": { "type": "object", "description": "JSON Schema for tool output" },
                            "environment_id": { "type": "string", "description": "Environment identifier" }
                        },
                        "required": ["name", "version", "description", "entrypoint"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "invoke_project_tool".into(),
                    description: "Invoke a previously registered agent-authored project tool.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Registered tool name" },
                            "version": { "type": "string", "description": "Registered tool version" },
                            "input": { "type": "object", "description": "Input arguments matching descriptor schema" }
                        },
                        "required": ["name", "version", "input"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "inspect_dataset".into(),
                    description: "Inspect schema, column types, row counts, and target configuration of the registered dataset.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_dataset".into(),
                    description: "Register an explicitly selected CSV dataset for project scientific evaluation.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "dataset_id": { "type": "string", "description": "Unique identifier for the registered dataset" },
                            "path": { "type": "string", "description": "Path to the dataset CSV file" }
                        },
                        "required": ["dataset_id", "path"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "delegate".into(),
                    description: "Delegate a subtask to a specialist role with a testable hypothesis and assigned branch.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "role_id": { "type": "string", "description": "Specialist role to delegate to" },
                            "intent": { "type": "string", "description": "Goal and specification of delegated subtask" },
                            "hypothesis": { "type": "string", "description": "Testable hypothesis being investigated" },
                            "sub_task_id": { "type": "string", "description": "Optional explicit child task identifier" }
                        },
                        "required": ["role_id", "intent"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "delegate_task".into(),
                    description: "Delegate a subtask to a specialist role with durable intent binding and crash recovery.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "role_id": { "type": "string", "description": "Specialist role to delegate to" },
                            "intent": { "type": "string", "description": "Goal and specification of delegated subtask" },
                            "hypothesis": { "type": "string", "description": "Testable hypothesis being investigated" },
                            "sub_task_id": { "type": "string", "description": "Optional explicit child task identifier" }
                        },
                        "required": ["role_id", "intent"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "schedule_specialist".into(),
                    description: "Durably enqueue a specialist subtask in M0 and agent_tasks for scheduled execution and restart-safe reconciliation.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "role_id": { "type": "string", "description": "Specialist role to schedule" },
                            "intent": { "type": "string", "description": "Goal and specification of scheduled subtask" },
                            "hypothesis": { "type": "string", "description": "Testable hypothesis being investigated" },
                            "sub_task_id": { "type": "string", "description": "Optional explicit child task identifier" }
                        },
                        "required": ["role_id", "intent"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "query_budget".into(),
                    description: "Query remaining budget dimensions (calls, tokens, wall_ms) and worker capacity.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "request_finalization".into(),
                    description: "Submit a finalization request for a verified candidate or baseline after critic audit.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Identifier of the verified candidate or baseline" },
                            "rationale": { "type": "string", "description": "Summary of evidence justifying final freeze under declared protocol" },
                            "approval_token": { "type": "string", "description": "Authenticated user approval token for this candidate" }
                        },
                        "required": ["candidate_id", "rationale", "approval_token"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "publish_delivery".into(),
                    description: "Publish verified delivery bundle for a candidate after critic audit and approval.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Identifier of the verified candidate or baseline" },
                            "rationale": { "type": "string", "description": "Summary of evidence justifying final freeze under declared protocol" },
                            "approval_token": { "type": "string", "description": "Authenticated user approval token for this candidate" }
                        },
                        "required": ["candidate_id", "rationale", "approval_token"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "retrieve_evidence".into(),
                    description: "Retrieve verified historical evidence, findings, or trial results by durable ID.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "evidence_id": { "type": "string", "description": "Durable ID of evidence or finding record" }
                        },
                        "required": ["evidence_id"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "submit_critic_audit".into(),
                    description: "Submit an independent critic review audit for a candidate against the declared evaluation protocol.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Candidate identifier being audited" },
                            "verdict": { "type": "string", "enum": ["approved", "rejected"], "description": "Audit decision" },
                            "protocol_compliance": { "type": "boolean", "description": "Whether candidate complies with declared evaluation protocol" },
                            "binding_sha256": { "type": "string", "description": "Exact current finalization binding; required for an audit that may authorize finalization" },
                            "evidence_summary": { "type": "string", "description": "Summary of verified evidence, metrics, and feasibility" }
                        },
                        "required": ["candidate_id", "verdict", "protocol_compliance", "evidence_summary"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_feature_transform".into(),
                    description: "Register a verified feature transform specification with parent dataset identity.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "transform_id": { "type": "string", "description": "Unique transform identifier" },
                            "transform_spec": { "type": "object", "description": "Declared transform parameters and scaling strategy" },
                            "parent_dataset": { "type": "string", "description": "Registered dataset ID" }
                        },
                        "required": ["transform_id", "transform_spec", "parent_dataset"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_calibration".into(),
                    description: "Register a probability calibration record tied to an evaluated base candidate.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "calibration_id": { "type": "string", "description": "Unique calibration identifier" },
                            "base_candidate_id": { "type": "string", "description": "Evaluated base candidate identifier" },
                            "method": { "type": "string", "enum": ["isotonic", "sigmoid"], "description": "Calibration method" },
                            "brier_score": { "type": "number", "description": "Brier score on calibration split" },
                            "log_loss": { "type": "number", "description": "Log loss on calibration split" }
                        },
                        "required": ["calibration_id", "base_candidate_id", "method"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_ensemble".into(),
                    description: "Register a scientific ensemble with verified parent lineage and weights.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "ensemble_id": { "type": "string", "description": "Unique ensemble identifier" },
                            "recipe_type": { "type": "string", "enum": ["voting", "stacking", "averaging"], "description": "Ensemble recipe type" },
                            "parent_ids": { "type": "array", "items": { "type": "string" }, "description": "Verified parent candidate identifiers" },
                            "weights": { "type": "array", "items": { "type": "number" }, "description": "Optional parent weights" },
                            "meta_params": { "type": "object", "description": "Optional meta-learner parameters" },
                            "is_feasible": { "type": "boolean", "description": "Whether ensemble satisfies feasibility constraints" }
                        },
                        "required": ["ensemble_id", "recipe_type", "parent_ids"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "inspect_scientific_lineage".into(),
                    description: "Inspect registered scientific ensembles, feature transforms, and calibration records for the project.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
        ]
    }

    /// Register a Category B agent-authored project tool.
    pub fn register_tool(
        root: &Path,
        project_id: &str,
        descriptor: ToolDescriptor,
    ) -> Result<()> {
        let now = now_ms();
        let desc_json = serde_json::to_string(&descriptor).map_err(|e| e.to_string())?;

        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO agent_tools(project, tool_name, version, descriptor_json, source_hash, environment_id, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(project, tool_name, version) DO NOTHING",
            rusqlite::params![
                project_id,
                descriptor.name,
                descriptor.version,
                desc_json,
                descriptor.source_hash,
                descriptor.environment_id,
                now,
            ],
        ).map_err(|e| e.to_string())?;

        let stored_json: String = db.query_row(
            "SELECT descriptor_json FROM agent_tools WHERE project=?1 AND tool_name=?2 AND version=?3",
            rusqlite::params![project_id, descriptor.name, descriptor.version],
            |r| r.get(0),
        ).map_err(|e| e.to_string())?;
        let stored: ToolDescriptor = serde_json::from_str(&stored_json).map_err(|e| e.to_string())?;
        if stored != descriptor {
            return Err(format!(
                "Tool {}:{} is already registered with a different descriptor. Tool versions are immutable.",
                descriptor.name, descriptor.version,
            ));
        }

        Ok(())
    }

    /// Invoke an agent-authored project tool inside the branch workspace.
    pub fn invoke_tool(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        tool_name: &str,
        version: &str,
        input: &Value,
    ) -> Result<ToolExecutionReceipt> {
        Self::invoke_tool_with_status(root, project_id, branch_id, tool_name, version, input, SandboxDetector::detect())
    }

pub fn validate_json_schema(val: &Value, schema: &Value) -> Result<()> {
    if !schema.is_object() || schema.as_object().unwrap().is_empty() {
        return Ok(());
    }
    if let Some(expected_type) = schema.get("type").and_then(|t| t.as_str()) {
        let matches = match expected_type {
            "object" => val.is_object(),
            "array" => val.is_array(),
            "string" => val.is_string(),
            "number" => val.is_number(),
            "integer" => val.is_i64() || val.is_u64() || (val.is_f64() && val.as_f64().unwrap().fract() == 0.0),
            "boolean" => val.is_boolean(),
            _ => true,
        };
        if !matches {
            return Err(format!("Schema validation failed: expected type '{expected_type}', found {:?}", val));
        }
    }
    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        if let Some(obj) = val.as_object() {
            for req in required {
                if let Some(field) = req.as_str() {
                    if !obj.contains_key(field) {
                        return Err(format!("Schema validation failed: missing required field '{field}'"));
                    }
                }
            }
        }
    }
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        if let Some(obj) = val.as_object() {
            for (field, prop_schema) in props {
                if let Some(field_val) = obj.get(field) {
                    Self::validate_json_schema(field_val, prop_schema)?;
                }
            }
        }
    }
    if let Some(enum_vals) = schema.get("enum").and_then(|e| e.as_array()) {
        if !enum_vals.contains(val) {
            return Err(format!("Schema validation failed: value {:?} is not one of enum {:?}", val, enum_vals));
        }
    }
    if let Some(items_schema) = schema.get("items") {
        if let Some(arr) = val.as_array() {
            for item in arr {
                Self::validate_json_schema(item, items_schema)?;
            }
        }
    }
    if let Some(min_val) = schema.get("minimum").and_then(|v| v.as_f64()) {
        if let Some(num) = val.as_f64() {
            if num < min_val {
                return Err(format!("Schema validation failed: numeric value {} < minimum bound {}", num, min_val));
            }
        }
    }
    if let Some(max_val) = schema.get("maximum").and_then(|v| v.as_f64()) {
        if let Some(num) = val.as_f64() {
            if num > max_val {
                return Err(format!("Schema validation failed: numeric value {} > maximum bound {}", num, max_val));
            }
        }
    }
    if let Some(min_len) = schema.get("minLength").and_then(|v| v.as_u64()) {
        if let Some(s) = val.as_str() {
            if (s.len() as u64) < min_len {
                return Err(format!("Schema validation failed: string length {} < minLength {}", s.len(), min_len));
            }
        }
    }
    if let Some(max_len) = schema.get("maxLength").and_then(|v| v.as_u64()) {
        if let Some(s) = val.as_str() {
            if (s.len() as u64) > max_len {
                return Err(format!("Schema validation failed: string length {} > maxLength {}", s.len(), max_len));
            }
        }
    }
    Ok(())
}

    fn invoke_tool_with_status(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        tool_name: &str,
        version: &str,
        input: &Value,
        status: SandboxStatus,
    ) -> Result<ToolExecutionReceipt> {
        // Refuse before mutating the workspace or attempting any execution.
        if let SandboxStatus::Unavailable { reason } = status {
            return Err(format!(
                "Sandbox execution refused: container runtime is unavailable ({reason}). Host Python substitution for untrusted code is prohibited."
            ));
        }
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let desc_json: String = db.query_row(
            "SELECT descriptor_json FROM agent_tools WHERE project=?1 AND tool_name=?2 AND version=?3",
            rusqlite::params![project_id, tool_name, version],
            |r| r.get(0),
        ).map_err(|_| format!("Tool {tool_name}:{version} is not registered in project {project_id}"))?;

        let desc: ToolDescriptor = serde_json::from_str(&desc_json).map_err(|e| e.to_string())?;

        // 1. Validate tool input against registered input_schema BEFORE execution
        Self::validate_json_schema(input, &desc.input_schema)?;

        // 2. Select execution directory: prefer registered immutable snapshot, or fallback to branch if hash matches
        let snapshot_dir = root.join("workflows").join(project_id).join("snapshots").join(&desc.source_hash);
        let (disposable_dir, exec_dir): (Option<std::path::PathBuf>, std::path::PathBuf) = if snapshot_dir.is_dir() {
            // Verify snapshot directory contents hash before invocation
            let current_hash = BranchWorkspaceManager::compute_dir_hash(&snapshot_dir).unwrap_or_default();
            if current_hash != desc.source_hash {
                return Err(format!(
                    "Snapshot directory content hash '{current_hash}' does not match registered tool source hash '{}'. Snapshot has been mutated or corrupted.",
                    desc.source_hash
                ));
            }
            // Execute in an isolated temporary staging copy so snapshot is never written to
            let staging = std::env::temp_dir().join(format!("mlr-snap-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()));
            BranchWorkspaceManager::copy_dir_recursive(&snapshot_dir, &staging)?;
            (Some(staging.clone()), staging)
        } else {
            let branch_dir = BranchWorkspaceManager::get_branch_dir(root, project_id, branch_id);
            let current_hash = BranchWorkspaceManager::compute_branch_hash(root, project_id, branch_id).unwrap_or_default();
            if current_hash != desc.source_hash {
                return Err(format!(
                    "Branch workspace hash '{current_hash}' does not match registered tool source hash '{}'. Cannot invoke tool against unverified working tree.",
                    desc.source_hash
                ));
            }
            // A matching mutable branch is only a source for disposable staging.
            // Revalidate the copy before execution, never run against the branch.
            let staging = std::env::temp_dir().join(format!("mlr-tool-{}-{}",std::process::id(),SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()));
            BranchWorkspaceManager::copy_dir_recursive(&branch_dir,&staging)?;
            if BranchWorkspaceManager::compute_dir_hash(&staging)?!=desc.source_hash {
                fs::remove_dir_all(&staging).map_err(|e|e.to_string())?;
                return Err("tool source changed during disposable staging".into());
            }
            (Some(staging.clone()), staging)
        };

        // 3. Write input to unique file to prevent concurrent collisions
        let unique_input_file = format!(".tool_input_{}_{}.json", SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos(), std::process::id());
        let input_path = exec_dir.join(&unique_input_file);
        fs::write(&input_path, serde_json::to_string(input).unwrap()).map_err(|e| e.to_string())?;

        let args = vec!["--input".to_string(), format!("/workspace/{unique_input_file}")];
        let limits = SandboxLimits {
            max_wall_ms: desc.max_wall_ms,
            ..Default::default()
        };

        let execution = SandboxRunner::execute(&exec_dir, &desc.entrypoint, &args, &limits, None);

        if input_path.exists() {
            let _ = fs::remove_file(input_path);
        }
        if let Some(ref disp) = disposable_dir {
            let _ = fs::remove_dir_all(disp);
        }
        let res = execution?;

        // 4. Validate tool output against registered output_schema AFTER execution
        let parsed_output = if res.exit_code == 0 {
            Some(serde_json::from_str::<Value>(&res.stdout).map_err(|e| format!("Tool returned invalid JSON: {e}"))?)
        } else { None };
        if let Some(ref out_val) = parsed_output {
            Self::validate_json_schema(out_val, &desc.output_schema)?;
        }

        Ok(ToolExecutionReceipt {
            tool_name: desc.name,
            version: desc.version,
            status: if res.exit_code == 0 { "succeeded".into() } else { "failed".into() },
            exit_code: res.exit_code,
            stdout: res.stdout,
            stderr: res.stderr,
            output_data: parsed_output,
            elapsed_ms: res.elapsed_ms,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_sandbox_refuses_tool_before_any_state_or_workspace_write() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("state-not-created");
        let error = ToolRegistry::invoke_tool_with_status(
            &root, "project", "branch", "tool", "v1", &json!({}),
            SandboxStatus::Unavailable { reason: "test runtime offline".into() },
        ).unwrap_err();
        assert!(error.contains("test runtime offline"));
        assert!(error.contains("Host Python substitution for untrusted code is prohibited"));
        assert!(!root.exists());
    }
}
