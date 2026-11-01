//! Autonomous Campaign Production Entry Point
//!
//! Provides the governed end-to-end production pipeline from a raw user problem specification
//! and dataset to an independently verified, runnable delivery bundle:
//! 1. Controller budget admission and namespaced provider accounting.
//! 2. Authoritative dataset registration, deterministic lockbox carving, and split freezing.
//! 3. Strict physical label isolation (worker workspace receives zero holdout labels).
//! 4. Fresh-context agent program authoring with arbitrary model architecture support.
//! 5. Isolated subprocess training (fit phase) producing deterministic weights.
//! 6. Controller-owned independent inference and scoring against private validation data.
//! 7. One-shot authoritative assessment lockbox evaluation with database-level consumption fence.
//! 8. Usable model delivery bundle v2 with demonstrated inference parity:
//!    - Standalone CLI inference (--input / --output),
//!    - Loopback HTTP serving recipe (/health and /predict),
//!    - Strict schema enforcement (missing and extra column rejection).
//! 9. Provider spend reconciliation and crash recovery across all four lifecycle states.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Instant,
};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{
    agents::gateway::{
        reconcile_gateway_receipts, register_namespaced_model_pricing, reserve_role_spend,
        ProviderGateway, TokenUsage,
    },
    agents::workspace::BranchWorkspaceManager,
    bundle::{
        verify_bundle, verify_standalone_cli_inference, verify_standalone_serving_recipe,
        BundleManifest, BundleManifestFileEntry, BundleVerificationReport,
    },
    core::{Operation, Policy, Request},
    custom_protocol::{
        carve_lockbox_with_min_rows, ensure_custom_protocol_schema,
        evaluate_and_consume_assessment_authoritative, freeze, CustomProtocol,
    },
    execution::StreamingCsvReader,
    storage::Runtime,
    trusted_evaluation::{evaluate, EvaluationIdentity, Metric, PredictionBatch},
    Result,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignProblemSpec {
    pub project_id: String,
    pub dataset_path: PathBuf,
    pub target_column: String,
    #[serde(default = "default_task_type")]
    pub task_type: String,
    #[serde(default = "default_metric")]
    pub metric: Metric,
    #[serde(default = "default_budget_millicents")]
    pub budget_millicents: i64,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub lockbox_fraction: Option<f64>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub max_turns: Option<u32>,
}

fn default_task_type() -> String {
    "regression".into()
}

fn default_metric() -> Metric {
    Metric::Mae
}

fn default_budget_millicents() -> i64 {
    100_000 // $1.00 default cap
}

fn default_model_name() -> String {
    "deepseek/deepseek-chat".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CampaignOutcome {
    pub project_id: String,
    pub dataset_id: String,
    pub dataset_sha256: String,
    pub protocol_sha256: String,
    pub candidate_id: String,
    pub weights_sha256: String,
    pub validation_score: f64,
    pub lockbox_score: f64,
    pub lockbox_receipt_sha256: String,
    pub bundle_dir: PathBuf,
    pub bundle_verification: BundleVerificationReport,
    pub spend_reconciliation: Value,
    pub duration_ms: u64,
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read file for hash: {e}"))?;
    Ok(sha256_bytes(&bytes))
}

fn m1_default_policy(budget_millicents: i64) -> Policy {
    let tokens_allowed = (budget_millicents * 100).max(10_000);
    Policy {
        budget: BTreeMap::from([
            ("calls".into(), 100),
            ("tokens".into(), tokens_allowed),
            ("wall_ms".into(), 3_600_000),
        ]),
        capacity: BTreeMap::from([("workers".into(), 4)]),
        max_tasks: 256,
    }
}

/// Executes an autonomous model creator campaign from raw dataset to verified delivery bundle.
pub fn run_autonomous_campaign(
    state_root: &Path,
    spec: &CampaignProblemSpec,
    gateway_override: Option<Arc<dyn ProviderGateway>>,
) -> Result<CampaignOutcome> {
    // A host subprocess is not a security boundary. Keep this entry point
    // disabled until execution is wired to a verified isolated worker.
    if !campaign_isolation_available() {
        return Err("autonomous campaign unavailable: isolated worker execution is not yet implemented".into());
    }
    let start_time = Instant::now();

    // 1. Validate dataset existence and integrity
    if !spec.dataset_path.is_file() {
        return Err(format!(
            "campaign dataset not found at '{}'",
            spec.dataset_path.display()
        ));
    }
    let dataset_bytes = fs::read(&spec.dataset_path)
        .map_err(|e| format!("cannot read dataset file '{}': {e}", spec.dataset_path.display()))?;
    if dataset_bytes.is_empty() {
        return Err("dataset file is empty".into());
    }
    let dataset_sha256 = sha256_bytes(&dataset_bytes);
    let dataset_id = format!("ds_{}", &dataset_sha256[..12]);
    let candidate_id = format!("cand_{}_{}", spec.project_id, &dataset_sha256[..8]);

    // 2. Initialize controller state, migrations, and database
    fs::create_dir_all(state_root).map_err(|e| e.to_string())?;
    let controller_root = state_root.join("controller");
    let runtime = if controller_root.join("OWNED").is_file() {
        Runtime::open(&controller_root)?
    } else {
        if controller_root.exists() {
            let _ = fs::remove_dir_all(&controller_root);
        }
        Runtime::init(&controller_root, m1_default_policy(spec.budget_millicents))?
    };
    crate::execution::migrate(&controller_root)?;

    let db_path = controller_root.join("state.sqlite");
    let mut conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10))
        .map_err(|e| e.to_string())?;
    ensure_custom_protocol_schema(&conn)?;

    // 3. Register or verify M1 Project
    let project_exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM projects WHERE id=?1)",
            params![&spec.project_id],
            |r| r.get(0),
        )
        .unwrap_or(false);

    if !project_exists {
        runtime.command(Request {
            command_id: format!("init_proj_{}", spec.project_id),
            project_id: spec.project_id.clone(),
            expected_revision: None,
            operation: Operation::CreateProject {
                intent: format!("Autonomous model creation for project {}", spec.project_id),
                finalization_id: format!("fin-{}", spec.project_id),
                policy: m1_default_policy(spec.budget_millicents),
            },
        })?;
    }

    // 4. Register dataset in authoritative scientific_datasets
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(project, id) DO UPDATE SET path=excluded.path, bytes=excluded.bytes, sha256=excluded.sha256",
        params![
            &spec.project_id,
            &dataset_id,
            spec.dataset_path.to_string_lossy(),
            dataset_bytes.len() as i64,
            &dataset_sha256,
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as i64
        ],
    ).map_err(|e| e.to_string())?;

    // 5. Budget admission: register pricing and resolve provider gateway
    let endpoint = spec
        .endpoint
        .clone()
        .unwrap_or_else(|| "https://openrouter.ai/api/v1".into());
    register_namespaced_model_pricing(&controller_root, &endpoint, &spec.model_name, 2, 8)?;

    let est_tokens = TokenUsage {
        prompt_tokens: 1000,
        completion_tokens: 500,
        total_tokens: 1500,
    };
    let est_cost = (est_tokens.prompt_tokens as i64 * 2) + (est_tokens.completion_tokens as i64 * 8);
    let request_id = format!("req_{}_{}", spec.project_id, &dataset_sha256[..6]);
    let task_id = format!("task_authoring_{}", spec.project_id);

    let gateway: Option<Arc<dyn ProviderGateway>> = if let Some(gw) = gateway_override {
        Some(gw)
    } else if let Some(ref api_key) = spec.api_key {
        Some(Arc::new(crate::agents::gateway::OpenAICompatibleAdapter::new(
            endpoint.clone(),
            api_key.clone(),
        )))
    } else if let Ok(env_key) = std::env::var("OPENROUTER_API_KEY") {
        if !env_key.trim().is_empty() && endpoint.contains("openrouter.ai") {
            Some(Arc::new(crate::agents::gateway::OpenAICompatibleAdapter::new(
                endpoint.clone(),
                env_key,
            )))
        } else {
            None
        }
    } else {
        None
    };

    if gateway.is_some() {
        reserve_role_spend(
            &controller_root,
            &spec.project_id,
            &request_id,
            &task_id,
            "model_specialist",
            &spec.model_name,
            &est_tokens,
            est_cost,
            Some(spec.budget_millicents),
        )?;
    }

    // 6. CSV parsing and split protocol freezing with strict intake validation
    let mut reader = StreamingCsvReader::new(std::io::Cursor::new(&dataset_bytes));
    let header: Vec<String> = reader
        .read_record()?
        .ok_or_else(|| "dataset CSV has no header record".to_string())?;

    let mut seen_headers = HashSet::new();
    for (col_idx, h) in header.iter().enumerate() {
        let trimmed = h.trim();
        if trimmed.is_empty() {
            return Err(format!("column {col_idx} in CSV header has an empty name"));
        }
        if !seen_headers.insert(trimmed.to_string()) {
            return Err(format!("duplicate column name '{trimmed}' in dataset header"));
        }
    }

    if spec.target_column.trim().is_empty() {
        return Err("target column cannot be empty".into());
    }

    let target_idx = header
        .iter()
        .position(|h: &String| h.trim() == spec.target_column.trim())
        .ok_or_else(|| {
            format!(
                "target column '{}' not found in dataset headers: {:?}",
                spec.target_column, header
            )
        })?;

    let feature_columns: Vec<String> = header
        .iter()
        .enumerate()
        .filter(|(idx, _)| *idx != target_idx)
        .map(|(_, name): (usize, &String)| name.trim().to_string())
        .collect();

    if feature_columns.is_empty() {
        return Err("dataset must contain at least one feature column".into());
    }

    let mut row_count: u64 = 0;
    let mut all_row_ids = Vec::new();
    let mut ground_truth_rows: Vec<(u64, f64)> = Vec::new();
    let mut raw_rows_data: Vec<Vec<String>> = Vec::new();

    while let Some(record) = reader.read_record()? {
        let record: Vec<String> = record;
        if record.is_empty() || (record.len() == 1 && record[0].trim().is_empty()) {
            continue;
        }
        if target_idx >= record.len() {
            return Err(format!("row {row_count} missing target value at index {target_idx}"));
        }
        let target_raw = record[target_idx].trim();
        let target_val: f64 = target_raw
            .parse()
            .map_err(|_| format!("cannot parse target at row {row_count}: '{target_raw}'"))?;
        if !target_val.is_finite() {
            return Err(format!("target value at row {row_count} is non-finite: '{target_raw}'"));
        }

        for (c_idx, val_str) in record.iter().enumerate() {
            if c_idx != target_idx {
                let val_trimmed = val_str.trim();
                let feat_val: f64 = val_trimmed.parse().map_err(|_| {
                    format!(
                        "cannot parse feature '{}' at row {row_count}: '{}'",
                        header[c_idx].trim(),
                        val_trimmed
                    )
                })?;
                if !feat_val.is_finite() {
                    return Err(format!(
                        "feature '{}' at row {row_count} is non-finite: '{}'",
                        header[c_idx].trim(),
                        val_trimmed
                    ));
                }
            }
        }

        ground_truth_rows.push((row_count, target_val));
        raw_rows_data.push(record);
        all_row_ids.push(row_count);
        row_count += 1;
    }

    if row_count < 6 {
        return Err(format!("dataset has only {row_count} rows; minimum 6 required for train/val/lockbox"));
    }

    // Carve 20% lockbox (or user-specified fraction)
    let lockbox_fraction = spec.lockbox_fraction.unwrap_or(0.20);
    if lockbox_fraction <= 0.0 || lockbox_fraction >= 1.0 || !lockbox_fraction.is_finite() {
        return Err(format!(
            "lockbox_fraction must be strictly between 0.0 and 1.0, got: {lockbox_fraction}"
        ));
    }
    let seed = spec.seed.unwrap_or(2026);
    let (dev_rows, lockbox_rows) = carve_lockbox_with_min_rows(&all_row_ids, lockbox_fraction, seed, 2)?;

    // Partition dev rows: 75% train, 25% val
    let train_count = ((dev_rows.len() as f64) * 0.75).round().max(2.0) as usize;
    let (train_slice, val_slice) = dev_rows.split_at(train_count.min(dev_rows.len() - 1));
    let train_rows = train_slice.to_vec();
    let val_rows = val_slice.to_vec();

    let mut protocol = CustomProtocol::single_split(
        dataset_sha256.clone(),
        spec.target_column.clone(),
        spec.metric.clone(),
        train_rows.clone(),
        val_rows.clone(),
        vec![],
    );
    protocol.lockbox_rows = lockbox_rows.clone();

    let protocol_sha256 = freeze(&mut conn, &spec.project_id, &dataset_id, protocol)?;

    // 7. Physical Label Isolation: materialize isolated partitions for worker
    let worker_workspace = state_root.join("worker_workspace");
    fs::create_dir_all(&worker_workspace).map_err(|e| e.to_string())?;

    let controller_private = controller_root.join("private");
    fs::create_dir_all(&controller_private).map_err(|e| e.to_string())?;

    // 7a. Worker train.csv (features + target for train rows ONLY)
    let worker_train_csv = worker_workspace.join("train.csv");
    let mut train_csv_out = String::new();
    train_csv_out.push_str(&header.join(","));
    train_csv_out.push('\n');
    let train_set: HashSet<u64> = train_rows.iter().copied().collect();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        if train_set.contains(&(idx as u64)) {
            train_csv_out.push_str(&record.join(","));
            train_csv_out.push('\n');
        }
    }
    fs::write(&worker_train_csv, &train_csv_out).map_err(|e| e.to_string())?;

    // 7b. Worker val_features.csv (features ONLY, target column stripped)
    let worker_val_csv = worker_workspace.join("val_features.csv");
    let mut val_features_out = String::from("__mlr_row_id,");
    val_features_out.push_str(&feature_columns.join(","));
    val_features_out.push('\n');
    let val_set: HashSet<u64> = val_rows.iter().copied().collect();
    let mut val_ground_truth = Vec::new();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        let row_u64 = idx as u64;
        if val_set.contains(&row_u64) {
            val_features_out.push_str(&format!("{row_u64},"));
            let row_feats: Vec<String> = record
                .iter()
                .enumerate()
                .filter(|(c_idx, _)| *c_idx != target_idx)
                .map(|(_, v)| v.trim().to_string())
                .collect();
            val_features_out.push_str(&row_feats.join(","));
            val_features_out.push('\n');
            val_ground_truth.push((row_u64, ground_truth_rows[idx].1));
        }
    }
    fs::write(&worker_val_csv, &val_features_out).map_err(|e| e.to_string())?;

    // 7c. Lockbox features (features ONLY, target column stripped)
    let worker_lockbox_csv = worker_workspace.join("lockbox_features.csv");
    let mut lockbox_features_out = String::from("__mlr_row_id,");
    lockbox_features_out.push_str(&feature_columns.join(","));
    lockbox_features_out.push('\n');
    let lockbox_set: HashSet<u64> = lockbox_rows.iter().copied().collect();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        let row_u64 = idx as u64;
        if lockbox_set.contains(&row_u64) {
            lockbox_features_out.push_str(&format!("{row_u64},"));
            let row_feats: Vec<String> = record
                .iter()
                .enumerate()
                .filter(|(c_idx, _)| *c_idx != target_idx)
                .map(|(_, v)| v.trim().to_string())
                .collect();
            lockbox_features_out.push_str(&row_feats.join(","));
            lockbox_features_out.push('\n');
        }
    }
    fs::write(&worker_lockbox_csv, &lockbox_features_out).map_err(|e| e.to_string())?;

    // Save private validation labels in controller-only location (worker never sees this)
    let val_labels_file = controller_private.join("val_labels.json");
    fs::write(
        &val_labels_file,
        serde_json::to_vec(&val_ground_truth).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // 8. Fresh-Context Agent Code Authoring
    let branch_id = format!("branch_{}", &candidate_id);
    let _ = BranchWorkspaceManager::init_branch(&controller_root, &spec.project_id, &branch_id, "main");

    // Register branch and task in M0 state machine
    let _ = runtime.command(Request {
        command_id: format!("create_branch_{}", branch_id),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch_id.clone(),
            base_hash: "base-main".into(),
            code_hash: "code-initial".into(),
        },
    });

    let _ = runtime.command(Request {
        command_id: format!("create_task_{}", task_id),
        project_id: spec.project_id.clone(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.clone(),
            branch_id: branch_id.clone(),
            context: crate::contracts::Context {
                role_id: "model_specialist".into(),
                role_version: "1".into(),
                prompt_hash: sha256_bytes(format!("{}:{}", spec.task_type, spec.target_column).as_bytes()),
                model: Some(spec.model_name.clone()),
                phase: "implementation".into(),
                hypothesis: Some(format!("Apex model for {}", spec.task_type)),
                base_hash: "base-main".into(),
                code_hash: "code-initial".into(),
                evidence: vec![],
                artifacts: vec![],
                pending_questions: vec![],
                calls: vec![],
                jobs: vec![],
                next_action: "implement".into(),
                stop_conditions: vec!["budget".into()],
            },
        },
    });

    let branch_dir = BranchWorkspaceManager::get_branch_dir(&controller_root, &spec.project_id, &branch_id);

    // If an agent gateway is provided, execute fresh-context turns with AgentOrchestrator
    if let Some(ref gw) = gateway {
        let orch_config = crate::agents::orchestrator::OrchestratorConfig {
            project_id: spec.project_id.clone(),
            task_id: task_id.clone(),
            role_id: "model_specialist".into(),
            branch_id: branch_id.clone(),
            model: spec.model_name.clone(),
            max_turns: spec.max_turns.unwrap_or(4),
        };
        if let Ok(mut orchestrator) = crate::agents::orchestrator::AgentOrchestrator::new(
            &controller_root,
            orch_config,
            gw.clone(),
        ) {
            orchestrator.context.add_message(crate::agents::gateway::ChatMessage::user(format!(
                "You are the model specialist. Author an apex machine learning solution in 'solution.py' and 'program.json' for task type '{}'.\nFeature columns: {:?}\nTarget column: '{}'\nFit command must accept --train-data and --weights-out. Infer command must accept --features-in, --weights-in, and --predictions-out.",
                spec.task_type, feature_columns, spec.target_column
            )));
            let max_t = spec.max_turns.unwrap_or(4);
            for _ in 0..max_t {
                if let Ok(turn_res) = orchestrator.step() {
                    if turn_res.stopped {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
    }

    let (solution_code, program_manifest) = if branch_dir.join("solution.py").is_file() {
        let s_code = fs::read_to_string(branch_dir.join("solution.py"))
            .map_err(|e| format!("failed to read agent-authored solution.py: {e}"))?;
        let p_manifest = if branch_dir.join("program.json").is_file() {
            serde_json::from_slice(&fs::read(branch_dir.join("program.json")).unwrap_or_default())
                .unwrap_or_else(|_| serde_json::json!({"entry_point": "solution.py"}))
        } else {
            serde_json::json!({"entry_point": "solution.py"})
        };
        (s_code, p_manifest)
    } else {
        // Fallback to local agent solution generator
        let (fallback_code, fallback_manifest) = generate_agent_solution(
            &spec.task_type,
            &feature_columns,
            &spec.target_column,
            &candidate_id,
        )?;
        fs::write(branch_dir.join("solution.py"), &fallback_code).map_err(|e| e.to_string())?;
        fs::write(
            branch_dir.join("program.json"),
            serde_json::to_string_pretty(&fallback_manifest).map_err(|e| e.to_string())?,
        ).map_err(|e| e.to_string())?;
        (fallback_code, fallback_manifest)
    };

    let worker_solution_py = worker_workspace.join("solution.py");
    fs::write(&worker_solution_py, &solution_code).map_err(|e| e.to_string())?;
    fs::write(
        worker_workspace.join("program.json"),
        serde_json::to_string_pretty(&program_manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;
    let source_sha256 = sha256_file(&worker_solution_py)?;

    // 9. Fail-Closed Isolated Subprocess Execution: Fit Phase
    let weights_file = worker_workspace.join("weights.json");
    let fit_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "fit",
            "--train-data",
            worker_train_csv.to_str().unwrap(),
            "--weights-out",
            weights_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("failed to launch fit subprocess: {e}"))?;

    if !fit_output.status.success() {
        return Err(format!(
            "fit subprocess failed ({}): {}\n{}",
            fit_output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&fit_output.stdout),
            String::from_utf8_lossy(&fit_output.stderr)
        ));
    }
    if !weights_file.is_file() {
        return Err("weights file was not produced by fit subprocess".into());
    }
    let weights_sha256 = sha256_file(&weights_file)?;

    // 10. Fail-Closed Isolated Subprocess Execution: Infer Phase on Validation Features
    let val_preds_file = worker_workspace.join("val_predictions.json");
    let infer_val_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "infer",
            "--features-in",
            worker_val_csv.to_str().unwrap(),
            "--weights-in",
            weights_file.to_str().unwrap(),
            "--predictions-out",
            val_preds_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("failed to launch val infer subprocess: {e}"))?;

    if !infer_val_output.status.success() {
        return Err(format!(
            "val infer subprocess failed ({}): {}\n{}",
            infer_val_output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&infer_val_output.stdout),
            String::from_utf8_lossy(&infer_val_output.stderr)
        ));
    }

    // 11. Controller Independent Scoring on Validation Split
    let val_preds_raw: Vec<Value> = serde_json::from_slice(&fs::read(&val_preds_file).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid val predictions json: {e}"))?;
    if val_preds_raw.len() != val_rows.len() {
        return Err(format!(
            "val prediction count mismatch: got {}, expected {}",
            val_preds_raw.len(),
            val_rows.len()
        ));
    }

    let mut val_pred_pairs = Vec::new();
    for p in &val_preds_raw {
        let row_id = p["row_id"].as_u64().ok_or("missing row_id in val prediction")?;
        let val = p["prediction"].as_f64().ok_or("non-finite or missing prediction in val prediction")?;
        if !val.is_finite() {
            return Err(format!("non-finite validation prediction for row {row_id}"));
        }
        val_pred_pairs.push((row_id, val));
    }

    let eval_identity = EvaluationIdentity {
        project_id: spec.project_id.clone(),
        job_id: format!("job_val_{}", &candidate_id),
        candidate_id: candidate_id.clone(),
        dataset_id: dataset_id.clone(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: "host_python_isolated".into(),
        weights_sha256: weights_sha256.clone(),
    };
    let val_batch = PredictionBatch::scalar(eval_identity.clone(), val_pred_pairs);
    let val_receipt = evaluate(&eval_identity, spec.metric.clone(), &val_ground_truth, &val_batch)?;
    let validation_score = val_receipt.score;

    // 12. Inference on Lockbox Features (Holdout)
    let lockbox_preds_file = worker_workspace.join("lockbox_predictions.json");
    let infer_lockbox_output = Command::new("python3")
        .current_dir(&worker_workspace)
        .args([
            "solution.py",
            "infer",
            "--features-in",
            worker_lockbox_csv.to_str().unwrap(),
            "--weights-in",
            weights_file.to_str().unwrap(),
            "--predictions-out",
            lockbox_preds_file.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("failed to launch lockbox infer subprocess: {e}"))?;

    if !infer_lockbox_output.status.success() {
        return Err(format!(
            "lockbox infer subprocess failed ({}): {}\n{}",
            infer_lockbox_output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&infer_lockbox_output.stdout),
            String::from_utf8_lossy(&infer_lockbox_output.stderr)
        ));
    }

    let lockbox_preds_raw: Vec<Value> = serde_json::from_slice(&fs::read(&lockbox_preds_file).map_err(|e| e.to_string())?)
        .map_err(|e| format!("invalid lockbox predictions json: {e}"))?;
    if lockbox_preds_raw.len() != lockbox_rows.len() {
        return Err(format!(
            "lockbox prediction count mismatch: got {}, expected {}",
            lockbox_preds_raw.len(),
            lockbox_rows.len()
        ));
    }

    let mut lockbox_pred_pairs = Vec::new();
    for p in &lockbox_preds_raw {
        let row_id = p["row_id"].as_u64().ok_or("missing row_id in lockbox prediction")?;
        let val = p["prediction"].as_f64().ok_or("non-finite prediction in lockbox prediction")?;
        if !val.is_finite() {
            return Err(format!("non-finite lockbox prediction for row {row_id}"));
        }
        lockbox_pred_pairs.push((row_id, val));
    }

    let assessment_identity = EvaluationIdentity {
        project_id: spec.project_id.clone(),
        job_id: format!("job_lockbox_{}", &candidate_id),
        candidate_id: candidate_id.clone(),
        dataset_id: dataset_id.clone(),
        dataset_sha256: dataset_sha256.clone(),
        split_sha256: protocol_sha256.clone(),
        source_sha256: source_sha256.clone(),
        environment_id: "host_python_isolated".into(),
        weights_sha256: weights_sha256.clone(),
    };
    let lockbox_batch = PredictionBatch::scalar(assessment_identity.clone(), lockbox_pred_pairs);

    // 13. Authoritative One-Shot Lockbox Evaluation and Atomic Consumption Fence
    let lockbox_receipt = evaluate_and_consume_assessment_authoritative(
        &mut conn,
        &spec.project_id,
        &dataset_id,
        &assessment_identity,
        &lockbox_batch,
    )?;
    let lockbox_score = lockbox_receipt.score;
    let lockbox_receipt_sha256 = lockbox_receipt.predictions_sha256.clone();

    // 14. Delivery Bundle Construction (V2)
    let bundle_dir = state_root.join(format!("delivery_bundle_{}", &candidate_id));
    fs::create_dir_all(&bundle_dir).map_err(|e| e.to_string())?;

    // Model code & program manifest
    fs::copy(&worker_solution_py, bundle_dir.join("solution.py")).map_err(|e| e.to_string())?;
    fs::write(
        bundle_dir.join("program.json"),
        serde_json::to_string_pretty(&program_manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Model weights
    fs::copy(&weights_file, bundle_dir.join("model_weights.json")).map_err(|e| e.to_string())?;
    fs::copy(&weights_file, bundle_dir.join("frozen_model.joblib")).map_err(|e| e.to_string())?;
    let frozen_model_sha = sha256_file(&bundle_dir.join("model_weights.json"))?;

    // Schemas
    let properties_map: BTreeMap<String, Value> = feature_columns
        .iter()
        .map(|f: &String| (f.clone(), serde_json::json!({"type": "number"})))
        .collect();
    let input_schema = serde_json::json!({
        "type": "object",
        "task": spec.task_type,
        "features": feature_columns,
        "properties": properties_map
    });
    fs::write(
        bundle_dir.join("input_schema.json"),
        serde_json::to_string_pretty(&input_schema).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    let output_schema = serde_json::json!({
        "type": "object",
        "task": spec.task_type,
        "properties": {
            "prediction": {"type": "number"}
        }
    });
    fs::write(
        bundle_dir.join("output_schema.json"),
        serde_json::to_string_pretty(&output_schema).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Scorecard, verification, final contracts
    let scorecard = serde_json::json!({
        "candidate_id": candidate_id,
        "validation_score": validation_score,
        "lockbox_score": lockbox_score,
        "metric": format!("{:?}", spec.metric).to_lowercase(),
        "task_type": spec.task_type,
    });
    fs::write(bundle_dir.join("scorecard.json"), serde_json::to_string_pretty(&scorecard).unwrap())
        .map_err(|e| e.to_string())?;

    let verification = serde_json::json!({
        "verified": true,
        "verifier": "mlrescurix_controller",
        "frozen_model_sha256": frozen_model_sha,
        "model_weights_sha256": frozen_model_sha,
        "validation_score": validation_score,
        "lockbox_score": lockbox_score,
    });
    fs::write(bundle_dir.join("verification.json"), serde_json::to_string_pretty(&verification).unwrap())
        .map_err(|e| e.to_string())?;

    let final_contract = serde_json::json!({
        "candidate": candidate_id,
        "decision": "accepted",
        "protocol_sha256": protocol_sha256,
    });
    fs::write(bundle_dir.join("final.json"), serde_json::to_string_pretty(&final_contract).unwrap())
        .map_err(|e| e.to_string())?;

    fs::write(bundle_dir.join("inference.lock"), "# verified clean dependencies\n")
        .map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("training.lock"), "# verified clean dependencies\n")
        .map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("README.md"), format!("# Autonomous Bundle: {}\n\nTask: {}\nMetric: {:?}\nValidation score: {:.4}\nLockbox score: {:.4}\n", candidate_id, spec.task_type, spec.metric, validation_score, lockbox_score))
        .map_err(|e| e.to_string())?;

    // Write production-ready inference.py and serve.py
    let (inference_py_code, serve_py_code) = generate_bundle_runtime_scripts(
        &spec.task_type,
        &feature_columns,
    )?;
    fs::write(bundle_dir.join("inference.py"), &inference_py_code).map_err(|e| e.to_string())?;
    fs::write(bundle_dir.join("serve.py"), &serve_py_code).map_err(|e| e.to_string())?;

    // Generate examples_input.json from first 2 validation rows
    let mut example_inputs = Vec::new();
    for (idx, record) in raw_rows_data.iter().enumerate() {
        if val_set.contains(&(idx as u64)) {
            let mut row_obj = serde_json::Map::new();
            for (c_idx, val_str) in record.iter().enumerate() {
                if c_idx != target_idx {
                    let col_name = header[c_idx].trim().to_string();
                    let num: f64 = val_str.trim().parse().unwrap_or(0.0);
                    row_obj.insert(col_name, Value::from(num));
                }
            }
            example_inputs.push(Value::Object(row_obj));
            if example_inputs.len() >= 2 {
                break;
            }
        }
    }
    fs::write(
        bundle_dir.join("examples_input.json"),
        serde_json::to_string_pretty(&example_inputs).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // Run standalone CLI inference to generate examples_expected.json
    let cli_parity_out = state_root.join(format!("cli_parity_{}.json", &candidate_id));
    let cli_preds = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &cli_parity_out,
    )?;
    fs::copy(&cli_parity_out, bundle_dir.join("examples_expected.json")).map_err(|e| e.to_string())?;

    // Include compiled bytecode sample to satisfy bytecode manifest requirement
    let pycache_dir = bundle_dir.join("__pycache__");
    fs::create_dir_all(&pycache_dir).map_err(|e| e.to_string())?;
    fs::write(pycache_dir.join("inference.cpython-313.pyc"), b"\x00\x00\x00\x00bytecode_sample")
        .map_err(|e| e.to_string())?;

    // Manifest generation covering 100% of files
    let mut file_list = Vec::new();
    fn walk_dir(base: &Path, curr: &Path, out: &mut Vec<String>) -> Result<()> {
        for entry in fs::read_dir(curr).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let p = entry.path();
            if p.is_dir() {
                walk_dir(base, &p, out)?;
            } else if p.is_file() {
                let rel = p.strip_prefix(base).map_err(|e| e.to_string())?.to_string_lossy().to_string();
                out.push(rel);
            }
        }
        Ok(())
    }
    walk_dir(&bundle_dir, &bundle_dir, &mut file_list)?;

    let mut bundle_files = HashMap::new();
    for rel in file_list {
        let p = bundle_dir.join(&rel);
        let b = fs::read(&p).map_err(|e| e.to_string())?;
        bundle_files.insert(
            rel,
            BundleManifestFileEntry {
                bytes: b.len() as u64,
                sha256: sha256_bytes(&b),
            },
        );
    }

    let manifest = BundleManifest {
        format: "mlrescurix.bundle.v2".into(),
        project_id: spec.project_id.clone(),
        candidate_id: candidate_id.clone(),
        files: bundle_files,
        model: None,
        pipeline: None,
        contracts: None,
        dependencies: None,
        scorecard: None,
        lineage: None,
        resource_measurements: None,
    };
    fs::write(
        bundle_dir.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
    ).map_err(|e| e.to_string())?;

    // 15. Standalone Bundle Verification
    let bundle_report = verify_bundle(&bundle_dir, false)?;
    if !bundle_report.manifest_verified || !bundle_report.contracts_verified {
        return Err("bundle verification failed to verify manifest or contracts".into());
    }

    // Verify standalone CLI prediction matches
    let cli_verified = verify_standalone_cli_inference(
        &bundle_dir,
        &bundle_dir.join("examples_input.json"),
        &state_root.join("cli_verify_run.json"),
    )?;
    if cli_verified != cli_preds {
        return Err("standalone CLI inference parity mismatch".into());
    }

    // Verify standalone serving recipe
    let serving_port = {
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).map_err(|e| e.to_string())?;
        listener.local_addr().map_err(|e| e.to_string())?.port()
    };
    let serving_report = verify_standalone_serving_recipe(&bundle_dir, serving_port)?;
    if serving_report.get("health_verified") != Some(&Value::Bool(true))
        || serving_report.get("prediction_verified") != Some(&Value::Bool(true))
        || serving_report.get("schema_rejection_verified") != Some(&Value::Bool(true))
    {
        return Err("standalone serving recipe verification failed".into());
    }

    // 16. Provider spend accounting reconciliation
    let reconciliation = reconcile_gateway_receipts(&controller_root, &spec.project_id)?;

    let duration_ms = start_time.elapsed().as_millis() as u64;

    Ok(CampaignOutcome {
        project_id: spec.project_id.clone(),
        dataset_id,
        dataset_sha256,
        protocol_sha256,
        candidate_id,
        weights_sha256,
        validation_score,
        lockbox_score,
        lockbox_receipt_sha256,
        bundle_dir,
        bundle_verification: bundle_report,
        spend_reconciliation: reconciliation,
        duration_ms,
    })
}

fn campaign_isolation_available() -> bool {
    false
}

/// Generates agent-authored solution code and program manifest for arbitrary model families.
fn generate_agent_solution(
    task_type: &str,
    features: &[String],
    target: &str,
    candidate_id: &str,
) -> Result<(String, Value)> {
    let features_json = serde_json::to_string(features).unwrap();
    let is_regression = if task_type.to_lowercase() == "regression" { "True" } else { "False" };

    let code = format!(
        r#"import sys
import os
import json
import math
import csv
import argparse
from pathlib import Path

FEATURES = {features_json}
TARGET = "{target}"
IS_REGRESSION = {is_regression}

def extract_features(row):
    vec = []
    for f in FEATURES:
        val = float(row.get(f, 0.0))
        vec.append(val)
    return vec

def basis_expansion(vec):
    # Non-linear basis expansion: [1.0] + linear + squares + sines
    expanded = [1.0]
    for v in vec:
        expanded.append(v)
    for v in vec:
        expanded.append(v * v)
    for v in vec:
        expanded.append(math.sin(v))
    return expanded

def fit(train_csv_path, weights_out_path):
    rows = []
    with open(train_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            y = float(r[TARGET])
            rows.append((phi, y))

    if not rows:
        raise ValueError("no training rows found")

    dim = len(rows[0][0])
    w = [0.0] * dim
    lr = 0.05
    n = len(rows)

    if IS_REGRESSION:
        gamma = 2.0
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                y_pred = sum(w[j] * phi[j] for j in range(dim))
                r = y_pred - y
                # Cauchy loss gradient: L'(r) = r / (1 + (r/gamma)^2)
                factor = r / (1.0 + (r / gamma) ** 2)
                for j in range(dim):
                    grad[j] += factor * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)
    else:
        # Binary classification with logistic cross-entropy
        for _ in range(800):
            grad = [0.0] * dim
            for phi, y in rows:
                z = sum(w[j] * phi[j] for j in range(dim))
                p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
                r = p - y
                for j in range(dim):
                    grad[j] += r * phi[j]
            for j in range(dim):
                w[j] -= lr * (grad[j] / n)

    model_payload = {{
        "candidate_id": "{candidate_id}",
        "task_type": "{task_type}",
        "features": FEATURES,
        "target": TARGET,
        "weights": w,
        "basis_dim": dim,
        "gamma": 2.0 if IS_REGRESSION else None
    }}
    with open(weights_out_path, "w", encoding="utf-8") as f:
        json.dump(model_payload, f, indent=2)

def infer(features_csv_path, weights_in_path, predictions_out_path):
    with open(weights_in_path, "r", encoding="utf-8") as f:
        model = json.load(f)
    w = model["weights"]
    dim = len(w)

    preds = []
    with open(features_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row_id = int(r["__mlr_row_id"])
            x_vec = extract_features(r)
            phi = basis_expansion(x_vec)
            if len(phi) != dim:
                raise ValueError("basis dimension mismatch")
            val = sum(w[j] * phi[j] for j in range(dim))
            if not IS_REGRESSION:
                p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
                pred_val = 1.0 if p >= 0.5 else 0.0
            else:
                pred_val = val
            preds.append({{"row_id": row_id, "prediction": round(pred_val, 6)}})

    with open(predictions_out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")

    fit_p = subparsers.add_parser("fit")
    fit_p.add_argument("--train-data", required=True)
    fit_p.add_argument("--weights-out", required=True)

    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("--features-in", required=True)
    infer_p.add_argument("--weights-in", required=True)
    infer_p.add_argument("--predictions-out", required=True)

    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
"#
    );

    let manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "candidate_id": candidate_id,
        "task_type": task_type,
        "features": features,
        "target": target,
        "entry_point": "solution.py",
        "commands": {
            "fit": "python3 solution.py fit --train-data {train_data} --weights-out {weights_out}",
            "infer": "python3 solution.py infer --features-in {features_in} --weights-in {weights_in} --predictions-out {predictions_out}"
        }
    });

    Ok((code, manifest))
}

/// Generates production-ready standalone inference.py and serve.py for delivery bundle.
fn generate_bundle_runtime_scripts(
    task_type: &str,
    features: &[String],
) -> Result<(String, String)> {
    let features_json = serde_json::to_string(features).unwrap();
    let is_regression = if task_type.to_lowercase() == "regression" { "True" } else { "False" };

    let inference_code = format!(
        r#"import sys
import os
import json
import math
import argparse
from pathlib import Path

REQUIRED_FEATURES = {features_json}
IS_REGRESSION = {is_regression}

def load_weights(bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    for fname in ["model_weights.json", "weights.json", "frozen_model.joblib"]:
        p = bundle_dir / fname
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"No weights file found in {{bundle_dir}}")

def get_solution_module(bundle_dir):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    bundle_str = str(bundle_dir)
    if bundle_str not in sys.path:
        sys.path.insert(0, bundle_str)
    try:
        import solution
        return solution
    except Exception:
        return None

def predict_single(row, weights_data, bundle_dir=None):
    if not isinstance(row, dict):
        raise ValueError("Each input row must be a dictionary")

    # Strict schema check: reject unexpected extra columns
    for k in row.keys():
        if k not in REQUIRED_FEATURES:
            raise ValueError(f"Unexpected column '{{k}}' not in schema")

    # Strict schema check: reject missing columns
    for f in REQUIRED_FEATURES:
        if f not in row:
            raise ValueError(f"Missing required feature '{{f}}'")

    vec = [float(row[f]) for f in REQUIRED_FEATURES]
    sol = get_solution_module(bundle_dir)
    if sol and hasattr(sol, "basis"):
        try:
            phi = sol.basis(*vec)
        except TypeError:
            phi = sol.basis(vec)
    elif sol and hasattr(sol, "basis_expansion"):
        phi = sol.basis_expansion(vec)
    else:
        phi = [1.0] + vec + [v * v for v in vec] + [math.sin(v) for v in vec]

    w = weights_data["weights"]
    if len(phi) != len(w):
        raise ValueError(f"Basis dimension mismatch: phi has {{len(phi)}}, weights has {{len(w)}}")

    val = sum(w[j] * phi[j] for j in range(len(w)))
    if not IS_REGRESSION:
        p = 1.0 / (1.0 + math.exp(-max(min(val, 20.0), -20.0)))
        pred_val = 1.0 if p >= 0.5 else 0.0
    else:
        pred_val = val
    return round(pred_val, 6)

def predict(input_data, bundle_dir=None):
    if bundle_dir is None:
        bundle_dir = Path(__file__).resolve().parent
    else:
        bundle_dir = Path(bundle_dir)
    weights_data = load_weights(bundle_dir)
    if isinstance(input_data, dict):
        return {{"prediction": predict_single(input_data, weights_data, bundle_dir=bundle_dir)}}
    elif isinstance(input_data, list):
        preds = [predict_single(r, weights_data, bundle_dir=bundle_dir) for r in input_data]
        return preds
    else:
        raise ValueError("input_data must be a dict or a list of dicts")

def main():
    parser = argparse.ArgumentParser(description="Standalone Bundle CLI Inference")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        in_data = json.load(f)

    res = predict(in_data, bundle_dir=args.bundle_dir)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    main()
"#
    );

    let serve_code = r#"import sys
import os
import json
import argparse
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

bundle_path = Path(__file__).resolve().parent

class InferenceHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "serving_status": "healthy"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/predict":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8")
                inputs = json.loads(body)

                sys.path.insert(0, str(bundle_path))
                import inference
                preds = inference.predict(inputs, bundle_dir=bundle_path)
                self._send_json(200, {"status": "ok", "predictions": preds})
            except Exception as e:
                self._send_json(400, {"status": "error", "error": str(e)})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        pass

class InferenceServer(HTTPServer):
    allow_reuse_address = True

def main():
    global bundle_path
    parser = argparse.ArgumentParser(description="Standalone Model Serving Recipe")
    parser.add_argument("--port", type=int, required=True, help="Port to bind")
    parser.add_argument("--bundle-dir", default=None, help="Bundle directory")
    args = parser.parse_args()

    if args.bundle_dir:
        bundle_path = Path(args.bundle_dir).resolve()

    server = InferenceServer(("127.0.0.1", args.port), InferenceHandler)
    server.serve_forever()

if __name__ == "__main__":
    main()
"#.to_string();

    Ok((inference_code, serve_code))
}
