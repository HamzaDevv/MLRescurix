use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    core::Policy,
    execution::{self, JobSpec, Probe, Supervisor},
    Result,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn sql<T>(x: rusqlite::Result<T>) -> Result<T> {
    x.map_err(|e| e.to_string())
}

fn valid_id(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 128 || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b)) {
        Err("invalid identity".into())
    } else {
        Ok(())
    }
}

fn hash_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

fn default_fidelities() -> Vec<f64> { vec![0.5, 1.0] }
fn default_recipes() -> Vec<String> { vec!["linear".into(), "tree".into()] }
fn default_max_trials() -> u32 { 10 }
fn default_max_screen_trials() -> u32 { 6 }
fn default_max_full_trials() -> u32 { 4 }
fn default_search_wall_ms() -> u64 { 300_000 }
fn default_job_timeout_ms() -> u64 { 30_000 }
fn default_finalization_allowance_ms() -> u64 { 60_000 }
fn default_max_repeats() -> u32 { 3 }
fn default_tie_tolerance() -> f64 { 0.005 }
fn default_concurrency() -> u32 { 1 }
fn default_max_retries() -> u32 { 2 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpec {
    pub session_id: String,
    pub project_id: String,
    pub dataset_id: String,
    pub target: String,
    pub task: String,
    pub metric: String,
    pub direction: String,
    pub split: String,
    #[serde(default = "default_fidelities")]
    pub fidelities: Vec<f64>,
    #[serde(default = "default_recipes")]
    pub recipes: Vec<String>,
    #[serde(default = "default_max_trials")]
    pub max_trials: u32,
    #[serde(default = "default_max_screen_trials")]
    pub max_screen_trials: u32,
    #[serde(default = "default_max_full_trials")]
    pub max_full_trials: u32,
    #[serde(default = "default_search_wall_ms")]
    pub search_wall_ms: u64,
    #[serde(default = "default_job_timeout_ms")]
    pub job_timeout_ms: u64,
    #[serde(default = "default_finalization_allowance_ms")]
    pub finalization_allowance_ms: u64,
    #[serde(default = "default_max_repeats")]
    pub max_repeats: u32,
    #[serde(default = "default_tie_tolerance")]
    pub tie_tolerance: f64,
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default)]
    pub constraints: Option<Value>,
}

impl SearchSpec {
    pub fn validate(&self) -> Result<()> {
        valid_id(&self.session_id)?;
        valid_id(&self.project_id)?;
        valid_id(&self.dataset_id)?;
        if self.target.is_empty() || self.target.len() > 128 { return Err("invalid target column".into()); }
        if !matches!(self.task.as_str(), "binary_classification" | "multiclass_classification" | "regression") {
            return Err("unsupported task".into());
        }
        if !matches!(self.direction.as_str(), "maximize" | "minimize") {
            return Err("direction must be maximize or minimize".into());
        }
        if !matches!(self.split.as_str(), "stratified" | "random" | "grouped" | "chronological") {
            return Err("unsupported split policy".into());
        }
        if self.fidelities.is_empty() || self.fidelities.iter().any(|&f| f <= 0.0 || f > 1.0) {
            return Err("fidelities must be non-empty and in (0, 1]".into());
        }
        if self.recipes.is_empty() || self.recipes.iter().any(|r| !matches!(r.as_str(), "dummy" | "linear" | "tree" | "hist_boosting")) {
            return Err("recipes must be a subset of dummy, linear, tree, hist_boosting".into());
        }
        if self.max_trials == 0 || self.max_trials > 1000 {
            return Err("max_trials must be in 1..=1000".into());
        }
        if self.search_wall_ms < 1000 {
            return Err("search_wall_ms must be at least 1000ms".into());
        }
        if self.job_timeout_ms < 1000 {
            return Err("job_timeout_ms must be at least 1000ms".into());
        }
        if self.concurrency != 1 {
            return Err("default concurrency of 1 active search trial is enforced".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSpec {
    pub experiment_id: String,
    pub session_id: String,
    pub project_id: String,
    pub recipe: String,
    pub parameters: Value,
    pub seed: u64,
    pub fidelity: f64,
    pub parent_experiment_id: Option<String>,
    pub node_type: String, // "baseline", "screening", "full_fidelity", "repeat"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub experiment_id: String,
    pub status: String,
    pub primary_metric: f64,
    pub metrics: Value,
    pub constraints: Value,
    pub is_feasible: bool,
    pub train_ms: f64,
    pub model_sha256: String,
    pub predictions_sha256: String,
    pub artifact_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContenderSummary {
    pub candidate_id: String,
    pub experiment_id: String,
    pub recipe: String,
    pub parameters: Value,
    pub mean_metric: f64,
    pub metric_std: f64,
    pub metric_min: f64,
    pub metric_max: f64,
    pub seeds: Vec<u64>,
    pub scores: Vec<f64>,
    pub is_feasible: bool,
    pub warm_p95_ms: Option<f64>,
    pub model_bytes: Option<u64>,
    pub failure_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionRecord {
    pub session_id: String,
    pub selected_candidate_id: Option<String>,
    pub apex_experiment_id: Option<String>,
    pub recipe: Option<String>,
    pub parameters: Option<Value>,
    pub status: String,
    pub rationale: String,
    pub contenders: Vec<ContenderSummary>,
    pub locked_at_ms: i64,
}

pub fn normalize_parameters(recipe: &str, task: &str, raw: &Value) -> Result<Value> {
    if recipe == "dummy" {
        return Ok(json!({}));
    }
    let obj = raw.as_object().ok_or("parameters must be an object")?;
    let mut normalized = BTreeMap::new();
    if recipe == "linear" {
        let is_regression = task == "regression";
        let param_name = if is_regression { "alpha" } else { "C" };
        for k in obj.keys() {
            if k != param_name {
                return Err(format!("unknown linear parameter: {k}"));
            }
        }
        let val = obj.get(param_name).and_then(|v| v.as_f64()).ok_or_else(|| format!("missing numeric {param_name}"))?;
        if !(0.001..=1000.0).contains(&val) || !val.is_finite() {
            return Err(format!("parameter {param_name}={val} out of bounds [0.001, 1000.0]"));
        }
        let rounded = (val * 1_000_000.0).round() / 1_000_000.0;
        normalized.insert(param_name.to_string(), json!(rounded));
        return Ok(json!(normalized));
    }
    if recipe == "tree" {
        for k in obj.keys() {
            if !matches!(k.as_str(), "n_estimators" | "max_depth" | "min_samples_leaf" | "max_features") {
                return Err(format!("unknown tree parameter: {k}"));
            }
        }
        let n_est = obj.get("n_estimators").and_then(|v| v.as_i64()).ok_or("missing integer n_estimators")?;
        if !(16..=128).contains(&n_est) {
            return Err(format!("n_estimators={n_est} out of bounds [16, 128]"));
        }
        let max_d = obj.get("max_depth").and_then(|v| v.as_i64()).ok_or("missing integer max_depth")?;
        if !(2..=20).contains(&max_d) {
            return Err(format!("max_depth={max_d} out of bounds [2, 20]"));
        }
        let min_leaf = obj.get("min_samples_leaf").and_then(|v| v.as_i64()).ok_or("missing integer min_samples_leaf")?;
        if !(1..=20).contains(&min_leaf) {
            return Err(format!("min_samples_leaf={min_leaf} out of bounds [1, 20]"));
        }
        let max_feat = obj.get("max_features").and_then(|v| v.as_f64()).ok_or("missing numeric max_features")?;
        if !(0.2..=1.0).contains(&max_feat) || !max_feat.is_finite() {
            return Err(format!("max_features={max_feat} out of bounds [0.2, 1.0]"));
        }
        let feat_rounded = (max_feat * 1_000_000.0).round() / 1_000_000.0;
        normalized.insert("n_estimators".to_string(), json!(n_est));
        normalized.insert("max_depth".to_string(), json!(max_d));
        normalized.insert("min_samples_leaf".to_string(), json!(min_leaf));
        normalized.insert("max_features".to_string(), json!(feat_rounded));
        return Ok(json!(normalized));
    }
    if recipe == "hist_boosting" {
        for k in obj.keys() {
            if !matches!(k.as_str(), "learning_rate" | "max_iter" | "max_leaf_nodes" | "min_samples_leaf" | "l2_regularization") {
                return Err(format!("unknown hist_boosting parameter: {k}"));
            }
        }
        let lr = obj.get("learning_rate").and_then(|v| v.as_f64()).ok_or("missing numeric learning_rate")?;
        if !(0.01..=0.3).contains(&lr) || !lr.is_finite() {
            return Err(format!("learning_rate={lr} out of bounds [0.01, 0.3]"));
        }
        let max_iter = obj.get("max_iter").and_then(|v| v.as_i64()).ok_or("missing integer max_iter")?;
        if !(20..=150).contains(&max_iter) {
            return Err(format!("max_iter={max_iter} out of bounds [20, 150]"));
        }
        let max_leaves = obj.get("max_leaf_nodes").and_then(|v| v.as_i64()).ok_or("missing integer max_leaf_nodes")?;
        if !(15..=63).contains(&max_leaves) {
            return Err(format!("max_leaf_nodes={max_leaves} out of bounds [15, 63]"));
        }
        let min_leaf = obj.get("min_samples_leaf").and_then(|v| v.as_i64()).ok_or("missing integer min_samples_leaf")?;
        if !(10..=50).contains(&min_leaf) {
            return Err(format!("min_samples_leaf={min_leaf} out of bounds [10, 50]"));
        }
        let l2 = obj.get("l2_regularization").and_then(|v| v.as_f64()).ok_or("missing numeric l2_regularization")?;
        if !(0.0..=10.0).contains(&l2) || !l2.is_finite() {
            return Err(format!("l2_regularization={l2} out of bounds [0.0, 10.0]"));
        }
        normalized.insert("learning_rate".to_string(), json!((lr * 1_000_000.0).round() / 1_000_000.0));
        normalized.insert("max_iter".to_string(), json!(max_iter));
        normalized.insert("max_leaf_nodes".to_string(), json!(max_leaves));
        normalized.insert("min_samples_leaf".to_string(), json!(min_leaf));
        normalized.insert("l2_regularization".to_string(), json!((l2 * 1_000_000.0).round() / 1_000_000.0));
        return Ok(json!(normalized));
    }
    Err(format!("unsupported recipe: {recipe}"))
}

pub fn compute_experiment_id(
    dataset_sha: &str,
    split: &str,
    recipe: &str,
    params: &Value,
    seed: u64,
    fidelity: f64,
) -> String {
    let canonical_params = serde_json::to_string(params).unwrap_or_default();
    let text = format!("{dataset_sha}|{split}|{recipe}|{canonical_params}|{seed}|{fidelity:.2}");
    let h = hash_bytes(text.as_bytes());
    format!("exp-{}", &h[..16])
}

fn python_exec(root: &Path, env_id: Option<&str>) -> Result<PathBuf> {
    if let Some(env) = env_id {
        let p = root.join("environments").join(env).join("bin/python3");
        if p.is_file() {
            return Ok(p);
        }
        return Err(format!(
            "Managed environment python '{}' not found; host Python fallback is prohibited.",
            p.display()
        ));
    }
    let p = root.join("environments").join("scientific").join("bin/python3");
    if p.is_file() {
        return Ok(p);
    }
    Err("Managed scientific environment not found; host Python fallback is prohibited.".into())
}

pub fn ask_sidecar(
    root: &Path,
    workflow_dir: &Path,
    session_id: &str,
    recipe: &str,
    fidelity: f64,
    task: &str,
    direction: &str,
    seed: u64,
    env_id: Option<&str>,
) -> Result<(u32, Value)> {
    let py = python_exec(root, env_id)?;
    let pythonpath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
    let output = Command::new(py)
        .args([
            "-m", "mlrescurix_worker.search.cli", "ask",
            "--dir", workflow_dir.to_str().unwrap(),
            "--session", session_id,
            "--recipe", recipe,
            "--fidelity", &fidelity.to_string(),
            "--task", task,
            "--direction", direction,
            "--seed", &seed.to_string(),
        ])
        .env("PYTHONPATH", pythonpath)
        .output()
        .map_err(|e| format!("cannot run search sidecar: {e}"))?;

    if !output.status.success() {
        return Err(format!("search sidecar ask failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    let res: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    let trial_id = res["optimizer_trial_id"].as_u64().ok_or("optimizer_trial_id missing")? as u32;
    let params = res["parameters"].clone();
    Ok((trial_id, params))
}

pub fn tell_sidecar(
    root: &Path,
    workflow_dir: &Path,
    session_id: &str,
    recipe: &str,
    fidelity: f64,
    trial_id: u32,
    score: Option<f64>,
    state: &str,
    direction: &str,
    seed: u64,
    env_id: Option<&str>,
) -> Result<()> {
    let py = python_exec(root, env_id)?;
    let pythonpath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../python/src");
    let mut cmd = Command::new(py);
    cmd.args([
        "-m", "mlrescurix_worker.search.cli", "tell",
        "--dir", workflow_dir.to_str().unwrap(),
        "--session", session_id,
        "--recipe", recipe,
        "--fidelity", &fidelity.to_string(),
        "--trial-id", &trial_id.to_string(),
        "--state", state,
        "--direction", direction,
        "--seed", &seed.to_string(),
    ]);
    if let Some(val) = score {
        cmd.arg("--value").arg(val.to_string());
    }
    let output = cmd.env("PYTHONPATH", pythonpath).output().map_err(|e| format!("cannot run search sidecar tell: {e}"))?;
    if !output.status.success() {
        return Err(format!("search sidecar tell failed: {}", String::from_utf8_lossy(&output.stderr)));
    }
    Ok(())
}

fn connect_db(root: &Path) -> Result<Connection> {
    crate::execution::migrate(root)?;
    let c = sql(Connection::open(root.join("state.sqlite")))?;
    sql(c.busy_timeout(std::time::Duration::from_secs(2)))?;
    sql(c.execute_batch("PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;"))?;
    Ok(c)
}

pub fn init_search(root: &Path, spec: SearchSpec) -> Result<Value> {
    spec.validate()?;
    execution::migrate(root)?;
    let c = connect_db(root)?;

    // Check if research admission is closed due to final test reservation or project completion
    let final_gate: Option<String> = sql(c.query_row(
        "SELECT status FROM final_test_gates WHERE project=?1 AND status IN ('reserved', 'consumed')",
        params![spec.project_id],
        |r| r.get(0),
    ).optional())?;
    if final_gate.is_some() {
        return Err("research admission is closed: final test access has been reserved or consumed".into());
    }
    let proj_state: Option<String> = sql(c.query_row(
        "SELECT state FROM projects WHERE id=?1",
        params![spec.project_id],
        |r| r.get(0),
    ).optional())?;
    if proj_state.as_deref() == Some("finalizing") || proj_state.as_deref() == Some("completed") {
        return Err("research admission is closed: project is finalizing or completed".into());
    }

    let exists: bool = sql(c.query_row("SELECT EXISTS(SELECT 1 FROM search_sessions WHERE project=?1 AND id=?2)", params![spec.project_id, spec.session_id], |r| r.get(0)))?;
    if exists {
        return inspect_search(root, &spec.project_id, &spec.session_id);
    }
    let dataset_exists: bool = sql(c.query_row("SELECT EXISTS(SELECT 1 FROM scientific_datasets WHERE project=?1 AND id=?2)", params![spec.project_id, spec.dataset_id], |r| r.get(0)))?;
    if !dataset_exists {
        return Err("dataset not registered for project".into());
    }
    let spec_json = serde_json::to_string(&spec).map_err(|e| e.to_string())?;
    let now = now_ms();
    sql(c.execute(
        "INSERT INTO search_sessions(project, id, dataset_id, spec, status, created_ms, updated_ms) VALUES(?1, ?2, ?3, ?4, 'created', ?5, ?5)",
        params![spec.project_id, spec.session_id, spec.dataset_id, spec_json, now],
    ))?;

    // Ingest baselines if results.json exists
    let workflow = root.join("workflows").join(&spec.project_id);
    let results_path = workflow.join("results.json");
    if results_path.is_file() {
        if let Ok(content) = fs::read(&results_path) {
            if let Ok(results) = serde_json::from_slice::<Value>(&content) {
                if let Some(candidates) = results["candidates"].as_array() {
                    let _sha = results["dataset_sha256"].as_str().unwrap_or("");
                    for cand in candidates {
                        let recipe = cand["recipe"].as_str().unwrap_or("dummy");
                        let exp_id = format!("baseline-{}", cand["candidate_id"].as_str().unwrap_or(recipe));
                        let dag_exists: bool = sql(c.query_row("SELECT EXISTS(SELECT 1 FROM experiment_dag WHERE project=?1 AND id=?2)", params![spec.project_id, exp_id], |r| r.get(0)))?;
                        if !dag_exists {
                            // Derive canonical default parameters from recipe provenance
                            let baseline_params = if cand.get("parameters").and_then(|p| p.as_object()).is_some_and(|o| !o.is_empty()) {
                                cand["parameters"].clone()
                            } else {
                                match recipe {
                                    "linear" => if spec.task == "regression" { json!({"alpha": 1.0}) } else { json!({"C": 1.0}) },
                                    "tree" => json!({"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1, "max_features": 1.0}),
                                    "hist_boosting" => json!({"learning_rate": 0.1, "max_iter": 50, "max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.0}),
                                    _ => json!({}),
                                }
                            };
                            let norm_params = normalize_parameters(recipe, &spec.task, &baseline_params).unwrap_or(baseline_params);
                            sql(c.execute(
                                "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, 2026, 1.0, 'baseline', NULL, ?6)",
                                params![spec.project_id, spec.session_id, exp_id, recipe, serde_json::to_string(&norm_params).unwrap(), now],
                            ))?;
                            let scores = &cand["metrics"];
                            let primary = scores.get(&spec.metric).and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let train_ms = cand.get("train_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let model_sha = cand.get("model_sha256").and_then(|v| v.as_str()).unwrap_or("");
                            let pred_sha = cand.get("validation_predictions_sha256").and_then(|v| v.as_str()).unwrap_or("");
                            let constraints_val = cand.get("constraints").cloned().unwrap_or(json!({}));
                            let mut feasible = cand.get("is_feasible").and_then(|v| v.as_bool()).unwrap_or(true);
                            if let Some(c_obj) = constraints_val.as_object() {
                                for (_, item) in c_obj {
                                    if item.get("verdict").and_then(|v| v.as_str()) == Some("fail") {
                                        feasible = false;
                                        break;
                                    }
                                }
                            }
                            sql(c.execute(
                                "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms) VALUES(?1, ?2, 'evaluated', ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'results.json', ?10)",
                                params![spec.project_id, exp_id, primary, serde_json::to_string(scores).unwrap(), serde_json::to_string(&constraints_val).unwrap(), if feasible { 1 } else { 0 }, train_ms, model_sha, pred_sha, now],
                            ))?;
                        }
                    }
                }
            }
        }
    }
    inspect_search(root, &spec.project_id, &spec.session_id)
}

pub fn inspect_search(root: &Path, project: &str, session_id: &str) -> Result<Value> {
    let c = connect_db(root)?;
    let session_row: Option<(String, String, i64, i64, Option<String>, Option<String>)> = sql(c.query_row(
        "SELECT spec, status, created_ms, updated_ms, apex_candidate_id, selection_record FROM search_sessions WHERE project=?1 AND id=?2",
        params![project, session_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
    ).optional())?;
    let Some((spec_str, status, created_ms, updated_ms, apex_id, selection_str)) = session_row else {
        return Err("search session not found".into());
    };
    let spec: Value = serde_json::from_str(&spec_str).map_err(|e| e.to_string())?;
    let selection: Option<Value> = selection_str.and_then(|s| serde_json::from_str(&s).ok());

    let mut q = sql(c.prepare("SELECT d.id, d.recipe, d.parameters, d.seed, d.fidelity, d.node_type, d.parent_id, r.status, r.primary_metric, r.is_feasible, r.metrics, r.constraints, r.train_ms, r.model_sha256 FROM experiment_dag d LEFT JOIN trial_results r ON r.project=d.project AND r.experiment_id=d.id WHERE d.project=?1 AND d.session_id=?2 ORDER BY d.created_ms"))?;
    let experiments = sql(q.query_map(params![project, session_id], |r| {
        Ok(json!({
            "experiment_id": r.get::<_, String>(0)?,
            "recipe": r.get::<_, String>(1)?,
            "parameters": serde_json::from_str::<Value>(&r.get::<_, String>(2)?).unwrap_or(json!({})),
            "seed": r.get::<_, u64>(3)?,
            "fidelity": r.get::<_, f64>(4)?,
            "node_type": r.get::<_, String>(5)?,
            "parent_id": r.get::<_, Option<String>>(6)?,
            "status": r.get::<_, Option<String>>(7)?.unwrap_or_else(|| "pending".into()),
            "primary_metric": r.get::<_, Option<f64>>(8)?,
            "is_feasible": r.get::<_, Option<i64>>(9)?.map(|v| v == 1),
            "metrics": r.get::<_, Option<String>>(10)?.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
            "constraints": r.get::<_, Option<String>>(11)?.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
            "train_ms": r.get::<_, Option<f64>>(12)?,
            "model_sha256": r.get::<_, Option<String>>(13)?,
        }))
    }))?.collect::<std::result::Result<Vec<Value>, _>>().map_err(|e| e.to_string())?;

    Ok(json!({
        "project_id": project,
        "session_id": session_id,
        "status": status,
        "spec": spec,
        "created_ms": created_ms,
        "updated_ms": updated_ms,
        "apex_candidate_id": apex_id,
        "selection_record": selection,
        "experiments": experiments,
    }))
}

pub fn reconcile_search(root: &Path, project: &str, session_id: &str) -> Result<()> {
    let c = connect_db(root)?;
    let spec_row: String = sql(c.query_row("SELECT spec FROM search_sessions WHERE project=?1 AND id=?2", params![project, session_id], |r| r.get(0)))?;
    let spec: SearchSpec = serde_json::from_str(&spec_row).map_err(|e| e.to_string())?;
    let workflow = root.join("workflows").join(project);

    // Look for optimizer_trials in "dispatched" or "failed" state that have trial_results
    let mut q = sql(c.prepare("SELECT o.study_name, o.optuna_trial_id, o.experiment_id, o.fidelity, r.primary_metric, r.status, d.recipe, d.seed FROM optimizer_trials o JOIN trial_results r ON r.project=o.project AND r.experiment_id=o.experiment_id JOIN experiment_dag d ON d.project=o.project AND d.id=o.experiment_id WHERE o.project=?1 AND o.session_id=?2 AND o.status IN ('dispatched', 'failed')"))?;
    let pending = sql(q.query_map(params![project, session_id], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, u32>(1)?, r.get::<_, String>(2)?, r.get::<_, f64>(3)?, r.get::<_, f64>(4)?, r.get::<_, String>(5)?, r.get::<_, String>(6)?, r.get::<_, u64>(7)?))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    for (study, trial_id, _exp_id, fidelity, score, status, recipe, seed) in pending {
        let state = if status == "evaluated" { "COMPLETE" } else { "FAIL" };
        let tell_res = tell_sidecar(root, &workflow, session_id, &recipe, fidelity, trial_id, Some(score), state, &spec.direction, seed, None);
        if tell_res.is_ok() {
            sql(c.execute("UPDATE optimizer_trials SET status='told', score=?5 WHERE project=?1 AND session_id=?2 AND study_name=?3 AND optuna_trial_id=?4", params![project, session_id, study, trial_id, score]))?;
        } else {
            sql(c.execute("UPDATE optimizer_trials SET status='failed', score=?5 WHERE project=?1 AND session_id=?2 AND study_name=?3 AND optuna_trial_id=?4", params![project, session_id, study, trial_id, score]))?;
        }
    }
    Ok(())
}

fn check_wall_budget_guard(root: &Path, project: &str, spec: &SearchSpec) -> Result<bool> {
    let c = connect_db(root)?;
    let policy_str: String = sql(c.query_row("SELECT policy FROM projects WHERE id=?1", [project], |r| r.get(0)))?;
    let policy: Policy = serde_json::from_str(&policy_str).map_err(|e| e.to_string())?;
    let total_wall = policy.budget.get("wall_ms").copied().unwrap_or(i64::MAX);

    let held: i64 = sql(c.query_row(
        "SELECT COALESCE(SUM(CAST(json_extract(budget,'$.wall_ms') AS INTEGER)),0) FROM reservations WHERE held=1 AND project=?1",
        [project],
        |r| r.get(0),
    ))?;
    let charged: i64 = sql(c.query_row(
        "SELECT COALESCE(SUM(CAST(json_extract(amount,'$.wall_ms') AS INTEGER)),0) FROM charges WHERE project=?1",
        [project],
        |r| r.get(0),
    ))?;
    let remaining = total_wall.saturating_sub(held.saturating_add(charged));
    let allowance = (spec.finalization_allowance_ms as i64).max((total_wall as f64 * 0.20) as i64);
    // Ensure the prospective job reservation will not breach the allowance:
    if remaining.saturating_sub(spec.job_timeout_ms as i64) <= allowance {
        return Ok(false); // Guard triggered: close search admission to protect finalization
    }
    Ok(true)
}

fn run_single_trial_job(
    root: &Path,
    project: &str,
    task_id: &str,
    generation: u64,
    spec: &SearchSpec,
    exp: &ExperimentSpec,
    supervisor: &mut Supervisor,
    env_id: Option<&str>,
    attempt: u32,
) -> Result<TrialResult> {
    let job_id = if attempt == 0 { format!("trial-{}", exp.experiment_id) } else { format!("trial-{}-{}", exp.experiment_id, attempt) };
    let workflow = root.join("workflows").join(project);
    let trial_dir = workflow.join("trials").join(&exp.experiment_id);
    let trial_file = trial_dir.join("trial.json");

    let c = connect_db(root)?;
    if trial_file.is_file() {
        if let Ok(bytes) = fs::read(&trial_file) {
            if let Ok(tr) = serde_json::from_slice::<TrialResult>(&bytes) {
                // Verify committed outcome in database matches
                let committed: Option<(String, f64, i64)> = sql(c.query_row(
                    "SELECT status, primary_metric, is_feasible FROM trial_results WHERE project=?1 AND experiment_id=?2",
                    params![project, exp.experiment_id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                ).optional())?;
                if let Some((st, pm, feas)) = committed {
                    if st == "evaluated" && (tr.primary_metric - pm).abs() < 1e-6 && (if tr.is_feasible { 1 } else { 0 }) == feas {
                        let art_path = workflow.join(&tr.artifact_path);
                        let mut cache_verified = false;
                        if tr.model_sha256.is_empty() {
                            if art_path.is_file() {
                                cache_verified = true;
                            }
                        } else if art_path.is_file() {
                            if let Ok(art_bytes) = fs::read(&art_path) {
                                if hash_bytes(&art_bytes) == tr.model_sha256 {
                                    cache_verified = true;
                                }
                            }
                        }
                        if cache_verified {
                            if !tr.predictions_sha256.is_empty() {
                                let pred_path = trial_dir.join("predictions.json");
                                if pred_path.is_file() {
                                    if let Ok(pred_bytes) = fs::read(&pred_path) {
                                        if hash_bytes(&pred_bytes) == tr.predictions_sha256 {
                                            return Ok(tr);
                                        }
                                    }
                                }
                            } else {
                                return Ok(tr);
                            }
                        }
                    }
                }
            }
        }
    }

    let job_spec = JobSpec {
        job_id: job_id.clone(),
        project_id: project.to_string(),
        task_id: task_id.to_string(),
        generation,
        timeout_ms: spec.job_timeout_ms,
        threads: 1,
        max_output_bytes: 4 * 1024 * 1024,
        environment_id: env_id.map(|s| s.to_string()).or_else(|| Some("scientific".into())),
        probe: Probe::Scientific {
            operation: "trial".into(),
            dataset_id: spec.dataset_id.clone(),
            spec: Some(serde_json::to_value(exp).map_err(|e| e.to_string())?),
            candidate_id: None,
            test_access_id: None,
        },
    };

    let _ = execution::submit(root, job_spec)?;

    // Run until job reaches terminal state or deadline expires
    let deadline = Instant::now() + Duration::from_millis(spec.job_timeout_ms.saturating_add(5000));
    while Instant::now() < deadline {
        supervisor.tick()?;
        let inspect = execution::inspect(root, project, &job_id)?;
        if !matches!(inspect.status.as_str(), "queued" | "running" | "terminating" | "recovering") {
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }

    let inspect = execution::inspect(root, project, &job_id)?;
    if inspect.status != "succeeded" {
        return Ok(TrialResult {
            experiment_id: exp.experiment_id.clone(),
            status: if inspect.status == "timed_out" { "timed_out".into() } else { "failed".into() },
            primary_metric: if spec.direction == "maximize" { f64::NEG_INFINITY } else { f64::INFINITY },
            metrics: json!({}),
            constraints: json!({}),
            is_feasible: false,
            train_ms: 0.0,
            model_sha256: "".into(),
            predictions_sha256: "".into(),
            artifact_path: "".into(),
        });
    }

    if trial_file.is_file() {
        let bytes = fs::read(&trial_file).map_err(|e| e.to_string())?;
        let tr: TrialResult = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if !tr.model_sha256.is_empty() {
            let _ = sql(c.execute(
                "INSERT INTO refs(project, id, category, digest) VALUES(?1, ?2, 'artifact', ?3) ON CONFLICT DO NOTHING",
                params![project, format!("trial-art-{}", tr.experiment_id), tr.model_sha256],
            ));
        }
        return Ok(tr);
    }
    Err("trial worker completed but trial.json missing".into())
}

fn run_single_trial_job_with_retries(
    root: &Path,
    project: &str,
    task_id: &str,
    generation: u64,
    spec: &SearchSpec,
    exp: &ExperimentSpec,
    supervisor: &mut Supervisor,
    env_id: Option<&str>,
) -> Result<TrialResult> {
    let mut last_tr = None;
    let max_retries = spec.max_retries.min(10);
    for attempt in 0..=max_retries {
        let tr = run_single_trial_job(root, project, task_id, generation, spec, exp, supervisor, env_id, attempt)?;
        if tr.status == "evaluated" {
            return Ok(tr);
        }
        last_tr = Some(tr);
    }
    Ok(last_tr.unwrap())
}

pub fn execute_search(
    root: &Path,
    project: &str,
    session_id: &str,
    task_id: &str,
    generation: u64,
    supervisor: &mut Supervisor,
    env_id: Option<&str>,
) -> Result<SelectionRecord> {
    let c = connect_db(root)?;

    // Research admission permanently closed if final test gate reserved or consumed, or project finalizing/completed
    let final_gate: Option<String> = sql(c.query_row(
        "SELECT status FROM final_test_gates WHERE project=?1 AND status IN ('reserved', 'consumed')",
        params![project],
        |r| r.get(0),
    ).optional())?;
    if final_gate.is_some() {
        return Err("research admission is closed: final test access has been reserved or consumed".into());
    }
    let proj_state: Option<String> = sql(c.query_row(
        "SELECT state FROM projects WHERE id=?1",
        params![project],
        |r| r.get(0),
    ).optional())?;
    if proj_state.as_deref() == Some("finalizing") || proj_state.as_deref() == Some("completed") {
        return Err("research admission is closed: project is finalizing or completed".into());
    }

    let (spec_str, dataset_sha, session_status, session_created_ms): (String, String, String, i64) = sql(c.query_row(
        "SELECT s.spec, d.sha256, s.status, s.created_ms FROM search_sessions s JOIN scientific_datasets d ON d.project=s.project AND d.id=s.dataset_id WHERE s.project=?1 AND s.id=?2",
        params![project, session_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    ))?;

    if session_status == "completed" || session_status == "provisional" {
        let selection_str: Option<String> = sql(c.query_row(
            "SELECT selection_record FROM search_sessions WHERE project=?1 AND id=?2",
            params![project, session_id],
            |r| r.get(0),
        ).optional())?;
        if let Some(s) = selection_str {
            if let Ok(rec) = serde_json::from_str::<SelectionRecord>(&s) {
                return Ok(rec);
            }
        }
    }

    let spec: SearchSpec = serde_json::from_str(&spec_str).map_err(|e| e.to_string())?;
    let workflow = root.join("workflows").join(project);

    reconcile_search(root, project, session_id)?;

    // Update status to screening
    sql(c.execute("UPDATE search_sessions SET status='screening', updated_ms=?3 WHERE project=?1 AND id=?2", params![project, session_id, now_ms()]))?;

    let is_maximize = spec.direction == "maximize";
    let cmp_scores = |a: f64, b: f64| -> std::cmp::Ordering {
        if is_maximize {
            b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        }
    };

    let mut total_trials: u32 = sql(c.query_row(
        "SELECT count(*) FROM experiment_dag WHERE project=?1 AND session_id=?2 AND node_type != 'baseline'",
        params![project, session_id],
        |r| r.get(0),
    ))?;

    let check_limits = |trials: u32, created_ms: i64, sp: &SearchSpec, rt: &Path, prj: &str| -> Result<bool> {
        if trials >= sp.max_trials {
            return Ok(false);
        }
        if (now_ms() - created_ms) >= sp.search_wall_ms as i64 {
            return Ok(false);
        }
        check_wall_budget_guard(rt, prj, sp)
    };

    // Stage 1: Screening (fidelity = 0.5)
    let screen_fidelity = 0.5;
    let mut screen_count = 0;
    for &recipe in &["linear", "tree"] {
        if !spec.recipes.iter().any(|r| r == recipe) {
            continue;
        }
        for _ in 0..(spec.max_screen_trials / 2).max(1) {
            if screen_count >= spec.max_screen_trials {
                break;
            }
            if !check_limits(total_trials, session_created_ms, &spec, root, project)? {
                break;
            }
            let (trial_id, params) = ask_sidecar(
                root, &workflow, session_id, recipe, screen_fidelity, &spec.task, &spec.direction, 2026, env_id,
            )?;
            let exp_id = compute_experiment_id(&dataset_sha, &spec.split, recipe, &params, 2026, screen_fidelity);
            let exp = ExperimentSpec {
                experiment_id: exp_id.clone(),
                session_id: session_id.to_string(),
                project_id: project.to_string(),
                recipe: recipe.to_string(),
                parameters: params.clone(),
                seed: 2026,
                fidelity: screen_fidelity,
                parent_experiment_id: None,
                node_type: "screening".to_string(),
            };

            let study_name = format!("{session_id}_{recipe}_f50");
            sql(c.execute(
                "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'screening', NULL, ?8) ON CONFLICT(project, id) DO NOTHING",
                params![project, session_id, exp_id, recipe, serde_json::to_string(&params).unwrap(), 2026, screen_fidelity, now_ms()],
            ))?;
            sql(c.execute(
                "INSERT INTO optimizer_trials(project, session_id, study_name, optuna_trial_id, experiment_id, fidelity, status, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, 'dispatched', ?7) ON CONFLICT(project, session_id, study_name, optuna_trial_id) DO NOTHING",
                params![project, session_id, study_name, trial_id, exp_id, screen_fidelity, now_ms()],
            ))?;

            let tr = run_single_trial_job_with_retries(root, project, task_id, generation, &spec, &exp, supervisor, env_id)?;
            sql(c.execute(
                "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) ON CONFLICT(project, experiment_id) DO UPDATE SET status=excluded.status, primary_metric=excluded.primary_metric, metrics=excluded.metrics, is_feasible=excluded.is_feasible",
                params![project, exp_id, tr.status, tr.primary_metric, serde_json::to_string(&tr.metrics).unwrap(), serde_json::to_string(&tr.constraints).unwrap(), if tr.is_feasible { 1 } else { 0 }, tr.train_ms, tr.model_sha256, tr.predictions_sha256, tr.artifact_path, now_ms()],
            ))?;

            let tell_state = if tr.status == "evaluated" { "COMPLETE" } else { "FAIL" };
            let tell_res = tell_sidecar(
                root, &workflow, session_id, recipe, screen_fidelity, trial_id,
                Some(tr.primary_metric), tell_state, &spec.direction, 2026, env_id,
            );
            if tell_res.is_ok() {
                sql(c.execute(
                    "UPDATE optimizer_trials SET status='told', score=?5 WHERE project=?1 AND session_id=?2 AND study_name=?3 AND optuna_trial_id=?4",
                    params![project, session_id, study_name, trial_id, tr.primary_metric],
                ))?;
            } else {
                sql(c.execute(
                    "UPDATE optimizer_trials SET status='failed', score=?5 WHERE project=?1 AND session_id=?2 AND study_name=?3 AND optuna_trial_id=?4",
                    params![project, session_id, study_name, trial_id, tr.primary_metric],
                ))?;
            }
            screen_count += 1;
            total_trials += 1;
        }
    }

    // Rank screening trials (screening never qualifies for Apex directly)
    let mut q = sql(c.prepare("SELECT d.id, d.recipe, d.parameters, r.primary_metric, r.is_feasible FROM experiment_dag d JOIN trial_results r ON r.project=d.project AND r.experiment_id=d.id WHERE d.project=?1 AND d.session_id=?2 AND d.node_type='screening' AND r.status='evaluated' AND r.is_feasible=1"))?;
    let mut screening_feasible: Vec<(String, String, Value, f64)> = sql(q.query_map(params![project, session_id], |r| {
        Ok((r.get(0)?, r.get(1)?, serde_json::from_str(&r.get::<_, String>(2)?).unwrap(), r.get(3)?))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    screening_feasible.sort_by(|a, b| cmp_scores(a.3, b.3));

    // Stage 2: Full Fidelity Promotion (fidelity = 1.0)
    sql(c.execute("UPDATE search_sessions SET status='evaluating', updated_ms=?3 WHERE project=?1 AND id=?2", params![project, session_id, now_ms()]))?;

    let top_screened: Vec<_> = screening_feasible.into_iter().take(spec.max_full_trials as usize).collect();
    let mut full_fidelity_experiments = Vec::new();

    for (parent_id, recipe, params, _) in top_screened {
        if !check_limits(total_trials, session_created_ms, &spec, root, project)? {
            break;
        }
        let full_exp_id = compute_experiment_id(&dataset_sha, &spec.split, &recipe, &params, 2026, 1.0);
        let full_exp = ExperimentSpec {
            experiment_id: full_exp_id.clone(),
            session_id: session_id.to_string(),
            project_id: project.to_string(),
            recipe: recipe.clone(),
            parameters: params.clone(),
            seed: 2026,
            fidelity: 1.0,
            parent_experiment_id: Some(parent_id),
            node_type: "full_fidelity".to_string(),
        };

        sql(c.execute(
            "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, 1.0, 'full_fidelity', ?7, ?8) ON CONFLICT(project, id) DO NOTHING",
            params![project, session_id, full_exp_id, recipe, serde_json::to_string(&params).unwrap(), 2026, full_exp.parent_experiment_id, now_ms()],
        ))?;

        let tr = run_single_trial_job_with_retries(root, project, task_id, generation, &spec, &full_exp, supervisor, env_id)?;
        sql(c.execute(
            "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) ON CONFLICT(project, experiment_id) DO UPDATE SET status=excluded.status, primary_metric=excluded.primary_metric, metrics=excluded.metrics, is_feasible=excluded.is_feasible",
            params![project, full_exp_id, tr.status, tr.primary_metric, serde_json::to_string(&tr.metrics).unwrap(), serde_json::to_string(&tr.constraints).unwrap(), if tr.is_feasible { 1 } else { 0 }, tr.train_ms, tr.model_sha256, tr.predictions_sha256, tr.artifact_path, now_ms()],
        ))?;
        total_trials += 1;

        if tr.is_feasible && tr.status == "evaluated" {
            full_fidelity_experiments.push((full_exp_id, recipe, params, tr.primary_metric));
        }
    }

    // Include strongest feasible baseline
    let mut baseline_q = sql(c.prepare("SELECT d.id, d.recipe, d.parameters, r.primary_metric FROM experiment_dag d JOIN trial_results r ON r.project=d.project AND r.experiment_id=d.id WHERE d.project=?1 AND d.session_id=?2 AND d.node_type='baseline' AND r.status='evaluated' AND r.is_feasible=1"))?;
    let mut baselines: Vec<(String, String, Value, f64)> = sql(baseline_q.query_map(params![project, session_id], |r| {
        Ok((r.get(0)?, r.get(1)?, serde_json::from_str(&r.get::<_, String>(2)?).unwrap(), r.get(3)?))
    }))?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;

    baselines.sort_by(|a, b| cmp_scores(a.3, b.3));
    let strongest_baseline = baselines.into_iter().next();

    // Stage 3: Finalist Confirmation (up to 2 full-fidelity + 1 baseline = max 3 contenders)
    sql(c.execute("UPDATE search_sessions SET status='confirming', updated_ms=?3 WHERE project=?1 AND id=?2", params![project, session_id, now_ms()]))?;

    full_fidelity_experiments.sort_by(|a, b| cmp_scores(a.3, b.3));
    let mut contenders = Vec::new();
    for (id, recipe, params, score) in full_fidelity_experiments.into_iter().take(2) {
        contenders.push((id, recipe, params, score));
    }
    if let Some((b_id, b_recipe, b_params, b_score)) = strongest_baseline {
        if !contenders.iter().any(|(_, r, p, _)| r == &b_recipe && p == &b_params) {
            contenders.push((b_id, b_recipe, b_params, b_score));
        }
    }

    let mut confirmed_contenders = Vec::new();
    let confirmation_seeds: Vec<u64> = (2026..=2026 + (spec.max_repeats.max(1) as u64 - 1)).collect();
    let mut budget_exhausted_during_confirmation = false;

    for (c_id, recipe, params, seed2026_score) in contenders {
        let mut scores = vec![seed2026_score];
        let mut seeds = vec![2026];
        let mut all_feasible = true;
        let mut warm_p95 = None;
        let mut model_bytes = None;

        // Fetch initial trial latency & size
        if let Ok(initial_tr) = sql(c.query_row("SELECT constraints, model_sha256 FROM trial_results WHERE project=?1 AND experiment_id=?2", params![project, c_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))) {
            if let Ok(c_json) = serde_json::from_str::<Value>(&initial_tr.0) {
                warm_p95 = c_json.get("inference_latency_ms").and_then(|v| v.get("value")).and_then(|v| v.as_f64());
                model_bytes = c_json.get("model_bytes").and_then(|v| v.get("value")).and_then(|v| v.as_u64());
            }
        }

        if confirmation_seeds.len() > 1 {
            for &repeat_seed in &confirmation_seeds[1..] {
                if !check_limits(total_trials, session_created_ms, &spec, root, project)? {
                    budget_exhausted_during_confirmation = true;
                    break;
                }
                let rep_exp_id = compute_experiment_id(&dataset_sha, &spec.split, &recipe, &params, repeat_seed, 1.0);
                let rep_exp = ExperimentSpec {
                    experiment_id: rep_exp_id.clone(),
                    session_id: session_id.to_string(),
                    project_id: project.to_string(),
                    recipe: recipe.clone(),
                    parameters: params.clone(),
                    seed: repeat_seed,
                    fidelity: 1.0,
                    parent_experiment_id: Some(c_id.clone()),
                    node_type: "repeat".to_string(),
                };

                sql(c.execute(
                    "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, 1.0, 'repeat', ?7, ?8) ON CONFLICT(project, id) DO NOTHING",
                    params![project, session_id, rep_exp_id, recipe, serde_json::to_string(&params).unwrap(), repeat_seed, c_id, now_ms()],
                ))?;

                let tr = run_single_trial_job_with_retries(root, project, task_id, generation, &spec, &rep_exp, supervisor, env_id)?;
                sql(c.execute(
                    "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12) ON CONFLICT(project, experiment_id) DO UPDATE SET status=excluded.status, primary_metric=excluded.primary_metric, metrics=excluded.metrics, is_feasible=excluded.is_feasible",
                    params![project, rep_exp_id, tr.status, tr.primary_metric, serde_json::to_string(&tr.metrics).unwrap(), serde_json::to_string(&tr.constraints).unwrap(), if tr.is_feasible { 1 } else { 0 }, tr.train_ms, tr.model_sha256, tr.predictions_sha256, tr.artifact_path, now_ms()],
                ))?;
                total_trials += 1;

                if !tr.is_feasible || tr.status != "evaluated" {
                    all_feasible = false;
                }
                scores.push(tr.primary_metric);
                seeds.push(repeat_seed);
            }
        }

        let n = scores.len() as f64;
        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        confirmed_contenders.push(ContenderSummary {
            candidate_id: c_id.clone(),
            experiment_id: c_id,
            recipe,
            parameters: params,
            mean_metric: mean,
            metric_std: std,
            metric_min: min,
            metric_max: max,
            seeds,
            scores,
            is_feasible: all_feasible,
            warm_p95_ms: warm_p95,
            model_bytes,
            failure_reason: if all_feasible { None } else { Some("failed repeat feasibility".into()) },
        });
    }

    // Stage 4: Apex Selection
    let selection = if budget_exhausted_during_confirmation && !confirmed_contenders.iter().any(|c| c.is_feasible && c.seeds.len() == confirmation_seeds.len()) {
        SelectionRecord {
            session_id: session_id.to_string(),
            selected_candidate_id: None,
            apex_experiment_id: None,
            recipe: None,
            parameters: None,
            status: "provisional_budget_exhausted".to_string(),
            rationale: "search wall budget exhausted before finalist confirmation could be completed".to_string(),
            contenders: confirmed_contenders,
            locked_at_ms: now_ms(),
        }
    } else {
        select_apex_from_contenders(session_id, &spec.metric, &spec.direction, spec.tie_tolerance, confirmed_contenders)
    };

    let selection_json = serde_json::to_string(&selection).map_err(|e| e.to_string())?;
    let status_str = if selection.status == "apex_selected" { "completed" } else { "provisional" };
    sql(c.execute(
        "UPDATE search_sessions SET status=?3, updated_ms=?4, apex_candidate_id=?5, selection_record=?6 WHERE project=?1 AND id=?2",
        params![project, session_id, status_str, now_ms(), selection.selected_candidate_id, selection_json],
    ))?;

    // Write selection and lineage to workflow directory
    let _ = fs::write(workflow.join("selection.json"), &selection_json);
    let inspect = inspect_search(root, project, session_id)?;
    let lineage = json!({
        "session_id": session_id,
        "apex_candidate_id": selection.selected_candidate_id,
        "status": selection.status,
        "rationale": selection.rationale,
        "experiments": inspect["experiments"],
    });
    let _ = fs::write(workflow.join("lineage.json"), serde_json::to_string_pretty(&lineage).unwrap());

    // Update results.json to include the Apex candidate so that finalize can locate and verify it
    if let Some(apex_id) = &selection.selected_candidate_id {
        let results_path = workflow.join("results.json");
        if results_path.is_file() {
            if let Ok(content) = fs::read(&results_path) {
                if let Ok(mut res) = serde_json::from_slice::<Value>(&content) {
                    if let Some(cands) = res["candidates"].as_array_mut() {
                        let already_present = cands.iter().any(|c| c["candidate_id"] == *apex_id);
                        if !already_present {
                            cands.push(json!({
                                "candidate_id": apex_id,
                                "recipe": selection.recipe,
                                "parameters": selection.parameters,
                                "status": "evaluated",
                                "metrics": { spec.metric.clone(): selection.contenders.iter().find(|c| c.candidate_id == *apex_id).map(|c| c.mean_metric).unwrap_or(0.0) },
                            }));
                            let _ = fs::write(&results_path, serde_json::to_string_pretty(&res).unwrap());
                        }
                    }
                }
            }
        }
    }

    Ok(selection)
}

pub fn select_apex_from_contenders(
    session_id: &str,
    metric_name: &str,
    direction: &str,
    tie_tolerance: f64,
    contenders: Vec<ContenderSummary>,
) -> SelectionRecord {
    let is_maximize = direction == "maximize";
    let cmp_scores = |a: f64, b: f64| -> std::cmp::Ordering {
        if is_maximize {
            b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
        } else {
            a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
        }
    };
    let feasible: Vec<_> = contenders.iter().filter(|c| c.is_feasible && c.seeds.len() >= 3).cloned().collect();
    if feasible.is_empty() {
        return SelectionRecord {
            session_id: session_id.to_string(),
            selected_candidate_id: None,
            apex_experiment_id: None,
            recipe: None,
            parameters: None,
            status: "infeasible".to_string(),
            rationale: "no contender passed development feasibility constraints across all confirmation seeds".to_string(),
            contenders,
            locked_at_ms: now_ms(),
        };
    }
    let mut sorted = feasible;
    sorted.sort_by(|a, b| cmp_scores(a.mean_metric, b.mean_metric));
    let best_mean = sorted[0].mean_metric;
    let tied: Vec<_> = sorted.into_iter().filter(|c| (c.mean_metric - best_mean).abs() <= tie_tolerance).collect();
    let mut tie_sorted = tied;
    tie_sorted.sort_by(|a, b| {
        let lat_a = a.warm_p95_ms.unwrap_or(f64::INFINITY);
        let lat_b = b.warm_p95_ms.unwrap_or(f64::INFINITY);
        match lat_a.partial_cmp(&lat_b).unwrap_or(std::cmp::Ordering::Equal) {
            std::cmp::Ordering::Equal => {
                let size_a = a.model_bytes.unwrap_or(u64::MAX);
                let size_b = b.model_bytes.unwrap_or(u64::MAX);
                match size_a.cmp(&size_b) {
                    std::cmp::Ordering::Equal => a.candidate_id.cmp(&b.candidate_id),
                    other => other,
                }
            }
            other => other,
        }
    });
    let winner = tie_sorted.into_iter().next().unwrap();
    let rationale = format!(
        "selected Apex configuration {} ({}) with mean {}={:.4} across 3 seeds (tie tolerance={:.3})",
        winner.candidate_id, winner.recipe, metric_name, winner.mean_metric, tie_tolerance
    );
    SelectionRecord {
        session_id: session_id.to_string(),
        selected_candidate_id: Some(winner.candidate_id.clone()),
        apex_experiment_id: Some(winner.experiment_id.clone()),
        recipe: Some(winner.recipe.clone()),
        parameters: Some(winner.parameters.clone()),
        status: "apex_selected".to_string(),
        rationale,
        contenders,
        locked_at_ms: now_ms(),
    }
}
