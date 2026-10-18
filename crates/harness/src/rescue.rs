//! Autonomous End-to-End Model Rescue Gate (Phase P10.4).
//!
//! Provides:
//! - Domain models and serialization for `AutonomousRescueSession` and `DatasetProfile`
//!   adhering to Draft 2020-12 schemas.
//! - Dataset inspection and profiling bridge.
//! - Failure interception, hypothesis-driven mutation, and Pareto optimization.
//! - Candidate delivery packaging producing Apex-certified `mlrescurix.bundle.v2`.
//! - Additive SQLite persistence (`autonomous_rescue_sessions`) under `PRAGMA user_version = 10`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DatasetProfile {
    pub columns: Vec<String>,
    pub row_count: u64,
    pub column_count: u64,
    #[serde(default)]
    pub null_counts: BTreeMap<String, u64>,
    #[serde(default)]
    pub null_percentages: BTreeMap<String, f64>,
    #[serde(default)]
    pub cardinalities: BTreeMap<String, u64>,
    #[serde(default)]
    pub dtypes: BTreeMap<String, String>,
    pub feature_columns: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_column: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id_column: Option<String>,
    pub inferred_task_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_balance: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RescueTurnRecord {
    pub turn_index: usize,
    pub phase: String,
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    pub status: String,
    #[serde(default)]
    pub details: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutonomousRescueSession {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    pub session_id: String,
    pub project_id: String,
    pub dataset_id: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub apex_candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundle_path: Option<String>,
    #[serde(default)]
    pub turns_executed: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_profile: Option<DatasetProfile>,
    #[serde(default)]
    pub turns: Vec<RescueTurnRecord>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_interception: Option<Value>,
    #[serde(default)]
    pub hypotheses: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_frontier_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finalization: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resource_accounting: Option<Value>,
    pub created_ms: i64,
    pub updated_ms: i64,
}

fn default_schema_version() -> String {
    "v2".to_string()
}

/// Record an autonomous rescue session into SQLite `autonomous_rescue_sessions`.
pub fn record_rescue_session(
    root: &Path,
    project: &str,
    session: &AutonomousRescueSession,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let profile_json = session
        .dataset_profile
        .as_ref()
        .map(|p| serde_json::to_string(p).unwrap_or_default());
    let session_json = serde_json::to_string(session).map_err(|e| e.to_string())?;

    db.execute(
        "INSERT INTO autonomous_rescue_sessions(
            project, session_id, status, dataset_id, initial_candidate_id, apex_candidate_id,
            bundle_path, turns_executed, profile_json, session_json, created_ms, updated_ms
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
         ON CONFLICT(project, session_id) DO UPDATE SET
            status = excluded.status,
            initial_candidate_id = excluded.initial_candidate_id,
            apex_candidate_id = excluded.apex_candidate_id,
            bundle_path = excluded.bundle_path,
            turns_executed = excluded.turns_executed,
            profile_json = excluded.profile_json,
            session_json = excluded.session_json,
            updated_ms = excluded.updated_ms",
        params![
            project,
            session.session_id,
            session.status,
            session.dataset_id,
            session.initial_candidate_id,
            session.apex_candidate_id,
            session.bundle_path,
            session.turns_executed,
            profile_json,
            session_json,
            session.created_ms,
            session.updated_ms,
        ],
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// Retrieve an autonomous rescue session by project and session_id.
pub fn get_rescue_session(
    root: &Path,
    project: &str,
    session_id: &str,
) -> Result<AutonomousRescueSession> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let row = db
        .query_row(
            "SELECT session_json FROM autonomous_rescue_sessions WHERE project=?1 AND session_id=?2",
            params![project, session_id],
            |r| r.get::<_, String>(0),
        )
        .map_err(|_| format!("Autonomous rescue session '{session_id}' not found in project '{project}'"))?;

    let session: AutonomousRescueSession =
        serde_json::from_str(&row).map_err(|e| format!("Corrupt autonomous rescue session JSON: {e}"))?;
    Ok(session)
}

/// List autonomous rescue sessions for a project.
pub fn list_rescue_sessions(
    root: &Path,
    project: &str,
    limit: usize,
) -> Result<Vec<AutonomousRescueSession>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let mut stmt = db
        .prepare(
            "SELECT session_json FROM autonomous_rescue_sessions
             WHERE project=?1 ORDER BY created_ms DESC LIMIT ?2",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map(params![project, limit as i64], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;

    let mut sessions = Vec::new();
    for r in rows {
        if let Ok(json_str) = r {
            if let Ok(sess) = serde_json::from_str::<AutonomousRescueSession>(&json_str) {
                sessions.push(sess);
            }
        }
    }
    Ok(sessions)
}

/// Update status, apex candidate ID, bundle path, and turns executed for an active session.
pub fn update_rescue_session_status(
    root: &Path,
    project: &str,
    session_id: &str,
    status: &str,
    apex_id: Option<&str>,
    bundle_path: Option<&str>,
    turns: i64,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let now = now_ms();

    let mut sess = get_rescue_session(root, project, session_id)?;
    sess.status = status.to_string();
    if let Some(aid) = apex_id {
        sess.apex_candidate_id = Some(aid.to_string());
    }
    if let Some(bp) = bundle_path {
        sess.bundle_path = Some(bp.to_string());
    }
    sess.turns_executed = turns;
    sess.updated_ms = now;

    let session_json = serde_json::to_string(&sess).map_err(|e| e.to_string())?;

    db.execute(
        "UPDATE autonomous_rescue_sessions
         SET status=?1, apex_candidate_id=COALESCE(?2, apex_candidate_id),
             bundle_path=COALESCE(?3, bundle_path), turns_executed=?4, session_json=?5, updated_ms=?6
         WHERE project=?7 AND session_id=?8",
        params![
            status,
            apex_id,
            bundle_path,
            turns,
            session_json,
            now,
            project,
            session_id,
        ],
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

/// Execute data profiling via Python scientific worker CLI `dataset-profile`.
pub fn profile_dataset_csv(
    root: &Path,
    csv_path: &Path,
    target_col: Option<&str>,
    id_col: Option<&str>,
) -> Result<DatasetProfile> {
    if !csv_path.is_file() {
        return Err(format!("Dataset CSV file not found: {}", csv_path.display()));
    }

    let embedded_py = crate::assets::ensure_python_root(root)?;
    let embedded_src = embedded_py.join("src");
    let mut pypaths = vec![embedded_src.to_string_lossy().to_string()];
    if let Ok(repo_src) = fs::canonicalize(Path::new("python/src")) {
        pypaths.push(repo_src.to_string_lossy().to_string());
    }
    if let Ok(cur_py) = std::env::var("PYTHONPATH") {
        if !cur_py.is_empty() {
            pypaths.push(cur_py);
        }
    }

    let mut cmd = Command::new("python3");
    cmd.args([
        "-m",
        "mlrescurix_worker.scientific.cli",
        "dataset-profile",
        "--csv",
        &csv_path.to_string_lossy(),
    ]);

    if let Some(tgt) = target_col {
        cmd.args(["--target", tgt]);
    }
    if let Some(id) = id_col {
        cmd.args(["--id-col", id]);
    }

    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");

    let output = cmd
        .output()
        .map_err(|e| format!("Failed to spawn python dataset-profile: {e}"))?;

    if !output.status.success() {
        let err_msg = String::from_utf8_lossy(&output.stderr).to_string();
        return Err(format!("dataset-profile failed: {err_msg}"));
    }

    let stdout_str = String::from_utf8_lossy(&output.stdout);
    let profile: DatasetProfile = serde_json::from_str(stdout_str.trim())
        .map_err(|e| format!("Failed to parse dataset profile JSON: {e}\nstdout: {stdout_str}"))?;

    Ok(profile)
}

/// Execute candidate packaging via Python scientific worker CLI `candidate-package-bundle`.
pub fn package_candidate_bundle(root: &Path, spec_payload: &Value) -> Result<Value> {
    let now = now_ms();
    let scratch_dir = root
        .join("workflows")
        .join(spec_payload.get("project_id").and_then(Value::as_str).unwrap_or("default"))
        .join("bundle_staging")
        .join(format!("pkg-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let spec_path = scratch_dir.join("spec.json");
    let output_path = scratch_dir.join("output.json");
    fs::write(&spec_path, serde_json::to_string_pretty(spec_payload).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let embedded_py = crate::assets::ensure_python_root(root)?;
    let embedded_src = embedded_py.join("src");
    let mut pypaths = vec![embedded_src.to_string_lossy().to_string()];
    if let Ok(repo_src) = fs::canonicalize(Path::new("python/src")) {
        pypaths.push(repo_src.to_string_lossy().to_string());
    }
    if let Ok(cur_py) = std::env::var("PYTHONPATH") {
        if !cur_py.is_empty() {
            pypaths.push(cur_py);
        }
    }

    let mut cmd = Command::new("python3");
    cmd.args([
        "-m",
        "mlrescurix_worker.scientific.cli",
        "candidate-package-bundle",
        "--spec",
        &spec_path.to_string_lossy(),
        "--output",
        &output_path.to_string_lossy(),
    ]);
    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");

    let out = cmd
        .output()
        .map_err(|e| format!("Failed to spawn candidate-package-bundle: {e}"))?;

    if !out.status.success() && !output_path.is_file() {
        let err_msg = String::from_utf8_lossy(&out.stderr).to_string();
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("candidate-package-bundle failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&output_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);

    Ok(res)
}

fn base_scientific_cmd(root: &Path, subcommand: &str) -> Result<Command> {
    let embedded_py = crate::assets::ensure_python_root(root)?;
    let embedded_src = embedded_py.join("src");
    let mut pypaths = vec![embedded_src.to_string_lossy().to_string()];
    if let Ok(repo_src) = fs::canonicalize(Path::new("python/src")) {
        pypaths.push(repo_src.to_string_lossy().to_string());
    }
    if let Ok(cur_py) = std::env::var("PYTHONPATH") {
        if !cur_py.is_empty() {
            pypaths.push(cur_py);
        }
    }

    let mut cmd = Command::new("python3");
    if let Ok(cur_dir) = std::env::current_dir() {
        let repo_root = if cur_dir.ends_with("crates/harness") {
            cur_dir.parent().unwrap_or(&cur_dir).to_path_buf()
        } else {
            cur_dir
        };
        cmd.current_dir(&repo_root);
        cmd.env("MLRESCURIX_REPO_ROOT", &repo_root);
    }
    cmd.args([
        "-m",
        "mlrescurix_worker.scientific.cli",
        subcommand,
    ]);
    cmd.env("PYTHONPATH", pypaths.join(":"));
    cmd.env("PYTHONDONTWRITEBYTECODE", "1");
    Ok(cmd)
}

/// Execute M2-EDA exploratory forensics via `eda-forensics`.
pub fn eda_forensics(
    root: &Path,
    dataset_path: &Path,
    target_col: Option<&str>,
    problem_statement: Option<&str>,
    output_dir: Option<&Path>,
    db_path: Option<&Path>,
    project_id: Option<&str>,
) -> Result<Value> {
    if !dataset_path.is_file() {
        return Err(format!("Dataset file not found: {}", dataset_path.display()));
    }
    let now = now_ms();
    let project = project_id.unwrap_or("default");
    let scratch_dir = root
        .join("workflows")
        .join(project)
        .join("eda_staging")
        .join(format!("eda-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let out_path = scratch_dir.join("output.json");
    let mut cmd = base_scientific_cmd(root, "eda-forensics")?;
    cmd.args(["--dataset", &dataset_path.to_string_lossy()]);
    if let Some(t) = target_col {
        cmd.args(["--target", t]);
    }
    if let Some(p) = problem_statement {
        cmd.args(["--problem-statement", p]);
    }
    let art_dir = output_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| root.join(".m2").join("artifacts").join("eda"));
    cmd.args(["--output-dir", &art_dir.to_string_lossy()]);
    cmd.args(["--output", &out_path.to_string_lossy()]);
    if let Some(db) = db_path {
        cmd.args(["--db-path", &db.to_string_lossy()]);
    }
    cmd.args(["--project-id", project]);

    let output = cmd.output().map_err(|e| format!("Failed to spawn python eda-forensics: {e}"))?;
    if !output.status.success() && !out_path.is_file() {
        let err_msg = if !output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stderr).to_string()
        } else {
            String::from_utf8_lossy(&output.stdout).to_string()
        };
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("eda-forensics failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);
    Ok(res)
}

/// Execute M2-FE paired feature ablation sprint via `ablation-sprint`.
pub fn ablation_sprint(root: &Path, spec_payload: &Value) -> Result<Value> {
    let now = now_ms();
    let project = spec_payload.get("project_id").and_then(Value::as_str).unwrap_or("default");
    let scratch_dir = root
        .join("workflows")
        .join(project)
        .join("ablation_staging")
        .join(format!("abl-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let spec_path = scratch_dir.join("spec.json");
    let out_path = scratch_dir.join("output.json");
    fs::write(&spec_path, serde_json::to_string_pretty(spec_payload).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let mut cmd = base_scientific_cmd(root, "ablation-sprint")?;
    cmd.args([
        "--spec",
        &spec_path.to_string_lossy(),
        "--output",
        &out_path.to_string_lossy(),
    ]);

    let output = cmd.output().map_err(|e| format!("Failed to spawn python ablation-sprint: {e}"))?;
    if !output.status.success() && !out_path.is_file() {
        let err_msg = if !output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stderr).to_string()
        } else {
            String::from_utf8_lossy(&output.stdout).to_string()
        };
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("ablation-sprint failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);
    Ok(res)
}

/// Execute M2-Model multi-paradigm evaluation and error forensics via `model-portfolio-eval`.
pub fn model_portfolio_eval(root: &Path, spec_payload: &Value) -> Result<Value> {
    let now = now_ms();
    let project = spec_payload.get("project_id").and_then(Value::as_str).unwrap_or("default");
    let scratch_dir = root
        .join("workflows")
        .join(project)
        .join("portfolio_staging")
        .join(format!("port-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let spec_path = scratch_dir.join("spec.json");
    let out_path = scratch_dir.join("output.json");
    fs::write(&spec_path, serde_json::to_string_pretty(spec_payload).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let mut cmd = base_scientific_cmd(root, "model-portfolio-eval")?;
    cmd.args([
        "--spec",
        &spec_path.to_string_lossy(),
        "--output",
        &out_path.to_string_lossy(),
    ]);

    let output = cmd.output().map_err(|e| format!("Failed to spawn python model-portfolio-eval: {e}"))?;
    if !output.status.success() && !out_path.is_file() {
        let err_msg = if !output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stderr).to_string()
        } else {
            String::from_utf8_lossy(&output.stdout).to_string()
        };
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("model-portfolio-eval failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);
    Ok(res)
}

/// Execute M2-Critic adversarial audit via `critic-audit`.
pub fn critic_audit(
    root: &Path,
    candidate_val: &Value,
    topology_val: &Value,
    forensics_val: Option<&Value>,
    dossier_val: Option<&Value>,
    db_path: Option<&Path>,
    project_id: Option<&str>,
    session_id: Option<&str>,
    epoch: Option<i64>,
) -> Result<Value> {
    let now = now_ms();
    let project = project_id.unwrap_or("default");
    let scratch_dir = root
        .join("workflows")
        .join(project)
        .join("critic_staging")
        .join(format!("crit-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let cand_path = scratch_dir.join("candidate.json");
    let topo_path = scratch_dir.join("topology.json");
    fs::write(&cand_path, serde_json::to_string_pretty(candidate_val).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    fs::write(&topo_path, serde_json::to_string_pretty(topology_val).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let mut cmd = base_scientific_cmd(root, "critic-audit")?;
    cmd.args([
        "--candidate",
        &cand_path.to_string_lossy(),
        "--topology",
        &topo_path.to_string_lossy(),
    ]);

    if let Some(f_val) = forensics_val {
        let f_path = scratch_dir.join("forensics.json");
        fs::write(&f_path, serde_json::to_string_pretty(f_val).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        cmd.args(["--forensics", &f_path.to_string_lossy()]);
    }
    if let Some(d_val) = dossier_val {
        let d_path = scratch_dir.join("dossier.json");
        fs::write(&d_path, serde_json::to_string_pretty(d_val).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        cmd.args(["--dossier", &d_path.to_string_lossy()]);
    }

    let out_path = scratch_dir.join("output.json");
    cmd.args(["--output", &out_path.to_string_lossy()]);
    if let Some(db) = db_path {
        cmd.args(["--db-path", &db.to_string_lossy()]);
    }
    cmd.args(["--project-id", project]);
    if let Some(sess) = session_id {
        cmd.args(["--session-id", sess]);
    }
    if let Some(ep) = epoch {
        cmd.args(["--epoch", &ep.to_string()]);
    }

    let output = cmd.output().map_err(|e| format!("Failed to spawn python critic-audit: {e}"))?;
    if !output.status.success() && !out_path.is_file() {
        let err_msg = if !output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stderr).to_string()
        } else {
            String::from_utf8_lossy(&output.stdout).to_string()
        };
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("critic-audit failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);
    Ok(res)
}

/// Execute M1 reflexive loop via `reflexive-orchestrate`.
pub fn reflexive_orchestrate(root: &Path, spec_payload: &Value) -> Result<Value> {
    let now = now_ms();
    let project = spec_payload.get("project_id").and_then(Value::as_str).unwrap_or("default");
    let scratch_dir = root
        .join("workflows")
        .join(project)
        .join("reflexive_staging")
        .join(format!("reflx-{now}"));
    fs::create_dir_all(&scratch_dir).map_err(|e| e.to_string())?;

    let spec_path = scratch_dir.join("spec.json");
    let out_path = scratch_dir.join("output.json");
    fs::write(&spec_path, serde_json::to_string_pretty(spec_payload).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;

    let mut cmd = base_scientific_cmd(root, "reflexive-orchestrate")?;
    cmd.args([
        "--spec",
        &spec_path.to_string_lossy(),
        "--output",
        &out_path.to_string_lossy(),
    ]);

    let output = cmd.output().map_err(|e| format!("Failed to spawn python reflexive-orchestrate: {e}"))?;
    if !output.status.success() && !out_path.is_file() {
        let err_msg = if !output.stderr.is_empty() {
            String::from_utf8_lossy(&output.stderr).to_string()
        } else {
            String::from_utf8_lossy(&output.stdout).to_string()
        };
        let _ = fs::remove_dir_all(&scratch_dir);
        return Err(format!("reflexive-orchestrate failed: {err_msg}"));
    }

    let out_str = fs::read_to_string(&out_path).map_err(|e| e.to_string())?;
    let res: Value = serde_json::from_str(&out_str).map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch_dir);
    Ok(res)
}

