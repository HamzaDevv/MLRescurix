//! Structured Diagnostic Feedback Bridge (Phase P10.2).
//!
//! Provides:
//! - Domain models and serialization for `DiagnosticReceipt` adhering to Draft 2020-12 schema.
//! - Failure classification (cgroup_oom, non_finite_loss, shape_mismatch, class_order_mismatch,
//!   execution_timeout, missing_dependency, syntax_error, runtime_exception).
//! - Root-cause attribution (file, line number, symbol, exception class) and bounded tracebacks.
//! - Actionable remediation suggestions for autonomous agent self-healing.
//! - Additive SQLite persistence (`diagnostic_receipts`) and query APIs under `PRAGMA user_version = 10`.

use std::path::Path;
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RootCause {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_number: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exception_class: Option<String>,
}

impl Default for RootCause {
    fn default() -> Self {
        Self {
            file: None,
            line_number: None,
            symbol: None,
            exception_class: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTelemetry {
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peak_rss_bytes: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_limit_bytes: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_elapsed_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub oom_killed: Option<bool>,
}

impl Default for ResourceTelemetry {
    fn default() -> Self {
        Self {
            exit_code: None,
            peak_rss_bytes: None,
            memory_limit_bytes: None,
            wall_elapsed_ms: None,
            timeout_ms: None,
            oom_killed: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticReceipt {
    pub schema_version: String,
    pub diagnostic_id: String,
    pub candidate_id: String,
    pub stage: String,
    pub failure_type: String,
    pub error_message: String,
    pub root_cause: RootCause,
    pub traceback_summary: String,
    pub resource_telemetry: ResourceTelemetry,
    pub remediation_suggestions: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_metadata: Option<Value>,
    #[serde(default)]
    pub created_ms: i64,
}

pub struct DiagnosticBridge;

impl DiagnosticBridge {
    pub const FAILURE_TYPES: &'static [&'static str] = &[
        "cgroup_oom",
        "non_finite_loss",
        "shape_mismatch",
        "class_order_mismatch",
        "execution_timeout",
        "missing_dependency",
        "syntax_error",
        "runtime_exception",
    ];

    /// Classify failure mode based on exit code, logs, and error strings.
    pub fn classify_failure(
        exit_code: Option<i32>,
        stderr: &str,
        stdout: &str,
        message: &str,
        oom_killed: Option<bool>,
    ) -> &'static str {
        if oom_killed == Some(true) || exit_code == Some(137) || exit_code == Some(-9) {
            return "cgroup_oom";
        }

        let combined = format!("{message}\n{stderr}\n{stdout}").to_lowercase();

        if combined.contains("cgroup_oom")
            || combined.contains("out of memory")
            || combined.contains("memoryerror")
            || combined.contains("oom-killer")
            || combined.contains("oomkilled")
            || combined.contains("allocate bytes")
            || combined.contains("std::bad_alloc")
            || combined.contains("cuda out of memory")
            || combined.contains("died from signal 9")
        {
            return "cgroup_oom";
        }

        if exit_code == Some(124)
            || combined.contains("timeout")
            || combined.contains("timed out")
            || combined.contains("timeoutexpired")
            || combined.contains("exceeded deadline")
        {
            return "execution_timeout";
        }

        if combined.contains("syntaxerror")
            || combined.contains("indentationerror")
            || combined.contains("taberror")
            || combined.contains("invalid syntax")
        {
            return "syntax_error";
        }

        if combined.contains("modulenotfounderror")
            || combined.contains("no module named")
            || combined.contains("importerror: cannot import name")
        {
            return "missing_dependency";
        }

        if combined.contains("floatingpointerror")
            || combined.contains("non-finite loss")
            || combined.contains("loss is nan")
            || combined.contains("loss is inf")
            || combined.contains("nan or non-finite")
            || combined.contains("non-finite values")
            || combined.contains("predictions contain nan")
            || combined.contains("probabilities contain nan")
            || combined.contains("overflowerror: math range error")
        {
            return "non_finite_loss";
        }

        if combined.contains("shape mismatch")
            || combined.contains("shapes (")
            || combined.contains("dimension mismatch")
            || combined.contains("expected shape")
            || combined.contains("number of features of the model must match")
            || combined.contains("feature_names mismatch")
            || combined.contains("row count mismatch")
            || combined.contains("inconsistent numbers of samples")
            || combined.contains("must have same first dimension")
            || combined.contains("row misalignment")
        {
            return "shape_mismatch";
        }

        if combined.contains("class order")
            || combined.contains("class_order")
            || combined.contains("missing probability columns")
            || combined.contains("does not match manifest class order")
            || combined.contains("is not in declared class_order")
            || combined.contains("probabilities do not sum to 1.0")
        {
            return "class_order_mismatch";
        }

        "runtime_exception"
    }

    /// Extract root-cause line/symbol attribution and bounded traceback from error text.
    pub fn parse_root_cause(text: &str) -> (RootCause, String) {
        let mut root = RootCause::default();
        let lines: Vec<&str> = text.lines().collect();

        // Find exception class from trailing lines
        for line in lines.iter().rev() {
            let l = line.trim();
            if let Some(colon_pos) = l.find(':') {
                let first_word = &l[..colon_pos].trim();
                if first_word.ends_with("Error") || first_word.ends_with("Exception") {
                    root.exception_class = Some(first_word.to_string());
                    break;
                }
            } else if l.ends_with("Error") || l.ends_with("Exception") {
                root.exception_class = Some(l.to_string());
                break;
            }
        }

        // Find candidate frame: File "...", line X, in Y
        for line in &lines {
            let l = line.trim();
            if l.starts_with("File \"") {
                if let Some(end_quote) = l[6..].find('"') {
                    let file_path = &l[6..6 + end_quote];
                    let rest = &l[6 + end_quote + 1..];
                    let mut line_no: Option<i64> = None;
                    let mut sym: Option<String> = None;

                    if let Some(line_pos) = rest.find("line ") {
                        let after_line = &rest[line_pos + 5..];
                        let digits: String = after_line.chars().take_while(|c| c.is_ascii_digit()).collect();
                        line_no = digits.parse().ok();
                    }

                    if let Some(in_pos) = rest.find(", in ") {
                        let after_in = rest[in_pos + 5..].trim();
                        if !after_in.is_empty() {
                            sym = Some(after_in.to_string());
                        }
                    }

                    let clean_file = if file_path.contains("candidate/") {
                        file_path[file_path.find("candidate/").unwrap()..].to_string()
                    } else if let Some(slash_pos) = file_path.rfind('/') {
                        file_path[slash_pos + 1..].to_string()
                    } else {
                        file_path.to_string()
                    };

                    root.file = Some(clean_file);
                    root.line_number = line_no;
                    root.symbol = sym;
                }
            }
        }

        // Bounded traceback summary
        let max_lines = 40;
        let summary_text = if lines.len() > max_lines {
            let head = &lines[..15];
            let tail = &lines[lines.len() - 15..];
            format!("{}\n... [truncated {} lines] ...\n{}", head.join("\n"), lines.len() - 30, tail.join("\n"))
        } else {
            lines.join("\n")
        };

        let bounded_summary = if summary_text.len() > 4096 {
            format!("{}...", &summary_text[..4090])
        } else {
            summary_text
        };

        (root, bounded_summary)
    }

    /// Generate actionable remediation suggestions based on failure mode and attribution.
    pub fn generate_remediations(failure_type: &str, root: &RootCause) -> Vec<String> {
        match failure_type {
            "cgroup_oom" => vec![
                "Reduce mini-batch size during training (e.g. batch_size=32 or 64) to lower peak memory occupancy.".into(),
                "Downcast numeric feature matrices from float64 to float32 or int32 using .astype(np.float32).".into(),
                "Reduce tree depth or model complexity (e.g. max_depth <= 6, n_estimators <= 100).".into(),
                "Use out-of-core batch streaming or memory-mapped arrays instead of inflating dense matrices.".into(),
            ],
            "non_finite_loss" => vec![
                "Check feature matrices for unhandled NaNs, nulls, or infs; apply SimpleImputer or median imputation before model fitting.".into(),
                "Add epsilon smoothing (e.g. 1e-7) to denominators and log arguments to prevent division by zero or log(0).".into(),
                "Apply gradient clipping (e.g. clip_grad_norm_ <= 1.0) or reduce learning rate to prevent exploding activations.".into(),
                "Verify target variable domain; ensure targets do not contain negative values if using Poisson/Gamma/Log losses.".into(),
            ],
            "shape_mismatch" => vec![
                "Verify that feature transformations produce identical columns and shapes between fit and predict.".into(),
                "If using one-hot encoding, configure handle_unknown='ignore' so unexpected test categories do not alter column counts.".into(),
                "Ensure target column is strictly isolated and dropped from feature matrices before passing to model.predict().".into(),
                "Check matrix transpose and dimension alignment: ensure (n_samples, n_features) orientation.".into(),
            ],
            "class_order_mismatch" => vec![
                "Ensure prediction output DataFrame contains probability columns named prob_<class> for each class in manifest class_order.".into(),
                "Ensure prediction probabilities are bounded in [0.0, 1.0] and sum to exactly 1.0 across classes for each sample.".into(),
                "Ensure predicted discrete values in pred column strictly match classes declared in manifest class_order.".into(),
            ],
            "execution_timeout" => vec![
                "Vectorize feature engineering using NumPy or Polars array operations instead of Python row iteration loops.".into(),
                "Reduce iteration limits (e.g. n_estimators, max_iter, or training epochs) to fit within execution deadline.".into(),
                "Implement early stopping with a validation loss patience parameter to abort unpromising training early.".into(),
                "Subsample training data for screening iterations before attempting full-dataset convergence.".into(),
            ],
            "missing_dependency" => vec![
                "The sandbox runs in an offline isolated container; third-party packages outside the certified stack cannot be installed.".into(),
                "Restrict imports to standard library and pre-installed packages: scikit-learn, lightgbm, xgboost, catboost, torch, numpy, pandas, scipy.".into(),
            ],
            "syntax_error" => {
                let loc = root.line_number.map(|l| format!(" at line {l}")).unwrap_or_default();
                let file_ref = root.file.as_ref().map(|f| format!(" in {f}")).unwrap_or_default();
                vec![
                    format!("Fix Python syntax error{loc}{file_ref}; check for missing colons, unmatched parentheses, or indentation discrepancies."),
                ]
            }
            _ => vec![
                "Review traceback lines and verify candidate code preconditions, input shapes, and variable states.".into(),
                "Add defensive guards or assertions before accessing dictionary keys or calling specialized model methods.".into(),
            ],
        }
    }

    /// Construct a verified `DiagnosticReceipt`.
    pub fn build_receipt(
        candidate_id: &str,
        stage: &str,
        exit_code: Option<i32>,
        stderr: &str,
        stdout: &str,
        message: &str,
        peak_rss_bytes: Option<i64>,
        memory_limit_bytes: Option<i64>,
        wall_elapsed_ms: Option<i64>,
        timeout_ms: Option<i64>,
        oom_killed: Option<bool>,
        context_metadata: Option<Value>,
    ) -> DiagnosticReceipt {
        let (mut root, tb_summary) = Self::parse_root_cause(&format!("{stderr}\n{stdout}"));
        let failure_type = Self::classify_failure(exit_code, stderr, stdout, message, oom_killed);

        if failure_type == "cgroup_oom" && root.exception_class.is_none() {
            root.exception_class = Some("MemoryError".into());
        }
        if failure_type == "execution_timeout" && root.exception_class.is_none() {
            root.exception_class = Some("TimeoutExpired".into());
        }

        let is_oom = if failure_type == "cgroup_oom" { Some(true) } else { oom_killed };

        let err_msg = if !message.is_empty() {
            message.to_string()
        } else if let Some(code) = exit_code {
            if code == 137 {
                "Process terminated with exit code 137 (SIGKILL / cgroup out-of-memory)".into()
            } else if code == 124 {
                format!("Process execution timed out after deadline ({}ms)", wall_elapsed_ms.or(timeout_ms).unwrap_or(0))
            } else {
                format!("Process failed with exit code {code}")
            }
        } else {
            "Execution failure".into()
        };

        let telemetry = ResourceTelemetry {
            exit_code,
            peak_rss_bytes,
            memory_limit_bytes,
            wall_elapsed_ms,
            timeout_ms,
            oom_killed: is_oom,
        };

        let remediations = Self::generate_remediations(failure_type, &root);

        DiagnosticReceipt {
            schema_version: "v2".into(),
            diagnostic_id: format!("diag-{}", &uuid_v4_short()),
            candidate_id: candidate_id.into(),
            stage: stage.into(),
            failure_type: failure_type.into(),
            error_message: if err_msg.len() > 2048 { format!("{}...", &err_msg[..2040]) } else { err_msg },
            root_cause: root,
            traceback_summary: tb_summary,
            resource_telemetry: telemetry,
            remediation_suggestions: remediations,
            context_metadata,
            created_ms: now_ms(),
        }
    }
}

fn uuid_v4_short() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let seed = format!("{}:{}", now_ms(), std::process::id());
    hasher.update(seed.as_bytes());
    format!("{:x}", hasher.finalize())[..12].to_string()
}

/// Record a diagnostic receipt into SQLite table `diagnostic_receipts`.
pub fn record_diagnostic_receipt(
    root: &Path,
    project: &str,
    receipt: &DiagnosticReceipt,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let receipt_json = serde_json::to_string_pretty(receipt).map_err(|e| e.to_string())?;
    let telemetry_json = serde_json::to_string(&receipt.resource_telemetry).map_err(|e| e.to_string())?;
    let remediations_json = serde_json::to_string(&receipt.remediation_suggestions).map_err(|e| e.to_string())?;

    db.execute(
        "INSERT INTO diagnostic_receipts (
            project, diagnostic_id, candidate_id, stage, failure_type, error_message,
            root_cause_file, root_cause_line, root_cause_symbol, exception_class,
            traceback_summary, resource_telemetry_json, remediations_json, receipt_json, created_ms
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
        ON CONFLICT(project, diagnostic_id) DO UPDATE SET
            receipt_json = excluded.receipt_json,
            error_message = excluded.error_message",
        params![
            project,
            receipt.diagnostic_id,
            receipt.candidate_id,
            receipt.stage,
            receipt.failure_type,
            receipt.error_message,
            receipt.root_cause.file,
            receipt.root_cause.line_number,
            receipt.root_cause.symbol,
            receipt.root_cause.exception_class,
            receipt.traceback_summary,
            telemetry_json,
            remediations_json,
            receipt_json,
            receipt.created_ms,
        ],
    ).map_err(|e| format!("Failed to record diagnostic receipt: {e}"))?;

    Ok(())
}

/// Retrieve a diagnostic receipt by diagnostic_id.
pub fn get_diagnostic_receipt(
    root: &Path,
    project: &str,
    diagnostic_id: &str,
) -> Result<DiagnosticReceipt> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let receipt_json: String = db.query_row(
        "SELECT receipt_json FROM diagnostic_receipts WHERE project = ?1 AND diagnostic_id = ?2",
        params![project, diagnostic_id],
        |r| r.get(0),
    ).map_err(|_| format!("Diagnostic receipt '{diagnostic_id}' not found in project '{project}'"))?;

    serde_json::from_str(&receipt_json).map_err(|e| format!("Corrupt diagnostic receipt JSON: {e}"))
}

/// List diagnostic receipts for a project, optionally filtered by candidate_id or failure_type.
pub fn list_diagnostic_receipts(
    root: &Path,
    project: &str,
    candidate_id: Option<&str>,
    failure_type: Option<&str>,
    limit: usize,
) -> Result<Vec<DiagnosticReceipt>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

    let query = match (candidate_id, failure_type) {
        (Some(cid), Some(ft)) => {
            let mut stmt = db.prepare(
                "SELECT receipt_json FROM diagnostic_receipts
                 WHERE project = ?1 AND candidate_id = ?2 AND failure_type = ?3
                 ORDER BY created_ms DESC LIMIT ?4"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, cid, ft, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        }
        (Some(cid), None) => {
            let mut stmt = db.prepare(
                "SELECT receipt_json FROM diagnostic_receipts
                 WHERE project = ?1 AND candidate_id = ?2
                 ORDER BY created_ms DESC LIMIT ?3"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, cid, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        }
        (None, Some(ft)) => {
            let mut stmt = db.prepare(
                "SELECT receipt_json FROM diagnostic_receipts
                 WHERE project = ?1 AND failure_type = ?2
                 ORDER BY created_ms DESC LIMIT ?3"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, ft, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        }
        (None, None) => {
            let mut stmt = db.prepare(
                "SELECT receipt_json FROM diagnostic_receipts
                 WHERE project = ?1
                 ORDER BY created_ms DESC LIMIT ?2"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project, limit as i64], |r| r.get::<_, String>(0)).map_err(|e| e.to_string())?;
            rows.filter_map(|r| r.ok()).collect::<Vec<_>>()
        }
    };

    let mut receipts = Vec::new();
    for json_str in query {
        if let Ok(receipt) = serde_json::from_str::<DiagnosticReceipt>(&json_str) {
            receipts.push(receipt);
        }
    }

    Ok(receipts)
}
