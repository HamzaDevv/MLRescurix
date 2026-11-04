//! Single authoritative controller resource management across tokens, monetary spend,
//! web search queries, and Kaggle compute quotas.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn rand_suffix() -> u32 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    nanos ^ ((std::process::id() as u32) << 16)
}

/// Ensure the operational_search_attempts schema exists on the SQLite connection.
pub fn ensure_resource_authority_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS operational_search_attempts (
            project TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            query_text TEXT NOT NULL,
            sanitized_query TEXT NOT NULL,
            provider TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')),
            cost_millicents INTEGER NOT NULL DEFAULT 0,
            created_ms INTEGER NOT NULL,
            settled_ms INTEGER,
            error_message TEXT,
            PRIMARY KEY(project, attempt_id)
        );
        CREATE INDEX IF NOT EXISTS idx_search_attempts_status ON operational_search_attempts(project, status);",
    ).map_err(|e| format!("cannot initialize resource authority schema: {e}"))?;
    Ok(())
}

/// Atomically admit a search query under the configured operational budget limit.
/// Returns Ok(Some(attempt_id)) if admitted, Ok(None) if limit reached.
pub fn admit_search_attempt(
    db_path: &Path,
    project: &str,
    query: &str,
    sanitized_query: &str,
    provider: &str,
    limit: usize,
) -> Result<Option<String>> {
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ?1 AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
        params![project],
        |r| r.get(0),
    ).map_err(|e| e.to_string())?;

    if (count as usize) >= limit {
        tx.rollback().map_err(|e| e.to_string())?;
        return Ok(None);
    }

    let timestamp = now_ms();
    let attempt_id = format!("search_attempt_{}_{:x}", timestamp, rand_suffix());
    tx.execute(
        "INSERT INTO operational_search_attempts(project, attempt_id, query_text, sanitized_query, provider, status, cost_millicents, created_ms, settled_ms, error_message)
         VALUES(?1, ?2, ?3, ?4, ?5, 'admitted', 0, ?6, NULL, NULL)",
        params![project, &attempt_id, query, sanitized_query, provider, timestamp],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(Some(attempt_id))
}

/// Idempotently settle an admitted search attempt. Rejects conflicting mutations if already terminal.
pub fn settle_search_attempt(
    db_path: &Path,
    project: &str,
    attempt_id: &str,
    status: &str,
    error: Option<&str>,
    cost_millicents: i64,
) -> Result<()> {
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let existing: Option<(String, i64, Option<String>)> = tx.query_row(
        "SELECT status, cost_millicents, error_message FROM operational_search_attempts WHERE project=?1 AND attempt_id=?2",
        params![project, attempt_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional().map_err(|e| e.to_string())?;

    let (current, previous_cost, previous_error) = existing.ok_or_else(|| format!("search attempt '{attempt_id}' not found for project '{project}'"))?;
    let terminal_states = ["success", "failed", "timeout"];
    if terminal_states.contains(&current.as_str()) {
        if current == status && previous_cost == cost_millicents && previous_error.as_deref() == error {
            return Ok(()); // idempotent no-op
        }
        return Err(format!(
            "rejected conflicting settlement for search attempt '{attempt_id}': already terminal ({current}), requested ({status})"
        ));
    }

    let timestamp = now_ms();
    tx.execute(
        "UPDATE operational_search_attempts
         SET status=?1, settled_ms=?2, error_message=?3, cost_millicents=?4
         WHERE project=?5 AND attempt_id=?6",
        params![status, timestamp, error, cost_millicents, project, attempt_id],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

/// Query the total count of durable search attempts for a project.
pub fn query_search_attempt_count(db_path: &Path, project: &str) -> Result<usize> {
    if !db_path.is_file() {
        return Ok(0);
    }
    let conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ?1 AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
        params![project],
        |r| r.get(0),
    ).map_err(|e| format!("cannot count search attempts: {e}"))?;
    Ok(count.max(0) as usize)
}

/// Provide a unified summary of all operational and monetary resources for a project.
pub fn get_unified_resource_status(state_dir: &Path, project: &str) -> Result<Value> {
    let state_file = if state_dir.is_file() {
        Some(state_dir.to_path_buf())
    } else {
        let candidates = [
            state_dir.join("controller/state.sqlite"),
            state_dir.join("state.sqlite"),
        ];
        candidates.into_iter().find(|p| p.is_file())
    };

    let (spend_val, prompt_val, comp_val, tot_val, search_val, accounting_status, accounting_verified) = if let Some(p) = state_file {
        match Connection::open(&p) {
            Ok(conn) => {
                let _ = conn.busy_timeout(std::time::Duration::from_secs(5));
                let _ = ensure_resource_authority_schema(&conn);

                let provider_res: std::result::Result<(i64, i64, i64), rusqlite::Error> = conn.query_row(
                    "SELECT COALESCE(SUM(cost_millicents), 0), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0)
                     FROM provider_accounting WHERE project=?1 AND status IN ('ok', 'reserved', 'outcome_unknown')",
                    params![project],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
                );

                let search_res: std::result::Result<i64, rusqlite::Error> = conn.query_row(
                    "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ?1 AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
                    params![project],
                    |r| r.get(0),
                );

                match (provider_res, search_res) {
                    (Ok((total_spend, prompt_tokens, completion_tokens)), Ok(search_attempts)) => (
                        serde_json::json!(total_spend),
                        serde_json::json!(prompt_tokens),
                        serde_json::json!(completion_tokens),
                        serde_json::json!(prompt_tokens + completion_tokens),
                        serde_json::json!(search_attempts),
                        "verified",
                        true,
                    ),
                    (Err(_), Ok(search_attempts)) => {
                        (
                            Value::Null,
                            Value::Null,
                            Value::Null,
                            Value::Null,
                            serde_json::json!(search_attempts),
                            "provider_accounting_unverified",
                            false,
                        )
                    },
                    _ => (
                        Value::Null,
                        Value::Null,
                        Value::Null,
                        Value::Null,
                        Value::Null,
                        "query_failed_uncertain",
                        false,
                    ),
                }
            }
            Err(_) => (
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
                "db_open_failed_uncertain",
                false,
            ),
        }
    } else {
        (
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            "missing_state_db",
            false,
        )
    };

    let pool_dir = if state_dir.is_file() {
        state_dir.parent().unwrap_or(state_dir)
    } else {
        state_dir
    };
    let pool_candidates = [
        pool_dir.join(".kaggle_pool/pool.sqlite"),
        pool_dir.join(".kaggle_pool/pool_state.sqlite"),
        pool_dir.join("pool_state.sqlite"),
        pool_dir.join("pool.sqlite"),
        pool_dir.join("controller/pool_state.sqlite"),
    ];
    let pool_path = pool_candidates.iter().find(|p| p.is_file());

    let (kaggle_accts, kaggle_rem, kaggle_used, kaggle_jobs, kaggle_status, kaggle_verified, kaggle_scope) = if let Some(p) = pool_path {
        match Connection::open(p) {
            Ok(pool_conn) => {
                let acct_res: std::result::Result<(i64, f64), rusqlite::Error> = pool_conn.query_row(
                    "SELECT COUNT(*), COALESCE(SUM(gpu_hours_remaining), 0.0) FROM kaggle_accounts",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                );
                let jobs_res: std::result::Result<i64, rusqlite::Error> = pool_conn.query_row(
                    "SELECT COUNT(*) FROM kaggle_jobs",
                    [],
                    |r| r.get(0),
                );
                match (acct_res, jobs_res) {
                    (Ok((accts, rem)), Ok(jobs)) => {
                        if accts == 0 {
                            (
                                serde_json::json!(0),
                                Value::Null,
                                Value::Null,
                                serde_json::json!(jobs),
                                "no_registered_accounts",
                                false,
                                "local pool has zero registered accounts; remote telemetry not connected",
                            )
                        } else {
                            (
                                serde_json::json!(accts),
                                serde_json::json!(rem),
                                Value::Null,
                                serde_json::json!(jobs),
                                "registered_accounts_present_remote_unverified",
                                false,
                                "remote quota and usage uncertain; remote telemetry not verified",
                            )
                        }
                    }
                    _ => (
                        Value::Null,
                        Value::Null,
                        Value::Null,
                        Value::Null,
                        "pool_query_failed_uncertain",
                        false,
                        "failed to query local pool database tables",
                    ),
                }
            }
            Err(_) => (
                Value::Null,
                Value::Null,
                Value::Null,
                Value::Null,
                "pool_open_failed_uncertain",
                false,
                "failed to open local pool database",
            ),
        }
    } else {
        (
            Value::Null,
            Value::Null,
            Value::Null,
            Value::Null,
            "unconnected_or_missing_pool",
            false,
            "no pool database located; remote compute not configured",
        )
    };

    Ok(serde_json::json!({
        "project": project,
        "total_spend_millicents": spend_val,
        "prompt_tokens": prompt_val,
        "completion_tokens": comp_val,
        "total_tokens": tot_val,
        "search_attempts": search_val,
        "accounting_status": accounting_status,
        "accounting_verified": accounting_verified,
        "kaggle_accounts_count": kaggle_accts,
        "kaggle_gpu_hours_remaining": kaggle_rem,
        "kaggle_hours_used": kaggle_used,
        "kaggle_jobs_count": kaggle_jobs,
        "kaggle_status": kaggle_status,
        "kaggle_remote_usage_verified": kaggle_verified,
        "accounting_scope": kaggle_scope,
    }))
}
