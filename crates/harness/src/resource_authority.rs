//! Single authoritative controller resource management across tokens, monetary spend,
//! web search queries, and Kaggle compute quotas.

use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use sha2::{Digest, Sha256};
use crate::agents::gateway::accounting_request_key;
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

/// Ensure the operational resource schemas exist on the SQLite connection.
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
            response_evidence TEXT,
            PRIMARY KEY(project, attempt_id)
        );
        CREATE INDEX IF NOT EXISTS idx_search_attempts_status ON operational_search_attempts(project, status);

        CREATE TABLE IF NOT EXISTS operational_kaggle_jobs (
            project TEXT NOT NULL,
            job_id TEXT NOT NULL,
            account_username TEXT NOT NULL,
            slug TEXT NOT NULL,
            accelerator TEXT NOT NULL,
            requested_hours REAL NOT NULL,
            actual_hours REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL CHECK(status IN ('reserved', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')),
            created_ms INTEGER NOT NULL,
            settled_ms INTEGER,
            error_message TEXT,
            PRIMARY KEY(project, job_id)
        );
        CREATE INDEX IF NOT EXISTS idx_kaggle_jobs_status ON operational_kaggle_jobs(project, status);

        CREATE TABLE IF NOT EXISTS operational_token_reservations (
            project TEXT NOT NULL,
            reservation_id TEXT NOT NULL,
            role_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL,
            completion_tokens INTEGER NOT NULL,
            cost_millicents INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('reserved', 'ok', 'failed', 'outcome_unknown', 'cancelled')),
            created_ms INTEGER NOT NULL,
            settled_ms INTEGER,
            error_message TEXT,
            PRIMARY KEY(project, reservation_id)
        );
        CREATE INDEX IF NOT EXISTS idx_token_res_status ON operational_token_reservations(project, status);",
    ).map_err(|e| format!("cannot initialize resource authority schema: {e}"))?;

    // Safe backward compatibility migration for pre-existing tables missing new columns
    let _ = conn.execute("ALTER TABLE operational_search_attempts ADD COLUMN response_evidence TEXT", []);

    // Safe backward compatibility migration for pre-existing tables missing 'cancelled' in CHECK constraint
    let op_table_sql: Option<String> = conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='operational_token_reservations'",
        [],
        |r| r.get(0),
    ).optional().map_err(|e| e.to_string())?;

    if let Some(sql) = op_table_sql {
        if !sql.contains("'cancelled'") {
            let tx = conn.unchecked_transaction().map_err(|e| e.to_string())?;
            tx.execute_batch(
                "CREATE TABLE operational_token_reservations_v2 (
                     project TEXT NOT NULL,
                     reservation_id TEXT NOT NULL,
                     role_id TEXT NOT NULL,
                     model_id TEXT NOT NULL,
                     prompt_tokens INTEGER NOT NULL,
                     completion_tokens INTEGER NOT NULL,
                     cost_millicents INTEGER NOT NULL,
                     status TEXT NOT NULL CHECK(status IN ('reserved', 'ok', 'failed', 'outcome_unknown', 'cancelled')),
                     created_ms INTEGER NOT NULL,
                     settled_ms INTEGER,
                     error_message TEXT,
                     PRIMARY KEY(project, reservation_id)
                 );
                 INSERT INTO operational_token_reservations_v2 SELECT * FROM operational_token_reservations;
                 DROP TABLE operational_token_reservations;
                 ALTER TABLE operational_token_reservations_v2 RENAME TO operational_token_reservations;
                 CREATE INDEX IF NOT EXISTS idx_token_res_status ON operational_token_reservations(project, status);"
            ).map_err(|e| format!("failed to migrate operational_token_reservations check constraint: {e}"))?;
            tx.commit().map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

/// Fence a legacy operational ledger, then import its liabilities into the
/// controller database. Fencing commits first: a crash can leave the old ledger
/// read-only, but cannot make its remaining obligations disappear. Repeating
/// the import is safe only for the same frozen logical contents.
pub fn import_legacy_operational_ledger(target: &Path, legacy: &Path) -> Result<usize> {
    let meta = std::fs::symlink_metadata(legacy).map_err(|e| e.to_string())?;
    if meta.file_type().is_symlink() || !meta.is_file() { return Err("legacy operational ledger must be a regular file".into()); }
    let legacy = std::fs::canonicalize(legacy).map_err(|e| e.to_string())?;
    let target = if target.exists() {
        let meta = std::fs::symlink_metadata(target).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err("target operational ledger must be a regular file".into());
        }
        std::fs::canonicalize(target).map_err(|e| e.to_string())?
    } else {
        let parent = target.parent().ok_or("target operational ledger has no parent directory")?;
        let parent = std::fs::canonicalize(parent).map_err(|e| e.to_string())?;
        let name = target.file_name().ok_or("target operational ledger has no filename")?;
        parent.join(name)
    };
    if legacy == target { return Err("legacy and target operational ledgers are identical".into()); }
    let mut old = Connection::open(&legacy).map_err(|e| e.to_string())?;
    old.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&old)?;
    let fence = old.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    let provider_table: bool = fence.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='provider_accounting')", [], |r| r.get(0)).map_err(|e| e.to_string())?;
    if provider_table {
        let count: i64 = fence.query_row("SELECT COUNT(*) FROM provider_accounting", [], |r| r.get(0)).map_err(|e| e.to_string())?;
        if count != 0 { return Err("legacy ledger contains provider accounting requiring separate reconciliation".into()); }
    }
    for table in ["operational_search_attempts", "operational_kaggle_jobs", "operational_token_reservations"] {
        for operation in ["INSERT", "UPDATE", "DELETE"] {
            let name = format!("migration_fence_{table}_{}", operation.to_ascii_lowercase());
            let sql = format!("CREATE TRIGGER IF NOT EXISTS {name} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'operational ledger migrated; use controller authority'); END");
            fence.execute_batch(&sql).map_err(|e| e.to_string())?;
        }
    }
    fence.commit().map_err(|e| e.to_string())?;
    drop(old);

    let mut dest = Connection::open(&target).map_err(|e| e.to_string())?;
    dest.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&dest)?;
    dest.execute_batch("CREATE TABLE IF NOT EXISTS operational_ledger_imports (source_path TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, row_count INTEGER NOT NULL, imported_ms INTEGER NOT NULL)").map_err(|e| e.to_string())?;
    dest.execute("ATTACH DATABASE ?1 AS legacy_operational", [legacy.to_string_lossy().as_ref()]).map_err(|e| e.to_string())?;
    let tx = dest.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    let tables = [
        ("operational_search_attempts", "project,attempt_id,query_text,sanitized_query,provider,status,cost_millicents,created_ms,settled_ms,error_message,response_evidence", "project,attempt_id"),
        ("operational_kaggle_jobs", "project,job_id,account_username,slug,accelerator,requested_hours,actual_hours,status,created_ms,settled_ms,error_message", "project,job_id"),
        ("operational_token_reservations", "project,reservation_id,role_id,model_id,prompt_tokens,completion_tokens,cost_millicents,status,created_ms,settled_ms,error_message", "project,reservation_id"),
    ];
    let mut hasher = Sha256::new();
    let mut count = 0usize;
    for (table, columns, order) in tables {
        let query = format!("SELECT COALESCE(json_group_array(json_array({columns})), '[]') FROM (SELECT {columns} FROM legacy_operational.{table} ORDER BY {order})");
        let data: String = tx.query_row(&query, [], |r| r.get(0)).map_err(|e| e.to_string())?;
        hasher.update(table.as_bytes()); hasher.update((data.len() as u64).to_be_bytes()); hasher.update(data.as_bytes());
        let n: i64 = tx.query_row(&format!("SELECT COUNT(*) FROM legacy_operational.{table}"), [], |r| r.get(0)).map_err(|e| e.to_string())?;
        count = count.checked_add(usize::try_from(n).map_err(|_| "invalid legacy row count")?).ok_or("legacy row count overflow")?;
    }
    let digest = format!("{:x}", hasher.finalize());
    let source = legacy.to_string_lossy();
    let previous: Option<(String, i64)> = tx.query_row("SELECT source_sha256,row_count FROM operational_ledger_imports WHERE source_path=?1", [source.as_ref()], |r| Ok((r.get(0)?, r.get(1)?))).optional().map_err(|e| e.to_string())?;
    if let Some((old_digest, old_count)) = previous {
        if old_digest != digest || old_count != count as i64 { return Err("fenced legacy ledger changed after import; reconciliation required".into()); }
        tx.commit().map_err(|e| e.to_string())?;
        return Ok(count);
    }
    let previous_by_digest: Option<(String, i64)> = tx.query_row("SELECT source_path, row_count FROM operational_ledger_imports WHERE source_sha256=?1", [&digest], |r| Ok((r.get(0)?, r.get(1)?))).optional().map_err(|e| e.to_string())?;
    if let Some((_, old_count)) = previous_by_digest {
        if old_count == count as i64 {
            tx.execute("INSERT OR REPLACE INTO operational_ledger_imports(source_path,source_sha256,row_count,imported_ms) VALUES(?1,?2,?3,?4)", params![source.as_ref(), digest, count as i64, now_ms()]).map_err(|e| e.to_string())?;
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(count);
        }
    }
    for (table, columns, _) in tables {
        let sql = format!("INSERT INTO main.{table} ({columns}) SELECT {columns} FROM legacy_operational.{table}");
        tx.execute(&sql, []).map_err(|e| format!("legacy {table} import failed; existing liabilities retained: {e}"))?;
    }
    tx.execute("INSERT INTO operational_ledger_imports(source_path,source_sha256,row_count,imported_ms) VALUES(?1,?2,?3,?4)", params![source.as_ref(), digest, count as i64, now_ms()]).map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(count)
}

/// Sanitize an outbound research query, blocking private tokens, file extensions, and data rows.
pub fn sanitize_research_query(query: &str) -> Result<String> {
    let clean = query.trim();
    if clean.is_empty() {
        return Err("search query cannot be empty".into());
    }
    if clean.len() > 256 {
        return Err("search query exceeds 256 character bound".into());
    }
    if clean.contains("sk-") || clean.contains("ghp_") || clean.to_lowercase().contains("bearer ") {
        return Err("search query contains potential secret or credential tokens".into());
    }
    if clean.contains(".csv") || clean.contains(".tsv") || clean.contains(".parquet") {
        return Err("search query contains forbidden dataset file extensions".into());
    }
    if clean.matches(',').count() >= 3 || clean.matches('\t').count() >= 2 {
        return Err("search query appears to contain raw tabular row delimiters".into());
    }
    Ok(clean.to_string())
}

/// Sanitize external search snippets to neutralize prompt injection vectors.
pub fn sanitize_research_content(text: &str) -> String {
    let patterns = [
        "ignore all previous instructions",
        "ignore previous instructions",
        "system: you are",
        "disregard all prior prompts",
        "<|im_start|>",
        "<|im_end|>",
    ];
    let mut result: String = text.chars().take(16_384).collect();
    if text.chars().count() > 16_384 { result.push_str(" [TRUNCATED]"); }
    for pat in patterns {
        // ASCII lowercasing preserves byte offsets even when preceding text is Unicode.
        while let Some(pos) = result.to_ascii_lowercase().find(pat) {
            result.replace_range(pos..pos + pat.len(), "[BLOCKED_INJECTION_PHRASE]");
        }
    }
    result
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
    let query_clean = sanitize_research_query(query)?;
    let sanitized_clean = if sanitized_query.trim().is_empty() {
        query_clean.clone()
    } else {
        sanitize_research_query(sanitized_query)?
    };
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
        "INSERT INTO operational_search_attempts(project, attempt_id, query_text, sanitized_query, provider, status, cost_millicents, created_ms, settled_ms, error_message, response_evidence)
         VALUES(?1, ?2, ?3, ?4, ?5, 'admitted', 0, ?6, NULL, NULL, NULL)",
        params![project, &attempt_id, &query_clean, &sanitized_clean, provider, timestamp],
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
    settle_search_attempt_with_evidence(db_path, project, attempt_id, status, error, cost_millicents, None)
}

/// Settle an admitted search attempt with structured response evidence receipts.
pub fn settle_search_attempt_with_evidence(
    db_path: &Path,
    project: &str,
    attempt_id: &str,
    status: &str,
    error: Option<&str>,
    cost_millicents: i64,
    response_evidence: Option<&str>,
) -> Result<()> {
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let existing: Option<(String, i64, Option<String>, Option<String>)> = tx.query_row(
        "SELECT status, cost_millicents, error_message, response_evidence FROM operational_search_attempts WHERE project=?1 AND attempt_id=?2",
        params![project, attempt_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    ).optional().map_err(|e| e.to_string())?;

    let (current, previous_cost, previous_error, previous_evidence) = existing.ok_or_else(|| format!("search attempt '{attempt_id}' not found for project '{project}'"))?;
    let terminal_states = ["success", "failed", "timeout", "outcome_unknown"];
    if terminal_states.contains(&current.as_str()) {
        if current == status && previous_cost == cost_millicents && previous_error.as_deref() == error && previous_evidence.as_deref() == response_evidence {
            return Ok(()); // idempotent no-op
        }
        return Err(format!(
            "rejected conflicting settlement for search attempt '{attempt_id}': already terminal ({current}), requested ({status})"
        ));
    }

    let timestamp = now_ms();
    tx.execute(
        "UPDATE operational_search_attempts
         SET status=?1, settled_ms=?2, error_message=?3, cost_millicents=?4, response_evidence=?5
         WHERE project=?6 AND attempt_id=?7",
        params![status, timestamp, error, cost_millicents, response_evidence, project, attempt_id],
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

/// Atomically admit a Kaggle compute job under the configured accelerator hours budget.
pub fn admit_kaggle_job(
    db_path: &Path,
    project: &str,
    job_id: &str,
    account_username: &str,
    slug: &str,
    accelerator: &str,
    requested_hours: f64,
    limit_hours: f64,
) -> Result<Option<String>> {
    if !requested_hours.is_finite() || requested_hours <= 0.0 {
        return Err("requested_hours must be positive and finite".into());
    }
    if !limit_hours.is_finite() || limit_hours < 0.0 {
        return Err("limit_hours must be non-negative and finite".into());
    }
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let (reserved_hours, actual_hours): (f64, f64) = tx.query_row(
        "SELECT COALESCE(SUM(CASE WHEN status IN ('reserved', 'dispatched', 'timeout', 'outcome_unknown') THEN MAX(requested_hours, actual_hours) ELSE 0.0 END), 0.0),
                COALESCE(SUM(CASE WHEN status NOT IN ('reserved', 'dispatched', 'timeout', 'outcome_unknown') THEN actual_hours ELSE 0.0 END), 0.0)
         FROM operational_kaggle_jobs WHERE project = ?1",
        params![project],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).map_err(|e| e.to_string())?;

    if (reserved_hours + actual_hours + requested_hours) > limit_hours + 1e-6 {
        tx.rollback().map_err(|e| e.to_string())?;
        return Ok(None);
    }

    let timestamp = now_ms();
    tx.execute(
        "INSERT INTO operational_kaggle_jobs(project, job_id, account_username, slug, accelerator, requested_hours, actual_hours, status, created_ms, settled_ms, error_message)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, 0.0, 'reserved', ?7, NULL, NULL)",
        params![project, job_id, account_username, slug, accelerator, requested_hours, timestamp],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(Some(job_id.to_string()))
}

/// Idempotently settle a Kaggle compute job. Rejects conflicting mutations if already terminal.
pub fn settle_kaggle_job(
    db_path: &Path,
    project: &str,
    job_id: &str,
    status: &str,
    actual_hours: f64,
    error: Option<&str>,
) -> Result<()> {
    if !actual_hours.is_finite() || actual_hours < 0.0 || !["success", "failed", "timeout", "outcome_unknown"].contains(&status) {
        return Err("invalid Kaggle settlement".into());
    }
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let existing: Option<(String, f64, Option<String>)> = tx.query_row(
        "SELECT status, actual_hours, error_message FROM operational_kaggle_jobs WHERE project=?1 AND job_id=?2",
        params![project, job_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional().map_err(|e| e.to_string())?;

    let (current, prev_hours, prev_error) = existing.ok_or_else(|| format!("kaggle job '{job_id}' not found for project '{project}'"))?;
    let terminal_states = ["success", "failed", "timeout", "outcome_unknown"];
    if terminal_states.contains(&current.as_str()) {
        if current == status && (prev_hours - actual_hours).abs() < 1e-6 && prev_error.as_deref() == error {
            return Ok(()); // idempotent no-op
        }
        return Err(format!(
            "rejected conflicting settlement for kaggle job '{job_id}': already terminal ({current}), requested ({status})"
        ));
    }

    let timestamp = now_ms();
    tx.execute(
        "UPDATE operational_kaggle_jobs
         SET status=?1, settled_ms=?2, error_message=?3, actual_hours=?4
         WHERE project=?5 AND job_id=?6",
        params![status, timestamp, error, actual_hours, project, job_id],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

/// Query deduplicated token and spend liability across provider_accounting and operational_token_reservations.
///
/// If a request has an operational reservation AND a corresponding provider_accounting record,
/// they represent the same physical request and must be counted exactly once to avoid double-counting.
pub fn query_deduplicated_token_and_spend(
    conn: &Connection,
    project: &str,
) -> Result<(u64, i64, i64, i64, bool)> {
    let has_pa: bool = conn.query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provider_accounting'",
        [],
        |r| r.get(0),
    ).map(|c| c > 0).map_err(|e| e.to_string())?;

    let mut unpriced_provider_usage = false;
    let mut pa_records: Vec<(String, String, i64, i64, i64)> = Vec::new(); // (request_id, task_id, prompt, comp, cost)

    if has_pa {
        unpriced_provider_usage = conn.query_row::<i64, _, _>(
            "SELECT COUNT(*) FROM provider_accounting WHERE project=?1 AND status='success_cost_unpriced'",
            params![project],
            |r| r.get(0),
        ).map_err(|e| e.to_string())? > 0;

        let mut pragma_stmt = conn.prepare("PRAGMA table_info(provider_accounting)").map_err(|e| e.to_string())?;
        let columns: Vec<String> = pragma_stmt.query_map([], |r| r.get::<_, String>(1))
            .map_err(|e| e.to_string())?
            .filter_map(|r| r.ok())
            .collect();
        let has_request_id = columns.iter().any(|c| c == "request_id");
        let has_task_id = columns.iter().any(|c| c == "task_id");

        if has_request_id && has_task_id {
            let mut stmt = conn.prepare(
                "SELECT request_id, task_id, COALESCE(prompt_tokens, 0), COALESCE(completion_tokens, 0), COALESCE(cost_millicents, 0)
                 FROM provider_accounting WHERE project = ?1 AND status IN ('ok', 'reserved', 'outcome_unknown', 'priced', 'outcome_unknown_ceiling_estimate', 'success_cost_unpriced')"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, i64>(4)?))
            }).map_err(|e| e.to_string())?;
            for row in rows {
                pa_records.push(row.map_err(|e| e.to_string())?);
            }
        } else {
            let mut stmt = conn.prepare(
                "SELECT COALESCE(prompt_tokens, 0), COALESCE(completion_tokens, 0), COALESCE(cost_millicents, 0)
                 FROM provider_accounting WHERE project = ?1 AND status IN ('ok', 'reserved', 'outcome_unknown', 'priced', 'outcome_unknown_ceiling_estimate', 'success_cost_unpriced')"
            ).map_err(|e| e.to_string())?;
            let rows = stmt.query_map(params![project], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))
            }).map_err(|e| e.to_string())?;
            for (idx, row) in rows.enumerate() {
                let (prompt, comp, cost) = row.map_err(|e| e.to_string())?;
                pa_records.push((format!("legacy_pa_{idx}"), String::new(), prompt, comp, cost));
            }
        }
    }

    let mut op_records: Vec<(String, i64, i64, i64)> = Vec::new(); // (reservation_id, prompt, comp, cost)
    let has_op: bool = conn.query_row::<i64, _, _>(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='operational_token_reservations'",
        [],
        |r| r.get(0),
    ).map(|c| c > 0).map_err(|e| e.to_string())?;

    if has_op {
        let mut stmt = conn.prepare(
            "SELECT reservation_id, COALESCE(prompt_tokens, 0), COALESCE(completion_tokens, 0), COALESCE(cost_millicents, 0)
             FROM operational_token_reservations WHERE project = ?1 AND status IN ('reserved', 'ok', 'outcome_unknown')"
        ).map_err(|e| e.to_string())?;
        let rows = stmt.query_map(params![project], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?))
        }).map_err(|e| e.to_string())?;
        for row in rows {
            op_records.push(row.map_err(|e| e.to_string())?);
        }
    }

    let mut matched_pa_indices = HashSet::new();
    let mut matched_op_indices = HashSet::new();
    let mut total_prompt: i64 = 0;
    let mut total_comp: i64 = 0;
    let mut total_cost: i64 = 0;

    for (op_idx, (res_id, op_prompt, op_comp, op_cost)) in op_records.iter().enumerate() {
        for (pa_idx, (pa_req_id, pa_task_id, pa_prompt, pa_comp, pa_cost)) in pa_records.iter().enumerate() {
            if matched_pa_indices.contains(&pa_idx) {
                continue;
            }
            let key = accounting_request_key(project, pa_task_id, res_id);
            if pa_req_id == res_id || pa_req_id == &key {
                matched_pa_indices.insert(pa_idx);
                matched_op_indices.insert(op_idx);
                total_prompt = total_prompt.checked_add((*op_prompt).max(*pa_prompt)).ok_or("prompt token overflow")?;
                total_comp = total_comp.checked_add((*op_comp).max(*pa_comp)).ok_or("completion token overflow")?;
                total_cost = total_cost.checked_add((*op_cost).max(*pa_cost)).ok_or("cost millicents overflow")?;
                break;
            }
        }
    }

    for (op_idx, (_, op_prompt, op_comp, op_cost)) in op_records.iter().enumerate() {
        if !matched_op_indices.contains(&op_idx) {
            total_prompt = total_prompt.checked_add(*op_prompt).ok_or("prompt token overflow")?;
            total_comp = total_comp.checked_add(*op_comp).ok_or("completion token overflow")?;
            total_cost = total_cost.checked_add(*op_cost).ok_or("cost millicents overflow")?;
        }
    }

    for (pa_idx, (_, _, pa_prompt, pa_comp, pa_cost)) in pa_records.iter().enumerate() {
        if !matched_pa_indices.contains(&pa_idx) {
            total_prompt = total_prompt.checked_add(*pa_prompt).ok_or("prompt token overflow")?;
            total_comp = total_comp.checked_add(*pa_comp).ok_or("completion token overflow")?;
            total_cost = total_cost.checked_add(*pa_cost).ok_or("cost millicents overflow")?;
        }
    }

    let total_tokens = (total_prompt as u64).checked_add(total_comp as u64).ok_or("token count overflow")?;
    Ok((total_tokens, total_cost, total_prompt, total_comp, unpriced_provider_usage))
}

/// Atomically admit an operational token reservation under token and monetary spend ceilings.
pub fn admit_token_reservation(
    db_path: &Path,
    project: &str,
    reservation_id: &str,
    role_id: &str,
    model_id: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    cost_millicents: i64,
    token_limit: u64,
    spend_limit: i64,
) -> Result<Option<String>> {
    if cost_millicents < 0 || spend_limit < 0 { return Err("negative monetary budget or reservation".into()); }
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let requested_tokens = (prompt_tokens as u64) + (completion_tokens as u64);

    let (total_tokens, total_spend, _, _, unpriced) = query_deduplicated_token_and_spend(&tx, project)?;
    if unpriced {
        return Err("unpriced provider usage requires reconciliation before admission".into());
    }

    if total_tokens.checked_add(requested_tokens).ok_or("token admission overflow")? > token_limit
        || total_spend.checked_add(cost_millicents).ok_or("spend admission overflow")? > spend_limit {
        tx.rollback().map_err(|e| e.to_string())?;
        return Ok(None);
    }

    let timestamp = now_ms();
    tx.execute(
        "INSERT INTO operational_token_reservations(project, reservation_id, role_id, model_id, prompt_tokens, completion_tokens, cost_millicents, status, created_ms, settled_ms, error_message)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'reserved', ?8, NULL, NULL)",
        params![project, reservation_id, role_id, model_id, prompt_tokens, completion_tokens, cost_millicents, timestamp],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(Some(reservation_id.to_string()))
}

/// Idempotently settle an operational token reservation. Rejects conflicting mutations if already terminal.
pub fn settle_token_reservation(
    db_path: &Path,
    project: &str,
    reservation_id: &str,
    status: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    cost_millicents: i64,
    error: Option<&str>,
) -> Result<()> {
    if cost_millicents < 0 || !["ok", "failed", "outcome_unknown", "cancelled"].contains(&status) { return Err("invalid token settlement".into()); }
    let mut conn = Connection::open(db_path).map_err(|e| e.to_string())?;
    conn.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
    ensure_resource_authority_schema(&conn)?;

    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    let existing: Option<(String, u32, u32, i64, Option<String>)> = tx.query_row(
        "SELECT status, prompt_tokens, completion_tokens, cost_millicents, error_message FROM operational_token_reservations WHERE project=?1 AND reservation_id=?2",
        params![project, reservation_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    ).optional().map_err(|e| e.to_string())?;

    let (current, prev_prompt_tokens, prev_comp_tokens, prev_cost, prev_error) = existing.ok_or_else(|| format!("token reservation '{reservation_id}' not found for project '{project}'"))?;
    let terminal_states = ["ok", "failed", "outcome_unknown", "cancelled"];
    if terminal_states.contains(&current.as_str()) {
        if current == status {
            if status == "cancelled" {
                return Ok(()); // idempotent no-op for cancelled
            }
            if prev_prompt_tokens == prompt_tokens && prev_comp_tokens == completion_tokens && prev_cost == cost_millicents && prev_error.as_deref() == error {
                return Ok(()); // idempotent no-op
            }
        }
        return Err(format!(
            "rejected conflicting settlement for token reservation '{reservation_id}': already terminal ({current}), requested ({status})"
        ));
    }

    if status == "cancelled" {
        let timestamp = now_ms();
        tx.execute(
            "UPDATE operational_token_reservations
             SET status='cancelled', settled_ms=?1, error_message=?2, prompt_tokens=0, completion_tokens=0, cost_millicents=0
             WHERE project=?3 AND reservation_id=?4",
            params![timestamp, error.unwrap_or("cancelled before dispatch"), project, reservation_id],
        ).map_err(|e| e.to_string())?;
        tx.commit().map_err(|e| e.to_string())?;
        return Ok(());
    }

    if status == "outcome_unknown" && (prompt_tokens < prev_prompt_tokens || completion_tokens < prev_comp_tokens || cost_millicents < prev_cost) {
        return Err("unknown outcome cannot release reserved liability".into());
    }
    let timestamp = now_ms();
    tx.execute(
        "UPDATE operational_token_reservations
         SET status=?1, settled_ms=?2, error_message=?3, prompt_tokens=?4, completion_tokens=?5, cost_millicents=?6
         WHERE project=?7 AND reservation_id=?8",
        params![status, timestamp, error, prompt_tokens, completion_tokens, cost_millicents, project, reservation_id],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
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

    let mut kag_job_count: i64 = 0;
    let mut kag_hours_spent: Option<f64> = None;

    let (spend_val, prompt_val, comp_val, tot_val, search_val, accounting_status, accounting_verified) = if let Some(p) = state_file {
        match Connection::open(&p) {
            Ok(conn) => {
                conn.busy_timeout(std::time::Duration::from_secs(5)).map_err(|e| e.to_string())?;
                ensure_resource_authority_schema(&conn)?;

                let pa_exists = conn.query_row::<i64, _, _>(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='provider_accounting'",
                    [],
                    |r| r.get(0),
                ).map_err(|e| e.to_string())?;

                let (total_tokens, total_spend, prompt_tokens, completion_tokens, unpriced_provider_usage) =
                    query_deduplicated_token_and_spend(&conn, project)?;

                // 2. Query search attempts
                let search_res: std::result::Result<i64, rusqlite::Error> = conn.query_row(
                    "SELECT COUNT(*) FROM operational_search_attempts WHERE project = ?1 AND status IN ('admitted', 'dispatched', 'success', 'failed', 'timeout', 'outcome_unknown')",
                    params![project],
                    |r| r.get(0),
                );

                // 3. Query operational Kaggle jobs from state db
                kag_job_count = conn.query_row(
                    "SELECT COUNT(*) FROM operational_kaggle_jobs WHERE project = ?1",
                    params![project],
                    |r| r.get(0),
                ).map_err(|e| format!("cannot query Kaggle job count: {e}"))?;
                kag_hours_spent = if kag_job_count > 0 {
                    conn.query_row(
                        "SELECT CASE WHEN SUM(CASE WHEN status IN ('reserved', 'dispatched', 'timeout', 'outcome_unknown') THEN 1 ELSE 0 END) > 0 THEN NULL ELSE SUM(actual_hours) END FROM operational_kaggle_jobs WHERE project = ?1",
                        params![project],
                        |r| r.get(0),
                    ).map_err(|e| format!("cannot query Kaggle usage: {e}"))?
                } else {
                    None
                };

                match search_res {
                    Ok(search_attempts) => (
                        if unpriced_provider_usage { Value::Null } else { serde_json::json!(total_spend) },
                        serde_json::json!(prompt_tokens),
                        serde_json::json!(completion_tokens),
                        serde_json::json!(total_tokens),
                        serde_json::json!(search_attempts),
                        if unpriced_provider_usage { "unpriced_provider_usage" } else if pa_exists > 0 || total_tokens > 0 { "local_accounting_queried" } else { "provider_accounting_unavailable" },
                        (pa_exists > 0 || total_tokens > 0) && !unpriced_provider_usage,
                    ),
                    Err(_) => (
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
                                kag_hours_spent.map(|h| serde_json::json!(h)).unwrap_or(Value::Null),
                                serde_json::json!(jobs.max(kag_job_count)),
                                "no_registered_accounts",
                                false,
                                "local pool has zero registered accounts; remote telemetry not connected",
                            )
                        } else {
                            (
                                serde_json::json!(accts),
                                serde_json::json!(rem),
                                kag_hours_spent.map(|h| serde_json::json!(h)).unwrap_or(Value::Null),
                                serde_json::json!(jobs.max(kag_job_count)),
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
    } else if kag_job_count > 0 {
        (
            serde_json::json!(0),
            Value::Null,
            kag_hours_spent.map(|h| serde_json::json!(h)).unwrap_or(Value::Null),
            serde_json::json!(kag_job_count),
            "operational_jobs_recorded_no_pool",
            false,
            "local jobs recorded in state db; remote telemetry not connected",
        )
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
