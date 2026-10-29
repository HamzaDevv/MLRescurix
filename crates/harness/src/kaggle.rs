use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::Result;

static RND_COUNTER: AtomicU64 = AtomicU64::new(1);

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn generate_token(prefix: &str) -> String {
    let count = RND_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut hasher = Sha256::new();
    hasher.update(format!("{prefix}:{}:{count}", now_ms()).as_bytes());
    let hex = format!("{:x}", hasher.finalize());
    format!("{prefix}-{}", &hex[..16])
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).map_err(|e| format!("cannot open file {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 16384];
    loop {
        let n = file.read(&mut buf).map_err(|e| format!("read error: {e}"))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KaggleLifecycleState {
    Planned,
    Staged,
    Queued,
    Running,
    Collecting,
    Verified,
    RetryWait,
    CancelRequested,
    Cancelled,
    Failed,
    NeedsResources,
}

impl KaggleLifecycleState {
    pub fn as_str(&self) -> &'static str {
        match self {
            KaggleLifecycleState::Planned => "planned",
            KaggleLifecycleState::Staged => "staged",
            KaggleLifecycleState::Queued => "queued",
            KaggleLifecycleState::Running => "running",
            KaggleLifecycleState::Collecting => "collecting",
            KaggleLifecycleState::Verified => "verified",
            KaggleLifecycleState::RetryWait => "retry_wait",
            KaggleLifecycleState::CancelRequested => "cancel_requested",
            KaggleLifecycleState::Cancelled => "cancelled",
            KaggleLifecycleState::Failed => "failed",
            KaggleLifecycleState::NeedsResources => "needs_resources",
        }
    }

    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "planned" => Ok(KaggleLifecycleState::Planned),
            "staged" => Ok(KaggleLifecycleState::Staged),
            "queued" => Ok(KaggleLifecycleState::Queued),
            "running" => Ok(KaggleLifecycleState::Running),
            "collecting" => Ok(KaggleLifecycleState::Collecting),
            "verified" => Ok(KaggleLifecycleState::Verified),
            "retry_wait" => Ok(KaggleLifecycleState::RetryWait),
            "cancel_requested" => Ok(KaggleLifecycleState::CancelRequested),
            "cancelled" => Ok(KaggleLifecycleState::Cancelled),
            "failed" => Ok(KaggleLifecycleState::Failed),
            "needs_resources" => Ok(KaggleLifecycleState::NeedsResources),
            other => Err(format!("unknown KaggleLifecycleState: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaggleAccount {
    pub username: String,
    pub credential_ref: String,
    pub is_anchor: bool,
    pub gpu_hours_remaining: f64,
    pub tpu_hours_remaining: f64,
    pub last_dispatched_at: u64,
}

impl KaggleAccount {
    pub fn new(username: &str, credential_ref: &str, is_anchor: bool) -> Result<Self> {
        if username.trim().is_empty() {
            return Err("username cannot be empty".into());
        }
        if !credential_ref.starts_with("env:") && !credential_ref.starts_with("file:") {
            return Err("credential_ref must start with 'env:' or 'file:' to prevent plain-text credential leaks".into());
        }
        Ok(Self {
            username: username.to_string(),
            credential_ref: credential_ref.to_string(),
            is_anchor,
            gpu_hours_remaining: 30.0,
            tpu_hours_remaining: 20.0,
            last_dispatched_at: 0,
        })
    }

    pub fn resolve_credential(&self, storage_dir: Option<&Path>) -> Result<String> {
        if self.credential_ref.starts_with("env:") {
            let var = &self.credential_ref[4..];
            let val = std::env::var(var).map_err(|_| format!("environment variable '{var}' not set"))?;
            let trimmed = val.trim();
            if trimmed.is_empty() {
                return Err(format!("environment variable '{var}' is empty"));
            }
            Ok(trimmed.to_string())
        } else if self.credential_ref.starts_with("file:") {
            let path_str = &self.credential_ref[5..];
            let path = PathBuf::from(path_str);
            let effective_path = if !path.is_absolute() && storage_dir.is_some() {
                storage_dir.unwrap().join(path)
            } else {
                path
            };
            if !effective_path.is_file() {
                return Err(format!("credential file '{}' not found", effective_path.display()));
            }
            let content = fs::read_to_string(&effective_path)
                .map_err(|e| format!("cannot read credential file: {e}"))?;
            let trimmed = content.trim();
            if trimmed.is_empty() {
                return Err("credential file is empty".into());
            }
            Ok(trimmed.to_string())
        } else {
            Err("invalid credential_ref prefix".into())
        }
    }
}

pub fn scrub_staged_content(content: &str) -> Result<()> {
    if content.contains("sk-") {
        return Err("security violation: potential API key (sk-...) detected in staged code".into());
    }
    if content.contains("DEEPSEEK_API_KEY") {
        return Err("security violation: reference to DEEPSEEK_API_KEY detected in staged code".into());
    }
    if content.contains("OPENROUTER_API_KEY") {
        return Err("security violation: reference to OPENROUTER_API_KEY detected in staged code".into());
    }
    Ok(())
}

pub fn scrub_staged_directory(dir: &Path) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Ok(content) = fs::read_to_string(&path) {
                scrub_staged_content(&content)?;
            }
        } else if path.is_dir() {
            scrub_staged_directory(&path)?;
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelJobRecord {
    pub job_id: String,
    pub account_username: String,
    pub slug: String,
    pub state: KaggleLifecycleState,
    pub accelerator: String,
    pub lease_token: String,
    pub lease_expires_at: u64,
    pub attempt: u32,
    pub is_competition: bool,
    pub created_at: u64,
    pub updated_at: u64,
}

pub trait KaggleApi: Send + Sync {
    fn push_kernel(&self, account: &KaggleAccount, slug: &str, staged_dir: &Path) -> Result<String>;
    fn get_kernel_status(&self, account: &KaggleAccount, slug: &str) -> Result<String>;
    fn download_kernel_output(&self, account: &KaggleAccount, slug: &str, dest_dir: &Path) -> Result<()>;
    fn submit_competition(&self, account: &KaggleAccount, competition: &str, file: &Path, message: &str) -> Result<String>;
}

pub struct KagglePoolManager {
    db_path: PathBuf,
    staging_dir: PathBuf,
    enable_secondary_accounts: bool,
}

impl KagglePoolManager {
    pub fn new(root_dir: &Path, enable_secondary_accounts: bool) -> Result<Self> {
        let pool_dir = root_dir.join(".kaggle_pool");
        fs::create_dir_all(&pool_dir).map_err(|e| e.to_string())?;
        let staging_dir = pool_dir.join("staging");
        fs::create_dir_all(&staging_dir).map_err(|e| e.to_string())?;

        let db_path = pool_dir.join("pool.sqlite");
        let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS kaggle_accounts (
                username TEXT PRIMARY KEY,
                credential_ref TEXT NOT NULL,
                is_anchor INTEGER NOT NULL DEFAULT 0,
                gpu_hours_remaining REAL NOT NULL DEFAULT 30.0,
                tpu_hours_remaining REAL NOT NULL DEFAULT 20.0,
                last_dispatched_at INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS kaggle_jobs (
                job_id TEXT PRIMARY KEY,
                account_username TEXT NOT NULL,
                slug TEXT NOT NULL,
                state TEXT NOT NULL,
                accelerator TEXT NOT NULL,
                lease_token TEXT NOT NULL,
                lease_expires_at INTEGER NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                is_competition INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY(account_username) REFERENCES kaggle_accounts(username)
            );
            "#,
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            db_path,
            staging_dir,
            enable_secondary_accounts,
        })
    }

    pub fn staging_dir(&self) -> &Path {
        &self.staging_dir
    }

    fn connect(&self) -> Result<Connection> {
        Connection::open(&self.db_path).map_err(|e| e.to_string())
    }

    pub fn register_account(&self, account: KaggleAccount) -> Result<()> {
        let conn = self.connect()?;
        if account.is_anchor {
            let existing_anchor: Option<String> = conn
                .query_row(
                    "SELECT username FROM kaggle_accounts WHERE is_anchor=1 AND username != ?1",
                    params![account.username],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| e.to_string())?;
            if let Some(existing) = existing_anchor {
                return Err(format!("anti-ban violation: another anchor account '{existing}' is already registered; only one anchor allowed"));
            }
        }
        conn.execute(
            r#"
            INSERT INTO kaggle_accounts (username, credential_ref, is_anchor, gpu_hours_remaining, tpu_hours_remaining, last_dispatched_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(username) DO UPDATE SET
                credential_ref = excluded.credential_ref,
                is_anchor = excluded.is_anchor,
                gpu_hours_remaining = excluded.gpu_hours_remaining,
                tpu_hours_remaining = excluded.tpu_hours_remaining,
                last_dispatched_at = excluded.last_dispatched_at;
            "#,
            params![
                account.username,
                account.credential_ref,
                if account.is_anchor { 1 } else { 0 },
                account.gpu_hours_remaining,
                account.tpu_hours_remaining,
                account.last_dispatched_at,
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn list_accounts(&self) -> Result<Vec<KaggleAccount>> {
        let conn = self.connect()?;
        let mut stmt = conn
            .prepare("SELECT username, credential_ref, is_anchor, gpu_hours_remaining, tpu_hours_remaining, last_dispatched_at FROM kaggle_accounts ORDER BY is_anchor DESC, username ASC")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([], |r| {
                Ok(KaggleAccount {
                    username: r.get(0)?,
                    credential_ref: r.get(1)?,
                    is_anchor: r.get::<_, i64>(2)? == 1,
                    gpu_hours_remaining: r.get(3)?,
                    tpu_hours_remaining: r.get(4)?,
                    last_dispatched_at: r.get(5)?,
                })
            })
            .map_err(|e| e.to_string())?;
        let mut list = Vec::new();
        for r in rows {
            list.push(r.map_err(|e| e.to_string())?);
        }
        Ok(list)
    }

    pub fn get_account(&self, username: &str) -> Result<Option<KaggleAccount>> {
        let conn = self.connect()?;
        let res = conn
            .query_row(
                "SELECT username, credential_ref, is_anchor, gpu_hours_remaining, tpu_hours_remaining, last_dispatched_at FROM kaggle_accounts WHERE username = ?1",
                params![username],
                |r| {
                    Ok(KaggleAccount {
                        username: r.get(0)?,
                        credential_ref: r.get(1)?,
                        is_anchor: r.get::<_, i64>(2)? == 1,
                        gpu_hours_remaining: r.get(3)?,
                        tpu_hours_remaining: r.get(4)?,
                        last_dispatched_at: r.get(5)?,
                    })
                },
            )
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(res)
    }

    pub fn lease_worker_account(
        &self,
        accelerator: &str,
        duration_hours: f64,
        is_competition: bool,
    ) -> Result<KaggleAccount> {
        let accounts = self.list_accounts()?;
        if accounts.is_empty() {
            return Err("no Kaggle accounts registered in compute pool".into());
        }

        let is_tpu = accelerator.to_lowercase() == "tpu";
        let mut candidates: Vec<KaggleAccount> = accounts
            .into_iter()
            .filter(|a| {
                if is_competition && !a.is_anchor {
                    return false;
                }
                if !self.enable_secondary_accounts && !a.is_anchor {
                    return false;
                }
                let quota = if is_tpu { a.tpu_hours_remaining } else { a.gpu_hours_remaining };
                quota >= duration_hours
            })
            .collect();

        if candidates.is_empty() {
            return Err("quota exhausted: no eligible account has sufficient remaining accelerator quota".into());
        }

        // Sort by quota descending, then anchor priority
        candidates.sort_by(|a, b| {
            let qa = if is_tpu { a.tpu_hours_remaining } else { a.gpu_hours_remaining };
            let qb = if is_tpu { b.tpu_hours_remaining } else { b.gpu_hours_remaining };
            qb.partial_cmp(&qa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut leased = candidates.remove(0);
        let now = now_ms();
        leased.last_dispatched_at = now;
        if is_tpu {
            leased.tpu_hours_remaining = (leased.tpu_hours_remaining - duration_hours).max(0.0);
        } else {
            leased.gpu_hours_remaining = (leased.gpu_hours_remaining - duration_hours).max(0.0);
        }

        self.register_account(leased.clone())?;
        Ok(leased)
    }

    pub fn stage_job(
        &self,
        job_id: &str,
        account: &KaggleAccount,
        accelerator: &str,
        is_competition: bool,
        code_files: &HashMap<String, String>,
    ) -> Result<KernelJobRecord> {
        if is_competition && !account.is_anchor {
            return Err("anti-ban violation: secondary account cannot submit to competitions".into());
        }

        let stage_dir = self.staging_dir.join(job_id);
        fs::create_dir_all(&stage_dir).map_err(|e| e.to_string())?;

        // Write and scrub code files
        for (rel_path, content) in code_files {
            scrub_staged_content(content)?;
            let target = stage_dir.join(rel_path);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::write(&target, content).map_err(|e| e.to_string())?;
        }

        let now = now_ms();
        let slug = format!("mlr-job-{}", &job_id[..job_id.len().min(16)]);
        let lease_token = generate_token("lease");
        let lease_expires_at = now + 3600 * 1000; // 1 hour lease window

        let record = KernelJobRecord {
            job_id: job_id.to_string(),
            account_username: account.username.clone(),
            slug,
            state: KaggleLifecycleState::Staged,
            accelerator: accelerator.to_string(),
            lease_token: lease_token.clone(),
            lease_expires_at,
            attempt: 1,
            is_competition,
            created_at: now,
            updated_at: now,
        };

        let conn = self.connect()?;
        conn.execute(
            r#"
            INSERT INTO kaggle_jobs (job_id, account_username, slug, state, accelerator, lease_token, lease_expires_at, attempt, is_competition, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            ON CONFLICT(job_id) DO UPDATE SET
                account_username = excluded.account_username,
                slug = excluded.slug,
                state = excluded.state,
                accelerator = excluded.accelerator,
                lease_token = excluded.lease_token,
                lease_expires_at = excluded.lease_expires_at,
                attempt = excluded.attempt,
                is_competition = excluded.is_competition,
                updated_at = excluded.updated_at;
            "#,
            params![
                record.job_id,
                record.account_username,
                record.slug,
                record.state.as_str(),
                record.accelerator,
                record.lease_token,
                record.lease_expires_at as i64,
                record.attempt,
                if record.is_competition { 1 } else { 0 },
                record.created_at as i64,
                record.updated_at as i64,
            ],
        )
        .map_err(|e| e.to_string())?;

        Ok(record)
    }

    pub fn dispatch_job(&self, api: &dyn KaggleApi, job_id: &str) -> Result<KernelJobRecord> {
        let mut job = self.get_job(job_id)?.ok_or_else(|| format!("job '{job_id}' not found"))?;
        let account = self.get_account(&job.account_username)?
            .ok_or_else(|| format!("account '{}' not found", job.account_username))?;

        if job.is_competition && !account.is_anchor {
            return Err("anti-ban violation: secondary account cannot submit to competitions".into());
        }

        let stage_dir = self.staging_dir.join(job_id);
        if !stage_dir.is_dir() {
            return Err(format!("staged directory '{}' does not exist", stage_dir.display()));
        }

        // Push kernel to remote
        api.push_kernel(&account, &job.slug, &stage_dir)?;

        job.state = KaggleLifecycleState::Queued;
        job.updated_at = now_ms();
        self.update_job(&job)?;
        Ok(job)
    }

    pub fn poll_job(&self, api: &dyn KaggleApi, job_id: &str) -> Result<KernelJobRecord> {
        let mut job = self.get_job(job_id)?.ok_or_else(|| format!("job '{job_id}' not found"))?;
        let account = self.get_account(&job.account_username)?
            .ok_or_else(|| format!("account '{}' not found", job.account_username))?;

        let status = api.get_kernel_status(&account, &job.slug)?;
        match status.to_lowercase().as_str() {
            "queued" => job.state = KaggleLifecycleState::Queued,
            "running" => job.state = KaggleLifecycleState::Running,
            "complete" | "completed" | "succeeded" => job.state = KaggleLifecycleState::Collecting,
            "error" | "failed" => job.state = KaggleLifecycleState::Failed,
            "cancelled" => job.state = KaggleLifecycleState::Cancelled,
            other => return Err(format!("unknown remote status from Kaggle API: {other}")),
        }
        job.updated_at = now_ms();
        self.update_job(&job)?;
        Ok(job)
    }

    pub fn collect_and_verify_job(
        &self,
        api: &dyn KaggleApi,
        job_id: &str,
        lease_token: &str,
        dest_dir: &Path,
    ) -> Result<KernelJobRecord> {
        let mut job = self.get_job(job_id)?.ok_or_else(|| format!("job '{job_id}' not found"))?;
        let account = self.get_account(&job.account_username)?
            .ok_or_else(|| format!("account '{}' not found", job.account_username))?;

        let now = now_ms();

        // 1. Fenced atomic lease check
        if job.lease_token != lease_token {
            return Err(format!("stale lease: collection token '{lease_token}' does not match active lease '{}'", job.lease_token));
        }
        if now > job.lease_expires_at {
            return Err(format!("stale lease: lease expired at {} (current time: {now})", job.lease_expires_at));
        }

        // 2. Download kernel output
        fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
        api.download_kernel_output(&account, &job.slug, dest_dir)?;

        // 3. Checksum verification against manifest.json
        let manifest_file = dest_dir.join("manifest.json");
        if !manifest_file.is_file() {
            return Err(format!("missing manifest.json in downloaded artifacts at '{}'", dest_dir.display()));
        }
        let manifest_bytes = fs::read(&manifest_file).map_err(|e| e.to_string())?;
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| format!("invalid manifest.json format: {e}"))?;

        if let Some(files) = manifest.get("files").and_then(|f| f.as_object()) {
            for (filename, expected_val) in files {
                let expected_hash = expected_val.as_str().ok_or_else(|| format!("file '{filename}' hash must be string"))?;
                let file_path = dest_dir.join(filename);
                if !file_path.is_file() {
                    return Err(format!("checksum verification failed: required file '{filename}' missing from output"));
                }
                let actual_hash = file_sha256(&file_path)?;
                if actual_hash != expected_hash {
                    return Err(format!("checksum mismatch: file '{filename}' expected sha256 '{expected_hash}', got '{actual_hash}'"));
                }
            }
        } else {
            return Err("manifest.json missing 'files' dictionary for artifact checksum verification".into());
        }

        job.state = KaggleLifecycleState::Verified;
        job.updated_at = now;
        self.update_job(&job)?;
        Ok(job)
    }

    pub fn submit_competition(
        &self,
        api: &dyn KaggleApi,
        account: &KaggleAccount,
        competition: &str,
        file: &Path,
        message: &str,
    ) -> Result<String> {
        if !account.is_anchor {
            return Err("anti-ban violation: secondary account cannot submit to competitions".into());
        }
        api.submit_competition(account, competition, file, message)
    }

    pub fn get_job(&self, job_id: &str) -> Result<Option<KernelJobRecord>> {
        let conn = self.connect()?;
        let res = conn
            .query_row(
                r#"
                SELECT job_id, account_username, slug, state, accelerator, lease_token, lease_expires_at, attempt, is_competition, created_at, updated_at
                FROM kaggle_jobs WHERE job_id = ?1
                "#,
                params![job_id],
                |r| {
                    let state_str: String = r.get(3)?;
                    let state = KaggleLifecycleState::from_str(&state_str)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(std::io::Error::new(std::io::ErrorKind::Other, e))))?;
                    Ok(KernelJobRecord {
                        job_id: r.get(0)?,
                        account_username: r.get(1)?,
                        slug: r.get(2)?,
                        state,
                        accelerator: r.get(4)?,
                        lease_token: r.get(5)?,
                        lease_expires_at: r.get::<_, i64>(6)? as u64,
                        attempt: r.get(7)?,
                        is_competition: r.get::<_, i64>(8)? == 1,
                        created_at: r.get::<_, i64>(9)? as u64,
                        updated_at: r.get::<_, i64>(10)? as u64,
                    })
                },
            )
            .optional()
            .map_err(|e| e.to_string())?;
        Ok(res)
    }

    fn update_job(&self, job: &KernelJobRecord) -> Result<()> {
        let conn = self.connect()?;
        conn.execute(
            r#"
            UPDATE kaggle_jobs SET
                account_username = ?2,
                slug = ?3,
                state = ?4,
                accelerator = ?5,
                lease_token = ?6,
                lease_expires_at = ?7,
                attempt = ?8,
                is_competition = ?9,
                updated_at = ?10
            WHERE job_id = ?1;
            "#,
            params![
                job.job_id,
                job.account_username,
                job.slug,
                job.state.as_str(),
                job.accelerator,
                job.lease_token,
                job.lease_expires_at as i64,
                job.attempt,
                if job.is_competition { 1 } else { 0 },
                job.updated_at as i64,
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }
}
