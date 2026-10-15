//! Native Rust HTTP/SSE Engine and Observability Service for MLRescurix (Phase P6).
//!
//! Provides loopback (127.0.0.1) HTTP/SSE service with session token authentication,
//! revision-fenced mutation command ingress, monotonic event streaming, CAS blob streaming,
//! read-only pooled SQLite queries, and embedded single-page dashboard assets.

use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::{params, OptionalExtension, TransactionBehavior};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    artifacts::ArtifactRegistry,
    broker,
    cas::ContentAddressableStorage,
    core::Receipt,
    execution,
    read_pool::{self, ReadPool},
    Result,
};

pub const MAX_REQUEST_BODY_BYTES: usize = 65_536; // 64 KiB ceiling
pub const MAX_HEADER_BYTES: usize = 8_192;
pub const EMBEDDED_DASHBOARD_HTML: &str = include_str!("assets/dashboard.html");

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn valid_id(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 128
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b))
}

#[derive(Debug, Clone)]
pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub query: HashMap<String, String>,
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

pub struct ServerHandle {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

impl ServerHandle {
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn port(&self) -> u16 {
        self.addr.port()
    }

    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        // Ping listener to unblock accept()
        let _ = TcpStream::connect_timeout(&self.addr, Duration::from_millis(50));
        if let Some(h) = self.join_handle.take() {
            let _ = h.join();
        }
    }
}

pub struct DashboardServer {
    state_root: PathBuf,
    token: String,
    listener: TcpListener,
    shutdown: Arc<AtomicBool>,
    read_pool: Arc<ReadPool>,
    artifacts: ArtifactRegistry,
    cas: ContentAddressableStorage,
}

impl DashboardServer {
    pub fn bind(state_root: &Path, port: u16, token: impl Into<String>) -> Result<Self> {
        execution::migrate(state_root)?;
        let canonical_root = std::fs::canonicalize(state_root).map_err(|e| e.to_string())?;
        let addr = format!("127.0.0.1:{port}");
        let listener = TcpListener::bind(&addr).map_err(|e| format!("Failed to bind {addr}: {e}"))?;
        let read_pool = read_pool::shared(&canonical_root)?;
        let artifacts = ArtifactRegistry::new(&canonical_root);
        let cas = ContentAddressableStorage::new(&canonical_root);

        Ok(Self {
            state_root: canonical_root,
            token: token.into(),
            listener,
            shutdown: Arc::new(AtomicBool::new(false)),
            read_pool,
            artifacts,
            cas,
        })
    }

    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.listener.local_addr().map_err(|e| e.to_string())
    }

    pub fn port(&self) -> Result<u16> {
        Ok(self.local_addr()?.port())
    }

    pub fn serve(self) -> Result<()> {
        let handle = self.serve_background()?;
        while !handle.shutdown.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
        }
        Ok(())
    }

    pub fn serve_background(self) -> Result<ServerHandle> {
        let addr = self.listener.local_addr().map_err(|e| e.to_string())?;
        let shutdown = self.shutdown.clone();

        let state_root = self.state_root.clone();
        let token = self.token.clone();
        let listener = self.listener;
        let read_pool = self.read_pool.clone();
        let artifacts = self.artifacts.clone();
        let cas = self.cas.clone();
        let shutdown_loop = shutdown.clone();

        let join_handle = thread::spawn(move || {
            let _ = listener.set_nonblocking(false);
            while !shutdown_loop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _peer)) => {
                        if shutdown_loop.load(Ordering::SeqCst) {
                            break;
                        }
                        let s_root = state_root.clone();
                        let s_tok = token.clone();
                        let s_pool = read_pool.clone();
                        let s_art = artifacts.clone();
                        let s_cas = cas.clone();
                        let s_shutdown = shutdown_loop.clone();
                        let s_port = addr.port();

                        thread::spawn(move || {
                            let _ = handle_connection(
                                stream, s_port, &s_tok, &s_root, &s_pool, &s_art, &s_cas, s_shutdown,
                            );
                        });
                    }
                    Err(_) => {
                        if shutdown_loop.load(Ordering::SeqCst) {
                            break;
                        }
                    }
                }
            }
        });

        Ok(ServerHandle {
            addr,
            shutdown,
            join_handle: Some(join_handle),
        })
    }
}

fn handle_connection(
    mut stream: TcpStream,
    port: u16,
    token: &str,
    state_root: &Path,
    read_pool: &Arc<ReadPool>,
    _artifacts: &ArtifactRegistry,
    cas: &ContentAddressableStorage,
    shutdown: Arc<AtomicBool>,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

    let req = match parse_http_request(&mut stream) {
        Ok(r) => r,
        Err(e) => {
            let code = if e.contains("64 KiB") { 413 } else { 400 };
            send_json_response(&mut stream, code, &json!({"error": e}))?;
            return Ok(());
        }
    };

    // 1. Strict Host header verification
    let host_header = req.headers.get("host").map(String::as_str).unwrap_or("");
    let valid_hosts = [
        format!("127.0.0.1:{port}"),
        format!("localhost:{port}"),
        "127.0.0.1".into(),
        "localhost".into(),
    ];
    if !valid_hosts.iter().any(|vh| host_header.eq_ignore_ascii_case(vh)) {
        send_json_response(&mut stream, 400, &json!({"error": "invalid Host"}))?;
        return Ok(());
    }

    // 2. Route Static Dashboard
    if req.method == "GET" && req.path == "/" {
        let bytes = EMBEDDED_DASHBOARD_HTML.as_bytes();
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            bytes.len()
        );
        stream.write_all(header.as_bytes())?;
        stream.write_all(bytes)?;
        stream.flush()?;
        return Ok(());
    }

    // 3. Route Session Unlock: POST /session
    if req.method == "POST" && req.path == "/session" {
        // Strict Origin check
        if !validate_origin(&req, port) {
            send_json_response(&mut stream, 403, &json!({"error": "Origin rejected"}))?;
            return Ok(());
        }
        let parsed: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(_) => {
                send_json_response(&mut stream, 400, &json!({"error": "invalid bounded JSON body"}))?;
                return Ok(());
            }
        };
        let user_token = parsed.get("token").and_then(Value::as_str).unwrap_or("");
        if !constant_time_eq(user_token.as_bytes(), token.as_bytes()) {
            send_json_response(&mut stream, 401, &json!({"error": "invalid token"}))?;
            return Ok(());
        }

        let body = serde_json::to_vec(&json!({"authenticated": true}))?;
        let resp = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nSet-Cookie: mlr_session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
            body.len()
        );
        stream.write_all(resp.as_bytes())?;
        stream.write_all(&body)?;
        stream.flush()?;
        return Ok(());
    }

    // 4. Strict Authentication on all /v1/* routes
    if !validate_auth(&req, token) {
        send_json_response(&mut stream, 401, &json!({"error": "authentication required"}))?;
        return Ok(());
    }

    // 5. Route Snapshot: GET /v1/projects/:id/snapshot
    if req.method == "GET" && req.path.starts_with("/v1/projects/") && req.path.ends_with("/snapshot") {
        let parts: Vec<&str> = req.path.split('/').collect();
        if parts.len() == 5 && parts[1] == "v1" && parts[2] == "projects" && parts[4] == "snapshot" {
            let project_id = parts[3];
            if !valid_id(project_id) {
                send_json_response(&mut stream, 400, &json!({"error": "invalid project id"}))?;
                return Ok(());
            }
            match build_project_snapshot(state_root, read_pool, project_id) {
                Ok(snap) => send_json_response(&mut stream, 200, &snap)?,
                Err(e) => {
                    if e.contains("not found") {
                        send_json_response(&mut stream, 404, &json!({"error": e}))?;
                    } else if e.contains("read pool") {
                        send_json_response(&mut stream, 503, &json!({"error": "read pool busy, retry"}))?;
                    } else {
                        send_json_response(&mut stream, 500, &json!({"error": format!("snapshot failed: {e}")}))?;
                    }
                }
            }
            return Ok(());
        }
    }

    // 6. Route Persistent SSE: GET /v1/projects/:id/events?after=:cursor
    if req.method == "GET" && req.path.starts_with("/v1/projects/") && req.path.ends_with("/events") {
        let parts: Vec<&str> = req.path.split('/').collect();
        if parts.len() == 5 && parts[1] == "v1" && parts[2] == "projects" && parts[4] == "events" {
            let project_id = parts[3];
            if !valid_id(project_id) {
                send_json_response(&mut stream, 400, &json!({"error": "invalid project id"}))?;
                return Ok(());
            }
            let after_str = req
                .query
                .get("after")
                .or_else(|| req.headers.get("last-event-id"))
                .map(String::as_str)
                .unwrap_or("0");
            let after_cursor = match after_str.parse::<u64>() {
                Ok(c) => c,
                Err(_) => {
                    send_json_response(&mut stream, 400, &json!({"error": "invalid cursor"}))?;
                    return Ok(());
                }
            };

            handle_sse_stream(stream, project_id, after_cursor, read_pool, shutdown)?;
            return Ok(());
        }
    }

    // 7. Route Mutation Commands: POST /v1/projects/:id/commands
    if req.method == "POST" && req.path.starts_with("/v1/projects/") && req.path.ends_with("/commands") {
        let parts: Vec<&str> = req.path.split('/').collect();
        if parts.len() == 5 && parts[1] == "v1" && parts[2] == "projects" && parts[4] == "commands" {
            let project_id = parts[3];
            if !valid_id(project_id) {
                send_json_response(&mut stream, 400, &json!({"error": "invalid project id"}))?;
                return Ok(());
            }
            if !validate_origin(&req, port) {
                send_json_response(&mut stream, 403, &json!({"error": "Origin rejected"}))?;
                return Ok(());
            }
            if req.body.len() > MAX_REQUEST_BODY_BYTES {
                send_json_response(&mut stream, 413, &json!({"error": "request body exceeds 64 KiB limit"}))?;
                return Ok(());
            }

            let payload: Value = match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(_) => {
                    send_json_response(&mut stream, 400, &json!({"error": "invalid bounded JSON body"}))?;
                    return Ok(());
                }
            };

            match handle_command_ingress(state_root, project_id, payload) {
                Ok(receipt) => {
                    let code = if receipt.status == "rejected" { 409 } else { 200 };
                    send_json_response(&mut stream, code, &serde_json::to_value(&receipt).unwrap_or(json!({})))?;
                }
                Err(e) => {
                    send_json_response(&mut stream, 400, &json!({"error": e}))?;
                }
            }
            return Ok(());
        }
    }

    // 8. Route CAS Blob Streaming: GET /v1/projects/:id/artifacts/:artifact_id
    if req.method == "GET" && req.path.starts_with("/v1/projects/") && req.path.contains("/artifacts/") {
        let parts: Vec<&str> = req.path.split('/').collect();
        if parts.len() == 6 && parts[1] == "v1" && parts[2] == "projects" && parts[4] == "artifacts" {
            let project_id = parts[3];
            let artifact_target = parts[5];
            if !valid_id(project_id) || !valid_id(artifact_target) {
                send_json_response(&mut stream, 400, &json!({"error": "invalid identifier"}))?;
                return Ok(());
            }
            handle_artifact_blob_stream(&mut stream, project_id, artifact_target, read_pool, cas)?;
            return Ok(());
        }
    }

    // 9. Route Command Receipt Lookup: GET /v1/commands/:project/:command_id
    if req.method == "GET" && req.path.starts_with("/v1/commands/") {
        let parts: Vec<&str> = req.path.split('/').collect();
        if parts.len() == 5 && parts[1] == "v1" && parts[2] == "commands" {
            let project_id = parts[3];
            let command_id = parts[4];
            if !valid_id(project_id) || !valid_id(command_id) {
                send_json_response(&mut stream, 400, &json!({"error": "invalid identifier"}))?;
                return Ok(());
            }
            let conn = match read_pool.try_read() {
                Ok(c) => c,
                Err(_) => {
                    send_json_response(&mut stream, 503, &json!({"error": "read pool busy"}))?;
                    return Ok(());
                }
            };
            let row: Option<String> = conn
                .query_row(
                    "SELECT receipt FROM commands WHERE project=?1 AND id=?2",
                    params![project_id, command_id],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
            match row {
                Some(r_json) => {
                    let parsed: Value = serde_json::from_str(&r_json).unwrap_or(json!({}));
                    send_json_response(&mut stream, 200, &parsed)?;
                }
                None => {
                    send_json_response(&mut stream, 404, &json!({"status": "unknown"}))?;
                }
            }
            return Ok(());
        }
    }

    send_json_response(&mut stream, 404, &json!({"error": "not found"}))?;
    Ok(())
}

fn parse_http_request(stream: &mut TcpStream) -> std::result::Result<HttpRequest, String> {
    let mut reader = BufReader::new(stream);
    let mut header_bytes = Vec::new();

    // Read header line by line until \r\n\r\n or \n\n
    loop {
        let mut line = Vec::new();
        let bytes_read = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| format!("Header read failed: {e}"))?;
        if bytes_read == 0 {
            return Err("connection closed during request headers".into());
        }
        header_bytes.extend_from_slice(&line);
        if header_bytes.len() > MAX_HEADER_BYTES {
            return Err("request headers exceed 8 KiB bound".into());
        }
        if line == b"\r\n" || line == b"\n" {
            break;
        }
    }

    let header_str = String::from_utf8_lossy(&header_bytes);
    let mut lines = header_str.lines();
    let request_line = lines.next().ok_or("empty request")?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_uppercase();
    let full_path = parts.next().ok_or("missing path")?.to_string();

    let (path, query) = if let Some((p, q)) = full_path.split_once('?') {
        let mut q_map = HashMap::new();
        for pair in q.split('&') {
            if let Some((k, v)) = pair.split_once('=') {
                q_map.insert(k.to_string(), v.to_string());
            } else if !pair.is_empty() {
                q_map.insert(pair.to_string(), String::new());
            }
        }
        (p.to_string(), q_map)
    } else {
        (full_path, HashMap::new())
    };

    let mut headers = HashMap::new();
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            headers.insert(k.trim().to_lowercase(), v.trim().to_string());
        }
    }

    let content_length: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    if content_length > MAX_REQUEST_BODY_BYTES {
        let mut drain_buf = vec![0u8; 8192];
        let mut remaining = content_length.min(1024 * 1024);
        while remaining > 0 {
            let to_read = remaining.min(drain_buf.len());
            match reader.read(&mut drain_buf[..to_read]) {
                Ok(0) | Err(_) => break,
                Ok(n) => remaining = remaining.saturating_sub(n),
            }
        }
        return Err("request body exceeds 64 KiB ceiling".into());
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        reader
            .read_exact(&mut body)
            .map_err(|e| format!("body read failed: {e}"))?;
    }

    Ok(HttpRequest {
        method,
        path,
        query,
        headers,
        body,
    })
}

fn validate_origin(req: &HttpRequest, port: u16) -> bool {
    let origin = req.headers.get("origin").map(String::as_str).unwrap_or("");
    let valid_origins = [
        format!("http://127.0.0.1:{port}"),
        format!("http://localhost:{port}"),
    ];
    valid_origins.iter().any(|vo| origin.eq_ignore_ascii_case(vo))
}

fn validate_auth(req: &HttpRequest, expected_token: &str) -> bool {
    // 1. Authorization: Bearer <token>
    if let Some(auth) = req.headers.get("authorization") {
        if let Some(tok) = auth.strip_prefix("Bearer ") {
            if constant_time_eq(tok.trim().as_bytes(), expected_token.as_bytes()) {
                return true;
            }
        }
    }
    // 2. Cookie: mlr_session=<token>
    if let Some(cookie) = req.headers.get("cookie") {
        for part in cookie.split(';') {
            if let Some((k, v)) = part.trim().split_once('=') {
                if k == "mlr_session" && constant_time_eq(v.trim().as_bytes(), expected_token.as_bytes()) {
                    return true;
                }
            }
        }
    }
    false
}

fn send_json_response(stream: &mut TcpStream, status_code: u16, val: &Value) -> std::io::Result<()> {
    let body = serde_json::to_vec(val)?;
    let status_line = match status_code {
        200 => "200 OK",
        400 => "400 Bad Request",
        401 => "401 Unauthorized",
        403 => "403 Forbidden",
        404 => "404 Not Found",
        409 => "409 Conflict",
        413 => "413 Payload Too Large",
        500 => "500 Internal Server Error",
        503 => "503 Service Unavailable",
        _ => "200 OK",
    };
    let header = format!(
        "HTTP/1.1 {status_line}\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()
}

fn build_project_snapshot(
    state_root: &Path,
    read_pool: &Arc<ReadPool>,
    project_id: &str,
) -> Result<Value> {
    let conn = read_pool.try_read()?;

    // Read project
    let (revision, spec_revision, intent_revision, intent, state, finalization_id, policy_str): (
        u64,
        u64,
        u64,
        String,
        String,
        String,
        String,
    ) = conn
        .query_row(
            "SELECT revision, spec_revision, intent_revision, intent, state, finalization_id, policy FROM projects WHERE id=?1",
            params![project_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)),
        )
        .map_err(|_| format!("project '{project_id}' not found"))?;

    let policy: Value = serde_json::from_str(&policy_str).unwrap_or(json!({}));

    // Read highest cursor
    let event_cursor: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(cursor), 0) FROM durable_events WHERE project=?1",
            params![project_id],
            |r| r.get(0),
        )
        .unwrap_or(0);

    // Read jobs
    let mut jobs_stmt = conn
        .prepare("SELECT id, spec, status, reason, pid, submitted_ms, started_ms, ended_ms, artifacts FROM jobs WHERE project=?1 ORDER BY submitted_ms, id LIMIT 256")
        .map_err(|e| e.to_string())?;
    let jobs = jobs_stmt
        .query_map(params![project_id], |r| {
            let spec_val: Value = serde_json::from_str(&r.get::<_, String>(1)?).unwrap_or(json!({}));
            let arts_val: Value = serde_json::from_str(&r.get::<_, String>(8)?).unwrap_or(json!({}));
            Ok(json!({
                "job_id": r.get::<_, String>(0)?,
                "spec": spec_val,
                "status": r.get::<_, String>(2)?,
                "reason": r.get::<_, Option<String>>(3)?,
                "pid": r.get::<_, Option<u32>>(4)?,
                "submitted_ms": r.get::<_, i64>(5)?,
                "started_ms": r.get::<_, Option<i64>>(6)?,
                "ended_ms": r.get::<_, Option<i64>>(7)?,
                "artifacts": arts_val,
            }))
        })
        .map_err(|e| e.to_string())?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;

    // Read search sessions & DAG if table exists
    let has_search: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='search_sessions')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let search_sessions: Vec<Value> = if has_search {
        let mut s_stmt = conn.prepare("SELECT id, dataset_id, spec, status, created_ms, updated_ms, apex_candidate_id, selection_record FROM search_sessions WHERE project=?1 ORDER BY created_ms DESC LIMIT 16").map_err(|e| e.to_string())?;
        s_stmt.query_map(params![project_id], |r| {
            let spec_val: Value = serde_json::from_str(&r.get::<_, String>(2)?).unwrap_or(json!({}));
            let sel_rec: Option<Value> = r.get::<_, Option<String>>(7)?.and_then(|s| serde_json::from_str(&s).ok());
            Ok(json!({
                "session_id": r.get::<_, String>(0)?,
                "dataset_id": r.get::<_, String>(1)?,
                "spec": spec_val,
                "status": r.get::<_, String>(3)?,
                "created_ms": r.get::<_, i64>(4)?,
                "updated_ms": r.get::<_, i64>(5)?,
                "apex_candidate_id": r.get::<_, Option<String>>(6)?,
                "selection_record": sel_rec,
            }))
        }).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    let has_dag: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='experiment_dag')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let experiment_dag: Vec<Value> = if has_dag {
        let mut dag_stmt = conn.prepare(
            "SELECT d.session_id, d.id, d.recipe, d.parameters, d.seed, d.fidelity, d.node_type, d.parent_id,
                    COALESCE(r.status, 'pending'), r.primary_metric, r.is_feasible, r.constraints, r.train_ms,
                    r.model_sha256, r.predictions_sha256, r.metrics
             FROM experiment_dag d
             LEFT JOIN trial_results r ON r.project=d.project AND r.experiment_id=d.id
             WHERE d.project=?1 ORDER BY d.created_ms LIMIT 512"
        ).map_err(|e| e.to_string())?;

        dag_stmt.query_map(params![project_id], |r| {
            let params_val: Value = serde_json::from_str(&r.get::<_, String>(3)?).unwrap_or(json!({}));
            let is_f: Option<i64> = r.get(10)?;
            let constraints_val: Value = r.get::<_, Option<String>>(11)?.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
            let metrics_val: Value = r.get::<_, Option<String>>(15)?.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or(json!({}));
            Ok(json!({
                "session_id": r.get::<_, String>(0)?,
                "experiment_id": r.get::<_, String>(1)?,
                "recipe": r.get::<_, String>(2)?,
                "parameters": params_val,
                "seed": r.get::<_, u64>(4)?,
                "fidelity": r.get::<_, f64>(5)?,
                "node_type": r.get::<_, String>(6)?,
                "parent_id": r.get::<_, Option<String>>(7)?,
                "status": r.get::<_, String>(8)?,
                "primary_metric": r.get::<_, Option<f64>>(9)?,
                "is_feasible": is_f.map(|v| v == 1),
                "constraints": constraints_val,
                "train_ms": r.get::<_, Option<f64>>(12)?,
                "model_sha256": r.get::<_, Option<String>>(13)?,
                "predictions_sha256": r.get::<_, Option<String>>(14)?,
                "metrics": metrics_val,
            }))
        }).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    // Read artifacts
    let has_artifacts: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifacts')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let artifacts: Vec<Value> = if has_artifacts {
        let mut art_stmt = conn.prepare(
            "SELECT a.artifact_id, a.kind, a.producer_authority, a.current_version_id, a.pinned, a.visibility, a.created_ms, a.updated_ms,
                    v.blob_sha256, v.index_digest, v.summary_text, v.schema_version, b.bytes, b.media_type
             FROM artifacts a
             LEFT JOIN artifact_versions v ON v.project=a.project AND v.version_id=a.current_version_id
             LEFT JOIN blobs b ON b.sha256=v.blob_sha256
             WHERE a.project=?1 AND a.tombstoned_ms IS NULL
             ORDER BY a.created_ms DESC LIMIT 256"
        ).map_err(|e| e.to_string())?;

        art_stmt.query_map(params![project_id], |r| {
            Ok(json!({
                "artifact_id": r.get::<_, String>(0)?,
                "kind": r.get::<_, String>(1)?,
                "producer_authority": r.get::<_, String>(2)?,
                "current_version_id": r.get::<_, Option<String>>(3)?,
                "pinned": r.get::<_, i64>(4)? == 1,
                "visibility": r.get::<_, String>(5)?,
                "created_ms": r.get::<_, i64>(6)?,
                "updated_ms": r.get::<_, i64>(7)?,
                "blob_sha256": r.get::<_, Option<String>>(8)?,
                "index_digest": r.get::<_, Option<String>>(9)?,
                "summary_text": r.get::<_, Option<String>>(10)?,
                "schema_version": r.get::<_, Option<i64>>(11)?,
                "bytes": r.get::<_, Option<i64>>(12)?,
                "media_type": r.get::<_, Option<String>>(13)?,
            }))
        }).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    // Read steering directives
    let has_steering: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='steering_directives')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let steering_directives: Vec<Value> = if has_steering {
        let mut steer_stmt = conn.prepare("SELECT directive_id, task_id, actor, epoch, directive, created_ms, applied FROM steering_directives WHERE project=?1 ORDER BY epoch DESC, created_ms DESC LIMIT 64").map_err(|e| e.to_string())?;
        steer_stmt.query_map(params![project_id], |r| {
            Ok(json!({
                "directive_id": r.get::<_, String>(0)?,
                "task_id": r.get::<_, String>(1)?,
                "actor": r.get::<_, String>(2)?,
                "epoch": r.get::<_, i64>(3)?,
                "directive": r.get::<_, String>(4)?,
                "created_ms": r.get::<_, i64>(5)?,
                "applied": r.get::<_, i64>(6)? == 1,
            }))
        }).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    let handoff_version: u64 = if has_steering {
        conn.query_row(
            "SELECT COALESCE(MAX(epoch), 0) FROM steering_directives WHERE project=?1",
            params![project_id],
            |r| r.get::<_, i64>(0),
        )
        .unwrap_or(0) as u64
    } else {
        0
    };

    // Read user approvals
    let has_approvals: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_approvals')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let user_approvals: Vec<Value> = if has_approvals {
        let mut appr_stmt = conn.prepare("SELECT approval_id, action_type, target_id, actor, approved, created_ms, expires_ms FROM user_approvals WHERE project=?1 ORDER BY created_ms DESC LIMIT 64").map_err(|e| e.to_string())?;
        appr_stmt.query_map(params![project_id], |r| {
            Ok(json!({
                "approval_id": r.get::<_, String>(0)?,
                "action_type": r.get::<_, String>(1)?,
                "target_id": r.get::<_, String>(2)?,
                "actor": r.get::<_, String>(3)?,
                "approved": r.get::<_, i64>(4)? == 1,
                "created_ms": r.get::<_, i64>(5)?,
                "expires_ms": r.get::<_, i64>(6)?,
            }))
        }).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e| e.to_string())?
    } else {
        Vec::new()
    };

    // Read tasks & agent tasks
    let mut tasks_stmt = conn
        .prepare("SELECT id, data FROM tasks WHERE project=?1 ORDER BY id LIMIT 64")
        .map_err(|e| e.to_string())?;
    let tasks = tasks_stmt
        .query_map(params![project_id], |r| {
            let data: Value = serde_json::from_str(&r.get::<_, String>(1)?).unwrap_or(json!({}));
            Ok(json!({
                "id": r.get::<_, String>(0)?,
                "data": data,
            }))
        })
        .map_err(|e| e.to_string())?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;

    // Read resource broker telemetry
    let capabilities = broker::discover_effective_capabilities(state_root).ok();
    let available = broker::get_available_capacity(state_root).ok();
    let is_under_pressure = broker::is_under_pressure(state_root).unwrap_or(false);
    let active_leases = broker::list_active_leases(&conn, Some(project_id)).unwrap_or_default();
    let allocation = broker::get_active_capacity_allocation(&conn, None).ok();

    // Read Apex selection
    let selection: Option<Value> = if conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='project_selections')",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false)
    {
        conn.query_row(
            "SELECT selection_id, candidate_version_id, scorecard_version_id, protocol_sha256, rationale, created_ms FROM project_selections WHERE project=?1 AND is_apex=1 ORDER BY created_ms DESC LIMIT 1",
            params![project_id],
            |r| Ok(json!({
                "selection_id": r.get::<_, String>(0)?,
                "candidate_version_id": r.get::<_, String>(1)?,
                "scorecard_version_id": r.get::<_, String>(2)?,
                "protocol_sha256": r.get::<_, String>(3)?,
                "rationale": r.get::<_, String>(4)?,
                "created_ms": r.get::<_, i64>(5)?,
            }))
        ).optional().unwrap_or(None)
    } else {
        None
    };

    Ok(json!({
        "project_id": project_id,
        "revision": revision,
        "spec_revision": spec_revision,
        "intent_revision": intent_revision,
        "intent": intent,
        "state": state,
        "finalization_id": finalization_id,
        "event_cursor": event_cursor.to_string(),
        "handoff_version": handoff_version,
        "policy": policy,
        "tasks": tasks,
        "jobs": jobs,
        "search_sessions": search_sessions,
        "experiment_dag": experiment_dag,
        "artifacts": artifacts,
        "steering_directives": steering_directives,
        "user_approvals": user_approvals,
        "selection": selection,
        "resource_status": {
            "capabilities": capabilities,
            "available": available,
            "is_under_pressure": is_under_pressure,
            "active_leases": active_leases,
            "allocation": allocation,
        }
    }))
}

fn handle_sse_stream(
    mut stream: TcpStream,
    project_id: &str,
    mut after_cursor: u64,
    read_pool: &Arc<ReadPool>,
    shutdown: Arc<AtomicBool>,
) -> std::io::Result<()> {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
    stream.write_all(headers.as_bytes())?;
    stream.flush()?;

    let mut keepalive_ticks = 0;

    while !shutdown.load(Ordering::SeqCst) {
        let (reset_required, events) = {
            let conn = match read_pool.try_read() {
                Ok(c) => c,
                Err(_) => {
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }
            };

            let expired_through: i64 = conn
                .query_row(
                    "SELECT COALESCE(expired_through_cursor, 0) FROM durable_event_checkpoints WHERE project=?1",
                    params![project_id],
                    |r| r.get(0),
                )
                .unwrap_or(0);

            if after_cursor < expired_through as u64 {
                (true, Vec::new())
            } else {
                let mut stmt = match conn.prepare(
                    "SELECT cursor, event_type, version, payload_json, committed_ms FROM durable_events WHERE project=?1 AND cursor>?2 ORDER BY cursor ASC LIMIT 64"
                ) {
                    Ok(s) => s,
                    Err(_) => {
                        thread::sleep(Duration::from_millis(200));
                        continue;
                    }
                };

                let evs = stmt
                    .query_map(params![project_id, after_cursor as i64], |r| {
                        Ok((
                            r.get::<_, i64>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, i64>(2)?,
                            r.get::<_, String>(3)?,
                            r.get::<_, i64>(4)?,
                        ))
                    })
                    .and_then(|mapped| mapped.collect::<rusqlite::Result<Vec<_>>>())
                    .unwrap_or_default();

                (false, evs)
            }
        };

        if reset_required {
            let frame = "event: reset\ndata: {\"reset_required\":true}\n\n";
            if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
                break;
            }
            after_cursor = 0;
            thread::sleep(Duration::from_millis(500));
            continue;
        }

        if !events.is_empty() {
            for (cursor, ev_type, version, payload, committed_ms) in events {
                after_cursor = after_cursor.max(cursor as u64);
                let json_data = json!({
                    "cursor": cursor,
                    "project": project_id,
                    "event_type": ev_type,
                    "version": version,
                    "payload": serde_json::from_str::<Value>(&payload).unwrap_or(json!({})),
                    "committed_ms": committed_ms,
                });
                let frame = format!(
                    "id: {cursor}\nevent: {ev_type}\ndata: {}\n\n",
                    serde_json::to_string(&json_data).unwrap_or_default()
                );
                if stream.write_all(frame.as_bytes()).is_err() || stream.flush().is_err() {
                    return Ok(());
                }
            }
            keepalive_ticks = 0;
        } else {
            keepalive_ticks += 1;
            if keepalive_ticks >= 4 {
                if stream.write_all(b": keepalive\n\n").is_err() || stream.flush().is_err() {
                    return Ok(());
                }
                keepalive_ticks = 0;
            }
        }

        thread::sleep(Duration::from_millis(250));
    }

    Ok(())
}

fn handle_command_ingress(state_root: &Path, project_id: &str, payload: Value) -> Result<Receipt> {
    let command_id = payload
        .get("command_id")
        .and_then(Value::as_str)
        .ok_or("missing command_id")?;
    let proj = payload
        .get("project_id")
        .and_then(Value::as_str)
        .ok_or("missing project_id")?;
    if proj != project_id {
        return Err("project_id in path does not match payload".into());
    }
    if !valid_id(command_id) || !valid_id(proj) {
        return Err("invalid command or project identity".into());
    }

    let expected_rev = payload.get("expected_revision").and_then(Value::as_u64);
    let operation = payload.get("operation").ok_or("missing operation")?;

    let op_type = operation
        .get("type")
        .and_then(Value::as_str)
        .ok_or("missing operation.type")?;

    let mut conn = execution::connect(state_root)?;
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .map_err(|e| e.to_string())?;

    // Idempotency check in commands table
    let old: Option<(String, String)> = tx
        .query_row(
            "SELECT request, receipt FROM commands WHERE id=?1 AND project=?2",
            params![command_id, project_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(|e| e.to_string())?;

    let encoded_req = serde_json::to_string(&payload).map_err(|e| e.to_string())?;

    if let Some((old_req, old_receipt)) = old {
        let req_hash = format!("hash:{:x}", Sha256::digest(encoded_req.as_bytes()));
        if old_req != encoded_req && old_req != req_hash {
            return Err("command identity reused with different payload/scope".into());
        }
        return serde_json::from_str(&old_receipt).map_err(|e| e.to_string());
    }

    // Revision check
    let current_rev: u64 = tx
        .query_row(
            "SELECT revision FROM projects WHERE id=?1",
            params![project_id],
            |r| r.get(0),
        )
        .map_err(|_| format!("project '{project_id}' not found"))?;

    if let Some(exp) = expected_rev {
        if exp != current_rev {
            let receipt = Receipt {
                command_id: command_id.into(),
                project_id: project_id.into(),
                status: "rejected".into(),
                revision: current_rev,
                result: json!({"reason": "revision conflict"}),
            };
            let receipt_json = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
            tx.execute(
                "INSERT INTO commands VALUES(?1, ?2, ?3, ?4)",
                params![command_id, project_id, encoded_req, receipt_json],
            )
            .map_err(|e| e.to_string())?;
            tx.commit().map_err(|e| e.to_string())?;
            return Ok(receipt);
        }
    }

    // Execute supported operations
    let result = match op_type {
        "lifecycle" => {
            let state = operation
                .get("state")
                .and_then(Value::as_str)
                .ok_or("missing lifecycle state")?;
            if !["created", "paused", "cancelled"].contains(&state) {
                return Err("invalid lifecycle state".into());
            }
            if state == "cancelled" {
                tx.execute(
                    "UPDATE projects SET state=?2, intent_revision=intent_revision+1 WHERE id=?1",
                    params![project_id, state],
                )
                .map_err(|e| e.to_string())?;
            } else {
                tx.execute(
                    "UPDATE projects SET state=?2 WHERE id=?1",
                    params![project_id, state],
                )
                .map_err(|e| e.to_string())?;
            }
            json!({"state": state})
        }
        "steer" => {
            let intent = operation
                .get("intent")
                .and_then(Value::as_str)
                .ok_or("missing steer intent")?;
            tx.execute(
                "UPDATE projects SET intent=?2, intent_revision=intent_revision+1 WHERE id=?1",
                params![project_id, intent],
            )
            .map_err(|e| e.to_string())?;
            json!({"intent": intent})
        }
        "apply_steering" => {
            let directive = operation
                .get("directive")
                .and_then(Value::as_str)
                .ok_or("missing directive")?;
            let actor = operation
                .get("actor")
                .and_then(Value::as_str)
                .unwrap_or("human:operator");
            let task_id = operation
                .get("task_id")
                .and_then(Value::as_str)
                .unwrap_or("lead");

            let epoch: i64 = tx
                .query_row(
                    "SELECT COALESCE(MAX(epoch), 0) + 1 FROM steering_directives WHERE project=?1",
                    params![project_id],
                    |r| r.get(0),
                )
                .unwrap_or(1);

            let directive_id = format!(
                "steer-{}",
                &format!("{:x}", Sha256::digest(format!("{task_id}:{epoch}:{actor}:{directive}").as_bytes()))[..16]
            );

            tx.execute(
                "INSERT INTO steering_directives(project, directive_id, task_id, actor, epoch, directive, created_ms, applied)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)
                 ON CONFLICT(project, directive_id) DO UPDATE SET directive=excluded.directive",
                params![project_id, directive_id, task_id, actor, epoch, directive, now_ms()],
            )
            .map_err(|e| e.to_string())?;

            tx.execute(
                "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
                 VALUES(?1, 'steering.applied', 1, json_object('directive_id', ?2, 'actor', ?3, 'epoch', ?4), ?5)",
                params![project_id, directive_id, actor, epoch, now_ms()],
            )
            .map_err(|e| e.to_string())?;

            json!({"directive_id": directive_id, "epoch": epoch})
        }
        "user_approval" => {
            let approval_id = operation
                .get("approval_id")
                .and_then(Value::as_str)
                .ok_or("missing approval_id")?;
            let action_type = operation
                .get("action_type")
                .and_then(Value::as_str)
                .ok_or("missing action_type")?;
            let target_id = operation
                .get("target_id")
                .and_then(Value::as_str)
                .ok_or("missing target_id")?;
            let actor = operation
                .get("actor")
                .and_then(Value::as_str)
                .unwrap_or("human:operator");
            let approved = operation
                .get("approved")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let ttl_ms = operation
                .get("ttl_ms")
                .and_then(Value::as_i64)
                .unwrap_or(3600_000);

            if !matches!(action_type, "holdout_access" | "deployment_handoff" | "final_test") {
                return Err(format!("invalid approval action_type '{action_type}'"));
            }

            let now = now_ms();
            let expires_ms = now.saturating_add(ttl_ms);

            tx.execute(
                "INSERT INTO user_approvals(project, approval_id, action_type, target_id, actor, approved, created_ms, expires_ms)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                 ON CONFLICT(project, approval_id) DO UPDATE SET approved=excluded.approved, expires_ms=excluded.expires_ms, actor=excluded.actor",
                params![project_id, approval_id, action_type, target_id, actor, if approved { 1 } else { 0 }, now, expires_ms],
            ).map_err(|e| e.to_string())?;

            tx.execute(
                "INSERT INTO durable_events(project, event_type, version, payload_json, committed_ms)
                 VALUES(?1, 'user.approved', 1, json_object('approval_id', ?2, 'action_type', ?3, 'target_id', ?4), ?5)",
                params![project_id, approval_id, action_type, target_id, now],
            ).map_err(|e| e.to_string())?;

            json!({"approval_id": approval_id, "approved": approved})
        }
        "pin_artifact" => {
            let artifact_id = operation
                .get("artifact_id")
                .and_then(Value::as_str)
                .ok_or("missing artifact_id")?;
            let pinned = operation
                .get("pinned")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            let now = now_ms();

            tx.execute(
                "UPDATE artifacts SET pinned=?2, updated_ms=?3 WHERE project=?1 AND artifact_id=?4",
                params![project_id, if pinned { 1 } else { 0 }, now, artifact_id],
            )
            .map_err(|e| e.to_string())?;

            json!({"artifact_id": artifact_id, "pinned": pinned})
        }
        _ => {
            return Err(format!("unsupported operation type: {op_type}"));
        }
    };

    // Advance project revision monotonically
    tx.execute(
        "UPDATE projects SET revision=revision+1 WHERE id=?1",
        params![project_id],
    )
    .map_err(|e| e.to_string())?;

    let new_rev: u64 = tx
        .query_row(
            "SELECT revision FROM projects WHERE id=?1",
            params![project_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;

    let receipt = Receipt {
        command_id: command_id.into(),
        project_id: project_id.into(),
        status: "applied".into(),
        revision: new_rev,
        result,
    };

    let receipt_json = serde_json::to_string(&receipt).map_err(|e| e.to_string())?;
    tx.execute(
        "INSERT INTO commands VALUES(?1, ?2, ?3, ?4)",
        params![command_id, project_id, encoded_req, receipt_json],
    )
    .map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;

    Ok(receipt)
}

fn handle_artifact_blob_stream(
    stream: &mut TcpStream,
    project_id: &str,
    artifact_target: &str,
    read_pool: &Arc<ReadPool>,
    cas: &ContentAddressableStorage,
) -> std::io::Result<()> {
    let conn = match read_pool.try_read() {
        Ok(c) => c,
        Err(_) => {
            send_json_response(stream, 503, &json!({"error": "read pool busy"}))?;
            return Ok(());
        }
    };

    // Lookup blob sha256 by artifact_id or version_id or direct blob sha
    let blob_info: Option<(String, String)> = conn
        .query_row(
            "SELECT v.blob_sha256, b.media_type
             FROM artifacts a
             JOIN artifact_versions v ON v.project=a.project AND v.version_id=a.current_version_id
             JOIN blobs b ON b.sha256=v.blob_sha256
             WHERE a.project=?1 AND a.artifact_id=?2 AND a.tombstoned_ms IS NULL",
            params![project_id, artifact_target],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .unwrap_or(None)
        .or_else(|| {
            conn.query_row(
                "SELECT v.blob_sha256, b.media_type
                 FROM artifact_versions v
                 JOIN blobs b ON b.sha256=v.blob_sha256
                 WHERE v.project=?1 AND v.version_id=?2",
                params![project_id, artifact_target],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()
            .unwrap_or(None)
        });

    let (sha256, media_type) = match blob_info {
        Some(info) => info,
        None => {
            send_json_response(stream, 404, &json!({"error": "artifact blob not found"}))?;
            return Ok(());
        }
    };

    let bytes = match cas.read_blob(&sha256) {
        Ok(b) => b,
        Err(e) => {
            send_json_response(stream, 500, &json!({"error": format!("blob read error: {e}")}))?;
            return Ok(());
        }
    };

    let header = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: {media_type}\r\nContent-Length: {}\r\nETag: \"{sha256}\"\r\nCache-Control: public, max-age=31536000, immutable\r\nConnection: close\r\n\r\n",
        bytes.len()
    );
    stream.write_all(header.as_bytes())?;
    stream.write_all(&bytes)?;
    stream.flush()
}
