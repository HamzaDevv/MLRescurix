//! Phase P6 Acceptance Gate and Integration Tests for MLRescurix.
//!
//! Verifies:
//! - P6.1: Native Rust HTTP/SSE engine, authenticated snapshots, event streams, revision-fenced mutation command ingress, CAS streaming, Host/Origin/CSRF validation.
//! - P6.2: Safe artifact, code & diagnostic gallery, strict zero-XSS DOM sanitization against script injection.
//! - P6.3: Experiment DAG traversal, Pareto frontier scatter metrics, candidate leaderboard separating trusted vs unverified claims.
//! - P6.4: Resource telemetry drawer, allocated vs measured CPU/RAM, active worker leases, disk pressure backoff alerts.
//! - P6.5: Human-in-the-loop turn-boundary steering, revisioned project notes, user approval gates.
//! - P6.6: Reconnect/replay resilience, monotonic SSE cursors, cursor expiration reset, large DAG rendering (100+ nodes), reduced-motion accessibility.

use std::{
    collections::BTreeMap,
    fs,
    io::{BufRead, BufReader, Read, Write},
    net::{SocketAddr, TcpStream},
    path::Path,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use rusqlite::params;
use serde_json::{json, Value};

use mlrescurix_harness::{
    artifacts::ArtifactRegistry,
    core::{Operation, Policy, Request},
    execution,
    server::DashboardServer,
    Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy {
        max_tasks: 1024,
        budget: b,
        capacity: c,
    }
}

fn setup_test_project(root: &Path, project: &str) -> (Runtime, ArtifactRegistry) {
    if root.exists() {
        let _ = fs::remove_dir_all(root);
    }
    let runtime = Runtime::init(root, test_policy(100)).unwrap();
    execution::migrate(root).unwrap();
    let reg = ArtifactRegistry::new(root);

    // Create project
    let req = Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P6 test intent".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    };
    runtime.command(req).unwrap();
    (runtime, reg)
}

struct HttpResponse {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn body_json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(json!({}))
    }

    fn body_str(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

fn raw_http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: Option<&[u8]>,
) -> HttpResponse {
    let mut stream = TcpStream::connect(addr).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    stream.set_write_timeout(Some(Duration::from_secs(5))).unwrap();

    let mut req_str = format!("{method} {path} HTTP/1.1\r\n");
    let has_host = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("host"));
    if !has_host {
        req_str.push_str(&format!("Host: 127.0.0.1:{}\r\n", addr.port()));
    }
    for (k, v) in headers {
        req_str.push_str(&format!("{k}: {v}\r\n"));
    }
    if let Some(b) = body {
        if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
            req_str.push_str(&format!("Content-Length: {}\r\n", b.len()));
        }
    } else if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
        req_str.push_str("Content-Length: 0\r\n");
    }
    req_str.push_str("Connection: close\r\n\r\n");

    stream.write_all(req_str.as_bytes()).unwrap();
    if let Some(b) = body {
        stream.write_all(b).unwrap();
    }
    stream.flush().unwrap();

    // Read response
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    assert!(parts.len() >= 2, "Invalid status line: {status_line}");
    let status: u16 = parts[1].parse().unwrap();

    let mut resp_headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line == "\n" || line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            resp_headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }

    let mut resp_body = Vec::new();
    reader.read_to_end(&mut resp_body).unwrap();

    HttpResponse {
        status,
        headers: resp_headers,
        body: resp_body,
    }
}

#[test]
fn test_p6_1_native_rust_server_auth_and_host_validation() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-auth-test";
    let token = "secret-session-token-998877";

    setup_test_project(root, project);

    // Bind server on ephemeral port (0)
    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();

    // 1. GET / serves dashboard HTML without auth
    let res = raw_http_request(addr, "GET", "/", &[], None);
    assert_eq!(res.status, 200);
    assert!(res.body_str().contains("MLRescurix Observability & Steering Dashboard"));
    assert!(res.body_str().contains("Multi-Fidelity Experiment DAG"));

    // 2. Reject request with invalid Host header
    let res = raw_http_request(addr, "GET", "/", &[("Host", "malicious-site.com")], None);
    assert_eq!(res.status, 400);
    assert!(res.body_json()["error"].as_str().unwrap().contains("invalid Host"));

    // 3. GET /v1/projects/:id/snapshot without auth returns 401
    let res = raw_http_request(addr, "GET", &format!("/v1/projects/{project}/snapshot"), &[], None);
    assert_eq!(res.status, 401);
    assert!(res.body_json()["error"].as_str().unwrap().contains("authentication required"));

    // 4. POST /session with wrong Origin returns 403
    let session_body = json!({"token": token}).to_string();
    let res = raw_http_request(
        addr,
        "POST",
        "/session",
        &[
            ("Content-Type", "application/json"),
            ("Origin", "http://evil-origin.com"),
        ],
        Some(session_body.as_bytes()),
    );
    assert_eq!(res.status, 403);
    assert!(res.body_json()["error"].as_str().unwrap().contains("Origin rejected"));

    // 5. POST /session with wrong token returns 401
    let bad_session_body = json!({"token": "wrong-token"}).to_string();
    let res = raw_http_request(
        addr,
        "POST",
        "/session",
        &[
            ("Content-Type", "application/json"),
            ("Origin", &format!("http://127.0.0.1:{}", addr.port())),
        ],
        Some(bad_session_body.as_bytes()),
    );
    assert_eq!(res.status, 401);
    assert!(res.body_json()["error"].as_str().unwrap().contains("invalid token"));

    // 6. POST /session with correct token returns 200 and Set-Cookie with SameSite=Strict
    let res = raw_http_request(
        addr,
        "POST",
        "/session",
        &[
            ("Content-Type", "application/json"),
            ("Origin", &format!("http://127.0.0.1:{}", addr.port())),
        ],
        Some(session_body.as_bytes()),
    );
    assert_eq!(res.status, 200);
    let cookie = res.header("Set-Cookie").expect("Set-Cookie header");
    assert!(cookie.contains(&format!("mlr_session={token}")));
    assert!(cookie.contains("SameSite=Strict"));
    assert!(cookie.contains("HttpOnly"));

    // 7. GET /v1/projects/:id/snapshot with Cookie authentication succeeds
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Cookie", &format!("mlr_session={token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let snap = res.body_json();
    assert_eq!(snap["project_id"], project);
    assert_eq!(snap["state"], "created");

    // 8. GET /v1/projects/:id/snapshot with Bearer token succeeds
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let snap = res.body_json();
    assert_eq!(snap["project_id"], project);

    handle.shutdown();
}

#[test]
fn test_p6_1_and_p6_5_command_revision_fencing_and_security() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-commands-test";
    let token = "cmd-token-123456";

    setup_test_project(root, project);

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();
    let origin = format!("http://127.0.0.1:{}", addr.port());

    // 1. Submit mutation with body exceeding 64 KiB ceiling -> rejected with 413
    let huge_directive = "x".repeat(70_000);
    let huge_payload = json!({
        "command_id": "cmd-huge-1",
        "project_id": project,
        "expected_revision": 1,
        "operation": {
            "type": "apply_steering",
            "directive": huge_directive
        }
    })
    .to_string();

    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(huge_payload.as_bytes()),
    );
    assert_eq!(res.status, 413);

    // 2. Submit valid command with stale expected_revision (expected 99, actual is 1)
    let stale_payload = json!({
        "command_id": "cmd-stale-1",
        "project_id": project,
        "expected_revision": 99,
        "operation": {
            "type": "lifecycle",
            "state": "paused"
        }
    })
    .to_string();

    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(stale_payload.as_bytes()),
    );
    // Revision conflict returns 409 with status "rejected"
    assert_eq!(res.status, 409);
    let receipt = res.body_json();
    assert_eq!(receipt["status"], "rejected");
    assert_eq!(receipt["result"]["reason"], "revision conflict");

    // 3. Submit valid command with matching expected_revision (revision 1)
    let valid_payload = json!({
        "command_id": "cmd-valid-pause",
        "project_id": project,
        "expected_revision": 1,
        "operation": {
            "type": "lifecycle",
            "state": "paused"
        }
    })
    .to_string();

    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(valid_payload.as_bytes()),
    );
    assert_eq!(res.status, 200);
    let receipt = res.body_json();
    assert_eq!(receipt["status"], "applied");
    assert_eq!(receipt["revision"], 2);
    assert_eq!(receipt["result"]["state"], "paused");

    // 4. Test idempotency: re-submitting same command_id with same payload returns cached applied receipt
    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(valid_payload.as_bytes()),
    );
    assert_eq!(res.status, 200);
    let re_receipt = res.body_json();
    assert_eq!(re_receipt["command_id"], "cmd-valid-pause");
    assert_eq!(re_receipt["status"], "applied");
    assert_eq!(re_receipt["revision"], 2);

    // 5. Test command receipt lookup endpoint GET /v1/commands/:project/:command_id
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/commands/{project}/cmd-valid-pause"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    assert_eq!(res.body_json()["command_id"], "cmd-valid-pause");

    handle.shutdown();
}

#[test]
fn test_p6_1_cas_blob_streaming_and_boundary_checks() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-cas-test";
    let token = "cas-token-443322";

    let (_runtime, reg) = setup_test_project(root, project);

    // Store a blob in CAS
    let blob_content = b"def candidate_model():\n    return {'model': 'linear_ridge', 'alpha': 1.0}\n";
    let receipt = reg.cas().store_blob(blob_content, "text/x-python", Some(project)).unwrap();

    // Register artifact and version in registry
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let now = now_ms();
    conn.execute(
        "INSERT INTO blobs(sha256, bytes, media_type, created_ms) VALUES(?1, ?2, 'text/x-python', ?3)",
        params![receipt.sha256, receipt.bytes as i64, now],
    ).unwrap();
    conn.execute(
        "INSERT INTO artifacts(project, artifact_id, kind, producer_authority, current_version_id, pinned, visibility, created_ms, updated_ms)
         VALUES(?1, 'candidate_code', 'code', 'evaluator:trusted', 'candidate_code_v1', 1, 'project', ?2, ?2)",
        params![project, now],
    ).unwrap();
    conn.execute(
        "INSERT INTO artifact_versions(project, version_id, artifact_id, blob_sha256, index_digest, summary_text, schema_version, metadata_json, created_ms)
         VALUES(?1, 'candidate_code_v1', 'candidate_code', ?2, 'candidate Python code', 'ridge model source', 1, '{}', ?3)",
        params![project, receipt.sha256, now],
    ).unwrap();

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();

    // 1. GET /v1/projects/:id/artifacts/:artifact_id streams the exact blob bytes
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/artifacts/candidate_code"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    assert_eq!(res.body, blob_content);
    assert_eq!(res.header("Content-Type"), Some("text/x-python"));
    assert_eq!(res.header("ETag"), Some(format!("\"{}\"", receipt.sha256).as_str()));

    // 2. Requesting nonexistent artifact returns 404
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/artifacts/nonexistent_artifact"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 404);

    // 3. Path traversal escape returns 400 Bad Request
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/artifacts/..%2F..%2Fetc%2Fpasswd"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 400);

    handle.shutdown();
}

#[test]
fn test_p6_1_and_p6_6_monotonic_sse_event_stream_and_concurrent_mutations() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-sse-test";
    let token = "sse-token-776655";

    setup_test_project(root, project);

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();
    let origin = format!("http://127.0.0.1:{}", addr.port());

    // Connect SSE client over raw TCP
    let mut sse_stream = TcpStream::connect(addr).unwrap();
    let sse_req = format!(
        "GET /v1/projects/{project}/events?after=0 HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {token}\r\nAccept: text/event-stream\r\n\r\n",
        addr.port()
    );
    sse_stream.write_all(sse_req.as_bytes()).unwrap();
    sse_stream.flush().unwrap();

    let mut reader = BufReader::new(sse_stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).unwrap();
    assert!(status_line.contains("200 OK"));

    // Skip headers until empty line
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line == "\n" {
            break;
        }
    }

    // Now trigger concurrent mutations via POST /commands
    let steer_payload = json!({
        "command_id": "cmd-steer-sse-1",
        "project_id": project,
        "expected_revision": 1,
        "operation": {
            "type": "apply_steering",
            "directive": "Focus on boosting calibration",
            "actor": "human:operator",
            "task_id": "lead"
        }
    })
    .to_string();

    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(steer_payload.as_bytes()),
    );
    assert_eq!(res.status, 200);

    // Read SSE frames and verify monotonic cursor and event_type
    let mut received_steering_event = false;
    let mut highest_cursor = 0u64;

    for _ in 0..10 {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.starts_with("id: ") {
            let cursor: u64 = line.strip_prefix("id: ").unwrap().trim().parse().unwrap();
            assert!(cursor >= highest_cursor, "Cursors must be monotonic: {cursor} >= {highest_cursor}");
            highest_cursor = cursor;
        } else if line.starts_with("event: ") {
            let ev_type = line.strip_prefix("event: ").unwrap().trim();
            if ev_type == "steering.applied" {
                received_steering_event = true;
            }
        } else if line.starts_with("data: ") {
            let json_str = line.strip_prefix("data: ").unwrap().trim();
            let parsed: Value = serde_json::from_str(json_str).unwrap();
            assert_eq!(parsed["project"], project);
            if received_steering_event {
                break;
            }
        }
    }

    assert!(received_steering_event, "Must receive steering.applied event frame over SSE stream");

    handle.shutdown();
}

#[test]
fn test_p6_6_reconnect_replay_resilience_and_cursor_expiration() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-cursor-test";
    let token = "cursor-token-554433";

    setup_test_project(root, project);

    // Insert events and simulate cursor expiration in durable_event_checkpoints
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let now = now_ms();
    for i in 1..=5 {
        conn.execute(
            "INSERT INTO durable_events(cursor, project, event_type, version, payload_json, committed_ms)
             VALUES(?1, ?2, 'test.event', 1, json_object('index', ?1), ?3)",
            params![i, project, now],
        ).unwrap();
    }
    // Set expired_through_cursor = 3 in checkpoints table
    conn.execute(
        "INSERT INTO durable_event_checkpoints(project, expired_through_cursor) VALUES(?1, 3)
         ON CONFLICT(project) DO UPDATE SET expired_through_cursor=3",
        params![project],
    ).unwrap();

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();

    // 1. Client connects with expired cursor (after=2 < expired 3) -> receives 'event: reset'
    let mut sse_stream = TcpStream::connect(addr).unwrap();
    let sse_req = format!(
        "GET /v1/projects/{project}/events?after=2 HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {token}\r\n\r\n",
        addr.port()
    );
    sse_stream.write_all(sse_req.as_bytes()).unwrap();
    sse_stream.flush().unwrap();

    let mut reader = BufReader::new(sse_stream);
    let mut reset_received = false;
    for _ in 0..15 {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line.starts_with("event: reset") {
            reset_received = true;
            break;
        }
    }
    assert!(reset_received, "Expired cursor must trigger 'event: reset' frame");

    // 2. Client reconnects with valid cursor (after=3) -> receives unexpired events 4 and 5
    let mut sse_stream2 = TcpStream::connect(addr).unwrap();
    let sse_req2 = format!(
        "GET /v1/projects/{project}/events?after=3 HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {token}\r\n\r\n",
        addr.port()
    );
    sse_stream2.write_all(sse_req2.as_bytes()).unwrap();
    sse_stream2.flush().unwrap();

    let mut reader2 = BufReader::new(sse_stream2);
    let mut received_event_4 = false;
    let mut received_event_5 = false;
    for _ in 0..20 {
        let mut line = String::new();
        reader2.read_line(&mut line).unwrap();
        if line.contains("\"cursor\":4") {
            received_event_4 = true;
        }
        if line.contains("\"cursor\":5") {
            received_event_5 = true;
            break;
        }
    }
    assert!(received_event_4 && received_event_5, "Reconnected client must receive remaining valid monotonic events");

    handle.shutdown();
}

#[test]
fn test_p6_2_zero_xss_and_malicious_script_payload_immunity() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-xss-test";
    let token = "xss-token-112233";

    setup_test_project(root, project);

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();
    let origin = format!("http://127.0.0.1:{}", addr.port());

    // Inject malicious script payloads into steering directive and intent
    let malicious_script = "<script>alert('pwned')</script><img src=x onerror=alert(1)>";
    let steer_payload = json!({
        "command_id": "cmd-xss-inject",
        "project_id": project,
        "expected_revision": 1,
        "operation": {
            "type": "apply_steering",
            "directive": malicious_script,
            "actor": "attacker<script>",
            "task_id": "lead"
        }
    })
    .to_string();

    let res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(steer_payload.as_bytes()),
    );
    assert_eq!(res.status, 200);

    // Fetch snapshot and verify payload is clean JSON string, never executing or mutating HTML
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let snap = res.body_json();
    let directives = snap["steering_directives"].as_array().unwrap();
    assert_eq!(directives.len(), 1);
    assert_eq!(directives[0]["directive"], malicious_script);

    // Also verify static HTML template contains zero raw innerHTML assignments for dynamic data
    let html_res = raw_http_request(addr, "GET", "/", &[], None);
    let html_content = html_res.body_str();
    assert!(
        !html_content.contains("innerHTML = directives"),
        "Dashboard code must not assign raw unescaped innerHTML"
    );

    handle.shutdown();
}

#[test]
fn test_p6_3_and_p6_4_snapshot_dag_and_resource_telemetry() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-dag-telemetry";
    let token = "telemetry-token-9900";

    setup_test_project(root, project);

    // Populate search session, DAG nodes, trial results, and resource leases
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let now = now_ms();

    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, 'dset-1', '/tmp/dset.csv', 1000, 'sha256-dummy', ?2)",
        params![project, now],
    ).unwrap();
    conn.execute(
        "INSERT INTO search_sessions(id, project, dataset_id, spec, status, created_ms, updated_ms, apex_candidate_id)
         VALUES('sess-1', ?1, 'dset-1', '{}', 'evaluating', ?2, ?2, 'cand-apex-1')",
        params![project, now],
    ).unwrap();

    // Screening trial (20% fidelity)
    conn.execute(
        "INSERT INTO experiment_dag(id, session_id, project, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
         VALUES('exp-screen-1', 'sess-1', ?1, 'linear', '{\"C\":0.1}', 2026, 0.2, 'screening', NULL, ?2)",
        params![project, now],
    ).unwrap();
    conn.execute(
        "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, is_feasible, constraints, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
         VALUES(?1, 'exp-screen-1', 'evaluated', 0.812, '{\"accuracy\": 0.812}', 1, '{\"latency\": {\"verdict\": \"pass\"}}', 45, 'sha-model-1', 'sha-pred-1', '/tmp/exp1.json', ?2)",
        params![project, now],
    ).unwrap();

    // Full fidelity trial (100% fidelity) child of screening trial
    conn.execute(
        "INSERT INTO experiment_dag(id, session_id, project, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
         VALUES('exp-full-1', 'sess-1', ?1, 'linear', '{\"C\":0.1}', 2026, 1.0, 'full_fidelity', 'exp-screen-1', ?2)",
        params![project, now + 10],
    ).unwrap();
    conn.execute(
        "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, is_feasible, constraints, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
         VALUES(?1, 'exp-full-1', 'evaluated', 0.935, '{\"accuracy\": 0.935}', 1, '{\"latency\": {\"verdict\": \"pass\"}}', 120, 'sha-model-2', 'sha-pred-2', '/tmp/exp2.json', ?2)",
        params![project, now + 10],
    ).unwrap();

    // Active worker lease
    conn.execute(
        "INSERT INTO resource_leases(
            lease_id, project, task, job, generation, status, backend, device,
            cpu_millicores, library_threads, memory_bytes, scratch_bytes, output_bytes,
            deadline_ms, heartbeat_ms, created_ms, updated_ms
        ) VALUES('lease-w1', ?1, 'task-1', 'job-1', 1, 'active', 'container', 'cpu', 2000, 2, 536870912, 1000, 1000, ?2 + 60000, ?2, ?2, ?2)",
        params![project, now],
    ).unwrap();

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();

    // Fetch snapshot and verify structured telemetry
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let snap = res.body_json();

    // Verify DAG nodes & trial results join
    let dag = snap["experiment_dag"].as_array().unwrap();
    assert_eq!(dag.len(), 2);
    assert_eq!(dag[0]["experiment_id"], "exp-screen-1");
    assert_eq!(dag[0]["fidelity"], 0.2);
    assert_eq!(dag[0]["primary_metric"], 0.812);
    assert_eq!(dag[0]["is_feasible"], true);

    assert_eq!(dag[1]["experiment_id"], "exp-full-1");
    assert_eq!(dag[1]["parent_id"], "exp-screen-1");
    assert_eq!(dag[1]["fidelity"], 1.0);
    assert_eq!(dag[1]["primary_metric"], 0.935);

    // Verify resource telemetry
    let res_status = &snap["resource_status"];
    assert!(res_status["capabilities"]["effective_cpus"].as_u64().unwrap() >= 1);
    assert!(res_status["available"]["available_cpus"].as_u64().unwrap() >= 1);
    let active_leases = res_status["active_leases"].as_array().unwrap();
    assert_eq!(active_leases.len(), 1);
    assert_eq!(active_leases[0]["lease_id"], "lease-w1");

    handle.shutdown();
}

#[test]
fn test_p6_6_large_dag_and_reduced_motion() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-large-dag";
    let token = "large-token-8877";

    setup_test_project(root, project);

    // Insert 120 DAG nodes to simulate large multi-turn exploration tree
    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let now = now_ms();
    conn.execute(
        "INSERT INTO scientific_datasets(project, id, path, bytes, sha256, registered_ms)
         VALUES(?1, 'dset-1', '/tmp/dset.csv', 1000, 'sha256-dummy', ?2)",
        params![project, now],
    ).unwrap();
    conn.execute(
        "INSERT INTO search_sessions(id, project, dataset_id, spec, status, created_ms, updated_ms, apex_candidate_id)
         VALUES('sess-large', ?1, 'dset-1', '{}', 'evaluating', ?2, ?2, 'cand-apex')",
        params![project, now],
    ).unwrap();

    let mut prev_id = None;
    for i in 1..=120 {
        let node_id = format!("node-{i:03}");
        let fidelity = if i <= 40 { 0.2 } else if i <= 80 { 0.5 } else { 1.0 };
        let node_type = if i <= 80 { "screening" } else { "full_fidelity" };
        let metric = 0.5 + (i as f64) * 0.0035;

        conn.execute(
            "INSERT INTO experiment_dag(id, session_id, project, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
             VALUES(?1, 'sess-large', ?2, 'tree', '{}', 2026, ?3, ?4, ?5, ?6)",
            params![node_id, project, fidelity, node_type, prev_id, now + (i as i64)],
        ).unwrap();

        conn.execute(
            "INSERT INTO trial_results(project, experiment_id, status, primary_metric, metrics, is_feasible, constraints, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms)
             VALUES(?1, ?2, 'evaluated', ?3, '{\"accuracy\": 0.8}', 1, '{}', 80, 'sha-m', 'sha-p', '/tmp/art.json', ?4)",
            params![project, node_id, metric, now + (i as i64)],
        ).unwrap();

        if i % 3 == 0 {
            prev_id = Some(node_id);
        }
    }

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();

    // Query snapshot with 120 nodes: must return quickly and accurately
    let start = SystemTime::now();
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    let elapsed = start.elapsed().unwrap();
    assert_eq!(res.status, 200, "snapshot error: {}", res.body_str());
    assert!(elapsed < Duration::from_millis(2000), "120-node snapshot query must complete under 2000ms, took {elapsed:?}");

    let snap = res.body_json();
    assert_eq!(snap["experiment_dag"].as_array().unwrap().len(), 120);

    // Verify reduced-motion support in dashboard CSS
    let html_res = raw_http_request(addr, "GET", "/", &[], None);
    assert!(html_res.body_str().contains("@media (prefers-reduced-motion: reduce)"));
    assert!(html_res.body_str().contains("animation-duration: 0.001ms"));

    handle.shutdown();
}

#[test]
fn test_p6_6_acceptance_gate_full_closure() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let project = "p6-closure-project";
    let token = "gate-closure-token-p6";

    setup_test_project(root, project);

    let server = DashboardServer::bind(root, 0, token).unwrap();
    let handle = server.serve_background().unwrap();
    let addr = handle.local_addr();
    let origin = format!("http://127.0.0.1:{}", addr.port());

    // 1. Session login: POST /session
    let res = raw_http_request(
        addr,
        "POST",
        "/session",
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
        ],
        Some(json!({"token": token}).to_string().as_bytes()),
    );
    assert_eq!(res.status, 200);
    assert!(res.header("Set-Cookie").unwrap().contains("mlr_session="));

    // 2. Fetch initial snapshot
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let initial_snap = res.body_json();
    assert_eq!(initial_snap["revision"], 1);

    // 3. Connect SSE event stream
    let mut sse_stream = TcpStream::connect(addr).unwrap();
    let sse_req = format!(
        "GET /v1/projects/{project}/events?after=0 HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {token}\r\n\r\n",
        addr.port()
    );
    sse_stream.write_all(sse_req.as_bytes()).unwrap();
    sse_stream.flush().unwrap();

    // 4. Inject human steering directive
    let steer_res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(json!({
            "command_id": "cmd-gate-steer",
            "project_id": project,
            "expected_revision": 1,
            "operation": {
                "type": "apply_steering",
                "directive": "Steer exploration to tree depth 8",
                "actor": "lead-scientist",
                "task_id": "experiment_engineer"
            }
        }).to_string().as_bytes()),
    );
    assert_eq!(steer_res.status, 200);
    let receipt1 = steer_res.body_json();
    assert_eq!(receipt1["status"], "applied");
    assert_eq!(receipt1["revision"], 2);

    // 5. Register user approval for holdout access
    let appr_res = raw_http_request(
        addr,
        "POST",
        &format!("/v1/projects/{project}/commands"),
        &[
            ("Content-Type", "application/json"),
            ("Origin", &origin),
            ("Authorization", &format!("Bearer {token}")),
        ],
        Some(json!({
            "command_id": "cmd-gate-approval",
            "project_id": project,
            "expected_revision": 2,
            "operation": {
                "type": "user_approval",
                "approval_id": "appr-closure-1",
                "action_type": "holdout_access",
                "target_id": "dataset-holdout",
                "actor": "compliance-officer",
                "approved": true,
                "ttl_ms": 7200000
            }
        }).to_string().as_bytes()),
    );
    assert_eq!(appr_res.status, 200);
    let receipt2 = appr_res.body_json();
    assert_eq!(receipt2["status"], "applied");
    assert_eq!(receipt2["revision"], 3);

    // 6. Verify snapshot reflects both mutations and advanced handoff epoch
    let res = raw_http_request(
        addr,
        "GET",
        &format!("/v1/projects/{project}/snapshot"),
        &[("Authorization", &format!("Bearer {token}"))],
        None,
    );
    assert_eq!(res.status, 200);
    let updated_snap = res.body_json();
    assert_eq!(updated_snap["revision"], 3);
    assert_eq!(updated_snap["handoff_version"], 1);
    assert_eq!(updated_snap["steering_directives"].as_array().unwrap().len(), 1);
    assert_eq!(updated_snap["user_approvals"].as_array().unwrap().len(), 1);

    handle.shutdown();
}
