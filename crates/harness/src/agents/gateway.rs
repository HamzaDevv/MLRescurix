use std::{
    fs,
    path::Path,
    process::Command,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: "assistant".into(),
            content,
            tool_calls,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
            name: Some(name.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub model: String,
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub usage: TokenUsage,
    pub latency_ms: u64,
}

pub struct OutboundFilter;

fn is_raw_tabular_row(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty()
        || trimmed.starts_with('{')
        || trimmed.starts_with('}')
        || trimmed.starts_with('[')
        || trimmed.starts_with(']')
        || trimmed.starts_with('#')
    {
        return false;
    }
    // Disqualify natural sentences and markdown/bullet lines
    if trimmed.ends_with('.')
        || trimmed.ends_with(':')
        || trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
    {
        return false;
    }
    let comma_count = trimmed.matches(',').count();
    let comma_space_count = trimmed.matches(", ").count();
    let tab_count = trimmed.matches('\t').count();
    let pipe_count = trimmed.matches('|').count();

    // In raw CSV data, commas separate unspaced values (e.g. 1.2,3.4,5.6).
    // In prose English, commas are followed by a space (e.g. ", ").
    if comma_count >= 2 && comma_space_count == 0 {
        return true;
    }
    if tab_count >= 2 {
        return true;
    }
    if pipe_count >= 2 && !trimmed.contains("||") {
        return true;
    }
    false
}

impl OutboundFilter {
    /// Redact raw data rows and truncate unbounded context items to prevent leaking private row data to providers.
    pub fn sanitize_content(text: &str) -> String {
        // 1. Detect JSON array of data records
        if let Ok(serde_json::Value::Array(items)) = serde_json::from_str::<serde_json::Value>(text) {
            if items.len() > 3 && items.iter().all(|item| item.is_object()) {
                let sample_keys_len = items[0].as_object().map(|o| o.len()).unwrap_or(0);
                if sample_keys_len >= 2 {
                    let schema_keys: Vec<String> = items[0].as_object().map(|o| o.keys().cloned().collect()).unwrap_or_default();
                    let redacted_count = items.len();
                    return serde_json::json!({
                        "schema_features": schema_keys,
                        "redacted_record_count": redacted_count,
                        "notice": format!("[REDACTED_ROW_DATA: {redacted_count} raw tabular records omitted; use aggregate diagnostics]")
                    }).to_string();
                }
            }
        }

        let lines: Vec<&str> = text.lines().collect();

        // 2. Detect if content contains CSV, TSV or pipe-delimited raw tables
        let mut cleaned_lines: Vec<String> = Vec::new();
        let mut table_rows: Vec<String> = Vec::new();

        let flush_table = |cleaned: &mut Vec<String>, rows: &mut Vec<String>| {
            if rows.len() > 3 {
                // Keep the first row (header) and redact remaining raw data rows
                cleaned.push(rows[0].clone());
                cleaned.push(format!(
                    "[REDACTED_ROW_DATA: {} raw tabular rows omitted; use aggregate diagnostics]",
                    rows.len() - 1
                ));
            } else {
                cleaned.extend(rows.drain(..));
            }
            rows.clear();
        };

        for line in lines {
            if is_raw_tabular_row(line) {
                table_rows.push(line.to_string());
            } else {
                if !table_rows.is_empty() {
                    flush_table(&mut cleaned_lines, &mut table_rows);
                }
                cleaned_lines.push(line.to_string());
            }
        }
        if !table_rows.is_empty() {
            flush_table(&mut cleaned_lines, &mut table_rows);
        }

        let combined = cleaned_lines.join("\n");
        // Limit character ceiling per item
        if combined.len() > 16_384 {
            format!("{}... [TRUNCATED]", &combined[..combined.floor_char_boundary(16_384)])
        } else {
            combined
        }
    }

    pub fn sanitize_request(req: &ChatCompletionRequest) -> ChatCompletionRequest {
        let sanitized_messages = req.messages.iter().map(|msg| {
            let mut sanitized = msg.clone();
            if let Some(c) = &msg.content {
                sanitized.content = Some(Self::sanitize_content(c));
            }
            if let Some(calls) = &msg.tool_calls {
                let sanitized_calls = calls.iter().map(|c| {
                    let mut call = c.clone();
                    call.function.arguments = Self::sanitize_content(&c.function.arguments);
                    call
                }).collect();
                sanitized.tool_calls = Some(sanitized_calls);
            }
            sanitized
        }).collect();

        ChatCompletionRequest {
            model: req.model.clone(),
            messages: sanitized_messages,
            tools: req.tools.clone(),
            temperature: req.temperature,
            max_tokens: req.max_tokens,
            provider: req.provider.clone(),
            reasoning: req.reasoning.clone(),
        }
    }
}

pub trait ProviderGateway: Send + Sync {
    fn requires_policy_approval(&self)->bool {false}
    fn chat_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse>;
}

/// Scripted provider fixture for deterministic offline testing and CI.
pub struct ScriptedProviderFixture {
    responses: Mutex<Vec<ChatCompletionResponse>>,
}

impl ScriptedProviderFixture {
    pub fn new(responses: Vec<ChatCompletionResponse>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }

    pub fn queue_response(&self, resp: ChatCompletionResponse) {
        let mut guard = self.responses.lock().unwrap();
        guard.push(resp);
    }
}

impl ProviderGateway for ScriptedProviderFixture {
    fn chat_completion(&self, _request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let mut guard = self.responses.lock().unwrap();
        if guard.is_empty() {
            return Err("ScriptedProviderFixture has no remaining queued responses".into());
        }
        Ok(guard.remove(0))
    }
}

/// OpenAI / OpenRouter HTTP adapter using curl.
pub struct OpenAICompatibleAdapter {
    pub endpoint_url: String,
    pub api_key: String,
    pub timeout_seconds: u32,
    pub max_retries: u32,
}

impl OpenAICompatibleAdapter {
    pub fn new(endpoint_url: String, api_key: String) -> Self {
        let normalized = if endpoint_url.ends_with("/chat/completions") {
            endpoint_url
        } else if endpoint_url.ends_with('/') {
            format!("{endpoint_url}chat/completions")
        } else {
            format!("{endpoint_url}/chat/completions")
        };
        Self {
            endpoint_url: normalized,
            api_key,
            timeout_seconds: 60,
            max_retries: 1,
        }
    }
}

impl ProviderGateway for OpenAICompatibleAdapter {
    fn requires_policy_approval(&self)->bool {true}
    fn chat_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let sanitized = OutboundFilter::sanitize_request(request);
        let req_json = serde_json::to_string(&sanitized).map_err(|e| e.to_string())?;

        let payload_path = std::env::temp_dir().join(format!("mlr_req_{}_{}.json", now_ms(), std::process::id()));
        fs::write(&payload_path, &req_json).map_err(|e| e.to_string())?;

        let mut attempts = 0;
        let mut last_error = String::new();

        while attempts < self.max_retries.min(1) {
            attempts += 1;
            let start = Instant::now();

            let output = Command::new("curl")
                .arg("-s")
                .arg("-S")
                .arg("-X").arg("POST")
                .arg(&self.endpoint_url)
                .arg("-H").arg("Content-Type: application/json")
                .arg("-H").arg(format!("Authorization: Bearer {}", self.api_key))
                .arg("--data-binary").arg(format!("@{}", payload_path.to_string_lossy()))
                .arg("--max-time").arg(self.timeout_seconds.to_string())
                .output();

            let elapsed = start.elapsed().as_millis() as u64;

            match output {
                Ok(out) => {
                    if !out.status.success() {
                        last_error = format!("curl failed (code {:?}): {}", out.status.code(), String::from_utf8_lossy(&out.stderr));
                        std::thread::sleep(Duration::from_millis(100 * (1 << attempts)));
                        continue;
                    }

                    let body: Value = serde_json::from_slice(&out.stdout)
                        .map_err(|e| format!("cannot parse provider response JSON: {e} - stdout: {}", String::from_utf8_lossy(&out.stdout)))?;

                    if let Some(err) = body.get("error") {
                        return Err(format!("provider returned error: {err}"));
                    }

                    let choices = body["choices"].as_array().ok_or("missing choices array")?;
                    let first = choices.first().ok_or("choices array is empty")?;
                    let message = &first["message"];
                    let content = message["content"].as_str().map(|s| s.to_string());

                    let mut tool_calls = Vec::new();
                    if let Some(calls) = message["tool_calls"].as_array() {
                        for c in calls {
                            let id = c["id"].as_str().unwrap_or("").to_string();
                            let c_type = c["type"].as_str().unwrap_or("function").to_string();
                            let fn_name = c["function"]["name"].as_str().unwrap_or("").to_string();
                            let fn_args = c["function"]["arguments"].as_str().unwrap_or("{}").to_string();
                            tool_calls.push(ToolCall {
                                id,
                                call_type: c_type,
                                function: FunctionCall {
                                    name: fn_name,
                                    arguments: fn_args,
                                },
                            });
                        }
                    }

                    let prompt_tokens = body["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as u32;
                    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap_or(0) as u32;
                    let total_tokens = body["usage"]["total_tokens"].as_u64().unwrap_or(0) as u32;

                    let _ = fs::remove_file(&payload_path);
                    return Ok(ChatCompletionResponse {
                        id: body["id"].as_str().unwrap_or("resp-unknown").to_string(),
                        model: body["model"].as_str().unwrap_or(&request.model).to_string(),
                        content,
                        tool_calls,
                        usage: TokenUsage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                        },
                        latency_ms: elapsed,
                    });
                }
                Err(e) => {
                    last_error = format!("cannot execute curl: {e}");
                    std::thread::sleep(Duration::from_millis(100 * (1 << attempts)));
                }
            }
        }

        let _ = fs::remove_file(&payload_path);
        Err(format!("provider request failed after {attempts} attempts: {last_error}"))
    }
}

pub fn register_model_pricing(
    root: &Path,
    model: &str,
    prompt_millicents_per_1k: i64,
    completion_millicents_per_1k: i64,
) -> Result<Value> {
    if model.is_empty() || prompt_millicents_per_1k < 0 || completion_millicents_per_1k < 0 {
        return Err("invalid model pricing specification".into());
    }
    let pricing_path = root.join("pricing.json");
    let mut table: serde_json::Map<String, Value> = if pricing_path.is_file() {
        serde_json::from_slice(&fs::read(&pricing_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?
    } else {
        serde_json::Map::new()
    };
    table.insert(
        model.to_string(),
        serde_json::json!({
            "prompt_millicents_per_1k": prompt_millicents_per_1k,
            "completion_millicents_per_1k": completion_millicents_per_1k,
        }),
    );
    fs::write(
        &pricing_path,
        serde_json::to_vec_pretty(&Value::Object(table)).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "model": model,
        "prompt_millicents_per_1k": prompt_millicents_per_1k,
        "completion_millicents_per_1k": completion_millicents_per_1k,
    }))
}

pub fn resolve_model_pricing(root: &Path, model: &str) -> Result<Option<(i64, i64)>> {
    for candidate in [root.join("pricing.json"), std::path::PathBuf::from(".config/provider.json")] {
        if candidate.is_file() {
            if let Ok(bytes) = fs::read(&candidate) {
                if let Ok(val) = serde_json::from_slice::<Value>(&bytes) {
                    let entry = val.get(model).or_else(|| val.get("pricing").and_then(|p| p.get(model)));
                    if let Some(obj) = entry {
                        if let (Some(p), Some(c)) = (
                            obj.get("prompt_millicents_per_1k").and_then(|v| v.as_i64()),
                            obj.get("completion_millicents_per_1k").and_then(|v| v.as_i64()),
                        ) {
                            if p >= 0 && c >= 0 {
                                return Ok(Some((p, c)));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(None)
}

pub fn approve_outbound_view(
    root: &Path,
    project_id: &str,
    request_sha256: &str,
    actor: &str,
    approved: bool,
    ttl_ms: i64,
) -> Result<Value> {
    if actor.is_empty() || request_sha256.len() != 64 || !request_sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("invalid outbound view approval parameters".into());
    }
    if !(1..=86_400_000).contains(&ttl_ms) {
        return Err("outbound view approval ttl_ms must be 1..=86_400_000".into());
    }
    crate::execution::migrate(root)?;
    let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let view_json: String = db
        .query_row(
            "SELECT view_json FROM outbound_views WHERE project=?1 AND request_sha256=?2",
            rusqlite::params![project_id, request_sha256],
            |r| r.get(0),
        )
        .map_err(|_| format!("staged outbound view '{request_sha256}' not found for project '{project_id}'"))?;
    use sha2::{Digest, Sha256};
    let actual_sha = format!("{:x}", Sha256::digest(view_json.as_bytes()));
    if actual_sha != request_sha256 {
        return Err("staged outbound view content digest mismatch".into());
    }
    let expires_ms = now_ms().saturating_add(ttl_ms);
    db.execute(
        "UPDATE outbound_views SET actor=?3, approved=?4, expires_ms=?5 WHERE project=?1 AND request_sha256=?2",
        rusqlite::params![project_id, request_sha256, actor, approved as i64, expires_ms],
    )
    .map_err(|e| e.to_string())?;
    db.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, 'outbound_view.decision', ?2)",
        rusqlite::params![
            project_id,
            serde_json::json!({
                "request_sha256": request_sha256,
                "actor": actor,
                "approved": approved,
                "expires_ms": expires_ms
            })
            .to_string()
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "project_id": project_id,
        "request_sha256": request_sha256,
        "actor": actor,
        "approved": approved,
        "expires_ms": expires_ms,
    }))
}

fn ensure_project_record(conn: &rusqlite::Connection, project_id: &str) -> Result<()> {
    let exists: bool = conn
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM projects WHERE id=?1)",
            rusqlite::params![project_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if !exists {
        // This row only satisfies receipt foreign keys. It grants no M0 budget;
        // the campaign authority must reserve spend before calling this gateway.
        let dummy_policy = "{\"budget\":{\"calls\":0,\"tokens\":0,\"wall_ms\":0},\"capacity\":{\"workers\":0,\"cpu_threads\":0},\"max_tasks\":0}";
        conn.execute(
            "INSERT INTO projects(id, revision, spec_revision, intent_revision, intent, finalization_id, state, policy)
             VALUES(?1, 1, 1, 1, 'Auto-registered by RustGateway', ?2, 'running', ?3)",
            rusqlite::params![project_id, format!("fin-{project_id}"), dummy_policy],
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn cost_from_pricing(usage: &TokenUsage, rates: (i64, i64)) -> Result<i64> {
    let round_up = |tokens: u32, rate: i64| -> i128 {
        let product = i128::from(tokens) * i128::from(rate);
        (product + 999) / 1000
    };
    let cost = round_up(usage.prompt_tokens, rates.0) + round_up(usage.completion_tokens, rates.1);
    i64::try_from(cost).map_err(|_| "provider cost exceeds supported accounting range".into())
}

fn reserve_gateway_receipt(root: &Path, project: &str, task: &str, request_id: &str, digest: &str) -> Result<Option<ChatCompletionResponse>> {
    use rusqlite::OptionalExtension;
    if request_id.is_empty() || request_id.len() > 256 || request_id.chars().any(char::is_control) {
        return Err("invalid gateway request id".into());
    }
    crate::execution::migrate(root)?;
    let mut db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.busy_timeout(Duration::from_secs(10)).map_err(|e| e.to_string())?;
    let tx = db.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    ensure_project_record(&tx, project)?;
    let existing: Option<(String, String, Option<String>)> = tx.query_row(
        "SELECT request_sha256, status, response_json FROM gateway_dispatch_receipts WHERE project=?1 AND task_id=?2 AND request_id=?3",
        rusqlite::params![project, task, request_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    ).optional().map_err(|e| e.to_string())?;
    if let Some((old_digest, status, response_json)) = existing {
        if old_digest != digest {
            return Err("gateway request id was already used with different content".into());
        }
        match status.as_str() {
            "response_received" => {
                let raw = response_json.ok_or("gateway response receipt is missing its response")?;
                tx.commit().map_err(|e| e.to_string())?;
                return serde_json::from_str(&raw).map(Some).map_err(|e| format!("invalid durable gateway response: {e}"));
            }
            "awaiting_response" | "outcome_unknown" => {
                return Err(format!("gateway request {request_id} has an ambiguous prior outcome; refusing provider replay"));
            }
            other => return Err(format!("unsupported gateway receipt status: {other}")),
        }
    }
    crate::execution::enforce_state_write_admission(root, Some(&tx),
        (project.len() + task.len() + request_id.len() + digest.len()) as u64 * 4 + 16 * 1024,
        crate::execution::StorageWriteKind::ProviderReceipt, Some(project))?;
    tx.execute(
        "INSERT INTO gateway_dispatch_receipts(project, task_id, request_id, request_sha256, status, response_json, created_ms, updated_ms)
         VALUES(?1, ?2, ?3, ?4, 'awaiting_response', NULL, ?5, ?5)",
        rusqlite::params![project, task, request_id, digest, now_ms()],
    ).map_err(|e| e.to_string())?;
    tx.commit().map_err(|e| e.to_string())?;
    Ok(None)
}

fn finish_gateway_receipt(root: &Path, project: &str, task: &str, request_id: &str, response: Option<&ChatCompletionResponse>) -> Result<()> {
    let (status, json) = match response {
        Some(resp) => ("response_received", Some(serde_json::to_string(resp).map_err(|e| e.to_string())?)),
        None => ("outcome_unknown", None),
    };
    let bytes = json.as_ref().map_or(4096, |value| value.len() as u64 * 4 + 16 * 1024);
    crate::execution::with_state_write(root, bytes, crate::execution::StorageWriteKind::ProviderReceipt,
        Some(project), |tx| {
            let changed = tx.execute(
                "UPDATE gateway_dispatch_receipts SET status=?4, response_json=?5, updated_ms=?6 WHERE project=?1 AND task_id=?2 AND request_id=?3 AND status='awaiting_response'",
                rusqlite::params![project, task, request_id, status, json, now_ms()],
            ).map_err(|e| e.to_string())?;
            if changed != 1 { return Err("gateway receipt was concurrently changed".into()); }
            Ok(())
        })
}

pub fn record_provider_accounting(
    root: &Path,
    project_id: &str,
    request_id: &str,
    task_id: &str,
    role_id: &str,
    model: &str,
    usage: &TokenUsage,
    cost_millicents: i64,
    latency_ms: u64,
    status: &str,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let mut connection=rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e|e.to_string())?;
    connection.busy_timeout(Duration::from_secs(10)).map_err(|e|e.to_string())?;
    let db=connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e|e.to_string())?;
    ensure_project_record(&db, project_id)?;
    crate::execution::enforce_state_write_admission(root, Some(&db), (model.len()+status.len()+request_id.len()+task_id.len()+role_id.len()) as u64*4+16*1024, crate::execution::StorageWriteKind::ProviderReceipt, Some(project_id))?;
    db.execute(
        "INSERT INTO provider_accounting(project, request_id, task_id, role_id, model, prompt_tokens, completion_tokens, cost_millicents, latency_ms, status, created_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
         ON CONFLICT(project, request_id) DO UPDATE SET
           prompt_tokens=excluded.prompt_tokens,
           completion_tokens=excluded.completion_tokens,
           cost_millicents=excluded.cost_millicents,
           latency_ms=excluded.latency_ms,
           status=excluded.status",
        rusqlite::params![
            project_id,
            request_id,
            task_id,
            role_id,
            model,
            usage.prompt_tokens,
            usage.completion_tokens,
            cost_millicents,
            latency_ms as i64,
            status,
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    db.commit().map_err(|e|e.to_string())?;
    // M0 action settlement owns charges; this table is supplemental telemetry.
    Ok(())
}

pub fn stage_outbound_request(
    root: &Path,
    project_id: &str,
    request: &ChatCompletionRequest,
    actor: &str,
) -> Result<Value> {
    use sha2::{Digest, Sha256};
    crate::execution::migrate(root)?;
    let sanitized = OutboundFilter::sanitize_request(request);
    let request_bytes = serde_json::to_vec(&sanitized).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&request_bytes));
    let view_json = String::from_utf8(request_bytes).map_err(|e| e.to_string())?;
    let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    ensure_project_record(&db, project_id)?;
    db.execute(
        "INSERT INTO outbound_views(project, request_sha256, view_json, actor, approved, expires_ms)
         VALUES(?1, ?2, ?3, ?4, 0, 0)
         ON CONFLICT(project, request_sha256) DO UPDATE SET view_json=excluded.view_json, actor=excluded.actor WHERE outbound_views.approved=0",
        rusqlite::params![project_id, digest, view_json, actor],
    )
    .map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "project_id": project_id,
        "request_sha256": digest,
        "actor": actor,
        "approved": false,
    }))
}

pub fn dispatch_gateway(
    root: &Path,
    project_id: &str,
    task_id: &str,
    role_id: &str,
    request: &ChatCompletionRequest,
    gateway: &dyn ProviderGateway,
    require_policy_approval: bool,
) -> Result<Value> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
    dispatch_gateway_with_id(root, project_id, task_id, role_id,
        &format!("ephemeral-{}-{nonce}", std::process::id()), request, gateway, require_policy_approval)
}

/// Dispatch with caller-owned idempotency identity. A caller must reuse the same
/// ID when recovering an interrupted turn and choose a new ID for every new turn.
pub fn dispatch_gateway_with_id(
    root: &Path,
    project_id: &str,
    task_id: &str,
    role_id: &str,
    request_id: &str,
    request: &ChatCompletionRequest,
    gateway: &dyn ProviderGateway,
    require_policy_approval: bool,
) -> Result<Value> {
    use sha2::{Digest, Sha256};
    crate::execution::migrate(root)?;
    let sanitized = OutboundFilter::sanitize_request(request);
    let sanitized_bytes = serde_json::to_vec(&sanitized).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&sanitized_bytes));

    // Refuse before transport if the campaign has no registered ceiling for the
    // selected model. The Python campaign authority has already reserved against
    // the same verified ceiling before this call.
    let rates = resolve_model_pricing(root, &sanitized.model)?
        .ok_or_else(|| format!("model '{}' has no registered pricing; refusing provider dispatch", sanitized.model))?;

    if require_policy_approval || gateway.requires_policy_approval() {
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let approved: bool = db.query_row(
            "SELECT EXISTS(SELECT 1 FROM outbound_views WHERE project=?1 AND request_sha256=?2 AND approved=1 AND expires_ms>?3)",
            rusqlite::params![project_id, digest, now_ms()],
            |r| r.get(0),
        ).map_err(|e| e.to_string())?;
        if !approved {
            ensure_project_record(&db, project_id)?;
            let view_json = String::from_utf8(sanitized_bytes.clone()).unwrap_or_default();
            let _ = db.execute(
                "INSERT INTO outbound_views(project, request_sha256, view_json, actor, approved, expires_ms)
                 VALUES(?1, ?2, ?3, 'unapproved', 0, 0)
                 ON CONFLICT(project, request_sha256) DO UPDATE SET view_json=excluded.view_json WHERE outbound_views.approved=0",
                rusqlite::params![project_id, digest, view_json],
            );
            return Err(format!("outbound provider request requires a content-bound policy-approved view ({digest}); pattern redaction is not upload authorization"));
        }
    }

    let cached_response = reserve_gateway_receipt(root, project_id, task_id, request_id, &digest)?;
    let replayed = cached_response.is_some();
    let resp = match cached_response {
        Some(response) => response,
        None => match gateway.chat_completion(&sanitized) {
            Ok(response) => {
                // Persist exact response before accounting or returning it. If this
                // write fails, the `awaiting_response` row fences retries as unknown.
                finish_gateway_receipt(root, project_id, task_id, request_id, Some(&response))?;
                response
            }
            Err(error) => {
                finish_gateway_receipt(root, project_id, task_id, request_id, None)?;
                return Err(error);
            }
        },
    };

    let cost_millicents = cost_from_pricing(&resp.usage, rates)?;
    let cost_usd = cost_millicents as f64 / 100_000.0;

    let accounting_identity = format!("{project_id}\0{task_id}\0{request_id}");
    let request_key = format!("gw-{}", Sha256::digest(accounting_identity.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>());
    record_provider_accounting(
        root,
        project_id,
        &request_key,
        task_id,
        role_id,
        &resp.model,
        &resp.usage,
        cost_millicents,
        resp.latency_ms,
        "ok",
    )?;

    Ok(serde_json::json!({
        "status": "completed",
        "id": resp.id,
        "model": resp.model,
        "content": resp.content,
        "tool_calls": resp.tool_calls,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens
        },
        "cost_millicents": cost_millicents,
        "cost_usd": cost_usd,
        "latency_ms": resp.latency_ms,
        "request_sha256": digest,
        "request_id": request_id,
        "replayed": replayed,
        "cost_is_estimate": true
    }))
}
