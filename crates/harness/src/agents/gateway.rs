use std::{
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
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
    fn is_offline_fixture(&self) -> bool { false }
    fn requires_policy_approval(&self)->bool {false}
    fn adapter_protocol(&self) -> &str { "scripted_fixture" }
    fn endpoint_url(&self) -> &str { "" }
    fn campaign_budget_context(&self) -> Result<(std::path::PathBuf, String)> {
        Ok((std::env::var("MLRESCURIX_CAMPAIGN_LEDGER").map(std::path::PathBuf::from)
            .map_err(|_| "shared campaign ledger required for paid transport")?,
            std::env::var("MLRESCURIX_CAMPAIGN_RUN").map_err(|_| "campaign run identity required")?))
    }
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
    fn is_offline_fixture(&self) -> bool { true }
    fn adapter_protocol(&self) -> &str { "scripted_fixture" }
    fn endpoint_url(&self) -> &str { "fixture://scripted" }
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
    fn adapter_protocol(&self) -> &str { "openai_compatible" }
    fn endpoint_url(&self) -> &str { &self.endpoint_url }
    fn chat_completion(&self, request: &ChatCompletionRequest) -> Result<ChatCompletionResponse> {
        let sanitized = OutboundFilter::sanitize_request(request);
        let req_json = serde_json::to_string(&sanitized).map_err(|e| e.to_string())?;

        if self.api_key.is_empty() || !self.api_key.bytes().all(|b| b.is_ascii_alphanumeric() || b"-_.".contains(&b)) {
            return Err("provider API key contains unsupported characters".into());
        }

        let mut attempts = 0;
        let mut last_error = String::new();

        while attempts < self.max_retries.min(1) {
            attempts += 1;
            let start = Instant::now();

            let child = Command::new("curl")
                .arg("-s")
                .arg("-S")
                .arg("-X").arg("POST")
                .arg(&self.endpoint_url)
                .arg("-H").arg("Content-Type: application/json")
                .arg("--config").arg("-")
                .arg("--max-time").arg(self.timeout_seconds.to_string())
                .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn();
            let output = match child {
                Ok(mut child) => {
                    // Curl config and payload go through a pipe, not process arguments or shared temporary files.
                    let escaped = req_json.replace('\\', "\\\\").replace('"', "\\\"");
                    let config = format!("header = \"Authorization: Bearer {}\"\ndata-binary = \"{}\"\n", self.api_key, escaped);
                    let write_result = child.stdin.take().ok_or("curl stdin unavailable")?.write_all(config.as_bytes());
                    if let Err(e) = write_result {
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(format!("cannot write curl request: {e}"));
                    }
                    child.wait_with_output()
                }
                Err(e) => Err(e),
            };

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

        Err(format!("provider request failed after {attempts} attempts: {last_error}"))
    }
}

pub fn normalize_endpoint(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    if trimmed.starts_with("fixture://") {
        return trimmed.to_string();
    }
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn with_pricing_file_lock<F, R>(root: &Path, f: F) -> Result<R>
where
    F: FnOnce() -> Result<R>,
{
    use std::os::unix::io::AsRawFd;
    let lock_path = root.join("pricing.lock");
    let lock_file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| format!("cannot open pricing lock file: {e}"))?;
    let fd = lock_file.as_raw_fd();
    unsafe {
        if libc::flock(fd, libc::LOCK_EX) != 0 {
            return Err("failed to acquire pricing lock".into());
        }
    }
    struct FlockGuard(i32);
    impl Drop for FlockGuard {
        fn drop(&mut self) {
            unsafe {
                libc::flock(self.0, libc::LOCK_UN);
            }
        }
    }
    let _guard = FlockGuard(fd);
    f()
}

pub fn register_namespaced_model_pricing(
    root: &Path,
    provider_key: &str,
    model: &str,
    prompt_millicents_per_1k: i64,
    completion_millicents_per_1k: i64,
) -> Result<Value> {
    if model.is_empty() || prompt_millicents_per_1k < 0 || completion_millicents_per_1k < 0 {
        return Err("invalid model pricing specification".into());
    }
    with_pricing_file_lock(root, || {
        let pricing_path = root.join("pricing.json");
        let mut table: serde_json::Map<String, Value> = if pricing_path.is_file() {
            serde_json::from_slice(&fs::read(&pricing_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?
        } else {
            serde_json::Map::new()
        };
        let norm_key = if provider_key.is_empty() {
            String::new()
        } else {
            normalize_endpoint(provider_key)
        };
        let namespaced_key = if norm_key.is_empty() {
            model.to_string()
        } else {
            format!("{norm_key}::{model}")
        };
        let entry = serde_json::json!({
            "prompt_millicents_per_1k": prompt_millicents_per_1k,
            "completion_millicents_per_1k": completion_millicents_per_1k,
        });
        table.insert(namespaced_key.clone(), entry.clone());
        if norm_key.is_empty() && table.get(model).is_none() {
            table.insert(model.to_string(), entry);
        }
        let tmp_path = pricing_path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(
            &tmp_path,
            serde_json::to_vec_pretty(&Value::Object(table)).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        fs::rename(&tmp_path, &pricing_path).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({
            "provider": norm_key,
            "model": model,
            "prompt_millicents_per_1k": prompt_millicents_per_1k,
            "completion_millicents_per_1k": completion_millicents_per_1k,
        }))
    })
}

pub fn resolve_namespaced_model_pricing(root: &Path, provider_key: &str, model: &str) -> Result<Option<(i64, i64)>> {
    let norm_key = if provider_key.is_empty() {
        String::new()
    } else {
        normalize_endpoint(provider_key)
    };
    let namespaced_key = if norm_key.is_empty() {
        model.to_string()
    } else {
        format!("{norm_key}::{model}")
    };
    with_pricing_file_lock(root, || {
        for candidate in [root.join("pricing.json"), std::path::PathBuf::from(".config/provider.json")] {
            if candidate.is_file() {
                if let Ok(bytes) = fs::read(&candidate) {
                    if let Ok(val) = serde_json::from_slice::<Value>(&bytes) {
                        let entry = if norm_key.is_empty() {
                            val.get(model)
                                .or_else(|| val.get("pricing").and_then(|p| p.get(model)))
                        } else {
                            val.get(&namespaced_key)
                                .or_else(|| val.get("pricing").and_then(|p| p.get(&namespaced_key)))
                        };
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
    })
}

pub fn register_model_pricing(
    root: &Path,
    model: &str,
    prompt_millicents_per_1k: i64,
    completion_millicents_per_1k: i64,
) -> Result<Value> {
    register_namespaced_model_pricing(root, "", model, prompt_millicents_per_1k, completion_millicents_per_1k)
}

pub fn resolve_model_pricing(root: &Path, model: &str) -> Result<Option<(i64, i64)>> {
    resolve_namespaced_model_pricing(root, "", model)
}

/// Construct an AgentRoutingPolicy for an unrouted request, binding to the provider's protocol and endpoint.
/// Fails closed if pricing cannot be resolved for the model.
pub fn default_routing_policy_for_request(
    root: &Path,
    role_id: &str,
    request: &ChatCompletionRequest,
    gateway: &dyn ProviderGateway,
) -> Result<crate::contracts::AgentRoutingPolicy> {
    let (prompt_rate, comp_rate) = resolve_namespaced_model_pricing(root, gateway.endpoint_url(), &request.model)?
        .or_else(|| if gateway.endpoint_url().is_empty() || gateway.is_offline_fixture() { resolve_model_pricing(root, &request.model).ok().flatten() } else { None })
        .ok_or_else(|| format!("model '{}' has no registered pricing for endpoint '{}'; refusing unpriced dispatch", request.model, gateway.endpoint_url()))?;

    let raw_endpoint = gateway.endpoint_url();
    let endpoint = if raw_endpoint.is_empty() {
        "fixture://default_gateway".to_string()
    } else {
        normalize_endpoint(raw_endpoint)
    };
    let raw_proto = gateway.adapter_protocol();
    let adapter_protocol = if raw_proto.is_empty() {
        "scripted_fixture"
    } else {
        raw_proto
    };

    let mut providers = std::collections::BTreeMap::new();
    providers.insert("default_provider".into(), crate::contracts::ProviderProfile {
        adapter_protocol: adapter_protocol.into(),
        endpoint: endpoint.clone(),
        secret_ref: None,
        permitted_data_disclosure: "redacted_only".into(),
        timeout_seconds: 60,
        max_retries: 2,
    });
    let mut models = std::collections::BTreeMap::new();
    models.insert("default_model_profile".into(), crate::contracts::ModelProfile {
        provider_ref: "default_provider".into(),
        model_id: request.model.clone(),
        context_window: 128_000,
        max_output_tokens: 8192,
        prompt_millicents_per_1k: prompt_rate,
        completion_millicents_per_1k: comp_rate,
        supports_tools: true,
    });
    let mut roles = std::collections::BTreeMap::new();
    roles.insert(role_id.to_string(), crate::contracts::RoleRouting {
        model_profile_ref: "default_model_profile".into(),
        escalation_profile_ref: None,
        max_spend_millicents: None,
        temperature: request.temperature,
    });
    Ok(crate::contracts::AgentRoutingPolicy {
        schema_version: "mlrescurix.agent_routing.v1".into(),
        default_worker_profile_ref: "default_model_profile".into(),
        providers,
        models,
        roles,
    })
}

/// Local operator inspection; never sent to a provider by this function.
pub fn inspect_outbound_view(root: &Path, project: &str, digest: &str) -> Result<Value> {
    crate::execution::migrate(root)?;
    let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let (raw, actor, approved, expires): (String, String, bool, i64) = db.query_row(
        "SELECT view_json,actor,approved,expires_ms FROM outbound_views WHERE project=?1 AND request_sha256=?2",
        rusqlite::params![project, digest], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))).map_err(|e| e.to_string())?;
    use sha2::{Digest, Sha256};
    if format!("{:x}", Sha256::digest(raw.as_bytes())) != digest { return Err("outbound view digest mismatch".into()); }
    Ok(serde_json::json!({"project":project,"request_sha256":digest,"view":serde_json::from_str::<Value>(&raw).map_err(|e|e.to_string())?,
        "actor":actor,"approved":approved,"expires_ms":expires}))
}

pub fn approve_outbound_view(
    root: &Path,
    project_id: &str,
    request_sha256: &str,
    actor: &str,
    approved: bool,
    ttl_ms: i64,
) -> Result<Value> {
    if approved && matches!(actor, "controller" | "scientific-controller" | "scientific-agent") {
        return Err("controller actors cannot authorize their own outbound upload; independent operator approval required".into());
    }
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

fn abort_gateway_receipt(root: &Path, project: &str, task: &str, request_id: &str) -> Result<()> {
    if let Ok(db) = rusqlite::Connection::open(root.join("state.sqlite")) {
        let _ = db.execute(
            "DELETE FROM gateway_dispatch_receipts WHERE project=?1 AND task_id=?2 AND request_id=?3 AND status='awaiting_response'",
            rusqlite::params![project, task, request_id],
        );
    }
    Ok(())
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
    let request_id = format!("ephemeral-{}-{nonce}", std::process::id());
    let policy = default_routing_policy_for_request(root, role_id, request, gateway)?;
    dispatch_gateway_with_routing(
        root,
        project_id,
        task_id,
        role_id,
        &request_id,
        request,
        &policy,
        gateway,
        require_policy_approval,
    )
}

/// Dispatch with caller-owned idempotency identity. A caller must reuse the same
/// ID when recovering an interrupted turn and choose a new ID for every new turn.
fn campaign_nanodollars(value: &str) -> Result<u64> {
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) || fraction.len() > 9 || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return Err("invalid campaign dollar precision".into());
    }
    let whole: u64 = whole.parse().map_err(|_| "invalid campaign dollars")?;
    let fraction: u64 = format!("{fraction:0<9}").parse().map_err(|_| "invalid campaign fraction")?;
    whole.checked_mul(1_000_000_000).and_then(|w| w.checked_add(fraction)).ok_or("campaign amount overflow".into())
}

/// The JSON campaign ledger is the sole dollar authority. Hold its POSIX lock
/// through receipt fencing and transport; settlement is idempotent in its owner.
/// A crash leaves the reservation committed, never a free retry allowance.
fn campaign_dispatch_guard_at(path: &Path, run: &str, request_id: &str, digest: &str, root: &Path, required_nanodollars: u64, scope: &str) -> Result<std::fs::File> {
    use std::io::Write;
    use std::os::fd::AsRawFd;
    let lock = std::fs::OpenOptions::new().create(true).append(true).open(path.with_extension("lock")).map_err(|e| e.to_string())?;
    // SAFETY: flock operates on this owned descriptor; closing releases the lock.
    if unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err("cannot lock campaign ledger".into());
    }
    let mut data: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let amount = |v: &Value| -> Result<u64> { campaign_nanodollars(v.as_str().ok_or("campaign dollar amount must be a decimal string")?) };
    let committed = amount(&data["total_billed_usd"])?
        .checked_add(amount(&data["total_reserved_usd"])?).and_then(|v| v.checked_add(amount(&data["total_uncertain_usd"]).ok()?)).ok_or("campaign totals overflow")?;
    let ceiling = amount(&data["campaign_ceiling_usd"])?;
    if committed > ceiling { return Err("campaign ceiling violated".into()); }
    let reservation = &mut data["active_reservations"][request_id];
    if reservation["run_id"].as_str() != Some(run) || amount(&reservation["amount_usd"])? < required_nanodollars {
        return Err("matching active campaign reservation required".into());
    }
    let root_identity = root.canonicalize().map_err(|e| e.to_string())?.to_string_lossy().to_string();
    if let Some(old) = reservation["gateway_digest"].as_str() {
        if old != digest || reservation["gateway_state_root"].as_str() != Some(root_identity.as_str()) || reservation["gateway_scope"].as_str() != Some(scope) {
            return Err("campaign request already bound to different content or receipt store".into());
        }
    }
    reservation["gateway_scope"] = Value::String(scope.into());
    reservation["gateway_digest"] = Value::String(digest.into());
    reservation["gateway_state_root"] = Value::String(root_identity);
    let tmp = path.with_extension(format!("gateway-tmp-{}", std::process::id()));
    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    file.write_all(&serde_json::to_vec_pretty(&data).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(tmp, &path).map_err(|e| e.to_string())?;
    std::fs::File::open(path.parent().ok_or("ledger parent missing")?).and_then(|f| f.sync_all()).map_err(|e| e.to_string())?;
    Ok(lock)
}

pub fn accounting_request_key(project_id: &str, task_id: &str, request_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let identity = format!("{project_id}\0{task_id}\0{request_id}");
    format!("gw-{}", Sha256::digest(identity.as_bytes()).iter().map(|b| format!("{b:02x}")).collect::<String>())
}

/// Internal transport dispatch implementation. Public callers must route through
/// `dispatch_gateway_with_routing` or `dispatch_gateway_with_id`.
fn dispatch_gateway_with_id_inner(
    root: &Path,
    project_id: &str,
    task_id: &str,
    role_id: &str,
    request_id: &str,
    request: &ChatCompletionRequest,
    gateway: &dyn ProviderGateway,
    require_policy_approval: bool,
    spend_reservation: Option<(&TokenUsage, i64, Option<i64>)>,
) -> Result<Value> {
    use sha2::{Digest, Sha256};
    crate::execution::migrate(root)?;
    let sanitized = OutboundFilter::sanitize_request(request);
    let sanitized_bytes = serde_json::to_vec(&sanitized).map_err(|e| e.to_string())?;
    let digest = format!("{:x}", Sha256::digest(&sanitized_bytes));

    // Refuse before transport if the campaign has no registered ceiling for the
    // selected model. The Python campaign authority has already reserved against
    // the same verified ceiling before this call.
    let rates = resolve_namespaced_model_pricing(root, gateway.endpoint_url(), &sanitized.model)?
        .or_else(|| if gateway.endpoint_url().is_empty() || gateway.is_offline_fixture() { resolve_model_pricing(root, &sanitized.model).ok().flatten() } else { None })
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

    let _campaign_lock = if !gateway.is_offline_fixture() {
        let provider = sanitized.provider.as_ref().ok_or("explicit provider routing configuration required for live transport")?;
        validate_provider_price_cap(provider, rates)?;
        let input_bound = u32::try_from(serde_json::to_vec(&sanitized.messages).map_err(|e| e.to_string())?.len() + 4096).map_err(|_| "request too large")?;
        let maximum = TokenUsage { prompt_tokens: input_bound, completion_tokens: sanitized.max_tokens.ok_or("explicit max_tokens required")?, total_tokens: 0 };
        let required = u64::try_from(cost_from_pricing(&maximum, rates)?).map_err(|_| "negative pricing")?.checked_mul(10_000).ok_or("reservation overflow")?;
        let (ledger, run) = gateway.campaign_budget_context()?;
        Some(campaign_dispatch_guard_at(&ledger, &run, request_id, &digest, root, required.max(1), &format!("{project_id}\0{task_id}"))?)
    } else { None }; // Scripted offline fixtures have no provider charges.
    let cached_response = reserve_gateway_receipt(root, project_id, task_id, request_id, &digest)?;
    let replayed = cached_response.is_some();
    let request_key = accounting_request_key(project_id, task_id, request_id);
    let resp = match cached_response {
        Some(response) => response,
        None => {
            if let Some((usage, est_cost, max_spend)) = spend_reservation {
                if let Err(e) = reserve_role_spend(
                    root,
                    project_id,
                    &request_key,
                    task_id,
                    role_id,
                    &sanitized.model,
                    usage,
                    est_cost,
                    max_spend,
                ) {
                    let _ = abort_gateway_receipt(root, project_id, task_id, request_id);
                    return Err(e);
                }
            }
            match gateway.chat_completion(&sanitized) {
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
            }
        }
    };

    let cost_millicents = cost_from_pricing(&resp.usage, rates)?;
    let cost_usd = cost_millicents as f64 / 100_000.0;

    if !replayed {
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
    }

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

/// Public dispatch entry point with explicit request ID. Routes authoritatively
/// through the routing policy to prevent unrouted direct bypasses.
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
    let policy = default_routing_policy_for_request(root, role_id, request, gateway)?;
    dispatch_gateway_with_routing(
        root,
        project_id,
        task_id,
        role_id,
        request_id,
        request,
        &policy,
        gateway,
        require_policy_approval,
    )
}

pub fn query_role_spend(root: &Path, project_id: &str, role_id: &str) -> Result<u64> {
    crate::execution::migrate(root)?;
    let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let spend: i64 = db.query_row(
        "SELECT COALESCE(SUM(cost_millicents), 0) FROM provider_accounting WHERE project=?1 AND role_id=?2 AND status IN ('ok', 'reserved', 'outcome_unknown')",
        rusqlite::params![project_id, role_id],
        |r| r.get(0),
    ).map_err(|e| format!("cannot query role spend: {e}"))?;
    Ok(spend.max(0) as u64)
}

pub fn reserve_role_spend(
    root: &Path,
    project_id: &str,
    request_id: &str,
    task_id: &str,
    role_id: &str,
    model: &str,
    estimated_tokens: &TokenUsage,
    estimated_cost: i64,
    max_spend_millicents: Option<i64>,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let mut connection = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    connection.busy_timeout(Duration::from_secs(10)).map_err(|e| e.to_string())?;
    let tx = connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    ensure_project_record(&tx, project_id)?;

    use rusqlite::OptionalExtension;
    let existing_status: Option<String> = tx.query_row(
        "SELECT status FROM provider_accounting WHERE project=?1 AND request_id=?2",
        rusqlite::params![project_id, request_id],
        |r| r.get(0),
    ).optional().map_err(|e| e.to_string())?;

    if let Some(ref st) = existing_status {
        if st == "ok" || st == "outcome_unknown" {
            return Err(format!("request '{request_id}' has already been dispatched with status '{st}'; duplicate reservation rejected"));
        }
    }

    if let Some(max_spend) = max_spend_millicents {
        let max_spend_u64 = max_spend.max(0) as u64;
        let current_reserved_and_spent: i64 = tx.query_row(
            "SELECT COALESCE(SUM(cost_millicents), 0) FROM provider_accounting WHERE project=?1 AND role_id=?2 AND status IN ('ok', 'reserved', 'outcome_unknown') AND request_id != ?3",
            rusqlite::params![project_id, role_id, request_id],
            |r| r.get(0),
        ).map_err(|e| format!("cannot query role spend: {e}"))?;
        let current_u64 = current_reserved_and_spent.max(0) as u64;
        let candidate_total = current_u64.saturating_add(estimated_cost.max(0) as u64);
        if candidate_total > max_spend_u64 {
            return Err(format!(
                "role '{role_id}' spend limit exceeded: current {current_u64} + estimated {estimated_cost} > max {max_spend}"
            ));
        }
    }

    crate::execution::enforce_state_write_admission(
        root,
        Some(&tx),
        (model.len() + request_id.len() + task_id.len() + role_id.len()) as u64 * 4 + 16 * 1024,
        crate::execution::StorageWriteKind::ProviderReceipt,
        Some(project_id),
    )?;

    tx.execute(
        "INSERT INTO provider_accounting(project, request_id, task_id, role_id, model, prompt_tokens, completion_tokens, cost_millicents, latency_ms, status, created_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, 'reserved', ?9)
         ON CONFLICT(project, request_id) DO UPDATE SET
           prompt_tokens=excluded.prompt_tokens,
           completion_tokens=excluded.completion_tokens,
           cost_millicents=excluded.cost_millicents,
           status='reserved'
         WHERE provider_accounting.status = 'reserved'",
        rusqlite::params![
            project_id,
            request_id,
            task_id,
            role_id,
            model,
            estimated_tokens.prompt_tokens,
            estimated_tokens.completion_tokens,
            estimated_cost,
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;

    tx.commit().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn dispatch_gateway_with_routing(
    root: &Path,
    project_id: &str,
    task_id: &str,
    role_id: &str,
    request_id: &str,
    request: &ChatCompletionRequest,
    routing_policy: &crate::contracts::AgentRoutingPolicy,
    gateway: &dyn ProviderGateway,
    require_policy_approval: bool,
) -> Result<Value> {
    routing_policy.validate()?;
    let (model_profile, provider_profile) = routing_policy.resolve_role(role_id)?;

    // 1. Check protocol and transport matching for non-fixture gateways
    if !gateway.is_offline_fixture() && !gateway.endpoint_url().is_empty() {
        if gateway.adapter_protocol() != provider_profile.adapter_protocol {
            return Err(format!(
                "adapter protocol mismatch: gateway protocol '{}' does not match resolved provider profile '{}'",
                gateway.adapter_protocol(), provider_profile.adapter_protocol
            ));
        }
        let configured_endpoint = normalize_endpoint(&provider_profile.endpoint);
        let gateway_ep = normalize_endpoint(gateway.endpoint_url());
        if gateway_ep != configured_endpoint {
            return Err(format!(
                "endpoint mismatch: gateway endpoint '{}' does not match resolved provider profile '{}'",
                gateway.endpoint_url(), provider_profile.endpoint
            ));
        }
    }

    // 2. Check capabilities
    if !request.tools.is_empty() && !model_profile.supports_tools {
        return Err(format!("model '{}' does not support requested tool calling", model_profile.model_id));
    }

    // 3. Check disclosure policy
    if provider_profile.permitted_data_disclosure == "none" && !request.messages.is_empty() {
        return Err(format!("provider profile for role '{}' prohibits outbound data disclosure (permitted_data_disclosure: none)", role_id));
    }
    if provider_profile.permitted_data_disclosure == "isolated_local" {
        let ep = &provider_profile.endpoint;
        // Compare the complete HTTP authority, not a prefix: localhost.evil and
        // localhost@evil are remote hosts. Reject unrecognized forms closed.
        let authority = ep.strip_prefix("http://")
            .and_then(|rest| rest.split(['/', '?', '#']).next());
        let is_local = authority.is_some_and(|host| {
            let (name, port) = if let Some(rest) = host.strip_prefix("[::1]") {
                ("[::1]", rest)
            } else if let Some(rest) = host.strip_prefix("localhost") {
                ("localhost", rest)
            } else if let Some(rest) = host.strip_prefix("127.0.0.1") {
                ("127.0.0.1", rest)
            } else {
                return false;
            };
            !name.is_empty() && (port.is_empty() || port.strip_prefix(':').is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())))
        });
        if !is_local && !gateway.is_offline_fixture() {
            return Err(format!(
                "provider profile for role '{}' has disclosure policy 'isolated_local', prohibiting outbound transmission to non-local endpoint '{}'",
                role_id, ep
            ));
        }
    }

    // 4. Construct adjusted request with the resolved model
    let mut routed_request = request.clone();
    routed_request.model = model_profile.model_id.clone();
    if routed_request.temperature.is_none() {
        if let Some(role_cfg) = routing_policy.roles.get(role_id) {
            routed_request.temperature = role_cfg.temperature;
        }
    }
    if routed_request.max_tokens.is_none() {
        routed_request.max_tokens = Some(model_profile.max_output_tokens);
    }

    if provider_profile.permitted_data_disclosure == "redacted_only" {
        for msg in &mut routed_request.messages {
            if let Some(content) = &msg.content {
                let sanitized = OutboundFilter::sanitize_content(content);
                if content.contains("__mlr_row_id") || (content.lines().count() > 5 && content.contains(',')) {
                    if !sanitized.contains("[REDACTED_ROW_DATA") {
                        return Err(format!("request for role '{role_id}' contains unredacted tabular row data prohibited under disclosure policy 'redacted_only'"));
                    }
                }
                msg.content = Some(sanitized);
            }
        }
    }

    // 5. Conservative token admission: UTF-8 byte upper bound + framing across entire request
    let input_bytes = serde_json::to_vec(&routed_request).map_err(|e| e.to_string())?.len();
    let estimated_prompt_tokens = (input_bytes as u32).saturating_add(256);
    let completion_tokens = routed_request.max_tokens.unwrap_or(model_profile.max_output_tokens);
    if estimated_prompt_tokens.saturating_add(completion_tokens) > model_profile.context_window {
        return Err(format!(
            "request estimated tokens ({} prompt + {} completion = {}) exceeds model context window ({})",
            estimated_prompt_tokens, completion_tokens, estimated_prompt_tokens.saturating_add(completion_tokens), model_profile.context_window
        ));
    }

    // 6. Check and atomically reserve role spend
    let conservative_usage = TokenUsage {
        prompt_tokens: estimated_prompt_tokens,
        completion_tokens,
        total_tokens: estimated_prompt_tokens.saturating_add(completion_tokens),
    };
    let rates = (model_profile.prompt_millicents_per_1k as i64, model_profile.completion_millicents_per_1k as i64);
    let est_cost = cost_from_pricing(&conservative_usage, rates)?;
    let max_spend = routing_policy.roles.get(role_id).and_then(|r| r.max_spend_millicents);
    let request_key = accounting_request_key(project_id, task_id, request_id);

    // 7. Register model pricing (fail-closed on error, namespaced by endpoint)
    register_namespaced_model_pricing(
        root,
        &provider_profile.endpoint,
        &model_profile.model_id,
        model_profile.prompt_millicents_per_1k,
        model_profile.completion_millicents_per_1k,
    )?;
    if gateway.is_offline_fixture() && gateway.endpoint_url() != provider_profile.endpoint {
        register_namespaced_model_pricing(
            root,
            gateway.endpoint_url(),
            &model_profile.model_id,
            model_profile.prompt_millicents_per_1k,
            model_profile.completion_millicents_per_1k,
        )?;
    }

    // 8. If provider routing configuration is present on request, validate binding to policy; populate authoritative provider configuration
    let max_prompt_usd = (model_profile.prompt_millicents_per_1k as f64 / 100.0) / 1.05;
    let max_comp_usd = (model_profile.completion_millicents_per_1k as f64 / 100.0) / 1.05;
    if let Some(user_prov) = &routed_request.provider {
        if let Some(order) = user_prov.get("order").and_then(|o| o.as_array()) {
            for ep in order {
                if let Some(ep_str) = ep.as_str() {
                    let norm_order_ep = normalize_endpoint(ep_str);
                    let norm_policy_ep = normalize_endpoint(&provider_profile.endpoint);
                    let norm_gw_ep = normalize_endpoint(gateway.endpoint_url());
                    if (ep_str.starts_with("http://") || ep_str.starts_with("https://"))
                        && norm_order_ep != norm_policy_ep && norm_order_ep != norm_gw_ep
                    {
                        return Err(format!(
                            "caller-supplied provider order contains unauthorized endpoint '{ep_str}'; must match resolved policy endpoint '{}'",
                            provider_profile.endpoint
                        ));
                    }
                }
            }
        }
        if let Some(only) = user_prov.get("only").and_then(|o| o.as_array()) {
            for ep in only {
                if let Some(ep_str) = ep.as_str() {
                    let norm_only_ep = normalize_endpoint(ep_str);
                    let norm_policy_ep = normalize_endpoint(&provider_profile.endpoint);
                    let norm_gw_ep = normalize_endpoint(gateway.endpoint_url());
                    if (ep_str.starts_with("http://") || ep_str.starts_with("https://"))
                        && norm_only_ep != norm_policy_ep && norm_only_ep != norm_gw_ep
                    {
                        return Err(format!(
                            "caller-supplied provider 'only' contains unauthorized endpoint '{ep_str}'; must match resolved policy endpoint '{}'",
                            provider_profile.endpoint
                        ));
                    }
                }
            }
        }
        if user_prov.get("allow_fallbacks").and_then(|f| f.as_bool()) == Some(true) {
            return Err("caller-supplied provider routing cannot enable fallbacks; policy disallows unrouted fallbacks".into());
        }
        if let Some(max_price) = user_prov.get("max_price") {
            if let Some(p_usd) = max_price.get("prompt").and_then(|p| p.as_f64()) {
                if p_usd > max_prompt_usd + 1e-6 {
                    return Err(format!(
                        "caller-supplied prompt price cap {p_usd} exceeds policy price cap {max_prompt_usd}"
                    ));
                }
            }
            if let Some(c_usd) = max_price.get("completion").and_then(|c| c.as_f64()) {
                if c_usd > max_comp_usd + 1e-6 {
                    return Err(format!(
                        "caller-supplied completion price cap {c_usd} exceeds policy price cap {max_comp_usd}"
                    ));
                }
            }
        }
    }
    if !gateway.is_offline_fixture() {
        if routed_request.provider.is_none() {
            routed_request.provider = Some(serde_json::json!({
                "order": [&provider_profile.endpoint],
                "allow_fallbacks": false,
                "max_price": {
                    "prompt": max_prompt_usd,
                    "completion": max_comp_usd
                }
            }));
        } else if let Some(p) = routed_request.provider.as_mut().and_then(|v| v.as_object_mut()) {
            p.insert("allow_fallbacks".into(), serde_json::Value::Bool(false));
            if !p.contains_key("max_price") {
                p.insert("max_price".into(), serde_json::json!({
                    "prompt": max_prompt_usd,
                    "completion": max_comp_usd
                }));
            }
        }
    }

    let dispatch_res = dispatch_gateway_with_id_inner(
        root,
        project_id,
        task_id,
        role_id,
        request_id,
        &routed_request,
        gateway,
        require_policy_approval,
        Some((&conservative_usage, est_cost, max_spend)),
    );

    let mut res = match dispatch_res {
        Ok(r) => r,
        Err(err) => {
            // Check status in gateway_dispatch_receipts to distinguish pre-dispatch failure vs uncertain remote outcome
            if let Ok(db) = rusqlite::Connection::open(root.join("state.sqlite")) {
                let receipt_status: Option<String> = db.query_row(
                    "SELECT status FROM gateway_dispatch_receipts WHERE project=?1 AND task_id=?2 AND request_id=?3",
                    rusqlite::params![project_id, task_id, request_id],
                    |r| r.get(0),
                ).ok();

                if receipt_status.as_deref() == Some("outcome_unknown") {
                    // Ambiguous / uncertain remote outcome: preserve reservation as durable outcome_unknown so spend cannot be exceeded
                    let _ = db.execute(
                        "UPDATE provider_accounting SET status='outcome_unknown' WHERE project=?1 AND request_id=?2 AND status='reserved'",
                        rusqlite::params![project_id, &request_key],
                    );
                }
                // Other states may represent an in-flight or completed remote call,
                // including a failed receipt write. Preserve any reservation until
                // a trusted recovery procedure can establish the outcome.

            }
            return Err(err);
        }
    };

    if let Some(obj) = res.as_object_mut() {
        obj.insert("routed_role".into(), Value::String(role_id.into()));
        obj.insert("resolved_model_id".into(), Value::String(model_profile.model_id.clone()));
        obj.insert("resolved_provider_protocol".into(), Value::String(provider_profile.adapter_protocol.clone()));
        obj.insert("resolved_endpoint".into(), Value::String(provider_profile.endpoint.clone()));
    }

    Ok(res)
}

fn validate_provider_price_cap(provider: &Value, rates: (i64, i64)) -> Result<()> {
    // Registered rates are millicents per 1k tokens; OpenRouter caps are USD per 1M.
    for (name, rate) in [("prompt", rates.0), ("completion", rates.1)] {
        let price = provider["max_price"][name].as_f64().ok_or("explicit provider max_price required")?;
        if !price.is_finite() || price < 0.0 || rate < 0 || price * 1.05 > (rate as f64 / 100.0) + 1e-12 {
            return Err("provider max_price exceeds registered reservation rate".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod campaign_admission_tests {
    use super::*;
    #[test]
    fn transport_sends_exact_json_and_auth_over_pipe() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            let mut raw = Vec::new();
            let mut chunk = [0; 4096];
            loop {
                let n = stream.read(&mut chunk).unwrap();
                assert!(n > 0);
                raw.extend_from_slice(&chunk[..n]);
                if let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&raw[..end]);
                    let length: usize = headers.lines().find_map(|line| line.to_ascii_lowercase().strip_prefix("content-length:").map(|s| s.trim().parse().unwrap())).unwrap();
                    if raw.len() >= end + 4 + length {
                        assert!(headers.contains("Authorization: Bearer test-key"));
                        let body: Value = serde_json::from_slice(&raw[end+4..end+4+length]).unwrap();
                        assert_eq!(body["messages"][0]["content"], "quotes \" and slash \\");
                        break;
                    }
                }
            }
            let body = r#"{"id":"test","model":"test","choices":[{"message":{"content":"ok"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}"#;
            write!(stream, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body).unwrap();
        });
        let adapter = OpenAICompatibleAdapter::new(format!("http://{address}"), "test-key".into());
        let request = ChatCompletionRequest {model:"test".into(), messages:vec![ChatMessage::user("quotes \" and slash \\")], tools:vec![], temperature:None, max_tokens:Some(3), provider:None, reasoning:None};
        assert_eq!(adapter.chat_completion(&request).unwrap().content.as_deref(), Some("ok"));
        server.join().unwrap();
    }
    #[test]
    fn live_price_caps_must_fit_reservation_rates() {
        assert!(validate_provider_price_cap(&serde_json::json!({"max_price":{"prompt":0.08,"completion":0.55}}), (9,58)).is_ok());
        for p in [serde_json::json!({}), serde_json::json!({"max_price":{"prompt":0.09,"completion":0.55}}), serde_json::json!({"max_price":{"prompt":0.08,"completion":-1}})] {
            assert!(validate_provider_price_cap(&p, (9,58)).is_err());
        }
    }
    #[test]
    fn exact_dollar_precision() {
        assert_eq!(campaign_nanodollars("0.050724020").unwrap(), 50_724_020);
        for invalid in ["NaN", "-1", "0.0000000001", "1e2", "18446744073709551615"] {
            assert!(campaign_nanodollars(invalid).is_err());
        }
    }
    #[test]
    fn reservation_binding_survives_reopen_and_rejects_cross_scope_replay() {
        let dir = tempfile::tempdir().unwrap();
        let ledger = dir.path().join("campaign_budget_ledger.json");
        std::fs::write(&ledger, serde_json::json!({
            "campaign_ceiling_usd":"0.30", "total_billed_usd":"0.050724020",
            "total_reserved_usd":"0.01", "total_uncertain_usd":"0.00",
            "active_reservations":{"r":{"run_id":"run", "amount_usd":"0.01"}}
        }).to_string()).unwrap();
        let acquire = |run: &str, id: &str, digest: &str, scope: &str, cost| campaign_dispatch_guard_at(
            &ledger, run, id, digest, dir.path(), cost, scope);
        assert!(acquire("other", "r", "hash", "p/t", 1).is_err());
        assert!(acquire("run", "missing", "hash", "p/t", 1).is_err());
        assert!(acquire("run", "r", "hash", "p/t", 10_000_001).is_err());
        drop(acquire("run", "r", "hash", "p/t", 10_000_000).unwrap());
        // Crash before/after receipt creation still retains the full reservation.
        drop(acquire("run", "r", "hash", "p/t", 1).unwrap());
        assert!(acquire("run", "r", "changed", "p/t", 1).is_err());
        assert!(acquire("run", "r", "hash", "p/other", 1).is_err());
        let stored: Value = serde_json::from_slice(&std::fs::read(ledger).unwrap()).unwrap();
        assert_eq!(stored["total_billed_usd"], "0.050724020");
        assert_eq!(stored["total_reserved_usd"], "0.01");
    }

    #[test]
    fn multi_provider_role_routing_and_context_bounds() {
        use crate::contracts::{AgentRoutingPolicy, ModelProfile, ProviderProfile, RoleRouting};
        use std::collections::BTreeMap;

        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("state");
        let policy = crate::core::Policy {
            budget: BTreeMap::from([
                ("calls".into(), 100),
                ("tokens".into(), 100_000),
                ("wall_ms".into(), 100_000),
            ]),
            capacity: BTreeMap::from([("workers".into(), 2), ("cpu_threads".into(), 4)]),
            max_tasks: 1024,
        };
        let _runtime = crate::storage::Runtime::init(&root, policy).unwrap();

        let mut providers = BTreeMap::new();
        providers.insert(
            "prov_high_tier".into(),
            ProviderProfile {
                adapter_protocol: "openai_compatible".into(),
                endpoint: "https://api.high-tier.com/v1".into(),
                secret_ref: None,
                permitted_data_disclosure: "redacted_only".into(),
                timeout_seconds: 60,
                max_retries: 2,
            },
        );
        providers.insert(
            "prov_local_tier".into(),
            ProviderProfile {
                adapter_protocol: "native_curl".into(),
                endpoint: "http://127.0.0.1:8000/v1".into(),
                secret_ref: None,
                permitted_data_disclosure: "isolated_local".into(),
                timeout_seconds: 30,
                max_retries: 1,
            },
        );

        let mut models = BTreeMap::new();
        models.insert(
            "profile_strat_owner".into(),
            ModelProfile {
                provider_ref: "prov_high_tier".into(),
                model_id: "claude-or-opus-reasoner-mock".into(),
                context_window: 128_000,
                max_output_tokens: 4096,
                prompt_millicents_per_1k: 300,
                completion_millicents_per_1k: 1500,
                supports_tools: true,
            },
        );
        models.insert(
            "profile_worker_fast".into(),
            ModelProfile {
                provider_ref: "prov_local_tier".into(),
                model_id: "qwen-or-deepseek-fast-mock".into(),
                context_window: 2048, // Small context window for testing
                max_output_tokens: 512,
                prompt_millicents_per_1k: 10,
                completion_millicents_per_1k: 20,
                supports_tools: true,
            },
        );

        let mut roles = BTreeMap::new();
        roles.insert(
            "campaign_owner".into(),
            RoleRouting {
                model_profile_ref: "profile_strat_owner".into(),
                escalation_profile_ref: None,
                max_spend_millicents: None,
                temperature: Some(0.2),
            },
        );
        roles.insert(
            "modeler".into(),
            RoleRouting {
                model_profile_ref: "profile_worker_fast".into(),
                escalation_profile_ref: Some("profile_strat_owner".into()),
                max_spend_millicents: Some(500),
                temperature: Some(0.0),
            },
        );

        let routing = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers.clone(),
            models: models.clone(),
            roles: roles.clone(),
        };
        assert!(routing.validate().is_ok());

        // 1. Resolve role "campaign_owner"
        let (owner_model, owner_prov) = routing.resolve_role("campaign_owner").unwrap();
        assert_eq!(owner_model.model_id, "claude-or-opus-reasoner-mock");
        assert_eq!(owner_prov.adapter_protocol, "openai_compatible");

        // 2. Resolve role "modeler"
        let (worker_model, worker_prov) = routing.resolve_role("modeler").unwrap();
        assert_eq!(worker_model.model_id, "qwen-or-deepseek-fast-mock");
        assert_eq!(worker_prov.adapter_protocol, "native_curl");

        // 3. Fallback to default for unlisted role (e.g. "fe_specialist")
        let (default_model, default_prov) = routing.resolve_role("fe_specialist").unwrap();
        assert_eq!(default_model.model_id, "qwen-or-deepseek-fast-mock");
        assert_eq!(default_prov.adapter_protocol, "native_curl");

        // 4. Test offline dispatch through dispatch_gateway_with_routing
        let fixture = ScriptedProviderFixture::new(vec![ChatCompletionResponse {
            id: "resp-1".into(),
            model: "claude-or-opus-reasoner-mock".into(),
            content: Some("Strategic plan ready".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
            latency_ms: 120,
        }]);

        let req = ChatCompletionRequest {
            model: "any".into(),
            messages: vec![ChatMessage::user("Design ML strategy")],
            tools: vec![],
            temperature: None,
            max_tokens: Some(100),
            provider: None,
            reasoning: None,
        };

        let result = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_strategy",
            "campaign_owner",
            "req_strat_001",
            &req,
            &routing,
            &fixture,
            false,
        ).unwrap();

        assert_eq!(result["status"], "completed");
        assert_eq!(result["routed_role"], "campaign_owner");
        assert_eq!(result["resolved_model_id"], "claude-or-opus-reasoner-mock");
        assert_eq!(result["resolved_provider_protocol"], "openai_compatible");

        // 5. Test context window rejection: worker_fast has context_window=2048, max_tokens=2500 -> rejected!
        let huge_req = ChatCompletionRequest {
            model: "any".into(),
            messages: vec![ChatMessage::user("Do something")],
            tools: vec![],
            temperature: None,
            max_tokens: Some(3000), // exceeds 2048
            provider: None,
            reasoning: None,
        };

        let err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_worker",
            "modeler",
            "req_worker_001",
            &huge_req,
            &routing,
            &fixture,
            false,
        ).unwrap_err();

        assert!(err.contains("exceeds model context window"));

        // 6. Test tool rejection when model profile has supports_tools = false
        let mut models_no_tools = models.clone();
        models_no_tools.get_mut("profile_worker_fast").unwrap().supports_tools = false;
        let routing_no_tools = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers.clone(),
            models: models_no_tools,
            roles: roles.clone(),
        };
        let tool_req = ChatCompletionRequest {
            model: "any".into(),
            messages: vec![ChatMessage::user("Check data")],
            tools: vec![ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "sample_tool".into(),
                    description: "test".into(),
                    parameters: serde_json::json!({}),
                },
            }],
            temperature: None,
            max_tokens: Some(100),
            provider: None,
            reasoning: None,
        };
        let tool_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_tools",
            "modeler",
            "req_tools_001",
            &tool_req,
            &routing_no_tools,
            &fixture,
            false,
        ).unwrap_err();
        assert!(tool_err.contains("does not support requested tool calling"));

        // 7. Test transport protocol and endpoint mismatch checks for live gateway
        let live_adapter = OpenAICompatibleAdapter::new("https://wrong-endpoint.com/v1".into(), "test-key".into());
        let proto_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_proto",
            "modeler", // requires native_curl protocol
            "req_proto_001",
            &req,
            &routing,
            &live_adapter,
            false,
        ).unwrap_err();
        assert!(proto_err.contains("adapter protocol mismatch") || proto_err.contains("endpoint mismatch"));

        // 8. Test role spend limit rejection
        let mut roles_strict = roles.clone();
        roles_strict.get_mut("modeler").unwrap().max_spend_millicents = Some(0); // 0 max spend
        let routing_zero_spend = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers.clone(),
            models: models.clone(),
            roles: roles_strict,
        };
        let spend_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_spend",
            "modeler",
            "req_spend_001",
            &req,
            &routing_zero_spend,
            &fixture,
            false,
        ).unwrap_err();
        assert!(spend_err.contains("spend limit exceeded"));

        // 9. Test disclosure policy: 'none' rejects outbound data before I/O
        let mut providers_no_disclosure = providers.clone();
        providers_no_disclosure.get_mut("prov_high_tier").unwrap().permitted_data_disclosure = "none".into();
        let routing_no_disclosure = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers_no_disclosure,
            models: models.clone(),
            roles: roles.clone(),
        };
        let disc_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_disc",
            "campaign_owner",
            "req_disc_001",
            &req,
            &routing_no_disclosure,
            &fixture,
            false,
        ).unwrap_err();
        assert!(disc_err.contains("prohibits outbound data disclosure"));

        // 10. Test disclosure policy: 'isolated_local' rejects remote endpoint before I/O
        let mut providers_isolated = providers.clone();
        providers_isolated.get_mut("prov_high_tier").unwrap().permitted_data_disclosure = "isolated_local".into();
        let routing_isolated = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers_isolated,
            models: models.clone(),
            roles: roles.clone(),
        };
        let live_remote = OpenAICompatibleAdapter::new("https://api.high-tier.com/v1".into(), "test-key".into());
        let iso_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_iso",
            "campaign_owner",
            "req_iso_001",
            &req,
            &routing_isolated,
            &live_remote,
            false,
        ).unwrap_err();
        assert!(iso_err.contains("prohibiting outbound transmission to non-local endpoint"));

        // Neither a matching prefix nor a deceptive localhost authority permits
        // a live request to be routed to a different host.
        let spoofed_adapter = OpenAICompatibleAdapter::new("https://api.high-tier.com/v1.evil".into(), "test-key".into());
        let err = dispatch_gateway_with_routing(&root, "p1", "task_spoof", "campaign_owner", "req_spoof", &req, &routing, &spoofed_adapter, false).unwrap_err();
        assert!(err.contains("endpoint mismatch"));
        for endpoint in ["http://localhost.evil/v1", "http://localhost@evil/v1", "http://127.0.0.1.evil/v1"] {
            let mut spoofed_policy = routing_isolated.clone();
            spoofed_policy.providers.get_mut("prov_high_tier").unwrap().endpoint = endpoint.into();
            let adapter = OpenAICompatibleAdapter::new(endpoint.into(), "test-key".into());
            let err = dispatch_gateway_with_routing(&root, "p1", "task_spoof", "campaign_owner", "req_spoof", &req, &spoofed_policy, &adapter, false).unwrap_err();
            assert!(err.contains("prohibiting outbound transmission"), "{endpoint}: {err}");
        }

        // 11. Test auto-populated price cap margin fits validate_provider_price_cap
        let prompt_rate = 300i64;
        let comp_rate = 1500i64;
        let max_p = (prompt_rate as f64 / 100.0) / 1.05;
        let max_c = (comp_rate as f64 / 100.0) / 1.05;
        let auto_provider = serde_json::json!({
            "max_price": {
                "prompt": max_p,
                "completion": max_c
            }
        });
        assert!(validate_provider_price_cap(&auto_provider, (prompt_rate, comp_rate)).is_ok());

        // 12. Test tool schemas are included in context token bound
        let mut tools_req = req.clone();
        // Add a tool with a large parameters description to test bounding
        let big_params = serde_json::json!({
            "type": "object",
            "properties": {
                "features": { "type": "string", "description": "x".repeat(3000) }
            }
        });
        tools_req.tools = vec![ToolDefinition {
            tool_type: "function".into(),
            function: ToolFunction {
                name: "process_data".into(),
                description: "Process data function".into(),
                parameters: big_params,
            },
        }];
        let mut tight_model = models.clone();
        // Set context window to 2000 (smaller than big_params)
        tight_model.get_mut("profile_worker_fast").unwrap().context_window = 2000;
        let routing_tight = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers.clone(),
            models: tight_model,
            roles: roles.clone(),
        };
        let ctx_err = dispatch_gateway_with_routing(
            &root,
            "p1",
            "task_ctx_tools",
            "modeler",
            "req_tools_ctx",
            &tools_req,
            &routing_tight,
            &fixture,
            false,
        ).unwrap_err();
        assert!(ctx_err.contains("exceeds model context window"));

        // 13. Test atomic concurrent reservation: concurrent requests cannot exceed max_spend
        let mut roles_cap = roles.clone();
        // Modeler prompt rate is 10, completion is 20.
        // est_tokens prompt ~ 380, completion 100 -> cost ~ 6 millicents. Actual fixture cost ~ 2 millicents.
        // Set max_spend to 7 millicents: exactly one request can proceed (6 <= 7), the other (6+6=12 or 2+6=8 > 7) must be rejected.
        roles_cap.get_mut("modeler").unwrap().max_spend_millicents = Some(7);
        let routing_concurrency = AgentRoutingPolicy {
            schema_version: "mlrescurix.agent_routing.v1".into(),
            default_worker_profile_ref: "profile_worker_fast".into(),
            providers: providers.clone(),
            models: models.clone(),
            roles: roles_cap,
        };
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let root_c = root.clone();
        let routing_c = routing_concurrency.clone();
        let fixture_c1 = ScriptedProviderFixture::new(vec![ChatCompletionResponse {
            id: "resp-c1".into(),
            model: "qwen-or-deepseek-fast-mock".into(),
            content: Some("code 1".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
            latency_ms: 10,
        }]);
        let fixture_c2 = ScriptedProviderFixture::new(vec![ChatCompletionResponse {
            id: "resp-c2".into(),
            model: "qwen-or-deepseek-fast-mock".into(),
            content: Some("code 2".into()),
            tool_calls: vec![],
            usage: TokenUsage { prompt_tokens: 100, completion_tokens: 50, total_tokens: 150 },
            latency_ms: 10,
        }]);

        let b1 = barrier.clone();
        let r1 = req.clone();
        let rt1 = root_c.clone();
        let rc1 = routing_c.clone();
        let h1 = std::thread::spawn(move || {
            b1.wait();
            dispatch_gateway_with_routing(&rt1, "p1", "task_concurrent", "modeler", "req_conc_1", &r1, &rc1, &fixture_c1, false)
        });

        let b2 = barrier.clone();
        let r2 = req.clone();
        let rt2 = root_c.clone();
        let rc2 = routing_c.clone();
        let h2 = std::thread::spawn(move || {
            b2.wait();
            dispatch_gateway_with_routing(&rt2, "p1", "task_concurrent", "modeler", "req_conc_2", &r2, &rc2, &fixture_c2, false)
        });

        let res1 = h1.join().unwrap();
        let res2 = h2.join().unwrap();
        let successes = [res1.is_ok(), res2.is_ok()].iter().filter(|&&ok| ok).count();
        let errors = [res1.is_err(), res2.is_err()].iter().filter(|&&err| err).count();
        assert_eq!(successes, 1, "exactly one concurrent dispatch should succeed under tight role ceiling");
        assert_eq!(errors, 1, "the other concurrent dispatch must be rejected by atomic reservation");
    }
}
