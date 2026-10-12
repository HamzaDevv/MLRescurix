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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
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

impl OutboundFilter {
    /// Redact raw data rows and truncate unbounded context items to prevent leaking private row data to providers.
    pub fn sanitize_content(text: &str) -> String {
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() <= 6 {
            return text.to_string();
        }

        // Detect if content looks like a multi-line CSV or raw table
        let mut row_count = 0;
        let mut cleaned_lines: Vec<String> = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            let comma_count = line.matches(',').count();
            if comma_count >= 2 && idx > 0 {
                row_count += 1;
                if row_count <= 3 {
                    cleaned_lines.push(line.to_string());
                }
            } else {
                cleaned_lines.push(line.to_string());
            }
        }

        if row_count > 3 {
            cleaned_lines.push(format!("[REDACTED_ROW_DATA: {} raw tabular rows omitted; use aggregate diagnostics]", row_count - 3));
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
        }
    }
}

pub trait ProviderGateway: Send + Sync {
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
    let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.execute(
        "INSERT INTO provider_accounting(project, request_id, task_id, role_id, model, prompt_tokens, completion_tokens, cost_millicents, latency_ms, status, created_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
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

    // M0 action settlement owns charges; this table is supplemental telemetry.
    Ok(())
}
