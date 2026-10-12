use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agents::{
        context::AgentContextManager,
        gateway::{ChatMessage, ChatCompletionRequest, ProviderGateway, record_provider_accounting},
        roles::get_role,
        tools::{ToolDescriptor, ToolRegistry},
        workspace::BranchWorkspaceManager,
    },
    Result,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    pub project_id: String,
    pub task_id: String,
    pub role_id: String,
    pub branch_id: String,
    pub model: String,
    pub max_turns: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnExecutionResult {
    pub turn_index: u32,
    pub tool_calls_executed: u32,
    pub assistant_content: Option<String>,
    pub stopped: bool,
    pub stop_reason: Option<String>,
}

pub struct AgentOrchestrator {
    pub root: PathBuf,
    pub config: OrchestratorConfig,
    pub context: AgentContextManager,
    pub gateway: Arc<dyn ProviderGateway>,
    pub current_turn: u32,
}

impl AgentOrchestrator {
    pub fn new(
        root: &Path,
        config: OrchestratorConfig,
        gateway: Arc<dyn ProviderGateway>,
    ) -> Result<Self> {
        let role_spec = get_role(&config.role_id)
            .ok_or_else(|| format!("Unknown role: {}", config.role_id))?;

        let context = AgentContextManager::new(
            &config.project_id,
            &config.task_id,
            &config.role_id,
            role_spec.system_prompt,
        );

        BranchWorkspaceManager::init_branch(root, &config.project_id, &config.branch_id, "main")?;

        Ok(Self {
            root: root.to_path_buf(),
            config,
            context,
            gateway,
            current_turn: 0,
        })
    }

    /// Check whether research admission is permanently closed for this project.
    pub fn is_research_closed(&self) -> Result<bool> {
        let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let gate_count: i64 = db.query_row(
            "SELECT count(*) FROM final_test_gates WHERE project=?1 AND status IN ('reserved', 'consumed')",
            [&self.config.project_id],
            |r| r.get(0),
        ).unwrap_or(0);
        let proj_state: Option<String> = db.query_row(
            "SELECT state FROM projects WHERE id=?1",
            [&self.config.project_id],
            |r| r.get(0),
        ).ok();

        let is_finalizing_or_completed = matches!(proj_state.as_deref(), Some("finalizing") | Some("completed"));
        Ok(gate_count > 0 || is_finalizing_or_completed)
    }

    /// Execute a single turn boundary iteration.
    pub fn step(&mut self) -> Result<TurnExecutionResult> {
        if self.current_turn >= self.config.max_turns {
            return Ok(TurnExecutionResult {
                turn_index: self.current_turn,
                tool_calls_executed: 0,
                assistant_content: Some("Max turns reached".into()),
                stopped: true,
                stop_reason: Some("max_turns_exhausted".into()),
            });
        }

        self.current_turn += 1;
        let tools = ToolRegistry::get_trusted_tools();

        let req = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: self.context.messages.clone(),
            tools,
            temperature: Some(0.2),
            max_tokens: Some(4096),
        };

        let response = self.gateway.chat_completion(&req)?;

        // Record provider accounting
        let _ = record_provider_accounting(
            &self.root,
            &self.config.project_id,
            &response.id,
            &self.config.task_id,
            &self.config.role_id,
            &response.model,
            &response.usage,
            (response.usage.total_tokens as i64) * 2, // estimated 2 millicents per token
            response.latency_ms,
            "success",
        );

        if response.tool_calls.is_empty() {
            // Assistant produced a textual response without tools
            self.context.add_message(ChatMessage::assistant(response.content.clone(), None));
            let stopped = response.content.as_deref().map(|c| c.contains("[FINAL]") || c.contains("[STOP]")).unwrap_or(false);
            return Ok(TurnExecutionResult {
                turn_index: self.current_turn,
                tool_calls_executed: 0,
                assistant_content: response.content,
                stopped,
                stop_reason: if stopped { Some("agent_decided_stop".into()) } else { None },
            });
        }

        // Add assistant tool calls to context
        self.context.add_message(ChatMessage::assistant(response.content.clone(), Some(response.tool_calls.clone())));

        let mut tool_count = 0;
        for call in &response.tool_calls {
            tool_count += 1;
            let result_str = self.dispatch_tool_call(&call.function.name, &call.function.arguments)?;
            self.context.add_message(ChatMessage::tool(result_str, &call.id, &call.function.name));
        }

        self.context.maybe_compact();

        Ok(TurnExecutionResult {
            turn_index: self.current_turn,
            tool_calls_executed: tool_count,
            assistant_content: response.content,
            stopped: false,
            stop_reason: None,
        })
    }

    fn dispatch_tool_call(&mut self, name: &str, arguments_json: &str) -> Result<String> {
        let args: Value = serde_json::from_str(arguments_json).unwrap_or(json!({}));

        match name {
            "file_read" => {
                let path = args["path"].as_str().ok_or("path missing")?;
                let content = BranchWorkspaceManager::read_file(&self.root, &self.config.project_id, &self.config.branch_id, path)?;
                Ok(content)
            }
            "file_write" => {
                let path = args["path"].as_str().ok_or("path missing")?;
                let content = args["content"].as_str().ok_or("content missing")?;
                let hash = BranchWorkspaceManager::write_file(&self.root, &self.config.project_id, &self.config.branch_id, path, content)?;
                Ok(json!({"status": "written", "path": path, "sha256": hash}).to_string())
            }
            "file_patch" => {
                let path = args["path"].as_str().ok_or("path missing")?;
                let target = args["target"].as_str().ok_or("target missing")?;
                let rep = args["replacement"].as_str().ok_or("replacement missing")?;
                BranchWorkspaceManager::patch_file(&self.root, &self.config.project_id, &self.config.branch_id, path, target, rep)?;
                Ok(json!({"status": "patched", "path": path}).to_string())
            }
            "file_list" => {
                let files = BranchWorkspaceManager::list_files(&self.root, &self.config.project_id, &self.config.branch_id)?;
                let items: Vec<Value> = files.into_iter().map(|(rel, len, sha)| {
                    json!({"path": rel, "bytes": len, "sha256": sha})
                }).collect();
                Ok(json!({"files": items}).to_string())
            }
            "register_project_tool" => {
                let tool_name = args["name"].as_str().ok_or("tool name missing")?;
                let ver = args["version"].as_str().ok_or("version missing")?;
                let desc_text = args["description"].as_str().ok_or("description missing")?;
                let entry = args["entrypoint"].as_str().ok_or("entrypoint missing")?;
                let branch_hash = BranchWorkspaceManager::compute_branch_hash(&self.root, &self.config.project_id, &self.config.branch_id)?;

                let desc = ToolDescriptor {
                    name: tool_name.to_string(),
                    version: ver.to_string(),
                    description: desc_text.to_string(),
                    input_schema: args.get("input_schema").cloned().unwrap_or(json!({})),
                    output_schema: args.get("output_schema").cloned().unwrap_or(json!({})),
                    entrypoint: entry.to_string(),
                    source_hash: branch_hash,
                    environment_id: args["environment_id"].as_str().unwrap_or("scientific").to_string(),
                    permitted_artifacts: vec![],
                    max_wall_ms: 30_000,
                    provenance: json!({"author_role": self.config.role_id, "task_id": self.config.task_id}),
                };

                ToolRegistry::register_tool(&self.root, &self.config.project_id, desc)?;
                self.context.record_achievement(
                    &self.root,
                    "tool_authored",
                    &format!("Registered project tool {tool_name}:{ver}"),
                    None,
                    json!({"tool": tool_name, "version": ver}),
                )?;
                Ok(json!({"status": "registered", "tool": tool_name, "version": ver}).to_string())
            }
            "invoke_project_tool" => {
                let tool_name = args["name"].as_str().ok_or("tool name missing")?;
                let ver = args["version"].as_str().ok_or("version missing")?;
                let input_val = args.get("input").cloned().unwrap_or(json!({}));
                let receipt = ToolRegistry::invoke_tool(
                    &self.root,
                    &self.config.project_id,
                    &self.config.branch_id,
                    tool_name,
                    ver,
                    &input_val,
                )?;
                Ok(serde_json::to_string(&receipt).unwrap_or_default())
            }
            "inspect_dataset" => {
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                let mut stmt = db.prepare("SELECT id, sha256, bytes FROM scientific_datasets WHERE project=?1").map_err(|e| e.to_string())?;
                let datasets: Vec<Value> = stmt.query_map([&self.config.project_id], |r| {
                    Ok(json!({
                        "id": r.get::<_, String>(0)?,
                        "sha256": r.get::<_, String>(1)?,
                        "bytes": r.get::<_, i64>(2)?,
                    }))
                }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();
                Ok(json!({"datasets": datasets}).to_string())
            }
            "delegate" => {
                let target_role = args["role_id"].as_str().ok_or("role_id missing")?;
                let intent = args["intent"].as_str().ok_or("intent missing")?;
                let hypothesis = args.get("hypothesis").and_then(|v| v.as_str());
                let sub_task_id = format!("task-{}", &crate::agents::workspace::BranchWorkspaceManager::compute_branch_hash(&self.root, &self.config.project_id, &self.config.branch_id)?[..10]);

                let now = now_ms();
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                db.execute(
                    "INSERT INTO agent_tasks(project, task_id, role_id, parent_task_id, status, hypothesis, intent, assigned_branch, created_ms, updated_ms)
                     VALUES(?1, ?2, ?3, ?4, 'active', ?5, ?6, ?7, ?8, ?8)",
                    rusqlite::params![
                        self.config.project_id,
                        sub_task_id,
                        target_role,
                        self.config.task_id,
                        hypothesis,
                        intent,
                        self.config.branch_id,
                        now,
                    ],
                ).map_err(|e| e.to_string())?;

                Ok(json!({"status": "delegated", "sub_task_id": sub_task_id, "role": target_role}).to_string())
            }
            "query_budget" => {
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                let policy_str: String = db.query_row(
                    "SELECT policy FROM projects WHERE id=?1",
                    [&self.config.project_id],
                    |r| r.get(0),
                ).unwrap_or_else(|_| "{}".into());
                Ok(json!({"policy": policy_str}).to_string())
            }
            "request_finalization" => {
                if self.is_research_closed()? {
                    return Err("Research admission is permanently closed; cannot re-finalize".into());
                }
                let cand_id = args["candidate_id"].as_str().ok_or("candidate_id missing")?;
                let rationale = args["rationale"].as_str().ok_or("rationale missing")?;

                // Atomically close research admission before final test gate is reserved
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                db.execute(
                    "UPDATE projects SET state='finalizing' WHERE id=?1",
                    [&self.config.project_id],
                ).map_err(|e| e.to_string())?;

                self.context.record_achievement(
                    &self.root,
                    "finalization_requested",
                    &format!("Finalization requested for apex candidate {cand_id}: {rationale}"),
                    None,
                    json!({"candidate_id": cand_id, "rationale": rationale}),
                )?;

                Ok(json!({"status": "finalization_admitted", "candidate_id": cand_id}).to_string())
            }
            other => Err(format!("Unknown tool: {other}")),
        }
    }
}
