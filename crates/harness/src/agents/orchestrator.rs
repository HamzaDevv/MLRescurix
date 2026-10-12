use std::{
    fs,
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    agents::{
        context::AgentContextManager,
        gateway::{ChatMessage, ChatCompletionRequest, ProviderGateway, TokenUsage, record_provider_accounting},
        roles::get_role,
        sandbox::{SandboxLimits, SandboxRunner},
        tools::{ToolDescriptor, ToolRegistry},
        workspace::BranchWorkspaceManager,
    },
    Result, Runtime,
    core::{Operation, Request, Policy, Task},
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
        ).map_err(|e| e.to_string())?;
        let proj_state: String = db.query_row(
            "SELECT state FROM projects WHERE id=?1",
            [&self.config.project_id],
            |r| r.get(0),
        ).map_err(|e| e.to_string())?;

        let is_finalizing_or_completed = matches!(proj_state.as_str(), "finalizing" | "completed");
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
        self.context.ensure_context_within_bounds()?;

        if self.is_research_closed()? {
            return Ok(TurnExecutionResult { turn_index: self.current_turn, tool_calls_executed: 0, assistant_content: None, stopped: true, stop_reason: Some("research_closed".into()) });
        }
        let role = get_role(&self.config.role_id).ok_or("Unknown role")?;
        let tools = ToolRegistry::get_trusted_tools().into_iter()
            .filter(|tool| role.capabilities.contains(&tool.function.name)).collect();

        let req = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: self.context.messages.clone(),
            tools,
            temperature: Some(0.2),
            max_tokens: Some(4096),
        };

        // Use M0's serialized writer for admission, fencing and settlement.
        // Serialized UTF-8 byte length conservatively bounds prompt tokens,
        // including tool definitions and tool-call arguments.
        let request_bytes = serde_json::to_vec(&req).map_err(|e| e.to_string())?;
        let est_prompt = u32::try_from(request_bytes.len()).map_err(|_| "Provider request too large")?;
        if est_prompt as u64 + 4096 > self.context.max_context_tokens as u64 {
            return Err("Serialized provider request exceeds context limit".into());
        }
        let now = now_ms();
        let action_id = format!("provider-{}-{}", std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos());
        let runtime = Runtime::open(&self.root)?;
        let command = |suffix: &str, operation: Operation| -> Result<crate::core::Receipt> {
            let receipt = runtime.command(Request {
                command_id: format!("{action_id}-{suffix}"),
                project_id: self.config.project_id.clone(),
                expected_revision: None, operation,
            })?;
            if receipt.status != "applied" {
                return Err(format!("Provider {suffix} refused: {}", receipt.result));
            }
            Ok(receipt)
        };
        let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let task_json: String = db.query_row(
            "SELECT data FROM tasks WHERE project=?1 AND id=?2",
            rusqlite::params![self.config.project_id, self.config.task_id], |r| r.get(0),
        ).map_err(|e| format!("Provider requires an authoritative M0 task: {e}"))?;
        let task: Task = serde_json::from_str(&task_json).map_err(|e| e.to_string())?;
        let owner = format!("agent-{}", self.config.task_id);
        let (generation, lease_id) = if task.expires_ms.is_some_and(|end| end > now) {
            if task.owner.as_deref() != Some(&owner) {
                return Err("Provider task owned by another active lease".into());
            }
            (task.generation, task.lease_id.ok_or("Missing active task lease")?)
        } else {
            let lease_id = format!("lease-{action_id}");
            let claimed = command("claim", Operation::Claim {
                task_id: self.config.task_id.clone(), owner,
                lease_id: lease_id.clone(), now_ms: now, expires_ms: now + 300_000,
                expected_generation: task.generation,
            })?;
            (claimed.result["generation"].as_u64().ok_or("Claim missing generation")?, lease_id)
        };
        let policy_json: String = db.query_row("SELECT policy FROM projects WHERE id=?1",
            [&self.config.project_id], |r| r.get(0)).map_err(|e| e.to_string())?;
        let policy: Policy = serde_json::from_str(&policy_json).map_err(|e| e.to_string())?;
        let mut budget = BTreeMap::new();
        for dimension in policy.budget.keys() {
            let ceiling = match dimension.as_str() {
                "tokens" => (est_prompt + 4096) as i64,
                "calls" => 1,
                "wall_ms" => 60_000,
                other => return Err(format!("Provider cannot enforce budget dimension {other}")),
            };
            budget.insert(dimension.clone(), ceiling);
        }
        let capacity = policy.capacity.keys().map(|k| (k.clone(), if k == "workers" {1} else {0})).collect();
        let request_hash = format!("{:x}", Sha256::digest(&request_bytes));
        command("prepare", Operation::Prepare {
            task_id: self.config.task_id.clone(), generation, lease_id: lease_id.clone(),
            action_id: action_id.clone(), input: request_hash,
            budget: budget.clone(), capacity, deadline_ms: now + 60_000, now_ms: now,
        })?;
        let dispatched = command("dispatch", Operation::Dispatch {
            task_id: self.config.task_id.clone(), generation, lease_id: lease_id.clone(),
            action_id: action_id.clone(), attempt_id: action_id.clone(),
            external_request_id: None, now_ms: now_ms(),
        })?;
        if dispatched.result["dispatch_allowed"] != true {
            return Err("Provider action requires reconciliation; refusing replay".into());
        }
        let response = match self.gateway.chat_completion(&req) {
            Ok(resp) => resp,
            Err(e) => {
                command("unknown", Operation::MarkUnknown {
                    action_id: action_id.clone(), reason: "Provider attempt has uncertain remote outcome".into(),
                })?;
                record_provider_accounting(&self.root, &self.config.project_id, &action_id,
                    &self.config.task_id, &self.config.role_id, &self.config.model,
                    &TokenUsage { prompt_tokens: est_prompt, completion_tokens: 4096, total_tokens: est_prompt + 4096 },
                    0, 0, "outcome_unknown_ceiling_estimate")?;
                command("resolve", Operation::Resolve {
                    action_id: action_id.clone(), resolution: "failed".into(), actual: budget,
                    proof: "Unresolved provider attempt conservatively settled at admitted ceiling; remote billing unknown".into(),
                })?;
                return Err(e);
            }
        };
        record_provider_accounting(&self.root, &self.config.project_id, &action_id,
            &self.config.task_id, &self.config.role_id, &response.model,
            &response.usage, 0, response.latency_ms, "success_cost_unpriced")?;
        let frame = json!({"version":1,"kind":"observation","payload":{
            "project_id":self.config.project_id,"task_id":self.config.task_id,
            "action_id":action_id,"generation":generation,"phase":"provider_response",
            "detail":format!("response {}", response.id),
        }}).to_string();
        command("response", Operation::Response {
            task_id: self.config.task_id.clone(), generation, lease_id: lease_id.clone(),
            action_id: action_id.clone(), frame, now_ms: now_ms(),
        })?;
        let actual = budget.keys().map(|k| (k.clone(), match k.as_str() {
            "tokens" => i64::try_from(response.usage.total_tokens).unwrap_or(i64::MAX),
            "calls" => 1,
            "wall_ms" => now_ms().saturating_sub(now),
            _ => 0,
        })).collect();
        command("commit", Operation::Commit {
            task_id: self.config.task_id.clone(), generation, lease_id,
            action_id: action_id.clone(), actual, outcome: "succeeded".into(), now_ms: now_ms(),
        })?;

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
            let result_str = match self.dispatch_tool_call(&call.function.name, &call.function.arguments) {
                Ok(res) => res,
                Err(err) => json!({"status": "error", "error": err}).to_string(),
            };
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

    pub fn dispatch_tool_call(&mut self, name: &str, arguments_json: &str) -> Result<String> {
        // Enforce role RBAC before executing any tool
        let role_spec = get_role(&self.config.role_id)
            .ok_or_else(|| format!("Unknown role: {}", self.config.role_id))?;
        let is_authorized = role_spec.capabilities.iter().any(|c| c == name)
;
        if !is_authorized {
            return Err(format!("Role '{}' is not authorized to invoke tool '{}'", self.config.role_id, name));
        }

        let args: Value = serde_json::from_str(arguments_json).map_err(|e| format!("Invalid tool arguments: {e}"))?;

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
            "sandbox_python" => {
                let code = args["code"].as_str().ok_or("code missing")?;
                let filename = args.get("filename").and_then(|v| v.as_str()).unwrap_or("snippet.py");
                let branch_dir = BranchWorkspaceManager::get_branch_dir(&self.root, &self.config.project_id, &self.config.branch_id);
                let code_path = branch_dir.join(filename);
                fs::write(&code_path, code).map_err(|e| e.to_string())?;
                let limits = SandboxLimits::default();
                let res = SandboxRunner::execute(&branch_dir, filename, &[], &limits, None);
                if code_path.exists() {
                    let _ = fs::remove_file(code_path);
                }
                let execution_res = res?;
                Ok(serde_json::to_string(&json!({
                    "exit_code": execution_res.exit_code,
                    "stdout": execution_res.stdout,
                    "stderr": execution_res.stderr,
                    "elapsed_ms": execution_res.elapsed_ms,
                })).unwrap())
            }
            "retrieve_evidence" => {
                let id = args["evidence_id"].as_str().ok_or("evidence_id missing")?;
                crate::execution::migrate(&self.root)?;
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                let ach_row: Option<(String, String, Option<String>, String)> = db.query_row(
                    "SELECT kind, summary, evidence_sha256, metrics FROM achievements WHERE project=?1 AND achievement_id=?2",
                    rusqlite::params![self.config.project_id, id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                ).ok();
                if let Some((kind, summary, sha, metrics)) = ach_row {
                    if let Some(ref s) = sha {
                        let p1 = self.root.join(".m2/evidence").join(s);
                        let p2 = self.root.join("workflows").join(&self.config.project_id).join("evidence").join(s);
                        if p1.is_file() {
                            let content = fs::read(&p1).map_err(|e| e.to_string())?;
                            let actual = format!("{:x}", Sha256::digest(&content));
                            if &actual != s {
                                return Err(format!("Durable evidence corruption: expected {s}, found {actual}"));
                            }
                        } else if p2.is_file() {
                            let content = fs::read(&p2).map_err(|e| e.to_string())?;
                            let actual = format!("{:x}", Sha256::digest(&content));
                            if &actual != s {
                                return Err(format!("Durable evidence corruption: expected {s}, found {actual}"));
                            }
                        }
                    }
                    return Ok(json!({
                        "id": id,
                        "type": "achievement",
                        "kind": kind,
                        "summary": summary,
                        "evidence_sha256": sha,
                        "metrics": serde_json::from_str::<Value>(&metrics).unwrap_or(json!({})),
                    }).to_string());
                }
                let find_row: Option<(String, String, String, String)> = db.query_row(
                    "SELECT hypothesis_id, verdict, rationale, metrics FROM scientific_findings WHERE project=?1 AND finding_id=?2",
                    rusqlite::params![self.config.project_id, id],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                ).ok();
                if let Some((hyp, verdict, rationale, metrics)) = find_row {
                    return Ok(json!({
                        "id": id,
                        "type": "finding",
                        "hypothesis_id": hyp,
                        "verdict": verdict,
                        "rationale": rationale,
                        "metrics": serde_json::from_str::<Value>(&metrics).unwrap_or(json!({})),
                    }).to_string());
                }
                Err(format!("Evidence or finding with id '{id}' not found"))
            }
            "submit_critic_audit" => {
                let cand_id = args["candidate_id"].as_str().ok_or("candidate_id missing")?;
                let status = args["verdict"].as_str().ok_or("verdict missing")?;
                let rationale = args["evidence_summary"].as_str().ok_or("evidence_summary missing")?;
                let source_hash = args.get("source_hash").and_then(|v| v.as_str()).unwrap_or("");
                let protocol = args.get("protocol").and_then(|v| v.as_str()).unwrap_or("standard_holdout");
                let constraints_verified = if args.get("protocol_compliance").and_then(|v| v.as_bool()).unwrap_or(false) { 1 } else { 0 };
                crate::execution::migrate(&self.root)?;
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                db.execute(
                    "INSERT INTO critic_audits(project, candidate_id, auditor_role, status, rationale, source_hash, protocol, constraints_verified, created_ms)
                     VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    rusqlite::params![
                        self.config.project_id,
                        cand_id,
                        self.config.role_id,
                        status,
                        rationale,
                        source_hash,
                        protocol,
                        constraints_verified,
                        now_ms(),
                    ],
                ).map_err(|e| e.to_string())?;
                self.context.record_achievement(
                    &self.root,
                    "critic_audit_submitted",
                    &format!("Critic audit submitted for candidate {cand_id}: {status}"),
                    None,
                    json!({"candidate_id": cand_id, "status": status, "rationale": rationale}),
                )?;
                Ok(json!({"status": "audit_recorded", "candidate_id": cand_id, "audit_status": status}).to_string())
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
                let specialist_branch = format!("{}-{}", self.config.branch_id, target_role);
                let sub_task_id = format!("{}-{}", self.config.task_id, target_role);

                // Initialize isolated branch workspace for specialist
                BranchWorkspaceManager::init_branch(&self.root, &self.config.project_id, &specialist_branch, &self.config.branch_id)?;

                let now = now_ms();
                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
                db.execute(
                    "INSERT INTO agent_tasks(project, task_id, role_id, parent_task_id, status, hypothesis, intent, assigned_branch, created_ms, updated_ms)
                     VALUES(?1, ?2, ?3, ?4, 'active', ?5, ?6, ?7, ?8, ?8)
                     ON CONFLICT(project, task_id) DO UPDATE SET status='active', updated_ms=?8",
                    rusqlite::params![
                        self.config.project_id,
                        sub_task_id,
                        target_role,
                        self.config.task_id,
                        hypothesis,
                        intent,
                        specialist_branch,
                        now,
                    ],
                ).map_err(|e| e.to_string())?;

                Ok(json!({
                    "status": "delegated",
                    "sub_task_id": sub_task_id,
                    "role": target_role,
                    "assigned_branch": specialist_branch,
                }).to_string())
            }
            "query_budget" => {
                let budget_info = AgentContextManager::query_authoritative_budget(&self.root, &self.config.project_id)?;
                Ok(budget_info.to_string())
            }
            "request_finalization" => {
                if self.is_research_closed()? {
                    return Err("Research admission is permanently closed; cannot re-finalize".into());
                }
                let cand_id = args["candidate_id"].as_str().ok_or("candidate_id missing")?;
                let rationale = args["rationale"].as_str().ok_or("rationale missing")?;

                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;

                // Two-key validation check:
                // 1. Critic Audit check: must have an approved critic audit in critic_audits
                let audit_status: Option<String> = db.query_row(
                    "SELECT status FROM critic_audits WHERE project=?1 AND candidate_id=?2 ORDER BY created_ms DESC LIMIT 1",
                    rusqlite::params![self.config.project_id, cand_id],
                    |r| r.get(0),
                ).ok();

                if audit_status.as_deref() != Some("approved") {
                    return Err(format!("Finalization refused: candidate '{cand_id}' has not received an approved critic audit in critic_audits (found: {:?})", audit_status));
                }

                // 2. Authenticated User Approval check: must have approval in approvals ledger
                let is_user_approved: Option<i64> = db.query_row(
                    "SELECT approved FROM approvals WHERE project=?1 AND candidate_id=?2 ORDER BY created_ms DESC LIMIT 1",
                    rusqlite::params![self.config.project_id, cand_id],
                    |r| r.get(0),
                ).ok();

                if is_user_approved != Some(1) {
                    return Err(format!("Finalization refused: candidate '{cand_id}' lacks verified user approval in approvals ledger. An authenticated approval receipt is required."));
                }

                // Atomically close research admission before final test gate is reserved
                let changed = db.execute(
                    "UPDATE projects SET state='finalizing' WHERE id=?1 AND state NOT IN ('finalizing', 'completed')",
                    [&self.config.project_id],
                ).map_err(|e| e.to_string())?;
                if changed != 1 {
                    return Err("Research admission is permanently closed; cannot re-finalize".into());
                }

                // Freeze exact source snapshot without automatic git merge
                let frozen_source_hash = BranchWorkspaceManager::compute_branch_hash(&self.root, &self.config.project_id, &self.config.branch_id)?;

                self.context.record_achievement(
                    &self.root,
                    "finalization_admitted",
                    &format!("Finalization admitted for apex candidate {cand_id}: {rationale}"),
                    None,
                    json!({"candidate_id": cand_id, "rationale": rationale, "frozen_source_hash": frozen_source_hash}),
                )?;

                Ok(json!({
                    "status": "finalization_admitted",
                    "candidate_id": cand_id,
                    "frozen_source_hash": frozen_source_hash,
                }).to_string())
            }
            other => Err(format!("Unknown tool: {other}")),
        }
    }
}
