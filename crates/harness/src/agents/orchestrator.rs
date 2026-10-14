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
        gateway::{
            ChatMessage, ChatCompletionRequest, ChatCompletionResponse, OutboundFilter,
            ProviderGateway, TokenUsage, record_provider_accounting, resolve_model_pricing,
        },
        roles::get_role,
        sandbox::{SandboxLimits, SandboxRunner},
        tools::{ToolDescriptor, ToolRegistry},
        workspace::BranchWorkspaceManager,
    },
    Result, Runtime,
    core::{Action, Operation, Request, Policy, Task},
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

        crate::execution::migrate(root)?;
        let db=rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e|e.to_string())?;
        let checkpoints:i64=db.query_row("SELECT count(*) FROM agent_checkpoints WHERE project=?1 AND task_id=?2",rusqlite::params![config.project_id,config.task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
        let current_turn:u32=db.query_row("SELECT COALESCE(max(turn),0) FROM provider_turn_receipts WHERE project=?1 AND task=?2",rusqlite::params![config.project_id,config.task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
        let mut context = if checkpoints>0 {AgentContextManager::restore(root,&config.project_id,&config.task_id)?} else {AgentContextManager::new(
            &config.project_id,
            &config.task_id,
            &config.role_id,
            role_spec.system_prompt.clone(),
        )};
        if context.role_id!=config.role_id {
            context.role_id=config.role_id.clone();
            if context.messages.first().is_some_and(|m|m.role=="system") {
                context.messages[0]=ChatMessage::system(role_spec.system_prompt.clone());
            } else {context.messages.insert(0,ChatMessage::system(role_spec.system_prompt.clone()));}
        }

        BranchWorkspaceManager::init_branch(root, &config.project_id, &config.branch_id, "main")?;

        Ok(Self {
            root: root.to_path_buf(),
            config,
            context,
            gateway,
            current_turn,
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

    /// Build the exact sanitized outbound request payload and its SHA-256 digest.
    pub fn build_sanitized_request(&mut self) -> Result<(ChatCompletionRequest, Vec<u8>, String)> {
        self.context.ensure_context_within_bounds()?;
        let role = get_role(&self.config.role_id).ok_or("Unknown role")?;
        let tools = ToolRegistry::get_trusted_tools().into_iter()
            .filter(|tool| role.capabilities.contains(&tool.function.name)).collect();

        let mut req = ChatCompletionRequest {
            model: self.config.model.clone(),
            messages: self.context.messages.clone(),
            tools,
            temperature: Some(0.2),
            max_tokens: Some(4096),
        };
        OutboundFilter::sanitize_request(&mut req);
        let request_bytes = serde_json::to_vec(&req).map_err(|e| e.to_string())?;
        let digest = format!("{:x}", Sha256::digest(&request_bytes));
        Ok((req, request_bytes, digest))
    }

    /// Stage the exact sanitized outbound request payload into `outbound_views` for policy inspection/approval.
    pub fn stage_outbound_view(&mut self, actor: &str) -> Result<Value> {
        let (req, request_bytes, digest) = self.build_sanitized_request()?;
        let view_json = String::from_utf8(request_bytes).map_err(|e| e.to_string())?;
        crate::execution::migrate(&self.root)?;
        let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO outbound_views(project, request_sha256, view_json, actor, approved, expires_ms)
             VALUES(?1, ?2, ?3, ?4, 0, 0)
             ON CONFLICT(project, request_sha256) DO UPDATE SET view_json=excluded.view_json, actor=excluded.actor WHERE outbound_views.approved=0",
            rusqlite::params![self.config.project_id, digest, view_json, actor],
        ).map_err(|e| e.to_string())?;
        Ok(json!({
            "project_id": self.config.project_id,
            "request_sha256": digest,
            "actor": actor,
            "approved": false,
            "view": req,
        }))
    }

    /// Reconcile any unresolved provider turns (`awaiting_response`, `outcome_unknown`, `response_received`, `executing_tools`)
    /// and orphaned M0 actions safely across process restarts without blind replay or duplicate tool execution.
    pub fn reconcile(&mut self) -> Result<Value> {
        crate::execution::migrate(&self.root)?;
        let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let mut stmt = db.prepare(
            "SELECT turn, action, status, response_json FROM provider_turn_receipts
             WHERE project=?1 AND task=?2 AND status != 'completed' ORDER BY turn ASC"
        ).map_err(|e| e.to_string())?;
        let rows: Vec<(u32, String, String, Option<String>)> = stmt.query_map(
            rusqlite::params![self.config.project_id, self.config.task_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        ).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;
        drop(stmt);

        let runtime = Runtime::open(&self.root)?;
        let mut reconciled_turns = Vec::new();

        for (turn, action_id, status, response_json) in rows {
            let now = now_ms();
            let task_json: String = db.query_row(
                "SELECT data FROM tasks WHERE project=?1 AND id=?2",
                rusqlite::params![self.config.project_id, self.config.task_id],
                |r| r.get(0),
            ).map_err(|e| e.to_string())?;
            let task: Task = serde_json::from_str(&task_json).map_err(|e| e.to_string())?;
            let owner = format!("agent-{}", self.config.task_id);
            let (generation, lease_id) = if task.expires_ms.is_some_and(|end| end > now) && task.owner.as_deref() == Some(&owner) {
                (task.generation, task.lease_id.unwrap_or_else(|| format!("lease-reconcile-{action_id}")))
            } else {
                let lease_id = format!("lease-reconcile-{action_id}-{}", task.generation);
                let claimed = runtime.command(Request {
                    command_id: format!("reconcile-{action_id}-claim-{}", task.generation),
                    project_id: self.config.project_id.clone(),
                    expected_revision: None,
                    operation: Operation::Claim {
                        task_id: self.config.task_id.clone(),
                        owner,
                        lease_id: lease_id.clone(),
                        now_ms: now,
                        expires_ms: now + 300_000,
                        expected_generation: task.generation,
                    },
                })?;
                if claimed.status != "applied" {
                    return Err(format!("Reconcile claim refused: {}", claimed.result));
                }
                (claimed.result["generation"].as_u64().ok_or("Claim missing generation")?, lease_id)
            };

            let action_json:String=db.query_row(
                "SELECT data FROM actions WHERE project=?1 AND id=?2",
                rusqlite::params![self.config.project_id, action_id],
                |r|r.get(0),
            ).map_err(|e|format!("Provider reconciliation requires an authoritative action receipt: {e}"))?;
            let action_opt:Option<Action>=Some(serde_json::from_str(&action_json).map_err(|e|format!("Invalid authoritative action receipt: {e}"))?);

            match status.as_str() {
                "awaiting_response" | "outcome_unknown" => {
                    if let Some(ref act) = action_opt {
                        if act.status == "dispatched" {
                            let settlement = runtime.command(Request {
                                command_id: format!("reconcile-{action_id}-unknown"),
                                project_id: self.config.project_id.clone(),
                                expected_revision: None,
                                operation: Operation::MarkUnknown {
                                    action_id: action_id.clone(),
                                    reason: format!("Reconciling turn {turn} from {status} without blind provider replay"),
                                },
                            })?;
                            if settlement.status != "applied" {return Err(format!("Provider reconciliation refused: {}",settlement.result));}
                        }
                        let est_tokens = act.budget.get("tokens").copied().unwrap_or(4096) as u32;
                        let est_prompt = est_tokens.saturating_sub(4096);
                        let est_cost = act.budget.get("cost_millicents").copied().unwrap_or(0);
                        record_provider_accounting(
                            &self.root,
                            &self.config.project_id,
                            &action_id,
                            &self.config.task_id,
                            &self.config.role_id,
                            &self.config.model,
                            &TokenUsage { prompt_tokens: est_prompt, completion_tokens: 4096, total_tokens: est_tokens },
                            est_cost,
                            0,
                            "outcome_unknown_ceiling_estimate",
                        )?;
                        if act.status == "dispatched" || act.status == "outcome_unknown" {
                            let settlement = runtime.command(Request {
                                command_id: format!("reconcile-{action_id}-resolve"),
                                project_id: self.config.project_id.clone(),
                                expected_revision: None,
                                operation: Operation::Resolve {
                                    action_id: action_id.clone(),
                                    resolution: "failed".into(),
                                    actual: act.budget.clone(),
                                    proof: format!("Reconciled uncertain turn {turn} ({status}) at admitted ceiling without replay"),
                                },
                            })?;
                            if settlement.status != "applied" {return Err(format!("Provider reconciliation refused: {}",settlement.result));}
                        }
                    }
                    let summary_json = json!({
                        "reconciled": true,
                        "prior_status": status,
                        "resolution": "settled_at_admitted_ceiling_no_replay"
                    }).to_string();
                    db.execute(
                        "UPDATE provider_turn_receipts SET status='completed', response_json=COALESCE(response_json, ?4), updated_ms=?5
                         WHERE project=?1 AND task=?2 AND turn=?3",
                        rusqlite::params![self.config.project_id, self.config.task_id, turn, summary_json, now_ms()],
                    ).map_err(|e| e.to_string())?;
                    reconciled_turns.push(json!({
                        "turn": turn,
                        "action_id": action_id,
                        "prior_status": status,
                        "resolution": "settled_at_admitted_ceiling_no_replay"
                    }));
                }
                "response_received" | "executing_tools" => {
                    let resp_str = response_json.ok_or_else(|| format!("Turn {turn} in {status} is missing response_json"))?;
                    let response: ChatCompletionResponse = serde_json::from_str(&resp_str).map_err(|e| e.to_string())?;
                    let (actual_cost, pricing_status) = match resolve_model_pricing(&self.root, &response.model)?
                        .or(resolve_model_pricing(&self.root, &self.config.model)?)
                    {
                        Some((p_rate, c_rate)) => {
                            let c = ((response.usage.prompt_tokens as i64 * p_rate + 999) / 1000)
                                + ((response.usage.completion_tokens as i64 * c_rate + 999) / 1000);
                            (c, "priced")
                        }
                        None => (0, "success_cost_unpriced"),
                    };
                    record_provider_accounting(
                        &self.root,
                        &self.config.project_id,
                        &action_id,
                        &self.config.task_id,
                        &self.config.role_id,
                        &response.model,
                        &response.usage,
                        actual_cost,
                        response.latency_ms,
                        pricing_status,
                    )?;
                    if let Some(ref act) = action_opt {
                        let stale = act.generation != generation || act.lease_id != lease_id || act.deadline_ms <= now_ms();
                        if act.status == "dispatched" && !stale {
                            let frame = json!({
                                "version": 1,
                                "kind": "observation",
                                "payload": {
                                    "project_id": self.config.project_id,
                                    "task_id": self.config.task_id,
                                    "action_id": action_id,
                                    "generation": act.generation,
                                    "phase": "provider_response",
                                    "detail": format!("response {}", response.id),
                                }
                            }).to_string();
                            let settlement = runtime.command(Request {
                                command_id: format!("reconcile-{action_id}-response"),
                                project_id: self.config.project_id.clone(),
                                expected_revision: None,
                                operation: Operation::Response {
                                    task_id: self.config.task_id.clone(),
                                    generation,
                                    lease_id: lease_id.clone(),
                                    action_id: action_id.clone(),
                                    frame,
                                    now_ms: now_ms(),
                                },
                            })?;
                            if settlement.status != "applied" {return Err(format!("Provider reconciliation refused: {}",settlement.result));}
                        }
                        if matches!(act.status.as_str(), "dispatched" | "responded" | "outcome_unknown") {
                            let actual = act.budget.keys().map(|k| (k.clone(), match k.as_str() {
                                "tokens" => i64::try_from(response.usage.total_tokens).unwrap_or(i64::MAX),
                                "calls" => 1,
                                "wall_ms" => i64::try_from(response.latency_ms).unwrap_or(1),
                                "cost_millicents" => actual_cost,
                                _ => 0,
                            })).collect();
                            if (stale || act.status == "outcome_unknown") && act.status == "dispatched" {
                                let settlement = runtime.command(Request {
                                    command_id: format!("reconcile-{action_id}-cached-unknown"),
                                    project_id: self.config.project_id.clone(), expected_revision: None,
                                    operation: Operation::MarkUnknown {action_id: action_id.clone(), reason: "expired provider fence with durable cached response".into()},
                                })?;
                                if settlement.status != "applied" {return Err(format!("Provider reconciliation refused: {}",settlement.result));}
                            }
                            let settlement_operation = if stale || act.status == "outcome_unknown" {
                                Operation::Resolve {action_id:action_id.clone(), resolution:"succeeded".into(), actual,
                                    proof: format!("durable provider response sha256={:x}", Sha256::digest(resp_str.as_bytes()))}
                            } else { Operation::Commit {
                                task_id:self.config.task_id.clone(),generation,lease_id:lease_id.clone(),action_id:action_id.clone(),actual,outcome:"succeeded".into(),now_ms:now_ms(),
                            }};
                            let settlement = runtime.command(Request {
                                command_id: format!("reconcile-{action_id}-commit"),
                                project_id: self.config.project_id.clone(),
                                expected_revision: None,
                                operation: settlement_operation,
                            })?;
                            if settlement.status != "applied" {return Err(format!("Provider reconciliation refused: {}",settlement.result));}
                        }
                    }

                    super::recovery::require_settlement(&db, &self.config.project_id, &self.config.task_id, &action_id)?;
                    // Restore latest checkpoint before resuming message/tool progression
                    if let Ok(restored) = AgentContextManager::restore(&self.root, &self.config.project_id, &self.config.task_id) {
                        self.context = restored;
                    }
                    self.current_turn = turn;

                    if response.tool_calls.is_empty() {
                        let has_assistant = self.context.messages.last().is_some_and(|m| m.role == "assistant" && m.content == response.content);
                        if !has_assistant {
                            self.context.add_message(ChatMessage::assistant(response.content.clone(), None));
                        }
                        self.complete_turn_receipt()?;
                        reconciled_turns.push(json!({
                            "turn": turn,
                            "action_id": action_id,
                            "prior_status": status,
                            "resolution": "committed_cached_response_no_tools"
                        }));
                    } else {
                        let has_assistant_tools = self.context.messages.iter().any(|m| {
                            m.role == "assistant" && m.tool_calls.as_ref().is_some_and(|tc| tc.iter().any(|c| c.id == response.tool_calls[0].id))
                        });
                        if !has_assistant_tools {
                            self.context.add_message(ChatMessage::assistant(response.content.clone(), Some(response.tool_calls.clone())));
                            self.context.record_checkpoint(&self.root)?;
                        }
                        // A side effect may have completed before its tool checkpoint.
                        // An unresolved execution is not permission to repeat it.
                        let tool_protocol: i64 = db.query_row("SELECT tool_protocol FROM provider_turn_receipts WHERE project=?1 AND task=?2 AND turn=?3", rusqlite::params![self.config.project_id,self.config.task_id,turn], |r|r.get(0)).map_err(|e|e.to_string())?;
                        let allow_new = status == "response_received" || tool_protocol == 1;
                        db.execute(
                            "UPDATE provider_turn_receipts SET status='executing_tools', tool_protocol=?5, updated_ms=?4 WHERE project=?1 AND task=?2 AND turn=?3",
                            rusqlite::params![self.config.project_id, self.config.task_id, turn, now_ms(), if allow_new {1} else {0}],
                        ).map_err(|e| e.to_string())?;

                        let mut executed_now = 0;
                        let mut skipped_completed = 0;
                        for call in &response.tool_calls {
                            let already_done = self.context.messages.iter().any(|m| {
                                m.role == "tool" && m.tool_call_id.as_deref() == Some(call.id.as_str())
                            });
                            if already_done {
                                skipped_completed += 1;
                                continue;
                            }
                            executed_now += 1;
                            let result_str = self.execute_durable_tool(call, allow_new)?;
                            self.context.add_message(ChatMessage::tool(result_str, &call.id, &call.function.name));
                            self.context.record_checkpoint(&self.root)?;
                        }
                        self.context.maybe_compact();
                        self.complete_turn_receipt()?;
                        reconciled_turns.push(json!({
                            "turn": turn,
                            "action_id": action_id,
                            "prior_status": status,
                            "resolution": "resumed_tools_from_checkpoint",
                            "tools_executed": executed_now,
                            "tools_skipped_already_completed": skipped_completed,
                        }));
                    }
                }
                other => return Err(format!("Unexpected provider_turn_receipts status: {other}")),
            }
        }

        // Also reconcile any orphaned M0 actions on this task that never reached completed/committed/resolved
        let mut act_stmt = db.prepare("SELECT id, data FROM actions WHERE project=?1").map_err(|e| e.to_string())?;
        let orphan_rows = act_stmt.query_map([&self.config.project_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        }).map_err(|e|e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;
        let mut orphan_actions = Vec::new();
        for (id, data) in orphan_rows {
            let action: Action = serde_json::from_str(&data).map_err(|e|e.to_string())?;
            if action.task_id == self.config.task_id && matches!(action.status.as_str(), "dispatched" | "outcome_unknown") {
                orphan_actions.push((id,action));
            }
        }
        drop(act_stmt);

        for (act_id, act) in orphan_actions {
            if act.status == "dispatched" {
                let settlement = runtime.command(Request {
                    command_id: format!("reconcile-orphan-{act_id}-unknown"),
                    project_id: self.config.project_id.clone(),
                    expected_revision: None,
                    operation: Operation::MarkUnknown {
                        action_id: act_id.clone(),
                        reason: "Orphaned dispatched action reconciled after restart".into(),
                    },
                })?;
                if settlement.status != "applied" {return Err(format!("Orphan settlement refused: {}",settlement.result));}
            }
            let settlement = runtime.command(Request {
                command_id: format!("reconcile-orphan-{act_id}-resolve"),
                project_id: self.config.project_id.clone(),
                expected_revision: None,
                operation: Operation::Resolve {
                    action_id: act_id.clone(),
                    resolution: "failed".into(),
                    actual: act.budget.clone(),
                    proof: "Orphaned action settled at ceiling during reconciliation".into(),
                },
            })?;
            if settlement.status != "applied" {return Err(format!("Orphan settlement refused: {}",settlement.result));}
        }

        self.current_turn = db.query_row(
            "SELECT COALESCE(max(turn),0) FROM provider_turn_receipts WHERE project=?1 AND task=?2",
            rusqlite::params![self.config.project_id, self.config.task_id],
            |r| r.get(0),
        ).map_err(|e| e.to_string())?;

        Ok(json!({
            "project_id": self.config.project_id,
            "task_id": self.config.task_id,
            "reconciled_turns": reconciled_turns,
            "current_turn": self.current_turn,
        }))
    }

    /// Execute a single turn boundary iteration.
    pub fn step(&mut self) -> Result<TurnExecutionResult> {
        let recovery_db=rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e|e.to_string())?;
        let unresolved:i64=recovery_db.query_row("SELECT count(*) FROM provider_turn_receipts WHERE project=?1 AND task=?2 AND status!='completed'",rusqlite::params![self.config.project_id,self.config.task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
        if unresolved>0{return Err("durable provider/tool outcome requires reconciliation; refusing blind replay".into());}
        if self.current_turn >= self.config.max_turns {
            return Ok(TurnExecutionResult {
                turn_index: self.current_turn,
                tool_calls_executed: 0,
                assistant_content: Some("Max turns reached".into()),
                stopped: true,
                stop_reason: Some("max_turns_exhausted".into()),
            });
        }

        if self.is_research_closed()? {
            return Ok(TurnExecutionResult { turn_index: self.current_turn + 1, tool_calls_executed: 0, assistant_content: None, stopped: true, stop_reason: Some("research_closed".into()) });
        }

        let (req, request_bytes, digest) = self.build_sanitized_request()?;
        if self.gateway.requires_policy_approval() {
            let db=rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e|e.to_string())?;
            let approved:bool=db.query_row("SELECT EXISTS(SELECT 1 FROM outbound_views WHERE project=?1 AND request_sha256=?2 AND approved=1 AND expires_ms>?3)",rusqlite::params![self.config.project_id,digest,now_ms()],|r|r.get(0)).map_err(|e|e.to_string())?;
            if !approved{return Err(format!("outbound provider request requires a content-bound policy-approved view ({digest}); pattern redaction is not upload authorization"));}
        }
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
        let model_pricing = resolve_model_pricing(&self.root, &self.config.model)?;
        let mut budget = BTreeMap::new();
        for dimension in policy.budget.keys() {
            let ceiling = match dimension.as_str() {
                "tokens" => (est_prompt + 4096) as i64,
                "calls" => 1,
                "wall_ms" => 60_000,
                "cost_millicents" => {
                    let (prompt_rate, comp_rate) = model_pricing.ok_or_else(|| {
                        format!(
                            "Provider model '{}' has no registered pricing in model_pricing; refusing unpriced dispatch under cost_millicents budget policy",
                            self.config.model
                        )
                    })?;
                    ((est_prompt as i64 * prompt_rate + 999) / 1000) + ((4096_i64 * comp_rate + 999) / 1000)
                }
                other => return Err(format!("Provider cannot enforce budget dimension {other}")),
            };
            budget.insert(dimension.clone(), ceiling);
        }
        // Protect the 20% finalization reserve against exploratory specialist/agent turns
        let is_finalization_turn = self.config.role_id == "critic"
            || self.context.messages.last().and_then(|m| m.content.as_deref()).is_some_and(|c| c.contains("request_finalization") || c.contains("[FINALIZE]"));
        if !is_finalization_turn {
            AgentContextManager::enforce_exploratory_budget_reserve(&self.root, &self.config.project_id, &budget)?;
        }

        let capacity = policy.capacity.keys().map(|k| (k.clone(), if k == "workers" {1} else {0})).collect();
        command("prepare", Operation::Prepare {
            task_id: self.config.task_id.clone(), generation, lease_id: lease_id.clone(),
            action_id: action_id.clone(), input: digest,
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
        self.current_turn += 1;
        self.context.record_checkpoint(&self.root)?;
        crate::execution::with_state_write(&self.root,16*1024,crate::execution::StorageWriteKind::ProviderReceipt,Some(&self.config.project_id),|tx|{
        tx.execute("INSERT INTO provider_turn_receipts(project,task,turn,action,status,updated_ms) VALUES(?1,?2,?3,?4,'awaiting_response',?5)",rusqlite::params![self.config.project_id,self.config.task_id,self.current_turn,action_id,now_ms()]).map_err(|e|e.to_string())?;
            Ok(())
        })?;
        let response = match self.gateway.chat_completion(&req) {
            Ok(resp) => resp,
            Err(e) => {
                db.execute("UPDATE provider_turn_receipts SET status='outcome_unknown',updated_ms=?4 WHERE project=?1 AND task=?2 AND turn=?3",rusqlite::params![self.config.project_id,self.config.task_id,self.current_turn,now_ms()]).map_err(|e|e.to_string())?;
                command("unknown", Operation::MarkUnknown {
                    action_id: action_id.clone(), reason: "Provider attempt has uncertain remote outcome".into(),
                })?;
                let ceiling_cost = budget.get("cost_millicents").copied().unwrap_or(0);
                record_provider_accounting(&self.root, &self.config.project_id, &action_id,
                    &self.config.task_id, &self.config.role_id, &self.config.model,
                    &TokenUsage { prompt_tokens: est_prompt, completion_tokens: 4096, total_tokens: est_prompt + 4096 },
                    ceiling_cost, 0, "outcome_unknown_ceiling_estimate")?;
                command("resolve", Operation::Resolve {
                    action_id: action_id.clone(), resolution: "failed".into(), actual: budget,
                    proof: "Unresolved provider attempt conservatively settled at admitted ceiling; remote billing unknown".into(),
                })?;
                return Err(e);
            }
        };
        let resp_json = serde_json::to_string(&response).map_err(|e|e.to_string())?;
        crate::execution::with_state_write(&self.root,resp_json.len() as u64*4+16*1024,crate::execution::StorageWriteKind::ProviderReceipt,Some(&self.config.project_id),|tx|{
        tx.execute("UPDATE provider_turn_receipts SET response_json=?4,status='response_received',updated_ms=?5 WHERE project=?1 AND task=?2 AND turn=?3",rusqlite::params![self.config.project_id,self.config.task_id,self.current_turn,resp_json,now_ms()]).map_err(|e|e.to_string())?;
            Ok(())
        })?;
        let (actual_cost, pricing_status) = match resolve_model_pricing(&self.root, &response.model)?
            .or(model_pricing)
        {
            Some((p_rate, c_rate)) => {
                let c = ((response.usage.prompt_tokens as i64 * p_rate + 999) / 1000)
                    + ((response.usage.completion_tokens as i64 * c_rate + 999) / 1000);
                (c, "priced")
            }
            None => (0, "success_cost_unpriced"),
        };
        record_provider_accounting(&self.root, &self.config.project_id, &action_id,
            &self.config.task_id, &self.config.role_id, &response.model,
            &response.usage, actual_cost, response.latency_ms, pricing_status)?;
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
            "cost_millicents" => actual_cost,
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
            self.complete_turn_receipt()?;
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
        self.context.record_checkpoint(&self.root)?;

        let mut tool_count = 0;
        db.execute("UPDATE provider_turn_receipts SET status='executing_tools',tool_protocol=1,updated_ms=?4 WHERE project=?1 AND task=?2 AND turn=?3",rusqlite::params![self.config.project_id,self.config.task_id,self.current_turn,now_ms()]).map_err(|e|e.to_string())?;
        for call in &response.tool_calls {
            tool_count += 1;
            let result_str = self.execute_durable_tool(call, true)?;
            self.context.add_message(ChatMessage::tool(result_str, &call.id, &call.function.name));
            self.context.record_checkpoint(&self.root)?;
        }

        self.context.maybe_compact();
        self.complete_turn_receipt()?;

        Ok(TurnExecutionResult {
            turn_index: self.current_turn,
            tool_calls_executed: tool_count,
            assistant_content: response.content,
            stopped: false,
            stop_reason: None,
        })
    }

    fn complete_turn_receipt(&mut self)->Result<()> {
        self.context.record_checkpoint(&self.root)?;
        let db=rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e|e.to_string())?;
        db.execute("UPDATE provider_turn_receipts SET status='completed',updated_ms=?4 WHERE project=?1 AND task=?2 AND turn=?3",rusqlite::params![self.config.project_id,self.config.task_id,self.current_turn,now_ms()]).map_err(|e|e.to_string())?;Ok(())
    }

    fn execute_durable_tool(&mut self, call: &crate::agents::gateway::ToolCall, allow_new: bool) -> Result<String> {
        let role = get_role(&self.config.role_id).ok_or("unknown tool role")?;
        if !role.capabilities.iter().any(|c| c == &call.function.name) {
            return Err(format!("Role '{}' is not authorized to invoke tool '{}'", self.config.role_id, call.function.name));
        }
        let root = self.root.clone();
        let config = self.config.clone();
        super::recovery::execute(&root, super::recovery::ToolIdentity {
            project: &config.project_id, task: &config.task_id, branch: &config.branch_id,
            turn: self.current_turn, call_id: &call.id, name: &call.function.name, arguments: &call.function.arguments,
        }, allow_new, || self.dispatch_tool_call(&call.function.name, &call.function.arguments))
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
            "schedule_pipeline" => {
                if self.is_research_closed()?{return Err("research admission is closed".into());}
                AgentContextManager::enforce_exploratory_budget_reserve(
                    &self.root,
                    &self.config.project_id,
                    &BTreeMap::from([("wall_ms".to_string(), 45_000)]),
                )?;
                let db=rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e|e.to_string())?;
                let data:String=db.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",rusqlite::params![self.config.project_id,self.config.task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
                let task:Task=serde_json::from_str(&data).map_err(|e|e.to_string())?;
                let view=crate::execution::submit(&self.root,crate::execution::JobSpec { resources: Default::default(),
                    job_id:args["job_id"].as_str().ok_or("job_id missing")?.into(),project_id:self.config.project_id.clone(),task_id:self.config.task_id.clone(),generation:task.generation,
                    timeout_ms:45_000,threads:1,max_output_bytes:4*1024*1024,
                    environment_id:Some(args["environment_id"].as_str().ok_or("environment_id missing")?.into()),
                    probe:crate::execution::Probe::Scientific{operation:"pipeline".into(),dataset_id:args["dataset_id"].as_str().ok_or("dataset_id missing")?.into(),spec:Some(args["protocol"].clone()),candidate_id:None,test_access_id:None},
                })?;Ok(serde_json::to_string(&view).map_err(|e|e.to_string())?)
            }
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
                if let Some(binding)=args.get("binding_sha256").and_then(|v|v.as_str()) {
                    crate::execution::record_content_decision(&self.root,&self.config.project_id,cand_id,binding,"critic_audit",&self.config.task_id,status=="approved"&&constraints_verified==1,rationale)?;
                }
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
            "register_dataset" => {
                let ds_id = args.get("dataset_id").or(args.get("id")).and_then(Value::as_str).ok_or("dataset_id missing")?;
                let path_str = args.get("path").or(args.get("source")).and_then(Value::as_str).ok_or("path missing")?;
                let src_path = if Path::new(path_str).is_absolute() {
                    std::path::PathBuf::from(path_str)
                } else {
                    match BranchWorkspaceManager::checked_path(&self.root, &self.config.project_id, &self.config.branch_id, path_str) {
                        Ok(p) => p,
                        Err(_) => self.root.join(path_str),
                    }
                };
                let receipt = crate::execution::register_dataset(&self.root, &self.config.project_id, ds_id, &src_path)?;
                Ok(receipt.to_string())
            }
            "schedule_specialist" => {
                let target_role = args["role_id"].as_str().ok_or("role_id missing")?;
                let intent = args["intent"].as_str().ok_or("intent missing")?;
                let hypothesis = args.get("hypothesis").and_then(|v| v.as_str());
                let sub_task_id = args.get("sub_task_id").or(args.get("child_task_id")).or(args.get("task_id")).and_then(Value::as_str);
                let specialist_branch = args.get("assigned_branch").or(args.get("branch")).and_then(Value::as_str);
                let receipt = self.schedule_specialist_with_ids(target_role, intent, hypothesis, sub_task_id, specialist_branch)?;
                Ok(receipt.to_string())
            }
            "delegate" | "delegate_task" => {
                let target_role = args["role_id"].as_str().ok_or("role_id missing")?;
                let intent = args["intent"].as_str().ok_or("intent missing")?;
                let hypothesis = args.get("hypothesis").and_then(|v| v.as_str());
                let sub_task_id = args.get("sub_task_id").or(args.get("child_task_id")).or(args.get("task_id")).and_then(Value::as_str);
                let specialist_branch = args.get("assigned_branch").or(args.get("branch")).and_then(Value::as_str);
                let scheduled = self.schedule_specialist_with_ids(target_role, intent, hypothesis, sub_task_id, specialist_branch)?;
                let sub_task_id = scheduled["sub_task_id"].as_str().ok_or("missing sub_task_id")?.to_string();
                let specialist_branch = scheduled["assigned_branch"].as_str().ok_or("missing assigned_branch")?.to_string();
                let run_summary = Self::run_scheduled_specialists(&self.root, &self.config.project_id, self.gateway.clone(), 3)?;
                let item = run_summary["specialists"]
                    .as_array()
                    .and_then(|arr| arr.iter().find(|x| x["sub_task_id"].as_str() == Some(sub_task_id.as_str())))
                    .cloned()
                    .unwrap_or_else(|| json!({
                        "status": "scheduled",
                        "error": Value::Null,
                        "sub_task_id": sub_task_id,
                        "role": target_role,
                        "assigned_branch": specialist_branch,
                        "turns_executed": 0,
                        "findings": [],
                        "achievements": [],
                    }));
                Ok(item.to_string())
            }
            "query_budget" => {
                let budget_info = AgentContextManager::query_authoritative_budget(&self.root, &self.config.project_id)?;
                Ok(budget_info.to_string())
            }
            "request_finalization" | "publish_delivery" | "publish_bundle" => {
                if self.is_research_closed()? {
                    return Err("Research admission is permanently closed; cannot re-finalize".into());
                }
                let cand_id = args["candidate_id"].as_str().ok_or("candidate_id missing")?;
                let rationale = args["rationale"].as_str().ok_or("rationale missing")?;

                let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;

                // Two-key validation check:
                // 1. Critic Audit check: must have an approved critic audit with constraints_verified=1
                let (audit_status, constraints_verified): (String, i64) = db.query_row(
                    "SELECT status, constraints_verified FROM critic_audits WHERE project=?1 AND candidate_id=?2 ORDER BY created_ms DESC LIMIT 1",
                    rusqlite::params![self.config.project_id, cand_id],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                ).map_err(|_| format!("Finalization refused: candidate '{cand_id}' has not received a critic audit in critic_audits"))?;

                if audit_status != "approved" || constraints_verified != 1 {
                    return Err(format!("Finalization refused: candidate '{cand_id}' critic audit status is '{audit_status}', constraints_verified={constraints_verified}. An approved audit with constraints_verified=1 is required."));
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
                let approved=crate::execution::require_finalization_decisions(&self.root,&self.config.project_id,cand_id)?;
                let pipeline_job_id=approved["binding"]["job"].as_str().ok_or("finalization binding missing pipeline job")?;
                let pipeline_view=crate::execution::inspect(&self.root,&self.config.project_id,pipeline_job_id)?;
                let crate::execution::Probe::Scientific{ref dataset_id,..}=pipeline_view.spec.probe else {
                    return Err("pipeline job probe is not a scientific probe".into());
                };
                let task_id=pipeline_view.spec.task_id.clone();
                let task_json:String=db.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",rusqlite::params![self.config.project_id,task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
                let mut task:Task=serde_json::from_str(&task_json).map_err(|e|e.to_string())?;
                let now=now_ms();
                if task.status!="active"||task.expires_ms.is_none_or(|end|end<now+120_000) {
                    let runtime=Runtime::open(&self.root)?;
                    let claim_receipt=runtime.command(Request{
                        command_id:format!("claim-finalize-{cand_id}-{now}"),
                        project_id:self.config.project_id.clone(),
                        expected_revision:None,
                        operation:Operation::Claim{
                            task_id:task_id.clone(),
                            owner:format!("finalizer-{}",self.config.task_id),
                            lease_id:format!("lease-finalize-{cand_id}"),
                            now_ms:now,
                            expires_ms:now+600_000,
                            expected_generation:task.generation,
                        },
                    })?;
                    if claim_receipt.status!="applied" {
                        return Err(format!("Could not acquire task lease for finalization: {}",claim_receipt.result));
                    }
                    let refreshed:String=db.query_row("SELECT data FROM tasks WHERE project=?1 AND id=?2",rusqlite::params![self.config.project_id,task_id],|r|r.get(0)).map_err(|e|e.to_string())?;
                    task=serde_json::from_str(&refreshed).map_err(|e|e.to_string())?;
                }

                // 3. Atomically close research admission before final test gate is reserved
                let changed = db.execute(
                    "UPDATE projects SET state='finalizing' WHERE id=?1 AND state NOT IN ('finalizing', 'completed')",
                    [&self.config.project_id],
                ).map_err(|e| e.to_string())?;
                if changed != 1 {
                    return Err("Research admission is permanently closed; cannot re-finalize".into());
                }

                // 4. Freeze exact source snapshot without automatic git merge
                let frozen_source_hash = BranchWorkspaceManager::compute_branch_hash(&self.root, &self.config.project_id, &self.config.branch_id)?;
                let snapshot_dest = self.root.join("workflows").join(&self.config.project_id).join("snapshots").join(&frozen_source_hash);
                let branch_src = BranchWorkspaceManager::get_branch_dir(&self.root, &self.config.project_id, &self.config.branch_id);
                if !snapshot_dest.exists() {
                    fs::create_dir_all(&snapshot_dest).map_err(|e| format!("cannot create snapshot directory: {e}"))?;
                    BranchWorkspaceManager::copy_dir_recursive(&branch_src, &snapshot_dest)?;
                    let copied_hash = BranchWorkspaceManager::compute_dir_hash(&snapshot_dest)?;
                    if copied_hash != frozen_source_hash {
                        return Err(format!("Snapshot copy verification failed: expected hash {frozen_source_hash}, got {copied_hash}"));
                    }
                }

                // Drain active exploratory jobs and tasks for this project
                db.execute(
                    "UPDATE jobs SET status=CASE WHEN status='queued' THEN 'cancelled' ELSE 'terminating' END, cancel_requested=1, reason='draining for finalization' WHERE project=?1 AND status IN ('queued', 'running')",
                    [&self.config.project_id],
                ).map_err(|e|e.to_string())?;
                db.execute(
                    "UPDATE agent_tasks SET status='cancelled', updated_ms=?2 WHERE project=?1 AND task_id!=?3 AND status IN ('paused', 'active')",
                    rusqlite::params![self.config.project_id, now_ms(), self.config.task_id],
                ).map_err(|e|e.to_string())?;

                let finalize_job_id=format!("finalize-{cand_id}");
                let test_access_id=args.get("test_access_id").and_then(|v|v.as_str()).map(str::to_string).unwrap_or_else(||format!("final-test-{cand_id}"));
                let finalize_spec=crate::execution::JobSpec{ resources: Default::default(),
                    job_id:finalize_job_id.clone(),
                    project_id:self.config.project_id.clone(),
                    task_id:task.task_id.clone(),
                    generation:task.generation,
                    timeout_ms:60_000,
                    threads:1,
                    max_output_bytes:4*1024*1024,
                    environment_id:pipeline_view.spec.environment_id.clone(),
                    probe:crate::execution::Probe::Scientific{
                        operation:"finalize".into(),
                        dataset_id:dataset_id.clone(),
                        spec:None,
                        candidate_id:Some(cand_id.to_string()),
                        test_access_id:Some(test_access_id.clone()),
                    },
                };
                crate::execution::submit(&self.root,finalize_spec)?;

                if let Ok(sup)=crate::execution::Supervisor::start(&self.root,1) {
                    sup.drain()?;
                    let fin_view=crate::execution::inspect(&self.root,&self.config.project_id,&finalize_job_id)?;
                    if fin_view.status!="succeeded" {
                        let gate_consumed:bool=db.query_row("SELECT EXISTS(SELECT 1 FROM final_test_gates WHERE project=?1 AND status='consumed')",[&self.config.project_id],|r|r.get(0)).unwrap_or(false);
                        if !gate_consumed {
                            let _=db.execute("UPDATE projects SET state='created' WHERE id=?1 AND state='finalizing'",[&self.config.project_id]);
                        }
                        return Err(format!("Finalization job '{}' failed: {:?}",finalize_job_id,fin_view.reason));
                    }
                    let bundle_dir=self.root.join("workflows").join(&self.config.project_id).join("bundle");
                    self.context.record_achievement(
                        &self.root,
                        "finalization_completed",
                        &format!("Content-approved candidate {cand_id} refit on train+val, evaluated on sealed test ({test_access_id}), and exported to verified bundle: {rationale}"),
                        None,
                        json!({"candidate_id": cand_id, "rationale": rationale, "frozen_source_hash": frozen_source_hash, "finalize_job_id": finalize_job_id, "test_access_id": test_access_id, "bundle_dir": bundle_dir}),
                    )?;
                    return Ok(json!({
                        "status": "finalized_and_delivered",
                        "candidate_id": cand_id,
                        "frozen_source_hash": frozen_source_hash,
                        "finalize_job_id": finalize_job_id,
                        "test_access_id": test_access_id,
                        "bundle_dir": bundle_dir,
                    }).to_string());
                }

                self.context.record_achievement(
                    &self.root,
                    "finalization_queued",
                    &format!("Content-approved candidate {cand_id} queued for finalization ({finalize_job_id}): {rationale}"),
                    None,
                    json!({"candidate_id": cand_id, "rationale": rationale, "frozen_source_hash": frozen_source_hash, "finalize_job_id": finalize_job_id, "test_access_id": test_access_id}),
                )?;

                Ok(json!({
                    "status": "finalization_queued",
                    "candidate_id": cand_id,
                    "frozen_source_hash": frozen_source_hash,
                    "finalize_job_id": finalize_job_id,
                    "test_access_id": test_access_id,
                }).to_string())
            }
            "register_feature_transform" => {
                let transform_id = args["transform_id"].as_str().ok_or("transform_id missing")?;
                let spec_val = &args["transform_spec"];
                let parent_dataset = args["parent_dataset"].as_str().ok_or("parent_dataset missing")?;
                let receipt = crate::execution::register_feature_transform(&self.root, &self.config.project_id, transform_id, spec_val, parent_dataset)?;
                Ok(receipt.to_string())
            }
            "register_calibration" => {
                let calib_id = args["calibration_id"].as_str().ok_or("calibration_id missing")?;
                let base_cand = args["base_candidate_id"].as_str().ok_or("base_candidate_id missing")?;
                let method = args["method"].as_str().ok_or("method missing")?;
                let brier = args.get("brier_score").and_then(|v| v.as_f64());
                let log_loss = args.get("log_loss").and_then(|v| v.as_f64());
                let receipt = crate::execution::register_calibration_record(&self.root, &self.config.project_id, calib_id, base_cand, method, brier, log_loss)?;
                Ok(receipt.to_string())
            }
            "register_ensemble" => {
                let ensemble_id = args["ensemble_id"].as_str().ok_or("ensemble_id missing")?;
                let recipe_type = args["recipe_type"].as_str().ok_or("recipe_type missing")?;
                let parent_ids: Vec<String> = args["parent_ids"].as_array().ok_or("parent_ids missing")?
                    .iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect();
                let weights: Option<Vec<f64>> = args.get("weights").and_then(|w| w.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect());
                let meta_params = args.get("meta_params");
                let is_feasible = args.get("is_feasible").and_then(|v| v.as_bool()).unwrap_or(false);
                let receipt = crate::execution::register_scientific_ensemble(
                    &self.root, &self.config.project_id, ensemble_id, recipe_type, &parent_ids,
                    weights.as_deref(), meta_params, is_feasible,
                )?;
                Ok(receipt.to_string())
            }
            "inspect_scientific_lineage" => {
                let lineage = crate::execution::inspect_scientific_lineage(&self.root, &self.config.project_id)?;
                Ok(lineage.to_string())
            }
            other => Err(format!("Unknown tool: {other}")),
        }
    }

    /// Durably enqueue a specialist subtask in `agent_tasks`, M0 (`CreateBranch`, `CreateTask`),
    /// `handoffs`, and `agent_checkpoints` so it can be scheduled and reconciled across restarts.
    pub fn schedule_specialist(
        &self,
        target_role: &str,
        intent: &str,
        hypothesis: Option<&str>,
    ) -> Result<Value> {
        self.schedule_specialist_with_ids(target_role, intent, hypothesis, None, None)
    }

    pub fn schedule_specialist_with_ids(
        &self,
        target_role: &str,
        intent: &str,
        hypothesis: Option<&str>,
        custom_sub_task_id: Option<&str>,
        custom_branch: Option<&str>,
    ) -> Result<Value> {
        if self.is_research_closed()? {
            return Err("research admission is closed".into());
        }
        AgentContextManager::enforce_exploratory_budget_reserve(
            &self.root,
            &self.config.project_id,
            &BTreeMap::from([
                ("calls".to_string(), 1),
                ("tokens".to_string(), 4_096),
                ("wall_ms".to_string(), 60_000),
            ]),
        )?;
        let specialist_role = get_role(target_role)
            .ok_or_else(|| format!("Unknown specialist role: {target_role}"))?;

        let delegation_id = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_nanos();
        let specialist_branch = custom_branch.map(str::to_string).unwrap_or_else(|| format!("{}-{}-{delegation_id}", self.config.branch_id, target_role));
        let sub_task_id = custom_sub_task_id.map(str::to_string).unwrap_or_else(|| format!("{}-{}-{delegation_id}", self.config.task_id, target_role));

        BranchWorkspaceManager::init_branch(&self.root, &self.config.project_id, &specialist_branch, &self.config.branch_id)?;
        let parent_dir = BranchWorkspaceManager::get_branch_dir(&self.root, &self.config.project_id, &self.config.branch_id);
        let child_dir = BranchWorkspaceManager::get_branch_dir(&self.root, &self.config.project_id, &specialist_branch);
        let parent_hash = BranchWorkspaceManager::compute_dir_hash(&parent_dir)?;
        BranchWorkspaceManager::copy_dir_recursive(&parent_dir, &child_dir)?;
        if BranchWorkspaceManager::compute_dir_hash(&child_dir)? != parent_hash {
            return Err("Parent workspace changed during specialist snapshot copy".into());
        }

        let now = now_ms();
        crate::execution::migrate(&self.root)?;
        let db = rusqlite::Connection::open(self.root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO agent_tasks(project, task_id, role_id, parent_task_id, status, hypothesis, intent, assigned_branch, created_ms, updated_ms)
             VALUES(?1, ?2, ?3, ?4, 'paused', ?5, ?6, ?7, ?8, ?8)
             ON CONFLICT(project, task_id) DO UPDATE SET status='paused', updated_ms=?8",
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

        let handoff_id = format!("hoff-{delegation_id}");
        db.execute(
            "INSERT INTO handoffs(project, handoff_id, task_id, from_role, to_role, version, status, summary, context_payload, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, 'pending', ?7, ?8, ?9)",
            rusqlite::params![
                self.config.project_id,
                handoff_id,
                sub_task_id,
                self.config.role_id,
                target_role,
                self.context.handoff_version as i64,
                intent,
                json!({"parent_task_id": self.config.task_id, "hypothesis": hypothesis, "branch": specialist_branch}).to_string(),
                now,
            ],
        ).map_err(|e| e.to_string())?;

        let runtime = Runtime::open(&self.root)?;
        let source_hash = BranchWorkspaceManager::compute_branch_hash(&self.root, &self.config.project_id, &specialist_branch)?;
        let context = crate::contracts::Context {
            role_id: target_role.to_string(),
            role_version: specialist_role.version.clone(),
            prompt_hash: "hash".into(),
            model: Some(self.config.model.clone()),
            phase: "delegated".into(),
            hypothesis: hypothesis.map(|h| h.to_string()),
            base_hash: source_hash.clone(),
            code_hash: source_hash.clone(),
            evidence: vec![],
            artifacts: vec![],
            pending_questions: vec![],
            calls: vec![],
            jobs: vec![],
            next_action: "specialist_investigation".into(),
            stop_conditions: vec![],
        };
        for (suffix, operation) in [
            ("branch", Operation::CreateBranch { branch_id: specialist_branch.clone(), base_hash: source_hash.clone(), code_hash: source_hash }),
            ("task", Operation::CreateTask { task_id: sub_task_id.clone(), branch_id: specialist_branch.clone(), context }),
        ] {
            let receipt = runtime.command(Request {
                command_id: format!("schedule-spec-{delegation_id}-{suffix}"),
                project_id: self.config.project_id.clone(),
                expected_revision: None,
                operation,
            })?;
            if receipt.status != "applied" {
                return Err(format!("Specialist admission rejected: {:?}", receipt));
            }
        }
        drop(runtime);

        // Record initial durable checkpoint for the specialist so it can resume cleanly even after restart
        let mut init_ctx = AgentContextManager::new(
            &self.config.project_id,
            &sub_task_id,
            target_role,
            specialist_role.system_prompt,
        );
        if let Some(h) = hypothesis {
            init_ctx.set_hypothesis(h);
        }
        init_ctx.add_message(ChatMessage::user(format!(
            "Delegated specialist task for role {target_role}.\nIntent: {intent}\nHypothesis: {}\nPlease perform your specialist analysis and record findings or tools.",
            hypothesis.unwrap_or("none")
        )));
        init_ctx.record_checkpoint(&self.root)?;

        Ok(json!({
            "status": "scheduled",
            "sub_task_id": sub_task_id,
            "role": target_role,
            "assigned_branch": specialist_branch,
            "handoff_id": handoff_id,
        }))
    }

    /// Execute and reconcile all queued/active scheduled specialist tasks for a project.
    pub fn run_scheduled_specialists(
        root: &Path,
        project_id: &str,
        gateway: Arc<dyn ProviderGateway>,
        max_turns_per_specialist: u32,
    ) -> Result<Value> {
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let mut stmt = db.prepare(
            "SELECT task_id, role_id, parent_task_id, status, hypothesis, intent, assigned_branch
             FROM agent_tasks
             WHERE project=?1 AND parent_task_id IS NOT NULL AND status IN ('paused', 'active')
             ORDER BY created_ms ASC"
        ).map_err(|e| e.to_string())?;
        let tasks: Vec<(String, String, String, String, Option<String>, String, String)> = stmt.query_map(
            [project_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)),
        ).map_err(|e| e.to_string())?.collect::<rusqlite::Result<Vec<_>>>().map_err(|e|e.to_string())?;
        drop(stmt);

        let mut results = Vec::new();
        for (sub_task_id, target_role, parent_task_id, _prior_status, hypothesis, intent, specialist_branch) in tasks {
            let parent_model: String = db.query_row(
                "SELECT data FROM tasks WHERE project=?1 AND id=?2",
                rusqlite::params![project_id, sub_task_id],
                |r| r.get::<_, String>(0),
            ).ok()
            .and_then(|s| serde_json::from_str::<Task>(&s).ok())
            .and_then(|t| t.context.model)
            .unwrap_or_else(|| "local-deterministic-v1".into());

            db.execute(
                "UPDATE agent_tasks SET status='active', updated_ms=?1 WHERE project=?2 AND task_id=?3",
                rusqlite::params![now_ms(), project_id, sub_task_id],
            ).map_err(|e| e.to_string())?;

            let specialist_config = OrchestratorConfig {
                project_id: project_id.to_string(),
                task_id: sub_task_id.clone(),
                role_id: target_role.clone(),
                branch_id: specialist_branch.clone(),
                model: parent_model,
                max_turns: max_turns_per_specialist,
            };
            let mut specialist = AgentOrchestrator::new(root, specialist_config, gateway.clone())?;
            // First reconcile any interrupted turn on this specialist task
            let reconciled = specialist.reconcile()?;
            if specialist.context.messages.len() <= 1 {
                specialist.context.add_message(ChatMessage::user(format!(
                    "Delegated specialist task for role {target_role}.\nIntent: {intent}\nHypothesis: {}\nPlease perform your specialist analysis and record findings or tools.",
                    hypothesis.as_deref().unwrap_or("none")
                )));
                specialist.context.record_checkpoint(root)?;
            }

            let mut spec_turns = specialist.current_turn;
            let mut specialist_error = None;
            let mut specialist_stopped = false;
            while spec_turns < specialist.config.max_turns {
                match specialist.step() {
                    Ok(res) => {
                        spec_turns = res.turn_index;
                        if res.stopped {
                            specialist_stopped = true;
                            break;
                        }
                    }
                    Err(error) => {
                        specialist_error = Some(error);
                        break;
                    }
                }
            }

            let findings: Vec<String> = db.prepare(
                "SELECT finding_id, verdict, rationale FROM scientific_findings WHERE project=?1 AND task_id=?2"
            ).map_err(|e| e.to_string())?
            .query_map(rusqlite::params![project_id, sub_task_id], |r| {
                Ok(format!("{}: verdict={} rationale={}", r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
            }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();

            let achievements: Vec<String> = db.prepare(
                "SELECT achievement_id, kind, summary FROM achievements WHERE project=?1 AND task_id=?2"
            ).map_err(|e| e.to_string())?
            .query_map(rusqlite::params![project_id, sub_task_id], |r| {
                Ok(format!("{}: kind={} summary={}", r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
            }).map_err(|e| e.to_string())?.filter_map(|r| r.ok()).collect();

            let end_now = now_ms();
            let outcome = if specialist_error.is_some() {
                "failed"
            } else if specialist_stopped {
                "completed"
            } else {
                "turn_limit"
            };
            let durable_status = if outcome == "failed" { "failed" } else { "completed" };
            db.execute(
                "UPDATE agent_tasks SET status=?1, updated_ms=?2 WHERE project=?3 AND task_id=?4",
                rusqlite::params![durable_status, end_now, project_id, sub_task_id],
            ).map_err(|e| e.to_string())?;
            db.execute(
                "UPDATE handoffs SET status=?1, context_payload=?2 WHERE project=?3 AND task_id=?4 AND status='pending'",
                rusqlite::params![
                    if outcome == "failed" { "rejected" } else { "accepted" },
                    json!({
                        "parent_task_id": parent_task_id,
                        "outcome": outcome,
                        "turns_executed": spec_turns,
                        "findings": findings,
                        "achievements": achievements,
                        "error": specialist_error,
                    }).to_string(),
                    project_id,
                    sub_task_id,
                ],
            ).map_err(|e| e.to_string())?;

            results.push(json!({
                "status": outcome,
                "durable_status": durable_status,
                "error": specialist_error,
                "sub_task_id": sub_task_id,
                "parent_task_id": parent_task_id,
                "role": target_role,
                "assigned_branch": specialist_branch,
                "turns_executed": spec_turns,
                "reconciled": reconciled["reconciled_turns"],
                "findings": findings,
                "achievements": achievements,
            }));
        }

        Ok(json!({
            "project_id": project_id,
            "specialists_processed": results.len(),
            "specialists": results,
        }))
    }
}
