use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::{
    agents::gateway::ChatMessage,
    Result,
};

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn hash_str(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Achievement {
    pub achievement_id: String,
    pub task_id: String,
    pub kind: String,
    pub summary: String,
    pub evidence_sha256: Option<String>,
    pub metrics: Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ActiveJobReceipt {
    pub job_id: String,
    pub task_id: String,
    pub operation: String,
    pub status: String,
    pub started_ms: i64,
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ScientificFinding {
    pub finding_id: String,
    pub task_id: String,
    pub hypothesis_id: String,
    pub verdict: String, // "supported", "rejected", "inconclusive"
    pub rationale: String,
    pub metrics: Value,
    pub artifacts: Vec<String>,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Handoff {
    pub handoff_id: String,
    pub task_id: String,
    pub from_role: String,
    pub to_role: String,
    pub version: u32,
    pub status: String, // "pending", "accepted", "rejected", "superseded"
    pub summary: String,
    pub context_payload: Value,
    pub created_ms: i64,
}

#[derive(Debug, Clone)]
pub struct AgentContextManager {
    pub project_id: String,
    pub task_id: String,
    pub role_id: String,
    pub max_context_tokens: u32,
    pub reserve_tokens: u32,
    pub messages: Vec<ChatMessage>,
    pub achievements: Vec<Achievement>,
    pub active_jobs: Vec<ActiveJobReceipt>,
    pub findings: Vec<ScientificFinding>,
    pub current_hypothesis: Option<String>,
    pub current_steering: Option<String>,
    pub final_test_identity: Option<String>,
    pub constraints: Option<String>,
    pub approvals: Vec<String>,
    pub handoff_version: u32,
}

impl AgentContextManager {
    pub fn new(
        project_id: impl Into<String>,
        task_id: impl Into<String>,
        role_id: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            task_id: task_id.into(),
            role_id: role_id.into(),
            max_context_tokens: 16_384,
            reserve_tokens: 4_096,
            messages: vec![ChatMessage::system(system_prompt.into())],
            achievements: Vec::new(),
            active_jobs: Vec::new(),
            findings: Vec::new(),
            current_hypothesis: None,
            current_steering: None,
            final_test_identity: None,
            constraints: None,
            approvals: Vec::new(),
            handoff_version: 1,
        }
    }

    pub fn estimate_tokens(&self) -> u32 {
        let total_chars: usize = self.messages.iter()
            .map(|m| {
                let c_len = m.content.as_ref().map(|s| s.len()).unwrap_or(0);
                let t_len = m.tool_calls.as_ref().map(|calls| {
                    calls.iter().map(|c| c.function.name.len() + c.function.arguments.len()).sum::<usize>()
                }).unwrap_or(0);
                c_len + t_len + 10
            })
            .sum();
        (total_chars / 4) as u32
    }

    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
    }

    pub fn set_hypothesis(&mut self, hyp: impl Into<String>) {
        self.current_hypothesis = Some(hyp.into());
    }

    pub fn record_achievement(
        &mut self,
        root: &Path,
        kind: &str,
        summary: &str,
        evidence_sha256: Option<String>,
        metrics: Value,
    ) -> Result<Achievement> {
        let now = now_ms();
        let ach_id = format!("ach-{}", &hash_str(&format!("{}:{}:{}:{now}", self.task_id, kind, summary))[..12]);
        let ach = Achievement {
            achievement_id: ach_id.clone(),
            task_id: self.task_id.clone(),
            kind: kind.to_string(),
            summary: summary.to_string(),
            evidence_sha256,
            metrics: metrics.clone(),
            created_ms: now,
        };
        self.achievements.push(ach.clone());

        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO achievements(project, achievement_id, task_id, kind, summary, evidence_sha256, metrics, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                self.project_id,
                ach.achievement_id,
                self.task_id,
                ach.kind,
                ach.summary,
                ach.evidence_sha256,
                serde_json::to_string(&metrics).unwrap_or_default(),
                now,
            ],
        ).map_err(|e| e.to_string())?;

        if let Some(ref sha) = ach.evidence_sha256 {
            let _ = db.execute(
                "INSERT INTO refs(project, id, category, digest) VALUES(?1, ?2, 'evidence', ?3) ON CONFLICT DO NOTHING",
                rusqlite::params![self.project_id, ach.achievement_id, sha],
            );
        }

        // Achievement boundaries trigger checkpoint and compaction check
        self.record_checkpoint(root)?;
        self.maybe_compact();

        Ok(ach)
    }

    pub fn record_finding(
        &mut self,
        root: &Path,
        hypothesis_id: &str,
        verdict: &str,
        rationale: &str,
        metrics: Value,
        artifacts: Vec<String>,
    ) -> Result<ScientificFinding> {
        let now = now_ms();
        let finding_id = format!("find-{}", &hash_str(&format!("{}:{}:{now}", self.task_id, hypothesis_id))[..12]);
        let finding = ScientificFinding {
            finding_id: finding_id.clone(),
            task_id: self.task_id.clone(),
            hypothesis_id: hypothesis_id.to_string(),
            verdict: verdict.to_string(),
            rationale: rationale.to_string(),
            metrics: metrics.clone(),
            artifacts: artifacts.clone(),
            created_ms: now,
        };
        self.findings.push(finding.clone());

        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO scientific_findings(project, finding_id, task_id, hypothesis_id, verdict, rationale, metrics, artifacts, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                self.project_id,
                finding.finding_id,
                self.task_id,
                finding.hypothesis_id,
                finding.verdict,
                finding.rationale,
                serde_json::to_string(&metrics).unwrap_or_default(),
                serde_json::to_string(&artifacts).unwrap_or_default(),
                now,
            ],
        ).map_err(|e| e.to_string())?;

        Ok(finding)
    }

    pub fn record_checkpoint(&mut self, root: &Path) -> Result<String> {
        let now = now_ms();
        let tokens = self.estimate_tokens();
        let nonce=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
        let chk_id = format!("chk-{}", &hash_str(&format!("{}:{}:{nonce}", self.task_id, self.handoff_version))[..24]);

        let state_snapshot = json!({
            "hypothesis": self.current_hypothesis,
            "achievements": self.achievements,
            "findings": self.findings,
            "messages": self.messages,
            "active_jobs": self.active_jobs,
            "role_id": self.role_id,
            "handoff_version": self.handoff_version,
            "current_steering": self.current_steering,
            "final_test_identity": self.final_test_identity,
            "constraints": self.constraints,
            "approvals": self.approvals,
            "achievements_count": self.achievements.len(),
            "findings_count": self.findings.len(),
            "messages_count": self.messages.len(),
        });

        let snapshot_json = serde_json::to_string(&state_snapshot).unwrap_or_default();
        let serialized_len = snapshot_json.len() as u64;
        let active_jobs_json = serde_json::to_string(&self.active_jobs).unwrap_or_default();

        crate::execution::migrate(root)?;
        let mut connection=rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e|e.to_string())?;
        connection.busy_timeout(std::time::Duration::from_secs(10)).map_err(|e|e.to_string())?;
        let db=connection.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate).map_err(|e|e.to_string())?;
        crate::execution::enforce_state_write_admission(root, Some(&db), serialized_len, crate::execution::StorageWriteKind::Checkpoint, Some(&self.project_id))?;

        db.execute(
            "INSERT INTO agent_checkpoints(project, checkpoint_id, task_id, role_id, handoff_version, context_tokens, state_snapshot, active_jobs, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                self.project_id,
                chk_id,
                self.task_id,
                self.role_id,
                self.handoff_version as i64,
                tokens as i64,
                snapshot_json,
                active_jobs_json,
                now,
            ],
        ).map_err(|e| e.to_string())?;

         db.execute(
            "DELETE FROM agent_checkpoints WHERE project=?1 AND task_id=?2 AND rowid NOT IN (
                SELECT rowid FROM agent_checkpoints WHERE project=?1 AND task_id=?2 ORDER BY created_ms DESC, rowid DESC LIMIT ?3
            )",
            rusqlite::params![self.project_id, self.task_id, crate::execution::MAX_RETAINED_CHECKPOINTS_PER_TASK],
        ).map_err(|e|e.to_string())?;

        db.commit().map_err(|e|e.to_string())?;
        Ok(chk_id)
    }

    /// Structured compaction: preserves system prompt, active hypothesis, all achievements,
    /// scientific findings, current steering, constraints, approvals, and recent conversation turns
    /// while collapsing older tool interactions and keeping tool-call pairings intact.
    pub fn maybe_compact(&mut self) -> bool {
        let current_tokens = self.estimate_tokens();
        let target_limit = self.max_context_tokens.saturating_sub(self.reserve_tokens);

        if current_tokens <= target_limit || self.messages.len() <= 4 {
            return false;
        }

        let system_msg = self.messages[0].clone();

        // Build structured compaction context preserving steering, constraints, approvals, and final test identity
        let mut summary_lines = Vec::new();
        summary_lines.push("### Context Compaction & Authoritative State Summary".to_string());
        if let Some(hyp) = &self.current_hypothesis {
            summary_lines.push(format!("Active Hypothesis: {hyp}"));
        }
        if let Some(steer) = &self.current_steering {
            summary_lines.push(format!("Current Steering: {steer}"));
        }
        if let Some(c) = &self.constraints {
            summary_lines.push(format!("Constraints: {c}"));
        }
        if let Some(fti) = &self.final_test_identity {
            summary_lines.push(format!("Final Test Identity: {fti}"));
        }
        if !self.approvals.is_empty() {
            summary_lines.push(format!("Registered Approvals: {}", self.approvals.join(", ")));
        }
        if !self.achievements.is_empty() {
            summary_lines.push("Committed Achievements:".to_string());
            for ach in &self.achievements {
                let sha_suffix = ach.evidence_sha256.as_deref().map(|s| format!(" (evidence_sha256: {s})")).unwrap_or_default();
                summary_lines.push(format!("- [{}] {}: {}{sha_suffix}", ach.kind, ach.achievement_id, ach.summary));
            }
        }
        if !self.findings.is_empty() {
            summary_lines.push("Scientific Findings:".to_string());
            for f in &self.findings {
                summary_lines.push(format!("- [Verdict: {}] {}: {}", f.verdict, f.hypothesis_id, f.rationale));
            }
        }
        if !self.active_jobs.is_empty() {
            summary_lines.push("Active/Pending Jobs:".to_string());
            for j in &self.active_jobs {
                summary_lines.push(format!("- Job {} ({}) status: {}", j.job_id, j.operation, j.status));
            }
        }

        if let Some(user_msg) = self.messages.iter().skip(1).find(|m| m.role == "user") {
            if let Some(content) = &user_msg.content {
                summary_lines.push(format!("Original User Request & Constraints: {}", content.chars().take(300).collect::<String>()));
            }
        }
        let compaction_msg = ChatMessage::user(summary_lines.join("\n"));

        // Determine safe recent slice without breaking tool pairs
        let mut slice_start = self.messages.len().saturating_sub(4).max(1);
        while slice_start > 1 && self.messages[slice_start].role == "tool" {
            slice_start -= 1;
        }

        let recent_slice = self.messages[slice_start..].to_vec();
        let mut compacted = vec![system_msg, compaction_msg];
        compacted.extend(recent_slice);
        self.messages = compacted;
        true
    }

    /// Invalidate active handoffs, inject steering directive, and update handoff version.
    pub fn apply_steering(&mut self, root: &Path, directive: &str) -> Result<()> {
        self.apply_steering_with_actor(root, directive, "human:operator")
    }

    pub fn apply_steering_with_actor(&mut self, root: &Path, directive: &str, actor: &str) -> Result<()> {
        self.handoff_version += 1;

        // Invalidate prior pending handoffs in SQLite
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "UPDATE handoffs SET status='superseded' WHERE project=?1 AND task_id=?2 AND status='pending'",
            rusqlite::params![self.project_id, self.task_id],
        ).map_err(|e| e.to_string())?;

        // Durably record steering directive
        let directive_id = format!("steer-{}", &hash_str(&format!("{}:{}:{}:{}", self.task_id, self.handoff_version, actor, directive))[..16]);
        let _ = super::task_contract::record_steering_directive(root, &self.project_id, &directive_id, &self.task_id, actor, self.handoff_version as u64, directive);
        let _ = super::task_contract::mark_steering_directive_applied(root, &self.project_id, &directive_id);
        self.current_steering = Some(directive.to_string());

        // Inject steering guidance at turn boundary
        self.messages.push(ChatMessage::user(format!(
            "[STEERING DIRECTIVE - Turn Boundary (Epoch v{}) by {actor}]:\n{directive}\nPrevious uncommitted proposals superseded.",
            self.handoff_version
        )));

        self.record_checkpoint(root)?;
        Ok(())
    }

    /// Retrieve an earlier failed approach or finding by keyword.
    pub fn retrieve_historical_findings(&self, keyword: &str) -> Vec<ScientificFinding> {
        self.findings.iter()
            .filter(|f| f.hypothesis_id.contains(keyword) || f.rationale.contains(keyword) || f.verdict.contains(keyword))
            .cloned()
            .collect()
    }

    /// Reconstruct context state from latest checkpoint, verifying durable evidence hashes.
    pub fn restore(root: &Path, project_id: &str, task_id: &str) -> Result<Self> {
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;

        let (role_id, handoff_version, snapshot_json): (String, i64, String) = db.query_row(
            "SELECT role_id, handoff_version, state_snapshot FROM agent_checkpoints WHERE project=?1 AND task_id=?2 ORDER BY created_ms DESC LIMIT 1",
            rusqlite::params![project_id, task_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).map_err(|_| format!("No checkpoint found for task {task_id} in project {project_id}"))?;

        let snapshot: Value = serde_json::from_str(&snapshot_json).map_err(|e| e.to_string())?;

        let hypothesis = snapshot.get("hypothesis").and_then(|v| v.as_str()).map(|s| s.to_string());
        let achievements: Vec<Achievement> = snapshot.get("achievements")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let findings: Vec<ScientificFinding> = snapshot.get("findings")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let messages: Vec<ChatMessage> = snapshot.get("messages")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();
        let active_jobs: Vec<ActiveJobReceipt> = snapshot.get("active_jobs")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // Verify evidence integrity
        for ach in &achievements {
            if let Some(ref sha) = ach.evidence_sha256 {
                let evidence_path_1 = root.join(".m2/evidence").join(sha);
                let evidence_path_2 = root.join("workflows").join(project_id).join("evidence").join(sha);
                let evidence_file = if evidence_path_1.is_file() {
                    Some(evidence_path_1)
                } else if evidence_path_2.is_file() {
                    Some(evidence_path_2)
                } else {
                    None
                };

                if let Some(p) = evidence_file {
                    let content = std::fs::read(&p).map_err(|e| e.to_string())?;
                    let actual_sha = format!("{:x}", Sha256::digest(&content));
                    if &actual_sha != sha {
                        return Err(format!("Durable evidence hash mismatch for achievement {}: expected {sha}, found {actual_sha}", ach.achievement_id));
                    }
                } else {
                    return Err(format!("Referenced evidence {sha} for achievement {} is missing from managed storage; a refs row cannot verify missing bytes", ach.achievement_id));
                }
            }
        }
        let current_steering = snapshot.get("current_steering").and_then(|v| v.as_str()).map(|s| s.to_string());
        let final_test_identity = snapshot.get("final_test_identity").and_then(|v| v.as_str()).map(|s| s.to_string());
        let constraints = snapshot.get("constraints").and_then(|v| v.as_str()).map(|s| s.to_string());
        let approvals: Vec<String> = snapshot.get("approvals")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let mut mgr = Self {
            project_id: project_id.to_string(),
            task_id: task_id.to_string(),
            role_id,
            max_context_tokens: 16_384,
            reserve_tokens: 4_096,
            messages,
            achievements,
            active_jobs,
            findings,
            current_hypothesis: hypothesis,
            current_steering,
            final_test_identity,
            constraints,
            approvals,
            handoff_version: handoff_version as u32,
        };
        mgr.ensure_context_within_bounds()?;
        Ok(mgr)
    }

    /// Reload and calculate authoritative remaining allowances from M0 reservations and charges.
    pub fn query_authoritative_budget(root: &Path, project_id: &str) -> Result<Value> {
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let policy_str: String = db.query_row(
            "SELECT policy FROM projects WHERE id=?1",
            [project_id],
            |r| r.get(0),
        ).map_err(|_| format!("Project {project_id} not found"))?;
        let policy: Value = serde_json::from_str(&policy_str).map_err(|e| e.to_string())?;
        let total_wall = policy.get("budget").and_then(|b| b.get("wall_ms")).and_then(|v| v.as_i64()).unwrap_or(300_000);
        let total_tokens = policy.get("budget").and_then(|b| b.get("tokens")).and_then(|v| v.as_i64()).unwrap_or(1_000_000);
        let total_calls = policy.get("budget").and_then(|b| b.get("calls")).and_then(|v| v.as_i64()).unwrap_or(100);
        let total_cost = policy.get("budget").and_then(|b| b.get("cost_millicents")).and_then(|v| v.as_i64());

        let sum_dim = |table: &str, col: &str, filter: &str, dim: &str| -> Result<i64> {
            db.query_row(
                &format!("SELECT COALESCE(SUM(CAST(json_extract({col},'$.{dim}') AS INTEGER)),0) FROM {table} WHERE {filter} AND project=?1"),
                [project_id],
                |r| r.get(0),
            ).map_err(|e| e.to_string())
        };

        let held_wall = sum_dim("reservations", "budget", "held=1", "wall_ms")?;
        let charged_wall = sum_dim("charges", "amount", "1=1", "wall_ms")?;
        let held_tokens = sum_dim("reservations", "budget", "held=1", "tokens")?;
        let charged_tokens = sum_dim("charges", "amount", "1=1", "tokens")?;
        let held_calls = sum_dim("reservations", "budget", "held=1", "calls")?;
        let charged_calls = sum_dim("charges", "amount", "1=1", "calls")?;
        let held_cost = sum_dim("reservations", "budget", "held=1", "cost_millicents")?;
        let charged_cost = sum_dim("charges", "amount", "1=1", "cost_millicents")?;

        let finalization_reserve_wall = (total_wall as f64 * 0.20) as i64;
        let finalization_reserve_tokens = (total_tokens as f64 * 0.20) as i64;
        let finalization_reserve_calls = (total_calls as f64 * 0.20) as i64;
        let finalization_reserve_cost = total_cost.map(|c| (c as f64 * 0.20) as i64);

        let remaining_wall = total_wall.saturating_sub(held_wall + charged_wall);
        let remaining_wall_effective = remaining_wall.saturating_sub(finalization_reserve_wall);
        let remaining_tokens = total_tokens.saturating_sub(held_tokens + charged_tokens);
        let remaining_tokens_effective = remaining_tokens.saturating_sub(finalization_reserve_tokens);
        let remaining_calls = total_calls.saturating_sub(held_calls + charged_calls);
        let remaining_calls_effective = remaining_calls.saturating_sub(finalization_reserve_calls);
        let remaining_cost = total_cost.map(|c| c.saturating_sub(held_cost + charged_cost));
        let remaining_cost_effective = remaining_cost.zip(finalization_reserve_cost).map(|(r, res)| r.saturating_sub(res));

        Ok(json!({
            "project_id": project_id,
            "policy": policy,
            "budget": {
                "wall_ms": total_wall,
                "tokens": total_tokens,
                "calls": total_calls,
                "cost_millicents": total_cost,
            },
            "consumption": {
                "wall_ms_held": held_wall,
                "wall_ms_charged": charged_wall,
                "tokens_held": held_tokens,
                "tokens_charged": charged_tokens,
                "calls_held": held_calls,
                "calls_charged": charged_calls,
                "cost_millicents_held": held_cost,
                "cost_millicents_charged": charged_cost,
            },
            "remaining": {
                "wall_ms_gross": remaining_wall,
                "wall_ms_effective": remaining_wall_effective,
                "tokens": remaining_tokens,
                "tokens_effective": remaining_tokens_effective,
                "calls_gross": remaining_calls,
                "calls_effective": remaining_calls_effective,
                "cost_millicents_gross": remaining_cost,
                "cost_millicents_effective": remaining_cost_effective,
            },
            "finalization_reserve": {
                "wall_ms": finalization_reserve_wall,
                "tokens": finalization_reserve_tokens,
                "calls": finalization_reserve_calls,
                "cost_millicents": finalization_reserve_cost,
            },
            "finalization_reserve_wall_ms": finalization_reserve_wall,
        }))
    }

    /// Verify that an exploratory action does not breach the protected 20% finalization reserve.
    pub fn enforce_exploratory_budget_reserve(
        root: &Path,
        project_id: &str,
        requested: &std::collections::BTreeMap<String, i64>,
    ) -> Result<()> {
        let info = Self::query_authoritative_budget(root, project_id)?;
        let rem = &info["remaining"];
        for (dim, &req_val) in requested {
            if req_val <= 0 {
                continue;
            }
            let effective_key = match dim.as_str() {
                "wall_ms" => "wall_ms_effective",
                "tokens" => "tokens_effective",
                "calls" => "calls_effective",
                "cost_millicents" => "cost_millicents_effective",
                _ => continue,
            };
            if let Some(avail) = rem.get(effective_key).and_then(|v| v.as_i64()) {
                if req_val > avail {
                    return Err(format!(
                        "Exploratory admission blocked (budget exhausted): requested {dim}={req_val} exceeds exploratory allowance {avail} (20% finalization reserve protected)"
                    ));
                }
            }
        }
        Ok(())
    }

    /// Configure context window bounds, enforcing economical default (~16K) with configurable upper ceiling <= 64K (65,536).
    pub fn set_context_window(&mut self, max_tokens: u32, reserve_tokens: u32) -> Result<()> {
        if max_tokens > 65_536 {
            return Err("Configured context window exceeds hard upper ceiling of 64K (65,536 tokens)".into());
        }
        if reserve_tokens >= max_tokens {
            return Err("Reserve tokens must be strictly less than max context tokens".into());
        }
        self.max_context_tokens = max_tokens;
        self.reserve_tokens = reserve_tokens;
        Ok(())
    }

    /// Reject broken JSON arguments or broken tool pairings in messages.
    pub fn validate_tool_pairings(&self) -> Result<()> {
        let mut open_tool_calls: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

        for (idx, msg) in self.messages.iter().enumerate() {
            if msg.role == "assistant" {
                if let Some(ref calls) = msg.tool_calls {
                    for call in calls {
                        // Check arguments are valid JSON
                        if let Err(e) = serde_json::from_str::<Value>(&call.function.arguments) {
                            return Err(format!("Tool call '{}' at index {idx} contains invalid JSON arguments: {e}", call.id));
                        }
                        open_tool_calls.insert(call.id.clone());
                    }
                }
            } else if msg.role == "tool" {
                let call_id = msg.tool_call_id.as_deref().ok_or_else(|| {
                    format!("Tool message at index {idx} is missing tool_call_id")
                })?;
                if !open_tool_calls.remove(call_id) {
                    return Err(format!("Tool response for '{call_id}' at index {idx} does not match any preceding active tool call"));
                }
            }
        }
        Ok(())
    }

    /// Refuse dispatch when essential state (system prompt, schemas, constraints, approvals, reserves) cannot fit.
    pub fn check_essential_state_fits(&self, tools: &[crate::agents::gateway::ToolDefinition]) -> Result<()> {
        let tools_json = serde_json::to_string(tools).unwrap_or_default();
        let tools_tokens = (tools_json.len().div_ceil(4)) as u32;

        let system_tokens = self.messages.first()
            .and_then(|m| m.content.as_ref())
            .map(|s| s.len().div_ceil(4) as u32)
            .unwrap_or(0);

        let steering_tokens = self.current_steering.as_ref().map(|s| s.len().div_ceil(4) as u32).unwrap_or(0);
        let constraints_tokens = self.constraints.as_ref().map(|s| s.len().div_ceil(4) as u32).unwrap_or(0);
        let final_test_tokens = self.final_test_identity.as_ref().map(|s| s.len().div_ceil(4) as u32).unwrap_or(0);

        let essential_tokens = system_tokens + tools_tokens + steering_tokens + constraints_tokens + final_test_tokens + self.reserve_tokens;

        if essential_tokens > self.max_context_tokens {
            return Err(format!(
                "Essential state ({} tokens: system={}, tools={}, reserves={}, steering/constraints={}) cannot fit within configured context window ({} tokens); refusing oversized dispatch",
                essential_tokens, system_tokens, tools_tokens, self.reserve_tokens, steering_tokens + constraints_tokens + final_test_tokens, self.max_context_tokens
            ));
        }
        Ok(())
    }

    /// Enforce hard token ceiling, spilling older evidence/findings to durable IDs,
    /// or refusing oversized dispatch without dropping essential constraints/approvals.
    pub fn ensure_context_within_bounds(&mut self) -> Result<()> {
        self.validate_tool_pairings()?;

        let target_limit = self.max_context_tokens.saturating_sub(self.reserve_tokens);
        if self.estimate_tokens() <= target_limit {
            return Ok(());
        }

        // Run structured compaction
        self.maybe_compact();

        if self.estimate_tokens() <= target_limit {
            return Ok(());
        }

        // Spill historical evidence/findings summaries to durable retrieval IDs
        let mut spilled_lines = Vec::new();
        spilled_lines.push("### Context Compaction (Historical Evidence Spilled to Durable IDs)".to_string());
        if let Some(hyp) = &self.current_hypothesis {
            spilled_lines.push(format!("Active Hypothesis: {hyp}"));
        }
        if let Some(steer) = &self.current_steering {
            spilled_lines.push(format!("Current Steering: {steer}"));
        }
        if let Some(c) = &self.constraints {
            spilled_lines.push(format!("Constraints: {c}"));
        }
        if let Some(fti) = &self.final_test_identity {
            spilled_lines.push(format!("Final Test Identity: {fti}"));
        }
        if !self.approvals.is_empty() {
            spilled_lines.push(format!("Registered Approvals: {}", self.approvals.join(", ")));
        }
        if !self.achievements.is_empty() {
            spilled_lines.push("Committed Achievement References:".to_string());
            for ach in &self.achievements {
                spilled_lines.push(format!("- [{}] id: {} (use retrieve_evidence tool with id='{}' to inspect details)", ach.kind, ach.achievement_id, ach.achievement_id));
            }
        }
        if !self.findings.is_empty() {
            spilled_lines.push("Scientific Finding References:".to_string());
            for f in &self.findings {
                spilled_lines.push(format!("- [Verdict: {}] id: {} (use retrieve_evidence tool with id='{}' to inspect details)", f.verdict, f.finding_id, f.finding_id));
            }
        }

        let compaction_msg = ChatMessage::user(spilled_lines.join("\n"));
        let system_msg = self.messages[0].clone();

        let mut slice_start = self.messages.len().saturating_sub(2).max(1);
        while slice_start > 1 && self.messages[slice_start].role == "tool" {
            slice_start -= 1;
        }

        let recent_slice = self.messages[slice_start..].to_vec();
        self.messages = vec![system_msg, compaction_msg];
        self.messages.extend(recent_slice);

        if self.estimate_tokens() > target_limit {
            return Err(format!(
                "Context window hard bound exceeded ({}/{} tokens); refusing oversized dispatch without dropping required constraints/approvals",
                self.estimate_tokens(),
                target_limit
            ));
        }

        Ok(())
    }
}
