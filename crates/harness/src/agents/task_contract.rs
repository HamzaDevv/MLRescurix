use std::path::Path;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use crate::Result;

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskLimits {
    pub max_turns: u32,
    pub max_tokens: u32,
    pub wall_ms: u64,
}

impl Default for TaskLimits {
    fn default() -> Self {
        Self {
            max_turns: 10,
            max_tokens: 16_384,
            wall_ms: 60_000,
        }
    }
}

/// Versioned task assignment contract (v2 schema)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskAssignment {
    pub schema_version: String,
    pub task_id: String,
    pub project_id: String,
    pub role_id: String,
    pub goal: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hypothesis: Option<String>,
    #[serde(default)]
    pub immutable_inputs: Vec<String>,
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    pub branch_ownership: String,
    pub limits: TaskLimits,
    #[serde(default)]
    pub expected_artifacts: Vec<String>,
    #[serde(default)]
    pub acceptance_criteria: Vec<String>,
    #[serde(default)]
    pub dependency_ids: Vec<String>,
}

impl TaskAssignment {
    pub fn new(
        project_id: impl Into<String>,
        task_id: impl Into<String>,
        role_id: impl Into<String>,
        goal: impl Into<String>,
        branch_ownership: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: "v2".into(),
            task_id: task_id.into(),
            project_id: project_id.into(),
            role_id: role_id.into(),
            goal: goal.into(),
            hypothesis: None,
            immutable_inputs: Vec::new(),
            allowed_tools: Vec::new(),
            branch_ownership: branch_ownership.into(),
            limits: TaskLimits::default(),
            expected_artifacts: Vec::new(),
            acceptance_criteria: Vec::new(),
            dependency_ids: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "v2" {
            return Err("TaskAssignment requires schema_version = 'v2'".into());
        }
        crate::contracts::id(&self.task_id)?;
        crate::contracts::id(&self.project_id)?;
        crate::contracts::id(&self.role_id)?;
        crate::contracts::id(&self.branch_ownership)?;
        if self.goal.trim().is_empty() {
            return Err("TaskAssignment goal cannot be empty".into());
        }
        if self.goal.len() > 4096 {
            return Err("TaskAssignment goal cannot exceed 4096 characters".into());
        }
        if self.limits.max_turns == 0 || self.limits.max_tokens == 0 || self.limits.wall_ms == 0 {
            return Err("TaskAssignment limits must be strictly positive".into());
        }
        Ok(())
    }
}

/// Versioned task completion receipt (v2 schema)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TaskCompletionReceipt {
    pub schema_version: String,
    pub task_id: String,
    pub project_id: String,
    pub verdict: String, // "succeeded", "failed", "inconclusive"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub experiment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub candidate_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_delta: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
    #[serde(default)]
    pub failures: Vec<String>,
    pub bounded_learnings: String,
    #[serde(default)]
    pub next_actions: Vec<String>,
    #[serde(default)]
    pub artifacts_produced: Vec<String>,
}

impl TaskCompletionReceipt {
    pub fn new(
        project_id: impl Into<String>,
        task_id: impl Into<String>,
        verdict: impl Into<String>,
        bounded_learnings: impl Into<String>,
    ) -> Self {
        Self {
            schema_version: "v2".into(),
            task_id: task_id.into(),
            project_id: project_id.into(),
            verdict: verdict.into(),
            experiment_id: None,
            candidate_id: None,
            source_hash: None,
            metric_delta: None,
            protocol: None,
            failures: Vec::new(),
            bounded_learnings: bounded_learnings.into(),
            next_actions: Vec::new(),
            artifacts_produced: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "v2" {
            return Err("TaskCompletionReceipt requires schema_version = 'v2'".into());
        }
        crate::contracts::id(&self.task_id)?;
        crate::contracts::id(&self.project_id)?;
        if !matches!(self.verdict.as_str(), "succeeded" | "failed" | "inconclusive") {
            return Err(format!("Invalid task verdict: '{}'; must be 'succeeded', 'failed', or 'inconclusive'", self.verdict));
        }
        if self.bounded_learnings.trim().is_empty() {
            return Err("bounded_learnings cannot be empty".into());
        }
        if self.bounded_learnings.len() > 4096 {
            return Err("bounded_learnings exceeds maximum length 4096 bytes".into());
        }
        if let Some(ref sh) = self.source_hash {
            if sh.len() != 64 || !sh.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("source_hash must be a valid 64-character SHA-256 hex string".into());
            }
        }
        Ok(())
    }
}

pub fn record_task_assignment(root: &Path, assignment: &TaskAssignment) -> Result<()> {
    assignment.validate()?;
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.execute(
        "INSERT INTO task_assignments(project, task_id, role_id, goal, hypothesis, immutable_inputs, allowed_tools, branch_ownership, limits, expected_artifacts, acceptance_criteria, dependency_ids, created_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
         ON CONFLICT(project, task_id) DO UPDATE SET
            role_id=excluded.role_id,
            goal=excluded.goal,
            hypothesis=excluded.hypothesis,
            immutable_inputs=excluded.immutable_inputs,
            allowed_tools=excluded.allowed_tools,
            branch_ownership=excluded.branch_ownership,
            limits=excluded.limits,
            expected_artifacts=excluded.expected_artifacts,
            acceptance_criteria=excluded.acceptance_criteria,
            dependency_ids=excluded.dependency_ids",
        params![
            assignment.project_id,
            assignment.task_id,
            assignment.role_id,
            assignment.goal,
            assignment.hypothesis,
            serde_json::to_string(&assignment.immutable_inputs).map_err(|e| e.to_string())?,
            serde_json::to_string(&assignment.allowed_tools).map_err(|e| e.to_string())?,
            assignment.branch_ownership,
            serde_json::to_string(&assignment.limits).map_err(|e| e.to_string())?,
            serde_json::to_string(&assignment.expected_artifacts).map_err(|e| e.to_string())?,
            serde_json::to_string(&assignment.acceptance_criteria).map_err(|e| e.to_string())?,
            serde_json::to_string(&assignment.dependency_ids).map_err(|e| e.to_string())?,
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn get_task_assignment(root: &Path, project_id: &str, task_id: &str) -> Result<Option<TaskAssignment>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let row = db.query_row(
        "SELECT role_id, goal, hypothesis, immutable_inputs, allowed_tools, branch_ownership, limits, expected_artifacts, acceptance_criteria, dependency_ids
         FROM task_assignments WHERE project=?1 AND task_id=?2",
        params![project_id, task_id],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
                r.get::<_, String>(9)?,
            ))
        },
    ).optional().map_err(|e| e.to_string())?;

    if let Some((role_id, goal, hypothesis, inputs_s, tools_s, branch, limits_s, arts_s, crit_s, deps_s)) = row {
        let immutable_inputs: Vec<String> = serde_json::from_str(&inputs_s).unwrap_or_default();
        let allowed_tools: Vec<String> = serde_json::from_str(&tools_s).unwrap_or_default();
        let limits: TaskLimits = serde_json::from_str(&limits_s).unwrap_or_default();
        let expected_artifacts: Vec<String> = serde_json::from_str(&arts_s).unwrap_or_default();
        let acceptance_criteria: Vec<String> = serde_json::from_str(&crit_s).unwrap_or_default();
        let dependency_ids: Vec<String> = serde_json::from_str(&deps_s).unwrap_or_default();
        Ok(Some(TaskAssignment {
            schema_version: "v2".into(),
            task_id: task_id.to_string(),
            project_id: project_id.to_string(),
            role_id,
            goal,
            hypothesis,
            immutable_inputs,
            allowed_tools,
            branch_ownership: branch,
            limits,
            expected_artifacts,
            acceptance_criteria,
            dependency_ids,
        }))
    } else {
        Ok(None)
    }
}

pub fn record_task_completion(root: &Path, receipt: &TaskCompletionReceipt) -> Result<()> {
    receipt.validate()?;
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.execute(
        "INSERT INTO task_completions(project, task_id, verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced, completed_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
         ON CONFLICT(project, task_id) DO UPDATE SET
            verdict=excluded.verdict,
            experiment_id=excluded.experiment_id,
            candidate_id=excluded.candidate_id,
            source_hash=excluded.source_hash,
            metric_delta=excluded.metric_delta,
            protocol=excluded.protocol,
            failures=excluded.failures,
            bounded_learnings=excluded.bounded_learnings,
            next_actions=excluded.next_actions,
            artifacts_produced=excluded.artifacts_produced,
            completed_ms=excluded.completed_ms",
        params![
            receipt.project_id,
            receipt.task_id,
            receipt.verdict,
            receipt.experiment_id,
            receipt.candidate_id,
            receipt.source_hash,
            receipt.metric_delta.as_ref().map(|v| serde_json::to_string(v).unwrap_or_default()),
            receipt.protocol,
            serde_json::to_string(&receipt.failures).map_err(|e| e.to_string())?,
            receipt.bounded_learnings,
            serde_json::to_string(&receipt.next_actions).map_err(|e| e.to_string())?,
            serde_json::to_string(&receipt.artifacts_produced).map_err(|e| e.to_string())?,
            now_ms(),
        ],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn get_task_completion(root: &Path, project_id: &str, task_id: &str) -> Result<Option<TaskCompletionReceipt>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let row = db.query_row(
        "SELECT verdict, experiment_id, candidate_id, source_hash, metric_delta, protocol, failures, bounded_learnings, next_actions, artifacts_produced
         FROM task_completions WHERE project=?1 AND task_id=?2",
        params![project_id, task_id],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, Option<String>>(5)?,
                r.get::<_, String>(6)?,
                r.get::<_, String>(7)?,
                r.get::<_, String>(8)?,
                r.get::<_, String>(9)?,
            ))
        },
    ).optional().map_err(|e| e.to_string())?;

    if let Some((verdict, exp_id, cand_id, src_h, delta_s, proto, fails_s, learns, next_s, arts_s)) = row {
        let metric_delta: Option<Value> = delta_s.and_then(|s| serde_json::from_str(&s).ok());
        let failures: Vec<String> = serde_json::from_str(&fails_s).unwrap_or_default();
        let next_actions: Vec<String> = serde_json::from_str(&next_s).unwrap_or_default();
        let artifacts_produced: Vec<String> = serde_json::from_str(&arts_s).unwrap_or_default();
        Ok(Some(TaskCompletionReceipt {
            schema_version: "v2".into(),
            task_id: task_id.to_string(),
            project_id: project_id.to_string(),
            verdict,
            experiment_id: exp_id,
            candidate_id: cand_id,
            source_hash: src_h,
            metric_delta,
            protocol: proto,
            failures,
            bounded_learnings: learns,
            next_actions,
            artifacts_produced,
        }))
    } else {
        Ok(None)
    }
}

pub fn record_steering_directive(
    root: &Path,
    project_id: &str,
    directive_id: &str,
    task_id: &str,
    actor: &str,
    epoch: u64,
    directive: &str,
) -> Result<()> {
    crate::contracts::id(project_id)?;
    crate::contracts::id(directive_id)?;
    crate::contracts::id(task_id)?;
    if actor.trim().is_empty() || directive.trim().is_empty() {
        return Err("actor and directive cannot be empty".into());
    }
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.execute(
        "INSERT INTO steering_directives(project, directive_id, task_id, actor, epoch, directive, created_ms, applied)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 0)
         ON CONFLICT(project, directive_id) DO UPDATE SET directive=excluded.directive",
        params![project_id, directive_id, task_id, actor, epoch as i64, directive, now_ms()],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Versioned Guidance Contract for dynamic turn-boundary steering
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct GuidanceContract {
    pub contract_id: String,
    pub session_id: String,
    pub target_task_id: String,
    pub epoch: u64,
    pub actor: String,
    pub directive: String,
    pub veto_reasons: Vec<String>,
    pub recommended_actions: Vec<String>,
    pub created_ms: i64,
}

impl GuidanceContract {
    pub fn new(
        contract_id: impl Into<String>,
        session_id: impl Into<String>,
        target_task_id: impl Into<String>,
        epoch: u64,
        actor: impl Into<String>,
        directive: impl Into<String>,
        veto_reasons: Vec<String>,
        recommended_actions: Vec<String>,
    ) -> Self {
        Self {
            contract_id: contract_id.into(),
            session_id: session_id.into(),
            target_task_id: target_task_id.into(),
            epoch,
            actor: actor.into(),
            directive: directive.into(),
            veto_reasons,
            recommended_actions,
            created_ms: now_ms(),
        }
    }
}

pub fn record_guidance_contract(root: &Path, guidance: &GuidanceContract) -> Result<()> {
    record_steering_directive(
        root,
        &guidance.session_id,
        &guidance.contract_id,
        &guidance.target_task_id,
        &guidance.actor,
        guidance.epoch,
        &guidance.directive,
    )
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SteeringDirectiveRecord {
    pub directive_id: String,
    pub task_id: String,
    pub actor: String,
    pub epoch: u64,
    pub directive: String,
}

pub fn get_pending_steering_directives(
    root: &Path,
    project_id: &str,
    task_id: &str,
) -> Result<Vec<SteeringDirectiveRecord>> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let mut stmt = db.prepare(
        "SELECT directive_id, task_id, actor, epoch, directive FROM steering_directives
         WHERE project=?1 AND task_id=?2 AND applied=0
         ORDER BY epoch ASC, created_ms ASC"
    ).map_err(|e| e.to_string())?;
    let rows = stmt.query_map(params![project_id, task_id], |r| {
        Ok(SteeringDirectiveRecord {
            directive_id: r.get::<_, String>(0)?,
            task_id: r.get::<_, String>(1)?,
            actor: r.get::<_, String>(2)?,
            epoch: r.get::<_, i64>(3)? as u64,
            directive: r.get::<_, String>(4)?,
        })
    }).map_err(|e| e.to_string())?.collect::<std::result::Result<Vec<_>, _>>().map_err(|e| e.to_string())?;
    Ok(rows)
}

pub fn mark_steering_directive_applied(
    root: &Path,
    project_id: &str,
    directive_id: &str,
) -> Result<()> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    db.execute(
        "UPDATE steering_directives SET applied=1 WHERE project=?1 AND directive_id=?2",
        params![project_id, directive_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

pub fn register_user_approval(
    root: &Path,
    project_id: &str,
    approval_id: &str,
    action_type: &str,
    target_id: &str,
    actor: &str,
    approved: bool,
    ttl_ms: i64,
) -> Result<Value> {
    crate::contracts::id(project_id)?;
    crate::contracts::id(approval_id)?;
    if !matches!(action_type, "holdout_access" | "deployment_handoff" | "final_test") {
        return Err(format!("Invalid approval action_type: '{action_type}'; must be 'holdout_access', 'deployment_handoff', or 'final_test'"));
    }
    if actor.trim().is_empty() || target_id.trim().is_empty() {
        return Err("actor and target_id cannot be empty".into());
    }
    if !(1..=86_400_000).contains(&ttl_ms) {
        return Err("approval ttl_ms must be 1..=86_400_000".into());
    }
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let now = now_ms();
    let expires_ms = now.saturating_add(ttl_ms);
    db.execute(
        "INSERT INTO user_approvals(project, approval_id, action_type, target_id, actor, approved, created_ms, expires_ms)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(project, approval_id) DO UPDATE SET approved=excluded.approved, expires_ms=excluded.expires_ms, actor=excluded.actor",
        params![project_id, approval_id, action_type, target_id, actor, approved as i64, now, expires_ms],
    ).map_err(|e| e.to_string())?;
    Ok(json!({
        "project_id": project_id,
        "approval_id": approval_id,
        "action_type": action_type,
        "target_id": target_id,
        "actor": actor,
        "approved": approved,
        "expires_ms": expires_ms,
    }))
}

pub fn check_user_approval(
    root: &Path,
    project_id: &str,
    action_type: &str,
    target_id: &str,
) -> Result<bool> {
    crate::execution::migrate(root)?;
    let db = Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
    let now = now_ms();
    let is_approved: bool = db.query_row(
        "SELECT EXISTS(
            SELECT 1 FROM user_approvals
            WHERE project=?1 AND action_type=?2 AND target_id=?3 AND approved=1 AND expires_ms > ?4
        )",
        params![project_id, action_type, target_id, now],
        |r| r.get(0),
    ).unwrap_or(false);
    Ok(is_approved)
}
