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
        let chk_id = format!("chk-{}", &hash_str(&format!("{}:{}:{now}", self.task_id, self.handoff_version))[..12]);

        let state_snapshot = json!({
            "hypothesis": self.current_hypothesis,
            "achievements_count": self.achievements.len(),
            "findings_count": self.findings.len(),
            "messages_count": self.messages.len(),
        });

        let active_jobs_json = serde_json::to_string(&self.active_jobs).unwrap_or_default();

        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
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
                serde_json::to_string(&state_snapshot).unwrap_or_default(),
                active_jobs_json,
                now,
            ],
        ).map_err(|e| e.to_string())?;

        Ok(chk_id)
    }

    /// Structured compaction: preserves system prompt, active hypothesis, all achievements,
    /// scientific findings, and recent conversation turns while collapsing older tool interactions.
    pub fn maybe_compact(&mut self) -> bool {
        let current_tokens = self.estimate_tokens();
        let target_limit = self.max_context_tokens.saturating_sub(self.reserve_tokens);

        if current_tokens <= target_limit || self.messages.len() <= 6 {
            return false;
        }

        let system_msg = self.messages[0].clone();
        let recent_slice = self.messages[self.messages.len().saturating_sub(4)..].to_vec();

        // Build structured compaction context
        let mut summary_lines = Vec::new();
        summary_lines.push("### Context Compaction & Authoritative State Summary".to_string());
        if let Some(hyp) = &self.current_hypothesis {
            summary_lines.push(format!("Active Hypothesis: {hyp}"));
        }
        if !self.achievements.is_empty() {
            summary_lines.push("Committed Achievements:".to_string());
            for ach in &self.achievements {
                summary_lines.push(format!("- [{}] {}: {}", ach.kind, ach.achievement_id, ach.summary));
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

        let compaction_msg = ChatMessage::user(summary_lines.join("\n"));

        let mut compacted = vec![system_msg, compaction_msg];
        compacted.extend(recent_slice);
        self.messages = compacted;
        true
    }

    /// Invalidate active handoffs, inject steering directive, and update handoff version.
    pub fn apply_steering(&mut self, root: &Path, directive: &str) -> Result<()> {
        self.handoff_version += 1;

        // Invalidate prior pending handoffs in SQLite
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "UPDATE handoffs SET status='superseded' WHERE project=?1 AND task_id=?2 AND status='pending'",
            rusqlite::params![self.project_id, self.task_id],
        ).map_err(|e| e.to_string())?;

        // Inject steering guidance at turn boundary
        self.messages.push(ChatMessage::user(format!(
            "[STEERING DIRECTIVE - Turn Boundary (Epoch v{})]:\n{directive}\nPrevious uncommitted proposals superseded.",
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
}
