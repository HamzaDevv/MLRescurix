use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use crate::Result;
pub const MAX_FRAME: usize = 65_536;
pub const MAX_INTEGER: u64 = 9_007_199_254_740_991;
pub type Amounts = BTreeMap<String, i64>;
pub fn id(s: &str) -> Result<()> {
    if s.is_empty() || s.len() > 128 || !s.bytes().all(|b| b.is_ascii_alphanumeric() || b"_.:-".contains(&b)) { return Err("invalid identity".into()); } Ok(())
}
pub fn text(s: &str) -> Result<()> { if s.is_empty() || s.len() > 4096 { Err("invalid bounded text".into()) } else { Ok(()) } }
pub fn amounts(a: &Amounts) -> Result<()> {
    if a.is_empty() || a.len() > 16 { return Err("one to sixteen budget dimensions required".into()); }
    for (k,v) in a { id(k)?; if *v < 0 || *v as u64 > MAX_INTEGER { return Err("invalid budget amount".into()); } } Ok(())
}
pub fn cursor(s: &str) -> Result<u64> {
    if s.is_empty() || s.len() > 19 || !s.bytes().all(|b| b.is_ascii_digit()) || (s.len()>1 && s.starts_with('0')) { return Err("invalid decimal cursor".into()); }
    s.parse().map_err(|_| "cursor overflow".into())
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Context {
    pub role_id: String,
    pub role_version: String,
    pub prompt_hash: String,
    pub model: Option<String>,
    pub phase: String,
    pub hypothesis: Option<String>,
    pub base_hash: String,
    pub code_hash: String,
    pub evidence: Vec<String>,
    pub artifacts: Vec<String>,
    pub pending_questions: Vec<String>,
    pub calls: Vec<String>,
    pub jobs: Vec<String>,
    pub next_action: String,
    pub stop_conditions: Vec<String>,
}
impl Context {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.role_id,&self.role_version,&self.prompt_hash,&self.base_hash,&self.code_hash] { id(s)?; }
        for s in [&self.phase,&self.next_action] { text(s)?; }
        if let Some(s)=&self.model { text(s)?; } if let Some(s)=&self.hypothesis { text(s)?; }
        for list in [&self.evidence,&self.artifacts,&self.pending_questions,&self.calls,&self.jobs,&self.stop_conditions] {
            if list.len()>64 {return Err("context list exceeds bound".into());} for s in list {id(s)?;}
        } Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Checkpoint {
    pub checkpoint_id: String,
    pub sequence: u64,
    pub project_id: String,
    pub branch_id: String,
    pub task_id: String,
    pub spec_revision: u64,
    pub intent_revision: u64,
    pub generation: u64,
    pub event_cursor: String,
    pub boundary: String,
    pub context: Context,
}
impl Checkpoint {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.checkpoint_id,&self.project_id,&self.branch_id,&self.task_id] {id(s)?;}
        for v in [self.sequence,self.spec_revision,self.intent_revision,self.generation] { if v==0 || v>MAX_INTEGER {return Err("invalid checkpoint revision".into());} }
        cursor(&self.event_cursor)?; text(&self.boundary)?; self.context.validate()
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CustomJobResources {
    pub memory_bytes: u64,
    pub cpu_threads: u32,
    pub timeout_ms: u64,
}
impl CustomJobResources {
    pub fn validate(&self) -> Result<()> {
        if self.memory_bytes < 64 * 1024 * 1024 { return Err("memory limit must be at least 64 MiB".into()); }
        if self.cpu_threads == 0 || self.cpu_threads > 64 { return Err("cpu threads must be 1..=64".into()); }
        if self.timeout_ms < 100 { return Err("timeout must be at least 100 ms".into()); }
        Ok(())
    }
}
fn default_v2_schema_version() -> Option<String> {
    Some("v2".into())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CustomJobSpec {
    pub job_id: String,
    pub project_id: String,
    pub task_id: String,
    pub generation: u64,
    pub intent_revision: u64,
    pub operation: String,
    pub model_plugin: String,
    pub dataset_id: String,
    #[serde(default = "default_v2_schema_version")]
    pub schema_version: Option<String>,
    #[serde(default)]
    pub target_column: Option<String>,
    pub metric_name: String,
    pub metric_direction: String,
    pub max_steps: u64,
    pub batch_size: u32,
    #[serde(default)]
    pub resume_checkpoint: Option<String>,
    #[serde(default)]
    pub learning_rate: Option<f64>,
    #[serde(default)]
    pub hidden_dim: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub resources: Option<CustomJobResources>,
}
impl CustomJobSpec {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.job_id, &self.project_id, &self.task_id, &self.dataset_id, &self.model_plugin] { id(s)?; }
        if self.generation == 0 || self.generation > MAX_INTEGER || self.intent_revision == 0 || self.intent_revision > MAX_INTEGER {
            return Err("invalid revision/generation".into());
        }
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        if !["custom_train", "custom_resume", "custom_evaluate", "custom_export"].contains(&self.operation.as_str()) {
            return Err("unsupported custom operation".into());
        }
        if !["maximize", "minimize"].contains(&self.metric_direction.as_str()) {
            return Err("invalid metric direction".into());
        }
        if self.max_steps == 0 || self.max_steps > 1_000_000 {
            return Err("invalid max_steps".into());
        }
        if self.batch_size == 0 || self.batch_size > 65536 {
            return Err("invalid batch_size".into());
        }
        if let Some(res) = &self.resources { res.validate()?; }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingCheckpoint {
    pub checkpoint_id: String,
    pub job_id: String,
    #[serde(default = "default_v2_schema_version")]
    pub schema_version: Option<String>,
    pub step: u64,
    pub epoch: u64,
    pub batch_offset: u64,
    pub sampler_position: u64,
    #[serde(default)]
    pub loss: Option<f64>,
    pub files: BTreeMap<String, String>,
    #[serde(default)]
    pub training_state: Option<serde_json::Value>,
    pub timestamp_unix: f64,
}
impl TrainingCheckpoint {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.checkpoint_id, &self.job_id] { id(s)?; }
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        if self.files.is_empty() { return Err("checkpoint files cannot be empty".into()); }
        for (f, sha) in &self.files {
            text(f)?;
            if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid file sha256 in checkpoint manifest".into());
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainingResult {
    pub job_id: String,
    pub status: String,
    #[serde(default = "default_v2_schema_version")]
    pub schema_version: Option<String>,
    pub step_reached: u64,
    pub epoch_reached: u64,
    #[serde(default)]
    pub primary_metric_name: Option<String>,
    #[serde(default)]
    pub primary_metric_score: Option<f64>,
    #[serde(default)]
    pub validation_loss: Option<f64>,
    pub peak_memory_bytes: u64,
    pub duration_seconds: f64,
    pub artifacts: Vec<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
}
impl TrainingResult {
    pub fn validate(&self) -> Result<()> {
        id(&self.job_id)?;
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        if !["succeeded", "failed", "timed_out", "needs_resources", "interrupted"].contains(&self.status.as_str()) {
            return Err("invalid training result status".into());
        }
        if self.artifacts.len() > 64 { return Err("too many artifacts in result".into()); }
        for a in &self.artifacts { text(a)?; }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct WorkerHandshake {
    pub protocol: String,
    pub worker_id: String,
    #[serde(default = "default_v2_schema_version")]
    pub schema_version: Option<String>,
    pub capabilities: Vec<String>,
    pub dependency_profile: String,
    pub device: String,
    pub platform: String,
}
impl WorkerHandshake {
    pub fn validate(&self) -> Result<()> {
        id(&self.worker_id)?;
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        text(&self.protocol)?;
        text(&self.dependency_profile)?;
        text(&self.device)?;
        text(&self.platform)?;
        if self.capabilities.is_empty() || self.capabilities.len() > 64 {
            return Err("worker capabilities must be 1..=64 items".into());
        }
        for c in &self.capabilities { text(c)?; }
        Ok(())
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag="kind", content="payload", rename_all="snake_case", deny_unknown_fields)]
pub enum Message {
    Hello { protocol: String, worker_id: String },
    Observation { project_id: String, task_id: String, action_id: String, generation: u64, phase: String, detail: String },
    Checkpoint(Checkpoint),
    CustomJob(CustomJobSpec),
    TrainingCheckpoint(TrainingCheckpoint),
    TrainingResult(TrainingResult),
    WorkerHandshake(WorkerHandshake),
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Frame { pub version: u32, #[serde(flatten)] pub message: Message }
// Avoid flatten + deny_unknown_fields ambiguity by parsing the outer envelope explicitly.
pub fn decode_frame(bytes: &[u8]) -> Result<Frame> {
    if bytes.len()>MAX_FRAME { return Err("frame too large".into()); }
    let mut v: serde_json::Value=serde_json::from_slice(bytes).map_err(|e|e.to_string())?;
    let obj=v.as_object_mut().ok_or("envelope must be object")?;
    if obj.len()!=3 || !obj.contains_key("version") || !obj.contains_key("kind") || !obj.contains_key("payload") {return Err("envelope fields".into());}
    if obj.remove("version")!=Some(serde_json::json!(1)) { return Err("unsupported protocol version".into()); }
    let m:Message=serde_json::from_value(v).map_err(|e|e.to_string())?;
    match &m {
        Message::Hello{protocol,worker_id} => { if protocol!="mlrescurix/1" {return Err("incompatible protocol".into());} id(worker_id)?; }
        Message::Observation{project_id,task_id,action_id,generation,phase,detail} => {
            for s in [project_id,task_id,action_id] {id(s)?;} if *generation==0 || *generation>MAX_INTEGER {return Err("invalid generation".into());} text(phase)?;text(detail)?;
        }
        Message::Checkpoint(c) => c.validate()?,
        Message::CustomJob(j) => j.validate()?,
        Message::TrainingCheckpoint(tc) => tc.validate()?,
        Message::TrainingResult(tr) => tr.validate()?,
        Message::WorkerHandshake(wh) => wh.validate()?,
    } Ok(Frame{version:1,message:m})
}
