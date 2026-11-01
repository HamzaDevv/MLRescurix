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
    pub candidate_id: Option<String>,
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
        if let Some(cand) = &self.candidate_id { id(cand)?; }
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
        if let Some(ckpt) = &self.resume_checkpoint {
            if ckpt.is_empty()
                || ckpt.len() > 256
                || ckpt.contains('\0')
                || ckpt.contains("..")
                || ckpt.starts_with('/')
                || ckpt.starts_with('\\')
                || ckpt.contains(':')
                || std::path::Path::new(ckpt).is_absolute()
                || std::path::Path::new(ckpt).components().any(|c| matches!(c, std::path::Component::ParentDir | std::path::Component::RootDir | std::path::Component::Prefix(_)))
            {
                return Err("invalid resume_checkpoint reference; must be a valid relative checkpoint identity without directory traversal".into());
            }
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
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub weights_sha256: Option<String>,
    pub step: u64,
    pub epoch: u64,
    pub batch_offset: u64,
    pub sampler_position: u64,
    #[serde(default)]
    pub loss: Option<f64>,
    pub files: BTreeMap<String, String>,
    #[serde(default)]
    pub dataset_sha256: Option<String>,
    #[serde(default)]
    pub split_sha256: Option<String>,
    #[serde(default)]
    pub source_sha256: Option<String>,
    #[serde(default)]
    pub environment_id: Option<String>,
    #[serde(default)]
    pub training_config: Option<serde_json::Value>,
    #[serde(default)]
    pub replay_guarantee: Option<String>,
    #[serde(default)]
    pub training_state: Option<serde_json::Value>,
    pub timestamp_unix: f64,
}
impl TrainingCheckpoint {
    pub fn validate(&self) -> Result<()> {
        for s in [&self.checkpoint_id, &self.job_id] { id(s)?; }
        if let Some(p) = &self.project_id { id(p)?; }
        if let Some(cand) = &self.candidate_id { id(cand)?; }
        if let Some(env) = &self.environment_id { id(env)?; }
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        if let Some(w_sha) = &self.weights_sha256 {
            if w_sha.len() != 64 || !w_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid weights_sha256 in checkpoint".into());
            }
        }
        if let Some(d_sha) = &self.dataset_sha256 {
            if d_sha.len() != 64 || !d_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid dataset_sha256 in checkpoint".into());
            }
        }
        if let Some(s_sha) = &self.split_sha256 {
            if s_sha.len() != 64 || !s_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid split_sha256 in checkpoint".into());
            }
        }
        if let Some(src_sha) = &self.source_sha256 {
            if src_sha.len() != 64 || !src_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid source_sha256 in checkpoint".into());
            }
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
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub candidate_id: Option<String>,
    #[serde(default)]
    pub weights_sha256: Option<String>,
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
    pub dataset_sha256: Option<String>,
    #[serde(default)]
    pub split_sha256: Option<String>,
    #[serde(default)]
    pub source_sha256: Option<String>,
    #[serde(default)]
    pub environment_id: Option<String>,
    #[serde(default)]
    pub training_config: Option<serde_json::Value>,
    #[serde(default)]
    pub replay_guarantee: Option<String>,
    #[serde(default)]
    pub limitations: Vec<String>,
    #[serde(default)]
    pub reason: Option<String>,
}
impl TrainingResult {
    pub fn validate(&self) -> Result<()> {
        id(&self.job_id)?;
        if let Some(p) = &self.project_id { id(p)?; }
        if let Some(cand) = &self.candidate_id { id(cand)?; }
        if let Some(env) = &self.environment_id { id(env)?; }
        if self.schema_version.as_deref() != Some("v2") {
            return Err("schema_version must be 'v2'".into());
        }
        if let Some(w_sha) = &self.weights_sha256 {
            if w_sha.len() != 64 || !w_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid weights_sha256 in result".into());
            }
        }
        if let Some(d_sha) = &self.dataset_sha256 {
            if d_sha.len() != 64 || !d_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid dataset_sha256 in result".into());
            }
        }
        if let Some(s_sha) = &self.split_sha256 {
            if s_sha.len() != 64 || !s_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid split_sha256 in result".into());
            }
        }
        if let Some(src_sha) = &self.source_sha256 {
            if src_sha.len() != 64 || !src_sha.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err("invalid source_sha256 in result".into());
            }
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

// ==================== v3 Contracts (Master Plan Rev 2.3) ====================

fn hex64(s: &str) -> Result<()> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("expected 64-character lowercase hex digest".into());
    }
    Ok(())
}

fn default_five() -> u32 { 5 }
fn default_seed() -> u64 { 42 }
fn default_id_col() -> String { "row_id".into() }
fn default_max_worker_turns() -> u32 { 50 }
fn default_lockbox_fraction() -> f64 { 0.10 }
fn default_disclosure() -> String { "redacted_only".into() }
fn default_timeout() -> u32 { 60 }
fn default_retries() -> u32 { 2 }
fn default_true() -> bool { true }
fn default_assurance() -> String { "provisional_spec_checked".into() }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricSpec {
    pub name: String,
    pub direction: String,
    #[serde(default)]
    pub class_order: Option<Vec<String>>,
    #[serde(default)]
    pub epsilon: Option<f64>,
    #[serde(default)]
    pub decision_threshold: Option<f64>,
    #[serde(default)]
    pub variant: Option<String>,
}

impl MetricSpec {
    pub fn validate(&self) -> Result<()> {
        id(&self.name)?;
        if !["minimize", "maximize"].contains(&self.direction.as_str()) {
            return Err("metric direction must be 'minimize' or 'maximize'".into());
        }
        if let Some(eps) = self.epsilon {
            if !eps.is_finite() || eps < 0.0 {
                return Err("metric epsilon must be finite and non-negative".into());
            }
        }
        if let Some(t) = self.decision_threshold {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err("metric decision_threshold must be between 0.0 and 1.0".into());
            }
        }
        if let Some(v) = &self.variant {
            id(v)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValidationSpec {
    pub strategy: String,
    #[serde(default = "default_five")]
    pub folds: u32,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub embargo_horizon: Option<u64>,
}

impl ValidationSpec {
    pub fn validate(&self) -> Result<()> {
        text(&self.strategy)?;
        if self.folds < 2 || self.folds > 50 {
            return Err("validation folds must be 2..=50".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CampaignBudgetSpec {
    pub max_wall_clock_seconds: u64,
    pub max_cost_millicents: i64,
    pub max_total_tokens: u64,
    #[serde(default)]
    pub max_gpu_quota_seconds: u64,
    #[serde(default = "default_max_worker_turns")]
    pub max_worker_turns: u32,
}

impl CampaignBudgetSpec {
    pub fn validate(&self) -> Result<()> {
        if self.max_wall_clock_seconds == 0 {
            return Err("max_wall_clock_seconds must be positive".into());
        }
        if self.max_cost_millicents < 0 {
            return Err("max_cost_millicents must be non-negative".into());
        }
        if self.max_worker_turns == 0 {
            return Err("max_worker_turns must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssessmentPolicySpec {
    pub mode: String,
    #[serde(default = "default_lockbox_fraction")]
    pub lockbox_fraction: f64,
    #[serde(default)]
    pub post_assessment_refit_allowed: bool,
}

impl AssessmentPolicySpec {
    pub fn validate(&self) -> Result<()> {
        if !["auto_if_feasible", "external", "disabled"].contains(&self.mode.as_str()) {
            return Err("assessment mode must be 'auto_if_feasible', 'external', or 'disabled'".into());
        }
        if !self.lockbox_fraction.is_finite() || self.lockbox_fraction < 0.0 || self.lockbox_fraction > 0.5 {
            return Err("lockbox_fraction must be between 0.0 and 0.5".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProblemSpec {
    pub schema_version: String,
    pub campaign_id: String,
    pub task_type: String,
    #[serde(default)]
    pub modalities: Vec<String>,
    pub target_column: String,
    #[serde(default = "default_id_col")]
    pub id_column: String,
    #[serde(default)]
    pub group_column: Option<String>,
    #[serde(default)]
    pub time_column: Option<String>,
    pub metric_spec: MetricSpec,
    pub validation_spec: ValidationSpec,
    #[serde(default)]
    pub requested_artifacts: Vec<String>,
    pub budget: CampaignBudgetSpec,
    pub assessment_policy: AssessmentPolicySpec,
}

impl ProblemSpec {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.problem.v1" {
            return Err("schema_version must be 'mlrescurix.problem.v1'".into());
        }
        id(&self.campaign_id)?;
        id(&self.task_type)?;
        id(&self.target_column)?;
        id(&self.id_column)?;
        if let Some(grp) = &self.group_column { id(grp)?; }
        if let Some(tm) = &self.time_column { id(tm)?; }
        self.metric_spec.validate()?;
        self.validation_spec.validate()?;
        self.budget.validate()?;
        self.assessment_policy.validate()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StageInputContract {
    pub schema_version: String,
    pub campaign_id: String,
    pub task_id: String,
    pub role: String,
    pub protocol_sha256: String,
    pub campaign_epoch: u64,
    pub branch_id: String,
    pub attempt_generation: u64,
    pub problem_spec_ref: String,
    #[serde(default)]
    pub input_artifact_refs: Vec<String>,
    #[serde(default)]
    pub hypothesis_ref: Option<String>,
    #[serde(default)]
    pub prior_result_refs: Vec<String>,
    pub capability_profile: String,
    pub resource_profile_ref: String,
    pub reservation_id: String,
    #[serde(default)]
    pub expected_artifacts: Vec<String>,
    #[serde(default)]
    pub acceptance_policy_ref: Option<String>,
}

impl StageInputContract {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.stage.v1" {
            return Err("schema_version must be 'mlrescurix.stage.v1'".into());
        }
        id(&self.campaign_id)?;
        id(&self.task_id)?;
        id(&self.role)?;
        hex64(&self.protocol_sha256)?;
        if self.campaign_epoch == 0 || self.attempt_generation == 0 {
            return Err("campaign_epoch and attempt_generation must be positive".into());
        }
        id(&self.branch_id)?;
        text(&self.problem_spec_ref)?;
        id(&self.capability_profile)?;
        id(&self.resource_profile_ref)?;
        id(&self.reservation_id)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StageOutputArtifact {
    pub schema_version: String,
    pub campaign_id: String,
    pub task_id: String,
    pub role: String,
    pub attempt_generation: u64,
    pub status: String,
    pub produced_artifacts: BTreeMap<String, String>,
    #[serde(default)]
    pub diagnostic_codes: Vec<String>,
    #[serde(default)]
    pub fit_provenance_ref: Option<String>,
    #[serde(default)]
    pub advisory_score: Option<f64>,
}

impl StageOutputArtifact {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.stage.v1" {
            return Err("schema_version must be 'mlrescurix.stage.v1'".into());
        }
        id(&self.campaign_id)?;
        id(&self.task_id)?;
        id(&self.role)?;
        if self.attempt_generation == 0 {
            return Err("attempt_generation must be positive".into());
        }
        if !["succeeded", "failed", "inconclusive", "needs_extension"].contains(&self.status.as_str()) {
            return Err("invalid stage output status".into());
        }
        for (name, sha) in &self.produced_artifacts {
            text(name)?;
            hex64(sha)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ProviderProfile {
    pub adapter_protocol: String,
    pub endpoint: String,
    #[serde(default)]
    pub secret_ref: Option<String>,
    #[serde(default = "default_disclosure")]
    pub permitted_data_disclosure: String,
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u32,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
}

impl ProviderProfile {
    pub fn validate(&self) -> Result<()> {
        text(&self.adapter_protocol)?;
        text(&self.endpoint)?;
        if !["none", "redacted_only", "full_abstract_only", "isolated_local"].contains(&self.permitted_data_disclosure.as_str()) {
            return Err("permitted_data_disclosure must be 'none', 'redacted_only', 'full_abstract_only', or 'isolated_local'".into());
        }
        if self.timeout_seconds == 0 {
            return Err("timeout_seconds must be positive".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    pub provider_ref: String,
    pub model_id: String,
    pub context_window: u32,
    pub max_output_tokens: u32,
    pub prompt_millicents_per_1k: i64,
    pub completion_millicents_per_1k: i64,
    #[serde(default = "default_true")]
    pub supports_tools: bool,
}

impl ModelProfile {
    pub fn validate(&self) -> Result<()> {
        id(&self.provider_ref)?;
        text(&self.model_id)?;
        if self.context_window < 1024 {
            return Err("context_window must be at least 1024".into());
        }
        if self.max_output_tokens < 128 {
            return Err("max_output_tokens must be at least 128".into());
        }
        if self.prompt_millicents_per_1k < 0 || self.completion_millicents_per_1k < 0 {
            return Err("pricing rates must be non-negative".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RoleRouting {
    pub model_profile_ref: String,
    #[serde(default)]
    pub escalation_profile_ref: Option<String>,
    #[serde(default)]
    pub max_spend_millicents: Option<i64>,
    #[serde(default)]
    pub temperature: Option<f64>,
}

impl RoleRouting {
    pub fn validate(&self) -> Result<()> {
        id(&self.model_profile_ref)?;
        if let Some(esc) = &self.escalation_profile_ref { id(esc)?; }
        if let Some(spend) = self.max_spend_millicents {
            if spend < 0 { return Err("max_spend_millicents must be non-negative".into()); }
        }
        if let Some(t) = self.temperature {
            if !t.is_finite() || t < 0.0 || t > 2.0 {
                return Err("temperature must be in [0.0, 2.0]".into());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AgentRoutingPolicy {
    pub schema_version: String,
    pub default_worker_profile_ref: String,
    pub providers: BTreeMap<String, ProviderProfile>,
    pub models: BTreeMap<String, ModelProfile>,
    pub roles: BTreeMap<String, RoleRouting>,
}

impl AgentRoutingPolicy {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.agent_routing.v1" {
            return Err("schema_version must be 'mlrescurix.agent_routing.v1'".into());
        }
        id(&self.default_worker_profile_ref)?;
        if !self.models.contains_key(&self.default_worker_profile_ref) {
            return Err("default_worker_profile_ref not found in models".into());
        }
        for (pid, prov) in &self.providers {
            id(pid)?;
            prov.validate()?;
        }
        for (mid, model) in &self.models {
            id(mid)?;
            model.validate()?;
            if !self.providers.contains_key(&model.provider_ref) {
                return Err(format!("model '{mid}' references unknown provider '{}'", model.provider_ref));
            }
        }
        for (rid, role) in &self.roles {
            id(rid)?;
            role.validate()?;
            if !self.models.contains_key(&role.model_profile_ref) {
                return Err(format!("role '{rid}' references unknown model profile '{}'", role.model_profile_ref));
            }
            if let Some(esc) = &role.escalation_profile_ref {
                if !self.models.contains_key(esc) {
                    return Err(format!("role '{rid}' references unknown escalation profile '{esc}'"));
                }
            }
        }
        Ok(())
    }

    pub fn resolve_role(&self, role_id: &str) -> Result<(&ModelProfile, &ProviderProfile)> {
        let model_ref = if let Some(role) = self.roles.get(role_id) {
            &role.model_profile_ref
        } else {
            &self.default_worker_profile_ref
        };
        let model = self.models.get(model_ref)
            .ok_or_else(|| format!("resolved model profile '{model_ref}' not found"))?;
        let provider = self.providers.get(&model.provider_ref)
            .ok_or_else(|| format!("provider profile '{}' not found", model.provider_ref))?;
        Ok((model, provider))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityProposal {
    pub schema_version: String,
    pub capability_id: String,
    pub kind: String,
    pub purpose: String,
    pub entrypoint: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default)]
    pub fit_scope_kind: Option<String>,
    #[serde(default)]
    pub spec_ref: Option<String>,
    #[serde(default)]
    pub reference_implementation_ref: Option<String>,
    #[serde(default)]
    pub fixture_refs: Vec<String>,
    #[serde(default = "default_assurance")]
    pub assurance_requested: String,
}

impl CapabilityProposal {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.capability.v1" {
            return Err("schema_version must be 'mlrescurix.capability.v1'".into());
        }
        id(&self.capability_id)?;
        if !["splitter", "metric_evaluator", "tool", "method", "task_adapter"].contains(&self.kind.as_str()) {
            return Err("invalid capability kind".into());
        }
        text(&self.purpose)?;
        text(&self.entrypoint)?;
        if !["authoritative_reference", "provisional_spec_checked", "property_checked_only"].contains(&self.assurance_requested.as_str()) {
            return Err("invalid assurance_requested".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CapabilityReceipt {
    pub schema_version: String,
    pub capability_id: String,
    pub status: String,
    pub checked_properties: Vec<String>,
    pub skipped_properties: Vec<String>,
    pub admission_version: u32,
    pub package_sha256: String,
    #[serde(default)]
    pub rejection_reason: Option<String>,
}

impl CapabilityReceipt {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.capability.v1" {
            return Err("schema_version must be 'mlrescurix.capability.v1'".into());
        }
        id(&self.capability_id)?;
        if !["reference_conformant", "provisional_spec_checked", "rejected"].contains(&self.status.as_str()) {
            return Err("invalid capability receipt status".into());
        }
        hex64(&self.package_sha256)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OperationDeclaration {
    pub argv: Vec<String>,
    pub capability: String,
    pub fit_scope_kind: String,
}

impl OperationDeclaration {
    pub fn validate(&self) -> Result<()> {
        if self.argv.is_empty() {
            return Err("operation argv cannot be empty".into());
        }
        id(&self.capability)?;
        if !["stateless", "fixed_pretrained", "outer_fold_train", "inner_fold_train", "meta_fold_train", "full_development_train", "consumes_matching_scope"].contains(&self.fit_scope_kind.as_str()) {
            return Err("invalid fit_scope_kind".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AgentProgramManifest {
    pub schema_version: String,
    pub execution_kind: String,
    pub source_tree_ref: String,
    pub task_contract_ref: String,
    pub environment_ref: String,
    pub method_id: String,
    pub operations: BTreeMap<String, OperationDeclaration>,
}

impl AgentProgramManifest {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.program.v1" {
            return Err("schema_version must be 'mlrescurix.program.v1'".into());
        }
        if !["agent_authored", "recipe_scaffold"].contains(&self.execution_kind.as_str()) {
            return Err("execution_kind must be 'agent_authored' or 'recipe_scaffold'".into());
        }
        text(&self.source_tree_ref)?;
        text(&self.task_contract_ref)?;
        text(&self.environment_ref)?;
        text(&self.method_id)?;
        if self.operations.is_empty() {
            return Err("manifest must declare at least one operation".into());
        }
        for (name, op) in &self.operations {
            id(name)?;
            op.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FitScopeBinding {
    pub schema_version: String,
    pub binding_id: String,
    pub split_id: String,
    pub protocol_sha256: String,
    pub fit_scope_kind: String,
    pub permitted_feature_row_ids_hash: String,
    pub permitted_label_row_ids_hash: String,
    #[serde(default)]
    pub exclusion_row_ids_hash: Option<String>,
    #[serde(default)]
    pub transitive_fit_dependencies: Vec<String>,
}

impl FitScopeBinding {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.fit_scope.v1" {
            return Err("schema_version must be 'mlrescurix.fit_scope.v1'".into());
        }
        id(&self.binding_id)?;
        id(&self.split_id)?;
        hex64(&self.protocol_sha256)?;
        hex64(&self.permitted_feature_row_ids_hash)?;
        hex64(&self.permitted_label_row_ids_hash)?;
        if let Some(excl) = &self.exclusion_row_ids_hash {
            hex64(excl)?;
        }
        if !["stateless", "fixed_pretrained", "outer_fold_train", "inner_fold_train", "meta_fold_train", "full_development_train", "consumes_matching_scope"].contains(&self.fit_scope_kind.as_str()) {
            return Err("invalid fit_scope_kind".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AssessmentPolicyRecord {
    pub schema_version: String,
    pub policy_id: String,
    pub campaign_id: String,
    pub mode: String,
    pub lockbox_fraction: f64,
    #[serde(default)]
    pub lockbox_membership_sha256: Option<String>,
    #[serde(default)]
    pub total_development_rows: u64,
    #[serde(default)]
    pub lockbox_row_count: u64,
    pub consumed: bool,
    #[serde(default)]
    pub post_assessment_refit_allowed: bool,
    #[serde(default)]
    pub unavailability_reason: Option<String>,
}

impl AssessmentPolicyRecord {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "mlrescurix.assessment.v1" {
            return Err("schema_version must be 'mlrescurix.assessment.v1'".into());
        }
        id(&self.policy_id)?;
        id(&self.campaign_id)?;
        if !["auto_if_feasible", "external", "disabled"].contains(&self.mode.as_str()) {
            return Err("invalid assessment mode".into());
        }
        if !self.lockbox_fraction.is_finite() || self.lockbox_fraction < 0.0 || self.lockbox_fraction > 0.5 {
            return Err("lockbox_fraction must be between 0.0 and 0.5".into());
        }
        if let Some(sha) = &self.lockbox_membership_sha256 {
            hex64(sha)?;
        }
        Ok(())
    }
}
