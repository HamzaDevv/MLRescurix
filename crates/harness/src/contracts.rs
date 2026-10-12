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
#[serde(tag="kind", content="payload", rename_all="snake_case", deny_unknown_fields)]
pub enum Message {
    Hello { protocol: String, worker_id: String },
    Observation { project_id: String, task_id: String, action_id: String, generation: u64, phase: String, detail: String },
    Checkpoint(Checkpoint),
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
    } Ok(Frame{version:1,message:m})
}
