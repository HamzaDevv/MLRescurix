use serde::{Deserialize,Serialize};
use crate::contracts::{Amounts,Context,Checkpoint};

#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Policy { pub budget: Amounts, pub capacity: Amounts, pub max_tasks: u32 }
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Request { pub command_id:String, pub project_id:String, pub expected_revision:Option<u64>, pub operation:Operation }
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
#[serde(tag="type",rename_all="snake_case",deny_unknown_fields)]
pub enum Operation {
    CreateProject { intent:String, finalization_id:String, policy:Policy },
    Lifecycle { state:String },
    Steer { intent:String },
    CreateBranch { branch_id:String, base_hash:String, code_hash:String },
    CreateTask { task_id:String, branch_id:String, context:Context },
    Claim { task_id:String, owner:String, lease_id:String, now_ms:i64, expires_ms:i64, expected_generation:u64 },
    Prepare { task_id:String, generation:u64, lease_id:String, action_id:String, input:String, budget:Amounts, capacity:Amounts, deadline_ms:i64, now_ms:i64 },
    Dispatch { task_id:String, generation:u64, lease_id:String, action_id:String, attempt_id:String, external_request_id:Option<String>, now_ms:i64 },
    Response { task_id:String, generation:u64, lease_id:String, action_id:String, frame:String, now_ms:i64 },
    Commit { task_id:String, generation:u64, lease_id:String, action_id:String, actual:Amounts, outcome:String, now_ms:i64 },
    MarkUnknown { action_id:String, reason:String },
    Resolve { action_id:String, resolution:String, actual:Amounts, proof:String },
    Checkpoint { task_id:String, generation:u64, lease_id:String, context:Context, boundary:String, now_ms:i64 },
    Ask { question_id:String, text:String },
    Answer { question_id:String, answer:String },
    RegisterReference { reference_id:String, category:String, digest:String },
    Timer { timer_id:String, task_id:String, generation:u64, due_ms:i64, expected_project_revision:u64, reason:String },
    DeliverTimers { now_ms:i64 },
    Recover { now_ms:i64 },
}
#[derive(Debug,Clone,Serialize,Deserialize,PartialEq)]
pub struct Receipt { pub command_id:String, pub project_id:String, pub status:String, pub revision:u64, pub result:serde_json::Value }
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Event { pub cursor:String, pub project_id:String, pub kind:String, pub payload:serde_json::Value }
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Task {
    pub task_id:String, pub branch_id:String, pub generation:u64, pub intent_revision:u64,
    pub owner:Option<String>, pub lease_id:Option<String>, pub expires_ms:Option<i64>, pub status:String, pub context:Context,
}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct Action {
    pub action_id:String, pub task_id:String, pub generation:u64, pub intent_revision:u64,
    pub lease_id:String, pub input:String, pub budget:Amounts, pub capacity:Amounts, pub deadline_ms:i64,
    pub status:String, pub attempt_id:Option<String>, pub external_request_id:Option<String>,
    pub response:Option<String>, pub receipt:Option<serde_json::Value>, pub uncertainty:Option<String>,
}
#[derive(Debug,Clone,Serialize,Deserialize)]
pub struct RecoveryTask { pub task:Task, pub checkpoint:Checkpoint, pub events:Vec<Event>, pub replay_next_cursor:String, pub replay_has_more:bool, pub actions:Vec<Action>, pub eligibility:String }
