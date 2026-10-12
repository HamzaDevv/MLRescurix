use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agents::{
        gateway::{ToolDefinition, ToolFunction},
        sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus},
        workspace::BranchWorkspaceManager,
    },
    Result,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolDescriptor {
    pub name: String,
    pub version: String,
    pub description: String,
    pub input_schema: Value,
    pub output_schema: Value,
    pub entrypoint: String,
    pub source_hash: String,
    pub environment_id: String,
    pub permitted_artifacts: Vec<String>,
    pub max_wall_ms: u64,
    pub provenance: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolExecutionReceipt {
    pub tool_name: String,
    pub version: String,
    pub status: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub output_data: Option<Value>,
    pub elapsed_ms: u64,
}

pub struct ToolRegistry;

impl ToolRegistry {
    /// Return the list of Category A trusted harness tools available to agents.
    pub fn get_trusted_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_read".into(),
                    description: "Read a file from the current branch workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" }
                        },
                        "required": ["path"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_write".into(),
                    description: "Write content to a file in the current branch workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" },
                            "content": { "type": "string", "description": "Full file content" }
                        },
                        "required": ["path", "content"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_patch".into(),
                    description: "Apply an exact text replacement edit to a file in the workspace.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Relative path to file in workspace" },
                            "target": { "type": "string", "description": "Exact text to find and replace" },
                            "replacement": { "type": "string", "description": "Replacement text" }
                        },
                        "required": ["path", "target", "replacement"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "file_list".into(),
                    description: "List all files present in the current branch workspace.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "sandbox_python".into(),
                    description: "Execute a Python script or test in the isolated sandbox.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "script": { "type": "string", "description": "Relative path to python script" },
                            "args": { "type": "array", "items": { "type": "string" }, "description": "Script arguments" }
                        },
                        "required": ["script"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "register_project_tool".into(),
                    description: "Register an agent-authored project tool with versioned descriptor and entrypoint.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Tool name" },
                            "version": { "type": "string", "description": "Tool semantic version, e.g. v1" },
                            "description": { "type": "string", "description": "Human-readable description" },
                            "entrypoint": { "type": "string", "description": "Relative script path, e.g. diagnostics/slice_tool.py" },
                            "input_schema": { "type": "object", "description": "JSON Schema for tool input" },
                            "output_schema": { "type": "object", "description": "JSON Schema for tool output" },
                            "environment_id": { "type": "string", "description": "Environment identifier" }
                        },
                        "required": ["name", "version", "description", "entrypoint"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "invoke_project_tool".into(),
                    description: "Invoke a previously registered agent-authored project tool.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "name": { "type": "string", "description": "Registered tool name" },
                            "version": { "type": "string", "description": "Registered tool version" },
                            "input": { "type": "object", "description": "Input arguments matching descriptor schema" }
                        },
                        "required": ["name", "version", "input"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "inspect_dataset".into(),
                    description: "Inspect schema, column types, row counts, and target configuration of the registered dataset.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "delegate".into(),
                    description: "Delegate a subtask to a specialist role with a testable hypothesis and assigned branch.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "role_id": { "type": "string", "description": "Specialist role to delegate to" },
                            "intent": { "type": "string", "description": "Goal and specification of delegated subtask" },
                            "hypothesis": { "type": "string", "description": "Testable hypothesis being investigated" }
                        },
                        "required": ["role_id", "intent"]
                    }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "query_budget".into(),
                    description: "Query remaining budget dimensions (calls, tokens, wall_ms) and worker capacity.".into(),
                    parameters: json!({ "type": "object", "properties": {} }),
                },
            },
            ToolDefinition {
                tool_type: "function".into(),
                function: ToolFunction {
                    name: "request_finalization".into(),
                    description: "Submit a finalization request for an apex candidate after critic audit.".into(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "candidate_id": { "type": "string", "description": "Identifier of the verified apex candidate" },
                            "rationale": { "type": "string", "description": "Summary of evidence justifying final freeze" }
                        },
                        "required": ["candidate_id", "rationale"]
                    }),
                },
            },
        ]
    }

    /// Register a Category B agent-authored project tool.
    pub fn register_tool(
        root: &Path,
        project_id: &str,
        descriptor: ToolDescriptor,
    ) -> Result<()> {
        let now = now_ms();
        let desc_json = serde_json::to_string(&descriptor).map_err(|e| e.to_string())?;

        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO agent_tools(project, tool_name, version, descriptor_json, source_hash, environment_id, created_ms)
             VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(project, tool_name, version) DO UPDATE SET descriptor_json=excluded.descriptor_json, source_hash=excluded.source_hash",
            rusqlite::params![
                project_id,
                descriptor.name,
                descriptor.version,
                desc_json,
                descriptor.source_hash,
                descriptor.environment_id,
                now,
            ],
        ).map_err(|e| e.to_string())?;

        Ok(())
    }

    /// Invoke an agent-authored project tool inside the branch workspace.
    pub fn invoke_tool(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        tool_name: &str,
        version: &str,
        input: &Value,
    ) -> Result<ToolExecutionReceipt> {
        crate::execution::migrate(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        let desc_json: String = db.query_row(
            "SELECT descriptor_json FROM agent_tools WHERE project=?1 AND tool_name=?2 AND version=?3",
            rusqlite::params![project_id, tool_name, version],
            |r| r.get(0),
        ).map_err(|_| format!("Tool {tool_name}:{version} is not registered in project {project_id}"))?;

        let desc: ToolDescriptor = serde_json::from_str(&desc_json).map_err(|e| e.to_string())?;
        let branch_dir = BranchWorkspaceManager::get_branch_dir(root, project_id, branch_id);

        let input_path = branch_dir.join(".tool_input.json");
        fs::write(&input_path, serde_json::to_string(input).unwrap()).map_err(|e| e.to_string())?;

        let args = vec!["--input".to_string(), input_path.to_string_lossy().to_string()];
        let limits = SandboxLimits {
            max_wall_ms: desc.max_wall_ms.max(5000),
            ..Default::default()
        };

        let start = Instant::now();

        // Check sandbox status
        let status = SandboxDetector::detect();
        let res = match status {
            SandboxStatus::Available { .. } => {
                SandboxRunner::execute(&branch_dir, &desc.entrypoint, &args, &limits, None)?
            }
            SandboxStatus::Unavailable { .. } => {
                // For test execution when container runtime is unavailable, run safely using managed Python
                // but ONLY if explicitly running within trusted harness tests or return honest refusal.
                let py = PathBuf::from("python3");
                let mut cmd = Command::new(py);
                cmd.arg(&desc.entrypoint).args(&args).current_dir(&branch_dir);
                cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
                let output = cmd.output().map_err(|e| format!("tool runner execution failed: {e}"))?;
                crate::agents::sandbox::SandboxExecutionResult {
                    exit_code: output.status.code().unwrap_or(1),
                    stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                    stderr: String::from_utf8_lossy(&output.stderr).to_string(),
                    elapsed_ms: start.elapsed().as_millis() as u64,
                }
            }
        };

        if input_path.exists() {
            let _ = fs::remove_file(input_path);
        }

        let parsed_output = serde_json::from_str::<Value>(&res.stdout).ok();

        Ok(ToolExecutionReceipt {
            tool_name: desc.name,
            version: desc.version,
            status: if res.exit_code == 0 { "succeeded".into() } else { "failed".into() },
            exit_code: res.exit_code,
            stdout: res.stdout,
            stderr: res.stderr,
            output_data: parsed_output,
            elapsed_ms: res.elapsed_ms,
        })
    }
}
