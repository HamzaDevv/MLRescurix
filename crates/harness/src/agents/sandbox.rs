use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use serde::{Deserialize, Serialize};

use crate::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SandboxStatus {
    Available { engine: String, version: String },
    Unavailable { reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxLimits {
    pub max_wall_ms: u64,
    pub max_memory_mb: u64,
    pub max_cpus: f64,
    pub max_output_bytes: u64,
    pub max_scratch_mb: u64,
    pub pids_limit: u32,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            max_wall_ms: 30_000,
            max_memory_mb: 512,
            max_cpus: 1.0,
            max_output_bytes: 4 * 1024 * 1024,
            max_scratch_mb: 64,
            pids_limit: 64,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxExecutionResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub elapsed_ms: u64,
}

pub struct SandboxDetector;

impl SandboxDetector {
    /// Probe the container runtime. Refuses host Python substitution if Docker daemon is offline.
    pub fn detect() -> SandboxStatus {
        let output = Command::new("docker")
            .args(["version", "--format", "{{.Server.Version}}"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match output {
            Ok(out) if out.status.success() => {
                let ver = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if ver.is_empty() {
                    SandboxStatus::Unavailable {
                        reason: "Docker reported empty version string; daemon may be unresponsive".into(),
                    }
                } else {
                    SandboxStatus::Available {
                        engine: "docker".into(),
                        version: ver,
                    }
                }
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                SandboxStatus::Unavailable {
                    reason: format!("Docker daemon inactive or returned error: {err}"),
                }
            }
            Err(e) => {
                SandboxStatus::Unavailable {
                    reason: format!("docker executable unavailable or failed: {e}"),
                }
            }
        }
    }
}

pub struct SandboxRunner;

impl SandboxRunner {
    /// Execute arbitrary agent Python code inside a hardened, isolated OCI container.
    /// Strictly rejects host Python fallback if the container runtime is offline.
    pub fn execute(
        workspace_dir: &Path,
        script_relative_path: &str,
        args: &[String],
        limits: &SandboxLimits,
        sandbox_image: Option<&str>,
    ) -> Result<SandboxExecutionResult> {
        let status = SandboxDetector::detect();
        match status {
            SandboxStatus::Unavailable { reason } => {
                return Err(format!(
                    "Sandbox execution refused: container runtime is unavailable ({reason}). Host Python substitution for untrusted code is prohibited."
                ));
            }
            SandboxStatus::Available { .. } => {}
        }

        // Validate workspace path to avoid path traversal
        if fs::symlink_metadata(workspace_dir).map_err(|e| e.to_string())?.file_type().is_symlink() {
            return Err("Workspace cannot be a symlink".into());
        }

        let script_path = workspace_dir.join(script_relative_path);
        if !script_path.is_file() {
            return Err(format!("Script {script_relative_path} does not exist in workspace"));
        }

        let image = sandbox_image.unwrap_or("python:3.11-slim");
        let start = Instant::now();

        let mut cmd = Command::new("docker");
        cmd.args([
            "run",
            "--rm",
            "--network", "none",
            "--read-only",
            "--tmpfs", &format!("/tmp:rw,noexec,nosuid,size={}m", limits.max_scratch_mb),
            "--memory", &format!("{}m", limits.max_memory_mb),
            "--cpus", &limits.max_cpus.to_string(),
            "--pids-limit", &limits.pids_limit.to_string(),
            "-v", &format!("{}:/workspace:rw", workspace_dir.to_string_lossy()),
            "-w", "/workspace",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
            image,
            "python3",
            script_relative_path,
        ]);

        for arg in args {
            cmd.arg(arg);
        }

        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn().map_err(|e| format!("cannot spawn container: {e}"))?;

        let deadline = Instant::now() + Duration::from_millis(limits.max_wall_ms);
        let mut timed_out = false;

        loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(_) => break,
                None => {
                    if Instant::now() > deadline {
                        timed_out = true;
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;

        if timed_out {
            return Err(format!("Sandbox execution timed out after {}ms", limits.max_wall_ms));
        }

        let out = child.wait_with_output().map_err(|e| e.to_string())?;
        let stdout_str = String::from_utf8_lossy(&out.stdout).to_string();
        let stderr_str = String::from_utf8_lossy(&out.stderr).to_string();

        if (stdout_str.len() + stderr_str.len()) as u64 > limits.max_output_bytes {
            return Err("Sandbox execution output exceeded output byte bound".into());
        }

        Ok(SandboxExecutionResult {
            exit_code: out.status.code().unwrap_or(1),
            stdout: stdout_str,
            stderr: stderr_str,
            elapsed_ms: elapsed,
        })
    }
}
