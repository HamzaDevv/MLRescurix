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

        let env_image = std::env::var("MLRESCURIX_SANDBOX_IMAGE").ok();
        let default_image = "python:3.13-slim";
        let image = sandbox_image.or(env_image.as_deref()).unwrap_or(default_image);
        static BOX_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = BOX_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let start = Instant::now();
        let container_name = format!(
            "mlr-box-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos(),
            seq
        );

        let mut cmd = Command::new("docker");
        cmd.args([
            "run",
            "--pull=never",
            "--name", &container_name,
            "--rm",
            "--network", "none",
            "--read-only",
            "--user", "1000:1000",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true",
            "--tmpfs", &format!("/scratch:rw,noexec,nosuid,size={}m,uid=1000,gid=1000", limits.max_scratch_mb),
            "--memory", &format!("{}m", limits.max_memory_mb),
            "--cpus", &limits.max_cpus.to_string(),
            "--pids-limit", &limits.pids_limit.to_string(),
            "-v", &format!("{}:/workspace:ro", workspace_dir.to_string_lossy()),
            "-w", "/workspace",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--env", "PYTHONUNBUFFERED=1",
            "--env", "TMPDIR=/scratch",
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
        let stdout_pipe = child.stdout.take();
        let stderr_pipe = child.stderr.take();
        let max_out = limits.max_output_bytes as usize;
        let total_output = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let stdout_count = total_output.clone();
        let stderr_count = total_output.clone();

        let stdout_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut stream) = stdout_pipe {
                use std::io::Read;
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    if n == 0 { break; }
                    stdout_count.fetch_add(n as u64, std::sync::atomic::Ordering::SeqCst);
                    let keep = n.min(max_out.saturating_sub(buf.len()));
                    buf.extend_from_slice(&chunk[..keep]);
                }
            }
            buf
        });

        let stderr_handle = std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut stream) = stderr_pipe {
                use std::io::Read;
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    if n == 0 { break; }
                    stderr_count.fetch_add(n as u64, std::sync::atomic::Ordering::SeqCst);
                    let keep = n.min(max_out.saturating_sub(buf.len()));
                    buf.extend_from_slice(&chunk[..keep]);
                }
            }
            buf
        });

        let deadline = Instant::now() + Duration::from_millis(limits.max_wall_ms);
        let mut timed_out = false;

        loop {
            match child.try_wait().map_err(|e| e.to_string())? {
                Some(_) => break,
                None => {
                    if Instant::now() > deadline || total_output.load(std::sync::atomic::Ordering::SeqCst) > limits.max_output_bytes {
                        timed_out = true;
                        let _ = Command::new("docker").args(["kill", &container_name]).output();
                        let _ = Command::new("docker").args(["rm", "-f", &container_name]).output();
                        let _ = child.kill();
                        let _ = child.wait();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;

        let status = child.wait().map_err(|e| e.to_string())?;
        let stdout_bytes = stdout_handle.join().unwrap_or_default();
        let stderr_bytes = stderr_handle.join().unwrap_or_default();
        let _ = Command::new("docker").args(["rm", "-f", &container_name]).output();

        if total_output.load(std::sync::atomic::Ordering::SeqCst) > limits.max_output_bytes {
            return Err("Sandbox execution output exceeded output byte bound".into());
        }

        if timed_out {
            return Err(format!("Sandbox execution timed out after {}ms", limits.max_wall_ms));
        }
        let stdout_str = String::from_utf8_lossy(&stdout_bytes).to_string();
        let stderr_str = String::from_utf8_lossy(&stderr_bytes).to_string();

        Ok(SandboxExecutionResult {
            exit_code: status.code().unwrap_or(1),
            stdout: stdout_str,
            stderr: stderr_str,
            elapsed_ms: elapsed,
        })
    }
}
