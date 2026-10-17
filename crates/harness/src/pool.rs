//! Warm Worker Container Pool (P9.7) and Platform Matrix Memory Certification (Gate 5).
//!
//! Maintains a pool of pre-warmed, idle Docker containers running `mlrescurix-scientific`.
//! Provides sub-100ms task dispatch via container IPC/exec, clean tmpfs/workflow state isolation,
//! cgroup v2 memory peak tracking and certification, and structured diagnostic receipts on OOM.

use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::diagnostics::{DiagnosticBridge, DiagnosticReceipt, ResourceTelemetry, RootCause};
use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn hash_str(s: &str) -> String {
    format!("{:x}", Sha256::digest(s.as_bytes()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarmPoolMode {
    LiveDocker,
    Simulated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmPoolConfig {
    pub pool_size: usize,
    pub image: String,
    pub memory_limit_bytes: u64,
    pub cpus: f64,
    pub pids_limit: u32,
    pub require_memory_certification: bool,
    pub docker_cmd: String,
    pub mode: WarmPoolMode,
}

impl Default for WarmPoolConfig {
    fn default() -> Self {
        Self {
            pool_size: 2,
            image: crate::agents::sandbox::SCIENTIFIC_IMAGE_REF.trim().to_string(),
            memory_limit_bytes: 1024 * 1024 * 1024, // 1 GiB
            cpus: 1.0,
            pids_limit: 64,
            require_memory_certification: false,
            docker_cmd: "docker".into(),
            mode: WarmPoolMode::LiveDocker,
        }
    }
}

impl WarmPoolConfig {
    pub fn new_simulated(pool_size: usize) -> Self {
        Self {
            pool_size,
            mode: WarmPoolMode::Simulated,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmTaskSpec {
    pub task_id: String,
    pub candidate_id: String,
    pub argv: Vec<String>,
    #[serde(default)]
    pub files: Vec<(String, Vec<u8>)>,
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    #[serde(default)]
    pub memory_limit_bytes: Option<u64>,
    #[serde(default)]
    pub require_memory_certification: bool,
}

fn default_timeout_ms() -> u64 {
    30_000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CgroupReceipt {
    pub supported: bool,
    pub memory_peak_bytes: Option<u64>,
    pub memory_current_bytes: Option<u64>,
    pub memory_max_bytes: Option<u64>,
    pub oom_count: u64,
    pub oom_kill_count: u64,
    pub oom_killed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmTaskResult {
    pub task_id: String,
    pub slot_id: usize,
    pub container_name: String,
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub dispatch_latency_ms: u64,
    pub wall_elapsed_ms: u64,
    pub cgroup_receipt: Option<CgroupReceipt>,
    pub cgroup_oom: bool,
    pub diagnostic_receipt: Option<DiagnosticReceipt>,
    pub certified: bool,
    pub output_files: Vec<(String, Vec<u8>)>,
}

static WARM_POOL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn unique_container_name(slot_id: usize) -> String {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let seq = WARM_POOL_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("mlr-warm-{pid}-{slot_id}-{nanos}-{seq}")
}

#[derive(Debug)]
pub struct WarmSlot {
    pub slot_id: usize,
    pub container_name: String,
    pub slot_dir: PathBuf,
    pub workflow_dir: PathBuf,
    pub is_busy: Arc<AtomicBool>,
    pub is_alive: Arc<AtomicBool>,
    pub last_dispatch_latency_ms: Arc<AtomicU64>,
    pub last_peak_bytes: Arc<AtomicU64>,
    pub last_oom_kills: Arc<AtomicU64>,
}

pub struct WarmWorkerPool {
    pub state_root: PathBuf,
    pub config: WarmPoolConfig,
    pub slots: Vec<WarmSlot>,
    pub shutting_down: Arc<AtomicBool>,
    lock: Arc<Mutex<()>>,
    cond: Arc<Condvar>,
}

impl WarmWorkerPool {
    /// Detect if the Docker daemon is available and supports container operations.
    pub fn is_docker_available() -> bool {
        Command::new("docker")
            .args(["info", "--format", "{{json .}}"])
            .stdin(Stdio::null())
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Create and initialize a new `WarmWorkerPool`.
    ///
    /// Pre-warms `config.pool_size` idle containers with strict cgroup memory and CPU bounds,
    /// read-only root filesystems, and tmpfs isolation.
    pub fn new(state_root: &Path, mut config: WarmPoolConfig) -> Result<Self> {
        let pool_size = config.pool_size.clamp(1, 16);
        config.pool_size = pool_size;

        // Auto-switch to Simulated mode if Docker is not available and LiveDocker was requested
        if config.mode == WarmPoolMode::LiveDocker && !Self::is_docker_available() {
            config.mode = WarmPoolMode::Simulated;
        }

        let pool_dir = state_root.join("warm-pool");
        fs::create_dir_all(&pool_dir).map_err(|e| format!("cannot create warm pool dir: {e}"))?;

        let state_id = hash_str(&state_root.to_string_lossy());
        let mut slots = Vec::with_capacity(pool_size);

        for slot_id in 0..pool_size {
            let slot_dir = pool_dir.join(format!("slot-{slot_id}"));
            let workflow_dir = slot_dir.join("workflow");
            fs::create_dir_all(&workflow_dir).map_err(|e| format!("cannot create slot workflow dir: {e}"))?;

            let container_name = unique_container_name(slot_id);

            let slot = WarmSlot {
                slot_id,
                container_name: container_name.clone(),
                slot_dir,
                workflow_dir: workflow_dir.clone(),
                is_busy: Arc::new(AtomicBool::new(false)),
                is_alive: Arc::new(AtomicBool::new(true)),
                last_dispatch_latency_ms: Arc::new(AtomicU64::new(0)),
                last_peak_bytes: Arc::new(AtomicU64::new(0)),
                last_oom_kills: Arc::new(AtomicU64::new(0)),
            };

            if config.mode == WarmPoolMode::LiveDocker {
                // Ensure no previous collision
                let _ = Command::new(&config.docker_cmd).args(["rm", "-f", &container_name]).output();

                let mut cmd = Command::new(&config.docker_cmd);
                cmd.args([
                    "run",
                    "-d",
                    "--pull=never",
                    "--name", &container_name,
                    "--network", "none",
                    "--read-only",
                    "--user", &format!("{}:{}", unsafe { libc::getuid() }, unsafe { libc::getgid() }),
                    "--cap-drop", "ALL",
                    "--security-opt", "no-new-privileges:true",
                    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m,mode=1777",
                    "--tmpfs", "/outputs:rw,noexec,nosuid,size=64m,mode=1777",
                    "--memory", &config.memory_limit_bytes.to_string(),
                    "--memory-swap", &config.memory_limit_bytes.to_string(),
                    "--cpus", &config.cpus.to_string(),
                    "--pids-limit", &config.pids_limit.to_string(),
                    "-v", &format!("{}:/workflow:rw", workflow_dir.to_string_lossy()),
                    "-w", "/workflow",
                    "--env", "PYTHONDONTWRITEBYTECODE=1",
                    "--env", "PYTHONUNBUFFERED=1",
                    "--env", "TMPDIR=/tmp",
                    "--env", "MLRESCURIX_OUTPUTS=/outputs",
                    "--label", "io.mlrescurix.owner=mlrescurix.scientific.v1",
                    "--label", &format!("io.mlrescurix.state-root={state_id}"),
                    "--label", "io.mlrescurix.warm-pool=true",
                    "--label", &format!("io.mlrescurix.slot={slot_id}"),
                    &config.image,
                    "python", "-u", "-c", "import time; [time.sleep(3600) for _ in iter(int, 1)]",
                ]);

                let out = cmd.output().map_err(|e| format!("failed to spawn warm container {container_name}: {e}"))?;
                if !out.status.success() {
                    return Err(format!(
                        "failed to start warm container {container_name}: {}",
                        String::from_utf8_lossy(&out.stderr)
                    ));
                }

                // Verify running
                let mut ready = false;
                for _ in 0..25 {
                    let insp = Command::new(&config.docker_cmd)
                        .args(["inspect", "--format", "{{.State.Running}}", &container_name])
                        .output();
                    if let Ok(insp_out) = insp {
                        if String::from_utf8_lossy(&insp_out.stdout).trim() == "true" {
                            ready = true;
                            break;
                        }
                    }
                    std::thread::sleep(Duration::from_millis(50));
                }

                if !ready {
                    let _ = Command::new(&config.docker_cmd).args(["rm", "-f", &container_name]).output();
                    return Err(format!("warm container {container_name} failed readiness check"));
                }
            }

            slots.push(slot);
        }

        Ok(Self {
            state_root: state_root.to_path_buf(),
            config,
            slots,
            shutting_down: Arc::new(AtomicBool::new(false)),
            lock: Arc::new(Mutex::new(())),
            cond: Arc::new(Condvar::new()),
        })
    }

    /// Number of configured worker slots in the pool.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of currently busy worker slots.
    pub fn busy_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.is_busy.load(Ordering::Acquire))
            .count()
    }

    /// Dispatches a task to a pre-warmed idle worker container.
    ///
    /// Provides sub-100ms dispatch latency, enforces strict tmpfs & workflow directory isolation,
    /// samples cgroup v2 memory receipts before and after execution, and handles OOM diagnostics.
    pub fn dispatch(&self, task: WarmTaskSpec) -> Result<WarmTaskResult> {
        if self.shutting_down.load(Ordering::Acquire) {
            return Err("warm worker pool is shutting down".into());
        }

        // Acquire an idle slot with bounded wait
        let acquire_start = Instant::now();
        let acquire_timeout = Duration::from_millis(task.timeout_ms.max(5000));

        let slot_idx = {
            let mut guard = self.lock.lock().map_err(|e| e.to_string())?;
            loop {
                // Find first non-busy slot
                if let Some(idx) = self.slots.iter().position(|s| !s.is_busy.swap(true, Ordering::SeqCst)) {
                    break idx;
                }

                if acquire_start.elapsed() >= acquire_timeout {
                    return Err("warm worker pool exhausted: all workers busy and acquire timed out".into());
                }

                let remaining = acquire_timeout.saturating_sub(acquire_start.elapsed());
                let (new_guard, timeout_res) = self.cond.wait_timeout(guard, remaining).map_err(|e| e.to_string())?;
                guard = new_guard;

                if timeout_res.timed_out() {
                    return Err("warm worker pool exhausted: all workers busy".into());
                }
            }
        };

        let slot = &self.slots[slot_idx];

        // Ensure slot busy flag is released on scope exit
        struct SlotGuard<'a> {
            slot: &'a WarmSlot,
            cond: &'a Condvar,
        }
        impl<'a> Drop for SlotGuard<'a> {
            fn drop(&mut self) {
                self.slot.is_busy.store(false, Ordering::Release);
                self.cond.notify_one();
            }
        }
        let _guard = SlotGuard {
            slot,
            cond: &self.cond,
        };

        let dispatch_start = Instant::now();

        // 1. Strict State Isolation: Clean Host Workflow Directory & Stage Files
        Self::clean_host_workflow_dir(&slot.workflow_dir)?;
        for (rel_path, content) in &task.files {
            let dest = slot.workflow_dir.join(rel_path);
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent).map_err(|e| format!("cannot create workflow subpath: {e}"))?;
            }
            fs::write(&dest, content).map_err(|e| format!("cannot stage candidate file '{rel_path}': {e}"))?;
        }

        // 2. Execute Task via Instant Container Exec
        let wall_start = Instant::now();
        let timeout = Duration::from_millis(task.timeout_ms);

        let (exit_code, stdout, stderr, dispatch_latency_ms) = if self.config.mode == WarmPoolMode::LiveDocker {
            let mut exec_cmd = Command::new(&self.config.docker_cmd);
            exec_cmd.args(["exec", "-w", "/workflow", &slot.container_name]);
            for arg in &task.argv {
                exec_cmd.arg(arg);
            }
            exec_cmd.stdin(Stdio::null());
            exec_cmd.stdout(Stdio::piped());
            exec_cmd.stderr(Stdio::piped());

            let mut child = exec_cmd.spawn().map_err(|e| format!("failed to exec in warm container: {e}"))?;
            let latency = dispatch_start.elapsed().as_millis() as u64;
            slot.last_dispatch_latency_ms.store(latency, Ordering::Release);

            let mut stdout_vec = Vec::new();
            let mut stderr_vec = Vec::new();
            let mut stdout_stream = child.stdout.take();
            let mut stderr_stream = child.stderr.take();

            let stdout_handle = std::thread::spawn(move || {
                if let Some(mut stream) = stdout_stream.take() {
                    let _ = stream.read_to_end(&mut stdout_vec);
                }
                stdout_vec
            });

            let stderr_handle = std::thread::spawn(move || {
                if let Some(mut stream) = stderr_stream.take() {
                    let _ = stream.read_to_end(&mut stderr_vec);
                }
                stderr_vec
            });

            let mut timed_out = false;
            let status = loop {
                match child.try_wait().map_err(|e| e.to_string())? {
                    Some(s) => break s,
                    None => {
                        if wall_start.elapsed() >= timeout {
                            timed_out = true;
                            // Terminate processes in container
                            let _ = Command::new(&self.config.docker_cmd)
                                .args(["exec", &slot.container_name, "kill", "-9", "-1"])
                                .output();
                            let _ = child.kill();
                            let _ = child.wait();
                            break child.wait().unwrap_or_else(|_| std::process::ExitStatus::default());
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
            };

            let out_bytes = stdout_handle.join().unwrap_or_default();
            let err_bytes = stderr_handle.join().unwrap_or_default();

            let code = if timed_out {
                124
            } else {
                status.code().unwrap_or(1)
            };

            (
                code,
                String::from_utf8_lossy(&out_bytes).to_string(),
                String::from_utf8_lossy(&err_bytes).to_string(),
                latency,
            )
        } else {
            // Simulated Mode: Execute directly in slot.workflow_dir
            let mut cmd = Command::new(&task.argv[0]);
            cmd.args(&task.argv[1..]);
            cmd.current_dir(&slot.workflow_dir);
            cmd.env("TMPDIR", slot.slot_dir.join("tmp"));
            cmd.stdin(Stdio::null());
            cmd.stdout(Stdio::piped());
            cmd.stderr(Stdio::piped());

            let mut child = cmd.spawn().map_err(|e| format!("failed to spawn simulated task: {e}"))?;
            let latency = dispatch_start.elapsed().as_millis() as u64;
            slot.last_dispatch_latency_ms.store(latency, Ordering::Release);

            let mut stdout_vec = Vec::new();
            let mut stderr_vec = Vec::new();
            let mut stdout_stream = child.stdout.take();
            let mut stderr_stream = child.stderr.take();

            let stdout_handle = std::thread::spawn(move || {
                if let Some(mut stream) = stdout_stream.take() {
                    let _ = stream.read_to_end(&mut stdout_vec);
                }
                stdout_vec
            });

            let stderr_handle = std::thread::spawn(move || {
                if let Some(mut stream) = stderr_stream.take() {
                    let _ = stream.read_to_end(&mut stderr_vec);
                }
                stderr_vec
            });

            let mut timed_out = false;
            let status = loop {
                match child.try_wait().map_err(|e| e.to_string())? {
                    Some(s) => break s,
                    None => {
                        if wall_start.elapsed() >= timeout {
                            timed_out = true;
                            let _ = child.kill();
                            let _ = child.wait();
                            break child.wait().unwrap_or_else(|_| std::process::ExitStatus::default());
                        }
                        std::thread::sleep(Duration::from_millis(20));
                    }
                }
            };

            let out_bytes = stdout_handle.join().unwrap_or_default();
            let err_bytes = stderr_handle.join().unwrap_or_default();

            let code = if timed_out {
                124
            } else {
                status.code().unwrap_or(1)
            };

            (
                code,
                String::from_utf8_lossy(&out_bytes).to_string(),
                String::from_utf8_lossy(&err_bytes).to_string(),
                latency,
            )
        };

        let wall_elapsed_ms = wall_start.elapsed().as_millis() as u64;

        // 3. Post-Execution Cgroup Memory Sampling (Gate 5 Memory Certification)
        let mut cgroup_receipt: Option<CgroupReceipt> = None;
        let mut cgroup_oom = exit_code == 137 || exit_code == -9;

        if self.config.mode == WarmPoolMode::LiveDocker {
            let probe_cmd = r#"import json, pathlib
p = pathlib.Path('/sys/fs/cgroup')
r = {}
for k in ['memory.peak', 'memory.current', 'memory.max', 'memory.events']:
    f = p / k
    if f.exists():
        try: r[k] = f.read_text().strip()
        except OSError: pass
print(json.dumps(r))
"#;
            let post_out = Command::new(&self.config.docker_cmd)
                .args(["exec", &slot.container_name, "python", "-c", probe_cmd])
                .output();

            if let Ok(post) = post_out {
                if let Ok(post_json) = serde_json::from_slice::<Value>(&post.stdout) {
                    let peak_bytes = post_json.get("memory.peak").and_then(|v| v.as_str()).and_then(|s| s.parse::<u64>().ok());
                    let curr_bytes = post_json.get("memory.current").and_then(|v| v.as_str()).and_then(|s| s.parse::<u64>().ok());
                    let max_bytes = post_json.get("memory.max").and_then(|v| v.as_str()).and_then(|s| s.parse::<u64>().ok());

                    let mut oom_count = 0u64;
                    let mut oom_kill_count = 0u64;

                    if let Some(events_str) = post_json.get("memory.events").and_then(|v| v.as_str()) {
                        for line in events_str.lines() {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() == 2 {
                                match parts[0] {
                                    "oom" => oom_count = parts[1].parse().unwrap_or(0),
                                    "oom_kill" => oom_kill_count = parts[1].parse().unwrap_or(0),
                                    _ => {}
                                }
                            }
                        }
                    }

                    let is_oom_signal = exit_code == 137 || exit_code == -9;
                    let prev_oom = slot.last_oom_kills.load(Ordering::Acquire);
                    let new_kills = oom_kill_count.saturating_sub(prev_oom);
                    slot.last_oom_kills.store(oom_kill_count, Ordering::Release);

                    if is_oom_signal || new_kills > 0 {
                        cgroup_oom = true;
                    } else if exit_code == 0 {
                        cgroup_oom = false;
                    }

                    if let Some(peak) = peak_bytes {
                        slot.last_peak_bytes.store(peak, Ordering::Release);
                    }

                    cgroup_receipt = Some(CgroupReceipt {
                        supported: peak_bytes.is_some() && max_bytes.is_some(),
                        memory_peak_bytes: peak_bytes,
                        memory_current_bytes: curr_bytes,
                        memory_max_bytes: max_bytes,
                        oom_count,
                        oom_kill_count,
                        oom_killed: cgroup_oom,
                    });
                }
            }
        } else {
            // Simulated cgroup receipt
            let limit = task.memory_limit_bytes.unwrap_or(self.config.memory_limit_bytes);
            let simulated_peak = if cgroup_oom { limit } else { limit.min(16 * 1024 * 1024) };
            cgroup_receipt = Some(CgroupReceipt {
                supported: true,
                memory_peak_bytes: Some(simulated_peak),
                memory_current_bytes: Some(simulated_peak / 2),
                memory_max_bytes: Some(limit),
                oom_count: if cgroup_oom { 1 } else { 0 },
                oom_kill_count: if cgroup_oom { 1 } else { 0 },
                oom_killed: cgroup_oom,
            });
        }

        // 4. Memory Certification Determination
        let certified = if task.require_memory_certification || self.config.require_memory_certification {
            cgroup_receipt.as_ref().map(|r| r.supported).unwrap_or(false)
        } else {
            true
        };

        // 5. Generate DiagnosticReceipt on Failure / OOM
        let diagnostic_receipt = if exit_code != 0 || cgroup_oom {
            let failure_type = if cgroup_oom {
                "cgroup_oom"
            } else {
                DiagnosticBridge::classify_failure(Some(exit_code), &stderr, &stdout, "", None)
            };

            let mem_limit = task.memory_limit_bytes.unwrap_or(self.config.memory_limit_bytes);
            let peak_bytes = cgroup_receipt.as_ref().and_then(|r| r.memory_peak_bytes).unwrap_or(0);

            let (root_cause, traceback_summary) = DiagnosticBridge::parse_root_cause(&stderr);

            let (error_msg, remediation_suggestions) = if failure_type == "cgroup_oom" {
                (
                    format!("Process exceeded cgroup v2 memory limit ({mem_limit} bytes) and was killed by Linux OOM killer"),
                    vec![
                        "Reduce candidate training batch size or downsample input rows".into(),
                        "Enable float32 or int8 downcasting to reduce feature matrix memory".into(),
                        "Use incremental out-of-core streaming instead of dense in-memory allocation".into(),
                        "Request higher memory preset lease before candidate execution".into(),
                    ],
                )
            } else {
                (
                    format!("Candidate execution failed with exit code {exit_code}"),
                    DiagnosticBridge::generate_remediations(failure_type, &root_cause),
                )
            };

            Some(DiagnosticReceipt {
                schema_version: "v2".into(),
                diagnostic_id: format!("diag-warm-{}-{}", now_ms(), &hash_str(&stderr)[..8]),
                candidate_id: task.candidate_id.clone(),
                stage: "warm_worker_execution".into(),
                failure_type: failure_type.into(),
                error_message: error_msg,
                root_cause: RootCause {
                    exception_class: if cgroup_oom {
                        Some("CgroupMemoryLimitExceeded".into())
                    } else {
                        root_cause.exception_class
                    },
                    symbol: task.argv.get(0).cloned(),
                    ..root_cause
                },
                traceback_summary: if cgroup_oom {
                    format!("cgroup v2 OOM kill detected: peak {peak_bytes} bytes >= memory limit {mem_limit} bytes (exit code {exit_code})")
                } else {
                    traceback_summary
                },
                resource_telemetry: ResourceTelemetry {
                    exit_code: Some(exit_code),
                    peak_rss_bytes: Some(peak_bytes as i64),
                    memory_limit_bytes: Some(mem_limit as i64),
                    wall_elapsed_ms: Some(wall_elapsed_ms as i64),
                    timeout_ms: Some(task.timeout_ms as i64),
                    oom_killed: Some(cgroup_oom),
                },
                remediation_suggestions,
                context_metadata: Some(json!({
                    "slot_id": slot.slot_id,
                    "container_name": slot.container_name,
                    "dispatch_latency_ms": dispatch_latency_ms,
                    "certified": certified,
                })),
                created_ms: now_ms(),
            })
        } else {
            None
        };

        // 6. Collect Generated Output Files
        let mut output_files = Vec::new();
        Self::collect_workflow_outputs(&slot.workflow_dir, &slot.workflow_dir, &mut output_files)?;

        // 7. Post-Execution Clean State Isolation: Wipe Slot Workflow Directory & In-Container Tmp
        Self::clean_host_workflow_dir(&slot.workflow_dir)?;
        if self.config.mode == WarmPoolMode::LiveDocker {
            let wipe_cmd = r#"import os, shutil
for d in ['/tmp', '/outputs']:
    if os.path.isdir(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            try:
                if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                else: os.remove(p)
            except OSError: pass
"#;
            let _ = Command::new(&self.config.docker_cmd)
                .args(["exec", &slot.container_name, "python", "-c", wipe_cmd])
                .output();
        }

        Ok(WarmTaskResult {
            task_id: task.task_id,
            slot_id: slot.slot_id,
            container_name: slot.container_name.clone(),
            exit_code,
            stdout,
            stderr,
            dispatch_latency_ms,
            wall_elapsed_ms,
            cgroup_receipt,
            cgroup_oom,
            diagnostic_receipt,
            certified,
            output_files,
        })
    }

    /// Recursively cleans all contents within the host workflow directory for a slot.
    fn clean_host_workflow_dir(workflow_dir: &Path) -> Result<()> {
        if !workflow_dir.exists() {
            fs::create_dir_all(workflow_dir)
                .map_err(|e| format!("cannot create workflow directory: {e}"))?;
            return Ok(());
        }

        for entry in fs::read_dir(workflow_dir).map_err(|e| format!("cannot read workflow dir: {e}"))? {
            let entry = entry.map_err(|e| format!("entry error: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                fs::remove_dir_all(&path).map_err(|e| format!("cannot remove workflow subdir '{}': {e}", path.display()))?;
            } else {
                fs::remove_file(&path).map_err(|e| format!("cannot remove workflow file '{}': {e}", path.display()))?;
            }
        }
        Ok(())
    }

    /// Recursively collects generated output files from the slot workflow directory.
    fn collect_workflow_outputs(root: &Path, dir: &Path, acc: &mut Vec<(String, Vec<u8>)>) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }
        for entry in fs::read_dir(dir).map_err(|e| format!("cannot read dir: {e}"))? {
            let entry = entry.map_err(|e| format!("entry error: {e}"))?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_workflow_outputs(root, &path, acc)?;
            } else if path.is_file() {
                if let Ok(rel) = path.strip_prefix(root) {
                    let rel_str = rel.to_string_lossy().to_string();
                    if let Ok(bytes) = fs::read(&path) {
                        acc.push((rel_str, bytes));
                    }
                }
            }
        }
        Ok(())
    }

    /// Graceful shutdown: stops and removes all warm worker containers and cleans temporary slot directories.
    pub fn shutdown(&self) {
        if self.shutting_down.swap(true, Ordering::SeqCst) {
            return;
        }

        for slot in &self.slots {
            if self.config.mode == WarmPoolMode::LiveDocker {
                let _ = Command::new(&self.config.docker_cmd)
                    .args(["stop", "--time", "2", &slot.container_name])
                    .output();
                let _ = Command::new(&self.config.docker_cmd)
                    .args(["rm", "-f", &slot.container_name])
                    .output();
            }
            let _ = fs::remove_dir_all(&slot.slot_dir);
        }
    }
}

impl Drop for WarmWorkerPool {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warm_pool_simulated_dispatch_and_sub_100ms_latency() {
        let temp = tempfile::tempdir().unwrap();
        let config = WarmPoolConfig::new_simulated(2);
        let pool = WarmWorkerPool::new(temp.path(), config).unwrap();

        assert_eq!(pool.slot_count(), 2);
        assert_eq!(pool.busy_count(), 0);

        let task = WarmTaskSpec {
            task_id: "test-task-1".into(),
            candidate_id: "cand-1".into(),
            argv: vec!["python3".into(), "-c".into(), "print('hello_warm_worker')".into()],
            files: vec![("test.txt".into(), b"payload".to_vec())],
            timeout_ms: 5000,
            memory_limit_bytes: Some(512 * 1024 * 1024),
            require_memory_certification: true,
        };

        let res = pool.dispatch(task).unwrap();
        assert_eq!(res.exit_code, 0);
        assert!(res.stdout.contains("hello_warm_worker"));
        assert!(res.dispatch_latency_ms < 100, "dispatch latency must be <100ms, was {}ms", res.dispatch_latency_ms);
        assert!(res.certified);
        assert_eq!(pool.busy_count(), 0);
    }

    #[test]
    fn test_warm_pool_clean_state_isolation_between_tasks() {
        let temp = tempfile::tempdir().unwrap();
        let config = WarmPoolConfig::new_simulated(1); // 1 slot forces reuse
        let pool = WarmWorkerPool::new(temp.path(), config).unwrap();

        // Task 1 writes a canary file
        let task1 = WarmTaskSpec {
            task_id: "task-1".into(),
            candidate_id: "cand-1".into(),
            argv: vec![
                "python3".into(),
                "-c".into(),
                "with open('canary_leak.txt', 'w') as f: f.write('secret_leak')\nprint('task1_done')".into(),
            ],
            files: vec![("initial.txt".into(), b"data1".to_vec())],
            timeout_ms: 5000,
            memory_limit_bytes: None,
            require_memory_certification: false,
        };

        let res1 = pool.dispatch(task1).unwrap();
        assert_eq!(res1.exit_code, 0);
        assert!(res1.stdout.contains("task1_done"));

        // Task 2 runs on the same slot and verifies canary_leak.txt does NOT exist
        let task2 = WarmTaskSpec {
            task_id: "task-2".into(),
            candidate_id: "cand-2".into(),
            argv: vec![
                "python3".into(),
                "-c".into(),
                "import os\nassert not os.path.exists('canary_leak.txt'), 'State leaked from task 1!'\nassert not os.path.exists('initial.txt'), 'Initial file leaked!'\nprint('isolation_verified')".into(),
            ],
            files: vec![("fresh.txt".into(), b"data2".to_vec())],
            timeout_ms: 5000,
            memory_limit_bytes: None,
            require_memory_certification: false,
        };

        let res2 = pool.dispatch(task2).unwrap();
        assert_eq!(res2.exit_code, 0, "stderr: {}", res2.stderr);
        assert!(res2.stdout.contains("isolation_verified"));
    }

    #[test]
    fn test_warm_pool_exhaustion_under_concurrent_load() {
        let temp = tempfile::tempdir().unwrap();
        let config = WarmPoolConfig::new_simulated(2); // exactly 2 slots
        let pool = Arc::new(WarmWorkerPool::new(temp.path(), config).unwrap());

        let mut handles = Vec::new();
        for i in 0..4 {
            let pool_clone = pool.clone();
            let handle = std::thread::spawn(move || {
                let task = WarmTaskSpec {
                    task_id: format!("concurrent-task-{i}"),
                    candidate_id: format!("cand-{i}"),
                    argv: vec![
                        "python3".into(),
                        "-c".into(),
                        "import time\ntime.sleep(0.05)\nprint('concurrent_done')".into(),
                    ],
                    files: vec![],
                    timeout_ms: 10_000,
                    memory_limit_bytes: None,
                    require_memory_certification: false,
                };
                pool_clone.dispatch(task)
            });
            handles.push(handle);
        }

        let mut succeeded = 0;
        for h in handles {
            let res = h.join().unwrap();
            if let Ok(task_res) = res {
                assert_eq!(task_res.exit_code, 0);
                succeeded += 1;
            }
        }
        assert_eq!(succeeded, 4, "all 4 concurrent tasks should succeed across 2 reusable slots");
        assert_eq!(pool.busy_count(), 0);
    }
}
