//! Resource lease broker: capability discovery, presets, headroom, and admission feasibility.
//!
//! P3.1: Effective CPU, memory, disk, and accelerator discovery across host and container VM,
//! resource request presets, configurable headroom, and early infeasible-request rejection.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use rusqlite::{params, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::core::Policy;
use crate::execution::{JobSpec, Probe, OUTPUT_CAP};
use crate::execution::resources::ResourceFeatures;
use crate::Result;

/// Default CPU headroom reserved for host and control plane (1000 millicores = 1 core).
pub const DEFAULT_CPU_HEADROOM_MILLICORES: u32 = 1000;
/// Default memory headroom reserved for host and control plane (512 MiB).
pub const DEFAULT_MEMORY_HEADROOM_BYTES: u64 = 512 * 1024 * 1024;
/// Default disk headroom reserved for state root (64 MiB, matching DISK_HEADROOM).
pub const DEFAULT_DISK_HEADROOM_BYTES: u64 = 64 * 1024 * 1024;

/// Minimum allowable memory request (64 MiB).
pub const MIN_MEMORY_BYTES: u64 = 64 * 1024 * 1024;
/// Maximum allowable memory request (64 GiB).
pub const MAX_MEMORY_BYTES: u64 = 64 * 1024 * 1024 * 1024;

/// Supported hardware accelerator types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorType {
    AppleMetal,
    NvidiaCuda,
}

/// Discovered hardware accelerator details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AcceleratorInfo {
    pub kind: AcceleratorType,
    pub name: String,
    pub device_count: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

/// Host hardware capabilities discovered via system metrics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostCapabilities {
    pub logical_cpus: u32,
    pub total_memory_bytes: u64,
    pub disk_total_bytes: u64,
    pub disk_available_bytes: u64,
    pub accelerators: Vec<AcceleratorInfo>,
}

/// Container VM capabilities discovered via container daemon.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerVmCapabilities {
    pub docker_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vm_cpus: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vm_memory_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cgroup_version: Option<String>,
    pub features: ResourceFeatures,
    pub accelerators: Vec<AcceleratorInfo>,
}

/// Effective capabilities synthesized across host and container VM.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveCapabilities {
    pub host: HostCapabilities,
    pub container_vm: ContainerVmCapabilities,
    pub effective_cpus: u32,
    pub effective_memory_bytes: u64,
    pub effective_disk_bytes: u64,
    pub effective_accelerators: Vec<AcceleratorInfo>,
    pub memory_certification_supported: bool,
}

impl EffectiveCapabilities {
    pub fn compute(host: HostCapabilities, container_vm: ContainerVmCapabilities, is_container_job: bool) -> Self {
        let (effective_cpus, effective_memory_bytes) = if is_container_job && container_vm.docker_available {
            let cpus = host.logical_cpus.min(container_vm.vm_cpus.unwrap_or(host.logical_cpus));
            let mem = host.total_memory_bytes.min(container_vm.vm_memory_bytes.unwrap_or(host.total_memory_bytes));
            (cpus, mem)
        } else {
            (host.logical_cpus, host.total_memory_bytes)
        };
        let effective_disk_bytes = host.disk_available_bytes;
        let effective_accelerators = if is_container_job {
            if !container_vm.accelerators.is_empty() {
                container_vm.accelerators.clone()
            } else {
                host.accelerators.clone()
            }
        } else {
            host.accelerators.clone()
        };
        let memory_certification_supported = container_vm.features.require().is_ok();
        Self {
            host,
            container_vm,
            effective_cpus,
            effective_memory_bytes,
            effective_disk_bytes,
            effective_accelerators,
            memory_certification_supported,
        }
    }

    pub fn available(&self, headroom: &ResourceHeadroom) -> AvailableCapacity {
        let cpu_headroom_cores = (headroom.cpu_millicores as f64 / 1000.0).ceil() as u32;
        let available_cpus = self.effective_cpus.saturating_sub(cpu_headroom_cores).max(1);
        let available_memory_bytes = self.effective_memory_bytes.saturating_sub(headroom.memory_bytes);
        let available_disk_bytes = self.effective_disk_bytes.saturating_sub(headroom.disk_bytes);
        AvailableCapacity {
            available_cpus,
            available_memory_bytes,
            available_disk_bytes,
            memory_certification_supported: self.memory_certification_supported,
            accelerators: self.effective_accelerators.clone(),
        }
    }
}

/// Configured resource headroom for system stability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResourceHeadroom {
    pub cpu_millicores: u32,
    pub memory_bytes: u64,
    pub disk_bytes: u64,
}

impl Default for ResourceHeadroom {
    fn default() -> Self {
        Self {
            cpu_millicores: DEFAULT_CPU_HEADROOM_MILLICORES,
            memory_bytes: DEFAULT_MEMORY_HEADROOM_BYTES,
            disk_bytes: DEFAULT_DISK_HEADROOM_BYTES,
        }
    }
}

impl ResourceHeadroom {
    /// Load headroom from environment overrides, configuration file, or defaults.
    pub fn load(root: &Path) -> Self {
        let mut headroom = Self::default();

        // 1. Try file in root/resource_headroom.json
        let file_path = root.join("resource_headroom.json");
        if file_path.is_file() {
            if let Ok(bytes) = fs::read(&file_path) {
                if let Ok(loaded) = serde_json::from_slice::<ResourceHeadroom>(&bytes) {
                    headroom = loaded;
                }
            }
        }

        // 2. Environment variable overrides take precedence
        if let Ok(val) = std::env::var("MLRESCURIX_HEADROOM_CPU_MILLICORES") {
            if let Ok(parsed) = val.parse::<u32>() {
                headroom.cpu_millicores = parsed;
            }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_HEADROOM_MEMORY_BYTES") {
            if let Ok(parsed) = val.parse::<u64>() {
                headroom.memory_bytes = parsed;
            }
        }
        if let Ok(val) = std::env::var("MLRESCURIX_HEADROOM_DISK_BYTES") {
            if let Ok(parsed) = val.parse::<u64>() {
                headroom.disk_bytes = parsed;
            }
        }

        headroom
    }
}

/// Available capacity after subtracting configured headroom.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableCapacity {
    pub available_cpus: u32,
    pub available_memory_bytes: u64,
    pub available_disk_bytes: u64,
    pub memory_certification_supported: bool,
    pub accelerators: Vec<AcceleratorInfo>,
}

/// Named resource request presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePreset {
    Micro,
    Standard,
    Heavy,
    Custom,
}

/// Configuration for a concrete resource preset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetConfig {
    pub preset: ResourcePreset,
    pub threads: u32,
    pub memory_bytes: u64,
    pub max_output_bytes: u64,
}

impl PresetConfig {
    pub fn default_for(preset: ResourcePreset) -> Self {
        match preset {
            ResourcePreset::Micro => Self {
                preset,
                threads: 1,
                memory_bytes: 512 * 1024 * 1024,
                max_output_bytes: 1 * 1024 * 1024,
            },
            ResourcePreset::Standard => Self {
                preset,
                threads: 2,
                memory_bytes: 2 * 1024 * 1024 * 1024,
                max_output_bytes: 2 * 1024 * 1024,
            },
            ResourcePreset::Heavy => Self {
                preset,
                threads: 4,
                memory_bytes: 8 * 1024 * 1024 * 1024,
                max_output_bytes: 4 * 1024 * 1024,
            },
            ResourcePreset::Custom => Self {
                preset,
                threads: 1,
                memory_bytes: 1024 * 1024 * 1024,
                max_output_bytes: 4 * 1024 * 1024,
            },
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.threads == 0 || self.threads > 256 {
            return Err(format!("invalid preset threads {}: must be in 1..=256", self.threads));
        }
        if !(MIN_MEMORY_BYTES..=MAX_MEMORY_BYTES).contains(&self.memory_bytes) {
            return Err(format!(
                "invalid preset memory {}: must be between 64 MiB and 64 GiB",
                self.memory_bytes
            ));
        }
        if self.max_output_bytes == 0 || self.max_output_bytes > OUTPUT_CAP {
            return Err(format!(
                "invalid preset max_output_bytes {}: must be between 1 and {}",
                self.max_output_bytes, OUTPUT_CAP
            ));
        }
        Ok(())
    }
}

pub fn validate_preset(preset: ResourcePreset) -> Result<()> {
    PresetConfig::default_for(preset).validate()
}

/// Load presets from configuration file or return defaults.
pub fn load_presets(root: &Path) -> HashMap<ResourcePreset, PresetConfig> {
    let mut map = HashMap::new();
    for p in [ResourcePreset::Micro, ResourcePreset::Standard, ResourcePreset::Heavy, ResourcePreset::Custom] {
        map.insert(p, PresetConfig::default_for(p));
    }

    let file_path = root.join("resource_presets.json");
    if file_path.is_file() {
        if let Ok(bytes) = fs::read(&file_path) {
            if let Ok(loaded) = serde_json::from_slice::<HashMap<String, Value>>(&bytes) {
                for (name, val) in loaded {
                    let preset_opt = match name.as_str() {
                        "micro" => Some(ResourcePreset::Micro),
                        "standard" => Some(ResourcePreset::Standard),
                        "heavy" => Some(ResourcePreset::Heavy),
                        "custom" => Some(ResourcePreset::Custom),
                        _ => None,
                    };
                    if let Some(preset) = preset_opt {
                        let mut cfg = PresetConfig::default_for(preset);
                        if let Some(t) = val.get("threads").and_then(|v| v.as_u64()) {
                            cfg.threads = t as u32;
                        }
                        if let Some(m) = val.get("memory_bytes").and_then(|v| v.as_u64()) {
                            cfg.memory_bytes = m;
                        }
                        if let Some(o) = val.get("max_output_bytes").and_then(|v| v.as_u64()) {
                            cfg.max_output_bytes = o;
                        }
                        if cfg.validate().is_ok() {
                            map.insert(preset, cfg);
                        }
                    }
                }
            }
        }
    }

    map
}

/// Trait abstracting capability discovery for hermetic unit testing and live system discovery.
pub trait CapabilityProbe {
    fn probe_host(&self, root: &Path) -> Result<HostCapabilities>;
    fn probe_container_vm(&self) -> Result<ContainerVmCapabilities>;
}

/// Live system implementation of CapabilityProbe.
#[derive(Debug, Default, Clone)]
pub struct SystemCapabilityProbe;

impl CapabilityProbe for SystemCapabilityProbe {
    fn probe_host(&self, root: &Path) -> Result<HostCapabilities> {
        let logical_cpus = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);
        let total_memory_bytes = get_host_memory_bytes();
        let (disk_total_bytes, disk_available_bytes) = get_disk_capacity(root)?;
        let accelerators = probe_host_accelerators();

        Ok(HostCapabilities {
            logical_cpus,
            total_memory_bytes,
            disk_total_bytes,
            disk_available_bytes,
            accelerators,
        })
    }

    fn probe_container_vm(&self) -> Result<ContainerVmCapabilities> {
        match crate::execution::resources::docker_info() {
            Ok(info) => {
                let vm_cpus = info["NCPU"].as_u64().map(|n| n as u32);
                let vm_memory_bytes = info["MemTotal"].as_u64();
                let cgroup_version = info["CgroupVersion"].as_str().map(|s| s.to_string());
                let features = ResourceFeatures::from_info(&info);
                let accelerators = probe_container_accelerators(&info);
                Ok(ContainerVmCapabilities {
                    docker_available: true,
                    vm_cpus,
                    vm_memory_bytes,
                    cgroup_version,
                    features,
                    accelerators,
                })
            }
            Err(e) => Ok(ContainerVmCapabilities {
                docker_available: false,
                vm_cpus: None,
                vm_memory_bytes: None,
                cgroup_version: None,
                features: ResourceFeatures {
                    available: false,
                    cgroup_v2: false,
                    memory_limit: false,
                    swap_limit: false,
                    cpu_quota: false,
                    pids_limit: false,
                    reason: Some(e),
                },
                accelerators: Vec::new(),
            }),
        }
    }
}

#[cfg(target_os = "macos")]
fn get_host_memory_bytes() -> u64 {
    let mut mem: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let res = unsafe {
        libc::sysctlbyname(
            b"hw.memsize\0".as_ptr() as *const libc::c_char,
            &mut mem as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if res == 0 && mem > 0 {
        mem
    } else {
        8 * 1024 * 1024 * 1024
    }
}

#[cfg(target_os = "linux")]
fn get_host_memory_bytes() -> u64 {
    let pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) };
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if pages > 0 && page_size > 0 {
        (pages as u64).saturating_mul(page_size as u64)
    } else {
        8 * 1024 * 1024 * 1024
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn get_host_memory_bytes() -> u64 {
    8 * 1024 * 1024 * 1024
}

fn get_disk_capacity(path: &Path) -> Result<(u64, u64)> {
    let mut check_path = path.to_path_buf();
    while !check_path.exists() {
        if let Some(parent) = check_path.parent() {
            check_path = parent.to_path_buf();
        } else {
            break;
        }
    }
    let c_path = std::ffi::CString::new(check_path.as_os_str().as_encoded_bytes())
        .map_err(|e| e.to_string())?;
    let mut stats = unsafe { std::mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stats) } != 0 {
        return Err("cannot inspect available state storage".into());
    }
    let available = (stats.f_bavail as u64).saturating_mul(stats.f_frsize as u64);
    let total = (stats.f_blocks as u64).saturating_mul(stats.f_frsize as u64);
    Ok((total, available))
}

#[cfg(target_os = "macos")]
fn probe_host_accelerators() -> Vec<AcceleratorInfo> {
    let mut is_arm64: i32 = 0;
    let mut len = std::mem::size_of::<i32>();
    let ret = unsafe {
        libc::sysctlbyname(
            b"hw.optional.arm64\0".as_ptr() as *const libc::c_char,
            &mut is_arm64 as *mut i32 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if ret == 0 && is_arm64 == 1 {
        let mut brand = [0u8; 128];
        let mut brand_len = brand.len();
        let _ = unsafe {
            libc::sysctlbyname(
                b"machdep.cpu.brand_string\0".as_ptr() as *const libc::c_char,
                brand.as_mut_ptr() as *mut libc::c_void,
                &mut brand_len,
                std::ptr::null_mut(),
                0,
            )
        };
        let brand_str = String::from_utf8_lossy(&brand[..brand_len.saturating_sub(1)]).trim().to_string();
        let name = if brand_str.is_empty() {
            "Apple Silicon Metal GPU".to_string()
        } else {
            format!("{brand_str} GPU")
        };
        vec![AcceleratorInfo {
            kind: AcceleratorType::AppleMetal,
            name,
            device_count: 1,
            memory_bytes: None,
        }]
    } else {
        Vec::new()
    }
}

#[cfg(not(target_os = "macos"))]
fn probe_host_accelerators() -> Vec<AcceleratorInfo> {
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .stdin(std::process::Stdio::null())
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut accels = Vec::new();
            for line in stdout.lines() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if !parts.is_empty() && !parts[0].is_empty() {
                    let name = parts[0].to_string();
                    let mem_bytes = parts.get(1).and_then(|m| m.parse::<u64>().ok()).map(|m| m * 1024 * 1024);
                    accels.push(AcceleratorInfo {
                        kind: AcceleratorType::NvidiaCuda,
                        name,
                        device_count: 1,
                        memory_bytes: mem_bytes,
                    });
                }
            }
            if !accels.is_empty() {
                return accels;
            }
        }
    }
    Vec::new()
}

fn probe_container_accelerators(_info: &Value) -> Vec<AcceleratorInfo> {
    // Docker VM container accelerators
    Vec::new()
}

/// Discover effective capabilities using the default system probe.
pub fn discover_effective_capabilities(root: &Path) -> Result<EffectiveCapabilities> {
    discover_effective_capabilities_with_probe(root, &SystemCapabilityProbe)
}

/// Discover effective capabilities with a specific capability probe.
pub fn discover_effective_capabilities_with_probe(
    root: &Path,
    probe: &impl CapabilityProbe,
) -> Result<EffectiveCapabilities> {
    let host = probe.probe_host(root)?;
    let container_vm = probe.probe_container_vm()?;
    Ok(EffectiveCapabilities::compute(host, container_vm, true))
}

/// Get available capacity accounting for configured headroom.
pub fn get_available_capacity(root: &Path) -> Result<AvailableCapacity> {
    get_available_capacity_with_probe(root, &SystemCapabilityProbe, &ResourceHeadroom::load(root))
}

/// Get available capacity with a specific probe and headroom configuration.
pub fn get_available_capacity_with_probe(
    root: &Path,
    probe: &impl CapabilityProbe,
    headroom: &ResourceHeadroom,
) -> Result<AvailableCapacity> {
    let eff = discover_effective_capabilities_with_probe(root, probe)?;
    Ok(eff.available(headroom))
}

/// Resolve job spec preset parameters if a preset was declared.
pub fn resolve_job_spec_preset(root: &Path, spec: &mut JobSpec) -> Result<()> {
    if let Some(preset) = spec.resources.preset {
        let presets = load_presets(root);
        let config = presets
            .get(&preset)
            .cloned()
            .unwrap_or_else(|| PresetConfig::default_for(preset));
        config.validate()?;

        if preset != ResourcePreset::Custom {
            // Apply preset parameters if using default values
            if spec.threads == 1 && config.threads > 1 {
                spec.threads = config.threads;
            }
            if spec.resources.memory_bytes == 1024 * 1024 * 1024 && config.memory_bytes != 1024 * 1024 * 1024 {
                spec.resources.memory_bytes = config.memory_bytes;
            }
            if spec.max_output_bytes == 1024 || spec.max_output_bytes == 4096 {
                spec.max_output_bytes = config.max_output_bytes.min(OUTPUT_CAP);
            }
        }
    }
    Ok(())
}

/// Check if a job request is feasible given discovered capabilities and configured headroom.
///
/// Rejects early before queue admission:
/// 1. Exceeds available memory capacity (effective minus headroom)
/// 2. Exceeds available CPU threads (effective minus headroom)
/// 3. Exceeds available disk storage (effective minus headroom)
/// 4. Requests an unsupported accelerator
/// 5. Requests memory certification on unsupported environments
pub fn check_request_feasibility(root: &Path, spec: &JobSpec) -> Result<()> {
    check_request_feasibility_with_probe(root, spec, &SystemCapabilityProbe, &ResourceHeadroom::load(root))
}

/// Check request feasibility with a specific probe and headroom.
pub fn check_request_feasibility_with_probe(
    root: &Path,
    spec: &JobSpec,
    probe: &impl CapabilityProbe,
    headroom: &ResourceHeadroom,
) -> Result<()> {
    let is_container_job = matches!(spec.probe, Probe::Scientific { .. }) || spec.environment_id.is_some();
    let host = probe.probe_host(root)?;
    let container_vm = probe.probe_container_vm()?;

    if is_container_job && !container_vm.docker_available {
        return Err("infeasible request: scientific container job requested but Docker environment is unavailable".into());
    }

    let effective = EffectiveCapabilities::compute(host, container_vm, is_container_job);
    let available = effective.available(headroom);

    // 1. Check memory certification requirement (preserves D24 / P1.5 invariant)
    if spec.resources.require_memory_certification {
        if !is_container_job {
            return Err("required memory certification unsupported for host control probes".into());
        }
        if !available.memory_certification_supported {
            return Err(
                "required memory certification unsupported: Docker cgroup v2 memory/swap/CPU/PID enforcement unavailable"
                    .into(),
            );
        }
    }

    // 2. Check requested memory against available capacity
    if spec.resources.memory_bytes > available.available_memory_bytes {
        let req_mb = spec.resources.memory_bytes / (1024 * 1024);
        let avail_mb = available.available_memory_bytes / (1024 * 1024);
        let eff_mb = effective.effective_memory_bytes / (1024 * 1024);
        let head_mb = headroom.memory_bytes / (1024 * 1024);
        return Err(format!(
            "infeasible request: requested memory {req_mb} MiB exceeds available capacity {avail_mb} MiB (effective {eff_mb} MiB minus {head_mb} MiB headroom)"
        ));
    }

    // 3. Check requested CPU threads against available CPU cores
    if spec.threads > available.available_cpus {
        return Err(format!(
            "infeasible request: requested threads {} exceeds available CPU capacity {} (effective {} minus {} headroom)",
            spec.threads,
            available.available_cpus,
            effective.effective_cpus,
            headroom.cpu_millicores / 1000
        ));
    }

    // 4. Check requested max_output_bytes against available disk
    if spec.max_output_bytes > available.available_disk_bytes {
        let req_mb = spec.max_output_bytes / (1024 * 1024);
        let avail_mb = available.available_disk_bytes / (1024 * 1024);
        return Err(format!(
            "infeasible request: requested output bound {req_mb} MiB exceeds available disk storage {avail_mb} MiB"
        ));
    }

    // 5. Check requested accelerator
    if let Some(requested_accel) = spec.resources.accelerator {
        let found = available.accelerators.iter().any(|a| a.kind == requested_accel);
        if !found {
            return Err(format!(
                "infeasible request: requested accelerator '{:?}' is not available on this system",
                requested_accel
            ));
        }
    }

    Ok(())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(i64::MAX as u128) as i64
}

fn default_pid_cap() -> u32 {
    64
}

/// Headroom reserved exclusively for finalization and control-plane jobs.
pub const FINALIZATION_RESERVE_WORKERS: i64 = 1;
pub const FINALIZATION_RESERVE_THREADS: i64 = 1;
pub const FINALIZATION_RESERVE_MEMORY_BYTES: i64 = 512 * 1024 * 1024; // 512 MiB

/// Default aging factor for weighted fair queues (1 minute = 1 rank score reduction).
pub const WFQ_DEFAULT_AGING_FACTOR_MS: f64 = 60_000.0;

/// Persistent resource lease lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseStatus {
    Reserved,
    Active,
    Heartbeated,
    Recovering,
    Reaped,
    Released,
}

impl LeaseStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            LeaseStatus::Reserved => "reserved",
            LeaseStatus::Active => "active",
            LeaseStatus::Heartbeated => "heartbeated",
            LeaseStatus::Recovering => "recovering",
            LeaseStatus::Reaped => "reaped",
            LeaseStatus::Released => "released",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "reserved" => Some(LeaseStatus::Reserved),
            "active" => Some(LeaseStatus::Active),
            "heartbeated" => Some(LeaseStatus::Heartbeated),
            "recovering" => Some(LeaseStatus::Recovering),
            "reaped" => Some(LeaseStatus::Reaped),
            "released" => Some(LeaseStatus::Released),
            _ => None,
        }
    }
}

/// Resource lease allocated for an admitted job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResourceLease {
    pub lease_id: String,
    pub project: String,
    pub task: String,
    pub job: String,
    pub generation: u64,
    pub status: LeaseStatus,
    pub backend: String,
    pub device: String,
    pub cpu_millicores: u32,
    pub library_threads: u32,
    pub memory_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accelerator_type: Option<AcceleratorType>,
    #[serde(default)]
    pub accelerator_units: u32,
    pub scratch_bytes: u64,
    pub output_bytes: u64,
    #[serde(default = "default_pid_cap")]
    pub pid_cap: u32,
    pub deadline_ms: i64,
    pub heartbeat_ms: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment_hash: Option<String>,
    pub created_ms: i64,
    pub updated_ms: i64,
}

/// Instantaneous capacity allocation across concurrent workers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct CapacityAllocation {
    pub workers: i64,
    pub cpu_threads: i64,
    pub memory_bytes: i64,
}

/// Atomically persist a new resource lease in 'reserved' status and record its initial transition.
pub fn reserve_lease(
    tx: &rusqlite::Transaction<'_>,
    lease: &ResourceLease,
    reason: Option<&str>,
) -> Result<()> {
    let now = now_ms();
    let accel_str = lease.accelerator_type.map(|a| {
        serde_json::to_string(&a)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string()
    });

    tx.execute(
        "INSERT INTO resource_leases(
            lease_id, project, task, job, generation, status, backend, device,
            cpu_millicores, library_threads, memory_bytes, accelerator_type,
            accelerator_units, scratch_bytes, output_bytes, pid_cap,
            deadline_ms, heartbeat_ms, environment_hash, created_ms, updated_ms
        ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)",
        params![
            lease.lease_id,
            lease.project,
            lease.task,
            lease.job,
            lease.generation,
            lease.status.as_str(),
            lease.backend,
            lease.device,
            lease.cpu_millicores,
            lease.library_threads,
            lease.memory_bytes as i64,
            accel_str,
            lease.accelerator_units,
            lease.scratch_bytes as i64,
            lease.output_bytes as i64,
            lease.pid_cap,
            lease.deadline_ms,
            lease.heartbeat_ms,
            lease.environment_hash,
            lease.created_ms,
            lease.updated_ms,
        ],
    )
    .map_err(|e| format!("cannot insert resource lease {}: {}", lease.lease_id, e))?;

    tx.execute(
        "INSERT INTO resource_lease_transitions(
            lease_id, project, job, from_status, to_status, reason, transitioned_ms
        ) VALUES(?1, ?2, ?3, NULL, ?4, ?5, ?6)",
        params![
            lease.lease_id,
            lease.project,
            lease.job,
            lease.status.as_str(),
            reason.unwrap_or("lease reserved"),
            now,
        ],
    )
    .map_err(|e| format!("cannot record lease transition: {}", e))?;

    let _ = tx.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, 'resource_lease.reserved', ?2)",
        params![
            lease.project,
            serde_json::to_string(lease).unwrap_or_default(),
        ],
    );

    Ok(())
}

/// Transition an existing resource lease to a new state and record the durable transition.
pub fn transition_lease(
    conn: &rusqlite::Connection,
    lease_id: &str,
    to_status: LeaseStatus,
    reason: Option<&str>,
) -> Result<()> {
    let now = now_ms();
    let (project, job, from_status_str): (String, String, String) = conn
        .query_row(
            "SELECT project, job, status FROM resource_leases WHERE lease_id=?1",
            [lease_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|e| format!("lease {} not found: {}", lease_id, e))?;

    if to_status == LeaseStatus::Heartbeated {
        conn.execute(
            "UPDATE resource_leases SET status=?2, heartbeat_ms=?3, updated_ms=?3 WHERE lease_id=?1",
            params![lease_id, to_status.as_str(), now],
        )
        .map_err(|e| format!("cannot update lease {}: {}", lease_id, e))?;
    } else {
        conn.execute(
            "UPDATE resource_leases SET status=?2, updated_ms=?3 WHERE lease_id=?1",
            params![lease_id, to_status.as_str(), now],
        )
        .map_err(|e| format!("cannot update lease {}: {}", lease_id, e))?;
    }

    conn.execute(
        "INSERT INTO resource_lease_transitions(
            lease_id, project, job, from_status, to_status, reason, transitioned_ms
        ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            lease_id,
            project,
            job,
            from_status_str,
            to_status.as_str(),
            reason.unwrap_or(""),
            now,
        ],
    )
    .map_err(|e| format!("cannot record lease transition: {}", e))?;

    let event_kind = format!("resource_lease.{}", to_status.as_str());
    let _ = conn.execute(
        "INSERT INTO events(project, kind, payload) VALUES(?1, ?2, ?3)",
        params![
            project,
            event_kind,
            serde_json::json!({
                "lease_id": lease_id,
                "job_id": job,
                "from_status": from_status_str,
                "to_status": to_status.as_str(),
                "reason": reason,
                "timestamp_ms": now,
            })
            .to_string(),
        ],
    );

    Ok(())
}

/// Update lease heartbeat.
pub fn heartbeat_lease(conn: &rusqlite::Connection, lease_id: &str) -> Result<()> {
    transition_lease(conn, lease_id, LeaseStatus::Heartbeated, Some("heartbeat"))
}

/// Release a lease when execution terminates and resources are reclaimed.
pub fn release_lease(conn: &rusqlite::Connection, lease_id: &str, reason: Option<&str>) -> Result<()> {
    transition_lease(conn, lease_id, LeaseStatus::Released, reason.or(Some("released on reap")))
}

/// Fetch a resource lease by its unique ID.
pub fn get_lease(conn: &rusqlite::Connection, lease_id: &str) -> Result<Option<ResourceLease>> {
    let mut stmt = conn
        .prepare(
            "SELECT lease_id, project, task, job, generation, status, backend, device,
                    cpu_millicores, library_threads, memory_bytes, accelerator_type,
                    accelerator_units, scratch_bytes, output_bytes, pid_cap,
                    deadline_ms, heartbeat_ms, environment_hash, created_ms, updated_ms
             FROM resource_leases WHERE lease_id=?1",
        )
        .map_err(|e| e.to_string())?;

    let row = stmt
        .query_row([lease_id], |r| {
            let status_str: String = r.get(5)?;
            let status = LeaseStatus::from_str(&status_str).unwrap_or(LeaseStatus::Reserved);
            let accel_str: Option<String> = r.get(11)?;
            let accelerator_type = accel_str.and_then(|s| match s.as_str() {
                "apple_metal" => Some(AcceleratorType::AppleMetal),
                "nvidia_cuda" => Some(AcceleratorType::NvidiaCuda),
                _ => None,
            });

            Ok(ResourceLease {
                lease_id: r.get(0)?,
                project: r.get(1)?,
                task: r.get(2)?,
                job: r.get(3)?,
                generation: r.get(4)?,
                status,
                backend: r.get(6)?,
                device: r.get(7)?,
                cpu_millicores: r.get(8)?,
                library_threads: r.get(9)?,
                memory_bytes: r.get::<_, i64>(10)? as u64,
                accelerator_type,
                accelerator_units: r.get(12)?,
                scratch_bytes: r.get::<_, i64>(13)? as u64,
                output_bytes: r.get::<_, i64>(14)? as u64,
                pid_cap: r.get(15)?,
                deadline_ms: r.get(16)?,
                heartbeat_ms: r.get(17)?,
                environment_hash: r.get(18)?,
                created_ms: r.get(19)?,
                updated_ms: r.get(20)?,
            })
        })
        .optional()
        .map_err(|e| e.to_string())?;

    Ok(row)
}

/// Fetch the active lease for a specific project job.
pub fn get_lease_for_job(conn: &rusqlite::Connection, project: &str, job: &str) -> Result<Option<ResourceLease>> {
    let lease_id: Option<String> = conn
        .query_row(
            "SELECT lease_id FROM resource_leases WHERE project=?1 AND job=?2 ORDER BY rowid DESC LIMIT 1",
            params![project, job],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;

    match lease_id {
        Some(id) => get_lease(conn, &id),
        None => Ok(None),
    }
}

/// List all active leases (reserved, active, heartbeated, recovering).
pub fn list_active_leases(conn: &rusqlite::Connection, project: Option<&str>) -> Result<Vec<ResourceLease>> {
    let mut stmt = conn
        .prepare(
            "SELECT lease_id, project, task, job, generation, status, backend, device,
                    cpu_millicores, library_threads, memory_bytes, accelerator_type,
                    accelerator_units, scratch_bytes, output_bytes, pid_cap,
                    deadline_ms, heartbeat_ms, environment_hash, created_ms, updated_ms
             FROM resource_leases
             WHERE status IN ('reserved', 'active', 'heartbeated', 'recovering')
               AND (?1 IS NULL OR project=?1)
             ORDER BY created_ms ASC",
        )
        .map_err(|e| e.to_string())?;

    let rows = stmt
        .query_map([project], |r| {
            let status_str: String = r.get(5)?;
            let status = LeaseStatus::from_str(&status_str).unwrap_or(LeaseStatus::Reserved);
            let accel_str: Option<String> = r.get(11)?;
            let accelerator_type = accel_str.and_then(|s| match s.as_str() {
                "apple_metal" => Some(AcceleratorType::AppleMetal),
                "nvidia_cuda" => Some(AcceleratorType::NvidiaCuda),
                _ => None,
            });

            Ok(ResourceLease {
                lease_id: r.get(0)?,
                project: r.get(1)?,
                task: r.get(2)?,
                job: r.get(3)?,
                generation: r.get(4)?,
                status,
                backend: r.get(6)?,
                device: r.get(7)?,
                cpu_millicores: r.get(8)?,
                library_threads: r.get(9)?,
                memory_bytes: r.get::<_, i64>(10)? as u64,
                accelerator_type,
                accelerator_units: r.get(12)?,
                scratch_bytes: r.get::<_, i64>(13)? as u64,
                output_bytes: r.get::<_, i64>(14)? as u64,
                pid_cap: r.get(15)?,
                deadline_ms: r.get(16)?,
                heartbeat_ms: r.get(17)?,
                environment_hash: r.get(18)?,
                created_ms: r.get(19)?,
                updated_ms: r.get(20)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Retrieve the durable transition history for a lease.
pub fn get_lease_transitions(conn: &rusqlite::Connection, lease_id: &str) -> Result<Vec<Value>> {
    let mut stmt = conn
        .prepare(
            "SELECT id, lease_id, project, job, from_status, to_status, reason, transitioned_ms
             FROM resource_lease_transitions WHERE lease_id=?1 ORDER BY transitioned_ms ASC, id ASC",
        )
        .map_err(|e| e.to_string())?;

    let transitions = stmt
        .query_map([lease_id], |r| {
            Ok(serde_json::json!({
                "id": r.get::<_, i64>(0)?,
                "lease_id": r.get::<_, String>(1)?,
                "project": r.get::<_, String>(2)?,
                "job": r.get::<_, String>(3)?,
                "from_status": r.get::<_, Option<String>>(4)?,
                "to_status": r.get::<_, String>(5)?,
                "reason": r.get::<_, Option<String>>(6)?,
                "transitioned_ms": r.get::<_, i64>(7)?,
            }))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(transitions)
}

/// Query active capacity allocation across held reservations.
pub fn get_active_capacity_allocation(
    conn: &rusqlite::Connection,
    project: Option<&str>,
) -> Result<CapacityAllocation> {
    let (workers, cpu_threads, memory_bytes): (i64, i64, i64) = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.workers')),0),
                    COALESCE(SUM(json_extract(capacity,'$.cpu_threads')),0),
                    COALESCE(SUM(json_extract(capacity,'$.memory_bytes')),0)
             FROM reservations WHERE held=1 AND (?1 IS NULL OR project=?1)",
            [project],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|e| e.to_string())?;

    Ok(CapacityAllocation {
        workers,
        cpu_threads,
        memory_bytes,
    })
}

/// Verify atomic capacity admission against global and project limits.
///
/// Distinguishes:
/// - Concurrency caps (`workers`) from CPU allocation (`cpu_threads`) and RAM allocation (`memory_bytes`).
/// - Preserves control-plane and finalization headroom (`FINALIZATION_RESERVE_*`).
pub fn check_capacity_admission(
    root: &Path,
    conn: &rusqlite::Connection,
    project_id: &str,
    spec: &JobSpec,
    worker_ceiling: usize,
) -> Result<()> {
    let (device_json, project_json): (String, String) = conn
        .query_row(
            "SELECT d.policy, p.policy FROM device d JOIN projects p ON p.id=?1 WHERE d.id=1",
            [project_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| e.to_string())?;

    let d: Policy = serde_json::from_str(&device_json).map_err(|e| e.to_string())?;
    let p: Policy = serde_json::from_str(&project_json).map_err(|e| e.to_string())?;

    let available = get_available_capacity(root)?;
    let is_finalize = matches!(&spec.probe, Probe::Scientific { operation, .. } if operation == "finalize");

    // 1. Worker slot concurrency ceiling
    let active_workers: i64 = conn
        .query_row(
            "SELECT count(*) FROM jobs WHERE status IN('running','terminating','recovering')",
            [],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if active_workers >= worker_ceiling as i64 {
        return Err("worker slot unavailable".into());
    }

    // 2. Global device workers capacity
    let device_workers_used: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.workers')),0) FROM reservations WHERE held=1",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let device_worker_limit = d.capacity.get("workers").copied().unwrap_or(worker_ceiling as i64);

    let has_finalization_work: bool = conn
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM projects WHERE state='finalizing'
                UNION
                SELECT 1 FROM scientific_jobs sj JOIN jobs j ON j.project=sj.project AND j.id=sj.job
                WHERE sj.operation='finalize' AND j.status='queued'
            )",
            [],
            |r| r.get(0),
        )
        .unwrap_or(false);

    let reserved_workers = if (worker_ceiling > 2 || has_finalization_work) && !is_finalize {
        FINALIZATION_RESERVE_WORKERS
    } else {
        0
    };
    if device_workers_used + 1 > device_worker_limit - reserved_workers {
        if device_workers_used + 1 <= device_worker_limit && reserved_workers > 0 {
            return Err("capacity unavailable: workers (reserved for finalization)".into());
        }
        return Err("device worker capacity unavailable".into());
    }

    // 3. Project worker concurrency cap
    let project_workers_used: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.workers')),0) FROM reservations WHERE held=1 AND project=?1",
            [project_id],
            |r| r.get(0),
        )
        .unwrap_or(0);
    if let Some(limit) = p.capacity.get("workers") {
        if project_workers_used + 1 > *limit {
            return Err("project worker capacity unavailable".into());
        }
    }

    // 4. Global CPU threads capacity
    let device_cpu_used: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.cpu_threads')),0) FROM reservations WHERE held=1",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let device_cpu_limit = d
        .capacity
        .get("cpu_threads")
        .copied()
        .unwrap_or(available.available_cpus as i64);

    let reserved_cpu = if (device_cpu_limit > FINALIZATION_RESERVE_THREADS * 2 || has_finalization_work) && !is_finalize {
        FINALIZATION_RESERVE_THREADS
    } else {
        0
    };
    let req_threads = spec.threads as i64;
    if device_cpu_used + req_threads > device_cpu_limit - reserved_cpu {
        if device_cpu_used + req_threads <= device_cpu_limit && reserved_cpu > 0 {
            return Err("capacity unavailable: cpu_threads (reserved for finalization)".into());
        }
        return Err("capacity unavailable: cpu_threads".into());
    }

    // 5. Project CPU threads capacity
    if let Some(limit) = p.capacity.get("cpu_threads") {
        let project_cpu_used: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(json_extract(capacity,'$.cpu_threads')),0) FROM reservations WHERE held=1 AND project=?1",
                [project_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if project_cpu_used + req_threads > *limit {
            return Err("project capacity unavailable: cpu_threads".into());
        }
    }

    // 6. Global Memory capacity
    let device_mem_used: i64 = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.memory_bytes')),0) FROM reservations WHERE held=1",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let device_mem_limit = d
        .capacity
        .get("memory_bytes")
        .copied()
        .unwrap_or(available.available_memory_bytes as i64);

    let reserved_mem = if device_mem_limit > FINALIZATION_RESERVE_MEMORY_BYTES * 2 && !is_finalize {
        FINALIZATION_RESERVE_MEMORY_BYTES
    } else {
        0
    };
    let req_mem = spec.resources.memory_bytes as i64;
    if device_mem_used + req_mem > device_mem_limit - reserved_mem {
        if device_mem_used + req_mem <= device_mem_limit && reserved_mem > 0 {
            return Err("capacity unavailable: memory_bytes (reserved for finalization)".into());
        }
        return Err("capacity unavailable: memory_bytes".into());
    }

    // 7. Project Memory capacity
    if let Some(limit) = p.capacity.get("memory_bytes") {
        let project_mem_used: i64 = conn
            .query_row(
                "SELECT COALESCE(SUM(json_extract(capacity,'$.memory_bytes')),0) FROM reservations WHERE held=1 AND project=?1",
                [project_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if project_mem_used + req_mem > *limit {
            return Err("project capacity unavailable: memory_bytes".into());
        }
    }

    Ok(())
}

/// Check if system is currently under resource pressure.
pub fn is_under_pressure(root: &Path) -> Result<bool> {
    if std::env::var("MLRESCURIX_PRESSURE_BACKOFF").ok().as_deref() == Some("1") {
        return Ok(true);
    }
    let (_, avail) = get_disk_capacity(root)?;
    if avail < DEFAULT_DISK_HEADROOM_BYTES {
        return Ok(true);
    }
    Ok(false)
}

/// Apply thread caps to all runtime and library thread environments.
pub fn apply_thread_caps_to_command(cmd: &mut std::process::Command, threads: u32) {
    let t = threads.to_string();
    cmd.env("MLRESCURIX_NUM_THREADS", &t);
    cmd.env("OMP_NUM_THREADS", &t);
    cmd.env("OPENBLAS_NUM_THREADS", &t);
    cmd.env("MKL_NUM_THREADS", &t);
    cmd.env("VECLIB_MAXIMUM_THREADS", &t);
    cmd.env("NUMEXPR_NUM_THREADS", &t);
    cmd.env("POLARS_MAX_THREADS", &t);
}

/// Record a point-in-time capacity allocation and queue wait snapshot.
pub fn record_allocation_snapshot(conn: &rusqlite::Connection) -> Result<Value> {
    let now = now_ms();
    let (active_jobs, running_jobs): (i64, i64) = conn
        .query_row(
            "SELECT (SELECT count(*) FROM jobs WHERE status IN('running','terminating','recovering')),
                    (SELECT count(*) FROM jobs WHERE status='running')",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(|e| e.to_string())?;

    let (allocated_workers, allocated_cpus, allocated_mem): (i64, i64, i64) = conn
        .query_row(
            "SELECT COALESCE(SUM(json_extract(capacity,'$.workers')),0),
                    COALESCE(SUM(json_extract(capacity,'$.cpu_threads')),0),
                    COALESCE(SUM(json_extract(capacity,'$.memory_bytes')),0)
             FROM reservations WHERE held=1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|e| e.to_string())?;

    let queued_count: i64 = conn
        .query_row(
            "SELECT count(*) FROM jobs WHERE status='queued'",
            [],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;

    let mut waits_stmt = conn
        .prepare(
            "SELECT id, project, (?1 - submitted_ms) as wait_ms
             FROM jobs WHERE status='queued' ORDER BY submitted_ms ASC",
        )
        .map_err(|e| e.to_string())?;

    let queued_waits: Vec<Value> = waits_stmt
        .query_map([now], |r| {
            Ok(serde_json::json!({
                "job_id": r.get::<_, String>(0)?,
                "project": r.get::<_, String>(1)?,
                "wait_ms": r.get::<_, i64>(2)?,
            }))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(serde_json::json!({
        "timestamp_ms": now,
        "active_jobs": active_jobs,
        "running_jobs": running_jobs,
        "queued_jobs": queued_count,
        "allocated_workers": allocated_workers,
        "allocated_cpus": allocated_cpus,
        "allocated_memory_bytes": allocated_mem,
        "queued_waits": queued_waits,
    }))
}
