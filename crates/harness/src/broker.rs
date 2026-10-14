//! Resource lease broker: capability discovery, presets, headroom, and admission feasibility.
//!
//! P3.1: Effective CPU, memory, disk, and accelerator discovery across host and container VM,
//! resource request presets, configurable headroom, and early infeasible-request rejection.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
