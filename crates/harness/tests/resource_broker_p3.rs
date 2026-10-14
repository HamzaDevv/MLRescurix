use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use mlrescurix_harness::broker::*;
use mlrescurix_harness::execution::{self, JobSpec, Probe, resources::ResourceFeatures, resources::ResourceRequirements};
use mlrescurix_harness::{Runtime, Policy, Request, Operation, contracts::Context};
use mlrescurix_harness::Result;

fn m1policy(limit: i64, workers: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), workers);
    c.insert("cpu_threads".into(), 8);
    Policy { max_tasks: 1024, budget: b, capacity: c }
}

fn context(base: &str, code: &str) -> Context {
    Context {
        role_id: "researcher".into(),
        role_version: "1".into(),
        prompt_hash: "prompt-hash".into(),
        model: None,
        phase: "research".into(),
        hypothesis: None,
        base_hash: base.into(),
        code_hash: code.into(),
        evidence: vec![],
        artifacts: vec![],
        pending_questions: vec![],
        calls: vec![],
        jobs: vec![],
        next_action: "inspect".into(),
        stop_conditions: vec!["budget".into()],
    }
}

fn setup_project(root: &Path, project: &str) {
    let r = Runtime::open(root).unwrap();
    let req = Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P3 resource broker test".into(),
            finalization_id: format!("final-{project}"),
            policy: m1policy(100, 8),
        },
    };
    r.command(req).unwrap();

    r.command(Request {
        command_id: format!("branch-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "b".into(),
            base_hash: "base-b".into(),
            code_hash: "code-b".into(),
        },
    }).unwrap();

    r.command(Request {
        command_id: format!("task-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: "t".into(),
            branch_id: "b".into(),
            context: context("base-b", "code-b"),
        },
    }).unwrap();

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    r.command(Request {
        command_id: format!("claim-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::Claim {
            task_id: "t".into(),
            owner: "lead".into(),
            lease_id: "leaset".into(),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        },
    }).unwrap();
}

#[derive(Debug, Clone)]
struct MockCapabilityProbe {
    host_cpus: u32,
    host_memory: u64,
    host_disk_total: u64,
    host_disk_avail: u64,
    host_accelerators: Vec<AcceleratorInfo>,

    docker_available: bool,
    vm_cpus: Option<u32>,
    vm_memory: Option<u64>,
    cgroup_v2: bool,
    memory_limit: bool,
    swap_limit: bool,
    cpu_quota: bool,
    pids_limit: bool,
    container_accelerators: Vec<AcceleratorInfo>,
}

impl MockCapabilityProbe {
    fn mac_m2_with_docker_vm() -> Self {
        Self {
            host_cpus: 8,
            host_memory: 16 * 1024 * 1024 * 1024, // 16 GiB
            host_disk_total: 500 * 1024 * 1024 * 1024,
            host_disk_avail: 100 * 1024 * 1024 * 1024,
            host_accelerators: vec![AcceleratorInfo {
                kind: AcceleratorType::AppleMetal,
                name: "Apple M2 GPU".into(),
                device_count: 1,
                memory_bytes: None,
            }],
            docker_available: true,
            vm_cpus: Some(4),                     // Docker VM allocated 4 CPUs
            vm_memory: Some(8 * 1024 * 1024 * 1024), // Docker VM allocated 8 GiB
            cgroup_v2: true,
            memory_limit: true,
            swap_limit: true,
            cpu_quota: true,
            pids_limit: true,
            container_accelerators: vec![],
        }
    }

    fn linux_with_cuda() -> Self {
        Self {
            host_cpus: 16,
            host_memory: 32 * 1024 * 1024 * 1024,
            host_disk_total: 1000 * 1024 * 1024 * 1024,
            host_disk_avail: 200 * 1024 * 1024 * 1024,
            host_accelerators: vec![AcceleratorInfo {
                kind: AcceleratorType::NvidiaCuda,
                name: "NVIDIA RTX 4090".into(),
                device_count: 1,
                memory_bytes: Some(24 * 1024 * 1024 * 1024),
            }],
            docker_available: true,
            vm_cpus: Some(16),
            vm_memory: Some(32 * 1024 * 1024 * 1024),
            cgroup_v2: true,
            memory_limit: true,
            swap_limit: true,
            cpu_quota: true,
            pids_limit: true,
            container_accelerators: vec![AcceleratorInfo {
                kind: AcceleratorType::NvidiaCuda,
                name: "NVIDIA RTX 4090".into(),
                device_count: 1,
                memory_bytes: Some(24 * 1024 * 1024 * 1024),
            }],
        }
    }

    fn unsupported_docker() -> Self {
        Self {
            host_cpus: 4,
            host_memory: 8 * 1024 * 1024 * 1024,
            host_disk_total: 100 * 1024 * 1024 * 1024,
            host_disk_avail: 20 * 1024 * 1024 * 1024,
            host_accelerators: vec![],
            docker_available: true,
            vm_cpus: Some(4),
            vm_memory: Some(8 * 1024 * 1024 * 1024),
            cgroup_v2: false, // Legacy cgroup v1 or missing limits
            memory_limit: true,
            swap_limit: false,
            cpu_quota: true,
            pids_limit: false,
            container_accelerators: vec![],
        }
    }
}

impl CapabilityProbe for MockCapabilityProbe {
    fn probe_host(&self, _root: &Path) -> Result<HostCapabilities> {
        Ok(HostCapabilities {
            logical_cpus: self.host_cpus,
            total_memory_bytes: self.host_memory,
            disk_total_bytes: self.host_disk_total,
            disk_available_bytes: self.host_disk_avail,
            accelerators: self.host_accelerators.clone(),
        })
    }

    fn probe_container_vm(&self) -> Result<ContainerVmCapabilities> {
        Ok(ContainerVmCapabilities {
            docker_available: self.docker_available,
            vm_cpus: self.vm_cpus,
            vm_memory_bytes: self.vm_memory,
            cgroup_version: if self.cgroup_v2 { Some("2".into()) } else { Some("1".into()) },
            features: ResourceFeatures {
                available: self.docker_available,
                cgroup_v2: self.cgroup_v2,
                memory_limit: self.memory_limit,
                swap_limit: self.swap_limit,
                cpu_quota: self.cpu_quota,
                pids_limit: self.pids_limit,
                reason: None,
            },
            accelerators: self.container_accelerators.clone(),
        })
    }
}

#[test]
fn p3_test_host_and_container_vm_capability_discovery() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();

    // 1. Live system discovery test
    let live_eff = discover_effective_capabilities(&root).unwrap();
    assert!(live_eff.host.logical_cpus >= 1);
    assert!(live_eff.host.total_memory_bytes >= MIN_MEMORY_BYTES);
    assert!(live_eff.host.disk_available_bytes > 0);
    assert!(live_eff.effective_cpus >= 1);
    assert!(live_eff.effective_memory_bytes >= MIN_MEMORY_BYTES);

    #[cfg(target_os = "macos")]
    {
        // On macOS Apple Silicon, AppleMetal should be discovered
        if !live_eff.host.accelerators.is_empty() {
            assert_eq!(live_eff.host.accelerators[0].kind, AcceleratorType::AppleMetal);
        }
    }

    // 2. Synthetic probe test: Host 16 GiB / 8 CPU, Container VM 8 GiB / 4 CPU
    let probe = MockCapabilityProbe::mac_m2_with_docker_vm();
    let eff = discover_effective_capabilities_with_probe(&root, &probe).unwrap();

    // Verify container VM limits cap effective capacity for container execution
    assert_eq!(eff.host.logical_cpus, 8);
    assert_eq!(eff.host.total_memory_bytes, 16 * 1024 * 1024 * 1024);
    assert_eq!(eff.container_vm.vm_cpus, Some(4));
    assert_eq!(eff.container_vm.vm_memory_bytes, Some(8 * 1024 * 1024 * 1024));

    // Effective capacity must be bounded by container VM
    assert_eq!(eff.effective_cpus, 4, "Effective CPU must be min(host_cpus, vm_cpus)");
    assert_eq!(eff.effective_memory_bytes, 8 * 1024 * 1024 * 1024, "Effective RAM must be min(host_ram, vm_ram)");
    assert!(eff.memory_certification_supported);
    assert_eq!(eff.effective_accelerators.len(), 1);
    assert_eq!(eff.effective_accelerators[0].kind, AcceleratorType::AppleMetal);
}

#[test]
fn p3_test_resource_headroom_accounting() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();

    let probe = MockCapabilityProbe::mac_m2_with_docker_vm();

    // 1. Default headroom: 1000 millicores (1 core), 512 MiB memory, 64 MiB disk
    let default_headroom = ResourceHeadroom::default();
    assert_eq!(default_headroom.cpu_millicores, 1000);
    assert_eq!(default_headroom.memory_bytes, 512 * 1024 * 1024);
    assert_eq!(default_headroom.disk_bytes, 64 * 1024 * 1024);

    let avail = get_available_capacity_with_probe(&root, &probe, &default_headroom).unwrap();
    // Effective 4 CPUs - 1 CPU headroom = 3 available CPUs
    assert_eq!(avail.available_cpus, 3);
    // Effective 8 GiB - 512 MiB headroom = 7.5 GiB available
    assert_eq!(avail.available_memory_bytes, 8 * 1024 * 1024 * 1024 - 512 * 1024 * 1024);
    // Effective 100 GiB - 64 MiB headroom
    assert_eq!(avail.available_disk_bytes, 100 * 1024 * 1024 * 1024 - 64 * 1024 * 1024);

    // 2. Custom headroom loaded from resource_headroom.json
    let custom_headroom = ResourceHeadroom {
        cpu_millicores: 2000,
        memory_bytes: 2 * 1024 * 1024 * 1024, // 2 GiB
        disk_bytes: 10 * 1024 * 1024 * 1024,  // 10 GiB
    };
    fs::write(
        root.join("resource_headroom.json"),
        serde_json::to_vec(&custom_headroom).unwrap(),
    ).unwrap();

    let loaded_headroom = ResourceHeadroom::load(&root);
    assert_eq!(loaded_headroom, custom_headroom);

    let custom_avail = get_available_capacity_with_probe(&root, &probe, &loaded_headroom).unwrap();
    // 4 CPUs - 2 CPU headroom = 2 available
    assert_eq!(custom_avail.available_cpus, 2);
    // 8 GiB - 2 GiB headroom = 6 GiB available
    assert_eq!(custom_avail.available_memory_bytes, 6 * 1024 * 1024 * 1024);
    // 100 GiB - 10 GiB headroom = 90 GiB available
    assert_eq!(custom_avail.available_disk_bytes, 90 * 1024 * 1024 * 1024);
}

#[test]
fn p3_test_resource_presets_bounds_and_resolution() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();

    // 1. Test built-in presets
    let micro = PresetConfig::default_for(ResourcePreset::Micro);
    assert_eq!(micro.threads, 1);
    assert_eq!(micro.memory_bytes, 512 * 1024 * 1024);
    assert_eq!(micro.max_output_bytes, 1 * 1024 * 1024);
    assert!(micro.validate().is_ok());

    let standard = PresetConfig::default_for(ResourcePreset::Standard);
    assert_eq!(standard.threads, 2);
    assert_eq!(standard.memory_bytes, 2 * 1024 * 1024 * 1024);
    assert_eq!(standard.max_output_bytes, 2 * 1024 * 1024);
    assert!(standard.validate().is_ok());

    let heavy = PresetConfig::default_for(ResourcePreset::Heavy);
    assert_eq!(heavy.threads, 4);
    assert_eq!(heavy.memory_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(heavy.max_output_bytes, 4 * 1024 * 1024);
    assert!(heavy.validate().is_ok());

    // 2. Test invalid preset bounds
    let invalid_threads = PresetConfig {
        preset: ResourcePreset::Custom,
        threads: 0,
        memory_bytes: 1024 * 1024 * 1024,
        max_output_bytes: 1024,
    };
    assert!(invalid_threads.validate().is_err());

    let invalid_mem_low = PresetConfig {
        preset: ResourcePreset::Custom,
        threads: 1,
        memory_bytes: 32 * 1024 * 1024, // < 64 MiB
        max_output_bytes: 1024,
    };
    assert!(invalid_mem_low.validate().is_err());

    let invalid_mem_high = PresetConfig {
        preset: ResourcePreset::Custom,
        threads: 1,
        memory_bytes: 128 * 1024 * 1024 * 1024, // > 64 GiB
        max_output_bytes: 1024,
    };
    assert!(invalid_mem_high.validate().is_err());

    let invalid_output = PresetConfig {
        preset: ResourcePreset::Custom,
        threads: 1,
        memory_bytes: 1024 * 1024 * 1024,
        max_output_bytes: 10 * 1024 * 1024, // > 4 MiB M1 ceiling
    };
    assert!(invalid_output.validate().is_err());

    // 3. Test preset resolution on JobSpec
    let mut spec = JobSpec {
        job_id: "preset-job".into(),
        project_id: "p".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            preset: Some(ResourcePreset::Standard),
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 10 },
    };

    resolve_job_spec_preset(&root, &mut spec).unwrap();
    assert_eq!(spec.threads, 2, "Standard preset must resolve threads=2");
    assert_eq!(spec.resources.memory_bytes, 2 * 1024 * 1024 * 1024, "Standard preset must resolve memory=2 GiB");
    assert_eq!(spec.max_output_bytes, 2 * 1024 * 1024, "Standard preset must resolve output=2 MiB");
}

#[test]
fn p3_test_infeasible_request_early_rejection_before_admission() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();
    setup_project(&root, "p-feas");
    execution::migrate(&root).unwrap();

    let probe = MockCapabilityProbe::mac_m2_with_docker_vm();
    let headroom = ResourceHeadroom::default();

    // 1. Excessive CPU threads rejection:
    // Container job: Available CPUs is 3 (4 VM CPUs - 1 CPU headroom). Requesting 8 threads is infeasible.
    let excessive_threads = JobSpec {
        job_id: "infeas-threads".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 8,
        max_output_bytes: 1024,
        environment_id: Some("science-env".into()),
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let res = check_request_feasibility_with_probe(&root, &excessive_threads, &probe, &headroom);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(err.contains("infeasible request: requested threads 8 exceeds available CPU capacity 3"));

    // 2. Excessive Memory rejection:
    // Container job: Available RAM is 7.5 GiB (8 GiB - 512 MiB). Requesting 12 GiB is infeasible.
    let excessive_ram = JobSpec {
        job_id: "infeas-ram".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: Some("science-env".into()),
        resources: ResourceRequirements {
            memory_bytes: 12 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 10 },
    };
    let res_ram = check_request_feasibility_with_probe(&root, &excessive_ram, &probe, &headroom);
    assert!(res_ram.is_err());
    let err_ram = res_ram.unwrap_err();
    assert!(err_ram.contains("infeasible request: requested memory 12288 MiB exceeds available capacity 7680 MiB"));

    // 3. Accelerator testing:
    // Mac M2 has AppleMetal, not NvidiaCuda. Requesting NvidiaCuda must be rejected on Mac M2.
    let cuda_req = JobSpec {
        job_id: "infeas-cuda".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            accelerator: Some(AcceleratorType::NvidiaCuda),
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 10 },
    };
    let res_cuda = check_request_feasibility_with_probe(&root, &cuda_req, &probe, &headroom);
    assert!(res_cuda.is_err());
    let err_cuda = res_cuda.unwrap_err();
    assert!(err_cuda.contains("infeasible request: requested accelerator 'NvidiaCuda' is not available on this system"));

    // Linux with CUDA accepts NvidiaCuda, but rejects AppleMetal
    let linux_probe = MockCapabilityProbe::linux_with_cuda();
    let res_cuda_linux = check_request_feasibility_with_probe(&root, &cuda_req, &linux_probe, &headroom);
    assert!(res_cuda_linux.is_ok(), "Linux probe with CUDA must accept NvidiaCuda request");

    let metal_req_on_linux = JobSpec {
        job_id: "infeas-metal-linux".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            accelerator: Some(AcceleratorType::AppleMetal),
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 10 },
    };
    let res_metal_linux = check_request_feasibility_with_probe(&root, &metal_req_on_linux, &linux_probe, &headroom);
    assert!(res_metal_linux.is_err());
    assert!(res_metal_linux.unwrap_err().contains("infeasible request: requested accelerator 'AppleMetal' is not available on this system"));

    // 4. Memory certification on unsupported cgroups rejection:
    let probe_unsupported = MockCapabilityProbe::unsupported_docker();
    let cert_req = JobSpec {
        job_id: "infeas-cert".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: Some("science-env".into()),
        resources: ResourceRequirements {
            require_memory_certification: true,
            ..Default::default()
        },
        probe: Probe::Scientific {
            operation: "trial".into(),
            dataset_id: "d".into(),
            spec: Some(serde_json::json!({"recipe": "logistic"})),
            candidate_id: None,
            test_access_id: None,
        },
    };
    let res_cert = check_request_feasibility_with_probe(&root, &cert_req, &probe_unsupported, &headroom);
    assert!(res_cert.is_err());
    let err_cert = res_cert.unwrap_err();
    assert_eq!(
        err_cert,
        "required memory certification unsupported: Docker cgroup v2 memory/swap/CPU/PID enforcement unavailable"
    );

    // 5. Submit rejection before admission (0 jobs inserted into SQLite):
    // Test submitting an infeasible job via execution::submit
    let infeas_submit_job = JobSpec {
        job_id: "infeas-submit".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 128, // 128 threads > host available CPUs
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let submit_res = execution::submit(&root, infeas_submit_job);
    assert!(submit_res.is_err());
    let submit_err = submit_res.unwrap_err();
    assert!(submit_err.starts_with("infeasible request: requested threads 128 exceeds available CPU capacity"));

    // Verify 0 rows inserted into jobs table
    let db = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let job_count: i64 = db.query_row("SELECT count(*) FROM jobs", [], |r| r.get(0)).unwrap();
    assert_eq!(job_count, 0, "Infeasible request must NOT be admitted or queued in jobs table");

    // 6. Feasible job submission succeeds
    let feasible_job = JobSpec {
        job_id: "feas-submit".into(),
        project_id: "p-feas".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let view = execution::submit(&root, feasible_job).unwrap();
    assert_eq!(view.status, "queued");
    let job_count_after: i64 = db.query_row("SELECT count(*) FROM jobs", [], |r| r.get(0)).unwrap();
    assert_eq!(job_count_after, 1, "Feasible job must be admitted to queued state");
}

#[test]
fn p3_test_container_vm_memory_cap_prevents_host_overestimate() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();

    // Probe where Host has 16 GiB RAM, but Docker VM only has 8 GiB
    let probe = MockCapabilityProbe::mac_m2_with_docker_vm();
    let headroom = ResourceHeadroom::default(); // 512 MiB

    // Job requesting 10 GiB RAM:
    // Fits host RAM (10 GiB < 16 GiB - 512 MiB = 15.5 GiB)
    // BUT fails container VM RAM (10 GiB > 8 GiB - 512 MiB = 7.5 GiB)
    let container_job = JobSpec {
        job_id: "container-10g".into(),
        project_id: "p-test".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: Some("science-env".into()),
        resources: ResourceRequirements {
            memory_bytes: 10 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Scientific {
            operation: "trial".into(),
            dataset_id: "d".into(),
            spec: Some(serde_json::json!({"recipe": "logistic"})),
            candidate_id: None,
            test_access_id: None,
        },
    };

    let res = check_request_feasibility_with_probe(&root, &container_job, &probe, &headroom);
    assert!(res.is_err(), "Container job requesting 10 GiB must be rejected when Docker VM only has 8 GiB");
    let err = res.unwrap_err();
    assert!(err.contains("infeasible request: requested memory 10240 MiB exceeds available capacity 7680 MiB (effective 8192 MiB minus 512 MiB headroom)"));

    // On the other hand, a host non-container job (with probe Sleep) requesting 10 GiB
    // on a 16 GiB host has effective memory of 16 GiB:
    let host_job = JobSpec {
        job_id: "host-10g".into(),
        project_id: "p-test".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 10_000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 10 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 10 },
    };

    let host_res = check_request_feasibility_with_probe(&root, &host_job, &probe, &headroom);
    assert!(host_res.is_ok(), "Host job requesting 10 GiB is feasible on 16 GiB host (available 15.5 GiB)");
}
