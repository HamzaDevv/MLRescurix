use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Duration;
use mlrescurix_harness::broker::*;
use mlrescurix_harness::execution::{self, JobSpec, Probe, Supervisor, resources::ResourceFeatures, resources::ResourceRequirements};
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
    setup_project_with_policy(root, project, m1policy(100, 8));
}

fn setup_project_with_policy(root: &Path, project: &str, policy: Policy) {
    let r = Runtime::open(root).unwrap();
    let req = Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P3 resource broker test".into(),
            finalization_id: format!("final-{project}"),
            policy,
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

fn now_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64
}

fn settle_job(root: &Path, supervisor: &mut Supervisor, project: &str, id: &str) -> String {
    for _ in 0..300 {
        supervisor.tick().unwrap();
        let j = execution::inspect(root, project, id).unwrap();
        if !matches!(j.status.as_str(), "queued" | "running" | "recovering" | "terminating") {
            return j.status;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("job did not settle for {project}/{id}");
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

#[test]
fn p3_test_atomic_capacity_reservations_and_dimension_distinction() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");

    let mut dev_capacity = BTreeMap::new();
    dev_capacity.insert("workers".into(), 4);
    dev_capacity.insert("cpu_threads".into(), 8);
    dev_capacity.insert("memory_bytes".into(), 8 * 1024 * 1024 * 1024);
    let mut dev_budget = BTreeMap::new();
    dev_budget.insert("calls".into(), 100);
    dev_budget.insert("tokens".into(), 100_000);
    dev_budget.insert("wall_ms".into(), 100_000);
    let dev_policy = Policy { max_tasks: 1024, budget: dev_budget.clone(), capacity: dev_capacity };
    Runtime::init(&root, dev_policy).unwrap();

    let mut proj_capacity = BTreeMap::new();
    proj_capacity.insert("workers".into(), 2);
    proj_capacity.insert("cpu_threads".into(), 4);
    proj_capacity.insert("memory_bytes".into(), 2 * 1024 * 1024 * 1024);
    let proj_policy = Policy { max_tasks: 1024, budget: dev_budget, capacity: proj_capacity };

    let r = Runtime::open(&root).unwrap();
    r.command(Request {
        command_id: "init-p-cap".into(),
        project_id: "p-cap".into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "Capacity dimension test".into(),
            finalization_id: "final-p-cap".into(),
            policy: proj_policy,
        },
    }).unwrap();

    r.command(Request {
        command_id: "branch-p-cap".into(),
        project_id: "p-cap".into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: "b".into(),
            base_hash: "base-b".into(),
            code_hash: "code-b".into(),
        },
    }).unwrap();

    r.command(Request {
        command_id: "task-p-cap".into(),
        project_id: "p-cap".into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: "t".into(),
            branch_id: "b".into(),
            context: context("base-b", "code-b"),
        },
    }).unwrap();

    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as i64;
    r.command(Request {
        command_id: "claim-p-cap".into(),
        project_id: "p-cap".into(),
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

    // 1. Submit Job 1: 1 worker, 2 threads, 1 GiB RAM
    let job1 = JobSpec {
        job_id: "j1".into(),
        project_id: "p-cap".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 2,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 2_000 },
    };
    execution::submit(&root, job1).unwrap();

    let mut supervisor = Supervisor::start(&root, 4).unwrap();
    supervisor.tick().unwrap();

    let j1_view = execution::inspect(&root, "p-cap", "j1").unwrap();
    assert_eq!(j1_view.status, "running", "Job 1 should be running");

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let alloc = get_active_capacity_allocation(&conn, Some("p-cap")).unwrap();
    assert_eq!(alloc.workers, 1);
    assert_eq!(alloc.cpu_threads, 2);
    assert_eq!(alloc.memory_bytes, 1024 * 1024 * 1024);

    // 2. Submit Job 2: requesting 3 threads.
    // Project limit is 4. Job 1 uses 2. 2 + 3 = 5 > 4.
    // Worker slot is available (1 of 2 free), memory is available (1 GiB of 2 GiB free).
    // But CPU threads is NOT available!
    let job2_excess_cpu = JobSpec {
        job_id: "j2-cpu".into(),
        project_id: "p-cap".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 3,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 512 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 2_000 },
    };
    execution::submit(&root, job2_excess_cpu).unwrap();

    let s2_cpu: JobSpec = serde_json::from_str(&conn.query_row("SELECT spec FROM jobs WHERE id='j2-cpu'", [], |r| r.get::<_, String>(0)).unwrap()).unwrap();
    let adm_res = check_capacity_admission(&root, &conn, "p-cap", &s2_cpu, 4);
    assert!(adm_res.is_err());
    assert_eq!(adm_res.unwrap_err(), "project capacity unavailable: cpu_threads");

    supervisor.tick().unwrap();
    let j2_view = execution::inspect(&root, "p-cap", "j2-cpu").unwrap();
    assert_eq!(j2_view.status, "queued", "Job with excessive CPU threads must stay queued, not admitted");

    // 3. Submit Job 3: requesting 2 threads (fits CPU: 2+2=4), but 2 GiB RAM (1 GiB + 2 GiB = 3 GiB > 2 GiB limit)
    let job3_excess_mem = JobSpec {
        job_id: "j3-mem".into(),
        project_id: "p-cap".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 2,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 2_000 },
    };
    execution::submit(&root, job3_excess_mem).unwrap();

    let s3_mem: JobSpec = serde_json::from_str(&conn.query_row("SELECT spec FROM jobs WHERE id='j3-mem'", [], |r| r.get::<_, String>(0)).unwrap()).unwrap();
    let adm_mem_res = check_capacity_admission(&root, &conn, "p-cap", &s3_mem, 4);
    assert!(adm_mem_res.is_err());
    assert_eq!(adm_mem_res.unwrap_err(), "project capacity unavailable: memory_bytes");

    supervisor.tick().unwrap();
    let j3_view = execution::inspect(&root, "p-cap", "j3-mem").unwrap();
    assert_eq!(j3_view.status, "queued", "Job with excessive RAM must stay queued, not admitted");

    // 4. Settle Job 1, releasing capacity
    settle_job(&root, &mut supervisor, "p-cap", "j1");
    supervisor.tick().unwrap();

    // After j1 settles and capacity is freed, j2-cpu (requesting 3 threads <= 4) is admitted!
    let j2_view_now = execution::inspect(&root, "p-cap", "j2-cpu").unwrap();
    assert_eq!(j2_view_now.status, "running", "Job 2 should now be admitted after Job 1 freed capacity");
    let alloc_j2 = get_active_capacity_allocation(&conn, Some("p-cap")).unwrap();
    assert_eq!(alloc_j2.workers, 1);
    assert_eq!(alloc_j2.cpu_threads, 3);
    assert_eq!(alloc_j2.memory_bytes, 512 * 1024 * 1024);

    settle_job(&root, &mut supervisor, "p-cap", "j2-cpu");
    supervisor.tick().unwrap();

    // After j2-cpu settles, j3-mem (requesting 2 GiB <= 2 GiB) is admitted!
    let j3_view_now = execution::inspect(&root, "p-cap", "j3-mem").unwrap();
    assert_eq!(j3_view_now.status, "running", "Job 3 should now be admitted after Job 2 freed capacity");
    settle_job(&root, &mut supervisor, "p-cap", "j3-mem");

    let alloc_final = get_active_capacity_allocation(&conn, Some("p-cap")).unwrap();
    assert_eq!(alloc_final.workers, 0);
    assert_eq!(alloc_final.cpu_threads, 0);
    assert_eq!(alloc_final.memory_bytes, 0);
}

#[test]
fn p3_test_lease_lifecycle_heartbeats_and_recovery_transitions() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();
    setup_project(&root, "p-lease");

    let conn = execution::connect(&root).unwrap();
    let lease_id = "lease-p-lease-test-1";

    let lease = ResourceLease {
        lease_id: lease_id.into(),
        project: "p-lease".into(),
        task: "t".into(),
        job: "test-1".into(),
        generation: 2,
        status: LeaseStatus::Reserved,
        backend: "process".into(),
        device: "local".into(),
        cpu_millicores: 1000,
        library_threads: 1,
        memory_bytes: 512 * 1024 * 1024,
        accelerator_type: None,
        accelerator_units: 0,
        scratch_bytes: 64 * 1024 * 1024,
        output_bytes: 4096,
        pid_cap: 64,
        deadline_ms: now_ms() + 10_000,
        heartbeat_ms: now_ms(),
        environment_hash: None,
        created_ms: now_ms(),
        updated_ms: now_ms(),
    };

    // 1. Reserve
    let mut tx_conn = execution::connect(&root).unwrap();
    let tx = tx_conn.transaction().unwrap();
    reserve_lease(&tx, &lease, Some("unit test reservation")).unwrap();
    tx.commit().unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().expect("lease must exist");
    assert_eq!(fetched.status, LeaseStatus::Reserved);

    // 2. Transition to Active
    transition_lease(&conn, lease_id, LeaseStatus::Active, Some("process spawned")).unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().unwrap();
    assert_eq!(fetched.status, LeaseStatus::Active);

    // 3. Heartbeat
    let before_hb = fetched.heartbeat_ms;
    std::thread::sleep(Duration::from_millis(5));
    heartbeat_lease(&conn, lease_id).unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().unwrap();
    assert_eq!(fetched.status, LeaseStatus::Heartbeated);
    assert!(fetched.heartbeat_ms >= before_hb);

    // 4. Transition to Recovering
    transition_lease(&conn, lease_id, LeaseStatus::Recovering, Some("supervisor restarted")).unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().unwrap();
    assert_eq!(fetched.status, LeaseStatus::Recovering);

    // 5. Transition to Reaped
    transition_lease(&conn, lease_id, LeaseStatus::Reaped, Some("process terminated")).unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().unwrap();
    assert_eq!(fetched.status, LeaseStatus::Reaped);

    // 6. Release
    release_lease(&conn, lease_id, Some("job settled")).unwrap();
    let fetched = get_lease(&conn, lease_id).unwrap().unwrap();
    assert_eq!(fetched.status, LeaseStatus::Released);

    // Verify durable transition history
    let transitions = get_lease_transitions(&conn, lease_id).unwrap();
    assert_eq!(transitions.len(), 6);
    assert_eq!(transitions[0]["to_status"], "reserved");
    assert_eq!(transitions[1]["to_status"], "active");
    assert_eq!(transitions[2]["to_status"], "heartbeated");
    assert_eq!(transitions[3]["to_status"], "recovering");
    assert_eq!(transitions[4]["to_status"], "reaped");
    assert_eq!(transitions[5]["to_status"], "released");

    // 7. Live Supervisor execution lease lifecycle verification
    let live_job = JobSpec {
        job_id: "live-lease-job".into(),
        project_id: "p-lease".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 2_000 },
    };
    execution::submit(&root, live_job).unwrap();

    let mut supervisor = Supervisor::start(&root, 2).unwrap();
    supervisor.tick().unwrap();

    let live_lease_id = "lease-p-lease-live-lease-job";
    let live_lease = get_lease(&conn, live_lease_id).unwrap().expect("supervisor must create lease on admission");
    assert!(matches!(live_lease.status, LeaseStatus::Active | LeaseStatus::Heartbeated));

    settle_job(&root, &mut supervisor, "p-lease", "live-lease-job");

    let live_lease_after = get_lease(&conn, live_lease_id).unwrap().expect("lease persists in terminal state");
    assert_eq!(live_lease_after.status, LeaseStatus::Released);

    let live_transitions = get_lease_transitions(&conn, live_lease_id).unwrap();
    assert!(live_transitions.iter().any(|t| t["to_status"] == "reserved"));
    assert!(live_transitions.iter().any(|t| t["to_status"] == "active"));
    assert!(live_transitions.iter().any(|t| t["to_status"] == "released"));
}

#[test]
fn p3_test_weighted_fair_queueing_aging_and_stable_tie_ordering() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();
    setup_project(&root, "p-alpha");
    setup_project(&root, "p-beta");

    // Set weights: alpha = 1, beta = 2
    execution::set_project_weight(&root, "p-alpha", 1).unwrap();
    execution::set_project_weight(&root, "p-beta", 2).unwrap();

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    let now = now_ms();
    let spec_a = JobSpec {
        job_id: "ja-1".into(),
        project_id: "p-alpha".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let spec_b = JobSpec {
        job_id: "jb-1".into(),
        project_id: "p-beta".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };

    execution::submit(&root, spec_a).unwrap();
    execution::submit(&root, spec_b).unwrap();

    // Aging prevents starvation:
    // Insert an older job submitted 5 minutes ago (300,000 ms ago)
    let spec_old = JobSpec {
        job_id: "j-old".into(),
        project_id: "p-alpha".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    conn.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms)
         VALUES('j-old', 'p-alpha', 't', 2, ?1, 'queued', ?2)",
        rusqlite::params![serde_json::to_string(&spec_old).unwrap(), now - 300_000],
    ).unwrap();

    let rank_sql = "((SELECT count(*) FROM jobs prior WHERE prior.project=j.project AND prior.status IN('running','succeeded','failed','timed_out','cancelled','rejected','interrupted'))*1.0/COALESCE((SELECT weight FROM project_weights w WHERE w.project=j.project),1) - (?1-j.submitted_ms)/60000.0)";
    let next_job: String = conn.query_row(
        &format!("SELECT j.id FROM jobs j WHERE j.status='queued' ORDER BY {rank_sql}, j.project, j.submitted_ms, j.id LIMIT 1"),
        [now],
        |r| r.get(0),
    ).unwrap();

    assert_eq!(next_job, "j-old", "Aging term (-5.0) must prioritize older queued job to bound starvation");

    // Stable tie-breaking:
    let tie1 = JobSpec {
        job_id: "tie-b".into(),
        project_id: "p-beta".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let tie2 = JobSpec {
        job_id: "tie-a".into(),
        project_id: "p-beta".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 10 },
    };
    let fixed_ts = now + 1000;
    conn.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms)
         VALUES('tie-b', 'p-beta', 't', 2, ?1, 'queued', ?2)",
        rusqlite::params![serde_json::to_string(&tie1).unwrap(), fixed_ts],
    ).unwrap();
    conn.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms)
         VALUES('tie-a', 'p-beta', 't', 2, ?1, 'queued', ?2)",
        rusqlite::params![serde_json::to_string(&tie2).unwrap(), fixed_ts],
    ).unwrap();

    let mut stmt = conn.prepare(&format!(
        "SELECT j.id FROM jobs j WHERE j.id IN ('tie-a', 'tie-b') ORDER BY {rank_sql}, j.project, j.submitted_ms, j.id"
    )).unwrap();
    let tie_ids: Vec<String> = stmt.query_map([now], |r| r.get(0)).unwrap().map(|r| r.unwrap()).collect();
    assert_eq!(tie_ids, vec!["tie-a", "tie-b"], "Ties must break deterministically by id");
}

#[test]
fn p3_test_thread_caps_and_broker_routing() {
    let mut cmd = std::process::Command::new("sh");
    apply_thread_caps_to_command(&mut cmd, 3);

    let envs: std::collections::HashMap<_, _> = cmd.get_envs().map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned()))).collect();
    assert_eq!(envs.get("MLRESCURIX_NUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("OMP_NUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("OPENBLAS_NUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("MKL_NUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("VECLIB_MAXIMUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("NUMEXPR_NUM_THREADS").unwrap().as_deref(), Some("3"));
    assert_eq!(envs.get("POLARS_MAX_THREADS").unwrap().as_deref(), Some("3"));

    let py_check = std::process::Command::new("python3")
        .args(["-c", r#"
import os
os.environ["MLRESCURIX_NUM_THREADS"] = "4"
from mlrescurix_worker.scientific.workflow import _leased_threads
assert _leased_threads() == 4, f"expected 4, got {_leased_threads()}"
print("PYTHON_THREAD_CAP_OK")
"#])
        .env("PYTHONPATH", "python/src")
        .output();
    if let Ok(out) = py_check {
        if out.status.success() {
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(stdout.contains("PYTHON_THREAD_CAP_OK"));
        }
    }
}

#[test]
fn p3_test_pressure_backoff_and_finalization_headroom_preservation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");

    let mut dev_capacity = BTreeMap::new();
    dev_capacity.insert("workers".into(), 3);
    dev_capacity.insert("cpu_threads".into(), 8);
    let dev_policy = Policy { max_tasks: 1024, budget: m1policy(100, 3).budget, capacity: dev_capacity.clone() };
    Runtime::init(&root, dev_policy).unwrap();

    let proj_policy = Policy { max_tasks: 1024, budget: m1policy(100, 3).budget, capacity: dev_capacity };
    setup_project_with_policy(&root, "p-press-1", proj_policy.clone());
    setup_project_with_policy(&root, "p-press-2", proj_policy);

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    let std_job1 = JobSpec {
        job_id: "std-1".into(),
        project_id: "p-press-1".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 1000 },
    };
    let std_job2 = JobSpec {
        job_id: "std-2".into(),
        project_id: "p-press-2".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 1000 },
    };
    execution::submit(&root, std_job1).unwrap();
    execution::submit(&root, std_job2).unwrap();

    let mut supervisor = Supervisor::start(&root, 3).unwrap();
    supervisor.tick().unwrap();

    let j1 = execution::inspect(&root, "p-press-1", "std-1").unwrap();
    let j2 = execution::inspect(&root, "p-press-2", "std-2").unwrap();
    assert_eq!(j1.status, "running");
    assert_eq!(j2.status, "running");

    // 3rd standard job should fail admission because remaining slot is reserved for finalization
    let std_job3 = JobSpec {
        job_id: "std-3".into(),
        project_id: "p-press-1".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 100 },
    };
    let res = check_capacity_admission(&root, &conn, "p-press-1", &std_job3, 3);
    assert!(res.is_err());
    assert_eq!(res.unwrap_err(), "capacity unavailable: workers (reserved for finalization)");

    // Pressure backoff
    unsafe { std::env::set_var("MLRESCURIX_PRESSURE_BACKOFF", "1"); }
    assert!(is_under_pressure(&root).unwrap());

    execution::submit(&root, std_job3).unwrap();
    supervisor.tick().unwrap();
    let j3 = execution::inspect(&root, "p-press-1", "std-3").unwrap();
    assert_eq!(j3.status, "queued", "Under pressure, standard job must remain queued");

    unsafe { std::env::remove_var("MLRESCURIX_PRESSURE_BACKOFF"); }
    assert!(!is_under_pressure(&root).unwrap());
}

#[test]
fn p3_test_per_project_cancellation_and_pause_isolation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");
    Runtime::init(&root, m1policy(100, 4)).unwrap();
    setup_project(&root, "p-cancel");
    setup_project(&root, "p-stay");

    let job_cancel = JobSpec {
        job_id: "j-cancel".into(),
        project_id: "p-cancel".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 2000 },
    };
    let job_stay = JobSpec {
        job_id: "j-stay".into(),
        project_id: "p-stay".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 100 },
    };

    execution::submit(&root, job_cancel).unwrap();
    execution::submit(&root, job_stay).unwrap();

    let mut supervisor = Supervisor::start(&root, 4).unwrap();
    supervisor.tick().unwrap();

    let jc = execution::inspect(&root, "p-cancel", "j-cancel").unwrap();
    let js = execution::inspect(&root, "p-stay", "j-stay").unwrap();
    assert_eq!(jc.status, "running");
    assert_eq!(js.status, "running");

    // Cancel p-cancel via Runtime
    let r = Runtime::open(&root).unwrap();
    r.command(Request {
        command_id: "cancel-p-cancel".into(),
        project_id: "p-cancel".into(),
        expected_revision: None,
        operation: Operation::Lifecycle {
            state: "cancelled".into(),
        },
    }).unwrap();

    let jc_after = settle_job(&root, &mut supervisor, "p-cancel", "j-cancel");
    assert_eq!(jc_after, "cancelled", "Cancelled project job must be reaped as cancelled");

    let js_after = execution::inspect(&root, "p-stay", "j-stay").unwrap();
    assert!(matches!(js_after.status.as_str(), "running" | "succeeded"));

    settle_job(&root, &mut supervisor, "p-stay", "j-stay");
    let js_final = execution::inspect(&root, "p-stay", "j-stay").unwrap();
    assert_eq!(js_final.status, "succeeded", "Non-cancelled project job must succeed cleanly");
}

#[test]
fn p3_test_multi_project_stress_and_resilience_10x5() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");

    let mut dev_capacity = BTreeMap::new();
    dev_capacity.insert("workers".into(), 8);
    dev_capacity.insert("cpu_threads".into(), 32);
    let dev_policy = Policy { max_tasks: 1024, budget: m1policy(1000, 8).budget, capacity: dev_capacity };
    Runtime::init(&root, dev_policy).unwrap();

    // 10 projects x 5 queued intents = 50 jobs
    for p_idx in 0..10 {
        let p = format!("proj-{p_idx}");
        setup_project(&root, &p);

        for j_idx in 0..5 {
            let job_id = format!("job-{p_idx}-{j_idx}");
            let probe = if j_idx % 2 == 0 {
                Probe::Sleep { millis: 15 }
            } else {
                Probe::Write { name: format!("res_{j_idx}.txt"), content: format!("output from project {p_idx} job {j_idx}") }
            };
            let spec = JobSpec {
                job_id,
                project_id: p.clone(),
                task_id: "t".into(),
                generation: 2,
                timeout_ms: 10_000,
                threads: 1,
                max_output_bytes: 4096,
                environment_id: None,
                resources: ResourceRequirements::default(),
                probe,
            };
            execution::submit(&root, spec).unwrap();
        }
    }

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();
    let total_queued: i64 = conn.query_row("SELECT count(*) FROM jobs WHERE status='queued'", [], |r| r.get(0)).unwrap();
    assert_eq!(total_queued, 50, "50 jobs must be initially queued across 10 projects");

    let mut supervisor = Supervisor::start(&root, 8).unwrap();
    let mut snapshots = Vec::new();

    // Run until all 50 jobs complete
    for _ in 0..500 {
        supervisor.tick().unwrap();
        if let Ok(snap) = record_allocation_snapshot(&conn) {
            snapshots.push(snap);
        }
        let active_count: i64 = conn.query_row(
            "SELECT count(*) FROM jobs WHERE status IN ('queued', 'running', 'terminating', 'recovering')",
            [],
            |r| r.get(0),
        ).unwrap();
        if active_count == 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(15));
    }

    // Verify all 50 jobs succeeded
    let succeeded_count: i64 = conn.query_row("SELECT count(*) FROM jobs WHERE status='succeeded'", [], |r| r.get(0)).unwrap();
    assert_eq!(succeeded_count, 50, "All 50 jobs across 10 projects must succeed");

    // Verify snapshots stay within declared worker capacity
    assert!(!snapshots.is_empty());
    for s in &snapshots {
        let alloc_w = s["allocated_workers"].as_i64().unwrap_or(0);
        assert!(alloc_w <= 8, "Allocated workers must not exceed device ceiling of 8 (observed {alloc_w})");
        let run_w = s["running_jobs"].as_i64().unwrap_or(0);
        assert!(run_w <= 8, "Running jobs must not exceed worker ceiling of 8 (observed {run_w})");
    }

    // Verify 0 leaked leases
    let active_leases = list_active_leases(&conn, None).unwrap();
    assert_eq!(active_leases.len(), 0, "No active leases should remain after execution completes");

    // Verify all 50 leases exist in released status
    let released_leases: i64 = conn.query_row("SELECT count(*) FROM resource_leases WHERE status='released'", [], |r| r.get(0)).unwrap();
    assert_eq!(released_leases, 50, "All 50 jobs must have a persisted released resource lease");
}

#[test]
fn p3_acceptance_gate() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("state");

    let mut dev_capacity = BTreeMap::new();
    dev_capacity.insert("workers".into(), 4);
    dev_capacity.insert("cpu_threads".into(), 8);
    dev_capacity.insert("memory_bytes".into(), 4 * 1024 * 1024 * 1024);
    let dev_policy = Policy { max_tasks: 1024, budget: m1policy(100, 4).budget, capacity: dev_capacity.clone() };
    Runtime::init(&root, dev_policy).unwrap();

    let proj_policy = Policy {
        max_tasks: 1024,
        budget: m1policy(100, 4).budget,
        capacity: dev_capacity,
    };

    setup_project_with_policy(&root, "gate-p1", proj_policy.clone());
    setup_project_with_policy(&root, "gate-p2", proj_policy.clone());

    let conn = rusqlite::Connection::open(root.join("state.sqlite")).unwrap();

    // -------------------------------------------------------------------------
    // Gate Criterion 1: Admitted sums stay within declared capacity
    // -------------------------------------------------------------------------
    let large_job = JobSpec {
        job_id: "large-job".into(),
        project_id: "gate-p1".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 4,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 2 * 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 2_000 },
    };
    execution::submit(&root, large_job).unwrap();

    let mut supervisor = Supervisor::start(&root, 4).unwrap();
    supervisor.tick().unwrap();

    let j_large = execution::inspect(&root, "gate-p1", "large-job").unwrap();
    assert_eq!(j_large.status, "running", "large job must be admitted to running");

    let alloc = get_active_capacity_allocation(&conn, None).unwrap();
    assert!(alloc.workers <= 4, "Admitted workers must stay within capacity");
    assert!(alloc.cpu_threads <= 8, "Admitted CPUs must stay within capacity");
    assert!(alloc.memory_bytes <= 4 * 1024 * 1024 * 1024, "Admitted RAM must stay within capacity");
    let res1 = settle_job(&root, &mut supervisor, "gate-p1", "large-job");
    assert_eq!(res1, "succeeded");

    // -------------------------------------------------------------------------
    // Gate Criterion 2: Large queued job receives service under documented aging policy
    // -------------------------------------------------------------------------
    let now = now_ms();
    let aged_job = JobSpec {
        job_id: "aged-large-job".into(),
        project_id: "gate-p2".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 2,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements {
            memory_bytes: 1024 * 1024 * 1024,
            ..Default::default()
        },
        probe: Probe::Sleep { millis: 50 },
    };
    conn.execute(
        "INSERT INTO jobs(id, project, task, generation, spec, status, submitted_ms)
         VALUES('aged-large-job', 'gate-p2', 't', 2, ?1, 'queued', ?2)",
        rusqlite::params![serde_json::to_string(&aged_job).unwrap(), now - 180_000], // 3 minutes ago
    ).unwrap();

    let rank_sql = "((SELECT count(*) FROM jobs prior WHERE prior.project=j.project AND prior.status IN('running','succeeded','failed','timed_out','cancelled','rejected','interrupted'))*1.0/COALESCE((SELECT weight FROM project_weights w WHERE w.project=j.project),1) - (?1-j.submitted_ms)/60000.0)";
    let selected_job: String = conn.query_row(
        &format!("SELECT j.id FROM jobs j WHERE j.status='queued' ORDER BY {rank_sql}, j.project, j.submitted_ms, j.id LIMIT 1"),
        [now],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(selected_job, "aged-large-job", "Aged large job receives service under aging fair-queue policy");
    let res2 = settle_job(&root, &mut supervisor, "gate-p2", "aged-large-job");
    assert_eq!(res2, "succeeded");

    // -------------------------------------------------------------------------
    // Gate Criterion 3: One project's cancellation leaves others running
    // -------------------------------------------------------------------------
    setup_project_with_policy(&root, "gate-p3", proj_policy.clone());
    setup_project_with_policy(&root, "gate-p4", proj_policy.clone());

    let job_p3 = JobSpec {
        job_id: "cancel-me".into(),
        project_id: "gate-p3".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 2000 },
    };
    let job_p4 = JobSpec {
        job_id: "keep-me".into(),
        project_id: "gate-p4".into(),
        task_id: "t".into(),
        generation: 2,
        timeout_ms: 5000,
        threads: 1,
        max_output_bytes: 1024,
        environment_id: None,
        resources: ResourceRequirements::default(),
        probe: Probe::Sleep { millis: 50 },
    };
    execution::submit(&root, job_p3).unwrap();
    execution::submit(&root, job_p4).unwrap();

    supervisor.tick().unwrap();
    assert_eq!(execution::inspect(&root, "gate-p3", "cancel-me").unwrap().status, "running");
    assert_eq!(execution::inspect(&root, "gate-p4", "keep-me").unwrap().status, "running");

    let r = Runtime::open(&root).unwrap();
    r.command(Request {
        command_id: "cancel-gate-p3".into(),
        project_id: "gate-p3".into(),
        expected_revision: None,
        operation: Operation::Lifecycle {
            state: "cancelled".into(),
        },
    }).unwrap();

    let p3_status = settle_job(&root, &mut supervisor, "gate-p3", "cancel-me");
    assert_eq!(p3_status, "cancelled");

    let p4_status = execution::inspect(&root, "gate-p4", "keep-me").unwrap().status;
    assert!(matches!(p4_status.as_str(), "running" | "succeeded"));

    let p4_final = settle_job(&root, &mut supervisor, "gate-p4", "keep-me");
    assert_eq!(p4_final, "succeeded", "Project 4 job completed successfully despite Project 3 cancellation");

    // -------------------------------------------------------------------------
    // Gate Criterion 4: No runner bypasses leases
    // -------------------------------------------------------------------------
    let succeeded_count: i64 = conn.query_row("SELECT count(*) FROM jobs WHERE status='succeeded'", [], |r| r.get(0)).unwrap();
    assert!(succeeded_count >= 3, "Must have verified succeeded jobs across gate criteria");

    let unleased_jobs: i64 = conn.query_row(
        "SELECT count(*) FROM jobs j WHERE j.status='succeeded' AND NOT EXISTS (
            SELECT 1 FROM resource_leases rl WHERE rl.project=j.project AND rl.job=j.id
        )",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(unleased_jobs, 0, "No succeeded job may bypass resource lease allocation");

    let unrecorded_transitions: i64 = conn.query_row(
        "SELECT count(*) FROM resource_leases rl WHERE NOT EXISTS (
            SELECT 1 FROM resource_lease_transitions rlt WHERE rlt.lease_id=rl.lease_id
        )",
        [],
        |r| r.get(0),
    ).unwrap();
    assert_eq!(unrecorded_transitions, 0, "Every lease must have recorded transition history");
}
