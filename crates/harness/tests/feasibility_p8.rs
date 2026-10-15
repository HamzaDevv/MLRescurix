//! Phase P8 Compute Feasibility & Local Alternatives Integration Tests.
//!
//! Verifies P8.1 through P8.5 and the Phase P8 Acceptance Gate:
//! - P8.1: Strict typed HardwareProfile, DatasetMetadata, FeasibilityReport, ExecutionPlan.
//! - P8.2: Analytical estimators (linear, boosting, tree, neural) and capped micro-pilots (N <= 100).
//! - P8.3: Transparent multi-tier feasibility verdicts (locally_feasible, feasible_with_adaptation, requires_external_compute).
//! - P8.4: Concrete evidence-backed local alternatives without silent semantic degradation.
//! - P8.5: Safe deterministic constraint negotiation preserving locked evaluation split protocols.
//! - Acceptance Gate: Explicit feasible/adapted/refused verdict emitted prior to heavy dispatch, micro-pilots respect tight caps.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{json, Value};

use mlrescurix_harness::{
    agents::{
        gateway::{ProviderGateway, ScriptedProviderFixture},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
    },
    broker::{CapabilityProbe, ContainerVmCapabilities, HostCapabilities},
    core::{Operation, Policy, Request},
    execution::{self, resources::ResourceFeatures},
    feasibility::{
        self, assess_feasibility, execute_micro_pilot, get_execution_plan, get_feasibility_report,
        negotiate_plan, record_execution_plan, record_feasibility_report, record_micro_pilot_run,
        suggest_alternatives, ConfidenceLevel, DatasetMetadata, FeasibilityVerdict,
        HardwareProfile, ModelFamily,
    },
    Runtime,
};

fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    b.insert("cost_millicents".into(), limit * 50);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy {
        max_tasks: 1024,
        budget: b,
        capacity: c,
    }
}

fn test_context(base: &str, code: &str) -> mlrescurix_harness::contracts::Context {
    mlrescurix_harness::contracts::Context {
        role_id: "lead".into(),
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

struct TestHarness {
    root: PathBuf,
}

impl TestHarness {
    fn new(prefix: &str) -> Self {
        let root = std::env::temp_dir().join(format!("mlrescurix_p8_{prefix}_{}", now_nanos()));
        if root.exists() {
            let _ = fs::remove_dir_all(&root);
        }
        Runtime::init(&root, test_policy(100)).unwrap();
        execution::migrate(&root).unwrap();
        Self { root }
    }

    fn create_project(&self, project: &str) {
        let r = Runtime::open(&self.root).unwrap();
        let _ = r.command(Request {
            command_id: format!("init-{project}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateProject {
                intent: "P8 Feasibility Test".into(),
                finalization_id: format!("final-{project}"),
                policy: test_policy(100),
            },
        }).unwrap();
    }

    fn setup_orchestrator(&self, project: &str, task_id: &str) -> AgentOrchestrator {
        self.create_project(project);
        let r = Runtime::open(&self.root).unwrap();
        let branch = format!("b-{task_id}");

        let _ = r.command(Request {
            command_id: format!("branch-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateBranch {
                branch_id: branch.clone(),
                base_hash: "base-p8".into(),
                code_hash: "code-p8".into(),
            },
        }).unwrap();

        let _ = r.command(Request {
            command_id: format!("task-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::CreateTask {
                task_id: task_id.into(),
                branch_id: branch.clone(),
                context: test_context("base-p8", "code-p8"),
            },
        }).unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
            .min(i64::MAX as u128) as i64;
        let _ = r.command(Request {
            command_id: format!("claim-{project}-{task_id}"),
            project_id: project.into(),
            expected_revision: None,
            operation: Operation::Claim {
                task_id: task_id.into(),
                owner: format!("agent-{task_id}"),
                lease_id: format!("lease-{task_id}"),
                now_ms: now,
                expires_ms: now + 300_000,
                expected_generation: 0,
            },
        }).unwrap();

        let gateway: Arc<dyn ProviderGateway> = Arc::new(ScriptedProviderFixture::new(vec![]));
        AgentOrchestrator::new(
            &self.root,
            OrchestratorConfig {
                project_id: project.into(),
                task_id: task_id.into(),
                role_id: "lead".into(),
                branch_id: branch,
                model: "mock-model".into(),
                max_turns: 5,
            },
            gateway,
        ).unwrap()
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

/// Hermetic mock capability probe for predictable test scenarios.
#[derive(Clone)]
struct MockFeasibilityProbe {
    host_cpus: u32,
    host_ram: u64,
    disk_avail: u64,
    docker_available: bool,
    vm_cpus: Option<u32>,
    vm_ram: Option<u64>,
}

impl CapabilityProbe for MockFeasibilityProbe {
    fn probe_host(&self, _root: &Path) -> mlrescurix_harness::Result<HostCapabilities> {
        Ok(HostCapabilities {
            logical_cpus: self.host_cpus,
            total_memory_bytes: self.host_ram,
            disk_total_bytes: 100 * 1024 * 1024 * 1024,
            disk_available_bytes: self.disk_avail,
            accelerators: vec![],
        })
    }

    fn probe_container_vm(&self) -> mlrescurix_harness::Result<ContainerVmCapabilities> {
        Ok(ContainerVmCapabilities {
            docker_available: self.docker_available,
            vm_cpus: self.vm_cpus,
            vm_memory_bytes: self.vm_ram,
            cgroup_version: Some("2".into()),
            features: ResourceFeatures {
                available: true,
                cgroup_v2: true,
                memory_limit: true,
                swap_limit: true,
                cpu_quota: true,
                pids_limit: true,
                reason: None,
            },
            accelerators: vec![],
        })
    }
}

#[test]
fn test_p8_1_hardware_profile_from_broker_discovery() {
    let harness = TestHarness::new("hw_profile");
    let probe = MockFeasibilityProbe {
        host_cpus: 8,
        host_ram: 16 * 1024 * 1024 * 1024,
        disk_avail: 40 * 1024 * 1024 * 1024,
        docker_available: true,
        vm_cpus: Some(4),
        vm_ram: Some(8 * 1024 * 1024 * 1024),
    };

    let profile = HardwareProfile::from_broker_with_probe(&harness.root, &probe).unwrap();

    assert_eq!(profile.schema_version, "v2");
    assert_eq!(profile.effective_cpus, 4); // min(8, 4)
    assert_eq!(profile.effective_memory_bytes, 8 * 1024 * 1024 * 1024); // min(16G, 8G)
    assert_eq!(profile.effective_disk_bytes, 40 * 1024 * 1024 * 1024);
    assert_eq!(profile.available.available_cpus, 3); // 4 - 1 core headroom
    assert!(profile.memory_certification_supported);
}

#[test]
fn test_p8_1_dataset_metadata_and_transformation_expansion() {
    let meta = DatasetMetadata::synthetic("ds_expansion_test", 10000, 20, 3, 50);

    assert_eq!(meta.schema_version, "v2");
    assert_eq!(meta.dataset_id, "ds_expansion_test");
    assert_eq!(meta.row_count, 10000);
    assert_eq!(meta.feature_count, 20);
    assert_eq!(meta.categorical_columns.len(), 3);
    assert_eq!(meta.numeric_columns.len(), 17);
    // 17 numerics + 3 * 50 categorical expansion = 167 features
    assert_eq!(meta.estimated_expanded_feature_count, 167);
    assert!(meta.transformed_expansion_factor > 8.0);
    assert_eq!(meta.memory_dense_bytes, 10000 * 167 * 8);
}

#[test]
fn test_p8_2_model_family_complexity_estimators() {
    let meta = DatasetMetadata::synthetic("ds_complexity", 5000, 15, 2, 20);

    for family in [
        ModelFamily::Linear,
        ModelFamily::Boosting,
        ModelFamily::Tree,
        ModelFamily::Neural,
        ModelFamily::Custom,
    ] {
        let (runtime, memory, assumptions) = feasibility::estimate_complexity(
            &meta,
            family,
            5,
            true,
            2,
        );

        assert!(runtime.min <= runtime.expected, "runtime.min <= runtime.expected for {:?}", family);
        assert!(runtime.expected <= runtime.max, "runtime.expected <= runtime.max for {:?}", family);
        assert!(memory.min <= memory.expected, "memory.min <= memory.expected for {:?}", family);
        assert!(memory.expected <= memory.max, "memory.expected <= memory.max for {:?}", family);

        assert!(assumptions.iter().any(|a| a.contains("Cross-validation with 5 folds")));
        assert!(assumptions.iter().any(|a| a.contains("Feature transformation expanded")));
        assert!(assumptions.iter().any(|a| a.contains("Concurrent worker lease count: 2")));
    }
}

#[test]
fn test_p8_2_capped_micro_pilots_respect_resource_and_time_caps() {
    let harness = TestHarness::new("micro_pilot");
    harness.create_project("proj_p8");

    // Succeeded pilot with N <= 100
    let pilot_ok = execute_micro_pilot("proj_p8", "ds_ok", ModelFamily::Linear, 100, 3000, 256 * 1024 * 1024);
    assert_eq!(pilot_ok.status, "succeeded");
    assert_eq!(pilot_ok.row_count, 100);
    assert!(pilot_ok.elapsed_ms > 0);

    record_micro_pilot_run(&harness.root, &pilot_ok).unwrap();

    // Verify row count clamping: requested 500 rows is clamped to 100
    let pilot_clamped = execute_micro_pilot("proj_p8", "ds_clamp", ModelFamily::Tree, 500, 3000, 256 * 1024 * 1024);
    assert_eq!(pilot_clamped.row_count, 100, "micro-pilot rows must be strictly <= 100");

    // Memory cap violation guard
    let pilot_oom = execute_micro_pilot("proj_p8", "ds_oom", ModelFamily::Boosting, 100, 3000, 1024); // 1 KB cap
    assert_eq!(pilot_oom.status, "memory_exceeded");
}

#[test]
fn test_p8_3_transparent_multi_tier_verdicts_and_reports() {
    let harness = TestHarness::new("verdicts");
    harness.create_project("p_feas");
    harness.create_project("p_adapt");
    harness.create_project("p_refuse");

    // Case 1: Locally feasible
    let probe_generous = MockFeasibilityProbe {
        host_cpus: 8,
        host_ram: 16 * 1024 * 1024 * 1024,
        disk_avail: 50 * 1024 * 1024 * 1024,
        docker_available: true,
        vm_cpus: Some(8),
        vm_ram: Some(16 * 1024 * 1024 * 1024),
    };
    let hw_gen = HardwareProfile::from_broker_with_probe(&harness.root, &probe_generous).unwrap();
    let meta_small = DatasetMetadata::synthetic("ds_small", 1000, 10, 0, 0);

    let report_feas = assess_feasibility(
        "p_feas",
        &hw_gen,
        &meta_small,
        ModelFamily::Linear,
        5,
        true,
        1,
        3600_000,
        None,
        None,
    );
    assert_eq!(report_feas.verdict, FeasibilityVerdict::LocallyFeasible);
    assert!(report_feas.verdict.is_feasible());
    record_feasibility_report(&harness.root, &report_feas).unwrap();

    let fetched_rep = get_feasibility_report(&harness.root, "p_feas", &report_feas.report_id).unwrap();
    assert_eq!(fetched_rep.report_id, report_feas.report_id);
    assert_eq!(fetched_rep.verdict, FeasibilityVerdict::LocallyFeasible);

    // Case 2: Feasible with adaptation
    let probe_tight = MockFeasibilityProbe {
        host_cpus: 2,
        host_ram: 700 * 1024 * 1024,
        disk_avail: 10 * 1024 * 1024 * 1024,
        docker_available: false,
        vm_cpus: None,
        vm_ram: None,
    };
    let hw_tight = HardwareProfile::from_broker_with_probe(&harness.root, &probe_tight).unwrap();
    let meta_mod = DatasetMetadata::synthetic("ds_mod", 20000, 50, 0, 0);

    let report_adapt = assess_feasibility(
        "p_adapt",
        &hw_tight,
        &meta_mod,
        ModelFamily::Tree,
        5,
        true,
        1,
        3600_000,
        None,
        None,
    );
    assert_eq!(report_adapt.verdict, FeasibilityVerdict::FeasibleWithAdaptation);
    assert!(report_adapt.verdict.is_feasible());
    assert!(!report_adapt.adaptation_proposals.is_empty());

    // Case 3: Requires external compute
    let meta_massive = DatasetMetadata::synthetic("ds_massive", 5_000_000, 200, 10, 50);
    let report_refuse = assess_feasibility(
        "p_refuse",
        &hw_tight,
        &meta_massive,
        ModelFamily::Neural,
        5,
        true,
        1,
        3600_000,
        None,
        None,
    );
    assert_eq!(report_refuse.verdict, FeasibilityVerdict::RequiresExternalCompute);
    assert!(!report_refuse.verdict.is_feasible());
}

#[test]
fn test_p8_4_concrete_evidence_backed_local_alternatives() {
    let meta = DatasetMetadata::synthetic("ds_alts", 40000, 30, 5, 25);
    let alternatives = suggest_alternatives(&meta, ModelFamily::Boosting, &FeasibilityVerdict::FeasibleWithAdaptation);

    let types: Vec<&str> = alternatives.iter().map(|a| a.adaptation_type.as_str()).collect();
    assert!(types.contains(&"quantization"));
    assert!(types.contains(&"sparse_linear"));
    assert!(types.contains(&"progressive_fidelity"));
    assert!(types.contains(&"downsampled_screening"));
    assert!(types.contains(&"smaller_architecture"));
    assert!(types.contains(&"streaming"));

    for alt in &alternatives {
        assert!(alt.predicted_speedup_factor >= 1.0);
        assert!(alt.predicted_memory_reduction_factor >= 1.0);
        assert!(!alt.quality_impact.is_empty());
    }
}

#[test]
fn test_p8_5_deterministic_constraint_negotiation_and_locked_splits() {
    let harness = TestHarness::new("negotiate");
    harness.create_project("p_neg");
    let probe = MockFeasibilityProbe {
        host_cpus: 4,
        host_ram: 4 * 1024 * 1024 * 1024,
        disk_avail: 20 * 1024 * 1024 * 1024,
        docker_available: true,
        vm_cpus: Some(4),
        vm_ram: Some(4 * 1024 * 1024 * 1024),
    };
    let hw = HardwareProfile::from_broker_with_probe(&harness.root, &probe).unwrap();
    let meta = DatasetMetadata::synthetic("ds_neg", 8000, 20, 2, 10);

    let report = assess_feasibility("p_neg", &hw, &meta, ModelFamily::Tree, 3, true, 1, 1800_000, None, None);
    let plan = negotiate_plan(
        &report,
        &["quantization".into(), "smaller_architecture".into()],
        Some(json!({ "target_p95_ms": 30 })),
        Some("STANDARD"),
    );

    assert_eq!(plan.selected_preset, "STANDARD");
    assert_eq!(plan.allocated_threads, 2);
    assert_eq!(plan.allocated_memory_bytes, 2 * 1024 * 1024 * 1024);
    assert!(plan.locked_split_protocol_preserved, "Locked evaluation splits must be strictly preserved!");
    assert_eq!(plan.downscaling_applied.len(), 2);

    record_feasibility_report(&harness.root, &report).unwrap();
    record_execution_plan(&harness.root, &plan).unwrap();
    let fetched_plan = get_execution_plan(&harness.root, "p_neg", &plan.plan_id).unwrap();
    assert_eq!(fetched_plan.plan_id, plan.plan_id);
    assert!(fetched_plan.locked_split_protocol_preserved);
}

#[test]
fn test_p8_5_heavy_dispatch_feasibility_gating() {
    let harness = TestHarness::new("gating");
    harness.create_project("p_gate");

    // Pre-record an infeasible feasibility report
    let probe = MockFeasibilityProbe {
        host_cpus: 4,
        host_ram: 4 * 1024 * 1024 * 1024,
        disk_avail: 20 * 1024 * 1024 * 1024,
        docker_available: true,
        vm_cpus: Some(4),
        vm_ram: Some(4 * 1024 * 1024 * 1024),
    };
    let hw = HardwareProfile::from_broker_with_probe(&harness.root, &probe).unwrap();
    let meta_huge = DatasetMetadata::synthetic("ds_gate_huge", 10_000_000, 500, 20, 100);

    let report_infeasible = assess_feasibility(
        "p_gate",
        &hw,
        &meta_huge,
        ModelFamily::Neural,
        5,
        true,
        1,
        3600_000,
        None,
        None,
    );
    assert_eq!(report_infeasible.verdict, FeasibilityVerdict::RequiresExternalCompute);
    record_feasibility_report(&harness.root, &report_infeasible).unwrap();

    // Verify gating strictly blocks heavy dispatch
    let err = feasibility::enforce_heavy_dispatch_feasibility(
        &harness.root,
        "p_gate",
        "ds_gate_huge",
        "neural",
        8 * 1024 * 1024 * 1024,
    );
    assert!(err.is_err(), "heavy dispatch must be refused when verdict is requires_external_compute");
    let err_msg = err.unwrap_err();
    assert!(err_msg.contains("Heavy training dispatch refused"));
    assert!(err_msg.contains("requires_external_compute"));
    assert!(err_msg.contains("Local alternatives available"));

    // Admissible dispatch
    let meta_small = DatasetMetadata::synthetic("ds_gate_small", 500, 10, 0, 0);
    let report_feasible = assess_feasibility(
        "p_gate",
        &hw,
        &meta_small,
        ModelFamily::Linear,
        3,
        true,
        1,
        3600_000,
        None,
        None,
    );
    record_feasibility_report(&harness.root, &report_feasible).unwrap();

    let ok = feasibility::enforce_heavy_dispatch_feasibility(
        &harness.root,
        "p_gate",
        "ds_gate_small",
        "linear",
        256 * 1024 * 1024,
    );
    assert!(ok.is_ok(), "heavy dispatch must be admitted when verdict is feasible");
}

#[test]
fn test_p8_orchestrator_tool_dispatch() {
    let harness = TestHarness::new("orch_tools");
    let mut orchestrator = harness.setup_orchestrator("p_orch_feas", "t_lead");

    // 1. estimate_feasibility tool call
    let estimate_args = json!({
        "dataset_id": "ds_orch_test",
        "model_family": "boosting",
        "cv_folds": 3,
        "include_final_refit": true,
        "concurrent_worker_leases": 1,
        "time_budget_ms": 1800000,
        "run_pilot": true
    }).to_string();

    let est_res = orchestrator.dispatch_tool_call("estimate_feasibility", &estimate_args).unwrap();
    let rep_json: Value = serde_json::from_str(&est_res).unwrap();
    assert_eq!(rep_json["schema_version"], "v2");
    assert_eq!(rep_json["model_family"], "boosting");
    assert!(rep_json["verdict"].is_string());
    assert!(!rep_json["adaptation_proposals"].as_array().unwrap().is_empty());
    let report_id = rep_json["report_id"].as_str().unwrap();

    // 2. negotiate_execution_plan tool call
    let negotiate_args = json!({
        "report_id": report_id,
        "accepted_adaptations": ["quantization", "progressive_fidelity"],
        "custom_constraints": { "budget_seconds": 60 },
        "target_preset": "MICRO"
    }).to_string();

    let neg_res = orchestrator.dispatch_tool_call("negotiate_execution_plan", &negotiate_args).unwrap();
    let plan_json: Value = serde_json::from_str(&neg_res).unwrap();
    assert_eq!(plan_json["schema_version"], "v2");
    assert_eq!(plan_json["selected_preset"], "MICRO");
    assert_eq!(plan_json["allocated_threads"], 1);
    assert_eq!(plan_json["locked_split_protocol_preserved"], true);
    assert_eq!(plan_json["downscaling_applied"].as_array().unwrap().len(), 2);
}

#[test]
fn test_p8_cli_commands_and_capabilities() {
    let harness = TestHarness::new("cli_p8");
    harness.create_project("p_cli_feas");
    let bin_path = Path::new(env!("CARGO_BIN_EXE_mlrescurix-harness"));

    // Test capabilities
    let mut cmd = Command::new(bin_path);
    cmd.arg("capabilities");
    let out = cmd.output().expect("capabilities command execution");
    assert!(out.status.success());
    let caps: Value = serde_json::from_slice(&out.stdout).unwrap();

    assert_eq!(caps["v2_release_gates"]["p8_feasibility"], "passed");
    let feas_caps = &caps["feasibility_capabilities"];
    assert_eq!(feas_caps["hardware_profile"], true);
    assert_eq!(feas_caps["dataset_metadata"], true);
    assert_eq!(feas_caps["analytical_estimators"], true);
    assert_eq!(feas_caps["capped_micro_pilots"], true);
    assert_eq!(feas_caps["transparent_verdicts"], true);
    assert_eq!(feas_caps["evidence_backed_alternatives"], true);
    assert_eq!(feas_caps["safe_downscaling_negotiation"], true);
    assert_eq!(feas_caps["locked_eval_split_preservation"], true);

    // Test estimate-feasibility CLI
    let mut est_cmd = Command::new(&bin_path);
    est_cmd.args([
        "estimate-feasibility",
        "--state", harness.root.to_str().unwrap(),
        "--project", "p_cli_feas",
        "--dataset", "ds_cli",
        "--family", "tree",
        "--folds", "3",
    ]);
    let est_out = est_cmd.output().expect("estimate-feasibility execution");
    assert!(est_out.status.success(), "estimate-feasibility CLI failed: {:?}", est_out);
    let est_json: Value = serde_json::from_slice(&est_out.stdout).unwrap();
    assert_eq!(est_json["model_family"], "tree");
    let rep_id = est_json["report_id"].as_str().unwrap();

    // Test negotiate-plan CLI
    let mut neg_cmd = Command::new(&bin_path);
    neg_cmd.args([
        "negotiate-plan",
        "--state", harness.root.to_str().unwrap(),
        "--project", "p_cli_feas",
        "--report", rep_id,
        "--preset", "STANDARD",
    ]);
    let neg_out = neg_cmd.output().expect("negotiate-plan execution");
    assert!(neg_out.status.success(), "negotiate-plan CLI failed: {:?}", neg_out);
    let plan_json: Value = serde_json::from_slice(&neg_out.stdout).unwrap();
    assert_eq!(plan_json["selected_preset"], "STANDARD");
    assert_eq!(plan_json["locked_split_protocol_preserved"], true);
}

#[test]
fn test_p8_acceptance_gate_full_closure() {
    let harness = TestHarness::new("p8_gate_closure");
    harness.create_project("p_gate_full");

    // 1. Hardware capability discovery from broker
    let hw = HardwareProfile::from_broker(&harness.root).unwrap();
    assert!(hw.effective_cpus >= 1);
    assert!(hw.effective_memory_bytes >= 64 * 1024 * 1024);

    // 2. Dataset metadata with expansion accounting
    let meta = DatasetMetadata::synthetic("ds_gate", 15000, 30, 4, 30);
    assert!(meta.transformed_expansion_factor > 1.0);

    // 3. Capped micro-pilot probe strictly respecting N <= 100
    let pilot = execute_micro_pilot("p_gate_full", "ds_gate", ModelFamily::Boosting, 100, 3000, 256 * 1024 * 1024);
    assert_eq!(pilot.status, "succeeded");
    assert_eq!(pilot.row_count, 100);
    record_micro_pilot_run(&harness.root, &pilot).unwrap();

    // 4. Feasibility report with explicit multi-tier verdict
    let report = assess_feasibility(
        "p_gate_full",
        &hw,
        &meta,
        ModelFamily::Boosting,
        5,
        true,
        2,
        3600_000,
        None,
        Some(&pilot),
    );
    assert!(report.verdict.is_feasible() || report.verdict == FeasibilityVerdict::RequiresExternalCompute);
    assert_eq!(report.confidence, ConfidenceLevel::High, "micro-pilot calibration raises confidence to High");
    record_feasibility_report(&harness.root, &report).unwrap();

    // 5. Safe deterministic downscaling negotiation preserving locked splits
    let plan = negotiate_plan(
        &report,
        &["quantization".into(), "smaller_architecture".into()],
        Some(json!({ "max_wall_ms": 30000 })),
        Some("STANDARD"),
    );
    assert!(plan.locked_split_protocol_preserved);
    record_execution_plan(&harness.root, &plan).unwrap();

    // 6. Feasibility gating verification
    let gate_res = feasibility::enforce_heavy_dispatch_feasibility(
        &harness.root,
        "p_gate_full",
        "ds_gate",
        "boosting",
        2 * 1024 * 1024 * 1024,
    );
    assert_eq!(gate_res.unwrap(), report.verdict);
}
