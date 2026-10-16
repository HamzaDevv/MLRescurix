use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentRoleSpec {
    pub role_id: String,
    pub version: String,
    pub description: String,
    pub system_prompt: String,
    pub capabilities: Vec<String>,
    pub max_tokens_per_turn: u32,
    pub model_hint: Option<String>,
}

pub fn get_standard_roles() -> Vec<AgentRoleSpec> {
    vec![
        AgentRoleSpec {
            role_id: "lead".into(),
            version: "v1".into(),
            description: "Research director orchestrating problem framing, hypothesis progression, delegation, and stopping decisions.".into(),
            system_prompt: r#"You are the Research Lead for tabular machine learning projects in MLRescurix.
Your responsibility is to direct research: frame the problem, formulate testable hypotheses, delegate focused investigations to specialist roles, review incoming evidence, and make evidence-backed decisions to iterate, pivot, or recommend finalization.
You operate within deterministic resource budgets and hard isolation boundaries.
Always ground your choices in concrete observations from the dataset and validated development metrics."#.into(),
            capabilities: vec![
                "inspect_dataset".into(),
                "inspect_data".into(),
                "register_dataset".into(),
                "delegate".into(),
                "delegate_task".into(),
                "schedule_specialist".into(),
                "propose_hypothesis".into(),
                "record_finding".into(),
                "request_finalization".into(),
                "publish_delivery".into(),
                "publish_bundle".into(),
                "schedule_pipeline".into(),
                "query_budget".into(),
                "retrieve_evidence".into(),
                "register_feature_transform".into(),
                "register_calibration".into(),
                "register_ensemble".into(),
                "inspect_scientific_lineage".into(),
                "workspace_fs".into(),
                "sandbox_exec".into(),
                "evaluate_submission".into(),
                "artifact_crud".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
                "inspect_arrow_dataset".into(),
                "evaluate_temporal_backtest".into(),
                "evaluate_anomaly_clustering".into(),
                "extract_frozen_embeddings".into(),
                "synthesize_candidate".into(),
                "inspect_diagnostics".into(),
                "propose_hypothesis".into(),
                "screen_candidate_fidelity".into(),
                "query_pareto_frontier".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "auditor".into(),
            version: "v1".into(),
            description: "Data auditor inspecting schema, missingness, target distribution, class imbalance, and data integrity.".into(),
            system_prompt: r#"You are the Data Auditor specialist.
Your responsibility is to thoroughly analyze the registered dataset: verify feature schemas, column data types, missing value patterns, target balance, potential target leakage, and train/validation split validity.
Report factual summaries without speculative assumptions."#.into(),
            capabilities: vec![
                "inspect_dataset".into(),
                "register_dataset".into(),
                "inspect_schema".into(),
                "compute_missingness".into(),
                "compute_class_balance".into(),
                "run_sandbox_diagnostic".into(),
                "sandbox_python".into(),
                "retrieve_evidence".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
                "inspect_arrow_dataset".into(),
                "evaluate_temporal_backtest".into(),
                "evaluate_anomaly_clustering".into(),
                "extract_frozen_embeddings".into(),
                "inspect_diagnostics".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "feature_engineer".into(),
            version: "v1".into(),
            description: "Feature engineer designing robust preprocessing, categorical encodings, scaling, and feature transformations.".into(),
            system_prompt: r#"You are the Feature Engineering specialist.
Your responsibility is to design, implement, and validate tabular preprocessing pipelines: handling categorical columns, missing value imputations, scaling, and feature interactions.
Ensure all transformations are strictly fitted on training splits to prevent leakage."#.into(),
            capabilities: vec![
                "file_read".into(),
                "file_write".into(),
                "file_patch".into(),
                "file_list".into(),
                "sandbox_python".into(),
                "run_preflight".into(),
                "retrieve_evidence".into(),
                "register_feature_transform".into(),
                "inspect_scientific_lineage".into(),
                "schedule_pipeline".into(),
                "workspace_fs".into(),
                "sandbox_exec".into(),
                "evaluate_submission".into(),
                "artifact_crud".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
                "extract_frozen_embeddings".into(),
                "synthesize_candidate".into(),
                "inspect_diagnostics".into(),
                "propose_hypothesis".into(),
                "screen_candidate_fidelity".into(),
                "query_pareto_frontier".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "model_specialist".into(),
            version: "v1".into(),
            description: "Model specialist formulating baseline and candidate tabular estimators.".into(),
            system_prompt: r#"You are the Model Specialist.
Your responsibility is to implement tabular estimators (linear baselines, tree-based models, and novel custom Python estimators).
Adhere strictly to the CustomEstimator contract (fit, predict, predict_proba) and verify serialization round-trips."#.into(),
            capabilities: vec![
                "file_read".into(),
                "file_write".into(),
                "file_patch".into(),
                "file_list".into(),
                "sandbox_python".into(),
                "register_candidate".into(),
                "retrieve_evidence".into(),
                "register_calibration".into(),
                "register_ensemble".into(),
                "inspect_scientific_lineage".into(),
                "schedule_pipeline".into(),
                "workspace_fs".into(),
                "sandbox_exec".into(),
                "evaluate_submission".into(),
                "artifact_crud".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
                "evaluate_anomaly_clustering".into(),
                "extract_frozen_embeddings".into(),
                "synthesize_candidate".into(),
                "inspect_diagnostics".into(),
                "propose_hypothesis".into(),
                "screen_candidate_fidelity".into(),
                "query_pareto_frontier".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "search_engineer".into(),
            version: "v1".into(),
            description: "Search engineer orchestrating multi-fidelity constrained hyperparameter searches.".into(),
            system_prompt: r#"You are the Search Engineer.
Your responsibility is to configure and invoke M3 multi-fidelity search routines: defining parameter bounds, screening trials, full-fidelity evaluations, and constraint boundaries.
Interpret search trajectories and convergence behavior."#.into(),
            capabilities: vec![
                "search_init".into(),
                "search_execute".into(),
                "search_run".into(),
                "search_inspect".into(),
                "query_budget".into(),
                "retrieve_evidence".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "diagnostic_engineer".into(),
            version: "v1".into(),
            description: "Diagnostic engineer authoring project tools for error analysis, slice metrics, and calibration.".into(),
            system_prompt: r#"You are the Diagnostic Engineer.
Your responsibility is to author reusable Python diagnostics and inspection utilities in the project branch.
Diagnose performance across slices, residual distributions, and prediction confidence. Remember: prediction-only tools cannot access private validation labels directly; inspect authorized development diagnostics produced by the trusted evaluator."#.into(),
            capabilities: vec![
                "file_read".into(),
                "file_write".into(),
                "file_patch".into(),
                "file_list".into(),
                "author_tool".into(),
                "invoke_tool".into(),
                "register_project_tool".into(),
                "invoke_project_tool".into(),
                "sandbox_python".into(),
                "inspect_diagnostics".into(),
                "retrieve_evidence".into(),
                "workspace_fs".into(),
                "sandbox_exec".into(),
                "evaluate_submission".into(),
                "artifact_crud".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "verifier".into(),
            version: "v1".into(),
            description: "Verification specialist running contract checks, preflight tests, and deterministic reproducibility checks.".into(),
            system_prompt: r#"You are the Verification Specialist.
Your responsibility is to subject candidate pipelines and estimators to strict preflight contract checks: test empty inputs, NaN handling, single-row prediction shapes, multi-class probabilities summing to 1, and joblib round-trip serialization."#.into(),
            capabilities: vec![
                "run_preflight".into(),
                "sandbox_python".into(),
                "verify_reproducibility".into(),
                "retrieve_evidence".into(),
                "workspace_fs".into(),
                "sandbox_exec".into(),
                "evaluate_submission".into(),
                "artifact_crud".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
        AgentRoleSpec {
            role_id: "critic".into(),
            version: "v1".into(),
            description: "Independent pre-freeze critic stress-testing assumptions and evaluating evidence before finalization.".into(),
            system_prompt: r#"You are the Independent Critic.
Your responsibility is to perform an adversarial evaluation of development evidence before any finalization request:
Verify that metric gains are not artifacts of high variance or data snooping; verify that baseline constraints hold across repeated seeds; confirm that no candidate has had access to test data.
Only approve finalization if evidence is unambiguous and robust."#.into(),
            capabilities: vec![
                "inspect_dataset".into(),
                "query_budget".into(),
                "retrieve_evidence".into(),
                "submit_critic_audit".into(),
                "inspect_evidence".into(),
                "compare_contenders".into(),
                "audit_lineage".into(),
                "inspect_scientific_lineage".into(),
                "approve_or_reject_finalization".into(),
                "discover_capabilities".into(),
                "dataset_query".into(),
                "experiment_tree".into(),
                "artifact_crud".into(),
                "estimate_feasibility".into(),
                "negotiate_execution_plan".into(),
            ],
            max_tokens_per_turn: 4096,
            model_hint: None,
        },
    ]
}

pub fn get_consolidated_default_roles() -> Vec<AgentRoleSpec> {
    let standard = get_standard_roles();
    let lead = standard.iter().find(|r| r.role_id == "lead").unwrap().clone();
    let verifier = standard.iter().find(|r| r.role_id == "verifier").unwrap().clone();
    let critic = standard.iter().find(|r| r.role_id == "critic").unwrap().clone();

    // Data/Diagnostics consolidates auditor + diagnostic_engineer
    let auditor = standard.iter().find(|r| r.role_id == "auditor").unwrap();
    let diagnostic_engineer = standard.iter().find(|r| r.role_id == "diagnostic_engineer").unwrap();
    let mut data_diag_caps = auditor.capabilities.clone();
    for cap in &diagnostic_engineer.capabilities {
        if !data_diag_caps.contains(cap) {
            data_diag_caps.push(cap.clone());
        }
    }
    let data_diagnostics = AgentRoleSpec {
        role_id: "data_diagnostics".into(),
        version: "v2".into(),
        description: "Data and diagnostics specialist inspecting dataset schemas, balance, missingness, and authoring/running diagnostics.".into(),
        system_prompt: r#"You are the Data and Diagnostics Specialist (consolidated role).
Your responsibility is to analyze the registered dataset, evaluate schemas, missingness patterns, target distributions, and author or execute diagnostic utilities on development outputs.
Report factual summaries without speculative assumptions; never attempt to inspect sealed holdout test rows."#.into(),
        capabilities: data_diag_caps,
        max_tokens_per_turn: 4096,
        model_hint: None,
    };

    // Experiment Engineer consolidates feature_engineer + model_specialist + search_engineer
    let feature_engineer = standard.iter().find(|r| r.role_id == "feature_engineer").unwrap();
    let model_specialist = standard.iter().find(|r| r.role_id == "model_specialist").unwrap();
    let search_engineer = standard.iter().find(|r| r.role_id == "search_engineer").unwrap();
    let mut exp_caps = feature_engineer.capabilities.clone();
    for cap in model_specialist.capabilities.iter().chain(search_engineer.capabilities.iter()) {
        if !exp_caps.contains(cap) {
            exp_caps.push(cap.clone());
        }
    }
    let experiment_engineer = AgentRoleSpec {
        role_id: "experiment_engineer".into(),
        version: "v2".into(),
        description: "Experiment engineer designing preprocessing, candidate models/pipelines, and multi-fidelity parameter searches.".into(),
        system_prompt: r#"You are the Experiment Engineer (consolidated role).
Your responsibility is to formulate, implement, and search over candidate ML models, feature engineering pipelines, and hyperparameter configurations.
Ensure all transformations strictly isolate train vs validation folds and adhere to executable candidate contracts."#.into(),
        capabilities: exp_caps,
        max_tokens_per_turn: 4096,
        model_hint: None,
    };

    vec![
        lead,
        data_diagnostics,
        experiment_engineer,
        verifier,
        critic,
    ]
}

pub fn get_role(role_id: &str) -> Option<AgentRoleSpec> {
    let normalized = match role_id {
        "data/diagnostics" | "data-diagnostics" => "data_diagnostics",
        "experiment/engineer" | "experiment-engineer" => "experiment_engineer",
        other => other,
    };
    get_standard_roles()
        .into_iter()
        .chain(get_consolidated_default_roles())
        .find(|r| r.role_id == normalized)
}
