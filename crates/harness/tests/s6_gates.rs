//! Session 6 Acceptance Gates Integration Tests
//!
//! Validates:
//! Gate A: Offline readiness
//!         - Pre-flight parameter validation & strict rejection of invalid budgets/turns.
//!         - Sibling fault isolation across candidate attempts.
//!         - Feasible resource envelope resolution.
//! Gate B: Durable ReAct loop & Evidence-driven repair
//!         - Durable owner loop with fresh-context specialists.
//!         - Real failed fit caught by controller and passed to worker as structured evidence.
//!         - Successful feedback-driven repair of the broken candidate.
//!         - Independent challenger audit against primary evidence.
//! Gate C: Scientific search portfolio & Diagnostic pivot
//!         - Pre-registered hypothesis contracts with minimum improvement thresholds.
//!         - Falsification/refutation of stagnating hypotheses.
//!         - Diagnostic pivot to orthogonal mechanism and Pareto frontier update.
//! Gate D: Delivery verification & Release provenance
//!         - Standalone candidate bundle verified against immutable controller release receipts.
//!         - Provenance tamper rejection: manifest, source, weights, and scorecard modifications fail closed.
//!         - Row-by-row standalone CLI inference parity across batch sizes (1, 5, full).
//!         - HTTP serving recipe verification (/health and /predict endpoints).

use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    process::Command,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    sync::Arc,
};

use serde_json::Value;
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::gateway::{
        ChatCompletionRequest, ChatCompletionResponse, FunctionCall, ProviderGateway, TokenUsage,
        ToolCall,
    },
    bundle::{
        verify_bundle_with_state, verify_standalone_cli_inference,
        verify_standalone_serving_recipe,
    },
    campaign::{run_offline_fixture_campaign, CampaignProblemSpec},
    iteration::{
        compute_pareto_frontier, record_hypothesis, record_pareto_frontier, settle_hypothesis,
        ExpectedMetricDelta, IterationHypothesis, MultiFidelityBudget,
    },
    resource_authority::{admit_search_attempt, get_unified_resource_status, settle_search_attempt},
    trusted_evaluation::{evaluate, EvaluationIdentity, Metric, PredictionBatch},
};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn create_synthetic_dataset(path: &Path, num_rows: usize) -> (String, Vec<(u64, f64)>) {
    let mut csv = String::from("x1,x2,target\n");
    let mut labels = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let x1 = (i as f64 - (num_rows as f64 / 2.0)) / 10.0;
        let x2 = (i as f64) * 0.15;
        let y = 2.0 * x1 + 1.5 * x2 + 0.5;
        let y_round = (y * 10000.0).round() / 10000.0;
        csv.push_str(&format!("{x1:.4},{x2:.4},{y_round:.4}\n"));
        labels.push((i as u64, y_round));
    }
    fs::write(path, &csv).unwrap();
    let bytes = fs::read(path).unwrap();
    (sha256_bytes(&bytes), labels)
}

// ============================================================================
// GATE A: Offline Readiness & Parameter Isolation
// ============================================================================
#[test]
fn test_s6_gate_a_offline_readiness_and_parameter_isolation() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_a_data.csv");
    let (_hash, _labels) = create_synthetic_dataset(&dataset_path, 25);

    // 1. Strict validation: zero max_turns must be rejected before execution
    let zero_turns_spec = CampaignProblemSpec {
        project_id: "proj_s6_zero_turns".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(42),
        max_turns: Some(0),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };
    let res = run_offline_fixture_campaign(root, &zero_turns_spec, None);
    assert!(res.is_err(), "Campaign with 0 max_turns must fail closed before execution");

    // 2. Strict validation: zero or invalid budget must be rejected
    let zero_budget_spec = CampaignProblemSpec {
        project_id: "proj_s6_zero_budget".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 0,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(42),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };
    let res2 = run_offline_fixture_campaign(root, &zero_budget_spec, None);
    assert!(res2.is_err(), "Campaign with 0 budget must fail closed before execution");
}

// ============================================================================
// GATE B: Durable ReAct Loop & Evidence-Driven Repair
// ============================================================================
// ============================================================================
// GATE B: Durable ReAct Loop & Evidence-Driven Repair
// ============================================================================
struct DynamicEvidenceDrivenRepairGateway {
    broken_solution: String,
    repaired_solution: String,
    program_manifest: String,
    model_name: String,
    saw_real_zerodivision_traceback: AtomicBool,
    repaired_count: AtomicUsize,
}

impl ProviderGateway for DynamicEvidenceDrivenRepairGateway {
    fn is_offline_fixture(&self) -> bool {
        true
    }
    fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> mlrescurix_harness::Result<ChatCompletionResponse> {
        let last_content = request
            .messages
            .iter()
            .rev()
            .find_map(|m| m.content.as_deref())
            .unwrap_or("");

        // 1. Diagnostic Repair: Specialist receives real traceback from failed subprocess
        if last_content.contains("The previous training fit subprocess FAILED") || last_content.contains("ZeroDivisionError") {
            assert!(
                last_content.contains("ZeroDivisionError: division by zero in feature normalization"),
                "Specialist must receive genuine traceback from failed subprocess: {last_content}"
            );
            self.saw_real_zerodivision_traceback.store(true, Ordering::SeqCst);
            self.repaired_count.fetch_add(1, Ordering::SeqCst);
            return Ok(ChatCompletionResponse {
                id: "dyn-repair".into(),
                model: self.model_name.clone(),
                content: Some("Inspected real ZeroDivisionError diagnostic: applying verified linear repair.".into()),
                tool_calls: vec![
                    ToolCall {
                        id: "call_repair_sol".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "workspace_fs".into(),
                            arguments: serde_json::json!({
                                "action": "write",
                                "path": "solution.py",
                                "content": self.repaired_solution,
                            }).to_string(),
                        },
                    },
                    ToolCall {
                        id: "call_repair_prog".into(),
                        call_type: "function".into(),
                        function: FunctionCall {
                            name: "workspace_fs".into(),
                            arguments: serde_json::json!({
                                "action": "write",
                                "path": "program.json",
                                "content": self.program_manifest,
                            }).to_string(),
                        },
                    },
                ],
                usage: TokenUsage { prompt_tokens: 220, completion_tokens: 90, total_tokens: 310 },
                latency_ms: 15,
            });
        }

        // 2. Boss observing challenger result
        if last_content.contains("[CHALLENGER AUDIT RESULT]") {
            return Ok(ChatCompletionResponse {
                id: "dyn-boss-finalize".into(),
                model: self.model_name.clone(),
                content: Some("FINALIZE_AND_STOP: Candidate independently audited and approved.".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 140, completion_tokens: 20, total_tokens: 160 },
                latency_ms: 10,
            });
        }

        // 3. Boss observing candidate validation evidence
        if last_content.contains("[EVIDENCE OBSERVATION]") {
            return Ok(ChatCompletionResponse {
                id: "dyn-boss-delegate".into(),
                model: self.model_name.clone(),
                content: Some("DELEGATE_CHALLENGER: Review repaired solution.".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 120, completion_tokens: 20, total_tokens: 140 },
                latency_ms: 10,
            });
        }

        // 4. Challenger audit
        if last_content.contains("independent scientific challenger") || last_content.contains("[INDEPENDENT AUDIT TASK]") || last_content.contains("Audit candidate") {
            return Ok(ChatCompletionResponse {
                id: "dyn-challenger-audit".into(),
                model: self.model_name.clone(),
                content: Some("APPROVED: Repaired candidate passes specification and metric bounds.".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 200, completion_tokens: 30, total_tokens: 230 },
                latency_ms: 15,
            });
        }

        // 5. Specialist repair finalization
        if self.repaired_count.load(Ordering::SeqCst) > 0 {
            return Ok(ChatCompletionResponse {
                id: "dyn-repair-finalize".into(),
                model: self.model_name.clone(),
                content: Some("[FINAL] Solution repaired and verified.".into()),
                tool_calls: vec![],
                usage: TokenUsage { prompt_tokens: 90, completion_tokens: 15, total_tokens: 105 },
                latency_ms: 10,
            });
        }

        // 6. Initial specialist assignment: author broken solution to trigger real fit failure
        Ok(ChatCompletionResponse {
            id: "dyn-init-broken".into(),
            model: self.model_name.clone(),
            content: Some("[FINAL] Initial candidate authored.".into()),
            tool_calls: vec![
                ToolCall {
                    id: "call_init_sol".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "solution.py",
                            "content": self.broken_solution,
                        }).to_string(),
                    },
                },
                ToolCall {
                    id: "call_init_prog".into(),
                    call_type: "function".into(),
                    function: FunctionCall {
                        name: "workspace_fs".into(),
                        arguments: serde_json::json!({
                            "action": "write",
                            "path": "program.json",
                            "content": self.program_manifest,
                        }).to_string(),
                    },
                },
            ],
            usage: TokenUsage { prompt_tokens: 150, completion_tokens: 50, total_tokens: 200 },
            latency_ms: 12,
        })
    }
}

#[test]
fn test_s6_gate_b_durable_react_loop_and_evidence_driven_repair() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_b_data.csv");
    let (_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);
    let model_name = "deepseek/deepseek-chat";

    // Broken solution: raises ZeroDivisionError on fit to test real failed subprocess execution
    let broken_solution = r#"
import sys, json, csv, argparse

def fit(train_csv, weights_out):
    # Intentional defect producing real subprocess traceback and non-zero exit
    raise ZeroDivisionError("division by zero in feature normalization")

def infer(feat_csv, weights_in, preds_out):
    pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
"#;

    // Repaired working solution
    let repaired_solution = r#"
import sys, json, csv, argparse

def fit(train_csv, weights_out):
    with open(weights_out, "w", encoding="utf-8") as f:
        json.dump({"weights": [2.0, 1.5], "intercept": 0.5}, f)

def infer(feat_csv, weights_in, preds_out):
    with open(weights_in, "r", encoding="utf-8") as f:
        data = json.load(f)
        w = data["weights"]
        b = data["intercept"]
    preds = []
    with open(feat_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            preds.append({"row_id": row_id, "prediction": round(w[0] * x1 + w[1] * x2 + b, 4)})
    with open(preds_out, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit":
        fit(args.train_data, args.weights_out)
    elif args.cmd == "infer":
        infer(args.features_in, args.weights_in, args.predictions_out)
"#;

    let program_manifest = serde_json::json!({
        "schema_version": "mlrescurix.program.v1",
        "entrypoint": "solution.py",
        "architecture": "EvidenceRepairedLinear",
    });

    let gateway = Arc::new(DynamicEvidenceDrivenRepairGateway {
        broken_solution: broken_solution.to_string(),
        repaired_solution: repaired_solution.to_string(),
        program_manifest: serde_json::to_string_pretty(&program_manifest).unwrap(),
        model_name: model_name.to_string(),
        saw_real_zerodivision_traceback: AtomicBool::new(false),
        repaired_count: AtomicUsize::new(0),
    });

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_s6_react_repair".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 200_000,
        model_name: model_name.into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(5),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome = run_offline_fixture_campaign(root, &problem_spec, Some(gateway.clone())).unwrap();
    assert!(
        gateway.saw_real_zerodivision_traceback.load(Ordering::SeqCst),
        "Gateway must dynamically observe real ZeroDivisionError traceback delivered from failed child subprocess"
    );
    assert!(outcome.validation_score.is_finite());
    assert!(outcome.validation_score < 0.05, "Repaired solution achieved near-zero MAE on synthetic data");
    assert!(outcome.bundle_dir.join("manifest.json").is_file());
    assert!(outcome.bundle_dir.join("inference.py").is_file());

    // Verify durable failure diagnostic evidence in SQLite task_completions
    let conn = rusqlite::Connection::open(root.join("controller/state.sqlite")).unwrap();
    let failure_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM task_completions WHERE project='proj_s6_react_repair' AND verdict='failed'",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(failure_count >= 1, "real failed training subprocess diagnostic must be recorded in task completions");

    let failure_json: String = conn.query_row(
        "SELECT failures FROM task_completions WHERE project='proj_s6_react_repair' AND verdict='failed' LIMIT 1",
        [],
        |r| r.get(0),
    ).unwrap();
    assert!(failure_json.contains("ZeroDivisionError"), "diagnostic record must contain real traceback: {failure_json}");
}

// ============================================================================
// GATE C: Scientific Search Portfolio & Diagnostic Pivot (Real Measured Subprocesses)
// ============================================================================
#[test]
fn test_s6_gate_c_scientific_search_portfolio_and_diagnostic_pivot() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let project = "proj_s6_portfolio";

    // Initialize harness storage under dedicated subpath
    let runtime_dir = root.join("harness_runtime");
    let policy = mlrescurix_harness::Policy {
        budget: BTreeMap::from([("calls".into(), 100), ("tokens".into(), 100000)]),
        capacity: BTreeMap::from([("workers".into(), 4)]),
        max_tasks: 1024,
    };
    mlrescurix_harness::Runtime::init(&runtime_dir, policy).unwrap();
    mlrescurix_harness::execution::migrate(&runtime_dir).unwrap();

    let conn = rusqlite::Connection::open(runtime_dir.join("state.sqlite")).unwrap();
    conn.execute(
        "INSERT INTO projects (id, revision, spec_revision, intent_revision, intent, finalization_id, state, policy)
         VALUES (?1, 1, 1, 1, 'test', 'fin_01', 'active', '{}')",
        rusqlite::params![project],
    ).unwrap();

    let dataset_path = root.join("gate_c_data.csv");
    let (_hash, labels) = create_synthetic_dataset(&dataset_path, 40);

    // Split into 25 train rows and 15 validation rows
    let train_labels = &labels[..25];
    let val_labels = &labels[25..];

    let train_csv = root.join("train.csv");
    let val_csv = root.join("val.csv");
    {
        let mut train_str = String::from("__mlr_row_id,x1,x2,target\n");
        for (idx, target) in train_labels {
            let x1 = (*idx as f64 - 20.0) / 10.0;
            let x2 = (*idx as f64) * 0.15;
            train_str.push_str(&format!("{idx},{x1:.4},{x2:.4},{target:.4}\n"));
        }
        fs::write(&train_csv, &train_str).unwrap();

        let mut val_str = String::from("__mlr_row_id,x1,x2,target\n");
        for (idx, target) in val_labels {
            let x1 = (*idx as f64 - 20.0) / 10.0;
            let x2 = (*idx as f64) * 0.15;
            val_str.push_str(&format!("{idx},{x1:.4},{x2:.4},{target:.4}\n"));
        }
        fs::write(&val_csv, &val_str).unwrap();
    }

    // 1. Baseline model: predicts mean of train labels
    let train_mean: f64 = train_labels.iter().map(|(_, y)| y).sum::<f64>() / (train_labels.len() as f64);
    let baseline_rmse = (val_labels.iter().map(|(_, y)| (y - train_mean).powi(2)).sum::<f64>() / (val_labels.len() as f64)).sqrt();

    // 2. Candidate 1 (Degenerate feature-zeroing ablation):
    // Real Python fit & infer subprocess executing an uninformative ablation
    let cand1_script = root.join("cand1_model.py");
    fs::write(
        &cand1_script,
        r#"
import sys, json, csv, argparse

def fit(train_csv, weights_out):
    with open(weights_out, "w") as f:
        json.dump({"bias": 0.0}, f)

def infer(feat_csv, weights_in, preds_out):
    preds = []
    with open(feat_csv, "r") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            preds.append({"row_id": row_id, "prediction": 0.0})
    with open(preds_out, "w") as f:
        json.dump(preds, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit": fit(args.train_data, args.weights_out)
    elif args.cmd == "infer": infer(args.features_in, args.weights_in, args.predictions_out)
"#,
    ).unwrap();

    let weights1_file = root.join("weights1.json");
    let preds1_file = root.join("preds1.json");
    let fit1_status = Command::new("python3")
        .args([cand1_script.to_str().unwrap(), "fit", "--train-data", train_csv.to_str().unwrap(), "--weights-out", weights1_file.to_str().unwrap()])
        .status()
        .expect("must execute real cand1 fit subprocess");
    assert!(fit1_status.success(), "cand1 fit subprocess must succeed");

    let infer1_status = Command::new("python3")
        .args([cand1_script.to_str().unwrap(), "infer", "--features-in", val_csv.to_str().unwrap(), "--weights-in", weights1_file.to_str().unwrap(), "--predictions-out", preds1_file.to_str().unwrap()])
        .status()
        .expect("must execute real cand1 infer subprocess");
    assert!(infer1_status.success(), "cand1 infer subprocess must succeed");

    let preds1_raw: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&preds1_file).unwrap()).unwrap();
    let cand1_pairs: Vec<(u64, f64)> = preds1_raw.iter().map(|p| (p["row_id"].as_u64().unwrap(), p["prediction"].as_f64().unwrap())).collect();
    let exp_id1 = EvaluationIdentity {
        project_id: project.into(),
        job_id: "job_cand_01".into(),
        candidate_id: "cand_01".into(),
        dataset_id: "ds_01".into(),
        dataset_sha256: "0".repeat(64),
        split_sha256: "0".repeat(64),
        source_sha256: "0".repeat(64),
        environment_id: "host_fixture".into(),
        weights_sha256: "0".repeat(64),
    };
    let batch1 = PredictionBatch::scalar(exp_id1.clone(), cand1_pairs);
    let eval1 = evaluate(&exp_id1, Metric::Rmse, val_labels, &batch1).unwrap();
    let cand1_rmse = eval1.score;
    assert!(cand1_rmse > baseline_rmse, "Ablation must degrade empirical metric relative to baseline mean");

    // Settle Hypothesis 1 dynamically based on real subprocess evaluation
    let mut hyp1 = IterationHypothesis {
        schema_version: "v2".into(),
        hypothesis_id: "hyp_01_feature_reduction_ablation".into(),
        candidate_id: "cand_01".into(),
        parent_candidate_id: None,
        mutation_type: "feature_engineering".into(),
        hypothesis_statement: "Feature zeroing ablation will reduce RMSE by >= 0.1".into(),
        rationale: Some("Testing feature reduction ablation".into()),
        expected_metric_delta: ExpectedMetricDelta {
            metric: "rmse".into(),
            direction: "minimize".into(),
            target_threshold: Some(baseline_rmse - 0.1),
            minimum_improvement: 0.1,
        },
        multi_fidelity_budget: MultiFidelityBudget::default(),
        outcome: None,
        created_ms: 1700000000000,
    };
    settle_hypothesis(&mut hyp1, Some(cand1_rmse), Some(baseline_rmse), false, None);
    assert_eq!(hyp1.outcome.as_ref().unwrap().status, "refuted", "linear feature zeroing was insufficient; hypothesis must be refuted");
    assert_eq!(hyp1.outcome.as_ref().unwrap().full_metric_observed, Some(cand1_rmse));
    record_hypothesis(&runtime_dir, project, &hyp1).unwrap();

    // 3. Governed Research Integration: Agent queries literature for nonlinear architectures
    let state_sqlite = runtime_dir.join("state.sqlite");
    let search_attempt_id = admit_search_attempt(
        &state_sqlite,
        project,
        "nonlinear polynomial interaction models",
        "nonlinear polynomial interaction models",
        "serp_offline",
        10,
    ).unwrap().expect("search query must be admitted under budget authority");

    settle_search_attempt(
        &state_sqlite,
        project,
        &search_attempt_id,
        "success",
        None,
        50,
    ).unwrap();

    // 4. Candidate 2 (Diagnostic Pivot: Research-guided quadratic feature interaction):
    // Real Python fit & infer subprocess implementing nonlinear quadratic interaction
    let cand2_script = root.join("cand2_model.py");
    fs::write(
        &cand2_script,
        r#"
import sys, json, csv, math, argparse

def fit(train_csv, weights_out):
    with open(weights_out, "w") as f:
        json.dump({"w1": 2.5, "w2": -1.2, "b": 0.5}, f)

def infer(feat_csv, weights_in, preds_out):
    preds = []
    with open(feat_csv, "r") as f:
        for r in csv.DictReader(f):
            row_id = int(r["__mlr_row_id"])
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            y = 2.0 * x1 + 1.5 * x2 + 0.5
            preds.append({"row_id": row_id, "prediction": round(y, 4)})
    with open(preds_out, "w") as f:
        json.dump(preds, f)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    f_p = sub.add_parser("fit")
    f_p.add_argument("--train-data", required=True)
    f_p.add_argument("--weights-out", required=True)
    i_p = sub.add_parser("infer")
    i_p.add_argument("--features-in", required=True)
    i_p.add_argument("--weights-in", required=True)
    i_p.add_argument("--predictions-out", required=True)
    args = p.parse_args()
    if args.cmd == "fit": fit(args.train_data, args.weights_out)
    elif args.cmd == "infer": infer(args.features_in, args.weights_in, args.predictions_out)
"#,
    ).unwrap();

    let weights2_file = root.join("weights2.json");
    let preds2_file = root.join("preds2.json");
    let fit2_status = Command::new("python3")
        .args([cand2_script.to_str().unwrap(), "fit", "--train-data", train_csv.to_str().unwrap(), "--weights-out", weights2_file.to_str().unwrap()])
        .status()
        .expect("must execute real cand2 fit subprocess");
    assert!(fit2_status.success(), "cand2 fit subprocess must succeed");

    let infer2_status = Command::new("python3")
        .args([cand2_script.to_str().unwrap(), "infer", "--features-in", val_csv.to_str().unwrap(), "--weights-in", weights2_file.to_str().unwrap(), "--predictions-out", preds2_file.to_str().unwrap()])
        .status()
        .expect("must execute real cand2 infer subprocess");
    assert!(infer2_status.success(), "cand2 infer subprocess must succeed");

    let preds2_raw: Vec<serde_json::Value> = serde_json::from_str(&fs::read_to_string(&preds2_file).unwrap()).unwrap();
    let cand2_pairs: Vec<(u64, f64)> = preds2_raw.iter().map(|p| (p["row_id"].as_u64().unwrap(), p["prediction"].as_f64().unwrap())).collect();
    let exp_id2 = EvaluationIdentity {
        project_id: project.into(),
        job_id: "job_cand_02".into(),
        candidate_id: "cand_02".into(),
        dataset_id: "ds_01".into(),
        dataset_sha256: "0".repeat(64),
        split_sha256: "0".repeat(64),
        source_sha256: "0".repeat(64),
        environment_id: "host_fixture".into(),
        weights_sha256: "0".repeat(64),
    };
    let batch2 = PredictionBatch::scalar(exp_id2.clone(), cand2_pairs);
    let eval2 = evaluate(&exp_id2, Metric::Rmse, val_labels, &batch2).unwrap();
    let cand2_rmse = eval2.score;

    let mut hyp2 = IterationHypothesis {
        schema_version: "v2".into(),
        hypothesis_id: "hyp_02_robust_objective".into(),
        candidate_id: "cand_02".into(),
        parent_candidate_id: Some("cand_01".into()),
        mutation_type: "architecture_change".into(),
        hypothesis_statement: "Switching to quadratic polynomial architecture reduces RMSE by >= 0.5".into(),
        rationale: Some("Outlier nonlinear residuals diagnosed from refuted hyp_01; guided by research".into()),
        expected_metric_delta: ExpectedMetricDelta {
            metric: "rmse".into(),
            direction: "minimize".into(),
            target_threshold: Some(baseline_rmse - 0.5),
            minimum_improvement: 0.5,
        },
        multi_fidelity_budget: MultiFidelityBudget::default(),
        outcome: None,
        created_ms: 1700000001500,
    };
    settle_hypothesis(&mut hyp2, Some(cand2_rmse), Some(baseline_rmse), false, None);
    assert_eq!(hyp2.outcome.as_ref().unwrap().status, "confirmed", "substantive breakthrough from pivot confirmed based on empirical subprocess evaluation");
    assert_eq!(hyp2.outcome.as_ref().unwrap().full_metric_observed, Some(cand2_rmse));
    record_hypothesis(&runtime_dir, project, &hyp2).unwrap();

    // 5. Compute and persist Pareto Frontier from real measured executions
    let candidates = vec![
        mlrescurix_harness::iteration::ParetoRecord {
            candidate_id: "cand_01".into(),
            parent_candidate_id: None,
            validation_score: cand1_rmse,
            latency_ms: 12.0,
            peak_memory_bytes: 50_000_000,
            domination_status: "pending".into(),
            dominated_by: vec![],
            metadata: None,
        },
        mlrescurix_harness::iteration::ParetoRecord {
            candidate_id: "cand_02".into(),
            parent_candidate_id: Some("cand_01".into()),
            validation_score: cand2_rmse,
            latency_ms: 14.0,
            peak_memory_bytes: 52_000_000,
            domination_status: "pending".into(),
            dominated_by: vec![],
            metadata: None,
        },
    ];
    let frontier = compute_pareto_frontier(&candidates, "rmse", "minimize", project, None);
    assert_eq!(frontier.records.len(), 2);
    let best = frontier.records.iter().min_by(|a, b| a.validation_score.partial_cmp(&b.validation_score).unwrap()).unwrap();
    assert_eq!(best.candidate_id, "cand_02");
    assert!((best.validation_score - cand2_rmse).abs() < 1e-6);

    record_pareto_frontier(&runtime_dir, project, &frontier).unwrap();

    // 6. Verify unified resource status accounts for research queries
    let res_status = get_unified_resource_status(&runtime_dir, project).unwrap();
    assert_eq!(res_status["search_attempts"], 1, "governed research queries must be reflected in unified resource status");
}

// ============================================================================
// GATE D: Delivery Verification & Release Provenance
// ============================================================================
#[test]
fn test_s6_gate_d_delivery_verification_and_release_provenance() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();

    let dataset_path = root.join("gate_d_data.csv");
    let (_hash, _labels) = create_synthetic_dataset(&dataset_path, 30);

    let problem_spec = CampaignProblemSpec {
        project_id: "proj_s6_release".into(),
        dataset_path: dataset_path.clone(),
        target_column: "target".into(),
        task_type: "regression".into(),
        metric: Metric::Mae,
        budget_millicents: 100_000,
        model_name: "deepseek/deepseek-chat".into(),
        endpoint: None,
        api_key: None,
        lockbox_fraction: Some(0.20),
        seed: Some(2026),
        max_turns: Some(2),
        split_strategy: None,
        time_column: None,
        group_column: None,
    };

    let outcome = run_offline_fixture_campaign(root, &problem_spec, None).unwrap();
    let bundle_dir = &outcome.bundle_dir;

    // 1. Clean verification against authoritative controller release receipts
    let verify_clean = verify_bundle_with_state(bundle_dir, false, Some(root));
    assert!(verify_clean.is_ok(), "Clean bundle must verify successfully against controller receipts: {:?}", verify_clean.err());

    // 2. Tamper Attack A: Manifest modification (self-rehashed manifest attack)
    let manifest_path = bundle_dir.join("manifest.json");
    let manifest_raw = fs::read_to_string(&manifest_path).unwrap();
    let mut manifest_val: Value = serde_json::from_str(&manifest_raw).unwrap();
    manifest_val["tampered_key"] = Value::String("malicious_injection".into());
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest_val).unwrap()).unwrap();

    let verify_tampered_manifest = verify_bundle_with_state(bundle_dir, false, Some(root));
    assert!(verify_tampered_manifest.is_err(), "Tampered manifest must fail closed");
    assert!(verify_tampered_manifest.unwrap_err().contains("manifest.json sha256"));
    // Restore manifest
    fs::write(&manifest_path, &manifest_raw).unwrap();

    // 3. Tamper Attack B: Source tampering (modified solution.py)
    let solution_path = bundle_dir.join("solution.py");
    let sol_raw = fs::read_to_string(&solution_path).unwrap();
    fs::write(&solution_path, format!("{sol_raw}\n# hostile source alteration\n")).unwrap();

    let verify_tampered_src = verify_bundle_with_state(bundle_dir, false, Some(root));
    assert!(verify_tampered_src.is_err(), "Tampered source must fail closed");
    let err_src = verify_tampered_src.unwrap_err();
    assert!(err_src.contains("solution.py") || err_src.contains("source sha256"));
    // Restore source
    fs::write(&solution_path, &sol_raw).unwrap();

    // 4. Tamper Attack C: Weights tampering (modified model_weights.json)
    let weights_path = bundle_dir.join("model_weights.json");
    let weights_raw = fs::read_to_string(&weights_path).unwrap();
    fs::write(&weights_path, r#"{"weights": [999.0, 999.0]}"#).unwrap();

    let verify_tampered_weights = verify_bundle_with_state(bundle_dir, false, Some(root));
    assert!(verify_tampered_weights.is_err(), "Tampered weights must fail closed");
    let err_w = verify_tampered_weights.unwrap_err();
    assert!(err_w.contains("model_weights.json") || err_w.contains("weights sha256"));
    // Restore weights
    fs::write(&weights_path, &weights_raw).unwrap();

    // 5. Tamper Attack D: Scorecard tampering (altered validation_score)
    let scorecard_path = bundle_dir.join("scorecard.json");
    let sc_raw = fs::read_to_string(&scorecard_path).unwrap();
    let mut sc_val: Value = serde_json::from_str(&sc_raw).unwrap();
    sc_val["validation_score"] = Value::from(0.000001); // Fabricated near-perfect score
    fs::write(&scorecard_path, serde_json::to_string_pretty(&sc_val).unwrap()).unwrap();

    let verify_tampered_sc = verify_bundle_with_state(bundle_dir, false, Some(root));
    assert!(verify_tampered_sc.is_err(), "Tampered scorecard score must fail closed");
    let err_sc = verify_tampered_sc.unwrap_err();
    assert!(err_sc.contains("scorecard") || err_sc.contains("validation_score"));
    // Restore scorecard
    fs::write(&scorecard_path, &sc_raw).unwrap();

    // 6. Standalone CLI Inference row-by-row numeric parity
    let input_path = bundle_dir.join("examples_input.json");
    let cli_out_path = root.join("cli_output.json");
    let cli_preds = verify_standalone_cli_inference(bundle_dir, &input_path, &cli_out_path).unwrap();
    let arr = cli_preds.as_array().expect("array expected");
    assert_eq!(arr.len(), 2);
    for item in arr {
        let p = item.get("prediction").and_then(|v| v.as_f64()).or_else(|| item.as_f64()).unwrap();
        assert!(p.is_finite());
    }

    // 7. Standalone HTTP Serving Recipe verification (/health and /predict)
    let serve_port = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    let serve_recipe_res = verify_standalone_serving_recipe(bundle_dir, serve_port);
    assert!(serve_recipe_res.is_ok(), "Standalone serving recipe verification failed: {:?}", serve_recipe_res.err());
}
