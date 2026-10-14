use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};
use rusqlite::params;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use mlrescurix_harness::{
    agents::{
        gateway::{ProviderGateway, ScriptedProviderFixture},
        orchestrator::{AgentOrchestrator, OrchestratorConfig},
        sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus, SCIENTIFIC_IMAGE_REF},
        workspace::BranchWorkspaceManager,
    },
    artifacts::ArtifactRegistry,
    core::{Operation, Policy, Request},
    execution,
    Runtime,
};

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn test_policy(limit: i64) -> Policy {
    let mut b = BTreeMap::new();
    b.insert("calls".into(), limit);
    b.insert("tokens".into(), limit * 1000);
    b.insert("wall_ms".into(), limit * 10000);
    let mut c = BTreeMap::new();
    c.insert("workers".into(), 4);
    c.insert("cpu_threads".into(), 4);
    Policy { max_tasks: 1024, budget: b, capacity: c }
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

fn setup_p4_project(root: &Path, project: &str, task_id: &str) -> AgentOrchestrator {
    if !root.join("OWNED").exists() {
        if root.exists() {
            let _ = fs::remove_dir(root);
        }
        Runtime::init(root, test_policy(100)).unwrap();
    }
    execution::migrate(root).unwrap();
    let r = Runtime::open(root).unwrap();
    let branch = format!("b-{task_id}");

    let _ = r.command(Request {
        command_id: format!("init-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateProject {
            intent: "P4 candidate tools test".into(),
            finalization_id: format!("final-{project}"),
            policy: test_policy(100),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("branch-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateBranch {
            branch_id: branch.clone(),
            base_hash: "base-p4".into(),
            code_hash: "code-p4".into(),
        },
    }).unwrap();

    let _ = r.command(Request {
        command_id: format!("task-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::CreateTask {
            task_id: task_id.into(),
            branch_id: branch.clone(),
            context: test_context("base-p4", "code-p4"),
        },
    }).unwrap();

    let now = now_ms();
    let _ = r.command(Request {
        command_id: format!("claim-{project}"),
        project_id: project.into(),
        expected_revision: None,
        operation: Operation::Claim {
            task_id: task_id.into(),
            owner: format!("agent-{task_id}"),
            lease_id: format!("lease-{task_id}"),
            now_ms: now,
            expires_ms: now + 3_600_000,
            expected_generation: 1,
        },
    }).unwrap();

    let config = OrchestratorConfig {
        project_id: project.into(),
        task_id: task_id.into(),
        role_id: "feature_engineer".into(),
        branch_id: branch,
        model: "dummy-model".into(),
        max_turns: 10,
    };

    let gw: Arc<dyn ProviderGateway> = Arc::new(ScriptedProviderFixture::new(vec![]));
    AgentOrchestrator::new(root, config, gw).unwrap()
}

fn compute_dir_source_hash(dir: &Path) -> String {
    let mut files = Vec::new();
    fn visit(base: &Path, curr: &Path, files: &mut Vec<(String, Vec<u8>)>) {
        for entry in fs::read_dir(curr).unwrap() {
            let entry = entry.unwrap();
            let p = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with('.') || name == "__pycache__" {
                continue;
            }
            if p.is_dir() {
                visit(base, &p, files);
            } else if p.is_file() {
                if name.ends_with(".pyc") || name.ends_with(".staging") || name == "manifest.json" {
                    continue;
                }
                let rel = p.strip_prefix(base).unwrap().to_string_lossy().to_string();
                let bytes = fs::read(&p).unwrap();
                files.push((rel, bytes));
            }
        }
    }
    visit(dir, dir, &mut files);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, content) in files {
        let f_sha = format!("{:x}", Sha256::digest(&content));
        let line = format!("{}:{}:{}\n", rel, content.len(), f_sha);
        hasher.update(line.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[test]
fn p4_1_discover_capabilities_reports_contract_boundaries() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-caps", "t-caps");

    let out = orch.dispatch_tool_call("discover_capabilities", "{}").unwrap();
    let val: Value = serde_json::from_str(&out).unwrap();

    // Verify tools list includes all P4 tools
    let tools = val["tools"].as_array().expect("tools array");
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    assert!(tool_names.contains(&"workspace_fs"), "missing workspace_fs");
    assert!(tool_names.contains(&"sandbox_exec"), "missing sandbox_exec");
    assert!(tool_names.contains(&"evaluate_submission"), "missing evaluate_submission");
    assert!(tool_names.contains(&"artifact_crud"), "missing artifact_crud");
    assert!(tool_names.contains(&"discover_capabilities"), "missing discover_capabilities");
    assert!(tool_names.contains(&"dataset_query"), "missing dataset_query");
    assert!(tool_names.contains(&"experiment_tree"), "missing experiment_tree");

    // Verify sandbox report and default scientific environment
    let env_prof = &val["environment_profile"];
    let default_img = env_prof["default_image"].as_str().unwrap();
    assert_eq!(default_img, SCIENTIFIC_IMAGE_REF.trim());
    assert!(env_prof["thread_caps_enforced"].as_bool().unwrap());

    // Verify resource presets
    assert!(val["resource_presets"]["MICRO"]["cpus"].as_u64().unwrap() >= 1);
    assert!(val["resource_presets"]["STANDARD"]["cpus"].as_u64().unwrap() >= 2);
    assert!(val["resource_presets"]["HEAVY"]["memory_mb"].as_u64().unwrap() >= 4096);

    // Verify query limits & state bounds
    assert_eq!(val["validation_query_budget"]["max_queries"].as_u64().unwrap(), 50);
    assert_eq!(val["state_limits"]["max_state_bytes"].as_u64().unwrap(), 134217728);
}

#[test]
fn p4_1_workspace_fs_operations_and_concurrency_fencing() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-ws", "t-ws");

    // 1. Write file
    let write_res = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "src/candidate.py",
        "content": "# initial content\nx = 10\n",
    }).to_string()).unwrap();
    let w_val: Value = serde_json::from_str(&write_res).unwrap();
    assert_eq!(w_val["status"], "written");
    let sha1 = w_val["sha256"].as_str().unwrap().to_string();

    // 2. Read file
    let read_res = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "read",
        "path": "src/candidate.py",
    }).to_string()).unwrap();
    let r_val: Value = serde_json::from_str(&read_res).unwrap();
    assert_eq!(r_val["status"], "read");
    assert_eq!(r_val["content"], "# initial content\nx = 10\n");

    // 3. Patch with valid expected hash (concurrency check)
    let patch_res = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "patch",
        "path": "src/candidate.py",
        "target": "x = 10",
        "replacement": "x = 42",
        "expected_hash": sha1,
    }).to_string()).unwrap();
    let p_val: Value = serde_json::from_str(&patch_res).unwrap();
    assert_eq!(p_val["status"], "patched");
    let sha2 = p_val["sha256"].as_str().unwrap().to_string();

    // 4. Stale expected hash check: patching with stale sha1 must fail
    let stale_patch = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "patch",
        "path": "src/candidate.py",
        "target": "x = 42",
        "replacement": "x = 100",
        "expected_hash": sha1, // stale!
    }).to_string());
    assert!(stale_patch.is_err(), "stale expected_hash must be rejected");
    assert!(stale_patch.unwrap_err().contains("expected SHA-256"));

    // 5. Write with valid expected hash sha2
    let write2 = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "src/candidate.py",
        "content": "# updated content\nx = 99\n",
        "expected_hash": sha2,
    }).to_string()).unwrap();
    assert!(write2.contains("written"));

    // 6. Path traversal rejection
    let escape = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "../escaped.txt",
        "content": "illegal escape",
    }).to_string());
    assert!(escape.is_err());
    assert!(escape.unwrap_err().contains("escapes branch workspace"));

    // 7. Glob and search
    let _ = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "tests/test_cand.py",
        "content": "def test_x(): assert x == 99\n",
    }).to_string()).unwrap();

    let glob_res = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "glob",
        "pattern": "*.py",
    }).to_string()).unwrap();
    let g_val: Value = serde_json::from_str(&glob_res).unwrap();
    let files = g_val["files"].as_array().unwrap();
    assert!(files.len() >= 2);

    let search_res = orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "search",
        "query": "assert x ==",
    }).to_string()).unwrap();
    let s_val: Value = serde_json::from_str(&search_res).unwrap();
    let matches = s_val["matches"].as_array().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0]["path"], "tests/test_cand.py");

    // 8. Backward compatibility: file_* operations remain functional
    let compat_read = orch.dispatch_tool_call("file_read", &json!({ "path": "src/candidate.py" }).to_string()).unwrap();
    assert!(compat_read.contains("x = 99"));
}

#[test]
fn p4_1_sandbox_exec_isolation_and_fallback() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-sbx", "t-sbx");

    // Check fallback isolation contract:
    // If container runtime is unavailable, host python fallback is strictly refused
    let unavailable_limits = SandboxLimits::default();
    let refusal = SandboxRunner::execute_argv(
        td.path(),
        &["python3".into(), "-c".into(), "print('untrusted')".into()],
        &unavailable_limits,
        Some("nonexistent-image"),
    );
    // If docker daemon is online, execute_argv proceeds with docker;
    // but when docker is unavailable, it returns an explicit error
    let status = SandboxDetector::detect();
    match status {
        SandboxStatus::Unavailable { .. } => {
            assert!(refusal.is_err());
            let err = refusal.unwrap_err();
            assert!(err.contains("Sandbox execution refused") && err.contains("Host Python substitution"));
        }
        SandboxStatus::Available { .. } => {
            // Docker is available on this system! Run a read-only workspace sandbox test
            let res = orch.dispatch_tool_call("sandbox_exec", &json!({
                "argv": ["python3", "-c", "import sys; sys.stdout.write('hello from sandbox')"],
            }).to_string()).unwrap();
            let val: Value = serde_json::from_str(&res).unwrap();
            assert_eq!(val["status"], "succeeded");
            assert_eq!(val["exit_code"], 0);
            assert!(val["stdout"].as_str().unwrap().contains("hello from sandbox"));
        }
    }
}

#[test]
fn p4_2_p4_3_executable_candidate_protocol_and_evaluation() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-eval", "t-eval");

    // 1. Create candidate directory and executable script in branch workspace
    let cand_code = r#"#!/usr/bin/env python3
import sys
import json
import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--train-path", required=True)
    fit_p.add_argument("--model-dir", required=True)
    
    pred_p = sub.add_parser("predict")
    pred_p.add_argument("--input-path", required=True)
    pred_p.add_argument("--model-dir", required=True)
    pred_p.add_argument("--output-path", required=True)
    
    args = parser.parse_args()
    
    if args.cmd == "fit":
        df = pd.read_csv(args.train_path)
        # Fit simple threshold rule
        m_val = float(df["feat_1"].median())
        import os
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, "model.json"), "w") as f:
            json.dump({"threshold": m_val}, f)
        print("fit completed")
        return 0
        
    elif args.cmd == "predict":
        df = pd.read_csv(args.input_path)
        # CRITICAL VERIFICATION: Label isolation
        if "target" in df.columns:
            raise RuntimeError("CRITICAL ERROR: target label leaked to predict input!")
            
        import os
        with open(os.path.join(args.model_dir, "model.json")) as f:
            m = json.load(f)
            
        th = m["threshold"]
        preds = []
        p0 = []
        p1 = []
        for val in df["feat_1"]:
            if val >= th:
                preds.append("class_1")
                p0.append(0.2)
                p1.append(0.8)
            else:
                preds.append("class_0")
                p0.append(0.8)
                p1.append(0.2)
                
        out_df = pd.DataFrame({
            "row_id": df["row_id"],
            "pred": preds,
            "prob_class_0": p0,
            "prob_class_1": p1,
        })
        out_df.to_csv(args.output_path, index=False)
        print("predict completed")
        return 0

if __name__ == "__main__":
    sys.exit(main())
"#;

    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "candidate/candidate.py",
        "content": cand_code,
    }).to_string()).unwrap();

    // Compute source hash for manifest
    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-eval", "b-t-eval");
    let cand_dir = branch_dir.join("candidate");
    let src_hash = compute_dir_source_hash(&cand_dir);

    let manifest = json!({
        "candidate_id": "cand_rule_model",
        "name": "Deterministic Rule Baseline",
        "task_type": "binary",
        "entrypoint": "candidate.py",
        "supported_operations": ["fit", "predict"],
        "stable_row_keys": "row_id",
        "class_order": ["class_0", "class_1"],
        "raw_input_schema": { "target": "target" },
        "source_hash": src_hash,
    });

    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "candidate/manifest.json",
        "content": serde_json::to_string_pretty(&manifest).unwrap(),
    }).to_string()).unwrap();

    // 2. Register synthetic dataset
    let mut csv_rows = vec!["row_id,feat_1,feat_2,slice_cat,target".to_string()];
    for i in 0..100 {
        let feat1 = (i as f64) * 0.1;
        let feat2 = (i as f64) * 2.5;
        let slice_cat = if i < 30 { "low" } else if i < 70 { "mid" } else { "high" };
        let target = if i >= 50 { "class_1" } else { "class_0" };
        csv_rows.push(format!("{i},{feat1:.2},{feat2:.2},{slice_cat},{target}"));
    }
    let csv_content = csv_rows.join("\n");
    let csv_file = td.path().join("synth_dataset.csv");
    fs::write(&csv_file, &csv_content).unwrap();

    let reg_ds = execution::register_dataset(td.path(), "p-eval", "synth_data", &csv_file).unwrap();
    assert!(reg_ds["sha256"].as_str().is_some());

    // 3. Query dataset via dataset_query tool
    let q_res = orch.dispatch_tool_call("dataset_query", &json!({
        "dataset_id": "synth_data",
    }).to_string()).unwrap();
    let q_val: Value = serde_json::from_str(&q_res).unwrap();
    assert_eq!(q_val["dataset_id"], "synth_data");
    assert_eq!(q_val["row_count"], 100);
    assert_eq!(q_val["columns"].as_array().unwrap().len(), 5);

    // 4. Trigger trusted evaluation
    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_rule_model",
        "candidate_path": "candidate",
        "dataset_id": "synth_data",
        "folds": 3,
        "seed": 2026,
        "diagnostic_slices": ["slice_cat"],
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    assert_eq!(eval_val["status"], "evaluated");
    assert_eq!(eval_val["candidate_id"], "cand_rule_model");
    assert!(eval_val["primary_metric"].as_f64().unwrap() >= 0.5);
    assert_eq!(eval_val["query_count"].as_u64().unwrap(), 1);
    assert_eq!(eval_val["max_queries"].as_u64().unwrap(), 50);

    // Verify official scorecard was registered in CAS and relational artifact registry
    let scorecard_ver = eval_val["scorecard_version_id"].as_str().unwrap();
    let reg = ArtifactRegistry::new(td.path());
    let ver_view = reg.get_version("p-eval", scorecard_ver).unwrap().expect("scorecard version exists");
    assert_eq!(ver_view.media_type, "application/json");

    // Verify scorecard artifact kind is 'scorecard' and authority is 'evaluator:trusted'
    let art_view = reg.get_artifact("p-eval", "scorecard_cand_rule_model").unwrap().expect("scorecard artifact exists");
    assert_eq!(art_view.kind, "scorecard");
    assert_eq!(art_view.producer_authority, "evaluator:trusted");
}

#[test]
fn p4_2_prediction_rejections_and_tamper_fencing() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-tamper", "t-tamper");

    // Prepare candidate with misbehaving prediction
    let bad_cand_code = r#"#!/usr/bin/env python3
import sys, argparse, pandas as pd

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
fit_p = sub.add_parser("fit")
fit_p.add_argument("--train-path", required=True)
fit_p.add_argument("--model-dir", required=True)
pred_p = sub.add_parser("predict")
pred_p.add_argument("--input-path", required=True)
pred_p.add_argument("--model-dir", required=True)
pred_p.add_argument("--output-path", required=True)
args = parser.parse_args()

if args.cmd == "fit":
    sys.exit(0)
elif args.cmd == "predict":
    df = pd.read_csv(args.input_path)
    # Malformed outputs: wrong row_ids!
    bad_df = pd.DataFrame({
        "row_id": [9999 + i for i in range(len(df))],
        "pred": ["class_0"] * len(df),
        "prob_class_0": [0.5] * len(df),
        "prob_class_1": [0.5] * len(df),
    })
    bad_df.to_csv(args.output_path, index=False)
    sys.exit(0)
"#;

    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "candidate/candidate.py",
        "content": bad_cand_code,
    }).to_string()).unwrap();

    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-tamper", "b-t-tamper");
    let cand_dir = branch_dir.join("candidate");
    let src_hash = compute_dir_source_hash(&cand_dir);

    let manifest = json!({
        "candidate_id": "cand_bad",
        "name": "Bad Candidate",
        "task_type": "binary",
        "entrypoint": "candidate.py",
        "supported_operations": ["fit", "predict"],
        "stable_row_keys": "row_id",
        "class_order": ["class_0", "class_1"],
        "raw_input_schema": { "target": "target" },
        "source_hash": src_hash,
    });

    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "candidate/manifest.json",
        "content": serde_json::to_string(&manifest).unwrap(),
    }).to_string()).unwrap();

    // Register synthetic dataset
    let mut csv_rows = vec!["row_id,feat_1,target".to_string()];
    for i in 0..50 {
        csv_rows.push(format!("{i},{i},class_0"));
    }
    let csv_file = td.path().join("tamper_dataset.csv");
    fs::write(&csv_file, csv_rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-tamper", "tamper_data", &csv_file).unwrap();

    // Rejection 1: Malformed row IDs in predictions
    let row_misalignment_err = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_bad",
        "candidate_path": "candidate",
        "dataset_id": "tamper_data",
        "folds": 2,
    }).to_string());
    assert!(row_misalignment_err.is_err());
    let err_msg = row_misalignment_err.unwrap_err();
    assert!(err_msg.contains("row misalignment") || err_msg.contains("candidate evaluation failed"));

    // Rejection 2: Tampered source hash
    // Modify candidate.py without updating manifest.json
    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "candidate/candidate.py",
        "content": "# tampered after manifest hashing\nimport sys\nsys.exit(1)\n",
    }).to_string()).unwrap();

    let tamper_err = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_bad",
        "candidate_path": "candidate",
        "dataset_id": "tamper_data",
        "folds": 2,
    }).to_string());
    assert!(tamper_err.is_err());
    assert!(tamper_err.unwrap_err().contains("source_hash mismatch"));
}

#[test]
fn p4_3_split_label_isolation_and_evaluator_authority() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-auth", "t-auth");

    // Authority Invariant: Agent cannot forge official scorecards via artifact_crud
    let forge_attempt = orch.dispatch_tool_call("artifact_crud", &json!({
        "action": "publish",
        "artifact_id": "fake_scorecard",
        "kind": "scorecard",
        "blob_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "version": "v1",
    }).to_string());

    assert!(forge_attempt.is_err(), "Agent must NOT be permitted to publish kind='scorecard'");
    let err = forge_attempt.unwrap_err();
    assert!(err.contains("scorecard publication is restricted to the trusted evaluator API"));

    // Publishing valid agent artifacts (e.g. 'code', 'research_note', 'model_weights') succeeds
    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "notes.txt",
        "content": "Initial EDA notes and observations",
    }).to_string()).unwrap();

    let valid_pub = orch.dispatch_tool_call("artifact_crud", &json!({
        "action": "publish",
        "artifact_id": "research_notes_1",
        "kind": "research_note",
        "source_path": "notes.txt",
        "version": "v1",
        "summary": "Initial EDA notes",
    }).to_string()).unwrap();

    let v_pub: Value = serde_json::from_str(&valid_pub).unwrap();
    assert_eq!(v_pub["status"], "published");
    assert_eq!(v_pub["artifact_id"], "research_notes_1");
}

#[test]
fn p4_3_decoupled_recipes_and_experiment_tree() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-dag", "t-dag");

    // Register dataset and search session required by foreign keys
    let ds_file = td.path().join("tree_ds.csv");
    fs::write(&ds_file, "row_id,f1,target\n0,1.0,class_0\n").unwrap();
    execution::register_dataset(td.path(), "p-dag", "tree_data", &ds_file).unwrap();

    let conn = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    let now = now_ms();

    conn.execute(
        "INSERT INTO search_sessions(project, id, dataset_id, spec, status, created_ms, updated_ms)
         VALUES(?1, 'sess_tree', 'tree_data', '{}', 'screening', ?2, ?2)",
        params!["p-dag", now],
    ).unwrap();

    // Verify experiment_dag accepts arbitrary recipe strings (decoupled from old 3-enum)
    let custom_recipes: [(&str, &str, f64, &str); 3] = [
        ("exp_deep_forest", "deep_forest_v1", 0.5, "screening"),
        ("exp_neural_slm", "neural_slm_qlora", 1.0, "full_fidelity"),
        ("exp_catboost_custom", "custom_catboost_optuna", 0.5, "screening"),
    ];

    for (idx, (id, recipe, fidelity, node_type)) in custom_recipes.iter().enumerate() {
        conn.execute(
            "INSERT INTO experiment_dag(project, session_id, id, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms)
             VALUES(?1, 'sess_tree', ?2, ?3, '{}', 42, ?4, ?5, NULL, ?6)",
            params!["p-dag", id, recipe, fidelity, node_type, now + idx as i64],
        ).expect("custom recipe must insert cleanly without check constraint failure");
    }

    // Query experiment_tree tool
    let tree_res = orch.dispatch_tool_call("experiment_tree", "{}").unwrap();
    let t_val: Value = serde_json::from_str(&tree_res).unwrap();
    let experiments = t_val["experiments"].as_array().unwrap();
    assert_eq!(experiments.len(), 3);
    assert_eq!(experiments[0]["recipe"], "deep_forest_v1");
    assert_eq!(experiments[1]["recipe"], "neural_slm_qlora");
    assert_eq!(experiments[2]["recipe"], "custom_catboost_optuna");
}

#[test]
fn p4_4_validation_query_budget_and_slice_suppression() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-budget", "t-budget");

    // 1. Diagnostic slice suppression test
    // Create candidate
    let cand_code = r#"#!/usr/bin/env python3
import sys, argparse, pandas as pd
parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
fit_p = sub.add_parser("fit")
fit_p.add_argument("--train-path", required=True)
fit_p.add_argument("--model-dir", required=True)
pred_p = sub.add_parser("predict")
pred_p.add_argument("--input-path", required=True)
pred_p.add_argument("--model-dir", required=True)
pred_p.add_argument("--output-path", required=True)
args = parser.parse_args()
if args.cmd == "fit": sys.exit(0)
elif args.cmd == "predict":
    df = pd.read_csv(args.input_path)
    pd.DataFrame({
        "row_id": df["row_id"],
        "pred": ["class_0"] * len(df),
        "prob_class_0": [0.6] * len(df),
        "prob_class_1": [0.4] * len(df),
    }).to_csv(args.output_path, index=False)
    sys.exit(0)
"#;
    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "cand/candidate.py",
        "content": cand_code,
    }).to_string()).unwrap();

    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-budget", "b-t-budget");
    let cand_dir = branch_dir.join("cand");
    let src_hash = compute_dir_source_hash(&cand_dir);

    let manifest = json!({
        "candidate_id": "cand_slices",
        "name": "Slice Test Candidate",
        "task_type": "binary",
        "entrypoint": "candidate.py",
        "supported_operations": ["fit", "predict"],
        "stable_row_keys": "row_id",
        "class_order": ["class_0", "class_1"],
        "raw_input_schema": { "target": "target" },
        "source_hash": src_hash,
    });
    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "cand/manifest.json",
        "content": serde_json::to_string(&manifest).unwrap(),
    }).to_string()).unwrap();

    // Create dataset: 100 rows. min_slice_support = max(20, 5% of 100 = 5) = 20.
    // Slice "rare_val" has only 5 rows (< 20 support) -> MUST be suppressed!
    // Slice "common_val" has 95 rows (>= 20 support) -> MUST be reported!
    let mut rows = vec!["row_id,feat_1,slice_col,target".to_string()];
    for i in 0..100 {
        let s_val = if i < 5 { "rare_val" } else { "common_val" };
        rows.push(format!("{i},1.0,{s_val},class_0"));
    }
    let ds_file = td.path().join("slices_ds.csv");
    fs::write(&ds_file, rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-budget", "slice_data", &ds_file).unwrap();

    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_slices",
        "candidate_path": "cand",
        "dataset_id": "slice_data",
        "folds": 2,
        "diagnostic_slices": ["slice_col"],
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    let diag = &eval_val["diagnostic_slices"]["slice_col"];
    assert_eq!(diag["rare_val"]["status"], "suppressed");
    assert!(diag["rare_val"]["reason"].as_str().unwrap().contains("below min_slice_support"));
    assert_eq!(diag["common_val"]["status"], "reported");

    // 2. Adaptive validation budget exhaustion test
    // Set query count to max (50) in DB
    let conn = rusqlite::Connection::open(td.path().join("state.sqlite")).unwrap();
    conn.execute(
        "UPDATE validation_query_counters SET query_count = 50 WHERE project=?1",
        params!["p-budget"],
    ).unwrap();

    // 51st query must be rejected
    let exhausted_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_slices",
        "candidate_path": "cand",
        "dataset_id": "slice_data",
        "folds": 2,
    }).to_string());

    assert!(exhausted_res.is_err());
    let err = exhausted_res.unwrap_err();
    assert!(err.contains("Validation query budget exhausted") && err.contains("adaptive validation limit exceeded"));
}

#[test]
fn p4_acceptance_gate_full_closure() {
    let td = tempfile::tempdir().unwrap();
    let mut orch = setup_p4_project(td.path(), "p-gate", "t-gate");

    // Acceptance Gate Criteria:
    // 1. Agents can create and fix arbitrary candidate code without modifying a recipe registry.
    // 2. Evaluation, scoring, and budgets remain under trusted authority.
    // 3. Split/label isolation, single-use final test gate, and generation/intent fencing are strictly enforced.

    // A. Create an unregistered novel candidate in branch workspace
    let cand_code = r#"#!/usr/bin/env python3
import sys, argparse, json, pandas as pd, numpy as np

parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest="cmd", required=True)
fit_p = sub.add_parser("fit")
fit_p.add_argument("--train-path", required=True)
fit_p.add_argument("--model-dir", required=True)
pred_p = sub.add_parser("predict")
pred_p.add_argument("--input-path", required=True)
pred_p.add_argument("--model-dir", required=True)
pred_p.add_argument("--output-path", required=True)
args = parser.parse_args()

if args.cmd == "fit":
    df = pd.read_csv(args.train_path)
    import os
    os.makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, "weights.json"), "w") as f:
        json.dump({"bias": 0.5}, f)
    sys.exit(0)
elif args.cmd == "predict":
    df = pd.read_csv(args.input_path)
    # Assert validation features NEVER leak target column
    if "target" in df.columns:
        sys.exit(99)
    out = pd.DataFrame({
        "row_id": df["row_id"],
        "pred": ["class_1" if x > 2.0 else "class_0" for x in df["feature_a"]],
        "prob_class_0": [0.3 if x > 2.0 else 0.7 for x in df["feature_a"]],
        "prob_class_1": [0.7 if x > 2.0 else 0.3 for x in df["feature_a"]],
    })
    out.to_csv(args.output_path, index=False)
    sys.exit(0)
"#;

    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "novel_algo/candidate.py",
        "content": cand_code,
    }).to_string()).unwrap();

    let branch_dir = BranchWorkspaceManager::get_branch_dir(td.path(), "p-gate", "b-t-gate");
    let cand_dir = branch_dir.join("novel_algo");
    let src_hash = compute_dir_source_hash(&cand_dir);

    let manifest = json!({
        "candidate_id": "cand_novel_gate",
        "name": "Novel Unregistered Algorithm",
        "task_type": "binary",
        "entrypoint": "candidate.py",
        "supported_operations": ["fit", "predict"],
        "stable_row_keys": "row_id",
        "class_order": ["class_0", "class_1"],
        "raw_input_schema": { "target": "target" },
        "source_hash": src_hash,
    });
    orch.dispatch_tool_call("workspace_fs", &json!({
        "action": "write",
        "path": "novel_algo/manifest.json",
        "content": serde_json::to_string(&manifest).unwrap(),
    }).to_string()).unwrap();

    // B. Register dataset
    let mut rows = vec!["row_id,feature_a,target".to_string()];
    for i in 0..60 {
        let fa = (i as f64) * 0.1;
        let target = if fa > 2.0 { "class_1" } else { "class_0" };
        rows.push(format!("{i},{fa:.2},{target}"));
    }
    let ds_file = td.path().join("gate_ds.csv");
    fs::write(&ds_file, rows.join("\n")).unwrap();
    execution::register_dataset(td.path(), "p-gate", "gate_data", &ds_file).unwrap();

    // C. Evaluate unregistered candidate without recipe modifications
    let eval_res = orch.dispatch_tool_call("evaluate_submission", &json!({
        "candidate_id": "cand_novel_gate",
        "candidate_path": "novel_algo",
        "dataset_id": "gate_data",
        "folds": 3,
        "seed": 2026,
    }).to_string()).unwrap();

    let eval_val: Value = serde_json::from_str(&eval_res).unwrap();
    assert_eq!(eval_val["status"], "evaluated");
    assert!(eval_val["primary_metric"].as_f64().unwrap() >= 0.7);

    // D. Verify trusted scorecard authority and tamper immunity
    let reg = ArtifactRegistry::new(td.path());
    let scorecard_ver_id = eval_val["scorecard_version_id"].as_str().unwrap();
    let scorecard_view = reg.get_version("p-gate", scorecard_ver_id).unwrap().unwrap();
    assert_eq!(scorecard_view.artifact_id, "scorecard_cand_novel_gate");

    // E. Verify agent cannot overwrite or forge scorecard
    let forgery = orch.dispatch_tool_call("artifact_crud", &json!({
        "action": "publish",
        "artifact_id": "scorecard_cand_novel_gate",
        "kind": "scorecard",
        "blob_sha256": scorecard_view.blob_sha256,
        "version": "v2",
    }).to_string());
    assert!(forgery.is_err());
    assert!(forgery.unwrap_err().contains("scorecard publication is restricted to the trusted evaluator API"));
}
