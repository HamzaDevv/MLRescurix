use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::Result;

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut bytes=[0u8;65536];
    let mut total=0u64;
    loop {let n=file.read(&mut bytes).map_err(|e|e.to_string())?;if n==0{break;}total+=n as u64;if total>256*1024*1024{return Err("bundle file exceeded bounded verification read".into());}hasher.update(&bytes[..n]);}
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifestFileEntry {
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub format: String,
    pub project_id: String,
    pub candidate_id: String,
    pub files: HashMap<String, BundleManifestFileEntry>,
    #[serde(default)]
    pub model: Option<Value>,
    #[serde(default)]
    pub pipeline: Option<Value>,
    #[serde(default)]
    pub contracts: Option<Value>,
    #[serde(default)]
    pub dependencies: Option<Value>,
    #[serde(default)]
    pub scorecard: Option<Value>,
    #[serde(default)]
    pub lineage: Option<Value>,
    #[serde(default)]
    pub resource_measurements: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVerificationReport {
    pub status: String,
    pub bundle_path: String,
    pub project_id: String,
    pub candidate_id: String,
    pub format: String,
    pub total_bytes: u64,
    pub file_count: usize,
    pub manifest_verified: bool,
    pub contracts_verified: bool,
    pub parity_check: String,
    pub constraints_verified: bool,
    pub details: Value,
    pub verified_at_ms: i64,
}

/// Recursively collect all relative file paths in `dir`. Rejects symlinks.
fn collect_bundle_files(base: &Path, current: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(current).map_err(|e| format!("cannot read dir {}: {e}", current.display()))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let p = entry.path();
        let meta = fs::symlink_metadata(&p).map_err(|e| format!("metadata error {}: {e}", p.display()))?;
        if meta.file_type().is_symlink() {
            return Err(format!("bundle file '{}' is a forbidden symlink", p.display()));
        }
        if meta.is_dir() {
            collect_bundle_files(base, &p, out)?;
        } else if meta.is_file() {
            let rel = p.strip_prefix(base).map_err(|e| e.to_string())?
                .to_string_lossy().to_string();
            out.push(rel);
        } else {
            return Err("bundle contains a forbidden special file".into());
        }
    }
    Ok(())
}

/// Authoritative bundle verification:
/// 1. Anti-tampering & manifest integrity (exact files, sizes, sha256 hashes, no untracked files, no symlinks).
/// 2. Required contract files (schema, scorecard, verification, final, inference, examples).
/// 3. Standalone inference parity & schema enforcement in managed/clean runtime.
/// 4. Constraint verification (latency, bundle size, feasibility).
pub fn verify_bundle(bundle_dir: &Path, run_parity_check: bool) -> Result<BundleVerificationReport> {
    verify_bundle_with_state(bundle_dir, run_parity_check, None)
}

/// Authoritative bundle verification with optional external state trust anchor (`state_root`):
/// 1. Anti-tampering & manifest integrity (exact files, sizes, sha256 hashes, no untracked files, no symlinks).
/// 2. Required contract files (schema, scorecard, verification, final, inference, examples).
/// 3. External state authenticity when `state_root` is provided: cross-checks `final_test_gates`, `managed_artifacts`,
///    `pipeline_receipts`, `content_decisions`, `environments`, and compiled-in `embedded_python_src_hash()`,
///    rejecting any self-rehashed or unapproved bundle.
/// 4. Standalone inference parity, schema enforcement, and container timing/RSS telemetry in the pinned runtime.
/// 5. Constraint verification (latency, bundle size, model size, feasibility).
pub fn verify_bundle_with_state(
    bundle_dir: &Path,
    run_parity_check: bool,
    state_root: Option<&Path>,
) -> Result<BundleVerificationReport> {
    let meta = fs::symlink_metadata(bundle_dir)
        .map_err(|e| format!("bundle path '{}' invalid: {e}", bundle_dir.display()))?;
    if meta.file_type().is_symlink() {
        return Err("bundle directory cannot be a symlink".into());
    }
    if !meta.is_dir() {
        return Err(format!("bundle path '{}' is not a directory", bundle_dir.display()));
    }

    let mut physical_files = Vec::new();
    collect_bundle_files(bundle_dir, bundle_dir, &mut physical_files)?;

    // 1. Load manifest.json
    let manifest_path = bundle_dir.join("manifest.json");
    if !manifest_path.is_file() {
        return Err("bundle missing manifest.json".into());
    }
    if fs::metadata(&manifest_path).map_err(|e|e.to_string())?.len()>1024*1024||physical_files.len()>4096 {
        return Err("bundle manifest/file-count verification ceiling exceeded".into());
    }
    let manifest_bytes = fs::read(&manifest_path).map_err(|e| format!("cannot read manifest.json: {e}"))?;
    let manifest_sha = format!("{:x}", Sha256::digest(&manifest_bytes));
    let manifest: BundleManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("malformed manifest.json: {e}"))?;

    if manifest.format != "mlrescurix.bundle.v1" && manifest.format != "mlrescurix.bundle.v2" {
        return Err(format!("unsupported bundle format: '{}'; expected 'mlrescurix.bundle.v1' or 'mlrescurix.bundle.v2'", manifest.format));
    }
    if manifest.project_id.is_empty() || manifest.candidate_id.is_empty() {
        return Err("manifest.json missing project_id or candidate_id".into());
    }

    // 2. Check required files
    let mut required_contracts = vec![
        "input_schema.json",
        "scorecard.json",
        "verification.json",
        "final.json",
        "inference.py",
        "inference.lock",
        "examples_input.json",
        "examples_expected.json",
        "README.md",
    ];
    if manifest.format == "mlrescurix.bundle.v2" {
        required_contracts.push("output_schema.json");
        required_contracts.push("training.lock");
        required_contracts.push("serve.py");
        if !manifest.files.contains_key("frozen_model.joblib") && !manifest.files.contains_key("weights.safetensors") {
            return Err("bundle manifest missing model weights ('frozen_model.joblib' or 'weights.safetensors')".into());
        }
        let inf_lock_p = bundle_dir.join("inference.lock");
        if inf_lock_p.is_file() {
            let inf_lock_content = fs::read_to_string(&inf_lock_p).map_err(|e| e.to_string())?.to_lowercase();
            for forbidden in &["pytest", "mlrescurix_harness", "cargo", "docker"] {
                if inf_lock_content.contains(forbidden) {
                    return Err(format!("dependency isolation violated: inference.lock contains prohibited tool '{forbidden}'"));
                }
            }
        }
    } else {
        required_contracts.push("frozen_model.joblib");
    }

    for req in &required_contracts {
        if !manifest.files.contains_key(*req) {
            return Err(format!("bundle manifest missing required contract file: '{req}'"));
        }
        let p = bundle_dir.join(req);
        if !p.is_file() {
            return Err(format!("bundle missing required physical contract file: '{req}'"));
        }
    }

    // 3. Manifest integrity & anti-tampering: verify all declared files
    let declared_bytes=manifest.files.values().try_fold(manifest_bytes.len() as u64,|sum,item|sum.checked_add(item.bytes).ok_or("bundle byte count overflow"))?;
    if declared_bytes>256*1024*1024{return Err("bundle exceeds 256 MiB managed verification ceiling".into());}
    let mut total_bytes = 0u64;
    for (rel_path, declared) in &manifest.files {
        if rel_path.contains("..") || rel_path.starts_with('/') || rel_path.starts_with('\\') {
            return Err(format!("illegal relative path in manifest: '{rel_path}'"));
        }
        let full_path = bundle_dir.join(rel_path);
        let m = fs::symlink_metadata(&full_path)
            .map_err(|_| format!("bundle manifest declares '{}' but file does not exist", rel_path))?;
        if m.file_type().is_symlink() {
            return Err(format!("bundle file '{rel_path}' is a forbidden symlink"));
        }
        if !m.is_file(){return Err("bundle manifest must declare regular files".into());}
        if m.len() != declared.bytes {
            return Err(format!(
                "tampered bundle: file '{rel_path}' size mismatch (declared {} bytes, actual {} bytes)",
                declared.bytes, m.len()
            ));
        }
        let actual_sha = hash_file(&full_path)?;
        if actual_sha != declared.sha256 {
            return Err(format!(
                "tampered bundle: file '{rel_path}' sha256 mismatch (declared {}, actual {})",
                declared.sha256, actual_sha
            ));
        }
        total_bytes = total_bytes.checked_add(m.len()).ok_or("bundle byte count overflow")?;
    }

    // 4. Anti-tampering: check that NO untracked files exist in bundle
    let declared_set: HashSet<&str> = manifest.files.keys().map(|s| s.as_str()).collect();
    for phys in &physical_files {
        if phys == "manifest.json" {
            continue;
        }
        if !declared_set.contains(phys.as_str()) {
            return Err(format!("tampered bundle: untracked file '{phys}' found in bundle directory"));
        }
    }

    total_bytes = total_bytes.checked_add(manifest_bytes.len() as u64).ok_or("bundle byte count overflow")?;

    // 5. Contract consistency check
    let verification_json: Value = serde_json::from_slice(
        &fs::read(bundle_dir.join("verification.json")).map_err(|e| e.to_string())?
    ).map_err(|e| format!("invalid verification.json: {e}"))?;

    let (frozen_model_sha, frozen_model_bytes) = if bundle_dir.join("frozen_model.joblib").is_file() {
        let p = bundle_dir.join("frozen_model.joblib");
        let sha = hash_file(&p)?;
        let bytes = fs::metadata(&p).map_err(|e| e.to_string())?.len();
        let declared = verification_json["frozen_model_sha256"].as_str().unwrap_or("");
        if declared != sha {
            return Err(format!(
                "tampered bundle: verification.json frozen_model_sha256 mismatch (declared {}, actual {})",
                declared, sha
            ));
        }
        (sha, bytes)
    } else if bundle_dir.join("weights.safetensors").is_file() {
        let p = bundle_dir.join("weights.safetensors");
        let sha = hash_file(&p)?;
        let bytes = fs::metadata(&p).map_err(|e| e.to_string())?.len();
        let declared = verification_json["weights_safetensors_sha256"].as_str()
            .or_else(|| verification_json["frozen_model_sha256"].as_str())
            .unwrap_or("");
        if !declared.is_empty() && declared != sha {
            return Err(format!(
                "tampered bundle: verification.json weights sha256 mismatch (declared {}, actual {})",
                declared, sha
            ));
        }
        (sha, bytes)
    } else {
        return Err("bundle missing required weights file ('frozen_model.joblib' or 'weights.safetensors')".into());
    };

    let final_json: Value = serde_json::from_slice(
        &fs::read(bundle_dir.join("final.json")).map_err(|e| e.to_string())?
    ).map_err(|e| format!("invalid final.json: {e}"))?;
    let final_sha = hash_file(&bundle_dir.join("final.json"))?;

    let scorecard_json: Value = serde_json::from_slice(
        &fs::read(bundle_dir.join("scorecard.json")).map_err(|e| e.to_string())?
    ).map_err(|e| format!("invalid scorecard.json: {e}"))?;

    if scorecard_json["candidate_id"].as_str() != Some(&manifest.candidate_id) {
        return Err("scorecard candidate_id does not match manifest candidate_id".into());
    }

    // 6. External state authenticity check when `state_root` is provided
    let mut external_binding = Value::Null;
    if let Some(root) = state_root {
        let _rt = crate::storage::Runtime::open(root)?;
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        use rusqlite::OptionalExtension;

        // Verify user approval token
        let has_user_approval: bool = db.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM user_approvals WHERE project=?1 AND target_id=?2 AND action_type IN ('deployment_handoff', 'final_test', 'holdout_access') AND approved=1
                UNION
                SELECT 1 FROM user_approvals WHERE project=?1 AND action_type IN ('deployment_handoff', 'final_test', 'holdout_access') AND approved=1
                UNION
                SELECT 1 FROM approvals WHERE project=?1 AND candidate_id=?2 AND approved=1
            )",
            rusqlite::params![&manifest.project_id, &manifest.candidate_id],
            |r| r.get(0),
        ).unwrap_or(false);
        if !has_user_approval {
            return Err(format!("external authenticity failed: candidate '{}' lacks verified user approval in user_approvals or approvals ledger", manifest.candidate_id));
        }

        let gate_row: Option<(String, String, String, String, String)> = db.query_row(
            "SELECT access_id, dataset_sha256, status, COALESCE(frozen_model_sha256, ''), COALESCE(result_sha256, '')
             FROM final_test_gates WHERE project=?1 AND candidate_id=?2 ORDER BY rowid DESC LIMIT 1",
            rusqlite::params![&manifest.project_id, &manifest.candidate_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        ).optional().map_err(|e| e.to_string())?;

        let (gate_access_id, gate_dataset_sha, gate_status, gate_frozen_sha, gate_result_sha) = gate_row
            .ok_or_else(|| format!(
                "external authenticity failed: no consumed final_test_gates record in state for project '{}' candidate '{}'",
                manifest.project_id, manifest.candidate_id
            ))?;

        if gate_status != "consumed" {
            return Err(format!("external authenticity failed: final_test_gates status is '{gate_status}', expected 'consumed'"));
        }
        if !gate_frozen_sha.is_empty() && gate_frozen_sha != frozen_model_sha {
            return Err(format!(
                "external authenticity failed: bundle weights sha256 ({frozen_model_sha}) does not match authoritative state gate ({gate_frozen_sha})"
            ));
        }
        if gate_result_sha != final_sha {
            return Err(format!(
                "external authenticity failed: bundle final.json sha256 ({final_sha}) does not match authoritative state gate ({gate_result_sha})"
            ));
        }

        // Verify no private evaluation row leakage from sealed holdout test split
        let wf_manifest_path = root.join("workflows").join(&manifest.project_id).join("dataset_manifest.json");
        if wf_manifest_path.is_file() {
            if let Ok(manifest_bytes) = fs::read(&wf_manifest_path) {
                if let Ok(ds_val) = serde_json::from_slice::<Value>(&manifest_bytes) {
                    if let Some(test_indices) = ds_val.get("splits").and_then(|s| s.get("test")).and_then(Value::as_array) {
                        let test_set: HashSet<i64> = test_indices.iter().filter_map(Value::as_i64).collect();
                        if let Ok(examples_val) = serde_json::from_slice::<Value>(&fs::read(bundle_dir.join("examples_input.json")).unwrap_or_default()) {
                            if let Some(ex_arr) = examples_val.as_array() {
                                for row in ex_arr {
                                    if let Some(row_id) = row.get("__row_id__").and_then(Value::as_i64) {
                                        if test_set.contains(&row_id) {
                                            return Err("private evaluation rows leak detected: holdout test rows present in golden examples".into());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if verification_json["test_access_id"].as_str() != Some(gate_access_id.as_str())
            || final_json["test_access_id"].as_str() != Some(gate_access_id.as_str())
        {
            return Err("external authenticity failed: bundle test_access_id does not match authoritative state gate".into());
        }
        if verification_json["dataset_sha256"].as_str() != Some(gate_dataset_sha.as_str())
            || scorecard_json["dataset_sha256"].as_str() != Some(gate_dataset_sha.as_str())
        {
            return Err("external authenticity failed: bundle dataset_sha256 does not match authoritative state gate".into());
        }

        // Cross-check authoritative manifest.json hash recorded in managed_artifacts during finalize
        let auth_manifest: Option<(u64, String)> = db.query_row(
            "SELECT bytes, sha256 FROM managed_artifacts WHERE project=?1 AND name='bundle/manifest.json' ORDER BY rowid DESC LIMIT 1",
            [&manifest.project_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        ).optional().map_err(|e| e.to_string())?;
        let (auth_bytes, auth_sha) = auth_manifest.ok_or("external authenticity failed: authoritative bundle manifest receipt is missing")?;
        {
            if manifest_bytes.len() as u64 != auth_bytes || manifest_sha != auth_sha {
                return Err(format!(
                    "external authenticity failed: bundle manifest.json sha256 ({manifest_sha}) does not match authoritative state record ({auth_sha}); self-rehashed bundle rejected"
                ));
            }
        }

        // Verify packaged training source files match compiled-in trusted worker source
        let expected_src_hash = crate::assets::embedded_python_src_hash();
        let embedded_map: HashMap<&str, &str> = crate::assets::EMBEDDED_PYTHON_FILES.iter().copied().collect();
        let wf_source = bundle_dir.join("training/source/workflow.py");
        if !wf_source.is_file() {
            return Err("external authenticity failed: bundle missing training/source/workflow.py".into());
        }
        for (rel_path, declared) in &manifest.files {
            if let Some(sub) = rel_path.strip_prefix("training/source/") {
                let asset_key = if sub == "workflow.py" {
                    "src/mlrescurix_worker/scientific/workflow.py".to_string()
                } else {
                    format!("src/{sub}")
                };
                let Some(expected_content) = embedded_map.get(asset_key.as_str()) else {
                    return Err(format!("external authenticity failed: unknown training source file '{rel_path}' in bundle"));
                };
                let expected_file_sha = format!("{:x}", Sha256::digest(expected_content.as_bytes()));
                if declared.sha256 != expected_file_sha {
                    return Err(format!(
                        "external authenticity failed: bundled training source '{rel_path}' sha256 ({}) does not match trusted embedded worker source ({expected_file_sha})",
                        declared.sha256
                    ));
                }
            }
        }

        // If candidate is a scheduled pipeline, verify pipeline_receipts, environment identity, and content-bound decisions
        let pipe_row: Option<(String, String, String)> = db.query_row(
            "SELECT source_sha256, environment_identity, dataset_sha256 FROM pipeline_receipts WHERE project=?1 AND candidate=?2",
            rusqlite::params![&manifest.project_id, &manifest.candidate_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        ).optional().map_err(|e| e.to_string())?;

        let mut verified_binding_sha = Value::Null;
        let mut verified_env_identity = Value::Null;
        if let Some((rec_src_sha, rec_env_id, rec_data_sha)) = pipe_row {
            if rec_src_sha != expected_src_hash {
                return Err("external authenticity failed: pipeline_receipts source_sha256 does not match trusted worker source".into());
            }
            if rec_data_sha != gate_dataset_sha {
                return Err("external authenticity failed: pipeline_receipts dataset_sha256 mismatch".into());
            }
            let env_python: Option<String> = db.query_row(
                "SELECT python FROM environments WHERE status='ready' AND python LIKE 'docker:%' AND identity=?1 LIMIT 1",
                [&rec_env_id],
                |r| r.get(0),
            ).optional().map_err(|e| e.to_string())?;
            let Some(env_python) = env_python else {
                return Err("external authenticity failed: pipeline environment_identity is not a ready container environment in state".into());
            };
            let auth_image = env_python.trim_start_matches("docker:");
            if run_parity_check {
                let requested_image = std::env::var("MLRESCURIX_BUNDLE_IMAGE").unwrap_or_default();
                if requested_image != auth_image {
                    return Err(format!(
                        "external authenticity failed: MLRESCURIX_BUNDLE_IMAGE ({requested_image}) does not match authoritative pipeline container image ({auth_image})"
                    ));
                }
            }
            let decisions = crate::execution::require_finalization_decisions(root, &manifest.project_id, &manifest.candidate_id)?;
            verified_binding_sha = decisions["binding_sha256"].clone();
            verified_env_identity = Value::String(rec_env_id);
            let wf_lineage = root.join("workflows").join(&manifest.project_id).join("lineage.json");
            let bundle_lineage = bundle_dir.join("lineage.json");
            if !bundle_lineage.is_file() || !wf_lineage.is_file() || hash_file(&bundle_lineage)? != hash_file(&wf_lineage)? {
                return Err("external authenticity failed: bundle lineage.json does not match authoritative workflow lineage.json".into());
            }
        }

        external_binding = serde_json::json!({
            "state_root": root.to_string_lossy(),
            "test_access_id": gate_access_id,
            "dataset_sha256": gate_dataset_sha,
            "frozen_model_sha256": gate_frozen_sha,
            "final_result_sha256": gate_result_sha,
            "bundle_manifest_sha256": manifest_sha,
            "trusted_source_sha256": expected_src_hash,
            "decision_binding_sha256": verified_binding_sha,
            "environment_identity": verified_env_identity,
        });
    }

    // 7. Constraint re-verification
    let mut constraints_verified = true;
    let mut failed_constraints = Vec::new();
    let mut declared_latency_threshold_ms: Option<f64> = None;
    if let Some(constraints) = scorecard_json.get("constraints").and_then(|c| c.as_object()) {
        for (c_name, c_val) in constraints {
            if c_val.get("threshold").is_some_and(Value::is_null) && c_val["enforcement"] == "unsupported" { continue; }
            if !matches!(c_name.as_str(),"inference_latency_ms"|"bundle_bytes"|"model_bytes"|"memory") && !c_val.get("threshold").is_none_or(Value::is_null) {
                return Err(format!("unsupported required bundle constraint: {c_name}"));
            }
            if c_name=="memory" && !c_val.get("threshold").is_none_or(Value::is_null) {
                return Err("required memory certification is unsupported by example parity".into());
            }
            if c_val.get("verdict").and_then(|v| v.as_str()) != Some("pass") {
                constraints_verified = false;
                failed_constraints.push(format!("{c_name}: verdict is not a verified pass"));
            }
            if c_name == "bundle_bytes" {
                if let Some(threshold) = c_val.get("threshold").and_then(|v| v.as_u64()) {
                    if total_bytes > threshold {
                        constraints_verified = false;
                        failed_constraints.push(format!("bundle_bytes exceeds threshold: {} > {}", total_bytes, threshold));
                    }
                }
            }
            if c_name == "model_bytes" {
                if let Some(threshold) = c_val.get("threshold").and_then(|v| v.as_u64()) {
                    if frozen_model_bytes > threshold {
                        constraints_verified = false;
                        failed_constraints.push(format!("model_bytes exceeds threshold: {} > {}", frozen_model_bytes, threshold));
                    }
                }
            }
            if c_name == "inference_latency_ms" {
                declared_latency_threshold_ms = c_val.get("threshold").and_then(|v| v.as_f64());
            }
        }
    }

    if !constraints_verified {
        return Err(format!("bundle constraint check failed: {}", failed_constraints.join("; ")));
    }

    // 8. Standalone inference parity & sandbox telemetry in the managed container only.
    let (parity_result, sandbox_telemetry) = if run_parity_check {
        let (status, telemetry) = verify_inference_parity(bundle_dir, &manifest_sha, state_root, Some(&manifest.project_id), total_bytes)?;
        if let Some(threshold_ms) = declared_latency_threshold_ms {
            if let Some(warm_p95) = telemetry.get("warm_p95_ms").and_then(|v| v.as_f64()) {
                if warm_p95 > threshold_ms {
                    return Err(format!(
                        "sandbox inference_latency_ms exceeds threshold: {:.3} ms > {:.3} ms",
                        warm_p95, threshold_ms
                    ));
                }
            }
        }
        (status, telemetry)
    } else {
        ("skipped_by_request".to_string(), Value::Null)
    };

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as i64;
    Ok(BundleVerificationReport {
        status: if run_parity_check { "sandbox_examples_verified" } else { "integrity_verified" }.to_string(),
        bundle_path: bundle_dir.to_string_lossy().to_string(),
        project_id: manifest.project_id,
        candidate_id: manifest.candidate_id,
        format: manifest.format,
        total_bytes,
        file_count: manifest.files.len(),
        manifest_verified: true,
        contracts_verified: true,
        parity_check: parity_result,
        constraints_verified,
        details: serde_json::json!({
            "scorecard_metrics": scorecard_json.get("final_test_metrics"),
            "constraints": scorecard_json.get("constraints"),
            "frozen_model_sha256": frozen_model_sha,
            "frozen_model_bytes": frozen_model_bytes,
            "authenticity": if state_root.is_some() {
                "verified_against_authoritative_state"
            } else {
                "not_established_without_external_trust"
            },
            "feasibility": if run_parity_check {
                "sandbox_measured_and_constraints_verified"
            } else {
                "scorecard_claims_not_independently_certified"
            },
            "sandbox_telemetry": sandbox_telemetry,
            "external_binding": external_binding,
            "scope": if state_root.is_some() && run_parity_check {
                "authoritative external state binding and pinned container sandbox parity verified"
            } else {
                "integrity and optional sandbox example parity; not release acceptance"
            },
        }),
        verified_at_ms: now,
    })
}

/// Executes standalone raw-input inference in a clean, isolated subprocess with the bundle mounted.
/// Asserts:
/// 1. Prediction on examples_input.json matches examples_expected.json.
/// 2. Schema rejection: missing feature column raises ValueError.
/// 3. Schema rejection: extra unexpected column raises ValueError.
/// 4. Captures container cold load time, warm p50/p95 latency, and peak RSS.
fn verify_inference_parity(bundle_dir: &Path, manifest_sha: &str, state_root: Option<&Path>, project_id: Option<&str>, total_bytes: u64) -> Result<(String, Value)> {
    let image = std::env::var("MLRESCURIX_BUNDLE_IMAGE").map_err(|_| "Sandbox parity requires MLRESCURIX_BUNDLE_IMAGE with a pinned scientific image; host execution is prohibited".to_string())?;
    let digest = image.rsplit_once("@sha256:").map(|(_, digest)| digest).unwrap_or("");
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("Bundle sandbox image must be pinned by sha256 digest".into());
    }
    let script = r#"
import json, sys, time, resource, importlib.metadata, subprocess
from pathlib import Path

bundle_path = Path(sys.argv[1]).resolve()
# Check the image's complete resolved dependency lock before candidate imports.
# This file is image-owned, never supplied by the bundle.
lock = Path('/opt/mlrescurix/requirements.lock')
if not lock.is_file():
    raise RuntimeError('scientific image lacks its verified dependency lock')
if sys.version_info[:2] != (3, 13):
    raise RuntimeError('scientific image requires CPython 3.13')
for line in lock.read_text().splitlines():
    if not line or line.startswith('#'): continue
    name, version = line.split(' --hash=sha256:')[0].split('==')
    if importlib.metadata.version(name) != version:
        raise RuntimeError('image dependency version mismatch: ' + name)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True, capture_output=True)
for line in (bundle_path / 'inference.lock').read_text().splitlines():
    if not line or line.startswith('#'): continue
    name, version = line.split(' --hash=sha256:')[0].split('==')
    if importlib.metadata.version(name) != version:
        raise RuntimeError('bundle dependency version mismatch: ' + name)
sys.path.insert(0, str(bundle_path))

t_load_start = time.perf_counter_ns()
try:
    import inference
except Exception as e:
    print(f"IMPORT_ERROR: {e}")
    sys.exit(10)
cold_load_ms = float((time.perf_counter_ns() - t_load_start) / 1e6)

inputs = json.loads((bundle_path / "examples_input.json").read_text())
expected = json.loads((bundle_path / "examples_expected.json").read_text())

# 1. Standalone prediction parity
try:
    res = inference.predict(inputs)
except Exception as e:
    print(f"PREDICT_ERROR: {e}")
    sys.exit(11)

actual_pred = res.get("prediction") if isinstance(res, dict) else res
expected_pred = expected.get("prediction") if isinstance(expected, dict) else expected

import numpy as np
schema = json.loads((bundle_path / 'input_schema.json').read_text())
regression = schema.get('task') == 'regression'
matches = actual_pred == expected_pred
if regression and actual_pred is not None and expected_pred is not None:
    a, b = np.asarray(actual_pred, dtype=float), np.asarray(expected_pred, dtype=float)
    matches = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a, b, rtol=1e-9, atol=1e-12)
if actual_pred is None or expected_pred is None or not matches:
    print(f"PARITY_MISMATCH: actual={actual_pred} != expected={expected_pred}")
    sys.exit(12)
if isinstance(expected, dict) and 'probabilities' in expected:
    if not isinstance(res, dict) or res.get('classes') != expected.get('classes'):
        raise RuntimeError('CLASS_ORDER_MISMATCH')
    a, b = np.asarray(res.get('probabilities'), dtype=float), np.asarray(expected['probabilities'], dtype=float)
    if a.shape != b.shape or a.ndim != 2 or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=1e-9,atol=1e-12):
        raise RuntimeError('PROBABILITY_PARITY_MISMATCH')
    if np.any(a < 0) or np.any(a > 1) or not np.allclose(a.sum(axis=1), 1):
        raise RuntimeError('INVALID_PROBABILITIES')

# Measure warm inference latency and peak RSS inside the container
warm_timings = []
for _ in range(5):
    t0 = time.perf_counter_ns()
    inference.predict(inputs)
    warm_timings.append((time.perf_counter_ns() - t0) / 1e6)

# 2. Schema check: missing column must raise ValueError
features = schema.get("features", [])
if not features or not all(isinstance(x, str) for x in features) or not inputs:
    raise RuntimeError('schema parity requires nonempty examples and named features')
if len(features) >= 1:
    col_to_drop = features[0]
    bad_missing = [{k: v for k, v in row.items() if k != col_to_drop} for row in inputs]
    try:
        inference.predict(bad_missing)
        print("SCHEMA_ERROR: missing column was not rejected")
        sys.exit(13)
    except (ValueError, KeyError):
        pass # expected rejection

# 3. Schema check: extra unexpected column must raise ValueError
bad_extra = [{**row, "__UNEXPECTED_COLUMN__": 12345} for row in inputs]
try:
    inference.predict(bad_extra)
    print("SCHEMA_ERROR: unexpected column was not rejected")
    sys.exit(14)
except (ValueError, KeyError):
    pass # expected rejection

peak_rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
print("PARITY_TELEMETRY:" + json.dumps({
    "cold_load_ms": cold_load_ms,
    "warm_p50_ms": float(np.percentile(warm_timings, 50)),
    "warm_p95_ms": float(np.percentile(warm_timings, 95)),
    "peak_rss_kb": peak_rss_kb,
    "batch_size": len(inputs),
    "repetitions": len(warm_timings),
}))
print("PARITY_VERIFIED_OK")
"#;

    if let Some(root) = state_root {
        crate::execution::enforce_state_write_admission(root, None, total_bytes, crate::execution::StorageWriteKind::BundleStaging, project_id)?;
    }
    let stage = if let Some(root) = state_root {
        let _ = fs::create_dir_all(root.join("staging"));
        root.join("staging").join(format!("bundle_verification_{}_{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()))
    } else {
        std::env::temp_dir().join(format!("mlr-bundle-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()))
    };
    fs::create_dir(&stage).map_err(|e| e.to_string())?;
    let result = (|| {
        crate::agents::workspace::BranchWorkspaceManager::copy_dir_recursive(bundle_dir, &stage.join("bundle"))?;
        if hash_file(&stage.join("bundle/manifest.json"))? != manifest_sha {
            return Err("Bundle manifest changed while staging sandbox verification".into());
        }
        verify_bundle(&stage.join("bundle"), false)?;
        fs::write(stage.join("verify.py"), script).map_err(|e| e.to_string())?;
        let result = crate::agents::sandbox::SandboxRunner::execute(
            &stage, "verify.py", &["/workspace/bundle".into()],
            &crate::agents::sandbox::SandboxLimits::default(), Some(&image),
        )?;
        if result.exit_code != 0 || !result.stdout.lines().any(|line| line == "PARITY_VERIFIED_OK") {
            return Err(format!("Sandbox inference parity failed ({}): {} {}", result.exit_code, result.stdout, result.stderr));
        }
        let mut telemetry = result.stdout.lines()
            .find_map(|line| line.strip_prefix("PARITY_TELEMETRY:"))
            .and_then(|json_str| serde_json::from_str::<Value>(json_str).ok())
            .unwrap_or_else(|| serde_json::json!({}));
        if let Some(obj) = telemetry.as_object_mut() {
            obj.insert("image_identity".into(), Value::String(image.clone()));
            obj.insert("image_sha256".into(), Value::String(digest.to_string()));
        }
        Ok(("parity_verified_ok".to_string(), telemetry))
    })();
    let cleanup = fs::remove_dir_all(&stage).map_err(|e| format!("bundle staging cleanup failed: {e}"));
    match result { Ok(value) => { cleanup?; Ok(value) }, Err(error) => { cleanup?; Err(error) } }
}

/// Executes standalone CLI prediction in a clean environment:
/// `python3 inference.py --input <inputs> --output <output>`
pub fn verify_standalone_cli_inference(
    bundle_dir: &Path,
    inputs_path: &Path,
    output_path: &Path,
) -> Result<Value> {
    let script = bundle_dir.join("inference.py");
    if !script.is_file() {
        return Err("bundle missing inference.py".into());
    }
    let mut cmd = std::process::Command::new("python3");
    cmd.arg(&script)
        .arg("--input")
        .arg(inputs_path)
        .arg("--output")
        .arg(output_path)
        .env("PYTHONDONTWRITEBYTECODE", "1");
    let status = cmd.output().map_err(|e| format!("failed to execute inference CLI: {e}"))?;
    if !status.status.success() {
        return Err(format!(
            "inference CLI failed ({}): {}\n{}",
            status.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&status.stdout),
            String::from_utf8_lossy(&status.stderr)
        ));
    }
    let out_bytes = fs::read(output_path).map_err(|e| format!("cannot read CLI output: {e}"))?;
    let val: Value = serde_json::from_slice(&out_bytes).map_err(|e| format!("invalid CLI output json: {e}"))?;
    Ok(val)
}

/// Spawns serve.py on a local loopback port, checks /health and /predict with examples_input.json,
/// verifies schema error rejection, and cleanly shuts down.
pub fn verify_standalone_serving_recipe(
    bundle_dir: &Path,
    port: u16,
) -> Result<Value> {
    let serve_script = bundle_dir.join("serve.py");
    if !serve_script.is_file() {
        return Err("bundle missing serve.py recipe".into());
    }
    let mut child = std::process::Command::new("python3")
        .arg(&serve_script)
        .arg("--port")
        .arg(port.to_string())
        .arg("--bundle-dir")
        .arg(bundle_dir)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .spawn()
        .map_err(|e| format!("failed to spawn serve.py: {e}"))?;

    // Wait briefly for server to bind
    std::thread::sleep(std::time::Duration::from_millis(500));

    let run_check = || -> Result<Value> {
        use std::io::{Read, Write};
        use std::net::TcpStream;

        // 1. Check /health
        let mut stream = TcpStream::connect(("127.0.0.1", port))
            .map_err(|e| format!("cannot connect to serve.py on port {port}: {e}"))?;
        stream.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .map_err(|e| e.to_string())?;
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).map_err(|e| e.to_string())?;
        let resp_str = String::from_utf8_lossy(&resp);
        if !resp_str.contains("200 OK") || !resp_str.contains("\"healthy\"") {
            return Err(format!("serve.py /health check failed: {resp_str}"));
        }

        // 2. Check /predict with examples_input.json
        let ex_in_bytes = fs::read(bundle_dir.join("examples_input.json")).map_err(|e| e.to_string())?;
        let mut stream2 = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        let req2 = format!(
            "POST /predict HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            ex_in_bytes.len(),
            String::from_utf8_lossy(&ex_in_bytes)
        );
        stream2.write_all(req2.as_bytes()).map_err(|e| e.to_string())?;
        let mut resp2 = Vec::new();
        stream2.read_to_end(&mut resp2).map_err(|e| e.to_string())?;
        let resp2_str = String::from_utf8_lossy(&resp2);
        if !resp2_str.contains("200 OK") || !resp2_str.contains("\"ok\"") {
            return Err(format!("serve.py /predict check failed: {resp2_str}"));
        }

        // 3. Check schema rejection with bad body
        let mut stream3 = TcpStream::connect(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        let bad_body = b"[{\"__BAD_COL__\": 123}]";
        let req3 = format!(
            "POST /predict HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            bad_body.len(),
            String::from_utf8_lossy(bad_body)
        );
        stream3.write_all(req3.as_bytes()).map_err(|e| e.to_string())?;
        let mut resp3 = Vec::new();
        stream3.read_to_end(&mut resp3).map_err(|e| e.to_string())?;
        let resp3_str = String::from_utf8_lossy(&resp3);
        if !resp3_str.contains("400 Bad Request") {
            return Err(format!("serve.py schema rejection check failed: {resp3_str}"));
        }

        Ok(serde_json::json!({
            "serving_status": "healthy",
            "health_verified": true,
            "prediction_verified": true,
            "schema_rejection_verified": true,
        }))
    };

    let outcome = run_check();
    let _ = child.kill();
    let _ = child.wait();
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bundle_collection_includes_bytecode_and_rejects_hidden_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("inference.pyc"), b"bytecode").unwrap();
        let mut files = Vec::new();
        collect_bundle_files(dir.path(), dir.path(), &mut files).unwrap();
        assert_eq!(files, vec!["inference.pyc"]);
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("inference.pyc", dir.path().join(".DS_Store")).unwrap();
            assert!(collect_bundle_files(dir.path(), dir.path(), &mut Vec::new()).is_err());
        }
    }
    #[test]
    fn integrity_does_not_certify_execution_or_unknown_constraints() {
        let dir = tempfile::tempdir().unwrap();
        let names = ["input_schema.json", "scorecard.json", "verification.json", "final.json", "frozen_model.joblib", "inference.py", "inference.lock", "examples_input.json", "examples_expected.json", "README.md"];
        for name in names { fs::write(dir.path().join(name), "{}").unwrap(); }
        let model_sha = hash_file(&dir.path().join("frozen_model.joblib")).unwrap();
        fs::write(dir.path().join("verification.json"), serde_json::json!({"frozen_model_sha256":model_sha}).to_string()).unwrap();
        fs::write(dir.path().join("scorecard.json"), r#"{"candidate_id":"c","constraints":{}}"#).unwrap();
        let update_manifest = || {
            let entries: HashMap<String, BundleManifestFileEntry> = names.iter().map(|name| {
                let path = dir.path().join(name);
                (name.to_string(), BundleManifestFileEntry { bytes: fs::metadata(&path).unwrap().len(), sha256: hash_file(&path).unwrap() })
            }).collect();
            let manifest = BundleManifest {
                format: "mlrescurix.bundle.v1".into(),
                project_id: "p".into(),
                candidate_id: "c".into(),
                files: entries,
                model: None, pipeline: None, contracts: None, dependencies: None, scorecard: None, lineage: None, resource_measurements: None,
            };
            fs::write(dir.path().join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        };
        update_manifest();
        let report = verify_bundle(dir.path(), false).unwrap();
        assert_eq!(report.status, "integrity_verified");
        assert_eq!(report.parity_check, "skipped_by_request");
        if std::env::var_os("MLRESCURIX_BUNDLE_IMAGE").is_none() {
            assert!(verify_bundle(dir.path(), true).unwrap_err().contains("host execution is prohibited"));
        }
        fs::write(dir.path().join("scorecard.json"), r#"{"candidate_id":"c","constraints":{"latency":{"verdict":"unverified"}}}"#).unwrap();
        update_manifest();
        assert!(verify_bundle(dir.path(), false).unwrap_err().contains("verified pass"));
    }

    #[test]
    fn v2_bundle_manifest_and_dependency_isolation() {
        let dir = tempfile::tempdir().unwrap();
        let names = [
            "input_schema.json", "output_schema.json", "scorecard.json", "verification.json",
            "final.json", "frozen_model.joblib", "inference.py", "inference.lock", "training.lock",
            "serve.py", "examples_input.json", "examples_expected.json", "README.md"
        ];
        for name in names { fs::write(dir.path().join(name), "{}").unwrap(); }
        fs::write(dir.path().join("inference.lock"), "numpy==2.0\n").unwrap();
        fs::write(dir.path().join("training.lock"), "numpy==2.0\npytest==8.0\n").unwrap();
        let model_sha = hash_file(&dir.path().join("frozen_model.joblib")).unwrap();
        fs::write(dir.path().join("verification.json"), serde_json::json!({"frozen_model_sha256":model_sha}).to_string()).unwrap();
        fs::write(dir.path().join("scorecard.json"), r#"{"candidate_id":"c2","constraints":{}}"#).unwrap();

        let entries: HashMap<String, BundleManifestFileEntry> = names.iter().map(|name| {
            let path = dir.path().join(name);
            (name.to_string(), BundleManifestFileEntry { bytes: fs::metadata(&path).unwrap().len(), sha256: hash_file(&path).unwrap() })
        }).collect();
        let manifest = BundleManifest {
            format: "mlrescurix.bundle.v2".into(),
            project_id: "p2".into(),
            candidate_id: "c2".into(),
            files: entries,
            model: Some(serde_json::json!({"weights_path": "frozen_model.joblib", "weights_sha256": model_sha, "format": "joblib"})),
            pipeline: Some(serde_json::json!({"feature_pipeline": "pipeline.py", "entrypoint": "inference.py"})),
            contracts: Some(serde_json::json!({"cli": "inference.py", "api": "predict(records)", "input_schema": "input_schema.json", "output_schema": "output_schema.json"})),
            dependencies: Some(serde_json::json!({"training_lock": "training.lock", "inference_lock": "inference.lock", "isolated_inference": true})),
            scorecard: Some(serde_json::json!({"path": "scorecard.json", "sha256": hash_file(&dir.path().join("scorecard.json")).unwrap()})),
            lineage: Some(serde_json::json!({"path": "scorecard.json", "sha256": hash_file(&dir.path().join("scorecard.json")).unwrap()})),
            resource_measurements: Some(serde_json::json!({"cold_load_ms": 10.0, "warm_p95_ms": 2.0, "peak_rss_kb": 1024, "package_bytes": 5000})),
        };
        fs::write(dir.path().join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();

        let report = verify_bundle(dir.path(), false).unwrap();
        assert_eq!(report.status, "integrity_verified");
        assert_eq!(report.format, "mlrescurix.bundle.v2");

        // Verify dependency isolation: inference.lock containing pytest must be rejected
        fs::write(dir.path().join("inference.lock"), "numpy==2.0\npytest==8.0\n").unwrap();
        let mut tampered_entries = manifest.files.clone();
        tampered_entries.insert("inference.lock".to_string(), BundleManifestFileEntry {
            bytes: fs::metadata(dir.path().join("inference.lock")).unwrap().len(),
            sha256: hash_file(&dir.path().join("inference.lock")).unwrap(),
        });
        let mut tampered_manifest = manifest.clone();
        tampered_manifest.files = tampered_entries;
        fs::write(dir.path().join("manifest.json"), serde_json::to_vec(&tampered_manifest).unwrap()).unwrap();

        let res = verify_bundle(dir.path(), false);
        assert!(res.is_err(), "Dependency isolation violation must be rejected");
        assert!(res.unwrap_err().contains("dependency isolation violated"));
    }
}
