//! Embedded Python runtime assets for standalone installation and upgrade independence.
//!
//! Embeds the trusted `mlrescurix_worker` package and requirement lockfiles at compile time
//! and materializes/verifies them into `<state_root>/runtime-assets/<source_sha256>/python`
//! on demand so installed binaries never depend on the source repository checkout.

use std::{
    fs,
    path::{Path, PathBuf},
};
use sha2::{Digest, Sha256};
use crate::Result;

pub const EMBEDDED_PYTHON_FILES: &[(&str, &str)] = &[
    ("pyproject.toml", include_str!("../../../python/pyproject.toml")),
    ("scientific-requirements.txt", include_str!("../../../python/scientific-requirements.txt")),
    ("search-requirements.txt", include_str!("../../../python/search-requirements.txt")),
    ("src/mlrescurix_worker/__init__.py", include_str!("../../../python/src/mlrescurix_worker/__init__.py")),
    ("src/mlrescurix_worker/custom/__init__.py", include_str!("../../../python/src/mlrescurix_worker/custom/__init__.py")),
    ("src/mlrescurix_worker/custom/contract.py", include_str!("../../../python/src/mlrescurix_worker/custom/contract.py")),
    ("src/mlrescurix_worker/custom/preflight.py", include_str!("../../../python/src/mlrescurix_worker/custom/preflight.py")),
    ("src/mlrescurix_worker/custom/runner.py", include_str!("../../../python/src/mlrescurix_worker/custom/runner.py")),
    ("src/mlrescurix_worker/protocol/__init__.py", include_str!("../../../python/src/mlrescurix_worker/protocol/__init__.py")),
    ("src/mlrescurix_worker/protocol/frame.schema.json", include_str!("../../../python/src/mlrescurix_worker/protocol/frame.schema.json")),
    ("src/mlrescurix_worker/scientific/__init__.py", include_str!("../../../python/src/mlrescurix_worker/scientific/__init__.py")),
    ("src/mlrescurix_worker/scientific/anomaly_clustering.py", include_str!("../../../python/src/mlrescurix_worker/scientific/anomaly_clustering.py")),
    ("src/mlrescurix_worker/scientific/arrow_adapter.py", include_str!("../../../python/src/mlrescurix_worker/scientific/arrow_adapter.py")),
    ("src/mlrescurix_worker/scientific/autonomous_rescue.py", include_str!("../../../python/src/mlrescurix_worker/scientific/autonomous_rescue.py")),
    ("src/mlrescurix_worker/scientific/benchmarks.py", include_str!("../../../python/src/mlrescurix_worker/scientific/benchmarks.py")),
    ("src/mlrescurix_worker/scientific/bundle.py", include_str!("../../../python/src/mlrescurix_worker/scientific/bundle.py")),
    ("src/mlrescurix_worker/scientific/calibration.py", include_str!("../../../python/src/mlrescurix_worker/scientific/calibration.py")),
    ("src/mlrescurix_worker/scientific/candidate_runner.py", include_str!("../../../python/src/mlrescurix_worker/scientific/candidate_runner.py")),
    ("src/mlrescurix_worker/scientific/candidate_synthesis.py", include_str!("../../../python/src/mlrescurix_worker/scientific/candidate_synthesis.py")),
    ("src/mlrescurix_worker/scientific/cli.py", include_str!("../../../python/src/mlrescurix_worker/scientific/cli.py")),
    ("src/mlrescurix_worker/scientific/dashboard.py", include_str!("../../../python/src/mlrescurix_worker/scientific/dashboard.py")),
    ("src/mlrescurix_worker/scientific/diagnostic_bridge.py", include_str!("../../../python/src/mlrescurix_worker/scientific/diagnostic_bridge.py")),
    ("src/mlrescurix_worker/scientific/ensemble.py", include_str!("../../../python/src/mlrescurix_worker/scientific/ensemble.py")),
    ("src/mlrescurix_worker/scientific/export.py", include_str!("../../../python/src/mlrescurix_worker/scientific/export.py")),
    ("src/mlrescurix_worker/scientific/feasibility.py", include_str!("../../../python/src/mlrescurix_worker/scientific/feasibility.py")),
    ("src/mlrescurix_worker/scientific/features.py", include_str!("../../../python/src/mlrescurix_worker/scientific/features.py")),
    ("src/mlrescurix_worker/scientific/frozen_embeddings.py", include_str!("../../../python/src/mlrescurix_worker/scientific/frozen_embeddings.py")),
    ("src/mlrescurix_worker/scientific/iteration_loop.py", include_str!("../../../python/src/mlrescurix_worker/scientific/iteration_loop.py")),
    ("src/mlrescurix_worker/scientific/pareto_ensemble.py", include_str!("../../../python/src/mlrescurix_worker/scientific/pareto_ensemble.py")),
    ("src/mlrescurix_worker/scientific/scheduled_pipeline.py", include_str!("../../../python/src/mlrescurix_worker/scientific/scheduled_pipeline.py")),
    ("src/mlrescurix_worker/scientific/serve.py", include_str!("../../../python/src/mlrescurix_worker/scientific/serve.py")),
    ("src/mlrescurix_worker/scientific/time_series.py", include_str!("../../../python/src/mlrescurix_worker/scientific/time_series.py")),
    ("src/mlrescurix_worker/scientific/workflow.py", include_str!("../../../python/src/mlrescurix_worker/scientific/workflow.py")),
    ("src/mlrescurix_worker/search/__init__.py", include_str!("../../../python/src/mlrescurix_worker/search/__init__.py")),
    ("src/mlrescurix_worker/search/adapter.py", include_str!("../../../python/src/mlrescurix_worker/search/adapter.py")),
    ("src/mlrescurix_worker/search/cli.py", include_str!("../../../python/src/mlrescurix_worker/search/cli.py")),
];

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Deterministic content digest of the embedded `python/src` tree matching `BranchWorkspaceManager::compute_dir_hash`.
pub fn embedded_python_src_hash() -> String {
    let mut entries: Vec<(String, usize, String)> = EMBEDDED_PYTHON_FILES
        .iter()
        .filter_map(|(rel, content)| {
            rel.strip_prefix("src/").map(|sub| {
                let bytes = content.as_bytes();
                (sub.to_string(), bytes.len(), sha256_hex(bytes))
            })
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hasher = Sha256::new();
    for (rel, len, sha) in entries {
        hasher.update(format!("{rel}:{len}:{sha}\n").as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn asset_directory(root:&Path,directory:&Path)->Result<()> {
    let mut current=root.to_path_buf();
    fs::create_dir_all(root).map_err(|e|e.to_string())?;
    let root_meta=fs::symlink_metadata(root).map_err(|e|e.to_string())?;
    if root_meta.file_type().is_symlink() || !root_meta.is_dir(){return Err("runtime asset root symlink or non-directory rejected".into());}
    for component in directory.strip_prefix(root).map_err(|e|e.to_string())?.components() {
        if !matches!(component,std::path::Component::Normal(_)){return Err("invalid runtime asset path".into());}
        current.push(component);
        match fs::create_dir(&current){Ok(())=>{},Err(e) if e.kind()==std::io::ErrorKind::AlreadyExists=>{},Err(e)=>return Err(e.to_string())}
        let meta=fs::symlink_metadata(&current).map_err(|e|e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_dir(){return Err("runtime asset directory symlink or non-directory rejected".into());}
    }
    Ok(())
}

/// Materializes and verifies the embedded `python/` tree inside `<state_root>/runtime-assets/<source_hash>/python`.
pub fn ensure_python_root(state_root: &Path) -> Result<PathBuf> {
    let src_hash = embedded_python_src_hash();
    let target_root = state_root.join("runtime-assets").join(&src_hash).join("python");
    let approx_bytes: u64 = EMBEDDED_PYTHON_FILES.iter().map(|(_, content)| content.len() as u64).sum();
    if !target_root.exists() {
        crate::execution::enforce_state_write_admission(state_root, None, approx_bytes, crate::execution::StorageWriteKind::RuntimeAsset, None)?;
    }
    asset_directory(state_root,&target_root)?;
    for (rel, content) in EMBEDDED_PYTHON_FILES {
        let dest = target_root.join(rel);
        asset_directory(state_root,dest.parent().ok_or("asset has no parent")?)?;
        if let Ok(meta)=fs::symlink_metadata(&dest) {
            if meta.file_type().is_symlink() || !meta.is_file(){return Err("runtime asset file symlink or non-file rejected".into());}
        }
        let expected_bytes = content.as_bytes();
        let needs_write = match fs::read(&dest) {
            Ok(existing) => existing != expected_bytes,
            Err(_) => true,
        };
        if needs_write {
            crate::execution::enforce_state_write_admission(state_root,None,expected_bytes.len() as u64,crate::execution::StorageWriteKind::RuntimeAsset,None)?;
            use std::io::Write;
            static NEXT_STAGE:std::sync::atomic::AtomicU64=std::sync::atomic::AtomicU64::new(0);
            let sequence=NEXT_STAGE.fetch_add(1,std::sync::atomic::Ordering::Relaxed);
            let tmp=dest.with_file_name(format!(".asset.{}.{}.staging",std::process::id(),sequence));
            let publish=(||->Result<()> {
                let mut stream=fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(|e|e.to_string())?;
                stream.write_all(expected_bytes).map_err(|e|e.to_string())?;
                stream.sync_all().map_err(|e|e.to_string())?;
                fs::rename(&tmp,&dest).map_err(|e|e.to_string())?;
                Ok(())
            })();
            if publish.is_err(){let _=fs::remove_file(&tmp);}
            publish?;
        }
        if fs::read(&dest).map_err(|e|e.to_string())?!=expected_bytes {
            return Err(format!("runtime asset verification failed: {rel}"));
        }
    }
    let actual_src_hash = crate::agents::workspace::BranchWorkspaceManager::compute_dir_hash(&target_root.join("src"))?;
    if actual_src_hash != src_hash {
        return Err(format!(
            "runtime python source integrity check failed: expected {src_hash}, got {actual_src_hash}"
        ));
    }
    Ok(target_root)
}

/// Materializes and returns the verified `<state_root>/runtime-assets/<source_hash>/python/src` directory.
pub fn ensure_python_src(state_root: &Path) -> Result<PathBuf> {
    Ok(ensure_python_root(state_root)?.join("src"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;
    #[test]
    fn runtime_assets_reject_directory_and_file_symlink_escape(){
        let temp=tempfile::tempdir().unwrap();let root=temp.path().join("state");let outside=temp.path().join("outside");
        fs::create_dir(&root).unwrap();fs::create_dir(&outside).unwrap();symlink(&outside,root.join("runtime-assets")).unwrap();
        assert!(ensure_python_root(&root).unwrap_err().contains("symlink"));assert_eq!(fs::read_dir(&outside).unwrap().count(),0);
        fs::remove_file(root.join("runtime-assets")).unwrap();let assets=ensure_python_root(&root).unwrap();
        let target=outside.join("victim");fs::write(&target,"must survive").unwrap();
        let file=assets.join("pyproject.toml");fs::remove_file(&file).unwrap();symlink(&target,&file).unwrap();
        assert!(ensure_python_root(&root).unwrap_err().contains("symlink"));assert_eq!(fs::read_to_string(target).unwrap(),"must survive");
    }
}
