use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use sha2::{Digest, Sha256};

use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn hash_bytes(data: &[u8]) -> String {
    format!("{:x}", Sha256::digest(data))
}

pub struct BranchWorkspaceManager;

impl BranchWorkspaceManager {
    fn checked_path(root: &Path, project: &str, branch: &str, relative: &str) -> Result<PathBuf> {
        crate::contracts::id(project)?;
        crate::contracts::id(branch)?;
        if relative.is_empty() || Path::new(relative).components().any(|c| !matches!(c, std::path::Component::Normal(_))) {
            return Err("File path escapes branch workspace boundary: invalid path component".into());
        }
        let target = Self::get_branch_dir(root, project, branch).join(relative);
        let mut current = root.to_path_buf();
        for component in target.strip_prefix(root).map_err(|e| e.to_string())?.components() {
            current.push(component);
            match fs::symlink_metadata(&current) {
                Ok(meta) if meta.file_type().is_symlink() => return Err("File path escapes branch workspace boundary: symlink rejected".into()),
                Ok(_) => {},
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {},
                Err(e) => return Err(e.to_string()),
            }
        }
        Ok(target)
    }
    pub fn get_branch_dir(root: &Path, project_id: &str, branch_id: &str) -> PathBuf {
        root.join("workflows").join(project_id).join("branches").join(branch_id)
    }

    pub fn init_branch(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        base_commit: &str,
    ) -> Result<PathBuf> {
        crate::execution::migrate(root)?;
        let branch_dir = Self::get_branch_dir(root, project_id, branch_id);
        fs::create_dir_all(&branch_dir).map_err(|e| e.to_string())?;

        let now = now_ms();
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "INSERT INTO branch_workspaces(project, branch_id, status, base_commit, current_commit, head_sha256, dirty_files, created_ms, updated_ms)
             VALUES(?1, ?2, 'active', ?3, ?3, 'empty', '[]', ?4, ?4)
             ON CONFLICT(project, branch_id) DO UPDATE SET updated_ms=excluded.updated_ms",
            rusqlite::params![project_id, branch_id, base_commit, now],
        ).map_err(|e| e.to_string())?;

        Ok(branch_dir)
    }

    pub fn write_file(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        relative_path: &str,
        content: &str,
    ) -> Result<String> {
        for comp in Path::new(relative_path).components() {
            match comp {
                std::path::Component::Normal(_) => {}
                _ => return Err("File path escapes branch workspace boundary: invalid path component".into()),
            }
        }
        let target_path = Self::checked_path(root, project_id, branch_id, relative_path)?;

        if let Some(parent) = target_path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        fs::write(&target_path, content.as_bytes()).map_err(|e| e.to_string())?;
        let file_hash = hash_bytes(content.as_bytes());

        Self::update_branch_head(root, project_id, branch_id)?;
        Ok(file_hash)
    }

    pub fn read_file(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        relative_path: &str,
    ) -> Result<String> {
        for comp in Path::new(relative_path).components() {
            match comp {
                std::path::Component::Normal(_) => {}
                _ => return Err("File path escapes branch workspace boundary: invalid path component".into()),
            }
        }
        let branch_dir = Self::get_branch_dir(root, project_id, branch_id);
        let target_path = Self::checked_path(root, project_id, branch_id, relative_path)?;

        if !target_path.starts_with(&branch_dir) {
            return Err("File path escapes branch workspace boundary".into());
        }

        fs::read_to_string(&target_path).map_err(|e| e.to_string())
    }

    pub fn patch_file(
        root: &Path,
        project_id: &str,
        branch_id: &str,
        relative_path: &str,
        target_content: &str,
        replacement_content: &str,
    ) -> Result<()> {
        let current = Self::read_file(root, project_id, branch_id, relative_path)?;
        if !current.contains(target_content) {
            return Err(format!("Target content to replace not found in {relative_path}"));
        }
        let updated = current.replacen(target_content, replacement_content, 1);
        Self::write_file(root, project_id, branch_id, relative_path, &updated)?;
        Ok(())
    }

    pub fn list_files(
        root: &Path,
        project_id: &str,
        branch_id: &str,
    ) -> Result<Vec<(String, u64, String)>> {
        let branch_dir = Self::get_branch_dir(root, project_id, branch_id);
        if !branch_dir.exists() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();
        Self::collect_files(&branch_dir, &branch_dir, &mut results)?;
        results.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(results)
    }

    fn collect_files(
        base: &Path,
        dir: &Path,
        results: &mut Vec<(String, u64, String)>,
    ) -> Result<()> {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if entry.file_type().map_err(|e| e.to_string())?.is_symlink() {
                return Err("Workspace symlink rejected".into());
            }
            if path.is_dir() {
                Self::collect_files(base, &path, results)?;
            } else if path.is_file() {
                let rel = path.strip_prefix(base).map_err(|e| e.to_string())?.to_string_lossy().to_string();
                let bytes = fs::read(&path).map_err(|e| e.to_string())?;
                let sha = hash_bytes(&bytes);
                results.push((rel, bytes.len() as u64, sha));
            }
        }
        Ok(())
    }

    pub fn compute_branch_hash(
        root: &Path,
        project_id: &str,
        branch_id: &str,
    ) -> Result<String> {
        let files = Self::list_files(root, project_id, branch_id)?;
        let mut hasher = Sha256::new();
        for (rel, len, sha) in files {
            hasher.update(format!("{rel}:{len}:{sha}\n").as_bytes());
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    pub fn update_branch_head(root: &Path, project_id: &str, branch_id: &str) -> Result<String> {
        let head_hash = Self::compute_branch_hash(root, project_id, branch_id)?;
        let now = now_ms();
        let db = rusqlite::Connection::open(root.join("state.sqlite")).map_err(|e| e.to_string())?;
        db.execute(
            "UPDATE branch_workspaces SET head_sha256=?1, updated_ms=?2 WHERE project=?3 AND branch_id=?4",
            rusqlite::params![head_hash, now, project_id, branch_id],
        ).map_err(|e| e.to_string())?;
        Ok(head_hash)
    }

    /// Export an immutable snapshot of a branch to enable safe cross-branch reuse without coupling workspaces.
    pub fn export_immutable_snapshot(
        root: &Path,
        project_id: &str,
        branch_id: &str,
    ) -> Result<(String, PathBuf)> {
        let head_hash = Self::compute_branch_hash(root, project_id, branch_id)?;
        let snapshot_dir = root.join("workflows").join(project_id).join("snapshots").join(&head_hash);
        if !snapshot_dir.exists() {
            fs::create_dir_all(&snapshot_dir).map_err(|e| e.to_string())?;
            let branch_dir = Self::get_branch_dir(root, project_id, branch_id);
            let files = Self::list_files(root, project_id, branch_id)?;
            for (rel, _, _) in files {
                let src = branch_dir.join(&rel);
                let dst = snapshot_dir.join(&rel);
                if let Some(parent) = dst.parent() {
                    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
                }
                fs::copy(&src, &dst).map_err(|e| e.to_string())?;
            }
        }
        Ok((head_hash, snapshot_dir))
    }

    /// Import files from an immutable snapshot into a target branch.
    pub fn import_snapshot(
        root: &Path,
        project_id: &str,
        target_branch_id: &str,
        snapshot_hash: &str,
    ) -> Result<()> {
        let snapshot_dir = root.join("workflows").join(project_id).join("snapshots").join(snapshot_hash);
        if !snapshot_dir.exists() {
            return Err(format!("Snapshot {snapshot_hash} does not exist"));
        }

        let target_dir = Self::get_branch_dir(root, project_id, target_branch_id);
        fs::create_dir_all(&target_dir).map_err(|e| e.to_string())?;

        let mut files = Vec::new();
        Self::collect_files(&snapshot_dir, &snapshot_dir, &mut files)?;
        for (rel, _, _) in files {
            let src = snapshot_dir.join(&rel);
            let dst = target_dir.join(&rel);
            if let Some(parent) = dst.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::copy(&src, &dst).map_err(|e| e.to_string())?;
        }

        Self::update_branch_head(root, project_id, target_branch_id)?;
        Ok(())
    }
}
