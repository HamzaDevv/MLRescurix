use std::{
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use crate::Result;

fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis().min(i64::MAX as u128) as i64
}

fn validate_sha256(s: &str) -> Result<()> {
    if s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(format!("invalid SHA-256 hex digest: '{s}'"));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobReceipt {
    pub sha256: String,
    pub bytes: u64,
    pub media_type: String,
    pub path: PathBuf,
    pub created_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestFileEntry {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DirectoryManifest {
    pub manifest_version: u32,
    pub kind: String,
    pub total_bytes: u64,
    pub file_count: usize,
    pub files: Vec<ManifestFileEntry>,
}

#[derive(Clone)]
pub struct ContentAddressableStorage {
    root: PathBuf,
}

impl ContentAddressableStorage {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn cas_dir(&self) -> PathBuf {
        self.root.join("cas")
    }

    pub fn staging_dir(&self) -> PathBuf {
        self.cas_dir().join(".staging")
    }

    pub fn bucket_dir(&self, sha256: &str) -> Result<PathBuf> {
        validate_sha256(sha256)?;
        Ok(self.cas_dir().join(&sha256[0..2]))
    }

    pub fn blob_path(&self, sha256: &str) -> Result<PathBuf> {
        let bucket = self.bucket_dir(sha256)?;
        Ok(bucket.join(sha256))
    }

    pub fn has_blob(&self, sha256: &str) -> bool {
        if let Ok(path) = self.blob_path(sha256) {
            if let Ok(meta) = fs::symlink_metadata(&path) {
                return !meta.file_type().is_symlink() && meta.is_file();
            }
        }
        false
    }

    pub fn read_blob(&self, sha256: &str) -> Result<Vec<u8>> {
        let path = self.blob_path(sha256)?;
        let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err(format!("CAS blob '{sha256}' is not a regular file"));
        }
        let bytes = fs::read(&path).map_err(|e| e.to_string())?;
        let computed = format!("{:x}", Sha256::digest(&bytes));
        if computed != sha256 {
            return Err(format!("CAS blob corruption: expected '{sha256}', found '{computed}'"));
        }
        Ok(bytes)
    }

    pub fn store_blob(&self, bytes: &[u8], media_type: &str, project_id: Option<&str>) -> Result<BlobReceipt> {
        let sha256 = format!("{:x}", Sha256::digest(bytes));
        let path = self.blob_path(&sha256)?;

        // Deduplication check: if blob already exists and is uncorrupted, reuse without re-allocating
        if self.has_blob(&sha256) {
            if let Ok(meta) = fs::metadata(&path) {
                if meta.len() == bytes.len() as u64 {
                    return Ok(BlobReceipt {
                        sha256,
                        bytes: bytes.len() as u64,
                        media_type: media_type.to_string(),
                        path,
                        created_ms: now_ms(),
                    });
                }
            }
        }

        // Pre-admission reservation accounting against the 128 MiB policy ceiling
        crate::execution::enforce_state_write_admission(
            &self.root,
            None,
            bytes.len() as u64,
            crate::execution::StorageWriteKind::ManagedBlob,
            project_id,
        )?;

        let staging_root = self.staging_dir();
        fs::create_dir_all(&staging_root).map_err(|e| e.to_string())?;

        let nonce = format!("{}_{}_{}.tmp", sha256, std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos());
        let staging_path = staging_root.join(nonce);

        // Write content to staging file
        fs::write(&staging_path, bytes).map_err(|e| e.to_string())?;

        // Fsync staging file
        let file = fs::File::open(&staging_path).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);

        // Set immutable read-only permissions
        let _ = fs::set_permissions(&staging_path, fs::Permissions::from_mode(0o444));

        // Ensure bucket directory exists
        let bucket = self.bucket_dir(&sha256)?;
        fs::create_dir_all(&bucket).map_err(|e| e.to_string())?;

        // Atomic publish to CAS destination
        if let Err(e) = fs::rename(&staging_path, &path) {
            // If destination already exists (concurrent write), clean staging and confirm
            if path.exists() {
                let _ = fs::remove_file(&staging_path);
            } else {
                let _ = fs::remove_file(&staging_path);
                return Err(format!("failed to atomically publish CAS blob: {e}"));
            }
        }

        // Fsync parent bucket directory
        if let Ok(parent_dir) = fs::File::open(&bucket) {
            let _ = parent_dir.sync_all();
        }

        Ok(BlobReceipt {
            sha256,
            bytes: bytes.len() as u64,
            media_type: media_type.to_string(),
            path,
            created_ms: now_ms(),
        })
    }

    pub fn store_file(&self, file_path: &Path, media_type: &str, project_id: Option<&str>) -> Result<BlobReceipt> {
        let meta = fs::symlink_metadata(file_path).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_file() {
            return Err("cannot store symlink or non-file into CAS".into());
        }
        let bytes = fs::read(file_path).map_err(|e| e.to_string())?;
        self.store_blob(&bytes, media_type, project_id)
    }

    pub fn create_directory_manifest(&self, dir: &Path, max_bytes: u64, project_id: Option<&str>) -> Result<(BlobReceipt, DirectoryManifest)> {
        let meta = fs::symlink_metadata(dir).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() || !meta.is_dir() {
            return Err("directory manifest target must be a real directory".into());
        }

        let mut files = Vec::new();
        let mut total_bytes: u64 = 0;

        fn collect_files(base: &Path, current: &Path, files: &mut Vec<ManifestFileEntry>, total: &mut u64, max_bytes: u64) -> Result<()> {
            for entry in fs::read_dir(current).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                let path = entry.path();
                let m = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
                if m.file_type().is_symlink() {
                    return Err(format!("symlink rejected in directory artifact: {:?}", path));
                }
                if m.is_dir() {
                    collect_files(base, &path, files, total, max_bytes)?;
                } else if m.is_file() {
                    let rel = path.strip_prefix(base).map_err(|e| e.to_string())?;
                    let rel_str = rel.to_string_lossy().into_owned();
                    if rel_str.contains('\\') || rel_str.starts_with('/') || rel_str.contains("..") {
                        return Err(format!("invalid relative path in manifest: '{rel_str}'"));
                    }
                    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
                    *total = total.checked_add(bytes.len() as u64).ok_or("directory artifact size overflow")?;
                    if *total > max_bytes {
                        return Err(format!("directory artifact exceeds max output bound of {max_bytes} bytes"));
                    }
                    let hash = format!("{:x}", Sha256::digest(&bytes));
                    files.push(ManifestFileEntry {
                        path: rel_str,
                        bytes: bytes.len() as u64,
                        sha256: hash,
                    });
                }
            }
            Ok(())
        }

        collect_files(dir, dir, &mut files, &mut total_bytes, max_bytes)?;
        if files.len() > 1024 {
            return Err("directory artifact exceeds maximum file count of 1024".into());
        }

        // Deterministic lexicographic sort by relative path
        files.sort_by(|a, b| a.path.cmp(&b.path));

        let manifest = DirectoryManifest {
            manifest_version: 1,
            kind: "directory_manifest".into(),
            total_bytes,
            file_count: files.len(),
            files,
        };

        let manifest_bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
        let receipt = self.store_blob(&manifest_bytes, "application/x-directory-manifest+json", project_id)?;

        Ok((receipt, manifest))
    }

    pub fn sweep_staging(&self, max_age: Duration) -> Result<u64> {
        let staging = self.staging_dir();
        if !staging.exists() {
            return Ok(0);
        }
        let now = SystemTime::now();
        let mut reclaimed: u64 = 0;
        for entry in fs::read_dir(&staging).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if let Ok(meta) = fs::symlink_metadata(&path) {
                if meta.file_type().is_symlink() || !meta.is_file() {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                let modified = meta.modified().unwrap_or(now);
                if now.duration_since(modified).unwrap_or_default() >= max_age {
                    let size = meta.len();
                    if fs::remove_file(&path).is_ok() {
                        reclaimed = reclaimed.saturating_add(size);
                    }
                }
            }
        }
        Ok(reclaimed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cas_blob_store_and_read() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("state");
        fs::create_dir_all(&root).unwrap();

        let cas = ContentAddressableStorage::new(&root);
        let data = b"hello artifact registry cas";
        let expected_sha = format!("{:x}", Sha256::digest(data));

        let receipt = cas.store_blob(data, "text/plain", None).unwrap();
        assert_eq!(receipt.sha256, expected_sha);
        assert_eq!(receipt.bytes, data.len() as u64);
        assert!(cas.has_blob(&expected_sha));

        let read_back = cas.read_blob(&expected_sha).unwrap();
        assert_eq!(read_back, data);

        // Deduplication returns existing without errors
        let receipt2 = cas.store_blob(data, "text/plain", None).unwrap();
        assert_eq!(receipt2.sha256, expected_sha);
    }

    #[test]
    fn test_cas_directory_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("state");
        let work = temp.path().join("bundle");
        fs::create_dir_all(work.join("sub")).unwrap();
        fs::write(work.join("b.txt"), b"file b").unwrap();
        fs::write(work.join("a.txt"), b"file a").unwrap();
        fs::write(work.join("sub").join("c.txt"), b"file c").unwrap();

        let cas = ContentAddressableStorage::new(&root);
        let (receipt, manifest) = cas.create_directory_manifest(&work, 1024 * 1024, None).unwrap();

        assert_eq!(manifest.file_count, 3);
        assert_eq!(manifest.files[0].path, "a.txt");
        assert_eq!(manifest.files[1].path, "b.txt");
        assert_eq!(manifest.files[2].path, "sub/c.txt");
        assert_eq!(receipt.media_type, "application/x-directory-manifest+json");
    }
}
