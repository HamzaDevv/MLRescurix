use mlrescurix_harness::campaign::compute_workspace_source_sha256;
use std::fs;

#[test]
fn nested_inference_and_serve_modules_are_bound_to_source_identity() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("solution.py"), "import sub.inference\n").unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();
    for name in ["inference.py", "serve.py"] {
        let path = dir.path().join("sub").join(name);
        fs::write(&path, "value = 1\n").unwrap();
        let before = compute_workspace_source_sha256(dir.path()).unwrap();
        fs::write(&path, "value = 2\n").unwrap();
        let after = compute_workspace_source_sha256(dir.path()).unwrap();
        assert_ne!(before, after, "{name} must be included in the source identity");
    }
}
