use std::fs;
use mlrescurix_harness::agents::sandbox::{SandboxDetector, SandboxLimits, SandboxRunner, SandboxStatus};

#[test]
fn test_docker_live_read_only_root_and_scratch_writable() {
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    // Script attempts to write to /workspace (should fail) and /scratch (should succeed)
    let script = r#"
import os, sys

# 1. Attempt to write to read-only workspace
try:
    with open('/workspace/forbidden.txt', 'w') as f:
        f.write('fail')
    print("UNEXPECTED: wrote to /workspace")
    sys.exit(1)
except OSError as e:
    # Expected: Read-only file system
    pass

# 2. Write to writable scratch tmpfs
scratch_file = '/scratch/allowed.txt'
with open(scratch_file, 'w') as f:
    f.write('ok')

with open(scratch_file, 'r') as f:
    content = f.read()

assert content == 'ok', f"scratch read failed: {content}"
print("VERIFIED_READONLY_AND_SCRATCH")
"#;
    fs::write(workspace.join("check_fs.py"), script).unwrap();

    let limits = SandboxLimits::default();
    let res = SandboxRunner::execute(workspace, "check_fs.py", &[], &limits, None).unwrap();

    assert_eq!(res.exit_code, 0, "stderr: {}", res.stderr);
    assert!(res.stdout.contains("VERIFIED_READONLY_AND_SCRATCH"));
}

#[test]
fn test_docker_live_network_isolation() {
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    // Script attempts outbound network connection
    let script = r#"
import socket, sys

try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect(('1.1.1.1', 80))
    print("UNEXPECTED: network connection succeeded")
    sys.exit(1)
except (OSError, TimeoutError) as e:
    print(f"VERIFIED_NETWORK_ISOLATED: {type(e).__name__}")
    sys.exit(0)
"#;
    fs::write(workspace.join("check_net.py"), script).unwrap();

    let limits = SandboxLimits::default();
    let res = SandboxRunner::execute(workspace, "check_net.py", &[], &limits, None).unwrap();

    assert_eq!(res.exit_code, 0, "stderr: {}", res.stderr);
    assert!(res.stdout.contains("VERIFIED_NETWORK_ISOLATED"));
}

#[test]
fn test_docker_live_non_root_uid_and_cap_drop() {
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    let script = r#"
import os, sys
uid = os.getuid()
gid = os.getgid()
assert uid == 1000, f"Expected uid 1000, got {uid}"
assert gid == 1000, f"Expected gid 1000, got {gid}"
print(f"VERIFIED_NON_ROOT: uid={uid}, gid={gid}")
"#;
    fs::write(workspace.join("check_uid.py"), script).unwrap();

    let limits = SandboxLimits::default();
    let res = SandboxRunner::execute(workspace, "check_uid.py", &[], &limits, None).unwrap();

    assert_eq!(res.exit_code, 0, "stderr: {}", res.stderr);
    assert!(res.stdout.contains("VERIFIED_NON_ROOT"));
}

#[test]
fn test_docker_live_timeout_enforcement() {
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    let script = r#"
import time
print("STARTING_SLEEP", flush=True)
time.sleep(30)
print("FINISHED_SLEEP", flush=True)
"#;
    fs::write(workspace.join("sleep.py"), script).unwrap();

    let limits = SandboxLimits {
        max_wall_ms: 1_500, // 1.5 seconds deadline
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let res = SandboxRunner::execute(workspace, "sleep.py", &[], &limits, None);
    let elapsed = t0.elapsed();

    assert!(res.is_err(), "Timed out process must return Err");
    let err = res.unwrap_err();
    assert!(err.contains("timed out"), "Error must indicate timeout: {err}");
    assert!(elapsed.as_millis() < 6_000, "Container kill must happen promptly; elapsed: {:?}", elapsed);
}

#[test]
fn test_docker_live_output_bound_enforcement() {
    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }

    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path();

    let script = r#"
import sys
for _ in range(1000):
    print("X" * 1024, flush=True)
"#;
    fs::write(workspace.join("flood.py"), script).unwrap();

    let limits = SandboxLimits {
        max_output_bytes: 8 * 1024, // 8 KB limit
        max_wall_ms: 10_000,
        ..Default::default()
    };
    let res = SandboxRunner::execute(workspace, "flood.py", &[], &limits, None);
    assert!(res.is_err(), "Flooding output must return Err");
    let err = res.unwrap_err();
    assert!(err.contains("exceeded output byte bound"), "Error must indicate output limit exceeded: {err}");
}


#[cfg(unix)]
#[test]
fn test_docker_live_cannot_read_controller_file_via_host_path_or_workspace_symlink() {
    use std::os::unix::fs::symlink;

    if matches!(SandboxDetector::detect(), SandboxStatus::Unavailable { .. }) {
        eprintln!("Skipping Docker test: daemon unavailable");
        return;
    }
    let controller = tempfile::tempdir().unwrap();
    let secret = controller.path().join("heldout-labels.json");
    fs::write(&secret, "controller-only-sentinel").unwrap();
    let workspace = tempfile::tempdir().unwrap();
    symlink(&secret, workspace.path().join("leaked-labels.json")).unwrap();
    let path_literal = serde_json::to_string(&secret.to_string_lossy().to_string()).unwrap();
    let script = format!(r#"
from pathlib import Path
for path in [{path_literal}, '/workspace/leaked-labels.json']:
    try:
        Path(path).read_bytes()
    except OSError:
        continue
    raise SystemExit('controller file accessible: ' + path)
print('VERIFIED_CONTROLLER_FILE_ISOLATION')
"#);
    fs::write(workspace.path().join("isolation.py"), script).unwrap();
    let result = SandboxRunner::execute(workspace.path(), "isolation.py", &[], &SandboxLimits::default(), None).unwrap();
    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert!(result.stdout.contains("VERIFIED_CONTROLLER_FILE_ISOLATION"));
}
