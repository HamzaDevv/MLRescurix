//! Integration tests for Warm Worker Container Pool (P9.7) and Platform Matrix Memory Certification (Gate 5).
//!
//! Validates:
//! 1. Sub-100ms task dispatch via pre-warmed idle worker containers.
//! 2. Negative guard: clean state isolation (zero state/artifact leakage across worker reuse).
//! 3. Negative guard: pool exhaustion and recovery under concurrent job load.
//! 4. Platform matrix memory certification: cgroup v2 peak receipts and structured OOM diagnostics.
//! 5. Graceful shutdown and container reaping on harness exit.

use std::{fs, sync::Arc};

use mlrescurix_harness::pool::{
    WarmPoolConfig, WarmPoolMode, WarmTaskSpec, WarmWorkerPool,
};

#[test]
fn test_warm_worker_pool_creation_and_sub_100ms_dispatch() {
    let temp = tempfile::tempdir().unwrap();
    let state_root = temp.path().join("state");
    fs::create_dir_all(&state_root).unwrap();

    // Use LiveDocker if daemon is online, otherwise Simulated
    let mode = if WarmWorkerPool::is_docker_available() {
        WarmPoolMode::LiveDocker
    } else {
        WarmPoolMode::Simulated
    };

    let config = WarmPoolConfig {
        pool_size: 2,
        mode,
        ..Default::default()
    };

    let pool = WarmWorkerPool::new(&state_root, config).unwrap();
    assert_eq!(pool.slot_count(), 2);
    assert_eq!(pool.busy_count(), 0);

    let task = WarmTaskSpec {
        task_id: "task-warm-fast-1".into(),
        candidate_id: "cand-fast-1".into(),
        argv: vec![
            "python".into(),
            "-c".into(),
            "import sys; print('warm_worker_dispatched_instantly')".into(),
        ],
        files: vec![("candidate_code.py".into(), b"# candidate payload\n".to_vec())],
        timeout_ms: 10_000,
        memory_limit_bytes: Some(1024 * 1024 * 1024),
        require_memory_certification: true,
    };

    let result = pool.dispatch(task).unwrap();
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("warm_worker_dispatched_instantly"));
    assert!(
        result.dispatch_latency_ms < 100,
        "dispatch latency must be <100ms, actual was {}ms",
        result.dispatch_latency_ms
    );
    assert!(result.certified);
    assert_eq!(pool.busy_count(), 0);
}

#[test]
fn test_warm_worker_clean_state_isolation_negative_guard() {
    let temp = tempfile::tempdir().unwrap();
    let state_root = temp.path().join("state");
    fs::create_dir_all(&state_root).unwrap();

    let mode = if WarmWorkerPool::is_docker_available() {
        WarmPoolMode::LiveDocker
    } else {
        WarmPoolMode::Simulated
    };

    // Pool size 1 strictly forces the second task to execute in the EXACT same worker slot
    let config = WarmPoolConfig {
        pool_size: 1,
        mode,
        ..Default::default()
    };

    let pool = WarmWorkerPool::new(&state_root, config).unwrap();
    assert_eq!(pool.slot_count(), 1);

    // Task 1: Write a secret canary file in /workflow and /tmp
    let task1 = WarmTaskSpec {
        task_id: "task-isolation-1".into(),
        candidate_id: "cand-leak-producer".into(),
        argv: vec![
            "python".into(),
            "-c".into(),
            r#"import os
with open('workflow_leak_canary.txt', 'w') as f:
    f.write('confidential_candidate_features_and_weights')
with open('/tmp/tmpfs_leak_canary.txt', 'w') as f:
    f.write('transient_secret_tokens')
print('task1_leaks_planted')
"#.into(),
        ],
        files: vec![("model_arch.py".into(), b"# model arch v1\n".to_vec())],
        timeout_ms: 10_000,
        memory_limit_bytes: None,
        require_memory_certification: false,
    };

    let result1 = pool.dispatch(task1).unwrap();
    assert_eq!(result1.exit_code, 0);
    assert!(result1.stdout.contains("task1_leaks_planted"));

    // Task 2: Executed in the same slot. Verifies zero leakage:
    // Neither 'workflow_leak_canary.txt', 'model_arch.py', nor '/tmp/tmpfs_leak_canary.txt' must exist!
    let task2 = WarmTaskSpec {
        task_id: "task-isolation-2".into(),
        candidate_id: "cand-leak-verifier".into(),
        argv: vec![
            "python".into(),
            "-c".into(),
            r#"import os
assert not os.path.exists('workflow_leak_canary.txt'), 'LEAK DETECTED: prior candidate file survived in /workflow'
assert not os.path.exists('model_arch.py'), 'LEAK DETECTED: prior staged file survived in /workflow'
assert not os.path.exists('/tmp/tmpfs_leak_canary.txt'), 'LEAK DETECTED: prior tempfile survived in /tmp'
print('clean_isolation_strictly_verified')
"#.into(),
        ],
        files: vec![("fresh_candidate.py".into(), b"# fresh candidate\n".to_vec())],
        timeout_ms: 10_000,
        memory_limit_bytes: None,
        require_memory_certification: false,
    };

    let result2 = pool.dispatch(task2).unwrap();
    assert_eq!(
        result2.exit_code, 0,
        "negative isolation guard failed: stderr={}",
        result2.stderr
    );
    assert!(result2.stdout.contains("clean_isolation_strictly_verified"));
}

#[test]
fn test_warm_worker_pool_exhaustion_and_recovery_negative_guard() {
    let temp = tempfile::tempdir().unwrap();
    let state_root = temp.path().join("state");
    fs::create_dir_all(&state_root).unwrap();

    let mode = if WarmWorkerPool::is_docker_available() {
        WarmPoolMode::LiveDocker
    } else {
        WarmPoolMode::Simulated
    };

    // Exactly 2 slots
    let config = WarmPoolConfig {
        pool_size: 2,
        mode,
        ..Default::default()
    };

    let pool = Arc::new(WarmWorkerPool::new(&state_root, config).unwrap());
    assert_eq!(pool.slot_count(), 2);

    let mut handles = Vec::new();

    // Spawn 4 concurrent tasks against a 2-slot pool
    for i in 0..4 {
        let pool_ref = pool.clone();
        let handle = std::thread::spawn(move || {
            let task = WarmTaskSpec {
                task_id: format!("concurrent-load-{i}"),
                candidate_id: format!("cand-concurrent-{i}"),
                argv: vec![
                    "python".into(),
                    "-c".into(),
                    r#"import time
time.sleep(0.1)
print('worker_completed_successfully')
"#.into(),
                ],
                files: vec![],
                timeout_ms: 15_000,
                memory_limit_bytes: None,
                require_memory_certification: false,
            };
            pool_ref.dispatch(task)
        });
        handles.push(handle);
    }

    let mut successful_jobs = 0;
    for handle in handles {
        let res = handle.join().unwrap();
        assert!(res.is_ok(), "task dispatch failed: {:?}", res.err());
        let task_res = res.unwrap();
        assert_eq!(task_res.exit_code, 0);
        assert!(task_res.stdout.contains("worker_completed_successfully"));
        successful_jobs += 1;
    }

    assert_eq!(successful_jobs, 4);
    assert_eq!(pool.busy_count(), 0);
}

#[test]
fn test_cgroup_v2_memory_certification_and_diagnostic_bridge_oom() {
    let temp = tempfile::tempdir().unwrap();
    let state_root = temp.path().join("state");
    fs::create_dir_all(&state_root).unwrap();

    let mode = if WarmWorkerPool::is_docker_available() {
        WarmPoolMode::LiveDocker
    } else {
        WarmPoolMode::Simulated
    };

    let memory_limit = 64 * 1024 * 1024; // 64 MiB strict cap

    let config = WarmPoolConfig {
        pool_size: 1,
        memory_limit_bytes: memory_limit,
        mode,
        require_memory_certification: true,
        ..Default::default()
    };

    let pool = WarmWorkerPool::new(&state_root, config).unwrap();

    // Task intentionally allocates 128 MiB (exceeding 64 MiB limit)
    let oom_task = WarmTaskSpec {
        task_id: "task-oom-trigger".into(),
        candidate_id: "cand-oversized-memory".into(),
        argv: vec![
            "python".into(),
            "-c".into(),
            r#"# Exceed 64 MiB cap by allocating 128 MiB
data = bytearray(128 * 1024 * 1024)
print('should_not_reach_here')
"#.into(),
        ],
        files: vec![],
        timeout_ms: 10_000,
        memory_limit_bytes: Some(memory_limit),
        require_memory_certification: true,
    };

    let result = pool.dispatch(oom_task).unwrap();

    // Cgroup v2 must detect OOM
    assert!(result.cgroup_oom, "cgroup_oom flag must be true");
    assert!(result.exit_code == 137 || result.exit_code == -9);

    // DiagnosticReceipt must be generated via DiagnosticBridge
    assert!(result.diagnostic_receipt.is_some());
    let diag = result.diagnostic_receipt.unwrap();
    assert_eq!(diag.failure_type, "cgroup_oom");
    assert_eq!(diag.resource_telemetry.exit_code, Some(137));
    assert_eq!(diag.resource_telemetry.oom_killed, Some(true));
    assert_eq!(diag.resource_telemetry.memory_limit_bytes, Some(memory_limit as i64));
    assert!(!diag.remediation_suggestions.is_empty());
    assert!(diag.remediation_suggestions.iter().any(|s| s.contains("batch size") || s.contains("downcast")));

    // Gate 5 Platform Memory Certification: cgroup_receipt supported must be true
    assert!(result.cgroup_receipt.is_some());
    let cgroup = result.cgroup_receipt.unwrap();
    assert!(cgroup.supported);
    assert!(cgroup.oom_killed);

    // Verify pool is still healthy after OOM recovery
    let recovery_task = WarmTaskSpec {
        task_id: "task-recovery-after-oom".into(),
        candidate_id: "cand-recovery".into(),
        argv: vec![
            "python".into(),
            "-c".into(),
            "print('recovered_and_healthy')".into(),
        ],
        files: vec![],
        timeout_ms: 5000,
        memory_limit_bytes: Some(memory_limit),
        require_memory_certification: true,
    };

    let recovery_result = pool.dispatch(recovery_task).unwrap();
    assert_eq!(recovery_result.exit_code, 0);
    assert!(recovery_result.stdout.contains("recovered_and_healthy"));
    assert!(!recovery_result.cgroup_oom);
}

#[test]
fn test_warm_worker_graceful_shutdown_and_reaping() {
    let temp = tempfile::tempdir().unwrap();
    let state_root = temp.path().join("state");
    fs::create_dir_all(&state_root).unwrap();

    let mode = if WarmWorkerPool::is_docker_available() {
        WarmPoolMode::LiveDocker
    } else {
        WarmPoolMode::Simulated
    };

    let config = WarmPoolConfig {
        pool_size: 2,
        mode,
        ..Default::default()
    };

    let pool = WarmWorkerPool::new(&state_root, config).unwrap();
    assert_eq!(pool.slot_count(), 2);

    let slot_dirs: Vec<_> = pool.slots.iter().map(|s| s.slot_dir.clone()).collect();
    for d in &slot_dirs {
        assert!(d.exists(), "slot directory should exist while pool is active");
    }

    // Explicit shutdown
    pool.shutdown();
    assert!(pool.shutting_down.load(std::sync::atomic::Ordering::Acquire));

    // After shutdown, slot directories are cleaned
    for d in &slot_dirs {
        assert!(!d.exists(), "slot directory should be cleaned after shutdown");
    }
}
