# P3.1 Capability Discovery, Headroom, Presets, and Infeasible-Request Rejection Evidence

Session date: 2026-10-14 (Asia/Kolkata)
Branch: `codex/p1-cgroup-orphan-recovery`
Baseline HEAD: `9cb2e3e` (`feat(p2): add artifact queries, event journal, and fenced mutations`)
Database schema: `user_version = 10`; zero numbered migration bump.

## Scope and Acceptance Summary

This session implements and explicitly verifies P3.1 (the first slice of Phase P3):
1. **Effective Host and Container-VM Capability Discovery**:
   - Discovers host logical CPU cores (`available_parallelism`), host total RAM (`sysctl hw.memsize` on macOS, `sysconf` on Linux), and filesystem storage via `statvfs`.
   - Discovers container-VM limits via `docker info` (`NCPU`, `MemTotal`, `CgroupVersion`, and cgroup v2 feature flags).
   - Synthesizes effective capabilities for container workloads by clamping to container-VM limits (`min(host, vm)`), preventing overestimation of host hardware when running inside bounded container VMs (e.g. Docker Desktop on macOS).
   - Implements passive hardware accelerator discovery (Apple Silicon Metal GPU on macOS arm64, NVIDIA CUDA via nvidia-smi on Linux) without executing untrusted candidate code on the host.
2. **Configured Resource Headroom Accounting**:
   - Accounts for configured headroom reserving CPU (default 1000 millicores), memory (default 512 MiB), and disk (default 64 MiB, matching `DISK_HEADROOM`) for host OS and harness control plane.
   - Computes available capacity as `Effective - Headroom`.
   - Supports configuration overrides from `<root>/resource_headroom.json` and environment variables (`MLRESCURIX_HEADROOM_CPU_MILLICORES`, `MLRESCURIX_HEADROOM_MEMORY_BYTES`, `MLRESCURIX_HEADROOM_DISK_BYTES`).
3. **Resource Request Presets**:
   - Built-in presets: `MICRO` (1 thread, 512 MiB RAM, 1 MiB output), `STANDARD` (2 threads, 2 GiB RAM, 2 MiB output), `HEAVY` (4 threads, 8 GiB RAM, 4 MiB output), and `CUSTOM`.
   - Enforces parameter bounds (1..=256 threads, 64 MiB..=64 GiB RAM, output within 4 MiB M1 ceiling).
   - Resolves declared presets into concrete job specifications via `resolve_job_spec_preset`.
   - Supports custom overrides via `<root>/resource_presets.json`.
4. **Early Infeasible-Request Rejection Before Admission**:
   - Deterministically rejects jobs requesting excessive threads (`threads > available_cpus`), excessive memory (`memory_bytes > available_memory`), excessive output storage (`max_output_bytes > available_disk`), unavailable accelerators, or memory certification on environments lacking cgroup v2.
   - Rejection occurs in `execution::submit` prior to database insertion; 0 rows are added to the SQLite `jobs` table (`SELECT count(*) FROM jobs` remains 0).
   - Secondary validation runs in `Supervisor::admit` to handle dynamic capacity shifts during queue residency.
5. **Container-VM Limit Enforcement**:
   - Verifies that a container job requesting 10 GiB RAM is rejected on an 8 GiB container VM even if the host machine has 16 GiB total RAM.

## Verification Logs

All test suites were executed with `--locked` from repository root:
- `resource_broker_p3.log`: 5 passed, 0 failed, 0 ignored.
- `lib.log`: 14 passed, 0 failed, 2 ignored.
- `artifact_registry_p2.log`: 14 passed, 0 failed, 0 ignored.
- `foundation.log`: 26 passed, 0 failed, 0 ignored (`--test-threads=1`).
- `recovery_cycles.log`: 8 passed, 0 failed, 2 ignored (`--test-threads=1`).
- `release_m6.log`: 5 passed, 0 failed, 3 ignored.
- `agent_m4.log`: 23 passed, 0 failed, 0 ignored.
- `search_m3.log`: 9 passed, 0 failed, 1 ignored.
- `python_protocol.log`: 3 passed, 0 failed.
- `git diff --check`: passed cleanly.

All historical evidence under `docs/evidence/m6/` remains completely untouched.
M6 release status remains independently **NOT READY**.
