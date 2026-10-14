# P2 Roadmap Gate Closure Acceptance Evidence

Session date: 2026-10-14 (Asia/Kolkata)
Branch: `codex/p1-cgroup-orphan-recovery`
Starting HEAD: `9cb2e3e` (`feat(p2): add artifact queries, event journal, and fenced mutations`)
Database schema: `user_version = 10`; zero numbered migration bump.

## Scope and Closure Summary

This session audits and provides explicit verification evidence closing all remaining P2 acceptance checks defined in `docs/v2/ROADMAP.md` and `docs/v2/PROGRESS.md`:
1. **Populated-state upgrade preservation**: Verified that upgrading a populated legacy database (v9 -> v10) preserves all pre-existing `final_test_gates` (both consumed and reserved records, dataset/frozen-model/result SHA-256 digests), `search_sessions` (apex candidate pointer and JSON selection records), `trial_results` (metrics and feasibility), `pipeline_receipts`, `tombstones`, and `action_fences` with zero foreign-key violations (`PRAGMA foreign_key_check` empty) and repeated migration idempotency.
2. **Partial CAS/filesystem/DB commit reconciliation**: Integrated `ContentAddressableStorage::sweep_staging` into `execution::collect_artifacts` to purge orphaned temporary staging files in `cas/.staging/`. Verified that CAS files created on disk prior to DB commit crashes are safely deduplicated/adopted on retry without double byte allocation, corrupted files on disk are detected by SHA-256 check and repaired atomically, and pre-admission byte reservations remain accurate.
3. **Protected references under concurrent GC**: Verified that running `collect_artifacts` concurrently with active reader/mutator threads preserves pinned artifacts (`artifacts.pinned = 1`), candidate and scorecard versions referenced by `project_selections` (Apex and historical), and lineage graph edges (`artifact_lineage`), while unpinned terminal job directories are safely reclaimed and active environment leases are protected.
4. **Event restart/reconnect consistency**: Verified that subscribing to events, dropping connections to simulate process restart, performing mutations (`publish_successor_command`, `annotate`, `tombstone_artifact_command`, `set_apex_selection`), and reopening `ArtifactRegistry` after restart allows subscribers to reconnect using their preserved cursor and receive all committed events in strict monotonic cursor order with zero duplicate deliveries. Also verified that expired cursors beyond the 10,000-event retention window trigger `reset_required = true` with clean snapshot resynchronization.

## Verification Logs

All test suites were executed with `--locked` from repository root:
- `artifact_registry_p2.log`: 14 passed, 0 failed, 0 ignored.
- `lib.log`: 14 passed, 0 failed, 2 ignored.
- `foundation.log`: 26 passed, 0 failed, 0 ignored (`--test-threads=1`).
- `recovery_cycles.log`: 8 passed, 0 failed, 2 ignored (`--test-threads=1`).
- `release_m6.log`: 5 passed, 0 failed, 3 ignored.
- `agent_m4.log`: 23 passed, 0 failed, 0 ignored.
- `search_m3.log`: 9 passed, 0 failed, 1 ignored.
- `python_protocol.log`: 3 passed, 0 failed.
- `git diff --check`: passed.

All historical evidence under `docs/evidence/m6/` remains completely untouched.
M6 release status remains independently **NOT READY**.
