# P2.6 successor and tombstone evidence

Session date: 2026-10-14 (Asia/Kolkata)
Branch: `codex/p1-cgroup-orphan-recovery`
Starting HEAD: `7e162a3` (`feat(p2): implement artifact registry, CAS, and schema v10 migration`)
Database schema: `user_version = 10`; this slice adds no numbered migration.

The worktree was already dirty with the previous P2.3–P2.5 implementation and
evidence. Those changes were preserved and extended. No M6 evidence was changed.

## Changes covered

- `publish_successor_command` compares the shared project revision, checks its
  `(project, command_id)` request fence, creates an immutable version, advances
  the logical artifact pointer, records `supersedes` lineage, and increments the
  project revision in one SQLite transaction.
- `tombstone_artifact_command` uses the same revision/idempotency fence and
  preserves artifact versions, lineage and bytes. It refuses pinned artifacts,
  any version referenced by project selection history, and versions produced by
  queued/running/terminating/recovering jobs.
- Tombstone events are journaled by a SQLite trigger. The idempotent event schema
  fragment is applied on every open/migrate so existing v10 state roots receive
  the additive trigger without a schema version bump.
- New regressions cover exact command retry, command ID payload reuse, stale
  revision rejection, immutable successor lineage, tombstone visibility/history,
  tombstone event emission, pins and selection history.

## Verification

Each command ran from the repository root with `--locked`; logs contain the
complete output and exit code.

- `cargo test --locked --test artifact_registry_p2`: 10 passed, 0 failed.
- `cargo test --locked --test recovery_cycles`: 8 passed, 2 ignored, 0 failed.
- `cargo test --locked --test foundation`: 26 passed, 0 failed.
- `cargo test --locked --test release_m6`: 5 passed, 3 ignored, 0 failed.
- `git diff --check`: passed.

One foundation test initially failed during three concurrent Cargo test
processes; it passed on an isolated rerun and again in the final serial suite.
`cargo fmt --check` could not run because this toolchain has no `cargo fmt`
subcommand installed.

P2 remains in progress. This session did not verify every roadmap gate for
partial CAS/filesystem reconciliation or event restart/reconnect under load.
M6 remains independently NOT READY; no scientific candidate code ran on host.
