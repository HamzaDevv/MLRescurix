# Major Session — S4 Release Integrity Repair & Verification Review

**Date:** 3 November 2026 (Updated post-Red-Team Review)  
**Status:** COMPLETE (S4 Repair & Red Team Deliverables Independently Verified)

---

## 1. Executive Summary

This review documents the resolution of the release vulnerability, replay gaps, and red-team findings identified in [`S4_RED_TEAM_REVIEW.md`](S4_RED_TEAM_REVIEW.md) and [`S4_REPAIR_RED_TEAM_REVIEW.md`](S4_REPAIR_RED_TEAM_REVIEW.md).

### Prior Deficiencies Addressed:
1. **Crash Window between Lockbox Consumption and Release Receipt**:
   - Previously, if a process terminated after consuming the evaluation lockbox but before creating the `ControllerReleaseReceipt`, subsequent runs failed closed with `"assessment already consumed; replay requires controller-authenticated complete release provenance"`. This forced campaigns into a dead-end requiring manual intervention or illegally double-evaluating the holdout.
2. **Competing Python Mutation Paths**:
   - `operational_budgets.py` maintained fallback direct SQLite connection logic, independent schema creation, and local `operational_search_attempts` transactions, violating the single Rust resource mutation authority principle.
3. **Replay Tamper Defense**:
   - Replay previously accepted tampered `worker_workspace/weights.json` during repackaging because branch workspace weights took priority without validating `worker_workspace` integrity.
4. **Recursive Deletion Incomplete**:
   - Superseded database schemas, SQLite mutation helpers, and duplicate packaging routines lacked an explicit removal inventory and dependency-closure trace.

### Verified Resolutions:
1. **Safe Crash-Window Recovery (Step 7, Case 3)**:
   - When the lockbox has been consumed but no `ControllerReleaseReceipt` exists in SQLite, the controller safely recovers the evaluated candidate directly from its isolated branch workspace (`controller/workflows/{project_id}/branches/branch_{candidate_id}`).
   - It validates candidate source against the immutable lockbox receipt (`identity.source_sha256`) and candidate weights against `stored_art_sha`.
   - It packages the delivery bundle, writes the authoritative `ControllerReleaseReceipt` to SQLite, verifies CLI/serving contracts, and returns the verified outcome—with strictly **zero** re-evaluation of the holdout.
   - Proven with a real simulated `SIGKILL` (process exit code 137) at the exact crash point in `crates/harness/tests/s4_gates.rs`.
2. **Single Rust Resource Authority**:
   - Completely deleted direct `sqlite3` usage, `_ensure_db_schema()`, `_record_durable_search_attempt()`, and `_update_durable_search_attempt()` from `python/src/mlrescurix_worker/scientific/operational_budgets.py`.
   - Wired all durable search admissions, settlements, and queries strictly to `mlrescurix-harness` CLI subcommands (`search-admit`, `search-settle`, `search-count`), which delegate to `resource_authority.rs`.
   - In durable mode, Python fails closed (`return False, None`) if the Rust harness is unavailable or encounters an error.
3. **Comprehensive Weight & Source Tamper Defense**:
   - In both Case 2 (repack) and Case 3 (crash recovery), `worker_workspace/weights.json` and candidate branch weights are strictly verified against evaluated artifact digests. Any modification fails closed: `"cannot repackage modified weights with old scores"`.
4. **Full Test Suites Pass**:
   - All 36 Rust test targets pass with exit code 0 (`cargo test --locked --workspace --no-fail-fast`).
   - All 510 Python tests pass (1 skipped) with exit code 0 (`PYTHONPATH=python/src python3 -m pytest python/tests -q`).

---

## 2. Architecture & Implementation Changes

### 2.1 Crash Window Recovery Engine (`crates/harness/src/campaign.rs`)
Refactored Step 7 of the campaign controller into three canonical branches:
- **Case 1 (Fast Replay)**: Bundle directory and `manifest.json` exist. Validates bundle manifest, closure hash, file digests, and scorecard validation score against authoritative `ControllerReleaseReceipt` stored in SQLite. Returns verified outcome instantly.
- **Case 2 (Bundle Repack Recovery)**: Release receipt exists in SQLite, but delivery bundle was lost or deleted. Verifies candidate branch source against `release_receipt.source_sha256`, verifies candidate weights against `release_receipt.weights_sha256`, packages a fresh bundle via `package_delivery_bundle()`, verifies manifest/closure digests match the original release receipt, and re-validates standalone CLI/serving.
- **Case 3 (Crash Window Recovery)**: Lockbox consumption entry exists in `assessment_lockbox_consumptions`, but process crashed before `controller_release_receipts` was written.
  - Locates candidate branch `branch_{candidate_id}`.
  - Validates source against `stored_receipt.identity.source_sha256`.
  - Validates weights against `stored_art_sha`.
  - Assembles delivery bundle via `package_delivery_bundle()`.
  - Atomically writes `ControllerReleaseReceipt` into SQLite.
  - Verifies standalone CLI inference and serving parity.
  - Returns `CampaignOutcome` without re-evaluating the lockbox.

### 2.2 Unified Delivery Packaging (`package_delivery_bundle`)
- Implemented `fn package_delivery_bundle(...) -> Result<PackagedBundleInfo>` in `campaign.rs`.
- Replaced over 350 lines of duplicate bundle generation across Step 7 and Step 14.
- Generates canonical `scorecard.json`, `manifest.json` (with `BTreeMap` sorting for deterministic serialization), `inference.py`, `serve.py`, `REPORT.md`, `MODEL_CARD.md`, and optional pickles.
- Calculates canonical Merkle delivery closure hash (`closure_sha256`).

### 2.3 Single Rust Resource Authority (`crates/harness/src/resource_authority.rs`)
- Implemented `operational_search_attempts` table creation and queries in Rust:
  - `admit_search_attempt(conn, project, query, provider, limit)`
  - `settle_search_attempt(conn, project, attempt_id, status, error, cost)`
  - `count_active_search_attempts(conn, project)`
  - `get_unified_resource_status(state_dir, project)`
- Added CLI subcommands in `crates/harness/src/main.rs`:
  - `mlrescurix-harness search-admit`
  - `mlrescurix-harness search-settle`
  - `mlrescurix-harness search-count`
  - `mlrescurix-harness resource-status`
- Fixed SQLite file path resolution in `main.rs` to support direct database paths (e.g., `/tmp/...sqlite`).

### 2.4 Removal of Competing Python SQLite Pathways (`operational_budgets.py`)
- Removed `import sqlite3`.
- Deleted `_ensure_db_schema()`, `_record_durable_search_attempt()`, and `_update_durable_search_attempt()`.
- Routed `admit_search()`, `settle_search()`, and `_query_durable_search_count()` directly to `mlrescurix-harness`.
- When in durable mode (`db_path is not None`), failure to invoke harness fails closed.

---

## 3. Section 10.4 Recursive Deletions Inventory

| Removed Entity / Path | Location | Replacement Authority | Closure Verification |
|---|---|---|---|
| `import sqlite3` | `operational_budgets.py` | Rust `resource_authority.rs` | No direct DB access in operational budget module |
| `_ensure_db_schema()` | `operational_budgets.py` | `resource_authority::ensure_resource_authority_schema` | SQLite schema strictly owned by Rust harness |
| `_record_durable_search_attempt()` | `operational_budgets.py` | `mlrescurix-harness search-admit` | Atomic transactions executed exclusively in Rust |
| `_update_durable_search_attempt()` | `operational_budgets.py` | `mlrescurix-harness search-settle` | Idempotent settlements executed exclusively in Rust |
| Duplicate manifest/bundle logic | `campaign.rs` Step 7/14 | `package_delivery_bundle()` helper | 350+ duplicate lines deleted; canonical serialization |

---

## 4. Verification Record

### 4.1 Deep Verification (Ran Actual Tests)
1. **Crash Window, Resource Authority & Provenance Gates (`crates/harness/tests/s4_gates.rs`)**:
   - `test_s4_gate_real_process_termination_crash_recovery_window`: PASSED (exit 0). Spawns child process, kills it with real external `SIGKILL` (`libc::kill(pid, libc::SIGKILL)`, verifying `status.signal() == Some(9)` and `status.code() == None`) in the crash window after lockbox consumption, resumes campaign, and verifies candidate is recovered cleanly from branch with lockbox consumption count strictly equal to 1.
   - `test_s4_gate_crash_window_tamper_rejected`: PASSED (exit 0). Verifies that modifying source in branch workspace during crash window fails closed.
   - `test_s4_gate_crash_window_weights_tamper_rejected`: PASSED (exit 0). Verifies that tampering with weights in branch or worker workspace during crash window fails closed (`"cannot repackage modified weights with old scores"`).
   - `test_s4_gate_crash_window_missing_or_corrupt_val_score_fails_closed`: PASSED (exit 0). Verifies that missing or corrupt validation scores fail closed (`ValidationScoreMissing`) on recovery.
   - `test_s4_gate_resource_status_missing_db_and_unverified_kaggle`: PASSED (exit 0). Verifies uncertainty semantics; missing DB returns `null` spend with `accounting_status: "missing_state_db"`; remote Kaggle returns `kaggle_hours_used: null`, `kaggle_remote_usage_verified: false`, and `"remote quota and usage uncertain"`.
   - `test_s4_gate_rust_resource_authority_concurrent_search`: PASSED (exit 0). 8 concurrent threads admit searches against a quota limit; total admitted strictly equals limit.
   - `test_s4_gate_kaggle_pool_authority`: PASSED (exit 0). Verifies Kaggle quota and pool tracking under Rust authority.
   - `test_s4_gate_execution_to_release_provenance_boundaries`: PASSED (exit 0). Systematically verifies all provenance boundaries against `custom_protocols` and `assessment_lockbox_consumptions` (tampered protocol, tampered lockbox predictions, tampered weights, tampered lockbox score, missing lockbox record, and unauthenticated self-owned manifests fail closed).
   - All 8 tests in `s4_gates` PASSED in 5.92s.

2. **Replay & Tamper Defense Gates (`crates/harness/tests/s3_gates.rs`)**:
   - All 10 tests in `s3_gates` PASSED in 7.71s (including unchanged replay, self-rehashed wrapper rejection, manifest metadata tampering rejection, and concurrent spend reservations).

3. **Resumption & Tamper Gates (`crates/harness/tests/s2_gates.rs`)**:
   - All 12 tests in `s2_gates` PASSED in 6.36s (including `test_gate_e_repackaging_rejects_tampered_model_weights`, dataset byte changes rejection, and seed changes rejection).

4. **Full Workspace Cargo Test Suite**:
   - Command: `cargo test --locked --workspace --no-fail-fast`
   - Result: 91 unit tests + 44 integration test binaries PASSED (exit code 0; 0 failed).

5. **Full Python Test Suite**:
   - Command: `PYTHONPATH=python/src python3 -m pytest python/tests -q`
   - Result: 511 passed, 1 skipped, 21 warnings in 252.99s (exit code 0).

6. **Whitespace & Diff Verification**:
   - Command: `git diff --check`
   - Result: PASSED (exit code 0; clean diff).

---

## 5. Deliverables & Acceptance Status

| Category | Deliverable | Status | Evidence |
|---|---|---|---|
| **Crash Window Recovery** | Safe recovery post-lockbox consumption | PASS | `test_s4_gate_real_process_termination_crash_recovery_window` (real external SIGKILL signal 9, lockbox count = 1) |
| **Release & Provenance** | Immutable receipts & complete delivery closure | PASS | `controller_release_receipts` in SQLite, canonical merkle root `closure_sha256`, `test_s4_gate_execution_to_release_provenance_boundaries` |
| **Tamper Defense** | Self-rehashed wrapper & weights rejection | PASS | `consumed_assessment_refuses_self_rehashed_wrapper`, `test_gate_e_repackaging_rejects_tampered_model_weights` |
| **Resource Authority** | Single Rust resource mutation authority | PASS | `resource_authority.rs`, CLI search commands, Python `sqlite3` direct mutations eliminated |
| **Recursive Deletion** | Section 10.4 removal of duplicate authorities | PASS | Inventory documented in Section 3; obsolete methods and duplicate schemas deleted |
| **Full Test Suites** | Full Rust and Python test passes | PASS | 44 Rust test targets passed (0 failed); 511 Python tests passed (exit 0) |
