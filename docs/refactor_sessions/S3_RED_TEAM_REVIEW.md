# Session 3 independent review

Reviewed 3 November 2026 against the staged implementation, author report and tests. **Session 3 remains partial; gates A–G are not all accepted.** The author packet's implementation and full-suite claims are historical author assertions; this review independently reran the focused checks below, not the full suites. Production isolation remains unavailable. No new live Amazon campaign, paid model request, search or Kaggle job was requested by this review.

## Useful progress

The offline fixture supports a multi-file Python example, manifest-directed commands and trusted Rust scoring. The previous specific `solution.py` repack probe now rejects its mutation. Search without a budget object is refused, contradictory terminal search settlement is rejected, and the built wheel's action file was reported present by the author's probe. An immediate-transaction fenced-write helper and process heartbeats were added. A challenger rejection fixture avoids assessment. These are useful narrower results, not the complete acceptance requirements.

## Findings requiring follow-up

1. **Release/source identity is incomplete (F, G).** `collect_py_files` omitted every file named `inference.py` or `serve.py`, even imported modules in subdirectories; the code also excludes non-Python assets and `program.json`. The fast path reads validation score from bundle `scorecard.json`, and receipt parsing is optional (`if let Some`), so missing/unparseable receipt does not fail closed. Manifest self-consistency does not prove controller-issued score provenance. The author tested one changed `solution.py`, not program, environment, prediction, wrapper and rewritten scorecard swaps. This review includes nested Python modules in the hash and rejects candidate root names reserved for generated wrappers; existing wrappers and other source/asset/environment/receipt bindings still require comprehensive design and tests. Old hashes using excluded nested filenames may fail closed on replay.
2. **One Rust resource authority was not delivered (D).** Python `OperationalBudgets` still directly mutates SQLite search admissions and maintains model token and Kaggle counters; connectivity is still used as an `enforced` status. The eight-thread test races *lease acquisition*, not token/search/Kaggle/financial admission. It does not verify atomic multi-resource accounting or historical obligations migration. Search refusal and conflicting settlement are separate improvements, not proof of the gate.
3. **Campaign recovery and atomic fencing remain unproven (C).** The purported killed controller is a `sleep 60` child; the test process acquires the lease using its PID, kills it, and exercises a helper. No actual campaign is killed around dispatch or completion, no unknown external effect is reconciled, and no stale artifact overwrite is tested. Some writes, such as specialist task assignment, remain outside `execute_fenced_write`; heartbeat errors are ignored. The helper is a partial improvement.
4. **Hostile-worker isolation is NOT_VERIFIED (E).** The test verifies production refusal, not a hostile process denied access. The offline worker shares host privileges and receives lockbox features in its mutable workspace before selection. A refusal test cannot certify a containment gate.
5. **Owner/independence and generality are partial (A, B).** The context test checks database `immutable_inputs`, not the actual serialized model request; its failed authoring is a scripted file edit, not an executed failed fit and evidence-driven repair. Challenger evidence remains summary based. The code still permits `[FINAL]` or `reference` to trigger default template generation without authored files. The supplied fixture has numeric-only data and a specific source layout; production containment, mixed text input and a second independent layout are unproven.
6. **Standalone parity and contract closure are not demonstrated (G).** The test named frozen candidate parity checks only that two self-generated example predictions are finite. It does not compare standalone predictions to scored frozen predictions or prove a fresh pinned environment. Packaging schemas by copying canonical files fixes installation but creates two copies needing drift control. Recursive legacy deletion and full cross-language schema parity remain open.
7. **Historical evidence was overwritten.** The repeat S2 probe wrote new results to S2's original observation file. This review preserved the new output as `docs/evidence/refactor/S3/s2_probe_after_s3.json`, restored the S2 file to its committed content and changed the probe to print results without overwriting history. The staged index still reflects the author's version; the working-tree restoration must be included when the user later stages changes.

## Corrected assessment

| Gate | Status | Reason |
| --- | --- | --- |
| A | PARTIAL | One offline unlisted example; general contained multi-layout/mixed-input execution unverified |
| B | PARTIAL | Scripted trajectories exist; serialized request and independent primary-evidence audit unproven |
| C | PARTIAL | Helper and killed surrogate process; no real killed campaign or complete fenced effects |
| D | FAIL for unified authority | Test races leases and Python still owns resource mutations |
| E | NOT_VERIFIED | Production refusal works; no hostile-worker containment test |
| F | PARTIAL | One mutation rejected; source, program, environment, receipts and scoring identity not fully bound |
| G | PARTIAL | Installed schema improvement; actual scored-candidate parity and deletion/contract closure unproven |

## Verification

The submitted `cargo test --locked -p mlrescurix-harness --test s3_gates` completed **8 passed** independently before the bounded hash fix. After the fix, `cargo test --locked -p mlrescurix-harness --test s3_gates --test s3_review_regressions --test s2_red_team` passed **12 tests** (8 + 1 + 3). The new test covers nested module hashing; the root-name refusal has not been independently exercised end to end. The focused Python research, accounting and S2 regression tests completed **11 passed**. The author reports full Rust and Python suite success; independent full-suite reruns were not performed for this review. A previous full Rust workspace run passed during S2 review, but a prior attempt showed a score just below its threshold; Session 3 changes that test to accuracy, so the original balanced-accuracy discrepancy is not proven resolved.
