# Prerequisite correctness and governance repairs — 2026-10-23

This session starts from clean `main` at `b4e4e8a`. No applicable AGENTS.md was found in the checkout or ancestor directories. No historical evidence, split manifest, campaign ledger amount, or dataset was replaced. No paid provider calls, five-task live campaign, or final scoring were performed. Architecture consolidation has not started.

## Implemented

**Development boundary.** `development_stage.py` and `scripts/stage_benchmark_development.py` import rows using the existing source/split SHA-256 commitments and frozen development identities. CSV records (including multiline fields) retain the exact selected order. Receipts bind source, split, selected identities, emitted bytes and row count. Changed inputs, duplicated/out-of-partition identities, incomplete imports and stale receipts fail. K1–K5 runners consume staged development CSVs. K5 retains its full frozen development observation history and a separately hash-bound panel protocol. Trusted derived preprocessing carries parent lineage.

Nonfixture agent sessions require staging receipts; explicit synthetic fixtures cannot count as benchmark evidence. Each session gives evaluators and generated scripts a dedicated development copy, checks its identity, and rejects tampered copies on restart. Generated scripts retain Docker-only execution and cannot approve candidates using their own metrics. Trusted host evaluation remains development execution, not a filesystem security boundary or production containment certification.

The trusted importer hashes/reads source bytes, including final-partition bytes. Its receipt explicitly records that access. Zero final rows in a worker input is **not** a claim that the importer never physically read final labels. Staging does not erase K1's contaminated 178-row final history, regenerated split, or Ticket overlap. No splits were regenerated here.

Reproduce an import without training or scoring:

```sh
python3 scripts/stage_benchmark_development.py --task K1 --destination /tmp/mlrescurix-development/K1.csv
```

Keep the CSV outside Git. K4's utility emits development transactions; trusted joins belong before derived sealing. The historical K4 subset cannot be substituted for missing frozen origins.

**Benchmark gates.** K1/K2 thresholds remain null. Their absolute acceptance status is explicitly inconclusive: the master plan supplies metric semantics, baseline comparisons and non-inferiority tolerance, but no independent absolute quality requirement. Neither historical scores nor their ad hoc thresholds justify inventing one. A product/domain owner must freeze an independently justified requirement before further adaptive search; no passing result is claimed.

K4's source manifest contains index ranges and some validation timestamps but no authoritative index-space declaration or prediction-context gap/units, no complete per-origin training timestamp bounds, and no compatible frozen mapping of every origin into the 50,000-row subset. Searches of local source and historical manifests did not recover those commitments. Raw timestamps cannot establish the intended embargo. The existing split hash is preserved; acceptance and execution stay blocked pending authoritative provenance. Do not infer metadata, truncate origins, or resplit to pass.

The panel evaluator now requires a `panel_protocol` containing:

- Explicit `required_entities` tuples matching `entity_columns`.
- An `origins` entry for every frozen fold, with `train_start`, `train_end`, and every exact daily `horizon_dates` value.
- Nonnegative integer `embargo_days`; the first horizon date must equal training end plus embargo plus one day. Origins must progress chronologically.
- Complete declared observation-calendar training history and complete daily horizon coverage for **every** required entity; unique/nonmissing entity/date keys. A nondaily observation calendar requires frozen source/split provenance and cannot omit dates present in the development frame.
- `covariate_availability` for every allowed predictor, with provenance and `kind` of `deterministic_calendar`, `static`, or `published`. Static/published inputs require `known_by` no later than that origin's training end. The target cannot be a covariate.
- Explicit `missing_entity_behavior` and `unseen_entity_behavior`; this adapter supports `reject` for both. Other behavior requires a separately implemented and validated adapter.

Missing/incompatible contracts are rejected before fitting; the controller also validates panel folds before provider dispatch. Historical K5 lacked this protocol and sufficient covariate provenance, including publication timing for oil and historical backfilling. The new `K5_PANEL_PROTOCOL.json` freezes 1,782 required entity tuples from the hash-verified official unlabeled query, exact 15-date horizons for all three existing origins, and the complete source observation calendar. The source has dates without any observations; inventing daily rows or omitting arbitrary observed dates is forbidden. The source-derived history row counts exactly match the unchanged frozen manifests: 2,893,968 / 2,920,698 / 2,947,428 training rows. Coverage validation groups once per fold instead of scanning all rows separately for every entity.

K5's embargo is explicitly zero under the declared prediction context: observations through the previous date are available, and the following 15 dates are forecast using deterministic Gregorian calendar predictors only. The justification is independent of observed model scores. Oil, promotions, source-derived historical priors and static fields without publication timing are excluded from this protocol. The K5 runner now uses the complete staged history, entity/date columns, target, and deterministic day-of-week/day-of-month/month features; it no longer builds the incompatible June-only history or backfilled oil view. Missing/unseen entities reject. This changes the predictor contract, not the original split or acceptance threshold. Full-scale model resource feasibility and quality remain unverified.

Reproduce the frozen protocol without fitting or scoring:

```sh
python3 scripts/freeze_k5_panel_contract.py --output /tmp/K5_PANEL_PROTOCOL.json
cmp docs/v2/K5_PANEL_PROTOCOL.json /tmp/K5_PANEL_PROTOCOL.json
```

The importer validates source hashes, per-date entity coverage and every frozen origin row/index count. Only entity/calendar protocol metadata, never copied sales labels, are versioned. A generic score still cannot certify panel forecasting.

**Campaign governance.** The JSON campaign ledger remains the single dollar authority. POSIX locking, atomic replacement, file and directory fsync protect new transitions. Reservations, settlements, and ambiguous outcomes have stable terminal request records and idempotent duplicate handling. Cross-run reuse, altered reservation amounts, settlement without reservation, nonfinite/negative amounts, and reopening ambiguous/settled requests fail.

Before nonfixture gateway dispatch, Rust obtains the **same ledger lock**, checks campaign totals and a sufficient matching active reservation against its rounded token-price upper bound, and durably binds the reservation to sanitized request digest, project/task scope, and canonical receipt-store path. The lock spans receipt fencing and transport. Integer nanodollar parsing avoids floating-point admission. Rust requires the requested model and `inference-net/fp8` route without fallback. Other routes fail. Offline fixture transports are explicitly identified and do not represent paid execution.

The Rust receipt precedes accounting/return; Python checkpoints the response before idempotent settlement. A crash before dispatch leaves the reservation committed; an awaiting/unknown receipt prevents another dispatch; a saved response can be inspected/reconciled without sending again; settlement replay cannot charge twice. There is no distributed exactly-once provider guarantee or automatic release/replay of uncertain requests. Receipt and ledger writes are conservative across the two stores, not one cross-store atomic database transaction. A response after a controller timeout may need operator reconciliation with the original gateway database. Keep that database and request ID. An unexpectedly higher observed charge is recorded rather than hidden, and stops continuation.

The direct Python and legacy specialist HTTP paths are disabled. The real Python gateway adapter accepts only this campaign’s preserved canonical ledger path, so choosing a new campaign directory cannot mint another live allowance; temporary authorities are limited to explicitly offline fixtures. Live agent calls require a campaign-bound Rust adapter. `provider_accounting` in Rust is supplemental telemetry, not another campaign balance.

**Outbound approval.** Staging is not authorization. The adapter never invokes approval. An independently authorized local operator can inspect exact sanitized bytes and authorize the digest with a TTL:

```sh
mlrescurix-harness inspect-outbound-view --state STATE --project PROJECT --request-sha256 DIGEST
mlrescurix-harness approve-outbound-view --state STATE --project PROJECT --request-sha256 DIGEST --actor AUTHORIZED_OPERATOR --ttl-ms 3600000
```

Approval is content-bound, expiring and recorded in durable events. Controller actor identities are rejected. These commands are an explicit local operator approval mechanism, not a signed remote identity system; actor names alone do not authenticate an adversarial process sharing the operator's OS identity. The controller/worker must not be granted the operator's approval execution authority. No approval for future provider uploads was granted during this offline session. Production separation of identities remains a deployment prerequisite.

**Accounting.** Calculated estimates, unverified historical accounting, and provider-verified charges are distinct. Estimated/unverified settlement retains the unused reservation as uncertain rather than releasing it. The verified-charge settlement branch is reserved for a trusted reconciliation process with actual charge evidence, never an agent's boolean assertion. No such verified-charge evidence was available here.

`scripts/reconcile_campaign_accounting.py` performs a read-only audit. All 29 run amounts sum to the preserved USD 0.050724020. Twenty runs have turn/response accounting totaling the same recorded run amounts (56 turns); nine lack matching turn-state records. The audit does not turn matching bookkeeping into verified billing. Historical Rust receipts with `cost_is_estimate=false` remain classified as calculated estimates. New summaries/provider-usage records explicitly label legacy billed fields as estimates or unverified accounting and keep verified provider charges null. No authoritative invoice/provider-charge receipt was available, so actual provider reconciliation is blocked and all existing commitments remain committed. Ledger SHA-256 is recorded in the audit; the ledger itself is unchanged.

**Execution and measurements.** Gateway timeout/interrupt cleanup now terminates and reaps the transport process group, including a spawned transport child; a synchronized subprocess regression exercises this crash boundary. Legacy candidate subprocesses now receive an environment allowlist, reject arbitrary extra environment variables, execute in process groups, kill descendants and reap on completion/timeout. This closes concrete credential-inheritance and child-lifecycle defects; it does not convert the legacy subprocess into a hostile-code sandbox. Partial process enumeration/measurement is labeled partial. Missing worker measurements remain unavailable. Generated-script feedback can expose bounded registered script/line locations and error classes while withholding data-bearing exception text; self-reported metrics stay advisory.

## Verified and blocked

Final validation: Python **308 passed / 1 skipped** (10 warnings, 202.10s), final canonical-ledger/gateway/prerequisite regressions **29 passed**, Rust **63 passed / 2 ignored**, explicit Linux cgroup fixtures **2 passed**, Rust build and Python compilation passed. The final canonical binding guard was tested in the 29-test affected suite after full-suite collection. Exact commands/counts and redacted logs are in `docs/evidence/v2/autonomy/20261023-prerequisite-repairs/summary.json`. Earlier failed attempts are retained separately. An initial cgroup test filter selected zero tests; the corrected selector ran both tests.

K1/K2/K3 official staging reproduced against pinned source and split hashes. The complete K5 metadata protocol reproduced byte-for-byte against pinned sources. Only hash/lineage receipts are committed. The importer did not score final labels.

Docker's Linux environment and cached scientific image `mlrescurix-scientific@sha256:d094b6becbbd1a1d86b944fd39370bfe26bb96f6290b429c0a51c512faf5f1bd` supported the existing cgroup tests: allocation measurement, enforced OOM/event receipt, durable recovery and cleanup passed. Receipts are `live-cgroup.json`, `live-oom.json`, and `live-survivor.json`. They certify those fixture assertions only. Campaign workers traversing the full production scheduler, per-job CPU accounting, concurrent sibling containment, soak/fault coverage and clean-room delivery remain unverified. The host Python suite skips the explicitly bounded 64m tmpfs ENOSPC check. A separate attempt in the cached digest-pinned container was blocked before test execution by missing `jsonschema`; its log is retained. Worker package metadata now declares the locally tested `jsonschema==4.26.0` and `psutil==6.1.1`, so future builds must fail dependency checks until matching hash-verified Linux wheels and an updated lock are supplied. No dependencies were downloaded and no unbuilt image digest was asserted. The cgroup fixture passes do not resolve that scientific-image dependency gap.

Remaining external prerequisites: independent K1/K2 quality requirements; authoritative K4 fold/gap provenance; actual provider-charge receipts; explicit outbound authorization; current route availability and prices before any later live dispatch. Live repaired-revision campaign evidence and final assessments remain outstanding. No production certification is claimed.

Campaign funds remain **USD 0.050724020 committed / USD 0.249275980 recorded remaining / USD 0.30 ceiling**. This session adds **USD 0**.
