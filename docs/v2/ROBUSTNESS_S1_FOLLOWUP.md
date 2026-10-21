# Independent review of S1 repair commit 653a8ee

Date: 2026-10-21. Verdict: **S1 remains partial; S2 live promotion is blocked.**
This review supersedes the completion claims in the supplied Antigravity report and
previous continuation prompt. Historical evidence is retained unchanged.

## Verified problems and changes in this follow-up

1. **RB-001 / P0:** Local `python -I` subprocesses still had host filesystem/network
   access, unbounded captured output, and no worker cgroup limits. Feature execution
   also loaded candidate-writable pickle output in the controller, allowing arbitrary
   controller execution during deserialization. Replaced both arbitrary EDA and string/
   file feature-code execution paths with explicit bounded refusals. The dispatcher
   returns inconclusive for CUSTOM_SANDBOX_CODE. This removes the unsafe path; it
   **does not implement or certify a sandbox**. Re-enable only through a governed
   executor with safe output transport. Built-in trusted callables/transformers remain
   in process and must never receive untrusted dynamically loaded code.
2. **RB-013 / P1:** Ledger reads/writes swallowed all exceptions. Corruption could
   silently erase replay state; ablation and dispatcher also shared one incompatible
   ledger path. Reads/writes now raise WorkflowError, and ablation receipts use a
   separate `.ablations.json` file. Atomic rename alone remains insufficient for
   concurrent transactional execution or authoritative Rust state binding.
3. **RB-016 / P1:** Presence-only gates overstated binary provenance, schema migration,
   dataset authenticity and gateway readiness. These gates now report inconclusive/
   blocked. Docker daemon metadata does not certify per-job enforcement. Corrupt run
   summaries now fail closed rather than resetting accounting. Campaign/run directory
   traversal is rejected. Controller lifetime RSS now has correct byte units for
   macOS/Linux and explicitly reports missing worker measurements. Aggregate status
   and CLI exit reflect non-passing gates instead of unconditional completion.
4. Budget/model policy now selects **USD 0.30 total**, **USD 0.05 per task run**,
   with USD 0.05 contingency included in the total. All sessions, retries, repairs and
   uncertain charges share this ceiling. Existing run spend/reservations are preserved
   when recomputing run-local remaining funds under the lower cap. Live mode stays
   blocked: these observations are **not an authoritative campaign-wide spend ledger**.

## Requested provider

The supplied screenshot selects OpenRouter's inference.net FP8 endpoint. An unauthenticated
GET of the public model endpoint catalog on 2026-10-21 returned:

- Model `deepseek/deepseek-v4.1-flash-1017`.
- Provider `InferenceNet`, endpoint `inference-net/fp8` (also a separate `/us` endpoint).
- Prompt USD 0.08/M, completion USD 0.50/M, cache read USD 0.008/M tokens.
- Context 1,040,000; maximum completion 393,216 (provider capability, not run allowance).

Source: https://openrouter.ai/api/v1/models/deepseek/deepseek-v4.1-flash-1017/endpoints
Routing documentation: https://openrouter.ai/docs/guides/routing/provider-selection
Use `only` and `order` containing `inference-net/fp8`, `allow_fallbacks: false`.
Revalidate pricing before live use; missing price/accounting means refuse. The model
page's cheapest aggregate rate is not this endpoint's rate. Policy is configured in
BENCHMARK_SUITE.json; production gateway enforcement is still required.

## Remaining S1 gaps for Antigravity

- Wire the existing governed harness executor into EDA/custom features, safe IPC,
  bounded streams and output artifacts, enforced mounts/network/CPU/memory/process
  policy and descendant cleanup; prove with disposable canaries and container tests.
- Locked OOF splits still lack authoritative row/topology binding and full validation
  (integer indices without coercion, duplicates across validation folds, coverage,
  time/group constraints). Temporal comparison permits equal timestamps. Missing
  declared columns can silently fall back to random folds. Joint time/group topology
  and purge/gap requirements remain incomplete. Meta learner is fitted and scored
  on the same OOF meta rows; independent stack assessment remains necessary.
- Ablation can still create private folds, fall back from group-aware classification
  to random KFold, and silently use balanced accuracy for unsupported metrics.
  Integrate evaluator-owned fold assignments and reject unsupported metric contracts.
- Strict JSON Schema does not prove supplied source/split/topology/budget identifiers
  match authoritative runtime state. Bind them before execution. Durable Python JSON
  files do not replace Rust's authority; add conflict/interruption/concurrency tests.
- Complete campaign resume identity/source/manifest matching and immutable artifact
  preservation. Add campaign-wide transactional reserve/settle accounting, including
  uncertain responses, before enabling any billed path. Do not reset at a new run ID.
- Official archives, readable metric/rules receipts, hashes and final split manifests
  remain unpinned. Kaggle authentication alone does not satisfy dataset gates.
- Preflight storage and source-built binary checks need actual runtime evidence.
  Synthetic fixtures do not demonstrate five authentic Kaggle solves.

The original `observations.json` is historical evidence from e1c8674, not the output
of the revised probe script. A zero probe exit does not mean every trust boundary was
exercised or fixed. Keep exact test scope and evidence revision in future reports.

## Validation scope

See `docs/evidence/v2/robustness/20261021-s1-followup/verification.json` and test log.
No Rust changes, billed calls, credential contents, authentic Kaggle downloads or final
holdout accesses were involved in this follow-up. Previously reported broad-suite
failure remains unresolved and was not rerun here.
