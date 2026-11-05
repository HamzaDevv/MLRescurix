# Independent review: five-turn live campaign

The trace and handoff record a real five-turn Amazon development campaign, selected neural candidate `cand_23fd26dd9d8e5346700121bb`, and development CV SMAPE 62.61695. The run stopped at `max_turns_reached`, and the handoff says production certification is false. This does not demonstrate unrestricted autonomous completion, independent specialist orchestration, or production readiness.

## Holdout exposure

An independent comparison using pandas row hashes showed that all 15,000 rows in `holdout_lockbox_15k.csv` occur in the earlier `staged/development.csv`, which previous Amazon campaigns used for model selection. Nine rows also have matching content and price in the current `staged/dev_60k.csv`. Row-hash equality identifies matching records, including duplicates; it does not by itself prove the exact original row identities. The reported 61.6392 SMAPE is an exploratory score on previously exposed data, not a pristine lockbox result. The sample has already been evaluated and must not be reused as an untouched holdout.

`scripts/evaluate_holdout.py` independently checks example replay and bundle hashes, but evaluates labeled data in the calling process and does not implement a trusted once-only lockbox, establish disjoint provenance, or anchor the manifest to an authoritative release receipt. The prior report's clean-subprocess claim applies to example replay, not the labeled evaluation. The test suite previously reevaluated the live sample; that test is now skipped. The evaluator is explicitly labeled exploratory, requires the harness, and no longer deletes bundle files to conceal unexpected changes.

## Budget and remaining work

The ledger records $0.00254 in calculated provider estimates and $0.01439 uncertain exposure for the five-turn run. These are not provider-verified final charges. The $0.12529 figure is campaign ledger headroom, not independent proof of a session-wide cap covering every paid service. The handoff marks global Kaggle and web enforcement unverified; neither was exercised live.

The change in this session primarily adds an evaluation script, test coverage, and a missing-authority error; it does not establish the requested independent boss/specialist workflow or unify all paid-service accounting. The five-turn launcher default was restored to no arbitrary turn limit. Complete safe worker fencing, controller-anchored provenance, certified once-only evaluation using genuinely unexposed data, independent specialists, broad search/pivot evidence, and Linux production containment. Audit obsolete code with evidence before deleting it.

The submitted report claims full Rust and Python suites passed. This review did not independently rerun those full suites. See the final response for focused checks run after review changes.
