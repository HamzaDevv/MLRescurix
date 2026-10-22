# MLRescurix session handoff

Session: run_20261022_230312
Stop: HANDOFF
Checkpoint: /Users/hamza/Developer/Imp_Projects/MLRescurix/docs/evidence/v2/robustness/20261021-robustness/run_20261022_230312/autonomous_loop_state.json
Accepted candidate: None

Production certification: false. Preserve data/fold hashes and shared campaign budget.

Next steps:
- Raise the worker RSS budget or enable cgroup-enforced containment so peak memory is bounded and observable rather than post-hoc terminated.
- Reduce the active feature matrix before modeling: drop near-empty columns (dist2 94.8% missing, D7/D8/D9/D12/D13/D14 ~90% missing, R_emaildomain 82% missing) and prune the V-block to a low-variance subset to cut the 434-column footprint.
- Evaluate a single low-memory family first (penalized_linear on a frequency-encoded, standardized subset) to establish a baseline ROC-AUC under the pinned folds before attempting hist_gbdt or lgbm.
- If hist_gbdt is retried, cap max_bins, set early_stopping with a small validation fraction, and cast float64 columns to float32 to reduce resident memory.
- Preserve folds_sha256 1b746fd844f7dd213539612826208a9114d2f559f8621e41a146d9de9726820a and the chronological rolling-origin split (3 development origins, 0.2 final fraction, locked gap) unchanged in the successor session.
- Record the known-vs-new entity assumption for TransactionID/card1 grouping in the successor session's evidence before any search.
