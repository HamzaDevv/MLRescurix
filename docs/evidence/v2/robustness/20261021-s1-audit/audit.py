"""Safe S1 claim audit. Run with PYTHONPATH=python/src python3 <this file>.

These probes demonstrate current defects, not desired regression assertions.
No provider calls, real credentials, external downloads or final labels are used.
Submodel subprocesses are mocked; generated files stay in temporary directories.
"""
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from mlrescurix_worker.scientific.eda_agent import run_exploratory_python
from mlrescurix_worker.scientific.fe_ablation import _resolve_and_apply_features
from mlrescurix_worker.scientific.hypothesis_executor import HypothesisActionDispatcher
from mlrescurix_worker.scientific.pareto_ensemble import ParetoEnsembleSynthesizer

ROOT = Path(__file__).resolve().parents[5]
results = {}
payload = dict(schema_version="v2", action_id="audit", parent_candidate_id="candidate",
               source_hash="12345678", split_id="split", topology_hash="12345678",
               seed=2026, action="DELETE_FEATURE", parameters={"columns": ["x"]},
               budget_id="budget", rationale_ref="audit")
df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 0]})

invalid = dict(payload, unexpected_field=True, seed="not an integer",
               parameters={"cv_folds": -1, "unknown_parameter": True})
try:
    HypothesisActionDispatcher().validate_action_payload(invalid)
    results["invalid_payload_accepted"] = True
except Exception:
    results["invalid_payload_accepted"] = False

with patch.dict(os.environ, {"MLRESCURIX_AUDIT_DUMMY": "dummy-canary"}):
    output = run_exploratory_python(
        "import os; print(os.environ.get('MLRESCURIX_AUDIT_DUMMY', 'missing'))", df)
results["eda_inherits_dummy_environment"] = "dummy-canary" in output
output = run_exploratory_python("print('界' * 3000)", df)
results["eda_output_bytes"] = len(output.encode("utf-8"))

code = "import os\ndef build_features(df, y=None):\n    return pd.DataFrame({'pid': [os.getpid()] * len(df)}, index=df.index)"
train, _ = _resolve_and_apply_features(code, df, df, "target", [])
results["custom_code_runs_in_controller"] = bool((train["pid"] == os.getpid()).all())

with tempfile.TemporaryDirectory() as tmp:
    ledger = Path(tmp) / "ledger.json"
    dispatcher = HypothesisActionDispatcher(ledger_path=ledger)
    target_payload = dict(payload, action="TRANSFORM_TARGET", parameters={"method": "log"})
    receipt = dispatcher.execute_action(target_payload, df, {}, ["x"])
    results["target_transform_receipt"] = receipt.status
    results["target_transform_mutated_input"] = not df["target"].equals(pd.Series([0, 1, 0], name="target"))
    results["ledger_created"] = ledger.exists()
    results["restart_cache_entries"] = len(HypothesisActionDispatcher(ledger_path=ledger).idempotency_cache)

# Ablation's real receipt uses baseline_cv_score, not base_cv_score; RMSE is negated.
with patch("mlrescurix_worker.scientific.hypothesis_executor.ablate_feature_hypothesis",
           return_value={"status": "ACCEPTED", "baseline_cv_score": -10.0,
                         "cv_score_delta": 1.0, "total_folds": 2}) as ablate:
    receipt = HypothesisActionDispatcher().execute_action(
        payload, df, {}, ["x"], metric_name="rmse", direction="minimize")
    results["improving_neg_rmse_receipt"] = receipt.status
    results["improving_neg_rmse_relative_gain"] = receipt.relative_gain
    results["ablation_metric_forwarded"] = "metric_name" in ablate.call_args.kwargs
    results["ablation_seed_forwarded"] = "random_state" in ablate.call_args.kwargs
    results["ablation_ledger_forwarded"] = "ledger_path" in ablate.call_args.kwargs

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    seen = []
    def fake_run(cmd, **kwargs):
        if "fit" in cmd:
            data = pd.read_csv(cmd[cmd.index("--train-path") + 1])
            seen.append({"train": data["time"].tolist()})
        else:
            data = pd.read_csv(cmd[cmd.index("--input-path") + 1])
            seen[-1]["val"] = data["time"].tolist()
            pd.DataFrame({"pred": np.full(len(data), 0.75)}).to_csv(
                cmd[cmd.index("--output-path") + 1], index=False)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    temporal = pd.DataFrame({"id": range(12), "time": range(12, 0, -1), "target": range(12)})
    with patch("subprocess.run", side_effect=fake_run):
        oof = ParetoEnsembleSynthesizer._collect_oof_predictions(
            root, ["model"], temporal, "target", "id", "regression", None,
            n_splits=2, topology={"temporal_col": "time"})
        results["temporal_oof_trains_on_future"] = any(max(s["train"]) > min(s["val"]) for s in seen)
        results["unassessed_oof_rows_filled_zero"] = int((oof[:, 0] == 0).sum())
        seen.clear()
        try:
            ParetoEnsembleSynthesizer._collect_oof_predictions(
                root, ["model"], temporal, "target", "id", "regression", None,
                split_assignments=[([0, 1], [1, 2])])
            results["overlapping_locked_split_accepted"] = True
        except Exception:
            results["overlapping_locked_split_accepted"] = False

spec = importlib.util.spec_from_file_location("campaign_audit", ROOT / "scripts/run_robustness_campaign.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
results["runner_noop_modes"] = {}
with tempfile.TemporaryDirectory() as tmp:
    module.EVIDENCE_BASE = Path(tmp)
    for mode in ("deterministic_dev", "live_dev", "certify"):
        with patch.object(sys, "argv", ["runner", "--mode", mode, "--run-id", mode]), contextlib.redirect_stdout(io.StringIO()) as stdout:
            try:
                ret = module.main()
                success = (ret == 0 or ret is None)
            except SystemExit as se:
                success = (se.code == 0)
        results["runner_noop_modes"][mode] = {
            "returned_successfully": success,
            "summary_written": (Path(tmp) / "20261021-robustness" / mode / "summary.json").exists(),
            "printed_run_complete": "Run Complete" in stdout.getvalue(),
        }

print(json.dumps(results, indent=2))
