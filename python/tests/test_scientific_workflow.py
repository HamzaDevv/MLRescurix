import csv
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.workflow import (
    WorkflowError, audit_csv, finalize_candidate, run_baselines,
)


class ScientificWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(7)
        n = 300
        x = rng.normal(size=n)
        self.frame = pd.DataFrame({
            "numeric": x,
            "category": np.where(x > 0, "positive", "negative"),
            "target": np.where(x > 0, "yes", "no"),
        })
        self.csv = self.root / "input.csv"
        self.frame.to_csv(self.csv, index=False)
        self.output = self.root / "project"
        self.spec = {"project_id": "fixture", "target": "target", "task": "binary_classification",
                     "metric": "balanced_accuracy", "split": "stratified", "positive_label": "yes"}

    def tearDown(self):
        self.temp.cleanup()

    def test_classification_audit_baselines_frozen_bundle_and_test_gate(self):
        manifest = audit_csv(self.csv, self.spec, self.output)
        self.assertEqual(len(set(manifest["splits"]["train"]) & set(manifest["splits"]["test"])), 0)
        result = run_baselines(self.output)
        self.assertEqual({x["candidate_id"] for x in result["candidates"]}, {"dummy", "linear", "tree"})
        scores = {x["candidate_id"]: x["metrics"]["balanced_accuracy"] for x in result["candidates"]}
        self.assertGreater(scores["linear"], scores["dummy"])
        final = finalize_candidate(self.output, "linear", test_access_id="test-once")
        self.assertEqual(final["status"], "verified_development_only")
        self.assertTrue((self.output / "bundle" / "frozen_model.joblib").is_file())
        bundle = self.output / "bundle"
        code = "import importlib.util,json; p=%r; s=importlib.util.spec_from_file_location('bundle_inference',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); rows=json.load(open(%r)); rows[0]['category']='never-seen'; print(json.dumps(m.predict(rows)))" % (str(bundle / "inference.py"), str(bundle / "examples_input.json"))
        actual = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
        self.assertEqual(len(actual["prediction"]), 5)
        self.assertTrue(np.isfinite(np.asarray(actual["probabilities"])).all())
        bundle_manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertIn("training/source/workflow.py", bundle_manifest["files"])
        gate = json.loads((self.output / "test_access.json").read_text())
        self.assertEqual(gate["status"], "consumed")
        with self.assertRaises(WorkflowError):
            finalize_candidate(self.output, "linear", test_access_id="test-again")

    def test_interrupted_sealed_assessment_cannot_be_replayed(self):
        audit_csv(self.csv, self.spec, self.output)
        run_baselines(self.output)
        with patch("mlrescurix_worker.scientific.workflow._run_isolated_predict", side_effect=RuntimeError("sealed scoring crash")):
            with self.assertRaisesRegex(RuntimeError, "sealed scoring crash"):
                finalize_candidate(self.output, "linear", test_access_id="crashed-assessment")
        self.assertEqual(json.loads((self.output / "test_access.json").read_text())["status"], "consumed")
        with patch("mlrescurix_worker.scientific.workflow._run_isolated_predict") as predict:
            with self.assertRaisesRegex(WorkflowError, "already consumed"):
                finalize_candidate(self.output, "linear", test_access_id="crashed-assessment")
            predict.assert_not_called()

    def test_grouped_split_has_no_group_leakage(self):
        frame = self.frame.copy()
        frame["group"] = np.repeat(np.arange(60), 5)
        frame["target"] = np.tile(["yes", "no", "yes", "no", "yes"], 60)
        frame.to_csv(self.csv, index=False)
        spec = {**self.spec, "split": "grouped", "group_column": "group"}
        manifest = audit_csv(self.csv, spec, self.output)
        groups = frame["group"].to_numpy()
        partitions = [set(groups[manifest["splits"][key]]) for key in ("train", "validation", "test")]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])

    def test_regression_random_split_and_metrics(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"x": rng.normal(size=120)})
        frame["target"] = frame["x"] * 2 + rng.normal(scale=.1, size=120)
        frame.to_csv(self.csv, index=False)
        spec = {"project_id": "reg", "target": "target", "task": "regression", "metric": "rmse", "split": "random"}
        audit_csv(self.csv, spec, self.output)
        results = run_baselines(self.output)
        self.assertIn("rmse", results["candidates"][1]["metrics"])

    def test_multiclass_log_loss_with_explicit_labels(self):
        x = np.tile(np.arange(90), 4)
        frame = pd.DataFrame({"x": x, "category": np.where(x < 30, "a", np.where(x < 60, "b", "c")),
                              "target": np.where(x < 30, "left", np.where(x < 60, "middle", "right"))})
        frame.to_csv(self.csv, index=False)
        spec = {"project_id": "multi", "target": "target", "task": "multiclass_classification",
                "metric": "log_loss", "split": "stratified"}
        audit_csv(self.csv, spec, self.output)
        results = run_baselines(self.output)
        self.assertIn("log_loss", results["candidates"][0]["metrics"])
        self.assertEqual(len(json.loads((self.output / "dataset_manifest.json").read_text())["labels"]), 3)

    def test_rejects_missing_targets_and_changed_snapshot(self):
        frame = self.frame.copy()
        frame.loc[0, "target"] = None
        frame.to_csv(self.csv, index=False)
        with self.assertRaises(WorkflowError):
            audit_csv(self.csv, self.spec, self.output)

    def test_chronological_split_preserves_input_order(self):
        frame = self.frame.copy()
        frame["target"] = np.tile(["yes", "no"], len(frame) // 2)
        frame.to_csv(self.csv, index=False)
        spec = {**self.spec, "split": "chronological"}
        manifest = audit_csv(self.csv, spec, self.output)
        parts = manifest["splits"]
        self.assertLess(max(parts["train"]), min(parts["validation"]))
        self.assertLess(max(parts["validation"]), min(parts["test"]))

    def test_declared_inference_limit_fails_before_test_gate(self):
        spec = {**self.spec, "max_latency_ms": 0.000001}
        audit_csv(self.csv, spec, self.output)
        run_baselines(self.output)
        final = finalize_candidate(self.output, "linear", test_access_id="blocked-test")
        self.assertEqual(final["status"], "provisional_infeasible")
        self.assertEqual(final["test_access"], "not_consumed")
        self.assertFalse((self.output / "test_access.json").exists())


if __name__ == "__main__":
    unittest.main()
