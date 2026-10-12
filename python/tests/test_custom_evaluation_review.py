"""Evaluator-only regressions: no candidate imports, fitting or deserialization."""
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from mlrescurix_worker.scientific.workflow import audit_csv, evaluate_custom_candidate, WorkflowError


class CustomEvaluationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        csv = self.root / "data.csv"
        pd.DataFrame({"feature": list(range(100)), "target": ["a", "b"] * 50}).to_csv(csv, index=False)
        self.workflow = self.root / "workflow"
        self.manifest = audit_csv(csv, {"project_id": "review", "target": "target", "task": "binary_classification", "metric": "balanced_accuracy"}, self.workflow)
        ids = self.manifest["splits"]["validation"]
        self.payload = {"row_ids": ids, "predictions": ["a"] * len(ids),
                        "classes": self.manifest["labels"], "probabilities": [[0.5, 0.5] for _ in ids]}
        self.predictions = self.root / "predictions.json"

    def evaluate(self, **kwargs):
        self.predictions.write_text(json.dumps(self.payload))
        return evaluate_custom_candidate(self.workflow, "candidate", self.predictions, **kwargs)

    def test_string_labels_are_valid(self):
        self.assertEqual(self.evaluate()["status"], "evaluated")

    def test_probability_rows_must_sum_to_one(self):
        self.payload["probabilities"][0] = [0.1, 0.1]
        with self.assertRaises(WorkflowError):
            self.evaluate()

    def test_class_order_is_required(self):
        self.payload["classes"] = list(reversed(self.payload["classes"]))
        with self.assertRaises(WorkflowError):
            self.evaluate()

    def test_untrusted_latency_cannot_certify_feasibility(self):
        self.payload["latency_ms"] = 0
        result = self.evaluate(constraints={"max_latency_ms": {"max": 100}})
        self.assertFalse(result["is_feasible"])
        self.assertEqual(result["constraints"]["max_latency_ms"]["verdict"], "unverified")
