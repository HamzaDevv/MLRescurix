import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from mlrescurix_worker.scientific.export import (
    DEFAULT_TOLERANCES,
    SafeTensorsPredictor,
    export_model,
    verify_export_parity,
)
from mlrescurix_worker.scientific.benchmarks import (
    BenchmarkSuiteReport,
    StrategyBenchmarkResult,
    TrialRecord,
    generate_benchmark_dataset,
    run_prospective_benchmark,
    save_benchmark_report,
)
from mlrescurix_worker.scientific.serve import (
    create_serving_server,
    BundleServingHandler,
)
import threading
import urllib.request


class TestP7ScientificDelivery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p7_test_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_model_export_safetensors_and_tolerances(self):
        # 1. Classification
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        y = np.array([0, 0, 1, 1], dtype=int)
        clf = LogisticRegression().fit(X, y)

        exp_dir = self.tmp / "clf_export"
        manifest = export_model(
            clf, exp_dir, formats=["joblib", "safetensors"],
            task="binary_classification", features=["f1", "f2"], classes=[0, 1]
        )
        self.assertTrue((exp_dir / "weights.safetensors").is_file())
        self.assertTrue((exp_dir / "tolerances.json").is_file())
        self.assertEqual(manifest["task"], "binary_classification")
        self.assertIn("joblib", manifest["formats"])
        self.assertIn("safetensors", manifest["formats"])

        # Parity check
        test_rows = [{"f1": 0.15, "f2": 0.25}, {"f1": 0.65, "f2": 0.75}]
        parity = verify_export_parity(clf, exp_dir, test_rows, task="binary_classification")
        self.assertTrue(parity["verified"])
        self.assertLess(parity["formats"]["safetensors"]["max_prob_diff"], 1e-5)

        # Standalone Predictor
        pred = SafeTensorsPredictor(exp_dir)
        out = pred.predict(test_rows)
        self.assertEqual(out["prediction"], [0, 1])

        metadata = exp_dir / "safetensors_meta.json"
        metadata.write_text(metadata.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "artifact integrity mismatch"):
            verify_export_parity(clf, exp_dir, test_rows, task="binary_classification")

        # Sandbox traversal check: must reject '..'
        with self.assertRaises(ValueError) as ctx:
            export_model(clf, exp_dir / ".." / "escaped", formats=["safetensors"])
        self.assertIn("escapes sandbox boundaries", str(ctx.exception))

    def test_joblib_parity_checks_the_saved_artifact(self):
        import joblib
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        model = Ridge().fit(X, [0.0, 1.0, 2.0, 3.0])
        export_dir = self.tmp / "joblib_export"
        export_model(model, export_dir, formats=["joblib"], task="regression", features=["x"])
        rows = [{"x": 0.5}, {"x": 2.5}]
        self.assertTrue(verify_export_parity(model, export_dir, rows, task="regression")["formats"]["joblib"]["verified"])
        for invalid in [[{}], [{"x": 0.5, "unexpected": 1}]]:
            with self.assertRaisesRegex(ValueError, "declared features exactly"):
                verify_export_parity(model, export_dir, invalid, task="regression")
        from unittest.mock import patch
        with patch.object(model, "predict", return_value=np.array([float("inf"), 0.0])):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                verify_export_parity(model, export_dir, rows, task="regression")
        joblib.dump(Ridge().fit(X, [3.0, 2.0, 1.0, 0.0]), export_dir / "frozen_model.joblib")
        with self.assertRaisesRegex(ValueError, "artifact integrity mismatch"):
            verify_export_parity(model, export_dir, rows, task="regression")
        # Even if a manifest is rewritten to match a substituted model, parity
        # must still compare predictions with the original trusted estimator.
        import hashlib
        manifest_path = export_dir / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["formats"]["joblib"]["sha256"] = hashlib.sha256((export_dir / "frozen_model.joblib").read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Joblib prediction parity violation"):
            verify_export_parity(model, export_dir, rows, task="regression")
        manifest = json.loads(manifest_path.read_text())
        manifest["formats"] = {}
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "No verifiable exported formats"):
            verify_export_parity(model, export_dir, rows, task="regression")

    def test_model_export_regression(self):
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32)
        y = np.array([3.0, 5.0, 7.0, 9.0], dtype=np.float32)
        reg = Ridge().fit(X, y)

        exp_dir = self.tmp / "reg_export"
        manifest = export_model(
            reg, exp_dir, formats=["safetensors"],
            task="regression", features=["a", "b"]
        )
        self.assertEqual(manifest["tolerances"]["prediction_atol"], DEFAULT_TOLERANCES["regression"]["prediction_atol"])

        test_rows = [{"a": 1.5, "b": 2.5}]
        pred = SafeTensorsPredictor(exp_dir)
        res = pred.predict(test_rows)
        parity = verify_export_parity(reg, exp_dir, test_rows, task="regression")
        self.assertEqual(set(parity["formats"]), {"safetensors"})
        with self.assertRaisesRegex(ValueError, "at least one input"):
            verify_export_parity(reg, exp_dir, [], task="regression")
        self.assertEqual(len(res["prediction"]), 1)
        expected_val = float(reg.predict(np.array([[1.5, 2.5]], dtype=np.float32))[0])
        self.assertAlmostEqual(res["prediction"][0], expected_val, places=5)

    def test_serve_http_endpoints(self):
        # Create minimal bundle
        bundle = self.tmp / "serve_bundle"
        bundle.mkdir()
        inf_code = '''
def predict(records):
    results = []
    for r in records:
        if 'a' not in r or 'b' not in r:
            raise ValueError(f"missing feature in {r}")
        results.append(int(r['a'] + r['b'] > 5))
    return {"prediction": results, "count": len(results)}
'''
        (bundle / "inference.py").write_text(inf_code)
        (bundle / "manifest.json").write_text(json.dumps({
            "format": "mlrescurix.bundle.v2",
            "candidate_id": "cand-serv-test",
        }))

        server = create_serving_server(bundle, host="127.0.0.1", port=8943)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        try:
            # 1. Health check
            with urllib.request.urlopen("http://127.0.0.1:8943/health", timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode())
                self.assertEqual(data["status"], "healthy")
                self.assertEqual(data["candidate_id"], "cand-serv-test")

            # 2. Predict valid
            req_data = json.dumps([{"a": 2.0, "b": 4.0}, {"a": 1.0, "b": 1.0}]).encode()
            req = urllib.request.Request("http://127.0.0.1:8943/predict", data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                pred_res = json.loads(resp.read().decode())
                self.assertEqual(pred_res["status"], "ok")
                self.assertEqual(pred_res["result"]["prediction"], [1, 0])

            # 3. Predict invalid (missing feature)
            bad_data = json.dumps([{"a": 2.0}]).encode()
            bad_req = urllib.request.Request("http://127.0.0.1:8943/predict", data=bad_data, headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(bad_req, timeout=5)
            self.assertEqual(ctx.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()

    def test_benchmarks_and_suboptimal_trials(self):
        report = run_prospective_benchmark(
            task="binary_classification",
            metric="accuracy",
            seeds=[2026, 42, 101],
            max_trials_per_strategy=6,
        )

        self.assertIn("random", report.results)
        self.assertIn("search", report.results)
        self.assertIn("agent", report.results)
        self.assertEqual(report.seeds, [2026, 42, 101])

        summary = report.comparative_summary
        self.assertTrue(summary["suboptimal_and_failing_trials_recorded"])
        self.assertIn("statistical_claim", summary)

        # Check that suboptimal / failing trials are recorded for agent
        agent_res = report.results["agent"]
        has_suboptimal = any(t.status in ("suboptimal", "failed_constraint") for t in agent_res.trials)
        self.assertTrue(has_suboptimal, "Agent must record suboptimal or failing trials")

        out_path = self.tmp / "benchmarks_out.json"
        save_benchmark_report(report, out_path)
        self.assertTrue(out_path.is_file())
        saved_data = json.loads(out_path.read_text())
        self.assertEqual(saved_data["suite_id"], report.suite_id)


if __name__ == "__main__":
    unittest.main()
