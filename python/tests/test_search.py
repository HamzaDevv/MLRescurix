"""Focused M3 tests for constrained search, adapter, and multi-fidelity execution."""
import http.client
import json
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.dashboard import create_server
from mlrescurix_worker.search.adapter import (
    DeterministicFallbackOptimizer,
    normalize_parameters,
    get_optimizer,
)
from mlrescurix_worker.scientific.workflow import (
    _get_fidelity_train_ids,
    audit_csv,
    finalize_candidate,
    run_baselines,
    run_trial,
    WorkflowError,
)


class SearchParameterTests(unittest.TestCase):
    def test_dummy_rejects_parameters(self):
        self.assertEqual(normalize_parameters("dummy", "binary_classification", {}), {})
        with self.assertRaises(ValueError):
            normalize_parameters("dummy", "binary_classification", {"C": 1.0})

    def test_linear_parameter_ranges(self):
        # Classification uses C
        self.assertEqual(normalize_parameters("linear", "binary_classification", {"C": 0.001}), {"C": 0.001})
        self.assertEqual(normalize_parameters("linear", "binary_classification", {"C": 1000.0}), {"C": 1000.0})
        with self.assertRaises(ValueError):
            normalize_parameters("linear", "binary_classification", {"C": 0.0001})
        with self.assertRaises(ValueError):
            normalize_parameters("linear", "binary_classification", {"C": 1000.1})
        with self.assertRaises(ValueError):
            normalize_parameters("linear", "binary_classification", {"alpha": 1.0})

        # Regression uses alpha
        self.assertEqual(normalize_parameters("linear", "regression", {"alpha": 1.0}), {"alpha": 1.0})
        with self.assertRaises(ValueError):
            normalize_parameters("linear", "regression", {"C": 1.0})

    def test_tree_parameter_ranges(self):
        valid = {"n_estimators": 64, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.5}
        norm = normalize_parameters("tree", "binary_classification", valid)
        self.assertEqual(norm["n_estimators"], 64)
        self.assertEqual(norm["max_depth"], 10)
        self.assertEqual(norm["min_samples_leaf"], 5)
        self.assertEqual(norm["max_features"], 0.5)

        # Bounds checks
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "n_estimators": 15})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "n_estimators": 129})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "max_depth": 1})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "max_depth": 21})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "min_samples_leaf": 0})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "min_samples_leaf": 21})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "max_features": 0.19})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "max_features": 1.01})
        with self.assertRaises(ValueError):
            normalize_parameters("tree", "binary_classification", {**valid, "unknown": 1})


class SearchOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlr-opt-test-")
        self.storage = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_fallback_optimizer_lifecycle(self):
        opt = DeterministicFallbackOptimizer(self.storage, "sess-1", "maximize", seed=2026)
        tid1, p1 = opt.ask("linear", 0.5, "binary_classification")
        self.assertEqual(tid1, 0)
        self.assertIn("C", p1)
        opt.tell("linear", 0.5, tid1, 0.85, "COMPLETE")

        tid2, p2 = opt.ask("linear", 0.5, "binary_classification")
        self.assertEqual(tid2, 1)
        self.assertIn("C", p2)

        # Separate study for fidelity=1.0
        tid_full, p_full = opt.ask("linear", 1.0, "binary_classification")
        self.assertEqual(tid_full, 0)

    def test_factory_creates_optimizer(self):
        opt = get_optimizer(self.storage, "sess-2", "minimize", seed=2026)
        tid, p = opt.ask("tree", 0.5, "regression")
        self.assertIn("n_estimators", p)
        opt.tell("tree", 0.5, tid, 1.25, "COMPLETE")


class MultiFidelityWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mlr-mf-test-")
        self.root = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_dataset(self, n=120):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "f1": rng.randn(n),
            "f2": rng.randn(n),
            "target": rng.choice([0, 1], size=n),
        })
        csv_path = self.root / "data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_multi_fidelity_screening_and_trial(self):
        csv_path = self._make_dataset(120)
        spec = {
            "project_id": "p1",
            "target": "target",
            "task": "binary_classification",
            "split": "stratified",
            "metric": "balanced_accuracy",
            "inference_batch_size": 16,
            "inference_repetitions": 3,
        }
        workflow = self.root / "workflow"
        manifest = audit_csv(csv_path, spec, workflow)
        self.assertEqual(manifest["format"], "mlrescurix.dataset.v1")

        # Screening trial at fidelity = 0.5
        exp_spec = {
            "experiment_id": "test-exp-1",
            "session_id": "session-1",
            "recipe": "linear",
            "parameters": {"C": 1.0},
            "seed": 2026,
            "fidelity": 0.5,
            "node_type": "screening",
        }
        trial = run_trial(workflow, exp_spec)
        self.assertEqual(trial["experiment_id"], "test-exp-1")
        self.assertEqual(trial["fidelity"], 0.5)
        self.assertTrue(trial["is_feasible"])
        self.assertIn("balanced_accuracy", trial["metrics"])
        self.assertTrue((workflow / "trials" / "test-exp-1" / "trial.json").is_file())

        # Full fidelity trial at fidelity = 1.0
        exp_full = {
            "experiment_id": "test-exp-full",
            "session_id": "session-1",
            "recipe": "linear",
            "parameters": {"C": 1.0},
            "seed": 2026,
            "fidelity": 1.0,
            "parent_experiment_id": "test-exp-1",
            "node_type": "full_fidelity",
        }
        full_trial = run_trial(workflow, exp_full)
        self.assertEqual(full_trial["fidelity"], 1.0)
        self.assertTrue(full_trial["is_feasible"])

    def test_bundle_size_checks_complete_bundle_staging(self):
        csv_path = self._make_dataset(120)
        spec = {
            "project_id": "p2",
            "target": "target",
            "task": "binary_classification",
            "split": "stratified",
            "metric": "balanced_accuracy",
            "inference_batch_size": 16,
            "inference_repetitions": 3,
            "max_bundle_bytes": 20_000, # 20KB is enough for model.joblib (~1.5KB) but less than staged bundle (~55KB)
        }
        workflow = self.root / "workflow"
        audit_csv(csv_path, spec, workflow)
        run_baselines(workflow)

        exp_spec = {
            "experiment_id": "candidate-1",
            "session_id": "session-1",
            "recipe": "linear",
            "parameters": {"C": 1.0},
            "seed": 2026,
            "fidelity": 1.0,
            "node_type": "full_fidelity",
        }
        tr = run_trial(workflow, exp_spec)
        self.assertEqual(tr["status"], "evaluated")
        self.assertTrue(tr["is_feasible"])

        # Finalization must check complete bundle size and reject before consuming test gate
        final_result = finalize_candidate(workflow, "candidate-1", test_access_id="test-access-1")
        self.assertEqual(final_result["status"], "provisional_infeasible")
        self.assertEqual(final_result["test_access"], "not_consumed")
        self.assertEqual(final_result["constraints"]["bundle_bytes"]["verdict"], "fail")


class DashboardSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state_dir = Path(self.temp.name) / "state"
        self.policy = Path(self.temp.name) / "policy.json"
        self.policy.write_text(json.dumps({
            "budget": {"calls": 100, "tokens": 50000, "wall_ms": 3600000},
            "capacity": {"workers": 2},
            "max_tasks": 100,
        }))
        self.binary = Path(__file__).resolve().parents[2] / "target" / "debug" / "mlrescurix-harness"

    def tearDown(self):
        self.temp.cleanup()

    def test_dashboard_login_and_snapshot_smoke(self):
        # 1. Initialize harness state
        init_res = subprocess.run([
            str(self.binary), "init",
            "--state", str(self.state_dir),
            "--policy", str(self.policy),
        ], capture_output=True, text=True)
        self.assertEqual(init_res.returncode, 0, init_res.stderr)

        # Trigger schema migrations up to v4
        subprocess.run([
            str(self.binary), "search-inspect",
            "--state", str(self.state_dir),
            "--project", "p-dash",
            "--session", "dummy",
        ], capture_output=True, text=True)

        # 2. Add a search session directly in sqlite to test projection
        token = "test-dashboard-secret-token-12345"
        with sqlite3.connect(self.state_dir / "state.sqlite") as db:
            now = int(time.time() * 1000)
            db.execute("INSERT INTO projects (id, revision, spec_revision, intent_revision, intent, finalization_id, state, policy) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       ("p-dash", 1, 1, 1, "test-intent", "final-p-dash", "created", self.policy.read_text()))
            db.execute("INSERT INTO scientific_datasets (project, id, path, bytes, sha256, registered_ms) VALUES (?, ?, ?, ?, ?, ?)",
                       ("p-dash", "dset-1", "/tmp/dset.csv", 1000, "sha256-dummy", now))
            db.execute("INSERT INTO search_sessions (id, project, dataset_id, spec, status, created_ms, updated_ms, apex_candidate_id, selection_record) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       ("sess-dash-1", "p-dash", "dset-1", json.dumps({"recipe": "linear", "metric": "balanced_accuracy"}), "completed", now, now, "cand-1", json.dumps({"apex_id": "cand-1", "rationale": "Strongest metric"})))
            db.execute("INSERT INTO experiment_dag (id, session_id, project, recipe, parameters, seed, fidelity, node_type, parent_id, created_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       ("exp-1", "sess-dash-1", "p-dash", "linear", json.dumps({"C": 1.0}), 2026, 1.0, "full_fidelity", None, now))
            db.execute("INSERT INTO trial_results (project, experiment_id, status, primary_metric, metrics, constraints, is_feasible, train_ms, model_sha256, predictions_sha256, artifact_path, committed_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       ("p-dash", "exp-1", "evaluated", 0.92, json.dumps({"balanced_accuracy": 0.92}), json.dumps({"latency_ms": {"verdict": "pass", "observed": 1.2}}), 1, 50, "sha-model", "sha-preds", "/tmp/trial.json", now))

        # 3. Create server on ephemeral port (port 0)
        server = create_server(self.state_dir, self.binary, 0, token)
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)

            # GET / -> 200, HTML page
            conn.request("GET", "/")
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            body = res.read().decode()
            self.assertIn("search-section", body)
            self.assertIn("renderSearch", body)

            # Snapshot without auth -> 401
            conn.request("GET", "/v1/projects/p-dash/snapshot")
            res = conn.getresponse()
            self.assertEqual(res.status, 401)
            res.read()

            # POST /session with bad token -> 401
            headers = {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"}
            conn.request("POST", "/session", body=json.dumps({"token": "wrong"}), headers=headers)
            res = conn.getresponse()
            self.assertEqual(res.status, 401)
            res.read()

            # POST /session with good token -> 200 and Set-Cookie
            conn.request("POST", "/session", body=json.dumps({"token": token}), headers=headers)
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            cookie_header = res.getheader("Set-Cookie")
            self.assertIn(f"mlr_session={token}", cookie_header)
            res.read()

            # GET /v1/projects/p-dash/snapshot with cookie -> 200 and search projection
            conn.request("GET", "/v1/projects/p-dash/snapshot", headers={"Cookie": f"mlr_session={token}"})
            res = conn.getresponse()
            self.assertEqual(res.status, 200)
            snap = json.loads(res.read().decode())
            self.assertIn("search_sessions", snap)
            self.assertEqual(len(snap["search_sessions"]), 1)
            self.assertEqual(snap["search_sessions"][0]["session_id"], "sess-dash-1")
            self.assertEqual(snap["search_sessions"][0]["apex_candidate_id"], "cand-1")

            self.assertIn("experiment_dag", snap)
            self.assertEqual(len(snap["experiment_dag"]), 1)
            self.assertEqual(snap["experiment_dag"][0]["experiment_id"], "exp-1")
            self.assertEqual(snap["experiment_dag"][0]["primary_metric"], 0.92)
            self.assertTrue(snap["experiment_dag"][0]["is_feasible"])

            conn.close()
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
