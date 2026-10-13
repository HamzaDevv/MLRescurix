"""M6 Bundle verification tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from mlrescurix_worker.scientific.bundle import verify_bundle, _sha256


class M6BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bundle_dir = Path(self.tmp.name) / "bundle"
        self.bundle_dir.mkdir()

        # Create minimal contract files
        self.files = {
            "input_schema.json": json.dumps({"features": [{"name": "f1", "type": "float"}]}),
            "scorecard.json": json.dumps({"candidate_id": "cand_1", "metric": "balanced_accuracy", "primary_metric": 0.85, "warm_p95_ms": 2.5, "constraints": {}}),
            "verification.json": json.dumps({"frozen_model_sha256": "placeholder_sha", "all_passed": True}),
            "final.json": json.dumps({"candidate_id": "cand_1", "status": "completed"}),
            "frozen_model.joblib": "model_bytes_12345",
            "inference.py": "def predict(data): return [0]",
            "inference.lock": json.dumps({"python": "3.13"}),
            "examples_input.json": json.dumps([{"f1": 1.0}]),
            "examples_expected.json": json.dumps({"predictions": [0]}),
            "README.md": "# Model Bundle",
        }
        for name, content in self.files.items():
            (self.bundle_dir / name).write_text(content)

        # Update verification.json with real sha
        real_sha = _sha256(self.bundle_dir / "frozen_model.joblib")
        (self.bundle_dir / "verification.json").write_text(
            json.dumps({"frozen_model_sha256": real_sha, "all_passed": True})
        )

        # Build manifest.json
        manifest_files = {}
        for name in (
            "input_schema.json",
            "scorecard.json",
            "verification.json",
            "final.json",
            "frozen_model.joblib",
            "inference.py",
            "inference.lock",
            "examples_input.json",
            "examples_expected.json",
            "README.md",
        ):
            p = self.bundle_dir / name
            manifest_files[name] = {"bytes": p.stat().st_size, "sha256": _sha256(p)}

        self.manifest = {
            "format": "mlrescurix.bundle.v1",
            "candidate_id": "cand_1",
            "project_id": "proj_1",
            "files": manifest_files,
        }
        (self.bundle_dir / "manifest.json").write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_bundle_verification(self):
        report = verify_bundle(self.bundle_dir, run_sandbox=False)
        self.assertEqual(report["status"], "integrity_verified")
        self.assertEqual(report["candidate_id"], "cand_1")
        self.assertTrue(report["manifest_verified"])

    def test_sandbox_request_refuses_host_execution(self):
        (self.bundle_dir / "inference.py").write_text("raise RuntimeError('HOST_EXECUTED')")
        entry = self.bundle_dir / "inference.py"
        self.manifest['files']['inference.py'] = {'bytes': entry.stat().st_size, 'sha256': _sha256(entry)}
        (self.bundle_dir / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Host execution is prohibited"):
            verify_bundle(self.bundle_dir, run_sandbox=True)

    def test_symlink_root_rejected(self):
        link = self.bundle_dir.parent / "link"
        link.symlink_to(self.bundle_dir, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify_bundle(link, run_sandbox=False)

    def test_unverified_constraint_rejected(self):
        path = self.bundle_dir / 'scorecard.json'
        score = json.loads(path.read_text())
        score['constraints'] = {'latency': {'verdict': 'unverified'}}
        path.write_text(json.dumps(score))
        self.manifest['files']['scorecard.json'] = {'bytes': path.stat().st_size, 'sha256': _sha256(path)}
        (self.bundle_dir / "manifest.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "verified pass"):
            verify_bundle(self.bundle_dir, run_sandbox=False)

    def test_tampered_model_rejected(self):
        model_path = self.bundle_dir / "frozen_model.joblib"
        model_path.write_text("tampered_content")
        with self.assertRaises(ValueError) as ctx:
            verify_bundle(self.bundle_dir, run_sandbox=False)
        self.assertIn("mismatch", str(ctx.exception).lower())

    def test_untracked_file_rejected(self):
        (self.bundle_dir / "untracked.py").write_text("malicious code")
        with self.assertRaises(ValueError) as ctx:
            verify_bundle(self.bundle_dir, run_sandbox=False)
        self.assertIn("untracked file", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
