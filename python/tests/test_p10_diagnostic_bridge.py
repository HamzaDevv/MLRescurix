"""Unit tests for Phase P10.2 Structured Diagnostic Feedback Bridge.

Verifies:
1. Schema validation against Draft 2020-12 and rejection of invalid receipts.
2. Failure mode classification and attribution across:
   a) Cgroup OOM / memory ceiling violation.
   b) Non-finite loss / NaN predictions.
   c) Feature shape / column mismatch between fit and predict.
   d) Execution timeout clamp.
   e) Syntax error and unhandled runtime exception.
   f) Class order mismatch.
   g) Missing dependency.
3. Traceback parsing and root-cause attribution (file, line number, symbol, exception class).
4. Candidate runner and CLI integration emitting structured DiagnosticReceipt on failure.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.candidate_runner import (
    CandidateExecutionError,
    evaluate_candidate_cv,
)
from mlrescurix_worker.scientific.diagnostic_bridge import (
    DiagnosticBridge,
    DiagnosticReceipt,
    FailureType,
    ResourceTelemetry,
    RootCause,
    get_diagnostic_schema,
    validate_diagnostic_receipt,
)


class TestP10DiagnosticBridge(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="mlr-test-p102-diag-")
        self.td = Path(self.temp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_p10_2_schema_validation_against_draft_2020_12(self) -> None:
        """Verify DiagnosticReceipt adheres to Draft 2020-12 schema and invalid payloads fail."""
        schema = get_diagnostic_schema()
        validator = jsonschema.Draft202012Validator(schema)

        # Valid receipt
        valid_receipt = DiagnosticReceipt(
            diagnostic_id="diag-test-123",
            candidate_id="cand_test",
            stage="fit",
            failure_type=FailureType.NON_FINITE_LOSS,
            error_message="Loss is NaN during epoch 2",
            root_cause=RootCause(file="model.py", line_number=42, symbol="train_epoch", exception_class="FloatingPointError"),
            traceback_summary="Traceback (most recent call last):\n  File 'model.py', line 42, in train_epoch\nFloatingPointError: Loss is NaN",
            resource_telemetry=ResourceTelemetry(exit_code=1, wall_elapsed_ms=1200),
            remediation_suggestions=["Apply gradient clipping", "Add epsilon smoothing"],
        ).to_dict()

        validator.validate(valid_receipt)

        # Invalid: missing required field 'failure_type'
        bad_receipt = dict(valid_receipt)
        del bad_receipt["failure_type"]
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(bad_receipt)

        # Invalid: unsupported failure_type enum
        bad_enum = dict(valid_receipt)
        bad_enum["failure_type"] = "random_unsupported_failure"
        with self.assertRaises(jsonschema.ValidationError):
            validator.validate(bad_enum)

    def test_p10_2_cgroup_oom_fault_injection(self) -> None:
        """Inject intentional Cgroup OOM / memory ceiling violation and verify diagnostic receipt."""
        # Case A: Exit code 137 (SIGKILL / cgroup OOM killer)
        stderr_oom = "Killed\nProcess killed by cgroup out of memory controller"
        diag = DiagnosticBridge.diagnose_failure(
            exit_code=137,
            stderr=stderr_oom,
            candidate_id="cand_oom_01",
            stage="fit",
            memory_limit_bytes=512 * 1024 * 1024,
            wall_elapsed_ms=4500,
        )

        self.assertEqual(diag.failure_type, FailureType.CGROUP_OOM)
        self.assertTrue(diag.resource_telemetry.oom_killed)
        self.assertEqual(diag.resource_telemetry.exit_code, 137)
        self.assertEqual(diag.resource_telemetry.memory_limit_bytes, 512 * 1024 * 1024)
        self.assertTrue(any("batch size" in s.lower() for s in diag.remediation_suggestions))
        self.assertTrue(any("float32" in s.lower() for s in diag.remediation_suggestions))
        diag.validate()

        # Case B: MemoryError exception in stderr
        mem_exc = (
            "Traceback (most recent call last):\n"
            "  File \"candidate/features.py\", line 55, in build_interactions\n"
            "    dense = np.zeros((100000, 50000), dtype=np.float64)\n"
            "MemoryError: Unable to allocate 37.3 GiB for an array with shape (100000, 50000)"
        )
        diag_b = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=mem_exc,
            candidate_id="cand_oom_02",
            stage="fit",
        )
        self.assertEqual(diag_b.failure_type, FailureType.CGROUP_OOM)
        self.assertEqual(diag_b.root_cause.line_number, 55)
        self.assertEqual(diag_b.root_cause.symbol, "build_interactions")
        diag_b.validate()

    def test_p10_2_non_finite_loss_fault_injection(self) -> None:
        """Inject non-finite loss / NaN predictions and verify diagnostic receipt."""
        stderr_nan = (
            "Epoch 1: loss=1.42\n"
            "Epoch 2: loss=nan\n"
            "Traceback (most recent call last):\n"
            "  File \"candidate/model.py\", line 104, in fit\n"
            "    raise FloatingPointError('Loss is nan during training')\n"
            "FloatingPointError: Loss is nan during training"
        )
        diag = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_nan,
            candidate_id="cand_nan_01",
            stage="fit",
        )

        self.assertEqual(diag.failure_type, FailureType.NON_FINITE_LOSS)
        self.assertEqual(diag.root_cause.line_number, 104)
        self.assertEqual(diag.root_cause.symbol, "fit")
        self.assertEqual(diag.root_cause.exception_class, "FloatingPointError")
        self.assertTrue(any("epsilon" in s.lower() or "gradient clipping" in s.lower() for s in diag.remediation_suggestions))
        diag.validate()

    def test_p10_2_shape_mismatch_fault_injection(self) -> None:
        """Inject feature shape / column mismatch and verify diagnostic receipt."""
        stderr_shape = (
            "Traceback (most recent call last):\n"
            "  File \"candidate/model.py\", line 68, in predict\n"
            "    return self.estimator.predict(X)\n"
            "  File \"/usr/local/lib/python3.11/site-packages/sklearn/base.py\", line 450, in predict\n"
            "ValueError: Number of features of the model must match the input. Model n_features is 12 and input n_features is 10"
        )
        diag = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_shape,
            candidate_id="cand_shape_01",
            stage="predict",
        )

        self.assertEqual(diag.failure_type, FailureType.SHAPE_MISMATCH)
        self.assertEqual(diag.root_cause.line_number, 68)
        self.assertEqual(diag.root_cause.symbol, "predict")
        self.assertEqual(diag.root_cause.exception_class, "ValueError")
        self.assertTrue(any("identical columns" in s.lower() or "one-hot" in s.lower() for s in diag.remediation_suggestions))
        diag.validate()

    def test_p10_2_execution_timeout_fault_injection(self) -> None:
        """Inject execution deadline timeout and verify diagnostic receipt."""
        diag = DiagnosticBridge.diagnose_failure(
            exit_code=124,
            stderr="TimeoutExpired: candidate execution exceeded deadline of 45s",
            candidate_id="cand_to_01",
            stage="fit",
            wall_elapsed_ms=45000,
            timeout_ms=45000,
        )

        self.assertEqual(diag.failure_type, FailureType.EXECUTION_TIMEOUT)
        self.assertEqual(diag.resource_telemetry.exit_code, 124)
        self.assertEqual(diag.resource_telemetry.wall_elapsed_ms, 45000)
        self.assertTrue(any("vectorize" in s.lower() or "early stopping" in s.lower() for s in diag.remediation_suggestions))
        diag.validate()

    def test_p10_2_syntax_error_and_runtime_exception_fault_injection(self) -> None:
        """Inject syntax error and unhandled runtime exceptions."""
        # 1. Syntax Error
        stderr_syn = (
            "  File \"candidate/features.py\", line 28\n"
            "    def preprocess(df)\n"
            "                      ^\n"
            "SyntaxError: expected ':'"
        )
        diag_syn = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_syn,
            candidate_id="cand_syn_01",
            stage="fit",
        )
        self.assertEqual(diag_syn.failure_type, FailureType.SYNTAX_ERROR)
        self.assertEqual(diag_syn.root_cause.line_number, 28)
        self.assertEqual(diag_syn.root_cause.exception_class, "SyntaxError")
        diag_syn.validate()

        # 2. Runtime Exception (ZeroDivisionError)
        stderr_zero = (
            "Traceback (most recent call last):\n"
            "  File \"candidate/features.py\", line 15, in compute_ratios\n"
            "    ratio = x / std_dev\n"
            "ZeroDivisionError: float division by zero"
        )
        diag_zero = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_zero,
            candidate_id="cand_zero_01",
            stage="fit",
        )
        self.assertEqual(diag_zero.failure_type, FailureType.RUNTIME_EXCEPTION)
        self.assertEqual(diag_zero.root_cause.line_number, 15)
        self.assertEqual(diag_zero.root_cause.symbol, "compute_ratios")
        self.assertEqual(diag_zero.root_cause.exception_class, "ZeroDivisionError")
        self.assertTrue(any("zero-guard" in s.lower() for s in diag_zero.remediation_suggestions))
        diag_zero.validate()

    def test_p10_2_class_order_and_missing_dependency(self) -> None:
        """Inject class order discrepancy and missing dependency."""
        # 1. Class Order Mismatch
        stderr_co = (
            "WorkflowError: class order discrepancy: prediction output missing probability columns ['prob_B']. "
            "Columns must match manifest class order ['A', 'B']"
        )
        diag_co = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_co,
            candidate_id="cand_co_01",
            stage="predict",
        )
        self.assertEqual(diag_co.failure_type, FailureType.CLASS_ORDER_MISMATCH)
        self.assertTrue(any("prob_<class>" in s for s in diag_co.remediation_suggestions))
        diag_co.validate()

        # 2. Missing Dependency
        stderr_mod = (
            "Traceback (most recent call last):\n"
            "  File \"candidate/entrypoint.py\", line 4, in <module>\n"
            "    import optuna\n"
            "ModuleNotFoundError: No module named 'optuna'"
        )
        diag_mod = DiagnosticBridge.diagnose_failure(
            exit_code=1,
            stderr=stderr_mod,
            candidate_id="cand_mod_01",
            stage="fit",
        )
        self.assertEqual(diag_mod.failure_type, FailureType.MISSING_DEPENDENCY)
        self.assertTrue(any("offline isolated container" in s for s in diag_mod.remediation_suggestions))
        diag_mod.validate()

    def test_p10_2_candidate_runner_cv_fault_injection(self) -> None:
        """Verify that evaluate_candidate_cv attaches high-signal DiagnosticReceipt on candidate failure."""
        cand_dir = self.td / "cand_failing"
        cand_dir.mkdir(parents=True, exist_ok=True)

        # Create entrypoint that intentionally crashes with a divide-by-zero runtime exception on fit
        entrypoint_py = (
            "import sys, argparse\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('action')\n"
            "    parser.add_argument('--train-path')\n"
            "    parser.add_argument('--model-dir')\n"
            "    args = parser.parse_args()\n"
            "    if args.action == 'fit':\n"
            "        # Injected runtime exception\n"
            "        x = 10 / 0\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        (cand_dir / "entrypoint.py").write_text(entrypoint_py)

        from mlrescurix_worker.scientific.candidate_runner import compute_source_hash
        src_hash = compute_source_hash(cand_dir)

        manifest = {
            "candidate_id": "cand_failing_cv",
            "name": "Failing Candidate",
            "task_type": "binary",
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": "row_id",
            "class_order": [0, 1],
            "source_hash": src_hash,
            "raw_input_schema": {"target": "target"},
        }
        (cand_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        # Create dummy dataset
        ds_csv = self.td / "dataset.csv"
        df = pd.DataFrame({
            "row_id": list(range(10)),
            "f1": np.linspace(0.0, 1.0, 10),
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })
        df.to_csv(ds_csv, index=False)

        with self.assertRaises(CandidateExecutionError) as ctx:
            evaluate_candidate_cv(
                dataset_csv=ds_csv,
                manifest=manifest,
                candidate_dir=cand_dir,
                folds=2,
            )

        err = ctx.exception
        self.assertIsNotNone(err.diagnostic_receipt, "Expected attached DiagnosticReceipt on CandidateExecutionError")
        receipt = err.diagnostic_receipt
        self.assertIsInstance(receipt, DiagnosticReceipt)
        self.assertEqual(receipt.candidate_id, "cand_failing_cv")
        self.assertEqual(receipt.stage, "fit")
        self.assertEqual(receipt.failure_type, FailureType.RUNTIME_EXCEPTION)
        self.assertEqual(receipt.root_cause.exception_class, "ZeroDivisionError")
        self.assertEqual(receipt.root_cause.line_number, 10)
        self.assertEqual(receipt.root_cause.symbol, "main")
        receipt.validate()


if __name__ == "__main__":
    unittest.main()
