"""Structured Diagnostic Feedback Bridge (Phase P10.2).

Captures execution failure classifications, root-cause line/symbol attribution,
bounded formatted tracebacks, peak resource telemetry, and actionable remediation
suggestions for autonomous agent self-healing and code debugging.
"""
from __future__ import annotations

import json
import os
import re
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jsonschema


SCHEMA_PATH = Path(__file__).resolve().parents[4] / "contracts" / "v2" / "diagnostic_receipt.schema.json"


class FailureType:
    CGROUP_OOM = "cgroup_oom"
    NON_FINITE_LOSS = "non_finite_loss"
    SHAPE_MISMATCH = "shape_mismatch"
    CLASS_ORDER_MISMATCH = "class_order_mismatch"
    EXECUTION_TIMEOUT = "execution_timeout"
    MISSING_DEPENDENCY = "missing_dependency"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_EXCEPTION = "runtime_exception"

    ALL = {
        CGROUP_OOM,
        NON_FINITE_LOSS,
        SHAPE_MISMATCH,
        CLASS_ORDER_MISMATCH,
        EXECUTION_TIMEOUT,
        MISSING_DEPENDENCY,
        SYNTAX_ERROR,
        RUNTIME_EXCEPTION,
    }


@dataclass
class RootCause:
    file: str | None = None
    line_number: int | None = None
    symbol: str | None = None
    exception_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line_number": self.line_number,
            "symbol": self.symbol,
            "exception_class": self.exception_class,
        }


@dataclass
class ResourceTelemetry:
    exit_code: int | None = None
    peak_rss_bytes: int | None = None
    memory_limit_bytes: int | None = None
    wall_elapsed_ms: int | None = None
    timeout_ms: int | None = None
    oom_killed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {"exit_code": self.exit_code}
        if self.peak_rss_bytes is not None:
            res["peak_rss_bytes"] = self.peak_rss_bytes
        if self.memory_limit_bytes is not None:
            res["memory_limit_bytes"] = self.memory_limit_bytes
        if self.wall_elapsed_ms is not None:
            res["wall_elapsed_ms"] = self.wall_elapsed_ms
        if self.timeout_ms is not None:
            res["timeout_ms"] = self.timeout_ms
        if self.oom_killed is not None:
            res["oom_killed"] = self.oom_killed
        return res


@dataclass
class DiagnosticReceipt:
    schema_version: str = "v2"
    diagnostic_id: str = field(default_factory=lambda: f"diag-{uuid.uuid4().hex[:12]}")
    candidate_id: str = "candidate"
    stage: str = "evaluate"
    failure_type: str = FailureType.RUNTIME_EXCEPTION
    error_message: str = ""
    root_cause: RootCause = field(default_factory=RootCause)
    traceback_summary: str = ""
    resource_telemetry: ResourceTelemetry = field(default_factory=ResourceTelemetry)
    remediation_suggestions: list[str] = field(default_factory=list)
    context_metadata: dict[str, Any] = field(default_factory=dict)
    created_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "diagnostic_id": self.diagnostic_id,
            "candidate_id": self.candidate_id,
            "stage": self.stage,
            "failure_type": self.failure_type,
            "error_message": self.error_message[:2048] if self.error_message else "Execution failure",
            "root_cause": self.root_cause.to_dict(),
            "traceback_summary": self.traceback_summary[:8192],
            "resource_telemetry": self.resource_telemetry.to_dict(),
            "remediation_suggestions": self.remediation_suggestions if self.remediation_suggestions else ["Inspect execution traceback and check candidate inputs."],
        }
        if self.context_metadata:
            d["context_metadata"] = self.context_metadata
        if self.created_ms > 0:
            d["created_ms"] = self.created_ms
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def validate(self) -> None:
        validate_diagnostic_receipt(self.to_dict())


_CACHED_VALIDATOR: jsonschema.Draft202012Validator | None = None


def get_diagnostic_schema() -> dict[str, Any]:
    if SCHEMA_PATH.is_file():
        return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    # Fallback inline schema if path resolution differs in isolated test sandbox
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:mlrescurix:diagnostic-receipt:2",
        "type": "object",
        "required": [
            "schema_version",
            "diagnostic_id",
            "candidate_id",
            "stage",
            "failure_type",
            "error_message",
            "root_cause",
            "traceback_summary",
            "resource_telemetry",
            "remediation_suggestions"
        ],
        "properties": {
            "schema_version": {"type": "string", "const": "v2"},
            "diagnostic_id": {"type": "string"},
            "candidate_id": {"type": "string"},
            "stage": {"type": "string"},
            "failure_type": {"type": "string"},
            "error_message": {"type": "string"},
            "root_cause": {"type": "object"},
            "traceback_summary": {"type": "string"},
            "resource_telemetry": {"type": "object"},
            "remediation_suggestions": {"type": "array"}
        }
    }


def validate_diagnostic_receipt(receipt_dict: dict[str, Any]) -> None:
    global _CACHED_VALIDATOR
    if _CACHED_VALIDATOR is None:
        schema = get_diagnostic_schema()
        _CACHED_VALIDATOR = jsonschema.Draft202012Validator(schema)
    _CACHED_VALIDATOR.validate(receipt_dict)


class DiagnosticBridge:
    """Diagnostic parser and feedback generator for candidate execution failures."""

    # Traceback frame pattern: File "...", line 123, in func_name
    FRAME_REGEX = re.compile(
        r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:, in (?P<symbol>.+))?'
    )

    # Common exception pattern: ExceptionClass: message
    EXC_REGEX = re.compile(
        r'^(?P<cls>[A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|Warning|LossException|LeakageError))(?::\s*(?P<msg>.*))?$',
        re.MULTILINE
    )

    @classmethod
    def parse_traceback(cls, text: str, candidate_dir: str | None = None) -> tuple[RootCause, str]:
        """Extract root-cause attribution (file, line, symbol, exception) and bounded summary."""
        lines = text.strip().splitlines()
        if not lines:
            return RootCause(), ""

        # Extract exception class and message from trailing lines
        exc_class: str | None = None
        exc_msg: str | None = None
        for line in reversed(lines):
            line_str = line.strip()
            match = cls.EXC_REGEX.match(line_str)
            if match:
                exc_class = match.group("cls").strip()
                exc_msg = (match.group("msg") or "").strip()
                break

        # Extract frames
        frames: list[tuple[str, int, str | None]] = []
        for i, line in enumerate(lines):
            m = cls.FRAME_REGEX.search(line)
            if m:
                f_path = m.group("file").strip()
                l_num = int(m.group("line"))
                sym = m.group("symbol")
                sym_str = sym.strip() if sym else None
                frames.append((f_path, l_num, sym_str))

        root_file: str | None = None
        root_line: int | None = None
        root_sym: str | None = None

        if frames:
            # Prefer candidate workspace frames over site-packages/internal python frames
            candidate_frames = []
            for f_path, l_num, sym in frames:
                # Normalize path
                posix_path = f_path.replace("\\", "/")
                is_candidate = (
                    "candidate" in posix_path
                    or (candidate_dir and candidate_dir in posix_path)
                    or ("site-packages" not in posix_path and "lib/python" not in posix_path and "dist-packages" not in posix_path)
                )
                if is_candidate:
                    candidate_frames.append((f_path, l_num, sym))

            # Use deepest candidate frame or deepest frame
            target_frame = candidate_frames[-1] if candidate_frames else frames[-1]
            root_file = target_frame[0]
            root_line = target_frame[1]
            root_sym = target_frame[2]

            # Simplify relative candidate path if possible
            if "candidate/" in root_file:
                root_file = root_file[root_file.find("candidate/"):]
            elif "/" in root_file:
                # Keep filename or relative subpath
                parts = root_file.split("/")
                if len(parts) >= 2 and parts[-2] in ("candidate", "transforms", "models", "features"):
                    root_file = f"{parts[-2]}/{parts[-1]}"
                else:
                    root_file = parts[-1]

        # Bounded formatted summary (max 4096 bytes)
        summary_lines = lines
        if len(summary_lines) > 40:
            summary_lines = summary_lines[:15] + [f"... [truncated {len(summary_lines) - 30} lines] ..."] + summary_lines[-15:]
        summary = "\n".join(summary_lines)
        if len(summary) > 4096:
            summary = summary[:4090] + "..."

        rc = RootCause(
            file=root_file,
            line_number=root_line,
            symbol=root_sym,
            exception_class=exc_class,
        )
        return rc, summary

    @classmethod
    def classify_failure(
        cls,
        exit_code: int | None,
        stderr: str,
        stdout: str = "",
        message: str = "",
        oom_killed: bool | None = None,
    ) -> str:
        """Classify failure mode based on exit status, output logs, and signals."""
        combined = f"{message}\n{stderr}\n{stdout}".lower()

        # 1. CGROUP OOM
        if oom_killed is True or exit_code in (137, -9):
            return FailureType.CGROUP_OOM
        if any(term in combined for term in [
            "cgroup_oom",
            "out of memory",
            "memoryerror",
            "oom-killer",
            "oomkilled",
            "killed process",
            "allocate bytes",
            "std::bad_alloc",
            "cuda out of memory",
            "died from signal 9",
        ]):
            return FailureType.CGROUP_OOM

        # 2. EXECUTION TIMEOUT
        if exit_code == 124 or any(term in combined for term in [
            "timeout",
            "timed out",
            "timeoutexpired",
            "exceeded deadline",
            "wall clock limit",
        ]):
            return FailureType.EXECUTION_TIMEOUT

        # 3. SYNTAX / PARSING ERROR
        if any(term in combined for term in [
            "syntaxerror",
            "indentationerror",
            "taberror",
            "invalid syntax",
        ]):
            return FailureType.SYNTAX_ERROR

        # 4. MISSING DEPENDENCY
        if any(term in combined for term in [
            "modulenotfounderror",
            "no module named",
            "importerror: cannot import name",
        ]):
            return FailureType.MISSING_DEPENDENCY

        # 5. NON-FINITE LOSS / PREDICTIONS
        if any(term in combined for term in [
            "floatingpointerror",
            "non-finite loss",
            "loss is nan",
            "loss is inf",
            "nan or non-finite",
            "non-finite values",
            "predictions contain nan",
            "probabilities contain nan",
            "overflowerror: math range error",
        ]):
            return FailureType.NON_FINITE_LOSS

        # 6. SHAPE / FEATURE MISMATCH
        if any(term in combined for term in [
            "shape mismatch",
            "shapes (",
            "dimension mismatch",
            "expected shape",
            "number of features of the model must match",
            "feature_names mismatch",
            "row count mismatch",
            "inconsistent numbers of samples",
            "must have same first dimension",
            "row misalignment",
        ]):
            return FailureType.SHAPE_MISMATCH

        # 7. CLASS ORDER MISMATCH
        if any(term in combined for term in [
            "class order",
            "class_order",
            "missing probability columns",
            "does not match manifest class order",
            "is not in declared class_order",
            "prediction value",
            "probabilities do not sum to 1.0",
        ]):
            return FailureType.CLASS_ORDER_MISMATCH

        # 8. GENERAL RUNTIME EXCEPTION
        return FailureType.RUNTIME_EXCEPTION

    @classmethod
    def generate_remediations(
        cls,
        failure_type: str,
        root_cause: RootCause,
        error_msg: str,
        resource_telemetry: ResourceTelemetry,
    ) -> list[str]:
        """Generate targeted, actionable remediation advice for autonomous agent self-healing."""
        suggestions: list[str] = []

        if failure_type == FailureType.CGROUP_OOM:
            suggestions.append("Reduce mini-batch size during training (e.g. batch_size=32 or 64) to lower peak memory occupancy.")
            suggestions.append("Downcast numeric feature matrices from float64 to float32 or int32 using `.astype(np.float32)`.")
            suggestions.append("Reduce tree depth or model complexity (e.g. max_depth <= 6, n_estimators <= 100).")
            suggestions.append("Use out-of-core batch streaming or memory-mapped arrays instead of inflating dense matrices.")
        elif failure_type == FailureType.NON_FINITE_LOSS:
            suggestions.append("Check feature matrices for unhandled NaNs, nulls, or infs; apply SimpleImputer or median imputation before model fitting.")
            suggestions.append("Add epsilon smoothing (e.g. 1e-7) to denominators and log arguments to prevent division by zero or log(0).")
            suggestions.append("Apply gradient clipping (e.g. clip_grad_norm_ <= 1.0) or reduce learning rate to prevent exploding activations.")
            suggestions.append("Verify target variable domain; ensure targets do not contain negative values if using Poisson/Gamma/Log losses.")
        elif failure_type == FailureType.SHAPE_MISMATCH:
            suggestions.append("Verify that feature transformations in `features.py` produce identical columns and shapes between `fit` and `predict`.")
            suggestions.append("If using one-hot encoding, configure `handle_unknown='ignore'` so unexpected test categories do not alter column counts.")
            suggestions.append("Ensure target column is strictly isolated and dropped from feature matrices before passing to `model.predict()`.")
            suggestions.append("Check matrix transpose and dimension alignment: ensure (n_samples, n_features) orientation.")
        elif failure_type == FailureType.CLASS_ORDER_MISMATCH:
            suggestions.append("Ensure prediction output DataFrame contains probability columns named `prob_<class>` for each class in manifest `class_order`.")
            suggestions.append("Ensure prediction probabilities are bounded in [0.0, 1.0] and sum to exactly 1.0 across classes for each sample.")
            suggestions.append("Ensure predicted discrete values in `pred` column strictly match classes declared in manifest `class_order`.")
        elif failure_type == FailureType.EXECUTION_TIMEOUT:
            suggestions.append("Vectorize feature engineering using NumPy or Polars array operations instead of Python row iteration loops.")
            suggestions.append("Reduce iteration limits (e.g. `n_estimators`, `max_iter`, or training epochs) to fit within execution deadline.")
            suggestions.append("Implement early stopping with a validation loss patience parameter to abort unpromising training early.")
            suggestions.append("Subsample training data for screening iterations before attempting full-dataset convergence.")
        elif failure_type == FailureType.MISSING_DEPENDENCY:
            suggestions.append("The sandbox runs in an offline isolated container; third-party packages outside the certified stack cannot be installed.")
            suggestions.append("Restrict imports to standard library and pre-installed packages: scikit-learn, lightgbm, xgboost, catboost, torch, numpy, pandas, scipy.")
        elif failure_type == FailureType.SYNTAX_ERROR:
            loc = f" at line {root_cause.line_number}" if root_cause.line_number else ""
            file_ref = f" in {root_cause.file}" if root_cause.file else ""
            suggestions.append(f"Fix Python syntax error{loc}{file_ref}; check for missing colons, unmatched parentheses, or indentation discrepancies.")
        else:
            # Runtime exception
            cls_name = root_cause.exception_class or ""
            if "ZeroDivision" in cls_name:
                suggestions.append("Add zero-guard checks (e.g. `denom if abs(denom) > 1e-6 else 1e-6`) before division operations.")
            elif "KeyError" in cls_name:
                suggestions.append("Verify column or dictionary keys exist before accessing; use `.get()` with safe defaults or check `.columns`.")
            elif "AttributeError" in cls_name:
                suggestions.append("Verify method and attribute names on the model/transformer instance; check API compatibility.")
            elif "TypeError" in cls_name:
                suggestions.append("Check data types and function argument counts; verify whether inputs should be NumPy arrays or pandas DataFrames.")
            else:
                suggestions.append("Review traceback lines and verify candidate code preconditions, input shapes, and variable states.")

        return suggestions

    @classmethod
    def diagnose_failure(
        cls,
        exit_code: int | None,
        stderr: str,
        stdout: str = "",
        message: str = "",
        candidate_id: str = "candidate",
        stage: str = "evaluate",
        peak_rss_bytes: int | None = None,
        memory_limit_bytes: int | None = None,
        wall_elapsed_ms: int | None = None,
        timeout_ms: int | None = None,
        oom_killed: bool | None = None,
        context_metadata: dict[str, Any] | None = None,
        candidate_dir: str | None = None,
    ) -> DiagnosticReceipt:
        """Compile a complete, validated DiagnosticReceipt from process execution outcomes."""
        # 1. Parse root-cause attribution from traceback
        combined_text = f"{stderr}\n{stdout}" if stdout else stderr
        root_cause, tb_summary = cls.parse_traceback(combined_text, candidate_dir=candidate_dir)

        # 2. Refine error message
        if not message:
            # Extract last non-empty line of stderr
            for line in reversed(stderr.strip().splitlines()):
                if line.strip():
                    message = line.strip()
                    break
        if not message and exit_code is not None and exit_code != 0:
            if exit_code == 137:
                message = "Process terminated with exit code 137 (SIGKILL / cgroup out-of-memory)"
            elif exit_code == 124:
                message = f"Process execution timed out after deadline ({wall_elapsed_ms or timeout_ms or 0}ms)"
            else:
                message = f"Process failed with exit code {exit_code}"

        # 3. Classify failure mode
        failure_type = cls.classify_failure(
            exit_code=exit_code,
            stderr=stderr,
            stdout=stdout,
            message=message,
            oom_killed=oom_killed,
        )

        # In OOM mode, update telemetry and root_cause if empty
        if failure_type == FailureType.CGROUP_OOM:
            oom_killed = True
            if not root_cause.exception_class:
                root_cause.exception_class = "MemoryError"

        # In timeout mode, update telemetry and root_cause if empty
        if failure_type == FailureType.EXECUTION_TIMEOUT:
            if not root_cause.exception_class:
                root_cause.exception_class = "TimeoutExpired"

        telemetry = ResourceTelemetry(
            exit_code=exit_code,
            peak_rss_bytes=peak_rss_bytes,
            memory_limit_bytes=memory_limit_bytes,
            wall_elapsed_ms=wall_elapsed_ms,
            timeout_ms=timeout_ms,
            oom_killed=oom_killed,
        )

        remediations = cls.generate_remediations(
            failure_type=failure_type,
            root_cause=root_cause,
            error_msg=message,
            resource_telemetry=telemetry,
        )

        import time
        created_ms = int(time.time() * 1000)

        receipt = DiagnosticReceipt(
            schema_version="v2",
            diagnostic_id=f"diag-{uuid.uuid4().hex[:12]}",
            candidate_id=candidate_id,
            stage=stage,
            failure_type=failure_type,
            error_message=message[:2048] if message else "Execution failure",
            root_cause=root_cause,
            traceback_summary=tb_summary,
            resource_telemetry=telemetry,
            remediation_suggestions=remediations,
            context_metadata=context_metadata or {},
            created_ms=created_ms,
        )

        receipt.validate()
        return receipt

    @classmethod
    def diagnose_exception(
        cls,
        exc: Exception,
        candidate_id: str = "candidate",
        stage: str = "evaluate",
        candidate_dir: str | None = None,
        context_metadata: dict[str, Any] | None = None,
    ) -> DiagnosticReceipt:
        """Compile a DiagnosticReceipt from a Python Exception instance."""
        tb_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        err_msg = str(exc) or exc.__class__.__name__

        return cls.diagnose_failure(
            exit_code=1,
            stderr=tb_text,
            stdout="",
            message=err_msg,
            candidate_id=candidate_id,
            stage=stage,
            candidate_dir=candidate_dir,
            context_metadata=context_metadata,
        )
