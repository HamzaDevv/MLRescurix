"""Clean-Room Custom Candidate Bundle Packager & Inference Verifier.

Packages custom neural networks and estimators into self-contained, verified delivery
bundles conforming to `mlrescurix.bundle.v2` and `ResultBundleContract`:
1. Self-Contained Clean-Room Portability:
   - Inference script (`inference.py`) runs in a clean subprocess with PYTHONPATH unset.
   - Zero controller or repository dependencies at inference time.
2. Robust Input Handling & Error Diagnostics:
   - Validates feature columns against input schema.
   - Catches missing required features and schema violations before model invocation.
   - Supports numeric dtype tolerances (int/float coercion, NaN handling).
3. Telemetry & Resource Reporting:
   - Measures inference execution latency (milliseconds) and process RSS memory delta.
4. Cryptographic Integrity:
   - Generates per-file SHA-256 manifest and model weights hashes.
   - Validates manifest against `contracts/v2/candidate_bundle.schema.json`.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from importlib.metadata import version as get_pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .contracts import ResultBundleContract, file_sha256
from ..scientific.evidence import streaming_file_sha256

logger = logging.getLogger("mlrescurix.delivery")


class BundleVerificationError(RuntimeError):
    """Raised when bundle packaging or clean-room verification fails."""


class CustomBundlePackager:
    """Packages custom models into portable, self-contained mlrescurix.bundle.v2 bundles."""

    def __init__(
        self,
        schema_path: Optional[Union[str, Path]] = None,
    ):
        if schema_path:
            self.schema_path = Path(schema_path).resolve()
        else:
            # Default schema path relative to repository
            parents = Path(__file__).resolve().parents
            repo_root = parents[4] if len(parents) > 4 else None
            self.schema_path = (repo_root / "contracts/v2/candidate_bundle.schema.json") if repo_root else None

    def package(
        self,
        candidate: Any,
        output_bundle_dir: Union[str, Path],
        target_column: str,
        task_type: str,
        primary_metric: str,
        primary_metric_score: float,
        metric_direction: str,
        feature_names: List[str],
        raw_example_records: List[Dict[str, Any]],
        candidate_id: Optional[str] = None,
        accepted_evaluation: Optional[Dict[str, Any]] = None,
        decision_threshold: float = 0.5,
        source_weights_path: Optional[Union[str, Path]] = None,
        project_id: Optional[str] = None,
    ) -> ResultBundleContract:
        """Packages fitted custom candidate into a complete, verified delivery bundle."""
        bundle_dir = Path(output_bundle_dir).resolve()
        if bundle_dir.exists():
            raise BundleVerificationError(f"Bundle destination already exists: {bundle_dir}")
        bundle_dir.mkdir(parents=True, exist_ok=True)

        cand_id = candidate_id or getattr(candidate, "plugin_id", "custom_neural_network")
        cand_identity = getattr(candidate, "candidate_identity", f"cand_{cand_id}_{int(time.time())}")

        if accepted_evaluation is not None:
            eval_status = accepted_evaluation.get("status")
            if eval_status not in ("evaluated", "evaluated_development", "verified_development_only"):
                raise BundleVerificationError(
                    f"Candidate {cand_id} has invalid/rejected evaluation status {eval_status!r}; export refused"
                )

        is_classification = "classification" in task_type.lower()
        classes = getattr(candidate, "classes_", [0, 1] if is_classification else None)
        if classes is not None:
            classes = [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in classes]

        # 1. Serialize model weights
        import joblib
        frozen_weights_file = "frozen_model.joblib"
        frozen_weights_path = bundle_dir / frozen_weights_file

        weights_file = "weights.pt"
        weights_path = bundle_dir / weights_file
        model_format = "pytorch_state_dict"

        if source_weights_path is not None:
            source_p = Path(source_weights_path).resolve()
            if not source_p.is_file():
                raise FileNotFoundError(f"Source weights file not found: {source_p}")
            if str(source_p).endswith(".joblib"):
                weights_file = "model.joblib"
                weights_path = bundle_dir / weights_file
                model_format = "joblib"
                shutil.copy2(source_p, weights_path)
                shutil.copy2(source_p, frozen_weights_path)
            else:
                shutil.copy2(source_p, weights_path)
                shutil.copy2(source_p, bundle_dir / "model.pt")
                try:
                    import torch
                    state_dict = torch.load(source_p, map_location="cpu", weights_only=True)
                    joblib.dump(state_dict, frozen_weights_path)
                except Exception:
                    shutil.copy2(source_p, frozen_weights_path)
        else:
            try:
                import torch
                if hasattr(candidate, "model") and isinstance(candidate.model, torch.nn.Module):
                    torch.save(candidate.model.state_dict(), weights_path)
                    shutil.copy2(weights_path, bundle_dir / "model.pt")
                    joblib.dump(candidate.model.state_dict(), frozen_weights_path)
                elif isinstance(candidate, torch.nn.Module):
                    torch.save(candidate.state_dict(), weights_path)
                    shutil.copy2(weights_path, bundle_dir / "model.pt")
                    joblib.dump(candidate.state_dict(), frozen_weights_path)
                else:
                    weights_file = "model.joblib"
                    weights_path = bundle_dir / weights_file
                    model_format = "joblib"
                    joblib.dump(candidate, weights_path)
                    joblib.dump(candidate, frozen_weights_path)
            except Exception:
                weights_file = "model.joblib"
                weights_path = bundle_dir / weights_file
                model_format = "joblib"
                joblib.dump(candidate, weights_path)
                joblib.dump(candidate, frozen_weights_path)

        weights_sha = streaming_file_sha256(weights_path)
        frozen_weights_sha = streaming_file_sha256(frozen_weights_path)

        # 2. Extract model architecture details
        in_features = len(feature_names)
        hidden_dim = getattr(candidate, "hidden_dim", 32)
        out_features = len(classes) if classes else 1

        # 3. Write input schema
        input_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "InferenceInputSchema",
            "type": "object",
            "required": feature_names,
            "properties": {
                name: {"type": ["number", "string", "null"]} for name in feature_names
            },
            "additionalProperties": True,
        }
        (bundle_dir / "input_schema.json").write_text(json.dumps(input_schema, indent=2), encoding="utf-8")

        # 4. Write output schema
        output_schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "InferenceOutputSchema",
            "type": "object",
            "required": ["prediction", "latency_ms", "peak_memory_bytes"],
            "properties": {
                "prediction": {"type": "array"},
                "probabilities": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                "classes": {"type": "array"},
                "latency_ms": {"type": "number"},
                "peak_memory_bytes": {"type": "integer"},
            },
        }
        (bundle_dir / "output_schema.json").write_text(json.dumps(output_schema, indent=2), encoding="utf-8")

        # 5. Write examples input and run candidate to generate expected outputs
        examples_records = copy.deepcopy(raw_example_records[:10])
        df_examples = pd.DataFrame(examples_records)
        for col in feature_names:
            if col not in df_examples.columns:
                df_examples[col] = 0.0
        df_examples = df_examples[feature_names]

        # Generate expected predictions
        preds = candidate.predict(df_examples)
        if isinstance(preds, np.ndarray):
            preds_list = preds.tolist()
        else:
            preds_list = list(preds)

        probs_list = None
        if is_classification and hasattr(candidate, "predict_proba"):
            probs = candidate.predict_proba(df_examples)
            if isinstance(probs, np.ndarray):
                probs_list = probs.tolist()

        (bundle_dir / "examples_input.json").write_text(
            json.dumps(examples_records, indent=2), encoding="utf-8"
        )

        expected_payload: Dict[str, Any] = {
            "prediction": preds_list,
        }
        if probs_list is not None:
            expected_payload["probabilities"] = probs_list
        if classes is not None:
            expected_payload["classes"] = classes

        (bundle_dir / "examples_expected.json").write_text(
            json.dumps(expected_payload, indent=2), encoding="utf-8"
        )

        # 6. Write standalone clean-room inference.py
        inference_code = f"""#!/usr/bin/env python3
\"\"\"Self-Contained Inference Runner for MLRescurix Delivery Bundle.

Zero repository dependencies. Standalone portable prediction CLI.
\"\"\"
import argparse
import json
import os
import sys
import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

FEATURE_NAMES = {repr(feature_names)}
IS_CLASSIFICATION = {repr(is_classification)}
CLASSES = {repr(classes)}
MODEL_FORMAT = {repr(model_format)}
WEIGHTS_FILE = {repr(weights_file)}
IN_FEATURES = {in_features}
HIDDEN_DIM = {hidden_dim}
OUT_FEATURES = {out_features}
THRESHOLD = {decision_threshold}

def load_model(bundle_dir):
    weights_path = os.path.join(bundle_dir, WEIGHTS_FILE)
    if MODEL_FORMAT == "pytorch_state_dict":
        import torch
        import torch.nn as nn

        class TinyMLP(nn.Module):
            def __init__(self, in_f, h_dim, out_f):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_f, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.ReLU(),
                    nn.Dropout(0.0),
                    nn.Linear(h_dim, h_dim // 2 if h_dim >= 4 else h_dim),
                    nn.LayerNorm(h_dim // 2 if h_dim >= 4 else h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim // 2 if h_dim >= 4 else h_dim, out_f),
                )
            def forward(self, x):
                return self.net(x)

        model = TinyMLP(IN_FEATURES, HIDDEN_DIM, OUT_FEATURES)
        if weights_path.endswith(".joblib"):
            import joblib
            state_dict = joblib.load(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    else:
        import joblib
        return joblib.load(weights_path)

def validate_and_prepare_matrix(records):
    if not isinstance(records, list):
        raise ValueError("Input data must be a JSON array of record dictionaries")

    matrix = []
    for i, r in enumerate(records):
        row = []
        for feat in FEATURE_NAMES:
            if feat not in r:
                raise KeyError(f"Record index {{i}} missing required feature '{{feat}}'")
            val = r[feat]
            if val is None:
                val = 0.0
            try:
                row.append(float(val))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature '{{feat}}' value '{{val}}' cannot be coerced to float: {{exc}}")
        matrix.append(row)
    return matrix

def main():
    parser = argparse.ArgumentParser(description="Standalone clean inference")
    parser.add_argument("--input", required=True, help="Input records JSON file")
    parser.add_argument("--output", required=True, help="Output predictions JSON file")
    args = parser.parse_args()

    bundle_dir = os.path.dirname(os.path.abspath(__file__))
    with open(args.input, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Validate schema and prepare numeric matrix
    matrix = validate_and_prepare_matrix(records)

    start_rss = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
    t0 = time.time()

    model = load_model(bundle_dir)

    if MODEL_FORMAT == "pytorch_state_dict":
        import torch
        with torch.no_grad():
            x_tensor = torch.tensor(matrix, dtype=torch.float32)
            logits = model(x_tensor)
            if IS_CLASSIFICATION:
                probs = torch.softmax(logits, dim=-1).cpu().numpy().tolist()
                cls_indices = [int(torch.argmax(logits, dim=-1)[i].item()) for i in range(len(matrix))]
                if CLASSES is not None and len(CLASSES) > 0:
                    preds = [CLASSES[idx] for idx in cls_indices]
                else:
                    preds = cls_indices
            else:
                probs = None
                preds = [float(v.item()) for v in logits.view(-1)]
    else:
        import numpy as np
        arr = np.array(matrix, dtype=float)
        preds_arr = model.predict(arr)
        preds = preds_arr.tolist() if isinstance(preds_arr, np.ndarray) else list(preds_arr)
        probs = None
        if IS_CLASSIFICATION and hasattr(model, "predict_proba"):
            probs_arr = model.predict_proba(arr)
            probs = probs_arr.tolist() if isinstance(probs_arr, np.ndarray) else list(probs_arr)

    latency_ms = (time.time() - t0) * 1000.0
    end_rss = psutil.Process().memory_info().rss if HAS_PSUTIL else 0
    peak_rss_delta = max(0, end_rss - start_rss)

    output = {{
        "prediction": preds,
        "latency_ms": round(latency_ms, 3),
        "peak_memory_bytes": peak_rss_delta,
    }}
    if probs is not None:
        output["probabilities"] = probs
    if CLASSES is not None:
        output["classes"] = CLASSES

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
"""
        (bundle_dir / "inference.py").write_text(inference_code, encoding="utf-8")
        try:
            os.chmod(bundle_dir / "inference.py", 0o755)
        except OSError:
            pass

        # 7. Write inference.lock and training.lock with pinned dependencies
        pinned_deps = []
        for pkg in ["numpy", "scipy", "pandas", "torch", "scikit-learn", "joblib"]:
            try:
                v = get_pkg_version(pkg)
                pinned_deps.append(f"{pkg}=={v}")
            except Exception:
                pass
        lock_content = "\n".join(pinned_deps) + "\n"
        (bundle_dir / "inference.lock").write_text(lock_content, encoding="utf-8")
        (bundle_dir / "training.lock").write_text(lock_content, encoding="utf-8")

        # 8. Write serve.py stub
        serve_code = """#!/usr/bin/env python3
\"\"\"Lightweight HTTP/CLI server stub for bundle invocation.\"\"\"
import sys
from inference import main

if __name__ == "__main__":
    main()
"""
        (bundle_dir / "serve.py").write_text(serve_code, encoding="utf-8")

        # 9. Ensure 64-char hex identity
        if len(cand_identity) == 64 and all(c in "0123456789abcdef" for c in cand_identity.lower()):
            clean_identity = cand_identity.lower()
        else:
            clean_identity = hashlib.sha256(cand_identity.encode("utf-8")).hexdigest()

        # 10. Write scorecard.json
        has_eval = accepted_evaluation is not None
        source_sc = None
        if has_eval and "scorecard_path" in accepted_evaluation and Path(accepted_evaluation["scorecard_path"]).is_file():
            source_sc = Path(accepted_evaluation["scorecard_path"])
        elif has_eval and (bundle_dir.parent / "trials" / cand_id / "scorecard.json").is_file():
            source_sc = bundle_dir.parent / "trials" / cand_id / "scorecard.json"

        if source_sc is not None:
            shutil.copy2(source_sc, bundle_dir / "scorecard.json")
        else:
            scorecard = {
                "bundle_version": "2.0.0",
                "candidate_id": cand_id,
                "candidate_identity": clean_identity,
                "evaluation_scope": "unverified_caller_supplied_metric",
                "primary_metric": primary_metric,
                "primary_metric_score": float(primary_metric_score),
                "primary_metric_direction": metric_direction,
                "task_type": task_type,
                "target_column": target_column,
                "constraints": {},
                "audit_status": "UNVERIFIED_DEVELOPMENT",
            }
            (bundle_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        scorecard_sha = streaming_file_sha256(bundle_dir / "scorecard.json")

        # 11. Write lineage.json
        lineage = {
            "candidate_id": cand_id,
            "candidate_identity": clean_identity,
            "created_at": time.time(),
            "provenance": "MLRescurix clean-room candidate packager",
        }
        (bundle_dir / "lineage.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
        lineage_sha = streaming_file_sha256(bundle_dir / "lineage.json")

        # 12. Write verification.json and final.json for contract conformity
        verification = {
            "weights_sha256": weights_sha,
            "scorecard_sha256": scorecard_sha,
            "frozen_model_sha256": frozen_weights_sha,
            "bundle_candidate": cand_id,
            "candidate_id": cand_id,
            "dataset_sha256": accepted_evaluation.get("dataset_sha256", "") if has_eval else "",
            "raw_input_api": "predict(records)",
            "all_passed": False,
        }
        (bundle_dir / "verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")

        final_data = {
            "status": "unverified_development_only",
            "candidate_id": cand_id,
            "test_metrics": {},
            "constraints": {},
            "bundle": "bundle",
        }
        (bundle_dir / "final.json").write_text(json.dumps(final_data, indent=2), encoding="utf-8")

        # 13. Write README.md
        readme = f"""# MLRescurix Delivery Bundle: {cand_id}

- **Candidate Identity:** `{clean_identity}`
- **Task Type:** {task_type}
- **Target Column:** {target_column}
- **Primary Metric:** {primary_metric} = {primary_metric_score:.4f} ({metric_direction})

## Clean-Room Inference Instructions

Run standalone prediction via CLI:
```bash
python inference.py --input examples_input.json --output predictions.json
```
Zero repository dependencies required. Pinned dependencies: `torch`, `numpy`, `pandas`.
"""
        (bundle_dir / "README.md").write_text(readme, encoding="utf-8")

        # 13. Generate manifest.json conforming strictly to contracts/v2/candidate_bundle.schema.json
        files_manifest: Dict[str, Dict[str, Any]] = {}
        for fname in sorted(os.listdir(bundle_dir)):
            if fname == "manifest.json":
                continue
            fpath = bundle_dir / fname
            if fpath.is_file():
                files_manifest[fname] = {
                    "sha256": streaming_file_sha256(fpath),
                    "bytes": fpath.stat().st_size,
                }

        manifest = {
            "format": "mlrescurix.bundle.v2",
            "project_id": project_id or "mlrescurix_project",
            "candidate_id": cand_id,
            "candidate_identity": clean_identity,
            "files": files_manifest,
            "model": {
                "weights_path": weights_file,
                "weights_sha256": weights_sha,
                "format": "native" if "pytorch" in model_format else "joblib",
                "decision_threshold": float(decision_threshold),
                "calibration_method": "none",
            },
            "pipeline": {
                "feature_pipeline": "inference.py",
                "entrypoint": "inference.py",
            },
            "contracts": {
                "cli": "inference.py",
                "api": "serve.py",
                "input_schema": "input_schema.json",
                "output_schema": "output_schema.json",
            },
            "dependencies": {
                "training_lock": "training.lock",
                "inference_lock": "inference.lock",
                "isolated_inference": True,
            },
            "scorecard": {
                "path": "scorecard.json",
                "sha256": scorecard_sha,
            },
            "lineage": {
                "path": "lineage.json",
                "sha256": lineage_sha,
            },
        }

        # Validate against bundle schema if available
        if self.schema_path and self.schema_path.exists():
            try:
                from jsonschema import Draft202012Validator
                schema_data = json.loads(self.schema_path.read_text(encoding="utf-8"))
                Draft202012Validator(schema_data).validate(manifest)
            except Exception as exc:
                logger.error("Bundle manifest schema validation failure: %s", exc)
                raise BundleVerificationError(f"Bundle manifest violates schema: {exc}") from exc

        (bundle_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # 14. Run Verification in Clean Subprocess
        self.verify_bundle(bundle_dir)

        # 12. Return ResultBundleContract
        return ResultBundleContract(
            candidate_id=cand_id,
            primary_metric_name=primary_metric,
            primary_metric_score=float(primary_metric_score),
            primary_metric_direction=metric_direction,
            aligned_predictions=preds_list,
            row_ids=list(range(len(preds_list))),
            probabilities=probs_list,
            classes=classes,
            measured_resources={"weights_size_bytes": weights_path.stat().st_size},
            model_artifacts={weights_file: weights_sha},
            transforms=[],
            dependency_lock={pkg.split("==")[0]: pkg.split("==")[1] for pkg in pinned_deps if "==" in pkg},
            inference_entrypoint="inference.py",
            failure_limitations=[],
        )

    def verify_bundle(self, bundle_dir: Union[str, Path]) -> Dict[str, Any]:
        """Runs inference.py in a clean subprocess with PYTHONPATH unset to prove clean-room execution."""
        b_dir = Path(bundle_dir).resolve()
        input_path = b_dir / "examples_input.json"
        output_path = b_dir / "subprocess_test_output.json"
        expected_path = b_dir / "examples_expected.json"

        if not input_path.exists() or not expected_path.exists():
            raise BundleVerificationError(f"Bundle {b_dir} missing examples_input.json or examples_expected.json")

        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}

        proc = subprocess.run(
            [sys.executable, str(b_dir / "inference.py"), "--input", str(input_path), "--output", str(output_path)],
            cwd=b_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if proc.returncode != 0:
            raise BundleVerificationError(
                f"Clean-room inference subprocess failed with exit code {proc.returncode}:\n"
                f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
            )

        if not output_path.exists():
            raise BundleVerificationError("Inference subprocess completed but output file was not written")

        output = json.loads(output_path.read_text(encoding="utf-8"))
        expected = json.loads(expected_path.read_text(encoding="utf-8"))

        # Verify predictions match within tolerance
        pred_actual = np.asarray(output["prediction"])
        pred_expected = np.asarray(expected["prediction"])
        if np.issubdtype(pred_actual.dtype, np.number) and np.issubdtype(pred_expected.dtype, np.number):
            if not np.allclose(pred_actual, pred_expected, atol=1e-5):
                raise BundleVerificationError(
                    f"Subprocess predictions {pred_actual} do not match expected {pred_expected}"
                )
        else:
            if not np.array_equal(pred_actual, pred_expected):
                raise BundleVerificationError(
                    f"Subprocess predictions {pred_actual} do not match expected {pred_expected}"
                )

        if "probabilities" in expected and "probabilities" in output:
            prob_actual = np.asarray(output["probabilities"])
            prob_expected = np.asarray(expected["probabilities"])
            if not np.allclose(prob_actual, prob_expected, atol=1e-5):
                raise BundleVerificationError("Subprocess probabilities do not match expected probabilities")

        # Cleanup test output file
        if output_path.exists():
            output_path.unlink()

        return {
            "status": "verified",
            "clean_room_verified": True,
            "latency_ms": output.get("latency_ms", 0.0),
            "peak_memory_bytes": output.get("peak_memory_bytes", 0),
        }
