"""Typed Hypothesis Action Contract & Dispatcher (MLRescurix v2 S1).

Implements strict validation, idempotent action execution, ten typed DSL mutations,
paired fold-local ablation, and governed sandbox coding with zero validation-label exposure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from .fe_ablation import ablate_feature_hypothesis
from .primitives import (
    InformativeMissingnessAndInteractions,
    LightweightTextEmbedder,
    NestedOOFEmpiricalBayesEncoder,
    QuasiIdDecomposer,
)
from .workflow import WorkflowError

SCHEMA_PATH = Path(__file__).resolve().parents[4] / "contracts" / "v2" / "hypothesis_action.schema.json"
if not SCHEMA_PATH.exists():
    for candidate_root in [Path(__file__).resolve().parents[3], Path.cwd(), Path.cwd().parent]:
        candidate_schema = candidate_root / "contracts" / "v2" / "hypothesis_action.schema.json"
        if candidate_schema.exists():
            SCHEMA_PATH = candidate_schema
            break


class ActionExecutionError(WorkflowError):
    """Raised when hypothesis action validation or dispatch fails."""


class IdempotencyConflictError(ActionExecutionError):
    """Raised when an existing action_id is submitted with conflicting parameters."""


@dataclass
class HypothesisActionReceipt:
    action_id: str
    status: str  # "confirmed", "refuted", "inconclusive", "failed"
    action: str
    parent_candidate_id: str
    split_id: str
    paired_metric_delta: Optional[float] = None
    relative_gain: Optional[float] = None
    versioned_transform_lineage: List[Dict[str, Any]] = field(default_factory=list)
    fitted_fold_references: List[str] = field(default_factory=list)
    resource_measurements: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    payload_hash: str = ""
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DomainSafeSkewTransformer(BaseEstimator, TransformerMixin):
    """Domain-safe power / log transformer with strict finiteness guarantees."""

    def __init__(self, columns: List[str], method: str = "yeo_johnson", clip_min: Optional[float] = None):
        self.columns = columns
        self.method = method
        self.clip_min = clip_min
        self.pt_map: Dict[str, PowerTransformer] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DomainSafeSkewTransformer":
        for col in self.columns:
            if col in X.columns and pd.api.types.is_numeric_dtype(X[col]):
                vals = X[col].dropna().to_numpy().reshape(-1, 1)
                if len(vals) > 0:
                    if self.method == "box_cox":
                        # Box-Cox requires strictly positive values
                        if np.any(vals <= 0):
                            shift = abs(float(np.min(vals))) + 1e-4
                            vals = vals + shift
                        pt = PowerTransformer(method="box-cox", standardize=False)
                    else:
                        pt = PowerTransformer(method="yeo-johnson", standardize=False)
                    try:
                        pt.fit(vals)
                        self.pt_map[col] = pt
                    except Exception:
                        pass
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, pt in self.pt_map.items():
            if col in X_out.columns:
                vals = X_out[col].fillna(0).to_numpy(dtype=float).reshape(-1, 1)
                if self.clip_min is not None:
                    vals = np.maximum(vals, self.clip_min)
                if self.method == "box_cox":
                    shift = 1e-4 if np.any(vals <= 0) else 0.0
                    vals = vals + shift
                try:
                    transformed = pt.transform(vals).flatten()
                    X_out[f"{col}_skew_{self.method}"] = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
                except Exception:
                    pass
        return X_out


class FeatureDeletionTransformer(BaseEstimator, TransformerMixin):
    """Reversible feature drop transformer."""

    def __init__(self, columns_to_delete: List[str]):
        self.columns_to_delete = columns_to_delete

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeatureDeletionTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=[c for c in self.columns_to_delete if c in X.columns], errors="ignore")


class BoundedInteractionTransformer(BaseEstimator, TransformerMixin):
    """Safe pairwise feature interaction generator with division guards."""

    def __init__(
        self,
        feature_pairs: List[List[str]],
        operations: Optional[List[str]] = None,
        max_interactions: int = 20,
    ):
        self.feature_pairs = feature_pairs
        self.operations = operations or ["multiply"]
        self.max_interactions = max_interactions

    def fit(self, X: pd.DataFrame, y: Any = None) -> "BoundedInteractionTransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        generated = 0
        for pair in self.feature_pairs:
            if generated >= self.max_interactions:
                break
            if len(pair) != 2:
                continue
            c1, c2 = pair
            if c1 in X.columns and c2 in X.columns and pd.api.types.is_numeric_dtype(X[c1]) and pd.api.types.is_numeric_dtype(X[c2]):
                s1 = X[c1].fillna(0).to_numpy(dtype=float)
                s2 = X[c2].fillna(0).to_numpy(dtype=float)
                for op in self.operations:
                    if op == "multiply":
                        res = s1 * s2
                        name = f"inter_{c1}_mul_{c2}"
                    elif op == "divide":
                        denom = np.where(np.abs(s2) < 1e-6, 1e-6, s2)
                        res = s1 / denom
                        name = f"inter_{c1}_div_{c2}"
                    elif op == "add":
                        res = s1 + s2
                        name = f"inter_{c1}_add_{c2}"
                    elif op == "subtract":
                        res = s1 - s2
                        name = f"inter_{c1}_sub_{c2}"
                    else:
                        continue
                    X_out[name] = np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
                    generated += 1
                    if generated >= self.max_interactions:
                        break
        return X_out


class HypothesisActionDispatcher:
    """Authoritative dispatcher for typed hypothesis actions.

    Maintains idempotency cache, strictly checks payload schemas, validates preconditions,
    and executes paired fold-local ablations.
    """

    def __init__(self, ledger_path: Optional[Union[str, Path]] = None):
        self.idempotency_cache: Dict[str, Tuple[str, HypothesisActionReceipt]] = {}
        self.ledger_path = Path(ledger_path) if ledger_path else None
        if self.ledger_path and self.ledger_path.exists():
            self._load_ledger()

    def _load_ledger(self) -> None:
        """Load persisted hypothesis actions from durable ledger."""
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("actions", []):
                aid = entry["action_id"]
                phash = entry.get("payload_hash", "")
                r_dict = dict(entry)
                r_dict.pop("payload_hash", None)
                receipt = HypothesisActionReceipt(**r_dict, payload_hash=phash)
                self.idempotency_cache[aid] = (phash, receipt)
        except Exception:
            pass

    def _save_ledger(self) -> None:
        """Atomically persist action cache to durable ledger file."""
        if not self.ledger_path:
            return
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            actions = []
            for aid, (phash, receipt) in self.idempotency_cache.items():
                r_dict = receipt.to_dict()
                r_dict["payload_hash"] = phash
                actions.append(r_dict)
            tmp_path = self.ledger_path.with_suffix(f".tmp.{os.getpid()}")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({"schema_version": "v2", "actions": actions}, f, indent=2)
            tmp_path.replace(self.ledger_path)
        except Exception:
            pass

    @staticmethod
    def _compute_payload_hash(payload: Dict[str, Any]) -> str:
        s = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def validate_action_payload(self, payload: Dict[str, Any]) -> None:
        """Validate payload against schema and required properties."""
        if not isinstance(payload, dict):
            raise ActionExecutionError("Action payload must be a JSON dictionary")

        if SCHEMA_PATH.exists():
            import jsonschema
            try:
                with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                jsonschema.validate(instance=payload, schema=schema)
            except jsonschema.ValidationError as ve:
                if "is a required property" in ve.message:
                    raise ActionExecutionError(f"Action payload missing required field: {ve.message}") from ve
                if "is not one of" in ve.message and any("action" in str(p) for p in ve.path):
                    raise ActionExecutionError(f"Unknown action type: {ve.message}") from ve
                raise ActionExecutionError(f"Action payload schema validation failed: {ve.message}") from ve
            except Exception as e:
                raise ActionExecutionError(f"Action payload validation error: {e}") from e

        required = [
            "schema_version",
            "action_id",
            "parent_candidate_id",
            "source_hash",
            "split_id",
            "topology_hash",
            "seed",
            "action",
            "parameters",
            "budget_id",
            "rationale_ref",
        ]
        for field_name in required:
            if field_name not in payload:
                raise ActionExecutionError(f"Action payload missing required field '{field_name}'")

        if payload["schema_version"] != "v2":
            raise ActionExecutionError(f"Unsupported schema_version '{payload['schema_version']}', expected 'v2'")

        valid_actions = {
            "APPLY_SKEW_TRANSFORM",
            "ENCODE_HIGH_CARD_CAT",
            "DECOMPOSE_QUASI_ID",
            "EMBED_SHORT_TEXT",
            "ADD_MNAR_INDICATOR",
            "ADD_INTERACTION",
            "DELETE_FEATURE",
            "REORDER_MODEL_PRIORITY",
            "UPDATE_SEARCH_SPACE",
            "TRANSFORM_TARGET",
            "CUSTOM_SANDBOX_CODE",
        }
        if payload["action"] not in valid_actions:
            raise ActionExecutionError(f"Unknown action type '{payload['action']}'")

    def execute_action(
        self,
        payload: Dict[str, Any],
        df: pd.DataFrame,
        topology: Dict[str, Any],
        base_features: Any,
        metric_name: str = "balanced_accuracy",
        direction: str = "maximize",
        fidelity: str = "fast",
    ) -> HypothesisActionReceipt:
        """Execute a typed hypothesis action with idempotency and paired ablation."""
        self.validate_action_payload(payload)

        action_id = payload["action_id"]
        action_name = payload["action"]
        params = payload["parameters"]
        payload_hash = self._compute_payload_hash(payload)

        # Idempotency check: replay identically, reject mutation with same action_id
        if action_id in self.idempotency_cache:
            cached_hash, cached_receipt = self.idempotency_cache[action_id]
            if cached_hash != payload_hash:
                raise IdempotencyConflictError(
                    f"Action ID '{action_id}' already executed with different parameters. "
                    f"Cached hash: '{cached_hash}', new hash: '{payload_hash}'"
                )
            return cached_receipt

        t0 = time.time()
        start_rss = 0
        try:
            import resource
            start_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            pass

        transformer_or_callable: Any = None
        lineage_record: Dict[str, Any] = {
            "action": action_name,
            "parameters": params,
            "applied_at": int(time.time() * 1000),
        }

        # Dispatch action to appropriate primitive transformer
        try:
            if action_name == "APPLY_SKEW_TRANSFORM":
                cols = params.get("columns", [])
                method = params.get("method", "yeo_johnson")
                clip_min = params.get("clip_min")
                transformer_or_callable = DomainSafeSkewTransformer(columns=cols, method=method, clip_min=clip_min)

            elif action_name == "ENCODE_HIGH_CARD_CAT":
                cols = params.get("columns", [])
                smoothing = float(params.get("smoothing", 10.0))
                cv_folds = int(params.get("cv_folds", 5))
                transformer_or_callable = NestedOOFEmpiricalBayesEncoder(
                    categorical_columns=cols,
                    min_smoothing=smoothing,
                    n_splits=cv_folds,
                    random_state=payload["seed"],
                )

            elif action_name == "DECOMPOSE_QUASI_ID":
                id_cols = params.get("id_columns", [])
                num_cols = params.get("numeric_columns", [])
                transformer_or_callable = QuasiIdDecomposer(
                    id_columns=id_cols,
                    numeric_columns=num_cols,
                )

            elif action_name == "EMBED_SHORT_TEXT":
                text_cols = params.get("text_columns", [])
                n_svd = int(params.get("n_svd_components", 8))
                top_k = int(params.get("top_k_subtokens", 16))
                transformer_or_callable = LightweightTextEmbedder(
                    text_columns=text_cols,
                    n_svd_components=n_svd,
                    top_k_subtokens=top_k,
                )

            elif action_name == "ADD_MNAR_INDICATOR":
                cols = params.get("columns", [])
                transformer_or_callable = InformativeMissingnessAndInteractions(
                    missing_columns=cols,
                    max_interactions=0,
                )

            elif action_name == "ADD_INTERACTION":
                pairs = params.get("feature_pairs", [])
                ops = params.get("operations", ["multiply"])
                max_int = int(params.get("max_interactions", 10))
                transformer_or_callable = BoundedInteractionTransformer(
                    feature_pairs=pairs,
                    operations=ops,
                    max_interactions=max_int,
                )

            elif action_name == "DELETE_FEATURE":
                cols_to_del = params.get("columns", [])
                transformer_or_callable = FeatureDeletionTransformer(columns_to_delete=cols_to_del)

            elif action_name == "REORDER_MODEL_PRIORITY":
                preferred = params.get("preferred_families", [])
                receipt = HypothesisActionReceipt(
                    action_id=action_id,
                    status="inconclusive",
                    action=action_name,
                    parent_candidate_id=payload["parent_candidate_id"],
                    split_id=payload["split_id"],
                    paired_metric_delta=0.0,
                    relative_gain=0.0,
                    versioned_transform_lineage=[lineage_record],
                    fitted_fold_references=["control_plane_schedule"],
                    message=f"Model priority reordered: {preferred}; returned inconclusive pending scheduler search sprint",
                    payload_hash=payload_hash,
                )
                self.idempotency_cache[action_id] = (payload_hash, receipt)
                self._save_ledger()
                return receipt

            elif action_name == "UPDATE_SEARCH_SPACE":
                target_fam = params.get("target_family", "general")
                bounds = params.get("parameter_bounds", {})
                receipt = HypothesisActionReceipt(
                    action_id=action_id,
                    status="inconclusive",
                    action=action_name,
                    parent_candidate_id=payload["parent_candidate_id"],
                    split_id=payload["split_id"],
                    paired_metric_delta=0.0,
                    relative_gain=0.0,
                    versioned_transform_lineage=[lineage_record],
                    fitted_fold_references=["search_space_updated"],
                    message=f"Search space updated for family '{target_fam}': {bounds}; returned inconclusive pending search trial execution",
                    payload_hash=payload_hash,
                )
                self.idempotency_cache[action_id] = (payload_hash, receipt)
                self._save_ledger()
                return receipt

            elif action_name == "TRANSFORM_TARGET":
                method = params.get("method", "log")
                receipt = HypothesisActionReceipt(
                    action_id=action_id,
                    status="inconclusive",
                    action=action_name,
                    parent_candidate_id=payload["parent_candidate_id"],
                    split_id=payload["split_id"],
                    paired_metric_delta=0.0,
                    relative_gain=0.0,
                    versioned_transform_lineage=[lineage_record],
                    fitted_fold_references=["target_transform_configured"],
                    message=f"Target transformation '{method}' requires candidate pipeline fitting and canonical inverse-metric evaluation; returned inconclusive until integrated into candidate training.",
                    payload_hash=payload_hash,
                )
                self.idempotency_cache[action_id] = (payload_hash, receipt)
                self._save_ledger()
                return receipt

            elif action_name == "CUSTOM_SANDBOX_CODE":
                code_snippet = params.get("code_snippet", "")
                transformer_or_callable = code_snippet

        except Exception as e:
            receipt = HypothesisActionReceipt(
                action_id=action_id,
                status="failed",
                action=action_name,
                parent_candidate_id=payload["parent_candidate_id"],
                split_id=payload["split_id"],
                message=f"Action initialization error: {e}",
                payload_hash=payload_hash,
            )
            self.idempotency_cache[action_id] = (payload_hash, receipt)
            self._save_ledger()
            return receipt

        # Run paired fold-local ablation
        ablation_receipt = ablate_feature_hypothesis(
            hypothesis_id=action_id,
            candidate_features_py=transformer_or_callable,
            base_features_py=base_features,
            topology=topology,
            df=df,
            fidelity=fidelity,
            metric_name=metric_name,
            random_state=payload["seed"],
            ledger_path=self.ledger_path,
        )

        cv_delta = ablation_receipt.get("cv_score_delta", 0.0)
        base_score = ablation_receipt.get("baseline_cv_score", ablation_receipt.get("base_cv_score", 0.5))

        # cv_score_delta in fe_ablation is utility-oriented (higher is always better)
        benefit = cv_delta
        eps = 1e-8
        relative_gain = benefit / max(abs(base_score), eps)

        # Settle: confirmed / refuted / inconclusive / failed
        if ablation_receipt.get("status") == "ACCEPTED" and relative_gain >= 0.005:
            status = "confirmed"
        elif benefit < 0:
            status = "refuted"
        else:
            status = "inconclusive"

        elapsed_s = time.time() - t0
        end_rss = start_rss
        try:
            import resource
            end_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except Exception:
            pass

        res_measurements = {
            "elapsed_seconds": elapsed_s,
            "peak_rss_kb": end_rss,
            "total_folds": ablation_receipt.get("total_folds", 5),
        }

        receipt = HypothesisActionReceipt(
            action_id=action_id,
            status=status,
            action=action_name,
            parent_candidate_id=payload["parent_candidate_id"],
            split_id=payload["split_id"],
            paired_metric_delta=cv_delta,
            relative_gain=relative_gain,
            versioned_transform_lineage=[lineage_record],
            fitted_fold_references=[f"fold_{i}" for i in range(ablation_receipt.get("total_folds", 5))],
            resource_measurements=res_measurements,
            message=f"Action settled as '{status}' (benefit={benefit:.5f}, relative_gain={relative_gain:.4%})",
            payload_hash=payload_hash,
        )

        self.idempotency_cache[action_id] = (payload_hash, receipt)
        self._save_ledger()
        return receipt
