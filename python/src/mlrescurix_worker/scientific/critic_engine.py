"""M2-Critic Adversarial Audit Engine & Hard Assertion Gate (Phase 5).

Conforms strictly to `contracts/v2/critique_receipt.schema.json`.
Provides:
- Independent pre-freeze audit of candidates against dataset topology contracts.
- Deterministic evaluation of the 4 Hard Assertions:
  * Hard Assertion 1 (Target Memorization & Leakage Veto):
    VETO if Metric_train - Metric_val > 0.12 (or Metric_train > 0.99 with Metric_val < 0.75),
    or if any single derived feature accounts for > 65% of total model split gain
    AND originates from a QUASI_ID_DECOMPOSE or high-cardinality column.
  * Hard Assertion 2 (Unexploited Signal Veto):
    VETO if any column classified as SHORT_TEXT_EMBED (e.g. Name) or QUASI_ID_DECOMPOSE
    (e.g. Ticket, Cabin) was completely dropped without extracting sub-tokens,
    prefixes, or group aggregations.
  * Hard Assertion 3 (Model Monoculture Veto):
    VETO if fewer than 3 distinct model families have been evaluated in Tier 1
    before attempting final packaging.
  * Hard Assertion 4 (Topology & Imbalance Veto):
    VETO if entity_group_col or temporal_col is non-null in topology but standard
    random KFold/StratifiedKFold was used, or if validation metric <= 0.52 (balanced
    accuracy) / <= 0.01 (R2).
- Generation of error slice hypotheses linking ErrorForensicsReport worst slices
  to concrete M2-FE remediation actions.
- SQLite synchronization to `critique_receipts` v11 table.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jsonschema

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path("contracts/v2/critique_receipt.schema.json")
_CACHED_SCHEMA: Optional[Dict[str, Any]] = None


def get_critique_receipt_schema() -> Dict[str, Any]:
    """Load and cache contracts/v2/critique_receipt.schema.json."""
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is None:
        p = SCHEMA_PATH
        if not p.is_file():
            for parent in Path(__file__).resolve().parents:
                cand = parent / "contracts" / "v2" / "critique_receipt.schema.json"
                if cand.is_file():
                    p = cand
                    break
        if not p.is_file():
            raise FileNotFoundError(f"critique_receipt.schema.json not found at {SCHEMA_PATH}")
        with open(p, "r", encoding="utf-8") as f:
            _CACHED_SCHEMA = json.load(f)
    return _CACHED_SCHEMA


@dataclass
class TargetMemorizationCheck:
    passed: bool
    train_metric: float
    val_metric: float
    gap: float
    max_allowed_gap: float = 0.12
    dominant_feature: Optional[str] = None
    dominant_gain_ratio: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "train_metric": float(self.train_metric),
            "val_metric": float(self.val_metric),
            "gap": float(round(self.gap, 6)),
            "max_allowed_gap": float(self.max_allowed_gap),
            "dominant_feature": self.dominant_feature,
            "dominant_gain_ratio": float(round(self.dominant_gain_ratio, 6)) if self.dominant_gain_ratio is not None else None,
        }


@dataclass
class UnexploitedSignalsCheck:
    passed: bool
    dropped_signals: List[str] = field(default_factory=list)
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "dropped_signals": list(self.dropped_signals),
            "details": self.details,
        }


@dataclass
class ModelDiversityCheck:
    passed: bool
    distinct_families_evaluated: int
    families: List[str] = field(default_factory=list)
    min_required: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "distinct_families_evaluated": int(self.distinct_families_evaluated),
            "families": list(self.families),
            "min_required": int(self.min_required),
        }


@dataclass
class TopologyAndImbalanceCheck:
    passed: bool
    cv_scheme_used: str
    required_cv_scheme: str
    score_floor_met: bool
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "cv_scheme_used": str(self.cv_scheme_used),
            "required_cv_scheme": str(self.required_cv_scheme),
            "score_floor_met": bool(self.score_floor_met),
            "score": float(round(self.score, 6)),
        }


@dataclass
class CritiqueReceipt:
    critique_id: str
    candidate_id: str
    epoch: int
    status: str  # "APPROVED" or "VETO"
    blocking_flaws: List[str]
    audit_checks: Dict[str, Any]
    soft_warnings: List[str] = field(default_factory=list)
    error_slice_hypotheses: List[str] = field(default_factory=list)
    reviewer_notes: str = ""
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    schema_version: str = "v2"

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "schema_version": "v2",
            "critique_id": self.critique_id,
            "candidate_id": self.candidate_id,
            "epoch": int(self.epoch),
            "status": self.status,
            "blocking_flaws": list(self.blocking_flaws),
            "soft_warnings": list(self.soft_warnings),
            "audit_checks": self.audit_checks,
            "error_slice_hypotheses": list(self.error_slice_hypotheses),
            "reviewer_notes": self.reviewer_notes[:8192],
            "timestamp_ms": int(self.timestamp_ms),
        }
        return res

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def validate(self) -> None:
        """Validate receipt against Draft 2020-12 critique_receipt schema."""
        schema = get_critique_receipt_schema()
        jsonschema.validate(instance=self.to_dict(), schema=schema)


def normalize_family_name(family: str) -> str:
    """Normalize model family string to canonical family token."""
    fam = family.lower().strip()
    if fam in ("lgbm", "lightgbm", "lgb"):
        return "lgbm"
    if fam in ("catboost", "cb"):
        return "catboost"
    if fam in ("xgboost", "xgb"):
        return "xgboost"
    if fam in ("hist_gbdt", "histgbdt", "hgb"):
        return "hist_gbdt"
    if fam in ("penalized_linear", "linear", "logistic", "ridge", "elasticnet"):
        return "penalized_linear"
    if fam in ("neural_tabular", "mlp", "tabnet", "resnet_tabular", "neural"):
        return "neural_tabular"
    return fam


class CriticEngine:
    """M2-Critic Adversarial Audit Engine implementing the 4 Hard Assertions."""

    def __init__(self, max_allowed_gap: float = 0.12, min_distinct_families: int = 3):
        self.max_allowed_gap = float(max_allowed_gap)
        self.min_distinct_families = int(min_distinct_families)

    def audit(
        self,
        candidate_id: Optional[str] = None,
        epoch: int = 1,
        topology: Optional[Union[Dict[str, Any], Any]] = None,
        eval_result: Optional[Union[Dict[str, Any], Any]] = None,
        forensics_report: Optional[Union[Dict[str, Any], Any]] = None,
        features_used: Optional[Sequence[str]] = None,
        distinct_families_evaluated: Optional[Sequence[str]] = None,
        cv_scheme_used: Optional[str] = None,
        reviewer_notes: Optional[str] = None,
        candidate_dir: Optional[Union[str, Path]] = None,
        ledger: Optional[Any] = None,
        eda_dossier: Optional[Any] = None,
        candidate: Optional[Union[Dict[str, Any], Any]] = None,
        error_forensics: Optional[Union[Dict[str, Any], Any]] = None,
        evaluated_model_families: Optional[Sequence[str]] = None,
        derived_feature_gains: Optional[Dict[str, float]] = None,
        db_path: Optional[Union[str, Path]] = None,
        project_id: str = "default",
        session_id: str = "session_001",
        **kwargs: Any,
    ) -> CritiqueReceipt:
        """Execute comprehensive adversarial audit against candidate and locked topology."""
        if candidate is not None:
            if candidate_id is None:
                candidate_id = candidate.get("candidate_id") if isinstance(candidate, dict) else getattr(candidate, "candidate_id", "cand_apex")
            if eval_result is None:
                eval_result = candidate
            if features_used is None:
                features_used = candidate.get("features") if isinstance(candidate, dict) else getattr(candidate, "features", None)
            if distinct_families_evaluated is None and isinstance(candidate, dict) and "submodels" in candidate:
                distinct_families_evaluated = candidate["submodels"]

        candidate_id = candidate_id or "candidate"
        if error_forensics is not None and forensics_report is None:
            forensics_report = error_forensics
        if evaluated_model_families is not None and distinct_families_evaluated is None:
            distinct_families_evaluated = evaluated_model_families

        blocking_flaws: List[str] = []
        soft_warnings: List[str] = []

        # Convert topology contract to dict if needed
        topo_dict = topology if isinstance(topology, dict) else (getattr(topology, "contract", {}) or getattr(topology, "to_dict", lambda: {})())
        task_type = str(topo_dict.get("task_type", "binary_classification"))
        is_classification = "classification" in task_type or task_type in ("binary", "multiclass")

        # 1. Parse evaluation metrics and generalization gap
        train_metric = 0.0
        val_metric = 0.0
        dominant_feature: Optional[str] = None
        dominant_gain_ratio: Optional[float] = None

        if eval_result is not None:
            if hasattr(eval_result, "mean_train_score") and hasattr(eval_result, "mean_cv_score"):
                train_metric = float(eval_result.mean_train_score)
                val_metric = float(eval_result.mean_cv_score)
                if hasattr(eval_result, "feature_importances") and hasattr(eval_result, "feature_names"):
                    imps = np.asarray(eval_result.feature_importances, dtype=float)
                    names = list(eval_result.feature_names)
                    total_gain = float(np.sum(imps))
                    if total_gain > 1e-12 and len(names) == len(imps):
                        max_idx = int(np.argmax(imps))
                        dominant_feature = names[max_idx]
                        dominant_gain_ratio = float(imps[max_idx] / total_gain)
            elif isinstance(eval_result, dict):
                train_metric = float(eval_result.get("train_score", eval_result.get("mean_train_score", 0.0)))
                val_metric = float(eval_result.get("cv_score", eval_result.get("mean_cv_score", eval_result.get("val_metric", 0.0))))
                dominant_feature = eval_result.get("dominant_feature")
                dominant_gain_ratio = eval_result.get("dominant_gain_ratio")

        gap = float(train_metric - val_metric)

        # Inspect feature importances from forensics_report if available
        if forensics_report is not None:
            report_dict = forensics_report if isinstance(forensics_report, dict) else getattr(forensics_report, "to_dict", lambda: {})()
            top_feats = report_dict.get("top_features_by_importance", [])
            if top_feats and dominant_gain_ratio is None:
                total_imp = sum(f.get("importance", 0.0) for f in top_feats)
                if total_imp > 1e-12:
                    dom = top_feats[0]
                    dominant_feature = dom.get("feature")
                    dominant_gain_ratio = dom.get("importance", 0.0) / total_imp

        if derived_feature_gains:
            for feat_name, gain_ratio in derived_feature_gains.items():
                if dominant_gain_ratio is None or gain_ratio > dominant_gain_ratio:
                    dominant_feature = feat_name
                    dominant_gain_ratio = float(gain_ratio)

        # =========================================================================
        # Hard Assertion 1: Target Memorization & Leakage Veto
        # =========================================================================
        target_mem_passed = True

        # Check gap threshold (> 0.12)
        if gap > self.max_allowed_gap:
            target_mem_passed = False
            blocking_flaws.append(
                f"Target memorization veto: Train-val generalization gap {gap:.4f} exceeds threshold "
                f"{self.max_allowed_gap:.2f} (train {train_metric:.4f} vs val {val_metric:.4f})."
            )

        # Check near-perfect train with collapsed val (train > 0.99 and val < 0.75)
        if train_metric > 0.99 and val_metric < 0.75:
            target_mem_passed = False
            blocking_flaws.append(
                f"Target memorization veto: Near-perfect training memorization ({train_metric:.4f} > 0.99) "
                f"with collapsed validation ({val_metric:.4f} < 0.75)."
            )

        # Check dominant feature gain (> 65% total gain from quasi-ID or high cardinality)
        if dominant_feature and dominant_gain_ratio is not None and dominant_gain_ratio > 0.65:
            base_col = dominant_feature.split("__")[0]
            col_classes = topo_dict.get("column_classifications", {})
            col_meta = topo_dict.get("column_metadata", {}).get(base_col, {})
            uniqueness_ratio = col_meta.get("uniqueness_ratio", 0.0)
            is_quasi_or_high_card = (
                col_classes.get(base_col) in ("QUASI_ID_DECOMPOSE", "DROP_SURROGATE_ID")
                or uniqueness_ratio > 0.30
                or col_meta.get("cardinality", 0) > 100
            )
            if is_quasi_or_high_card:
                target_mem_passed = False
                blocking_flaws.append(
                    f"Target leakage veto: Feature '{dominant_feature}' accounts for {dominant_gain_ratio * 100:.1f}% "
                    f"(> 65%) of total split gain and originates from quasi-ID or high-cardinality column '{base_col}'."
                )

        target_mem_check = TargetMemorizationCheck(
            passed=target_mem_passed,
            train_metric=train_metric,
            val_metric=val_metric,
            gap=gap,
            max_allowed_gap=self.max_allowed_gap,
            dominant_feature=dominant_feature,
            dominant_gain_ratio=dominant_gain_ratio,
        )

        # =========================================================================
        # Hard Assertion 2: Unexploited Signal Veto
        # =========================================================================
        active_features = set()
        if features_used is not None:
            active_features.update(features_used)
        if candidate_dir is not None:
            cand_p = Path(candidate_dir)
            feat_p = cand_p / "features.py"
            if feat_p.is_file():
                try:
                    feat_text = feat_p.read_text(encoding="utf-8")
                    for line in feat_text.splitlines():
                        if "=" in line:
                            active_features.add(line.split("=")[0].strip())
                except Exception:
                    pass
            man_p = cand_p / "manifest.json"
            if man_p.is_file():
                try:
                    man = json.loads(man_p.read_text(encoding="utf-8"))
                    for f_name in man.get("raw_input_schema", {}).get("features", []):
                        active_features.add(f_name)
                except Exception:
                    pass

        col_classes = topo_dict.get("column_classifications", {})
        dropped_signals: List[str] = []

        for col, classification in col_classes.items():
            if classification in ("SHORT_TEXT_EMBED", "QUASI_ID_DECOMPOSE"):
                # Check whether any active feature derives from col
                col_lower = col.lower()
                derived = False
                for feat in active_features:
                    feat_lower = feat.lower()
                    if (
                        feat == col
                        or feat.startswith(f"{col}__")
                        or feat.startswith(f"{col}_")
                        or feat_lower.startswith(f"{col_lower}__")
                        or feat_lower.startswith(f"{col_lower}_")
                    ):
                        derived = True
                        break
                    # Special semantic keywords
                    if col == "Name" and any(k in feat_lower for k in ("honorific", "title", "surname", "svd", "text")):
                        derived = True
                        break
                    if col in ("Ticket", "Cabin") and any(k in feat_lower for k in ("group_size", "prefix", "deck", "is_nan")):
                        derived = True
                        break

                if not derived:
                    dropped_signals.append(col)
                    blocking_flaws.append(
                        f"Column '{col}' was dropped without decomposition/embedding."
                    )

        unexploited_passed = len(dropped_signals) == 0
        unexploited_details = None
        if not unexploited_passed:
            unexploited_details = (
                f"Unexploited signal veto: Column(s) {dropped_signals} classified as SHORT_TEXT_EMBED "
                f"or QUASI_ID_DECOMPOSE were dropped without sub-token extraction, prefix EB, or group aggregations."
            )

        unexploited_check = UnexploitedSignalsCheck(
            passed=unexploited_passed,
            dropped_signals=dropped_signals,
            details=unexploited_details,
        )

        # =========================================================================
        # Hard Assertion 3: Model Monoculture Veto
        # =========================================================================
        evaluated_fams: Set[str] = set()
        if distinct_families_evaluated is not None:
            for f in distinct_families_evaluated:
                evaluated_fams.add(normalize_family_name(f))

        # Also inspect ledger epochs if passed
        if ledger is not None:
            epochs_data = getattr(ledger, "epochs", [])
            if isinstance(epochs_data, list):
                for ep in epochs_data:
                    m_results = ep.get("model_results", []) if isinstance(ep, dict) else getattr(ep, "model_results", [])
                    for mr in m_results:
                        fam_raw = mr.get("family") if isinstance(mr, dict) else getattr(mr, "family", "")
                        if fam_raw:
                            evaluated_fams.add(normalize_family_name(fam_raw))

        # If candidate is a blend, inspect submodels
        if eval_result is not None and isinstance(eval_result, dict) and "submodels" in eval_result:
            for sm in eval_result["submodels"]:
                if isinstance(sm, str):
                    evaluated_fams.add(normalize_family_name(sm))
                elif isinstance(sm, dict) and "family" in sm:
                    evaluated_fams.add(normalize_family_name(sm["family"]))

        diversity_count = len(evaluated_fams)
        model_diversity_passed = diversity_count >= self.min_distinct_families
        if not model_diversity_passed:
            blocking_flaws.append(
                f"Model monoculture veto: only {diversity_count} distinct model families evaluated "
                f"({sorted(evaluated_fams)}), minimum {self.min_distinct_families} distinct families required before final packaging."
            )

        model_diversity_check = ModelDiversityCheck(
            passed=model_diversity_passed,
            distinct_families_evaluated=diversity_count,
            families=sorted(evaluated_fams),
            min_required=self.min_distinct_families,
        )

        # =========================================================================
        # Hard Assertion 4: Topology & Imbalance Veto
        # =========================================================================
        entity_group_col = topo_dict.get("entity_group_col")
        temporal_col = topo_dict.get("temporal_col")
        required_cv_scheme = str(topo_dict.get("cv_scheme", "stratified_kfold"))
        effective_cv_scheme = str(cv_scheme_used or required_cv_scheme)

        topology_passed = True

        # Check entity grouping
        if entity_group_col:
            if effective_cv_scheme not in ("stratified_group_kfold", "group_kfold"):
                topology_passed = False
                blocking_flaws.append(
                    f"Topology veto: entity_group_col '{entity_group_col}' is non-null in topology contract, "
                    f"but standard CV scheme '{effective_cv_scheme}' was used instead of 'stratified_group_kfold' / 'group_kfold'."
                )

        # Check temporal ordering
        if temporal_col:
            if effective_cv_scheme != "purged_time_series_split":
                topology_passed = False
                blocking_flaws.append(
                    f"Topology veto: temporal_col '{temporal_col}' is non-null in topology contract, "
                    f"but CV scheme '{effective_cv_scheme}' was used instead of 'purged_time_series_split'."
                )

        # Check validation score floor
        score_floor_met = True
        if is_classification:
            if val_metric <= 0.52:
                score_floor_met = False
                topology_passed = False
                blocking_flaws.append(
                    f"Score floor veto: Validation balanced accuracy {val_metric:.4f} <= 0.52 (random chance floor)."
                )
        else:
            if val_metric <= 0.01:
                score_floor_met = False
                topology_passed = False
                blocking_flaws.append(
                    f"Score floor veto: Validation R2 score {val_metric:.4f} <= 0.01."
                )

        topology_imbalance_check = TopologyAndImbalanceCheck(
            passed=topology_passed,
            cv_scheme_used=effective_cv_scheme,
            required_cv_scheme=required_cv_scheme,
            score_floor_met=score_floor_met,
            score=val_metric,
        )

        # =========================================================================
        # Generate Error Slice Hypotheses from Forensics Report
        # =========================================================================
        error_slice_hypotheses: List[str] = []
        if forensics_report is not None:
            rep = forensics_report if isinstance(forensics_report, dict) else getattr(forensics_report, "to_dict", lambda: {})()
            for slice_rec in rep.get("worst_slices", [])[:3]:
                rule = slice_rec.get("slice_rule", "unknown")
                err_rate = float(slice_rec.get("slice_error_rate", 0.0))
                samples = int(slice_rec.get("sample_count", 0))
                lift = float(slice_rec.get("lift", 1.0))
                hyp_stmt = (
                    f"M2-FE Hypothesis: Worst slice '{rule}' exhibits {err_rate * 100:.1f}% error rate "
                    f"({samples} samples, lift {lift:.2f}); recommend target-encoding interactions or subgroup imputation."
                )
                error_slice_hypotheses.append(hyp_stmt)

        # Determine overall status
        status = "APPROVED" if not blocking_flaws else "VETO"

        # Construct notes
        notes = reviewer_notes or (
            "Candidate verified and approved for zero-skew bundle packaging."
            if status == "APPROVED"
            else f"Candidate VETOED due to {len(blocking_flaws)} blocking flaw(s)."
        )

        receipt = CritiqueReceipt(
            critique_id=f"critique_{candidate_id}_ep{epoch}_{uuid.uuid4().hex[:8]}",
            candidate_id=candidate_id,
            epoch=epoch,
            status=status,
            blocking_flaws=blocking_flaws,
            soft_warnings=soft_warnings,
            audit_checks={
                "target_memorization": target_mem_check.to_dict(),
                "unexploited_signals": unexploited_check.to_dict(),
                "model_diversity": model_diversity_check.to_dict(),
                "topology_and_imbalance": topology_imbalance_check.to_dict(),
            },
            error_slice_hypotheses=error_slice_hypotheses,
            reviewer_notes=notes,
        )

        if db_path is not None:
            record_critique_receipt_in_db(
                receipt=receipt,
                db_path=db_path,
                project_id=project_id,
                session_id=session_id,
            )

        receipt.validate()
        return receipt


def record_critique_receipt_in_db(
    receipt: Union[CritiqueReceipt, Dict[str, Any], str, Path],
    db_path: Optional[Union[str, Path, CritiqueReceipt, Dict[str, Any]]] = None,
    project_id: str = "default",
    session_id: str = "session_001",
) -> None:
    """Persist a CritiqueReceipt into the SQLite v11 critique_receipts table."""
    if isinstance(receipt, (str, Path)) and (isinstance(db_path, (CritiqueReceipt, dict)) or hasattr(db_path, "to_dict")):
        receipt, db_path = db_path, receipt

    rec_dict = receipt.to_dict() if isinstance(receipt, CritiqueReceipt) or hasattr(receipt, "to_dict") else receipt
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(p)
    try:
        cur = conn.cursor()
        # Ensure projects table exists if foreign keys are active
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT,
                created_ms INTEGER
            )
            """
        )
        cur.execute(
            "INSERT OR IGNORE INTO projects (id, name, created_ms) VALUES (?, ?, ?)",
            (project_id, project_id, int(time.time() * 1000)),
        )

        # Ensure critique_receipts table exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS critique_receipts (
                project TEXT NOT NULL REFERENCES projects(id),
                critique_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                epoch INTEGER NOT NULL CHECK(epoch >= 0),
                status TEXT NOT NULL CHECK(status IN ('APPROVED', 'VETO')),
                blocking_flaws_json TEXT NOT NULL,
                soft_warnings_json TEXT NOT NULL DEFAULT '[]',
                audit_checks_json TEXT NOT NULL,
                error_slice_hypotheses_json TEXT NOT NULL DEFAULT '[]',
                reviewer_notes TEXT,
                receipt_json TEXT NOT NULL,
                evaluated_at_ms INTEGER NOT NULL,
                PRIMARY KEY(project, critique_id)
            )
            """
        )

        cur.execute(
            """
            INSERT OR REPLACE INTO critique_receipts (
                project,
                critique_id,
                candidate_id,
                session_id,
                epoch,
                status,
                blocking_flaws_json,
                soft_warnings_json,
                audit_checks_json,
                error_slice_hypotheses_json,
                reviewer_notes,
                receipt_json,
                evaluated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                rec_dict["critique_id"],
                rec_dict["candidate_id"],
                session_id,
                int(rec_dict["epoch"]),
                rec_dict["status"],
                json.dumps(rec_dict["blocking_flaws"]),
                json.dumps(rec_dict.get("soft_warnings", [])),
                json.dumps(rec_dict["audit_checks"]),
                json.dumps(rec_dict.get("error_slice_hypotheses", [])),
                rec_dict.get("reviewer_notes", ""),
                json.dumps(rec_dict),
                int(rec_dict["timestamp_ms"]),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def critic_audit_candidate(
    candidate_id: Optional[str] = None,
    epoch: int = 1,
    topology: Optional[Union[Dict[str, Any], Any]] = None,
    eval_result: Optional[Union[Dict[str, Any], Any]] = None,
    forensics_report: Optional[Union[Dict[str, Any], Any]] = None,
    features_used: Optional[Sequence[str]] = None,
    distinct_families_evaluated: Optional[Sequence[str]] = None,
    cv_scheme_used: Optional[str] = None,
    reviewer_notes: Optional[str] = None,
    candidate_dir: Optional[Union[str, Path]] = None,
    ledger: Optional[Any] = None,
    eda_dossier: Optional[Any] = None,
    db_path: Optional[Union[str, Path]] = None,
    project_id: str = "default",
    session_id: str = "session_001",
    candidate: Optional[Union[Dict[str, Any], Any]] = None,
    error_forensics: Optional[Union[Dict[str, Any], Any]] = None,
    evaluated_model_families: Optional[Sequence[str]] = None,
    derived_feature_gains: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> CritiqueReceipt:
    """Tool 5: Execute candidate audit via CriticEngine, with optional SQLite persistence."""
    engine = CriticEngine()
    receipt = engine.audit(
        candidate_id=candidate_id,
        epoch=epoch,
        topology=topology,
        eval_result=eval_result,
        forensics_report=forensics_report,
        features_used=features_used,
        distinct_families_evaluated=distinct_families_evaluated,
        cv_scheme_used=cv_scheme_used,
        reviewer_notes=reviewer_notes,
        candidate_dir=candidate_dir,
        ledger=ledger,
        eda_dossier=eda_dossier,
        candidate=candidate,
        error_forensics=error_forensics,
        evaluated_model_families=evaluated_model_families,
        derived_feature_gains=derived_feature_gains,
        db_path=db_path,
        project_id=project_id,
        session_id=session_id,
        **kwargs,
    )
    return receipt
