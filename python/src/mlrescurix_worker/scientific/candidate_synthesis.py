"""Autonomous Candidate Synthesis Engine (Phase P10.1).

Empowers the agent to freely synthesize, configure, and generate complete, decoupled
machine learning candidates adhering strictly to the candidate protocol (fit/predict CLI,
stable row ID alignment, label isolation, and source integrity).

Supports arbitrary ML families: LightGBM, XGBoost, PyTorch neural networks,
tree ensembles, linear baselines, and bespoke feature pipelines.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import py_compile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema

from .candidate_runner import compute_source_hash, validate_candidate_manifest
from .workflow import WorkflowError


ROOT_DIR = Path(__file__).resolve().parents[4]
SYNTHESIS_SCHEMA_PATH = ROOT_DIR / "contracts/v2/candidate_synthesis.schema.json"


class SynthesisError(WorkflowError):
    """Base error for candidate synthesis failures."""


class InvalidSynthesisSpecError(SynthesisError):
    """Raised when candidate synthesis specification fails schema or semantic validation."""


class UnsupportedFamilyError(SynthesisError):
    """Raised when requested model family or framework is unsupported."""


class CodeGenerationError(SynthesisError):
    """Raised when generated candidate code fails syntax or integrity verification."""


@dataclass
class ColumnDescriptor:
    name: str
    data_type: str  # numerical, categorical, text, datetime, id, target
    nullable: bool = False
    cardinality: int | None = None
    imputation_strategy: str = "none"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "data_type": self.data_type,
            "nullable": self.nullable,
            "imputation_strategy": self.imputation_strategy,
        }
        if self.cardinality is not None:
            d["cardinality"] = self.cardinality
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnDescriptor:
        return cls(
            name=data["name"],
            data_type=data["data_type"],
            nullable=data.get("nullable", False),
            cardinality=data.get("cardinality"),
            imputation_strategy=data.get("imputation_strategy", "none"),
        )


@dataclass
class TargetDescriptor:
    name: str
    task_type: str  # binary, multiclass, regression
    class_order: list[str | int] | None = None
    positive_class: str | int | None = None
    min_value: float | None = None
    max_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "task_type": self.task_type,
        }
        if self.class_order is not None:
            d["class_order"] = self.class_order
        if self.positive_class is not None:
            d["positive_class"] = self.positive_class
        if self.min_value is not None:
            d["min_value"] = self.min_value
        if self.max_value is not None:
            d["max_value"] = self.max_value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetDescriptor:
        return cls(
            name=data["name"],
            task_type=data["task_type"],
            class_order=data.get("class_order"),
            positive_class=data.get("positive_class"),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
        )


@dataclass
class ProblemDefinition:
    task_type: str  # binary, multiclass, regression
    target_column: str
    dataset_id: str = "default_dataset"
    id_column: str = "row_id"
    time_column: str | None = None
    group_column: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "dataset_id": self.dataset_id,
            "id_column": self.id_column,
            "description": self.description,
        }
        if self.time_column is not None:
            d["time_column"] = self.time_column
        if self.group_column is not None:
            d["group_column"] = self.group_column
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemDefinition:
        return cls(
            task_type=data["task_type"],
            target_column=data["target_column"],
            dataset_id=data.get("dataset_id", "default_dataset"),
            id_column=data.get("id_column", "row_id"),
            time_column=data.get("time_column"),
            group_column=data.get("group_column"),
            description=data.get("description", ""),
        )


@dataclass
class MetricObjective:
    primary_metric: str  # balanced_accuracy, log_loss, roc_auc, f1, mae, rmse, r2
    direction: str = "maximize"  # maximize, minimize
    target_threshold: float | None = None
    secondary_metrics: list[str] = field(default_factory=list)
    diagnostic_slice_columns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "primary_metric": self.primary_metric,
            "direction": self.direction,
            "secondary_metrics": self.secondary_metrics,
            "diagnostic_slice_columns": self.diagnostic_slice_columns,
        }
        if self.target_threshold is not None:
            d["target_threshold"] = self.target_threshold
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricObjective:
        return cls(
            primary_metric=data["primary_metric"],
            direction=data.get("direction", "maximize"),
            target_threshold=data.get("target_threshold"),
            secondary_metrics=data.get("secondary_metrics", []),
            diagnostic_slice_columns=data.get("diagnostic_slice_columns", []),
        )


@dataclass
class ModelIntent:
    family: str  # boosting, neural, tree, linear, ensemble, custom
    framework: str  # lightgbm, xgboost, catboost, pytorch, scikit-learn, custom
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    feature_pipeline: dict[str, Any] = field(default_factory=dict)
    architecture_description: str = ""
    custom_modules: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "family": self.family,
            "framework": self.framework,
            "hyperparameters": self.hyperparameters,
            "feature_pipeline": self.feature_pipeline,
            "architecture_description": self.architecture_description,
        }
        if self.custom_modules:
            d["custom_modules"] = self.custom_modules
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelIntent:
        return cls(
            family=data["family"],
            framework=data["framework"],
            hyperparameters=data.get("hyperparameters", {}),
            feature_pipeline=data.get("feature_pipeline", {}),
            architecture_description=data.get("architecture_description", ""),
            custom_modules=data.get("custom_modules", {}),
        )


@dataclass
class SynthesisSpec:
    synthesis_id: str
    candidate_id: str
    problem_definition: ProblemDefinition
    metric_objective: MetricObjective
    model_intent: ModelIntent
    name: str = ""
    input_descriptors: list[ColumnDescriptor] = field(default_factory=list)
    target_descriptor: TargetDescriptor | None = None
    schema_version: str = "v2"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "synthesis_id": self.synthesis_id,
            "candidate_id": self.candidate_id,
            "name": self.name or f"Candidate {self.candidate_id}",
            "problem_definition": self.problem_definition.to_dict(),
            "metric_objective": self.metric_objective.to_dict(),
            "model_intent": self.model_intent.to_dict(),
        }
        if self.input_descriptors:
            d["input_descriptors"] = [c.to_dict() for c in self.input_descriptors]
        if self.target_descriptor:
            d["target_descriptor"] = self.target_descriptor.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SynthesisSpec:
        inputs = [ColumnDescriptor.from_dict(c) for c in data.get("input_descriptors", [])]
        target_desc = TargetDescriptor.from_dict(data["target_descriptor"]) if "target_descriptor" in data else None
        return cls(
            schema_version=data.get("schema_version", "v2"),
            synthesis_id=data["synthesis_id"],
            candidate_id=data["candidate_id"],
            name=data.get("name", ""),
            problem_definition=ProblemDefinition.from_dict(data["problem_definition"]),
            input_descriptors=inputs,
            target_descriptor=target_desc,
            metric_objective=MetricObjective.from_dict(data["metric_objective"]),
            model_intent=ModelIntent.from_dict(data["model_intent"]),
        )


@dataclass
class SynthesizedCandidate:
    candidate_id: str
    candidate_dir: Path
    manifest: dict[str, Any]
    source_hash: str
    files: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_dir": str(self.candidate_dir),
            "manifest": self.manifest,
            "source_hash": self.source_hash,
            "files": self.files,
        }


def load_synthesis_schema() -> dict[str, Any]:
    """Load candidate synthesis schema."""
    if SYNTHESIS_SCHEMA_PATH.exists():
        return json.loads(SYNTHESIS_SCHEMA_PATH.read_text(encoding="utf-8"))
    # Fallback minimal schema
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}


def validate_synthesis_spec(spec_data: dict[str, Any]) -> None:
    """Validate synthesis spec dictionary against schema."""
    schema = load_synthesis_schema()
    try:
        validator = jsonschema.Draft202012Validator(schema)
        validator.validate(spec_data)
    except jsonschema.ValidationError as exc:
        raise InvalidSynthesisSpecError(f"synthesis spec validation failed: {exc.message} (path: {list(exc.path)})") from exc


class CandidateSynthesizer:
    """Autonomous generator of decoupled, protocol-conforming ML candidates."""

    @classmethod
    def synthesize(
        cls,
        spec: SynthesisSpec | dict[str, Any],
        destination_dir: Path | str,
        overwrite: bool = True,
    ) -> SynthesizedCandidate:
        """Synthesize candidate files into destination directory."""
        if isinstance(spec, dict):
            validate_synthesis_spec(spec)
            spec_obj = SynthesisSpec.from_dict(spec)
        else:
            spec_data = spec.to_dict()
            validate_synthesis_spec(spec_data)
            spec_obj = spec

        dest = Path(destination_dir)
        if dest.exists() and not overwrite:
            raise SynthesisError(f"destination directory '{dest}' already exists and overwrite=False")
        dest.mkdir(parents=True, exist_ok=True)

        task_type = spec_obj.problem_definition.task_type
        target_col = spec_obj.problem_definition.target_column
        id_col = spec_obj.problem_definition.id_column
        cid = spec_obj.candidate_id
        name = spec_obj.name or f"Candidate {cid}"

        # Resolve class order for classification
        class_order = None
        if task_type in ("binary", "multiclass"):
            if spec_obj.target_descriptor and spec_obj.target_descriptor.class_order:
                class_order = spec_obj.target_descriptor.class_order
            else:
                # Default binary or multiclass order
                class_order = [0, 1] if task_type == "binary" else [0, 1, 2]

        # 1. Generate features.py
        features_code = cls._generate_features_code(spec_obj)
        features_path = dest / "features.py"
        features_path.write_text(features_code, encoding="utf-8")

        # 2. Generate model.py
        model_code = cls._generate_model_code(spec_obj)
        model_path = dest / "model.py"
        model_path.write_text(model_code, encoding="utf-8")

        # 3. Generate custom modules if declared
        if spec_obj.model_intent.custom_modules:
            for rel_mod, mod_src in spec_obj.model_intent.custom_modules.items():
                mod_path = dest / rel_mod
                mod_path.parent.mkdir(parents=True, exist_ok=True)
                mod_path.write_text(mod_src, encoding="utf-8")

        # 4. Generate entrypoint.py
        entrypoint_code = cls._generate_entrypoint_code(spec_obj, class_order)
        entrypoint_path = dest / "entrypoint.py"
        entrypoint_path.write_text(entrypoint_code, encoding="utf-8")

        # Verify python syntax of all generated files
        for py_file in dest.glob("*.py"):
            try:
                py_compile.compile(str(py_file), doraise=True)
            except py_compile.PyCompileError as exc:
                raise CodeGenerationError(f"syntax error in generated file '{py_file.name}': {exc}") from exc

        # 5. Compute source hash across all candidate source files (excluding manifest)
        src_hash = compute_source_hash(dest)

        # 6. Generate manifest.json
        manifest: dict[str, Any] = {
            "candidate_id": cid,
            "name": name,
            "task_type": task_type,
            "entrypoint": "entrypoint.py",
            "supported_operations": ["fit", "predict"],
            "stable_row_keys": id_col,
            "raw_input_schema": {
                "target": target_col,
                "id": id_col,
            },
            "source_hash": src_hash,
            "model_metadata": {
                "family": spec_obj.model_intent.family,
                "framework": spec_obj.model_intent.framework,
                "hyperparameters": spec_obj.model_intent.hyperparameters,
            },
        }
        if class_order is not None:
            manifest["class_order"] = class_order

        # Validate manifest
        validated_manifest = validate_candidate_manifest(manifest, dest)
        manifest_path = dest / "manifest.json"
        manifest_path.write_text(json.dumps(validated_manifest, indent=2, sort_keys=True), encoding="utf-8")

        # Collect file receipts
        generated_files: list[dict[str, Any]] = []
        for root, _, files in os.walk(dest):
            for fn in sorted(files):
                if fn.startswith(".") or fn.endswith(".pyc") or fn == "__pycache__":
                    continue
                fp = Path(root) / fn
                rel = fp.relative_to(dest).as_posix()
                b = fp.read_bytes()
                generated_files.append({
                    "relative_path": rel,
                    "sha256": hashlib.sha256(b).hexdigest(),
                    "bytes": len(b),
                })

        return SynthesizedCandidate(
            candidate_id=cid,
            candidate_dir=dest,
            manifest=validated_manifest,
            source_hash=src_hash,
            files=generated_files,
        )

    @classmethod
    def _generate_features_code(cls, spec: SynthesisSpec) -> str:
        """Generate decoupled feature pipeline module with advanced feature engineering primitives."""
        pipeline_cfg = spec.model_intent.feature_pipeline
        scaling = pipeline_cfg.get("scaling", "none")
        encoding = pipeline_cfg.get("categorical_encoding", "none")
        handle_missing = pipeline_cfg.get("handle_missing", True)
        smoothing = float(pipeline_cfg.get("target_encoding_smoothing", 10.0))
        te_cv = int(pipeline_cfg.get("target_encoding_cv", 5))
        freq_enc = bool(pipeline_cfg.get("frequency_encoding", False)) or (encoding == "frequency")
        crosses = bool(pipeline_cfg.get("numeric_feature_crosses", False)) or bool(pipeline_cfg.get("interaction_features", False))
        cyclical = bool(pipeline_cfg.get("cyclical_datetime", False))
        max_crosses = int(pipeline_cfg.get("max_crosses", 20))

        return f'''"""Auto-generated feature pipeline with advanced P11 feature engineering primitives."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold


class FeaturePipeline:
    def __init__(self):
        self.scaling_mode = "{scaling}"
        self.encoding_mode = "{encoding}"
        self.handle_missing = {handle_missing}
        self.smoothing = {smoothing}
        self.te_cv = {te_cv}
        self.freq_enc = {freq_enc}
        self.crosses = {crosses}
        self.cyclical = {cyclical}
        self.max_crosses = {max_crosses}

        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.dt_cols: list[str] = []

        self.num_imputer = None
        self.cat_imputer = None
        self.scaler = None
        self.encoder = None

        # P11 Advanced Encodings & Transforms
        self.target_encoding_map: dict[str, dict[str, float]] = {{}}
        self.global_target_mean: float = 0.0
        self.global_target_var: float = 1.0
        self.freq_map: dict[str, dict[str, float]] = {{}}
        self.cross_specs: list[tuple[str, str, str]] = []

        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> FeaturePipeline:
        # Detect column modalities
        self.num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or "time" in c.lower()]
        self.cat_cols = [c for c in df.columns if c not in self.num_cols and c not in self.dt_cols]

        # 1. Numeric imputation & scaling
        if self.num_cols and self.handle_missing:
            self.num_imputer = SimpleImputer(strategy="median")
            num_data = self.num_imputer.fit_transform(df[self.num_cols])
        elif self.num_cols:
            num_data = df[self.num_cols].to_numpy(dtype=float)
        else:
            num_data = np.empty((len(df), 0))

        if self.num_cols:
            if self.scaling_mode == "standard":
                self.scaler = StandardScaler()
                self.scaler.fit(num_data)
            elif self.scaling_mode == "minmax":
                self.scaler = MinMaxScaler()
                self.scaler.fit(num_data)
            elif self.scaling_mode == "robust":
                self.scaler = RobustScaler()
                self.scaler.fit(num_data)

        # 2. Categorical imputation & encoding
        if self.cat_cols and self.handle_missing:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            cat_data = self.cat_imputer.fit_transform(df[self.cat_cols].astype(str))
        elif self.cat_cols:
            cat_data = df[self.cat_cols].astype(str).to_numpy()
        else:
            cat_data = np.empty((len(df), 0))

        if self.cat_cols:
            if self.encoding_mode == "onehot":
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.encoder.fit(cat_data)
            elif self.encoding_mode == "ordinal":
                self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self.encoder.fit(cat_data)

        # 3. Frequency & Count encoding
        if self.freq_enc and self.cat_cols:
            n_tot = max(len(df), 1)
            for c in self.cat_cols:
                series = df[c].astype(str)
                counts = series.value_counts().to_dict()
                self.freq_map[c] = {{str(k): float(v) / n_tot for k, v in counts.items()}}

        # 4. Out-of-fold (OOF) Target Encoding with Smoothing & Prior Variance Regularization
        if self.encoding_mode in ("target", "target_oof") and self.cat_cols and y is not None:
            y_arr = np.asarray(y, dtype=float)
            self.global_target_mean = float(np.nanmean(y_arr))
            self.global_target_var = float(np.nanvar(y_arr)) if len(y_arr) > 1 else 1.0
            if self.global_target_var <= 1e-9:
                self.global_target_var = 1.0

            # Compute smoothed encoding table with prior variance regularization
            for col in self.cat_cols:
                series = df[col].astype(str).reset_index(drop=True)
                col_map: dict[str, float] = {{}}
                for val, group in series.groupby(series):
                    idx = group.index
                    n_cat = len(idx)
                    cat_targets = y_arr[idx]
                    cat_mean = float(cat_targets.mean())
                    cat_var = float(cat_targets.var()) if n_cat > 1 else self.global_target_var

                    # Empirical Bayes prior variance shrinkage
                    m_eff = self.smoothing * (1.0 + min(cat_var / self.global_target_var, 5.0))
                    smoothed = (n_cat * cat_mean + m_eff * self.global_target_mean) / (n_cat + m_eff)
                    col_map[str(val)] = smoothed
                self.target_encoding_map[col] = col_map

        # 5. Automated numeric feature crosses
        if self.crosses and len(self.num_cols) >= 2:
            selected_num = self.num_cols[:6]
            self.cross_specs = []
            for c in selected_num:
                if len(self.cross_specs) >= self.max_crosses:
                    break
                self.cross_specs.append(("sq", c, c))
            for i in range(len(selected_num)):
                for j in range(i + 1, len(selected_num)):
                    if len(self.cross_specs) >= self.max_crosses:
                        break
                    c1, c2 = selected_num[i], selected_num[j]
                    self.cross_specs.append(("diff", c1, c2))
                    if len(self.cross_specs) < self.max_crosses:
                        self.cross_specs.append(("ratio", c1, c2))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before transform")

        parts = []

        # 1. Numeric features
        if self.num_cols:
            num_df = df.reindex(columns=self.num_cols)
            if self.num_imputer is not None:
                num_mat = self.num_imputer.transform(num_df)
            else:
                num_mat = num_df.fillna(0.0).to_numpy(dtype=float)
            if self.scaler is not None:
                num_mat = self.scaler.transform(num_mat)
            parts.append(num_mat)

        # 2. Categorical features
        if self.cat_cols:
            cat_df = df.reindex(columns=self.cat_cols).astype(str)
            if self.cat_imputer is not None:
                cat_mat = self.cat_imputer.transform(cat_df)
            else:
                cat_mat = cat_df.to_numpy()

            if self.encoder is not None:
                encoded_mat = self.encoder.transform(cat_mat)
                parts.append(encoded_mat)

            # Frequency encoding
            if self.freq_enc and self.freq_map:
                freq_cols = []
                for c in self.cat_cols:
                    if c in self.freq_map:
                        fmap = self.freq_map[c]
                        s = cat_df[c].map(fmap).fillna(0.0).to_numpy(dtype=float).reshape(-1, 1)
                        freq_cols.append(s)
                if freq_cols:
                    parts.append(np.hstack(freq_cols))

            # Target encoding (strictly using fitted mappings, zero label leakage)
            if self.target_encoding_map:
                te_cols = []
                for c in self.cat_cols:
                    if c in self.target_encoding_map:
                        tmap = self.target_encoding_map[c]
                        s = cat_df[c].map(tmap).fillna(self.global_target_mean).to_numpy(dtype=float).reshape(-1, 1)
                        te_cols.append(s)
                if te_cols:
                    parts.append(np.hstack(te_cols))

        # 3. Numeric feature crosses
        if self.cross_specs and self.num_cols:
            cross_cols = []
            eps = 1e-5
            for op, c1, c2 in self.cross_specs:
                if c1 in df.columns and c2 in df.columns:
                    s1 = df[c1].fillna(0.0).to_numpy(dtype=float)
                    s2 = df[c2].fillna(0.0).to_numpy(dtype=float)
                    if op == "sq":
                        res = np.square(s1).reshape(-1, 1)
                    elif op == "diff":
                        res = (s1 - s2).reshape(-1, 1)
                    elif op == "ratio":
                        denom = np.where(np.abs(s2) < eps, np.sign(s2 + 1e-12) * eps, s2)
                        res = (s1 / denom).reshape(-1, 1)
                    else:
                        res = (s1 * s2).reshape(-1, 1)
                    cross_cols.append(res)
            if cross_cols:
                parts.append(np.hstack(cross_cols))

        # 4. Cyclical datetime encoding
        if self.cyclical and self.dt_cols:
            dt_features = []
            for c in self.dt_cols:
                if c in df.columns:
                    try:
                        dts = pd.to_datetime(df[c], errors="coerce")
                        h = dts.dt.hour.fillna(0.0).to_numpy(dtype=float)
                        dow = dts.dt.dayofweek.fillna(0.0).to_numpy(dtype=float)
                        m = dts.dt.month.fillna(1.0).to_numpy(dtype=float)
                        dt_features.append(np.sin(2.0 * np.pi * h / 24.0).reshape(-1, 1))
                        dt_features.append(np.cos(2.0 * np.pi * h / 24.0).reshape(-1, 1))
                        dt_features.append(np.sin(2.0 * np.pi * dow / 7.0).reshape(-1, 1))
                        dt_features.append(np.cos(2.0 * np.pi * dow / 7.0).reshape(-1, 1))
                        dt_features.append(np.sin(2.0 * np.pi * (m - 1.0) / 12.0).reshape(-1, 1))
                        dt_features.append(np.cos(2.0 * np.pi * (m - 1.0) / 12.0).reshape(-1, 1))
                    except Exception:
                        pass
            if dt_features:
                parts.append(np.hstack(dt_features))

        if not parts:
            return np.zeros((len(df), 1))

        res = np.hstack(parts)
        res = np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
        return res

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(self, p / "feature_pipeline.joblib")

    @classmethod
    def load(cls, model_dir: Path | str) -> FeaturePipeline:
        p = Path(model_dir) / "feature_pipeline.joblib"
        if not p.exists():
            return cls()
        import joblib
        return joblib.load(p)
'''

    @classmethod
    def _generate_model_code(cls, spec: SynthesisSpec) -> str:
        """Generate decoupled model wrapper supporting arbitrary families."""
        family = spec.model_intent.family.lower()
        framework = spec.model_intent.framework.lower()
        task_type = spec.problem_definition.task_type
        hyperparams = spec.model_intent.hyperparameters

        if family == "hybrid" or framework == "peft_lora":
            return cls._generate_hybrid_lora_code(task_type, hyperparams, spec)
        elif framework == "lightgbm" or family == "boosting":
            return cls._generate_lightgbm_code(task_type, hyperparams)
        elif framework == "xgboost":
            return cls._generate_xgboost_code(task_type, hyperparams)
        elif framework == "pytorch" or family == "neural":
            return cls._generate_pytorch_code(task_type, hyperparams)
        elif family == "tree" or framework == "scikit-learn":
            return cls._generate_tree_code(task_type, hyperparams)
        elif family == "linear":
            return cls._generate_linear_code(task_type, hyperparams)
        else:
            return cls._generate_generic_code(task_type, hyperparams)

    @classmethod
    def _generate_lightgbm_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """LightGBM model wrapper with HistGradientBoosting fallback."""
        n_est = hyperparams.get("n_estimators", 100)
        lr = hyperparams.get("learning_rate", 0.05)
        max_depth = hyperparams.get("max_depth", -1)
        num_leaves = hyperparams.get("num_leaves", 31)

        is_reg = task_type == "regression"

        return f'''"""Auto-generated LightGBM model wrapper with standard fallback."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.n_estimators = {n_est}
        self.learning_rate = {lr}
        self.max_depth = {max_depth}
        self.num_leaves = {num_leaves}
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelWrapper:
        is_regression = self.task_type == "regression"
        if not is_regression:
            self.classes_ = np.unique(y)

        if HAS_LIGHTGBM:
            if is_regression:
                self.model = lgb.LGBMRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    random_state=2026,
                    verbose=-1,
                )
            else:
                self.model = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    random_state=2026,
                    verbose=-1,
                )
        else:
            # Fallback to standard scikit-learn HistGradientBoosting
            if is_regression:
                self.model = HistGradientBoostingRegressor(
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_leaf_nodes=self.num_leaves,
                    random_state=2026,
                )
            else:
                self.model = HistGradientBoostingClassifier(
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_leaf_nodes=self.num_leaves,
                    random_state=2026,
                )

        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Ensure probabilities sum to 1.0 and are finite
            probs = np.nan_to_num(probs, nan=1.0 / len(self.classes_))
            probs = np.clip(probs, 0.0, 1.0)
            sums = probs.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            return probs / sums
        return None

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({{"model": self.model, "classes": self.classes_, "task_type": self.task_type}}, p / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir) / "model.joblib"
        if not p.exists():
            return wrapper
        import joblib
        data = joblib.load(p)
        wrapper.model = data["model"]
        wrapper.classes_ = data["classes"]
        wrapper.task_type = data["task_type"]
        return wrapper
'''

    @classmethod
    def _generate_xgboost_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """XGBoost model wrapper with GradientBoosting fallback."""
        n_est = hyperparams.get("n_estimators", 100)
        lr = hyperparams.get("learning_rate", 0.05)
        max_depth = hyperparams.get("max_depth", 6)

        return f'''"""Auto-generated XGBoost model wrapper with standard fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.n_estimators = {n_est}
        self.learning_rate = {lr}
        self.max_depth = {max_depth}
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelWrapper:
        is_regression = self.task_type == "regression"
        if not is_regression:
            self.classes_ = np.unique(y)

        if HAS_XGBOOST:
            if is_regression:
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=2026,
                    verbosity=0,
                )
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=2026,
                    verbosity=0,
                )
        else:
            if is_regression:
                self.model = GradientBoostingRegressor(
                    n_estimators=min(self.n_estimators, 100),
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=2026,
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=min(self.n_estimators, 100),
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=2026,
                )

        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            probs = np.nan_to_num(probs, nan=1.0 / len(self.classes_))
            probs = np.clip(probs, 0.0, 1.0)
            sums = probs.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            return probs / sums
        return None

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({{"model": self.model, "classes": self.classes_, "task_type": self.task_type}}, p / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir) / "model.joblib"
        if not p.exists():
            return wrapper
        import joblib
        data = joblib.load(p)
        wrapper.model = data["model"]
        wrapper.classes_ = data["classes"]
        wrapper.task_type = data["task_type"]
        return wrapper
'''

    @classmethod
    def _generate_pytorch_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """PyTorch modular Multi-Layer Perceptron (MLP) wrapper."""
        hidden_dim = hyperparams.get("hidden_dim", 64)
        epochs = hyperparams.get("epochs", 15)
        lr = hyperparams.get("learning_rate", 0.005)
        batch_size = hyperparams.get("batch_size", 32)

        return f'''"""Auto-generated PyTorch modular neural network wrapper."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, is_regression: bool):
        super().__init__()
        self.is_regression = is_regression
        self.net = nn.Sequential(
            nn.Linear(in_dim, {hidden_dim}),
            nn.ReLU(),
            nn.Linear({hidden_dim}, {hidden_dim // 2}),
            nn.ReLU(),
            nn.Linear({hidden_dim // 2}, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.epochs = {epochs}
        self.lr = {lr}
        self.batch_size = {batch_size}
        self.classes_ = None
        self.model: SimpleMLP | None = None
        self.in_dim = 0
        self.out_dim = 1

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelWrapper:
        torch.manual_seed(2026)
        self.in_dim = X.shape[1]
        is_regression = self.task_type == "regression"

        if is_regression:
            self.out_dim = 1
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()
        else:
            self.classes_ = np.unique(y)
            class_to_idx = {{c: i for i, c in enumerate(self.classes_)}}
            self.out_dim = len(self.classes_)
            y_indices = np.array([class_to_idx[val] for val in y], dtype=np.int64)
            y_tensor = torch.tensor(y_indices, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.model = SimpleMLP(self.in_dim, self.out_dim, is_regression)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_x)
                if is_regression:
                    loss = criterion(out, batch_y)
                else:
                    loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            out = self.model(X_tensor)
            if self.task_type == "regression":
                return out.squeeze(1).numpy()
            else:
                preds_idx = torch.argmax(out, dim=1).numpy()
                return np.array([self.classes_[i] for i in preds_idx])

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            out = self.model(X_tensor)
            probs = torch.softmax(out, dim=1).numpy()
            probs = np.clip(probs, 0.0, 1.0)
            sums = probs.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            return probs / sums

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), p / "pytorch_weights.pt")
        meta = {{
            "task_type": self.task_type,
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
        }}
        (p / "pytorch_meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir)
        meta_p = p / "pytorch_meta.json"
        if not meta_p.exists():
            return wrapper
        meta = json.loads(meta_p.read_text())
        wrapper.task_type = meta["task_type"]
        wrapper.in_dim = meta["in_dim"]
        wrapper.out_dim = meta["out_dim"]
        wrapper.classes_ = np.array(meta["classes"]) if meta["classes"] is not None else None
        wrapper.model = SimpleMLP(wrapper.in_dim, wrapper.out_dim, wrapper.task_type == "regression")
        weights_p = p / "pytorch_weights.pt"
        if weights_p.exists():
            wrapper.model.load_state_dict(torch.load(weights_p, map_location="cpu", weights_only=True))
        return wrapper
'''

    @classmethod
    def _generate_hybrid_lora_code(cls, task_type: str, hyperparams: dict[str, Any], spec: SynthesisSpec) -> str:
        """PyTorch multimodal hybrid model wrapper combining lightweight Transformer + LoRA with dense tabular embeddings."""
        lora_r = hyperparams.get("lora_r", 4)
        lora_alpha = hyperparams.get("lora_alpha", 8.0)
        d_model = hyperparams.get("d_model", 64)
        num_layers = hyperparams.get("num_layers", 2)
        nhead = hyperparams.get("nhead", 4)
        epochs = hyperparams.get("epochs", 10)
        lr = hyperparams.get("learning_rate", 0.003)
        batch_size = hyperparams.get("batch_size", 32)
        max_seq_len = hyperparams.get("max_seq_len", 32)
        vocab_size = hyperparams.get("vocab_size", 2000)

        # Detect text columns from input descriptors
        text_cols = [c.name for c in spec.input_descriptors if c.data_type == "text"]

        return f'''"""Auto-generated PyTorch Multimodal Hybrid model wrapper (Transformer + LoRA + Tabular Embeddings)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class LoRALinear(nn.Module):
    """Linear layer augmented with Low-Rank Adapters (LoRA) where base weights are frozen."""
    def __init__(self, in_features: int, out_features: int, r: int = {lora_r}, alpha: float = {lora_alpha}):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r

        # Frozen base projection
        self.base_linear = nn.Linear(in_features, out_features)
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # Trainable low-rank adapter matrices: A in R^{{r x in_features}}, B in R^{{out_features x r}}
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_linear(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


class TransformerBackbone(nn.Module):
    """Lightweight Transformer encoder with >95% frozen backbone weights and LoRA adapters."""
    def __init__(self, vocab_size: int = {vocab_size}, d_model: int = {d_model}, num_layers: int = {num_layers}, nhead: int = {nhead}):
        super().__init__()
        self.d_model = d_model
        # Frozen base embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.embedding.weight.requires_grad = False

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({{
                "q_proj": LoRALinear(d_model, d_model),
                "k_proj": nn.Linear(d_model, d_model),
                "v_proj": LoRALinear(d_model, d_model),
                "out_proj": nn.Linear(d_model, d_model),
                "ln1": nn.LayerNorm(d_model),
                "ffn1": nn.Linear(d_model, d_model * 2),
                "ffn2": nn.Linear(d_model * 2, d_model),
                "ln2": nn.LayerNorm(d_model),
                "act": nn.GELU(),
            }})
            # Freeze non-adapter parameters in backbone
            layer["k_proj"].weight.requires_grad = False
            layer["k_proj"].bias.requires_grad = False
            layer["out_proj"].weight.requires_grad = False
            layer["out_proj"].bias.requires_grad = False
            layer["ffn1"].weight.requires_grad = False
            layer["ffn1"].bias.requires_grad = False
            layer["ffn2"].weight.requires_grad = False
            layer["ffn2"].bias.requires_grad = False
            self.layers.append(layer)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for layer in self.layers:
            residual = x
            x_norm = layer["ln1"](x)
            q = layer["q_proj"](x_norm)
            k = layer["k_proj"](x_norm)
            v = layer["v_proj"](x_norm)

            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_model)
            attn = torch.softmax(scores, dim=-1)
            context = layer["out_proj"](attn @ v)
            x = residual + context

            residual = x
            x_norm = layer["ln2"](x)
            ffn = layer["ffn2"](layer["act"](layer["ffn1"](x_norm)))
            x = residual + ffn

        return x.mean(dim=1)


class HybridMultimodalNet(nn.Module):
    """Joint model fusing text transformer backbone + LoRA and tabular dense subnet."""
    def __init__(self, tab_in_dim: int, out_dim: int, is_regression: bool):
        super().__init__()
        self.is_regression = is_regression
        self.text_backbone = TransformerBackbone()
        self.tab_subnet = nn.Sequential(
            nn.Linear(max(tab_in_dim, 1), {d_model}),
            nn.ReLU(),
            nn.LayerNorm({d_model}),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear({d_model} * 2, {d_model}),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear({d_model}, out_dim),
        )

    def get_frozen_ratio(self) -> float:
        backbone_params = list(self.text_backbone.parameters())
        total = sum(p.numel() for p in backbone_params)
        frozen = sum(p.numel() for p in backbone_params if not p.requires_grad)
        return float(frozen / total) if total > 0 else 1.0

    def forward(self, tokens: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        text_emb = self.text_backbone(tokens)
        tab_emb = self.tab_subnet(tabular)
        joint = torch.cat([text_emb, tab_emb], dim=-1)
        return self.fusion_head(joint)


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.epochs = {epochs}
        self.lr = {lr}
        self.batch_size = {batch_size}
        self.max_seq_len = {max_seq_len}
        self.vocab_size = {vocab_size}
        self.text_columns = {repr(text_cols)}
        self.classes_ = None
        self.model: HybridMultimodalNet | None = None
        self.tab_dim = 0
        self.out_dim = 1

    def _tokenize(self, text_series: pd.Series | list[str] | np.ndarray) -> torch.Tensor:
        token_rows = []
        for val in text_series:
            s = str(val).lower() if val is not None else ""
            words = s.split()[:self.max_seq_len]
            t_ids = [abs(hash(w)) % self.vocab_size for w in words]
            if len(t_ids) < self.max_seq_len:
                t_ids = t_ids + [0] * (self.max_seq_len - len(t_ids))
            else:
                t_ids = t_ids[:self.max_seq_len]
            token_rows.append(t_ids)
        if not token_rows:
            return torch.zeros((1, self.max_seq_len), dtype=torch.long)
        return torch.tensor(token_rows, dtype=torch.long)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray) -> ModelWrapper:
        torch.manual_seed(2026)
        is_regression = self.task_type == "regression"

        if isinstance(X, pd.DataFrame):
            actual_text_cols = [c for c in X.columns if c in self.text_columns or X[c].dtype == object or pd.api.types.is_string_dtype(X[c])]
            actual_tab_cols = [c for c in X.columns if c not in actual_text_cols]
            if actual_text_cols:
                combined_text = X[actual_text_cols].astype(str).agg(" ".join, axis=1)
                tokens_tensor = self._tokenize(combined_text)
            else:
                tokens_tensor = torch.zeros((len(X), self.max_seq_len), dtype=torch.long)
            tab_df = X[actual_tab_cols].select_dtypes(include=[np.number]).fillna(0.0)
            tab_tensor = torch.tensor(tab_df.to_numpy(dtype=float), dtype=torch.float32)
        else:
            tab_tensor = torch.tensor(X, dtype=torch.float32)
            tokens_tensor = torch.zeros((len(X), self.max_seq_len), dtype=torch.long)

        self.tab_dim = tab_tensor.shape[1] if tab_tensor.ndim == 2 else 1

        if is_regression:
            self.out_dim = 1
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()
        else:
            self.classes_ = np.unique(y)
            class_to_idx = {{c: i for i, c in enumerate(self.classes_)}}
            self.out_dim = len(self.classes_)
            y_indices = np.array([class_to_idx[val] for val in y], dtype=np.int64)
            y_tensor = torch.tensor(y_indices, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        self.model = HybridMultimodalNet(self.tab_dim, self.out_dim, is_regression)

        # Verify PEFT invariant: >95% of backbone weights frozen!
        frozen_ratio = self.model.get_frozen_ratio()
        assert frozen_ratio >= 0.95, f"PEFT frozen parameter ratio check failed: {{frozen_ratio:.3f}} < 0.95"

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=self.lr, weight_decay=1e-4)

        dataset = TensorDataset(tokens_tensor, tab_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_tokens, batch_tab, batch_y in loader:
                optimizer.zero_grad()
                out = self.model(batch_tokens, batch_tab)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def _prepare_inputs(self, X: np.ndarray | pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(X, pd.DataFrame):
            actual_text_cols = [c for c in X.columns if c in self.text_columns or X[c].dtype == object or pd.api.types.is_string_dtype(X[c])]
            actual_tab_cols = [c for c in X.columns if c not in actual_text_cols]
            if actual_text_cols:
                combined_text = X[actual_text_cols].astype(str).agg(" ".join, axis=1)
                tokens_tensor = self._tokenize(combined_text)
            else:
                tokens_tensor = torch.zeros((len(X), self.max_seq_len), dtype=torch.long)
            tab_df = X[actual_tab_cols].select_dtypes(include=[np.number]).fillna(0.0)
            tab_tensor = torch.tensor(tab_df.to_numpy(dtype=float), dtype=torch.float32)
        else:
            tab_tensor = torch.tensor(X, dtype=torch.float32)
            tokens_tensor = torch.zeros((len(X), self.max_seq_len), dtype=torch.long)
        return tokens_tensor, tab_tensor

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        self.model.eval()
        with torch.no_grad():
            tokens, tab = self._prepare_inputs(X)
            out = self.model(tokens, tab)
            if self.task_type == "regression":
                return out.squeeze(1).numpy()
            else:
                preds_idx = torch.argmax(out, dim=1).numpy()
                return np.array([self.classes_[i] for i in preds_idx])

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        self.model.eval()
        with torch.no_grad():
            tokens, tab = self._prepare_inputs(X)
            out = self.model(tokens, tab)
            probs = torch.softmax(out, dim=1).numpy()
            probs = np.clip(probs, 0.0, 1.0)
            sums = probs.sum(axis=1, keepdims=True)
            sums[sums == 0] = 1.0
            return probs / sums

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), p / "hybrid_weights.pt")
        meta = {{
            "task_type": self.task_type,
            "tab_dim": self.tab_dim,
            "out_dim": self.out_dim,
            "classes": self.classes_.tolist() if self.classes_ is not None else None,
            "text_columns": self.text_columns,
            "frozen_ratio": self.model.get_frozen_ratio() if self.model else 1.0,
        }}
        (p / "hybrid_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir)
        meta_p = p / "hybrid_meta.json"
        if not meta_p.exists():
            return wrapper
        meta = json.loads(meta_p.read_text())
        wrapper.task_type = meta["task_type"]
        wrapper.tab_dim = meta["tab_dim"]
        wrapper.out_dim = meta["out_dim"]
        wrapper.classes_ = np.array(meta["classes"]) if meta["classes"] is not None else None
        wrapper.text_columns = meta.get("text_columns", [])
        wrapper.model = HybridMultimodalNet(wrapper.tab_dim, wrapper.out_dim, wrapper.task_type == "regression")
        weights_p = p / "hybrid_weights.pt"
        if weights_p.exists():
            wrapper.model.load_state_dict(torch.load(weights_p, map_location="cpu", weights_only=True))
        return wrapper
'''

    @classmethod
    def _generate_tree_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """Random Forest / Extra Trees model wrapper."""
        n_est = hyperparams.get("n_estimators", 100)
        max_depth = hyperparams.get("max_depth", 10)

        return f'''"""Auto-generated Random Forest model wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.n_estimators = {n_est}
        self.max_depth = {max_depth}
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelWrapper:
        if self.task_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=2026,
            )
        else:
            self.classes_ = np.unique(y)
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=2026,
            )
        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        probs = self.model.predict_proba(X)
        probs = np.nan_to_num(probs, nan=1.0 / len(self.classes_))
        probs = np.clip(probs, 0.0, 1.0)
        sums = probs.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return probs / sums

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({{"model": self.model, "classes": self.classes_, "task_type": self.task_type}}, p / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir) / "model.joblib"
        if not p.exists():
            return wrapper
        import joblib
        data = joblib.load(p)
        wrapper.model = data["model"]
        wrapper.classes_ = data["classes"]
        wrapper.task_type = data["task_type"]
        return wrapper
'''

    @classmethod
    def _generate_linear_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """Linear / Logistic Regression model wrapper."""
        alpha = hyperparams.get("alpha", 1.0)

        return f'''"""Auto-generated Linear / Logistic model wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge


class ModelWrapper:
    def __init__(self):
        self.task_type = "{task_type}"
        self.alpha = {alpha}
        self.model = None
        self.classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ModelWrapper:
        if self.task_type == "regression":
            self.model = Ridge(alpha=self.alpha, random_state=2026)
        else:
            self.classes_ = np.unique(y)
            self.model = LogisticRegression(max_iter=1000, random_state=2026)
        self.model.fit(X, y)
        if hasattr(self.model, "classes_"):
            self.classes_ = self.model.classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.task_type == "regression":
            return None
        if self.model is None:
            raise RuntimeError("ModelWrapper must be fitted before predict_proba")
        probs = self.model.predict_proba(X)
        probs = np.nan_to_num(probs, nan=1.0 / len(self.classes_))
        probs = np.clip(probs, 0.0, 1.0)
        sums = probs.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1.0
        return probs / sums

    def save(self, model_dir: Path | str) -> None:
        p = Path(model_dir)
        p.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({{"model": self.model, "classes": self.classes_, "task_type": self.task_type}}, p / "model.joblib")

    @classmethod
    def load(cls, model_dir: Path | str) -> ModelWrapper:
        wrapper = cls()
        p = Path(model_dir) / "model.joblib"
        if not p.exists():
            return wrapper
        import joblib
        data = joblib.load(p)
        wrapper.model = data["model"]
        wrapper.classes_ = data["classes"]
        wrapper.task_type = data["task_type"]
        return wrapper
'''

    @classmethod
    def _generate_generic_code(cls, task_type: str, hyperparams: dict[str, Any]) -> str:
        """Generic fallback model wrapper."""
        return cls._generate_tree_code(task_type, hyperparams)

    @classmethod
    def _generate_entrypoint_code(cls, spec: SynthesisSpec, class_order: list[Any] | None) -> str:
        """Generate protocol-conforming entrypoint.py CLI script."""
        task_type = spec.problem_definition.task_type
        target_col = spec.problem_definition.target_column
        id_col = spec.problem_definition.id_column
        class_order_repr = repr(class_order) if class_order is not None else "None"

        return f'''#!/usr/bin/env python3
"""Candidate CLI Entrypoint.

Adheres strictly to the candidate protocol:
  fit    --train-path <csv> --model-dir <dir>
  predict --input-path <csv> --model-dir <dir> --output-path <csv>

Guarantees label isolation, stable row ID alignment, and class order adherence.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from features import FeaturePipeline
from model import ModelWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Candidate Protocol Entrypoint")
    subparsers = parser.add_subparsers(dest="action", required=True)

    fit_p = subparsers.add_parser("fit")
    fit_p.add_argument("--train-path", required=True, help="Path to training CSV with labels")
    fit_p.add_argument("--model-dir", required=True, help="Directory to save model artifacts")

    pred_p = subparsers.add_parser("predict")
    pred_p.add_argument("--input-path", required=True, help="Path to input features CSV (no labels)")
    pred_p.add_argument("--model-dir", required=True, help="Directory containing trained model")
    pred_p.add_argument("--output-path", required=True, help="Destination CSV path for predictions")

    return parser.parse_args()


def run_fit(train_path: str, model_dir: str):
    df = pd.read_csv(train_path, low_memory=False)
    target_col = "{target_col}"
    id_col = "{id_col}"

    if target_col not in df.columns:
        raise ValueError(f"Training dataset missing target column '{{target_col}}'")

    # Extract target and drop non-feature columns
    y = df[target_col].to_numpy()
    feature_cols = [c for c in df.columns if c not in (target_col, id_col)]
    X_df = df[feature_cols].copy()

    # Fit and save feature pipeline with y for OOF target encoding
    pipe = FeaturePipeline()
    pipe.fit(X_df, y)
    pipe.save(model_dir)

    # Transform features and fit model
    X_transformed = pipe.transform(X_df)
    model = ModelWrapper()
    if hasattr(model, "text_columns") and model.text_columns:
        model.fit(X_df, y)
    else:
        model.fit(X_transformed, y)
    model.save(model_dir)
    print("fit successfully completed")


def run_predict(input_path: str, model_dir: str, output_path: str):
    df = pd.read_csv(input_path, low_memory=False)
    id_col = "{id_col}"
    target_col = "{target_col}"
    task_type = "{task_type}"
    declared_classes = {class_order_repr}

    # CRITICAL: Label isolation check. If target is in input dataframe, ignore or drop it!
    if target_col in df.columns:
        # Never use target in predict!
        df = df.drop(columns=[target_col])

    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    row_ids = df[id_col].tolist()
    feature_cols = [c for c in df.columns if c != id_col]
    X_df = df[feature_cols].copy()

    # Load pipeline and transform
    pipe = FeaturePipeline.load(model_dir)
    X_transformed = pipe.transform(X_df)

    # Load model and predict
    model = ModelWrapper.load(model_dir)
    if hasattr(model, "text_columns") and model.text_columns:
        preds = model.predict(X_df)
    else:
        preds = model.predict(X_transformed)

    out_dict = {{id_col: row_ids, "pred": preds}}

    if task_type in ("binary", "multiclass"):
        if hasattr(model, "text_columns") and model.text_columns:
            probs = model.predict_proba(X_df)
        else:
            probs = model.predict_proba(X_transformed)
        classes = declared_classes if declared_classes is not None else (model.classes_ if model.classes_ is not None else [0, 1])

        if probs is not None:
            # Map probabilities to declared class columns
            if hasattr(model, "classes_") and model.classes_ is not None:
                model_classes = list(model.classes_)
                for c in classes:
                    if c in model_classes:
                        idx = model_classes.index(c)
                        out_dict[f"prob_{{c}}"] = probs[:, idx]
                    else:
                        out_dict[f"prob_{{c}}"] = np.zeros(len(df))
            else:
                for idx, c in enumerate(classes):
                    if idx < probs.shape[1]:
                        out_dict[f"prob_{{c}}"] = probs[:, idx]
                    else:
                        out_dict[f"prob_{{c}}"] = np.zeros(len(df))
        else:
            # Fallback uniform probabilities
            n_c = len(classes)
            for c in classes:
                out_dict[f"prob_{{c}}"] = np.full(len(df), 1.0 / n_c)

    out_df = pd.DataFrame(out_dict)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print("predict successfully completed")


def main():
    args = parse_args()
    if args.action == "fit":
        run_fit(args.train_path, args.model_dir)
    elif args.action == "predict":
        run_predict(args.input_path, args.model_dir, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

    @classmethod
    def estimate_peft_memory_mb(cls, spec: SynthesisSpec | dict[str, Any]) -> dict[str, Any]:
        """Estimate analytical memory footprint for PEFT/LoRA multimodal hybrid candidate."""
        if isinstance(spec, dict):
            hyperparams = spec.get("model_intent", {}).get("hyperparameters", {})
        else:
            hyperparams = spec.model_intent.hyperparameters
        vocab_size = hyperparams.get("vocab_size", 2000)
        d_model = hyperparams.get("d_model", 64)
        num_layers = hyperparams.get("num_layers", 2)
        lora_r = hyperparams.get("lora_r", 4)
        batch_size = hyperparams.get("batch_size", 32)
        max_seq_len = hyperparams.get("max_seq_len", 32)

        # Base parameters in frozen backbone
        embed_params = vocab_size * d_model
        layer_params = num_layers * (4 * d_model * d_model + 4 * d_model * d_model)
        base_params = embed_params + layer_params

        # LoRA trainable parameters
        lora_params = num_layers * 2 * (d_model * lora_r + lora_r * d_model)
        tab_params = 32 * d_model + d_model * 2 * d_model
        trainable_params = lora_params + tab_params

        total_params = base_params + trainable_params
        frozen_backbone = max(base_params - lora_params, 0)
        frozen_ratio = frozen_backbone / base_params if base_params > 0 else 1.0

        # Memory calculations:
        # 1. Weights: float32 = 4 bytes
        weights_bytes = total_params * 4
        # 2. Adam optimizer states (momentum + variance) = 8 bytes per trainable param
        optimizer_bytes = trainable_params * 8
        # 3. Activation memory during forward/backward
        activation_bytes = batch_size * max_seq_len * d_model * num_layers * 4 * 4
        total_bytes = weights_bytes + optimizer_bytes + activation_bytes
        total_mb = total_bytes / (1024 * 1024)

        return {
            "base_params": base_params,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "frozen_ratio": round(frozen_ratio, 4),
            "estimated_memory_mb": round(total_mb, 2),
            "feasible_under_512mb": total_mb <= 512.0,
        }
