"""Scientific contracts and data preparation, without provider or worker loops."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import pandas as pd
from .development_stage import stage_development, verify_stage, seal_derived, sha256
from .benchmark_contracts import map_locked_index_folds, map_k4_manifest_folds, map_locked_temporal_folds


class TaskAdapter:
    def configure_topology(self, frame, folds, topology, target_column, caller_pinned_folds):
        if isinstance(self, PanelAdapter):
            temporal_col = topology.get("temporal_col") or topology.get("time_column") or topology.get("time")
            horizon = topology.get("forecast_horizon_dates") or topology.get("forecast_horizon")
            allowlist = topology.get("known_at_forecast_columns")
            if not temporal_col or temporal_col not in frame.columns:
                raise ValueError("Panel forecasting requires an authoritative temporal column")
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
                raise ValueError("Panel forecasting requires a positive forecast horizon in dates")
            if not isinstance(allowlist, list) or not allowlist or any(not isinstance(c, str) for c in allowlist):
                raise ValueError("Panel forecasting requires an explicit known_at_forecast_columns allowlist")
            entity_columns = topology.get("entity_columns", [])
            if not isinstance(entity_columns, list) or any(c not in frame.columns for c in entity_columns):
                raise ValueError("Panel entity_columns must name existing development columns")
            # Critic recognizes this canonical temporal split contract; the exact
            # caller-provided assignments remain bound by folds_sha256.
            topology["cv_scheme"] = "purged_time_series_split"
        topology.setdefault("cv_scheme", "caller_locked" if caller_pinned_folds else "stratified_kfold" if isinstance(self, ClassificationAdapter) and not isinstance(self, RegressionAdapter) else "kfold")
        topology.setdefault("column_classifications", {c: "NUMERIC_CONTINUOUS" if pd.api.types.is_numeric_dtype(frame[c]) else "CATEGORICAL" for c in frame if c != target_column})
        temporal = topology.get("temporal_col")
        group = topology.get("entity_group_col")
        if temporal:
            if temporal not in frame: raise ValueError("Authoritative temporal column missing from development data")
            clock = frame[temporal]
            if not pd.api.types.is_numeric_dtype(clock): clock = pd.to_datetime(clock, errors="raise")
            if clock.isna().any(): raise ValueError("Missing authoritative time values")
            embargo = topology.get("split", {}).get("embargo_seconds", 0)
            for train, valid in folds:
                latest, earliest = clock.iloc[train].max(), clock.iloc[valid].min()
                gap = float((earliest - latest).total_seconds()) if isinstance(latest, pd.Timestamp) else float(earliest - latest)
                if gap <= 0 or gap < float(embargo): raise ValueError("Pinned folds violate temporal order/embargo")
        if group:
            if group not in frame: raise ValueError("Authoritative entity group column missing; explicit group mapping required")
            for train, valid in folds:
                if set(frame.iloc[train][group].dropna()) & set(frame.iloc[valid][group].dropna()):
                    raise ValueError("Pinned folds violate group separation")
        self.validate(frame, folds, topology, target_column)
        return topology

    def initial_spec(self, metric):
        spec = {"families": ["hist_gbdt", "penalized_linear", "lgbm"], "weights": [1, 1, 1], "transforms": [], "seed": 42}
        if metric == 'rmse_log_price': spec['target_transform'] = 'log'
        elif metric == 'rmsle': spec['target_transform'] = 'log1p'
        return spec

    def validate(self, frame, folds, topology, target):
        if target not in frame or frame[target].isna().any():
            raise ValueError("Missing development targets")

    def evaluate(self, request, frame):
        from .candidate_pipeline import evaluate_candidate
        return evaluate_candidate(request['spec'], frame, request['target'], request['task_type'],
                                  request['folds'], request['metric'], request.get('topology'))

    def refit(self, request, frame):
        from .candidate_pipeline import CandidatePipeline
        return CandidatePipeline(request['spec'], request['target'], request['task_type']).fit(frame)


class ClassificationAdapter(TaskAdapter):
    """Classification candidate contract."""


class RegressionAdapter(TaskAdapter):
    """Regression metrics/transform semantics are implemented by CandidatePipeline."""


class PanelAdapter(RegressionAdapter):
    def validate(self, frame, folds, topology, target):
        super().validate(frame, folds, topology, target)
        from .panel_contract import validate_panel_fold
        if len(topology.get('panel_protocol', {}).get('origins', [])) != len(folds):
            raise ValueError("Incomplete panel_protocol: exact origin count must match frozen folds")
        for i, (train, valid) in enumerate(folds):
            validate_panel_fold(frame, train, valid, topology, i, target)


def task_adapter(task_type):
    if task_type in ('binary', 'multiclass', 'binary_classification', 'multiclass_classification'):
        return ClassificationAdapter()
    if task_type == 'regression': return RegressionAdapter()
    if task_type in ('panel_forecasting', 'forecasting'): return PanelAdapter()
    raise ValueError('Unsupported scientific task')


FOLDERS = {'K1': 'k1_titanic', 'K2': 'k2_house_prices', 'K3': 'k3_forest_cover',
           'K4': 'k4_ieee_fraud', 'K5': 'k5_store_sales'}


class BenchmarkAdapter:
    def __init__(self, policy, task_id, root):
        self.problem = copy.deepcopy(next(p for p in policy['problems'] if p['id'] == task_id))
        self.task_id, self.root = task_id, Path(root)
        self.science = task_adapter(self.problem['task'])

    def validate_contract(self):
        p = self.problem
        if p.get('execution_contract', {}).get('status', '').startswith('blocked'):
            raise ValueError(json.dumps(p['execution_contract'], sort_keys=True))
        if p.get('final_quality_criteria', {}).get('threshold') is None:
            raise ValueError('Independent absolute quality requirement missing; threshold remains null')
        if p.get('final_quality_criteria', {}).get('status') != 'locked':
            raise ValueError('Independent quality contract is not locked')
        if not p.get('authenticity', {}).get('split_manifest_sha256'):
            raise ValueError('Frozen split identity missing')
        return self.contract()

    def prepare(self, workspace):
        """Import exact frozen development order, then seal any trusted derivation."""
        self.validate_contract()
        workspace = Path(workspace)
        folder = self.root / 'data/official' / FOLDERS[self.task_id]
        filename = 'train_transaction.csv' if self.task_id == 'K4' else 'train.csv'
        staged = workspace / 'staged_development.csv'
        stage_development(folder / filename, folder / 'split_manifest.json', staged,
                          self.problem['authenticity']['files_sha256'][filename],
                          self.problem['authenticity']['split_manifest_sha256'])
        manifest = json.loads((folder / 'split_manifest.json').read_text())
        frame = pd.read_csv(staged)
        topology, statement = self.contract()
        if self.task_id == 'K4':
            # No speculative subset or optional unverified join. This path can be
            # reached only after authoritative provenance replaces the blocker.
            identity_path = folder / 'train_identity.csv'
            if sha256(identity_path) != self.problem['authenticity']['files_sha256']['train_identity.csv']:
                raise ValueError('K4 identity source hash mismatch')
            identity_frame = pd.read_csv(identity_path)
            frame = frame.merge(identity_frame, on='TransactionID', how='left', validate='one_to_one', sort=False)
            folds = map_k4_manifest_folds(manifest, frame, 0, len(frame) - 1)
        elif self.task_id == 'K5':
            frame = frame[['date', 'store_nbr', 'family', 'sales']].copy()
            calendar = pd.to_datetime(frame['date'])
            frame['day_of_week'] = calendar.dt.dayofweek
            frame['day_of_month'] = calendar.dt.day
            frame['month'] = calendar.dt.month
            folds = map_locked_temporal_folds(manifest.get('development_folds') or manifest.get('development_origins', []), frame, 'date')
        else:
            folds = map_locked_index_folds(manifest.get('development_folds', []), manifest['development_indices'])
        output = staged
        if self.task_id in ('K4', 'K5'):
            output = workspace / 'development.csv'
            frame.to_csv(output, index=False)
            seal_derived(output, verify_stage(staged))
        self.science.validate(frame, folds, topology, self.problem['target'])
        return output, frame, folds, topology, statement

    def contract(self):
        task_id = self.task_id
        problem = self.problem
        if problem is None:
            raise ValueError(f"Benchmark contract for {task_id} is missing")
        split = problem["split"]
        from .candidate_pipeline import identity
        topology = {
            "benchmark_contract_sha256": identity(problem),
            "benchmark_id": task_id,
            "target_column": problem["target"],
            "task_type": problem["task"],
            "benchmark_task_type": problem["task"],
            "metric": problem["metric"],
            "direction": problem["direction"],
            "split": split,
        }
        if task_id == "K1":
            topology["cv_scheme"] = "group_kfold"
            if split.get("group_column"):
                topology["entity_group_col"] = split["group_column"]
        elif task_id == "K2":
            topology["cv_scheme"] = "kfold"
        elif task_id == "K3":
            topology["cv_scheme"] = "stratified_kfold"
        elif task_id == "K4":
            topology.update(temporal_col=split.get("time"), cv_scheme="purged_time_series_split",
                            entity_join_key=split.get("join_key"))
        elif task_id == "K5":
            topology.update(temporal_col=split.get("time"), cv_scheme="purged_time_series_split",
                            entity_columns=split.get("entities"), forecast_horizon_dates=split.get("development_horizon_dates"))
        if task_id == "K5":
            protocol_path = self.root / problem["execution_contract"]["protocol_path"]
            if sha256(protocol_path) != problem["execution_contract"]["protocol_sha256"]:
                raise ValueError("Frozen K5 panel protocol hash mismatch")
            frozen = json.loads(protocol_path.read_text())
            if frozen["panel_protocol"]["training_calendar_provenance"]["split_sha256"] != problem["authenticity"]["split_manifest_sha256"]:
                raise ValueError("K5 protocol belongs to a different frozen split")
            topology.update({k: frozen[k] for k in ("temporal_col", "entity_columns", "forecast_horizon_dates", "known_at_forecast_columns", "panel_protocol")})
        statement = (f"{problem['name']}; target={problem['target']}; metric={problem['metric']} "
                     f"({problem['direction']}); pinned split={json.dumps(split, sort_keys=True)}")
        return topology, statement



def benchmark_adapter(policy, task_id, root):
    if task_id not in FOLDERS: raise ValueError('Unknown benchmark task')
    return BenchmarkAdapter(policy, task_id, root)
