"""Trusted, bounded full-pipeline jobs. Called only by the Rust container scheduler.

OOF values are generated here from fold fits, never accepted as caller arrays.
All component models and fold evidence are retained with content hashes.
"""
from __future__ import annotations
import json
import os
import time
import tempfile
import shutil
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression, Ridge
from .features import SplitAwareFeaturePipeline, FeatureTransformSpec
from .calibration import SplitAwareCalibrator
from .workflow import (_pipeline, _predict, _score, _read_csv, _normal_spec,
                       _atomic_json, _sha256, WorkflowError)
from mlrescurix_worker.search.adapter import normalize_parameters


class FeatureModel:
    def __init__(self, transformer, model):
        self.transformer = transformer
        self.model = model
        if hasattr(model, 'classes_'):
            self.classes_ = model.classes_

    def predict(self, frame):
        return self.model.predict(self.transformer.transform(frame) if self.transformer else frame)

    def predict_proba(self, frame):
        return self.model.predict_proba(self.transformer.transform(frame) if self.transformer else frame)


class DeliveredPipeline:
    def __init__(self, models, task, ensemble, weights, meta=None, calibrator=None):
        self.models = models
        self.task = task
        self.ensemble = ensemble
        self.weights = weights
        self.meta = meta
        self.calibrator = calibrator
        if task != 'regression':
            self.classes_ = models[0].classes_

    def _base(self, frame):
        outputs = [m.predict(frame) if self.task == 'regression' else m.predict_proba(frame)
                   for m in self.models]
        if self.meta is not None:
            inputs = meta_features(outputs, self.task)
            return self.meta.predict(inputs) if self.task == 'regression' else self.meta.predict_proba(inputs)
        return sum(w * np.asarray(p) for w, p in zip(self.weights, outputs))

    def predict(self, frame):
        if self.task == 'regression':
            return self._base(frame)
        return self.classes_[np.argmax(self.predict_proba(frame), axis=1)]

    def predict_proba(self, frame):
        probabilities = self._base(frame)
        return self.calibrator.transform(probabilities) if self.calibrator else probabilities


def meta_features(outputs, task):
    return np.column_stack([np.asarray(p).reshape(-1, 1) if task == 'regression'
                            else p[:, 1:2] if p.shape[1] == 2 else p for p in outputs])


def run_pipeline(output_dir, protocol):
    # All fits, parent models and OOF bytes first live in the scheduler's bounded
    # 64 MiB container tmpfs. ENOSPC fails before any managed model publication.
    with tempfile.TemporaryDirectory(prefix='mlr-pipeline-') as scratch:
        receipt = _execute_pipeline(output_dir, protocol, Path(scratch))
        final = Path(output_dir) / 'trials' / protocol['experiment_id']
        final.parent.mkdir(parents=True, exist_ok=True)
        stage = final.with_name('.' + final.name + '-staging')
        if stage.exists() or final.exists():
            raise WorkflowError('pipeline publication path already exists; reconcile before retry')
        try:
            shutil.copytree(scratch, stage)
            for path in stage.iterdir():
                with path.open('rb') as file:
                    os.fsync(file.fileno())
            os.rename(stage, final)
        except BaseException:
            if stage.exists():
                shutil.rmtree(stage)
            raise
        return receipt


def _execute_pipeline(output_dir, protocol, directory):
    root = Path(output_dir)
    if not os.environ.get('MLRESCURIX_ENVIRONMENT_IDENTITY') or not os.environ.get('MLRESCURIX_SOURCE_SHA256'):
        raise WorkflowError('pipeline must execute through a registered Rust container job')
    if set(protocol) - {'format','experiment_id','parents','features','ensemble','weights','calibration','seed','folds'}:
        raise WorkflowError('unknown pipeline protocol field')
    if protocol.get('format') != 'mlrescurix.pipeline.v1':
        raise WorkflowError('unsupported pipeline protocol')
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    spec = _normal_spec(manifest['spec'])
    frame, digest = _read_csv(root / 'dataset.csv')
    if digest != manifest['sha256']:
        raise WorkflowError('dataset snapshot changed')
    candidate = protocol['experiment_id']
    if not candidate or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-' for c in candidate):
        raise WorkflowError('unsafe candidate id')
    if (root / 'trials' / candidate).exists():
        raise WorkflowError('pipeline identity already has execution bytes; inspect/reconcile, never replay blindly')
    parents = protocol.get('parents', [])
    if not 1 <= len(parents) <= 3:
        raise WorkflowError('pipeline requires one to three declared parent recipes')
    ensemble = protocol.get('ensemble', 'voting')
    if ensemble not in {'voting','stacking'}:
        raise WorkflowError('unsupported ensemble')
    calibration = protocol.get('calibration')
    if calibration not in {None,'isotonic','sigmoid'} or (calibration and spec.task == 'regression'):
        raise WorkflowError('unsupported calibration')
    # Calibration of stacking needs a second nested cross-fitting level. Fail closed.
    if calibration and ensemble == 'stacking':
        raise WorkflowError('stacking plus calibration requires nested meta OOF; unsupported')
    folds = protocol.get('folds', 3)
    seed = protocol.get('seed', 2026)
    if type(folds) is not int or not 2 <= folds <= 5 or type(seed) is not int:
        raise WorkflowError('invalid bounded fold/seed protocol')
    weights = np.asarray(protocol.get('weights', [1.] * len(parents)), dtype=float)
    if weights.shape != (len(parents),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise WorkflowError('invalid component weights')
    weights /= weights.sum()
    feature_spec = FeatureTransformSpec(**protocol['features']) if protocol.get('features') else None
    if feature_spec and (feature_spec.discretize_bins not in {0,2,3,4,5,6,7,8,9,10} or not 0 <= feature_spec.max_interaction_features <= 10):
        raise WorkflowError('feature expansion outside bounds')
    features = manifest['features']
    train_ids = manifest['splits']['train']
    val_ids = manifest['splits']['validation']
    x = frame.iloc[train_ids][features]
    y = frame.iloc[train_ids][spec.target].to_numpy()
    classes = np.unique(y) if spec.task != 'regression' else None
    feature_y = y if spec.task == 'regression' else np.searchsorted(classes, y)
    if manifest['split_policy'] == 'grouped':
        splitter = GroupKFold(n_splits=folds)
        splits = list(splitter.split(x, y, frame.iloc[train_ids][spec.group_column]))
    elif manifest['split_policy'] == 'chronological':
        splits = list(TimeSeriesSplit(n_splits=folds).split(x))
    elif spec.task != 'regression':
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x,y))
    else:
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(x))
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    provenance = []
    artifacts = {}
    oof_outputs = []
    models = []
    fit_count = 0
    def fit(parent, positions):
        nonlocal fit_count
        if set(parent) - {'recipe','parameters'}:
            raise WorkflowError('parent protocols accept recipes, never asserted artifact hashes or OOF arrays')
        recipe = parent['recipe']
        parameters = normalize_parameters(recipe, spec.task, parent.get('parameters',{}))
        transformer = SplitAwareFeaturePipeline(feature_spec, spec.task, seed) if feature_spec else None
        transformed = transformer.fit_transform_training(x.iloc[positions],feature_y[positions]) if transformer else x.iloc[positions]
        model = _pipeline(transformed, list(transformed.columns), spec.task, recipe, seed, parameters)
        model.fit(transformed, y[positions])
        if classes is not None and list(model.classes_) != list(classes):
            raise WorkflowError('each OOF fit requires all training classes')
        fit_count += 1
        return FeatureModel(transformer, model)
    for parent_index, parent in enumerate(parents):
        shape = (len(x),) if classes is None else (len(x),len(classes))
        oof = np.full(shape,np.nan)
        for fold,(fit_positions,held_positions) in enumerate(splits):
            model = fit(parent,fit_positions)
            filename = f'parent-{parent_index}-fold-{fold}.joblib'
            path = directory / filename
            joblib.dump(model,path)
            artifacts[filename] = _sha256(path)
            prediction = model.predict(x.iloc[held_positions]) if classes is None else model.predict_proba(x.iloc[held_positions])
            oof[held_positions] = prediction
            provenance.append({'parent':parent_index,'fold':fold,'train_row_ids':[train_ids[i] for i in fit_positions],
                               'held_row_ids':[train_ids[i] for i in held_positions], 'model_file':filename,
                               'model_sha256':artifacts[filename]})
        model = fit(parent,np.arange(len(x)))
        filename = f'parent-{parent_index}-full.joblib'
        joblib.dump(model,directory / filename)
        artifacts[filename] = _sha256(directory / filename)
        models.append(model)
        oof_outputs.append(oof)
    covered = np.isfinite(oof_outputs[0]) if classes is None else np.isfinite(oof_outputs[0]).all(axis=1)
    if not covered.any():
        raise WorkflowError('no disjoint OOF rows')
    meta = None
    if ensemble == 'stacking':
        meta = Ridge(alpha=1.) if classes is None else LogisticRegression(max_iter=200,random_state=seed)
        meta.fit(meta_features([p[covered] for p in oof_outputs],spec.task),y[covered])
        fit_count += 1
    calibrator = None
    if calibration:
        calibrator = SplitAwareCalibrator(calibration)
        probabilities = sum(w * p[covered] for w,p in zip(weights,oof_outputs))
        calibrator.fit_from_oof(probabilities,y[covered],classes.tolist())
        fit_count += len(calibrator.calibrators_)
    delivered = DeliveredPipeline(models,spec.task,ensemble,weights,meta,calibrator)
    model_path = directory / 'candidate.joblib'
    joblib.dump(delivered,model_path)
    train_ms = (time.perf_counter_ns() - started) / 1e6
    load_start = time.perf_counter_ns()
    reloaded = joblib.load(model_path)
    cold_load_ms = (time.perf_counter_ns()-load_start)/1e6
    validation = frame.iloc[val_ids][features]
    prediction = _predict(reloaded,validation,features,spec.task)
    scores = _score(frame.iloc[val_ids][spec.target].to_numpy(),prediction,spec.task)
    pred_path = directory / 'predictions.json'
    _atomic_json(pred_path,{'row_ids':val_ids,**prediction})
    oof_path = directory / 'oof.json'
    _atomic_json(oof_path,{'dataset_sha256':digest,'split_sha256':_sha256(manifest_path),'folds':provenance,
                         'covered_row_ids':[train_ids[i] for i in np.flatnonzero(covered)],
                         'excluded_warmup_row_ids':[train_ids[i] for i in np.flatnonzero(~covered)],
                         'outputs':[p[covered].tolist() for p in oof_outputs],'artifacts':artifacts})
    sample = validation.iloc[:spec.inference_batch_size]
    timings = []
    for _ in range(spec.inference_repetitions):
        t = time.perf_counter_ns()
        _predict(reloaded,sample,features,spec.task)
        timings.append((time.perf_counter_ns()-t)/1e6)
    receipt = {'format':'mlrescurix.pipeline-receipt.v1','experiment_id':candidate,'status':'evaluated_development',
               'protocol':protocol,'dataset_sha256':digest,'split_sha256':_sha256(manifest_path),
               'source_sha256':os.environ['MLRESCURIX_SOURCE_SHA256'],
               'environment_identity':os.environ['MLRESCURIX_ENVIRONMENT_IDENTITY'],
               'model_sha256':_sha256(model_path),'predictions_sha256':_sha256(pred_path),'oof_sha256':_sha256(oof_path),
               'metrics':scores,'fit_count':fit_count,'train_ms':train_ms,'cold_load_ms':cold_load_ms,
               'warm_p95_ms':float(np.percentile(timings,95)),'delivered_model_bytes':model_path.stat().st_size,
               'retained_artifact_bytes':sum(p.stat().st_size for p in directory.iterdir()),
               'feasibility':'development measurements only; export bytes and fresh target memory certification pending'}
    _atomic_json(directory / 'trial.json',receipt)
    return receipt


def refit_pipeline_for_finalization(root: Path, snapshot, manifest: dict, spec, candidate: dict, frozen_path: Path) -> dict:
    """Refit a verified composite pipeline protocol on train + validation without test leakage."""
    protocol = candidate.get('protocol')
    if not isinstance(protocol, dict) or protocol.get('format') != 'mlrescurix.pipeline.v1':
        raise WorkflowError('invalid composite pipeline protocol for finalization')
    if set(protocol) - {'format','experiment_id','parents','features','ensemble','weights','calibration','seed','folds'}:
        raise WorkflowError('unknown pipeline protocol field')
    parents = protocol.get('parents', [])
    if not 1 <= len(parents) <= 3:
        raise WorkflowError('pipeline requires one to three declared parent recipes')
    ensemble = protocol.get('ensemble', 'voting')
    if ensemble not in {'voting', 'stacking'}:
        raise WorkflowError('unsupported ensemble')
    calibration = protocol.get('calibration')
    if calibration not in {None, 'isotonic', 'sigmoid'} or (calibration and spec.task == 'regression'):
        raise WorkflowError('unsupported calibration')
    if calibration and ensemble == 'stacking':
        raise WorkflowError('stacking plus calibration requires nested meta OOF; unsupported')
    folds = protocol.get('folds', 3)
    seed = protocol.get('seed', 2026)
    if type(folds) is not int or not 2 <= folds <= 5 or type(seed) is not int:
        raise WorkflowError('invalid bounded fold/seed protocol')
    weights = np.asarray(protocol.get('weights', [1.] * len(parents)), dtype=float)
    if weights.shape != (len(parents),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise WorkflowError('invalid component weights')
    weights /= weights.sum()
    feature_spec = FeatureTransformSpec(**protocol['features']) if protocol.get('features') else None
    if feature_spec and (feature_spec.discretize_bins not in {0,2,3,4,5,6,7,8,9,10} or not 0 <= feature_spec.max_interaction_features <= 10):
        raise WorkflowError('feature expansion outside bounds')

    features = manifest['features']
    train_ids = list(manifest['splits']['train'])
    val_ids = list(manifest['splits']['validation'])
    test_ids = set(manifest['splits']['test'])
    trainval_ids = sorted(train_ids + val_ids) if manifest['split_policy'] == 'chronological' else (train_ids + val_ids)
    if set(trainval_ids) & test_ids:
        raise WorkflowError('train+validation refit partition overlaps sealed test partition')

    x = snapshot.iloc[trainval_ids][features]
    y = snapshot.iloc[trainval_ids][spec.target].to_numpy()
    classes = np.unique(y) if spec.task != 'regression' else None
    feature_y = y if spec.task == 'regression' else np.searchsorted(classes, y)

    if manifest['split_policy'] == 'grouped':
        splitter = GroupKFold(n_splits=folds)
        splits = list(splitter.split(x, y, snapshot.iloc[trainval_ids][spec.group_column]))
    elif manifest['split_policy'] == 'chronological':
        splits = list(TimeSeriesSplit(n_splits=folds).split(x))
    elif spec.task != 'regression':
        splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y))
    else:
        splits = list(KFold(n_splits=folds, shuffle=True, random_state=seed).split(x))

    started = time.perf_counter_ns()
    provenance = []
    oof_outputs = []
    models = []
    fit_count = 0

    def fit_parent(parent, positions):
        nonlocal fit_count
        if set(parent) - {'recipe', 'parameters'}:
            raise WorkflowError('parent protocols accept recipes, never asserted artifact hashes or OOF arrays')
        recipe = parent['recipe']
        parameters = normalize_parameters(recipe, spec.task, parent.get('parameters', {}))
        transformer = SplitAwareFeaturePipeline(feature_spec, spec.task, seed) if feature_spec else None
        transformed = transformer.fit_transform_training(x.iloc[positions], feature_y[positions]) if transformer else x.iloc[positions]
        model = _pipeline(transformed, list(transformed.columns), spec.task, recipe, seed, parameters)
        model.fit(transformed, y[positions])
        if classes is not None and list(model.classes_) != list(classes):
            raise WorkflowError('each OOF fit requires all training classes')
        fit_count += 1
        return FeatureModel(transformer, model)

    for parent_index, parent in enumerate(parents):
        shape = (len(x),) if classes is None else (len(x), len(classes))
        oof = np.full(shape, np.nan)
        for fold, (fit_positions, held_positions) in enumerate(splits):
            fold_model = fit_parent(parent, fit_positions)
            prediction = fold_model.predict(x.iloc[held_positions]) if classes is None else fold_model.predict_proba(x.iloc[held_positions])
            oof[held_positions] = prediction
            provenance.append({
                'parent': parent_index,
                'fold': fold,
                'train_row_ids': [trainval_ids[i] for i in fit_positions],
                'held_row_ids': [trainval_ids[i] for i in held_positions],
            })
        full_model = fit_parent(parent, np.arange(len(x)))
        models.append(full_model)
        oof_outputs.append(oof)

    covered = np.isfinite(oof_outputs[0]) if classes is None else np.isfinite(oof_outputs[0]).all(axis=1)
    if not covered.any():
        raise WorkflowError('no disjoint OOF rows in finalization refit')
    meta = None
    if ensemble == 'stacking':
        meta = Ridge(alpha=1.) if classes is None else LogisticRegression(max_iter=200, random_state=seed)
        meta.fit(meta_features([p[covered] for p in oof_outputs], spec.task), y[covered])
        fit_count += 1
    calibrator = None
    if calibration:
        calibrator = SplitAwareCalibrator(calibration)
        probabilities = sum(w * p[covered] for w, p in zip(weights, oof_outputs))
        calibrator.fit_from_oof(probabilities, y[covered], classes.tolist())
        fit_count += len(calibrator.calibrators_)

    delivered = DeliveredPipeline(models, spec.task, ensemble, weights, meta, calibrator)
    joblib.dump(delivered, frozen_path, compress=3)
    train_ms = (time.perf_counter_ns() - started) / 1e6
    frozen_sha = _sha256(frozen_path)
    lineage_record = {
        'format': 'mlrescurix.pipeline-lineage.v1',
        'candidate_id': candidate.get('experiment_id', candidate.get('candidate_id')),
        'protocol': protocol,
        'development_model_sha256': candidate.get('model_sha256'),
        'development_oof_sha256': candidate.get('oof_sha256'),
        'development_predictions_sha256': candidate.get('predictions_sha256'),
        'final_refit_trainval_row_ids': trainval_ids,
        'final_refit_covered_row_ids': [trainval_ids[i] for i in np.flatnonzero(covered)],
        'final_refit_excluded_warmup_row_ids': [trainval_ids[i] for i in np.flatnonzero(~covered)],
        'final_refit_folds': provenance,
        'final_refit_fit_count': fit_count,
        'final_refit_train_ms': train_ms,
        'test_row_intersection': 0,
        'frozen_model_sha256': frozen_sha,
    }
    _atomic_json(root / 'lineage.json', lineage_record)
    return lineage_record
