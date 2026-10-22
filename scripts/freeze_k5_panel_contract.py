#!/usr/bin/env python3
"""Freeze panel schema/calendar metadata from pinned sources; never score labels."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def freeze(source_dir, problem):
    split_path = source_dir / 'split_manifest.json'
    source = source_dir / 'train.csv'
    query = source_dir / 'test.csv'
    auth = problem['authenticity']
    for path, expected in [(source, auth['files_sha256']['train.csv']), (query, auth['files_sha256']['test.csv']),
                           (split_path, auth['split_manifest_sha256'])]:
        if digest(path) != expected: raise ValueError('Pinned K5 provenance mismatch')
    split = json.loads(split_path.read_text())
    with query.open(newline='') as stream:
        entities = sorted({(int(r['store_nbr']), r['family']) for r in csv.DictReader(stream)})
    # Trusted metadata import reads source bytes, including target bytes. Only
    # development dates and per-date/entity coverage commitments leave this boundary.
    dates = {}
    allowed = set(entities)
    with source.open(newline='') as stream:
        for position, row in enumerate(csv.DictReader(stream)):
            if position > split['development_index_end']: break
            if position < split['development_index_start']: continue
            entity = (int(row['store_nbr']), row['family'])
            date = row['date']
            if entity not in allowed or entity in dates.setdefault(date, set()):
                raise ValueError('Duplicate or unknown K5 development identity')
            dates[date].add(entity)
    if not dates or any(v != allowed for v in dates.values()):
        raise ValueError('Incomplete authoritative K5 per-entity date coverage')
    for path, expected in [(source, auth['files_sha256']['train.csv']), (query, auth['files_sha256']['test.csv']),
                           (split_path, auth['split_manifest_sha256'])]:
        if digest(path) != expected: raise ValueError('K5 provenance changed during metadata import')
    calendar = sorted(dates)
    origins = []
    for origin in split['development_origins']:
        history = [d for d in calendar if d <= origin['train_end_date']]
        horizon = [d for d in calendar if origin['val_start_date'] <= d <= origin['val_end_date']]
        if len(history) * len(entities) != origin['train_rows'] or len(horizon) * len(entities) != origin['val_rows']:
            raise ValueError('Frozen K5 row counts incompatible with source calendar')
        if origin['val_index_start'] != len(history) * len(entities) or origin['val_index_end'] != (len(history) + len(horizon)) * len(entities) - 1:
            raise ValueError('Frozen K5 index bounds incompatible with calendar')
        origins.append({'train_start': history[0], 'train_end': history[-1], 'training_dates': history,
                        'horizon_dates': horizon})
    calendar_features = ['day_of_week', 'day_of_month', 'month']
    protocol = {'required_entities': [list(e) for e in entities], 'origins': origins, 'embargo_days': 0,
                'embargo_justification': 'Predict the declared next 15 dates after observations through the preceding date; deterministic calendar covariates have no future observation dependency. No contemporaneous external covariates or future target-derived features are admitted.',
                'missing_entity_behavior': 'reject', 'unseen_entity_behavior': 'reject',
                'covariate_availability': {c: {'kind': 'deterministic_calendar', 'provenance': 'Gregorian date arithmetic; known at every origin'} for c in calendar_features},
                'training_calendar_provenance': {'source_sha256': auth['files_sha256']['train.csv'], 'split_sha256': auth['split_manifest_sha256'],
                    'selection': 'exact frozen development row interval; all observed dates and every required entity; no invented missing-date labels'},
                'entity_universe_provenance': {'source_sha256': auth['files_sha256']['test.csv'], 'selection': 'unique store_nbr/family keys in official unlabeled query'},
                'trusted_metadata_import_reads_source_bytes': True}
    return {'format': 'mlrescurix.panel_protocol.v1', 'task_id': 'K5', 'temporal_col': 'date',
            'entity_columns': ['store_nbr', 'family'], 'forecast_horizon_dates': 15,
            'known_at_forecast_columns': calendar_features, 'panel_protocol': protocol}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/official/k5_store_sales')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    problem = next(p for p in json.loads((ROOT / 'docs/v2/BENCHMARK_SUITE.json').read_text())['problems'] if p['id'] == 'K5')
    args.output.write_text(json.dumps(freeze(args.source_dir, problem), sort_keys=True, separators=(',', ':')) + '\n')
