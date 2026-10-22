"""Strict daily panel contracts; unsupported cold-start/missing policies fail closed."""
import pandas as pd


def validate_panel_fold(frame, train, valid, topology, fold_number, target):
    protocol = topology.get('panel_protocol')
    required = {'required_entities', 'origins', 'embargo_days', 'missing_entity_behavior',
                'unseen_entity_behavior', 'covariate_availability'}
    if not isinstance(protocol, dict) or required - protocol.keys():
        raise ValueError('Incomplete panel_protocol: required entities, origins, embargo, entity behavior and covariate availability required')
    entities = topology.get('entity_columns')
    if not entities or frame[entities].isna().any().any():
        raise ValueError('Panel entity identities must be present and nonmissing')
    expected = protocol['required_entities']
    if not isinstance(expected, list) or not expected or any(not isinstance(e, list) or len(e) != len(entities) for e in expected):
        raise ValueError('required_entities must explicitly enumerate every entity tuple')
    expected = [tuple(e) for e in expected]
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate required entity identity')
    if protocol['missing_entity_behavior'] != 'reject' or protocol['unseen_entity_behavior'] != 'reject':
        raise ValueError('Unsupported missing/unseen entity behavior; this adapter requires reject')
    embargo = protocol['embargo_days']
    if type(embargo) is not int or embargo < 0:
        raise ValueError('Panel embargo_days must be a nonnegative integer')
    origins = protocol['origins']
    if not isinstance(origins, list) or fold_number >= len(origins):
        raise ValueError('Missing chronological panel origin')
    time_col = topology.get('temporal_col') or topology.get('time_column') or topology.get('time')
    times = pd.to_datetime(frame[time_col], errors='raise')
    previous = None
    for origin in origins:
        if not isinstance(origin, dict) or {'train_start', 'train_end', 'horizon_dates'} - origin.keys():
            raise ValueError('Origin requires exact training bounds and horizon_dates')
        dates = pd.DatetimeIndex(pd.to_datetime(origin['horizon_dates'], errors='raise'))
        if len(dates) != topology.get('forecast_horizon_dates', topology.get('forecast_horizon')) or not dates.equals(pd.date_range(dates.min(), periods=len(dates), freq='D')):
            raise ValueError('Panel horizon dates must be exact contiguous daily dates')
        start, end = pd.Timestamp(origin['train_start']), pd.Timestamp(origin['train_end'])
        if start > end or dates.min() != end + pd.Timedelta(days=embargo + 1) or (previous is not None and dates.min() <= previous):
            raise ValueError('Panel origins are not chronological or violate embargo')
        previous = dates.min()
    origin = origins[fold_number]
    horizon = pd.DatetimeIndex(pd.to_datetime(origin['horizon_dates']))
    history = pd.date_range(origin['train_start'], origin['train_end'], freq='D')
    if 'training_dates' in origin:
        provenance = protocol.get('training_calendar_provenance', {})
        if not all(isinstance(provenance.get(k), str) and len(provenance[k]) == 64 and
                   all(c in '0123456789abcdef' for c in provenance[k]) for k in ('source_sha256', 'split_sha256')):
            raise ValueError('An explicit observation calendar requires frozen source/split provenance')
        history = pd.DatetimeIndex(pd.to_datetime(origin['training_dates'], errors='raise'))
        start, end = pd.Timestamp(origin['train_start']), pd.Timestamp(origin['train_end'])
        if history.empty or not history.equals(history.sort_values().unique()) or history[0] != start or history[-1] != end:
            raise ValueError('Training observation calendar is unordered, duplicated or outside origin bounds')
        observed_calendar = pd.DatetimeIndex(sorted(pd.unique(times[(times >= start) & (times <= end)])))
        if not observed_calendar.equals(history):
            raise ValueError('Training observation calendar omits or invents development dates')
    # The complete frozen training history must be used, not just arbitrary earlier rows.
    for indices, dates, label in ((train, history, 'training'), (valid, horizon, 'validation')):
        selected = frame.iloc[indices][entities].copy()
        clock_key = '__panel_contract_time'
        while clock_key in entities: clock_key += '_'
        selected[clock_key] = times.iloc[indices].to_numpy()
        seen = set()
        expected_set = set(expected)
        for key, rows in selected.groupby(entities, sort=False, observed=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            if key not in expected_set:
                raise ValueError(f'Panel {label} missing/unseen required entity')
            seen.add(key)
            if not pd.DatetimeIndex(rows[clock_key]).sort_values().equals(dates):
                raise ValueError(f'Panel {label} incomplete per-entity date coverage')
        if seen != expected_set:
            raise ValueError(f'Panel {label} missing/unseen required entity')
    available = protocol['covariate_availability']
    known = topology['known_at_forecast_columns']
    if not isinstance(available, dict) or set(available) != set(known) or target in available:
        raise ValueError('Every covariate requires known-at-origin provenance; target cannot be a covariate')
    origin_time = pd.Timestamp(origin['train_end'])
    for column, record in available.items():
        if not isinstance(record, dict) or not record.get('provenance') or record.get('kind') not in ('deterministic_calendar', 'static', 'published'):
            raise ValueError(f'Unknown availability provenance for {column}')
        if record['kind'] != 'deterministic_calendar':
            if 'known_by' not in record or pd.Timestamp(record['known_by']) > origin_time:
                raise ValueError(f'Covariate {column} unavailable at origin')
    return {'required_entities': len(expected), 'training_dates': len(history),
            'horizon_dates': [d.isoformat() for d in horizon], 'embargo_days': embargo,
            'per_entity_coverage': 'complete', 'missing_entity_behavior': 'reject', 'unseen_entity_behavior': 'reject'}
