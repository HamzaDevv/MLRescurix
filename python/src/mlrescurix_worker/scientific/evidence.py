"""Durable, content-bound execution/evidence identities (not OS authentication)."""
import hashlib
import json
import os
from pathlib import Path
from .candidate_pipeline import identity


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(f'.tmp.{os.getpid()}')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def source_identity(root):
    root = Path(root)
    files = list((root / 'python/src/mlrescurix_worker').rglob('*.py'))
    if not files: raise ValueError('Worker source tree unavailable for identity binding')
    checkout = (root / 'Cargo.lock').is_file()
    if checkout:
        files += list((root / 'crates/harness/src').rglob('*.rs'))
        files += [root / p for p in ('scripts/run_robustness_campaign.py', 'Cargo.lock')]
    files.append(root / 'python/pyproject.toml')
    return identity({'scope': 'checkout' if checkout else 'installed_python_assets',
        'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}})


def seal_result(result, binding, trust):
    result = dict(result, execution_binding=binding, trust_level=trust)
    result['receipt_identity'] = identity(result)
    return result


def verify_result(result, binding):
    content = dict(result)
    receipt = content.pop('receipt_identity', None)
    if content.get('execution_binding') != binding or receipt != identity(content):
        raise ValueError('Evaluation receipt identity/binding mismatch')


def read_worker_result(path, request_identity):
    payload = json.loads(Path(path).read_text())
    if payload.get('request_identity') != request_identity:
        raise ValueError('Worker result request identity mismatch; preserve for reconciliation')
    content = payload.get('result') if 'result' in payload else {'error': payload.get('error')}
    if payload.get('payload_identity') != identity(content):
        raise ValueError('Worker result content identity mismatch')
    if 'error' in payload: raise ValueError(payload['error'])
    return payload['result']
