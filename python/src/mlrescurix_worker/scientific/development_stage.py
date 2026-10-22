"""Trusted import boundary. Source bytes are read only here, never in a worker.

Hashing/importing necessarily reads source bytes (including sealed rows); this is
not a claim of zero physical access. Only selected development records are emitted.
Receipts are integrity/lineage commitments, not authorization from an untrusted agent.
"""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def stage_development(source, split_path, destination, expected_source_sha256,
                      expected_split_sha256, indices=None):
    source, split_path, destination = map(Path, (source, split_path, destination))
    if sha256(source) != expected_source_sha256 or sha256(split_path) != expected_split_sha256:
        raise ValueError('Source/split hash differs from frozen commitment')
    split = json.loads(split_path.read_text())
    development = split.get('development_indices')
    if development is None:
        development = range(split['development_index_start'], split['development_index_end'] + 1)
    selected = development if indices is None else indices
    if not selected or any(type(i) is not int or i < 0 for i in selected):
        raise ValueError('Invalid development row identities')
    if len(set(selected)) != len(selected) or not set(selected).issubset(development):
        raise ValueError('Selection is duplicated or outside frozen development partition')
    if set(selected).intersection(split.get('holdout_indices', [])):
        raise ValueError('Development and final identities overlap')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        output = Path(scratch) / 'development.csv'
        wanted = set(selected)
        ordered = all(a < b for a, b in zip(selected, selected[1:]))
        retained = {}
        count = 0
        with source.open(newline='') as src, output.open('w', newline='') as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst, lineterminator='\n')
            writer.writerow(next(reader))
            for i, row in enumerate(reader):
                if i in wanted:
                    count += 1
                    if ordered:
                        writer.writerow(row)
                    else:
                        retained[i] = row
            if not ordered:
                writer.writerows(retained[i] for i in selected)
            dst.flush()
            os.fsync(dst.fileno())
        if count != len(selected) or sha256(source) != expected_source_sha256 or sha256(split_path) != expected_split_sha256:
            raise ValueError('Missing selected records or source changed during staging')
        receipt = {'version': 1, 'scope': 'development_only', 'source_sha256': expected_source_sha256,
                   'split_sha256': expected_split_sha256, 'csv_sha256': sha256(output),
                   'selection_sha256': hashlib.sha256(json.dumps(list(selected), separators=(',', ':')).encode()).hexdigest(),
                   'rows': count, 'trusted_import_reads_source_bytes': True,
                   'worker_final_rows': 0, 'task_id': split.get('task_id'),
                   'history': 'K1 final contamination is preserved; staging does not restore a clean final set' if split.get('task_id') == 'K1' else None}
        output.replace(destination)
        receipt_path = destination.with_suffix('.stage.json')
        temporary = Path(scratch) / 'receipt.json'
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)
    return destination


def verify_stage(path):
    path = Path(path)
    receipt = json.loads(path.with_suffix('.stage.json').read_text())
    if receipt.get('scope') != 'development_only' or receipt.get('csv_sha256') != sha256(path):
        raise ValueError('Development staging receipt missing or stale')
    return receipt


def seal_derived(path, parent_receipt):
    """Called only by trusted preprocessing, never from an agent tool."""
    receipt = dict(parent_receipt, csv_sha256=sha256(path), derived=True)
    Path(path).with_suffix('.stage.json').write_text(json.dumps(receipt, indent=2) + '\n')
