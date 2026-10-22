#!/usr/bin/env python3
"""Reproducibly import only frozen development rows; no training or final scoring."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python/src'))
from mlrescurix_worker.scientific.development_stage import stage_development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['K1', 'K2', 'K3', 'K4', 'K5'], required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    folders = {'K1': 'k1_titanic', 'K2': 'k2_house_prices', 'K3': 'k3_forest_cover',
               'K4': 'k4_ieee_fraud', 'K5': 'k5_store_sales'}
    problem = next(p for p in json.loads((ROOT / 'docs/v2/BENCHMARK_SUITE.json').read_text())['problems'] if p['id'] == args.task)
    folder = ROOT / 'data/official' / folders[args.task]
    filename = 'train_transaction.csv' if args.task == 'K4' else 'train.csv'
    stage_development(folder / filename, folder / 'split_manifest.json', args.destination,
                      problem['authenticity']['files_sha256'][filename], problem['authenticity']['split_manifest_sha256'])
    print(json.dumps({'task': args.task, 'status': 'staged', 'receipt': str(args.destination.with_suffix('.stage.json')),
                      'final_scored': False}))


if __name__ == '__main__': main()
