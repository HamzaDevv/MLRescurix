#!/usr/bin/env python3
"""Read-only historical accounting audit. Never infers verified bills from token rates."""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def reconcile(campaign):
    ledger_path = campaign / 'campaign_budget_ledger.json'
    ledger_bytes = ledger_path.read_bytes()
    ledger = json.loads(ledger_bytes)
    entries = []
    for run_id, run in ledger['runs'].items():
        state = campaign / run_id / 'autonomous_loop_state.json'
        receipts = []
        if state.exists():
            for turn in json.loads(state.read_text()).get('turns', []):
                r = turn.get('provider_receipt', {})
                receipts.append({'response_id': r.get('provider_response_id'),
                                 'recorded_usd': str(r.get('cost_usd', turn.get('provider_cost_usd', 'unknown'))),
                                 'basis': 'calculated_estimate' if r.get('gateway_route') == 'rust_mlrescurix_harness' else 'unverified_turn_accounting',
                                 'verified_provider_charge_usd': None})
        receipt_sum = sum((Decimal(r['recorded_usd']) for r in receipts if r['recorded_usd'] != 'unknown'), Decimal(0))
        entries.append({'run_id': run_id, 'committed_recorded_usd': run['billed_usd'],
                        'receipt_sum_usd': str(receipt_sum) if receipts and all(r['recorded_usd'] != 'unknown' for r in receipts) else None,
                        'receipt_sum_matches_ledger': receipt_sum == Decimal(run['billed_usd']) if receipts and all(r['recorded_usd'] != 'unknown' for r in receipts) else None,
                        'receipts': receipts, 'verified_provider_charge_usd': None})
    total = sum((Decimal(r['billed_usd']) for r in ledger['runs'].values()), Decimal(0))
    return {'ledger_sha256': hashlib.sha256(ledger_bytes).hexdigest(),
            'recorded_committed_usd': str(sum((Decimal(ledger[k]) for k in ('total_billed_usd', 'total_reserved_usd', 'total_uncertain_usd')), Decimal(0))),
            'legacy_recorded_billed_usd': ledger['total_billed_usd'],
            'reserved_usd': ledger['total_reserved_usd'], 'uncertain_usd': ledger['total_uncertain_usd'],
            'sum_runs_matches_ledger': total == Decimal(ledger['total_billed_usd']),
            'remaining_recorded_usd': ledger['remaining_campaign_usd'],
            'verified_provider_charge_usd': None, 'additional_spend_usd': '0',
            'status': 'provider_reconciliation_blocked_no_invoice_or_provider_charge_receipts',
            'note': 'False historical cost_is_estimate flags do not verify a provider charge. All committed funds remain committed.',
            'runs': entries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(reconcile(args.campaign), indent=2) + '\n')
