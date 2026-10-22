"""Offline trust-boundary regressions; no provider network access."""
import concurrent.futures
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from mlrescurix_worker.scientific.development_stage import stage_development, sha256, verify_stage
from mlrescurix_worker.scientific.panel_contract import validate_panel_fold
from mlrescurix_worker.scientific.candidate_runner import execute_candidate_cli

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('prerequisite_runner', ROOT / 'scripts/run_robustness_campaign.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def authority(tmp_path):
    return runner.CampaignBudgetAuthority(tmp_path, {'campaign_budget': {'provider_usd_max': '.30'},
                                                    'provider': {'usd_per_main_live_run_max': '.05'}})


def test_stage_hash_order_sealed_rows_and_tamper(tmp_path):
    source, split, output = [tmp_path / p for p in ('source.csv', 'split.json', 'stage/development.csv')]
    source.write_text('x,y\n"line\nbreak",0\nsealed-canary,999\ndev,1\n')
    split.write_text(json.dumps({'task_id': 'K1', 'development_indices': [2, 0], 'holdout_indices': [1]}))
    source_hash, split_hash = sha256(source), sha256(split)
    stage_development(source, split, output, source_hash, split_hash)
    assert 'sealed-canary' not in output.read_text()
    assert pd.read_csv(output).y.tolist() == [1, 0]
    original = output.read_bytes()
    receipt = verify_stage(output)
    assert receipt['trusted_import_reads_source_bytes'] and receipt['worker_final_rows'] == 0
    assert 'contamination' in receipt['history']
    stage_development(source, split, output, source_hash, split_hash)
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match='outside frozen'):
        stage_development(source, split, output, source_hash, split_hash, [1])
    output.write_text('x,y\nforged,0\n')
    with pytest.raises(ValueError, match='stale'):
        verify_stage(output)
    source.write_text('changed')
    with pytest.raises(ValueError, match='hash differs'):
        stage_development(source, split, output, source_hash, split_hash)


def test_concurrent_duplicate_reservation_and_settlement(tmp_path):
    budget = authority(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: budget.reserve('run', 'r', '.01'), range(16)))
    assert Decimal(budget.get_status()['total_reserved_usd']) == Decimal('.01')
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: budget.settle('run', 'r', '.001', cost_kind='calculated_estimate'), range(16)))
    state = budget.get_status()
    assert Decimal(state['total_billed_usd']) == Decimal('.001')
    assert Decimal(state['total_uncertain_usd']) == Decimal('.009')
    assert Decimal(state['remaining_campaign_usd']) == Decimal('.29')
    assert state['runs']['run']['requests'] == 1
    with pytest.raises(ValueError): budget.reserve('run', 'r', '.01')
    with pytest.raises(ValueError): budget.settle('other', 'r', '.001')
    with pytest.raises(ValueError): budget.settle('run', 'missing', '.001')


def test_ambiguous_crash_is_committed_once_after_reopen(tmp_path):
    budget = authority(tmp_path)
    budget.reserve('run', 'r', '.02')
    budget = authority(tmp_path)
    budget.mark_uncertain('run', 'r', '.02')
    budget.mark_uncertain('run', 'r', '.02')
    assert Decimal(budget.get_status()['total_uncertain_usd']) == Decimal('.02')
    with pytest.raises(ValueError): budget.reserve('run', 'r', '.02')
    with pytest.raises(ValueError): budget.settle('run', 'r', '.001')


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-1'])
def test_invalid_budget_amount_fails(tmp_path, value):
    with pytest.raises(ValueError): authority(tmp_path).reserve('run', 'r', value)


def panel():
    frame = pd.DataFrame([{'date': d, 'store': entity, 'sales': 1, 'calendar': 1}
                          for d in pd.date_range('2020-01-01', periods=6) for entity in (1, 2)])
    topology = {'temporal_col': 'date', 'entity_columns': ['store'], 'forecast_horizon_dates': 2,
                'known_at_forecast_columns': ['calendar'], 'panel_protocol': {
                    'required_entities': [[1], [2]], 'embargo_days': 1,
                    'missing_entity_behavior': 'reject', 'unseen_entity_behavior': 'reject',
                    'covariate_availability': {'calendar': {'kind': 'deterministic_calendar', 'provenance': 'Gregorian calendar'}},
                    'origins': [{'train_start': '2020-01-01', 'train_end': '2020-01-03',
                                 'horizon_dates': ['2020-01-05', '2020-01-06']}]}}
    return frame, topology


def test_complete_panel_and_incomplete_entity_coverage():
    frame, topology = panel()
    assert validate_panel_fold(frame, list(range(6)), list(range(8, 12)), topology, 0, 'sales')['per_entity_coverage'] == 'complete'
    for tr, va in [(list(range(6)), [8, 9, 10]), ([0, 1, 4, 5], list(range(8, 12)))]:
        with pytest.raises(ValueError, match='per-entity'):
            validate_panel_fold(frame, tr, va, topology, 0, 'sales')


@pytest.mark.parametrize('mutation', ['missing', 'embargo', 'covariate', 'unseen', 'origin'])
def test_panel_rejects_incompatible_contract(mutation):
    frame, topology = panel()
    p = topology['panel_protocol']
    if mutation == 'missing': del p['required_entities']
    if mutation == 'embargo': p['embargo_days'] = 0
    if mutation == 'covariate': p['covariate_availability']['calendar'] = {'kind': 'published', 'known_by': '2020-01-04', 'provenance': 'late publication'}
    if mutation == 'unseen': p['required_entities'] = [[1], [3]]
    if mutation == 'origin': p['origins'][0]['horizon_dates'] = ['2020-01-05', '2020-01-07']
    with pytest.raises(ValueError): validate_panel_fold(frame, list(range(6)), list(range(8, 12)), topology, 0, 'sales')


def test_legacy_candidate_env_drops_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-canary-never-in-child')
    (tmp_path / 'entry.py').write_text('import os; assert "OPENROUTER_API_KEY" not in os.environ; print("clean")')
    code, out, err = execute_candidate_cli(tmp_path, 'entry.py', 'fit', [])
    assert code == 0 and out.strip() == 'clean'
    with pytest.raises(Exception, match='unsupported variables'):
        execute_candidate_cli(tmp_path, 'entry.py', 'fit', [], extra_env={'OPENROUTER_API_KEY': 'blocked'})


def test_legacy_candidate_timeout_reaps_group(tmp_path):
    (tmp_path / 'entry.py').write_text('import time; time.sleep(10)')
    code, out, err = execute_candidate_cli(tmp_path, 'entry.py', 'fit', [], timeout_s=.1)
    assert code == 124 and 'deadline' in err


@pytest.mark.parametrize('after_replace', [False, True])
def test_settlement_crash_at_atomic_publish_reconciles_once(tmp_path, monkeypatch, after_replace):
    budget = authority(tmp_path)
    budget.reserve('run', 'r', '.01')
    original = Path.replace
    def interrupted(path, destination):
        if Path(destination) == budget.ledger_path:
            if after_replace: original(path, destination)
            raise RuntimeError('simulated crash at ledger publish')
        return original(path, destination)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'replace', interrupted)
        with pytest.raises(RuntimeError, match='simulated crash'):
            budget.settle('run', 'r', '.001', cost_kind='calculated_estimate')
    reopened = authority(tmp_path)
    reopened.settle('run', 'r', '.001', cost_kind='calculated_estimate')
    state = reopened.get_status()
    assert state['runs']['run']['requests'] == 1
    assert Decimal(state['total_billed_usd']) == Decimal('.001')
    assert Decimal(state['total_uncertain_usd']) == Decimal('.009')


def test_known_overrun_is_recorded_not_hidden(tmp_path):
    budget = authority(tmp_path)
    budget.reserve('run', 'r', '.001')
    budget.settle('run', 'r', '.002', cost_kind='calculated_estimate')
    state = budget.get_status()
    assert Decimal(state['total_billed_usd']) == Decimal('.002')
    assert state['request_records']['r']['reservation_exceeded'] is True


def test_historical_false_estimate_flag_does_not_verify_provider_charge(tmp_path):
    spec = importlib.util.spec_from_file_location('accounting_audit', ROOT / 'scripts/reconcile_campaign_accounting.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    budget = authority(tmp_path)
    budget.reserve('run', 'request', '.01')
    budget.settle('run', 'request', '.01')
    folder = tmp_path / 'run'; folder.mkdir()
    (folder / 'autonomous_loop_state.json').write_text(json.dumps({'turns': [{'provider_receipt': {
        'gateway_route': 'rust_mlrescurix_harness', 'cost_usd': '.01', 'cost_is_estimate': False}}]}))
    before = budget.ledger_path.read_bytes()
    report = audit.reconcile(tmp_path)
    assert report['runs'][0]['receipt_sum_matches_ledger']
    assert report['runs'][0]['receipts'][0]['basis'] == 'calculated_estimate'
    assert report['verified_provider_charge_usd'] is None
    assert budget.ledger_path.read_bytes() == before


def test_legacy_provider_helper_is_blocked_and_contracts_remain_inconclusive():
    with pytest.raises(RuntimeError, match='Legacy direct provider transport disabled'):
        runner.CampaignRunner._call_openrouter_specialist(None, 'agent', [])
    policy = json.loads((ROOT / 'docs/v2/BENCHMARK_SUITE.json').read_text())
    for task in ('K1', 'K2'):
        assert runner.locked_quality_threshold(policy, task) is None
    for task in ('K4',):
        problem = next(p for p in policy['problems'] if p['id'] == task)
        assert problem['execution_contract']['status'].startswith('blocked')
        assert problem['execution_contract']['missing']


def test_unstaged_labeled_file_is_rejected_before_csv_materialization(tmp_path, monkeypatch):
    from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
    source = tmp_path / 'sealed.csv'
    source.write_text('x,y\nfinal-canary,999\n')
    def forbidden_read(*args, **kwargs):
        raise AssertionError('Unstaged labels reached the dataframe parser')
    monkeypatch.setattr(pd, 'read_csv', forbidden_read)
    with pytest.raises(FileNotFoundError):
        AutonomousAgentLoop(tmp_path / 'workspace', 'p', 's', authority(tmp_path / 'budget'),
                            source, 'y', max_turns=0)


def test_frozen_k5_protocol_is_complete_source_bound_and_preserves_fold_counts():
    policy = json.loads((ROOT / 'docs/v2/BENCHMARK_SUITE.json').read_text())
    problem = next(p for p in policy['problems'] if p['id'] == 'K5')
    path = ROOT / problem['execution_contract']['protocol_path']
    assert sha256(path) == problem['execution_contract']['protocol_sha256']
    frozen = json.loads(path.read_text()); protocol = frozen['panel_protocol']
    assert len(protocol['required_entities']) == 1782
    assert len(set(map(tuple, protocol['required_entities']))) == 1782
    assert [len(o['training_dates']) * 1782 for o in protocol['origins']] == [2893968, 2920698, 2947428]
    assert all(len(o['horizon_dates']) == 15 for o in protocol['origins'])
    assert protocol['embargo_days'] == 0 and protocol['embargo_justification']
    assert protocol['training_calendar_provenance']['split_sha256'] == problem['authenticity']['split_manifest_sha256']
    assert all(r['kind'] == 'deterministic_calendar' for r in protocol['covariate_availability'].values())
    assert protocol['missing_entity_behavior'] == protocol['unseen_entity_behavior'] == 'reject'


def test_observation_calendar_allows_only_declared_structural_gaps():
    frame, topology = panel()
    frame = frame.drop(index=[2, 3]).reset_index(drop=True) # source has no observations on Jan 2
    protocol = topology['panel_protocol']
    protocol['training_calendar_provenance'] = {'source_sha256': 'a'*64, 'split_sha256': 'b'*64}
    protocol['origins'][0]['training_dates'] = ['2020-01-01', '2020-01-03']
    receipt = validate_panel_fold(frame, list(range(4)), list(range(6, 10)), topology, 0, 'sales')
    assert receipt['training_dates'] == 2
    protocol['origins'][0]['training_dates'] = ['2020-01-01', '2020-01-02', '2020-01-03']
    with pytest.raises(ValueError, match='omits or invents'):
        validate_panel_fold(frame, list(range(4)), list(range(6, 10)), topology, 0, 'sales')


def test_future_summary_compatibility_fields_never_claim_verified_charges(tmp_path):
    obj = runner.CampaignRunner.__new__(runner.CampaignRunner)
    path = tmp_path / 'provider_usage.json'
    obj._atomic_write_json(path, {'total_billed_cost_usd': .01})
    record = json.loads(path.read_text())
    assert record['total_billed_cost_usd'] == .01
    assert record['verified_provider_charge_usd'] is None
    assert record['legacy_billed_fields_are_verified_charges'] is False
