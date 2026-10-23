"""Offline service, human-control, recovery and cross-task delivery regressions."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlrescurix_worker.scientific import benchmark_runner
from mlrescurix_worker.scientific.agent_interface import AgentInterface, main as control_main
from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop, RecoveryRequired
from mlrescurix_worker.scientific.candidate_pipeline import identity
from mlrescurix_worker.scientific.development_stage import sha256
from mlrescurix_worker.scientific.evidence import atomic_json, read_worker_result
from mlrescurix_worker.scientific.governance import CampaignBudgetAuthority
from mlrescurix_worker.scientific.task_adapters import benchmark_adapter, FOLDERS
from test_autonomy_regressions import make_loop, action

ROOT = Path(__file__).resolve().parents[2]
POLICY = json.loads((ROOT / 'docs/v2/BENCHMARK_SUITE.json').read_text())


def test_human_steering_and_pause_survive_restart_without_provider_or_budget_work(tmp_path):
    loop, prompts = make_loop(tmp_path, [action('ANALYZE')])
    interface = AgentInterface(loop.workspace_root)
    interface.submit('steer', 'Prioritize interpretable models and explain measured failures.', command_id='guidance')
    interface.submit('pause', command_id='pause')
    assert loop.run()['stop_reason'] == 'human_pause'
    assert prompts == [] and not loop.campaign_authority.requests
    resumed, prompts = make_loop(tmp_path, [action('HANDOFF', reason='Human requested review')])
    interface.submit('resume', command_id='resume')
    state = resumed.run()
    assert len(prompts) == 1
    assert prompts[0]['human_guidance'][0]['message'].startswith('Prioritize interpretable')
    assert state['human_commands_applied'] == ['guidance', 'pause', 'resume']
    assert interface.status()['pending_commands'] == []
    assert state['stop_reason'] == 'HANDOFF'


def test_human_command_concurrency_idempotence_and_content_binding(tmp_path):
    loop, _ = make_loop(tmp_path, [action('ANALYZE')])
    interface = loop.interface
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: interface.submit('steer', 'Use measured feedback', command_id='same'), range(16)))
    assert len(json.loads(interface.inbox_path.read_text())) == 1
    with pytest.raises(ValueError, match='reused'): interface.submit('stop', command_id='same')
    data = json.loads(interface.inbox_path.read_text())
    data[0]['message'] = 'tampered'
    atomic_json(interface.inbox_path, data)
    with pytest.raises(ValueError, match='binding mismatch'): loop.run()
    assert loop.campaign_authority.requests == []


def test_human_stop_during_provider_turn_takes_effect_before_next_dispatch(tmp_path):
    loop, _ = make_loop(tmp_path, [action('ANALYZE'), action('ANALYZE')])
    def provider(messages, tokens):
        loop.interface.submit('stop', command_id='stop')
        return {'content': json.dumps(action('ANALYZE')), 'cost_usd': 0}
    loop.provider_caller = provider
    state = loop.run()
    assert state['turns_completed'] == 1 and state['stop_reason'] == 'human_stop'
    loop.interface.submit('resume')
    assert loop.run()['stop_reason'] == 'human_stop'
    assert len(loop.campaign_authority.requests) == 1


def test_control_cli_reports_queued_command_and_status(tmp_path, capsys):
    loop, _ = make_loop(tmp_path, [])
    assert control_main(['--workspace', str(loop.workspace_root), 'steer', '--message', 'Inspect residuals', '--command-id', 'cli']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'queued'
    control_main(['--workspace', str(loop.workspace_root), 'status'])
    assert json.loads(capsys.readouterr().out)['pending_commands'] == ['cli']
    with pytest.raises(ValueError): loop.interface.submit('approve_upload')


@pytest.mark.parametrize('task_id', ['K1', 'K2', 'K4'])
def test_incomplete_benchmark_contract_blocks_before_gateway_or_reservation(tmp_path, monkeypatch, task_id):
    budget = CampaignBudgetAuthority(tmp_path / 'budget', POLICY)
    before = budget.ledger_path.read_bytes()
    campaign = SimpleNamespace(policy=POLICY, summary={'gates': {}}, campaign_authority=budget, save_summary=lambda: None)
    def forbidden(*a, **k): raise AssertionError('Blocked contract initialized gateway')
    monkeypatch.setattr(benchmark_runner, 'RustGatewayAdapter', forbidden)
    outcome = benchmark_runner.run_benchmark(campaign, task_id)
    assert outcome['status'] == 'blocked'
    assert budget.ledger_path.read_bytes() == before


def test_foreign_control_binding_rejected(tmp_path):
    loop, _ = make_loop(tmp_path, [])
    loop.interface.submit('pause')
    data = json.loads(loop.interface.inbox_path.read_text())
    data[0]['binding_identity'] = 'foreign-session'
    body = dict(data[0]); body.pop('receipt_identity')
    data[0]['receipt_identity'] = identity(body)
    atomic_json(loop.interface.inbox_path, data)
    with pytest.raises(ValueError, match='binding mismatch'): loop.run()


def test_worker_cache_rejects_foreign_and_corrupt_receipts(tmp_path):
    path = tmp_path / 'result.json'
    result = {'cv_score': .5}
    atomic_json(path, {'request_identity': 'r', 'payload_identity': identity(result), 'result': result})
    assert read_worker_result(path, 'r') == result
    with pytest.raises(ValueError, match='request identity'): read_worker_result(path, 'other')
    payload = json.loads(path.read_text()); payload['result']['cv_score'] = .99
    atomic_json(path, payload)
    with pytest.raises(ValueError, match='content identity'): read_worker_result(path, 'r')


def test_controller_cache_and_approved_measurement_tamper_fail_closed(tmp_path):
    loop, _ = make_loop(tmp_path, [action('BLEND_AND_AUDIT', families=['penalized_linear'])])
    loop.critic_engine.audit = lambda **kw: SimpleNamespace(to_dict=lambda: {'status': 'APPROVED', 'candidate_id': kw['candidate']['candidate_id']})
    state = loop.run()
    assert state['accepted_candidate_id']
    spec = state['accepted_candidate']
    token = identity({'spec': spec, 'binding': loop.binding, 'operation': 'evaluate'})
    output = loop.workspace_root / 'evaluations' / f'{token}.result.json'
    payload = json.loads(output.read_text()); payload['result']['cv_score'] = .99
    atomic_json(output, payload)
    with pytest.raises(ValueError, match='content identity'): loop._evaluate(spec)
    state['accepted_evaluation']['cv_score'] = .99
    loop._save_state(state)
    with pytest.raises(RecoveryRequired, match='receipt identity'): loop.fit_accepted_candidate()


def test_changed_source_binding_rejects_resume(tmp_path, monkeypatch):
    loop, _ = make_loop(tmp_path, [])
    from mlrescurix_worker.scientific import autonomous_loop
    monkeypatch.setattr(autonomous_loop, 'source_identity', lambda root: 'changed-source')
    with pytest.raises(RecoveryRequired, match='binding mismatch'): make_loop(tmp_path, [])


def fixture_campaign(tmp_path, task_id):
    policy = copy.deepcopy(POLICY)
    p = next(p for p in policy['problems'] if p['id'] == task_id)
    p['target'] = 'y'
    # Explicit synthetic contract, never an official quality requirement.
    p['final_quality_criteria'] = {'status': 'locked', 'threshold': -1 if p['direction'] == 'maximize' else 100}
    folder = tmp_path / 'data/official' / FOLDERS[task_id]
    folder.mkdir(parents=True)
    if task_id == 'K5':
        frame = pd.DataFrame([{'date': str(d.date()), 'store_nbr': entity, 'family': 'a', 'sales': 1 + i % 3}
            for i, d in enumerate(pd.date_range('2020-01-01', periods=13)) for entity in (1, 2)])
        p['target'] = 'sales'
        manifest = {'task_id': task_id, 'development_indices': list(range(24)), 'holdout_indices': [24, 25],
            'development_origins': [{'train_end_date': '2020-01-10', 'validation_start_date': '2020-01-11', 'validation_end_date': '2020-01-12'}]}
    else:
        frame = pd.DataFrame({'x': list(range(42)), 'y': [i % 2 if task_id == 'K3' else i + 1 for i in range(42)]})
        manifest = {'task_id': task_id, 'development_indices': list(range(40)), 'holdout_indices': [40, 41],
                    'development_folds': [{'train_indices': list(range(30)), 'val_indices': list(range(30, 40))}]}
    source, split = folder / 'train.csv', folder / 'split_manifest.json'
    frame.to_csv(source, index=False); split.write_text(json.dumps(manifest))
    p['authenticity'] = {'files_sha256': {'train.csv': sha256(source)}, 'split_manifest_sha256': sha256(split)}
    if task_id == 'K5':
        protocol = {'temporal_col': 'date', 'entity_columns': ['store_nbr', 'family'], 'forecast_horizon_dates': 2,
            'known_at_forecast_columns': ['day_of_week', 'day_of_month', 'month'], 'panel_protocol': {
                'required_entities': [[1, 'a'], [2, 'a']], 'embargo_days': 0,
                'training_calendar_provenance': {'source_sha256': sha256(source), 'split_sha256': sha256(split)},
                'missing_entity_behavior': 'reject', 'unseen_entity_behavior': 'reject',
                'covariate_availability': {c: {'kind': 'deterministic_calendar', 'provenance': 'Gregorian calendar'} for c in ('day_of_week', 'day_of_month', 'month')},
                'origins': [{'train_start': '2020-01-01', 'train_end': '2020-01-10', 'horizon_dates': ['2020-01-11', '2020-01-12']}]}}
        (tmp_path / 'protocol.json').write_text(json.dumps(protocol))
        p['execution_contract'] = {'status': 'locked', 'protocol_path': 'protocol.json', 'protocol_sha256': sha256(tmp_path / 'protocol.json')}
    run = tmp_path / 'run'; run.mkdir()
    budget = CampaignBudgetAuthority(tmp_path / 'budget', policy)
    campaign = SimpleNamespace(policy=policy, run_id='fixture', run_dir=run, campaign_authority=budget,
        summary={'gates': {}}, save_summary=lambda: None, _live_gateway_block=lambda: None,
        _atomic_write_json=atomic_json, record_resource=lambda *a: None, record_trace=lambda *a: None,
        metrics_path=run / 'metrics.json', provider_usage_path=run / 'usage.json')
    return campaign


@pytest.mark.parametrize('task_id', ['K2', 'K3', 'K5'])
def test_common_lifecycle_real_workers_and_identity_bound_bundle(tmp_path, monkeypatch, task_id):
    campaign = fixture_campaign(tmp_path, task_id)
    monkeypatch.setattr(benchmark_runner, 'benchmark_adapter', lambda policy, task, root: benchmark_adapter(policy, task, tmp_path))
    monkeypatch.setattr(benchmark_runner, 'verified_provider_pricing', lambda policy: {'verified': True, 'input_per_million': 0, 'output_per_million': 0})
    monkeypatch.setattr(benchmark_runner, 'RustGatewayAdapter', lambda **kw: None)
    prompts = []
    def factory(**kw):
        def provider(messages, tokens):
            prompts.append(messages)
            a = action('BLEND_AND_AUDIT', families=['penalized_linear']) if len(prompts) == 1 else action('FINALIZE_AND_STOP')
            return {'content': json.dumps(a), 'cost_usd': 0}
        kw.update(provider_caller=provider, provider_is_fixture=True)
        loop = AutonomousAgentLoop(**kw)
        loop.critic_engine.audit = lambda **args: SimpleNamespace(to_dict=lambda: {'status': 'APPROVED', 'candidate_id': args['candidate']['candidate_id']})
        return loop
    monkeypatch.setattr(benchmark_runner, 'AutonomousAgentLoop', factory)
    outcome = benchmark_runner.run_benchmark(campaign, task_id)
    assert outcome['status'] == 'passed', outcome
    bundle = campaign.run_dir / 'artifacts/bundle'
    evidence = json.loads((bundle / 'delivery_evidence.json').read_text())
    manifest = json.loads((bundle / 'manifest.json').read_text())
    assert manifest['files']['delivery_evidence.json']['sha256'] == sha256(bundle / 'delivery_evidence.json')
    assert evidence['model_sha256'] == manifest['model']['weights_sha256']
    assert evidence['approved_audit']['evaluation_identity'] == evidence['evaluation']['receipt_identity']
    assert evidence['binding'] == evidence['evaluation']['execution_binding']
    assert evidence['binding']['stage_receipt'] is None  # explicitly synthetic fixture
    assert evidence['candidate_identity'] == outcome['candidate_identity']
    assert outcome['holdout_score'] is None and not outcome['production_certified']
    # Existing standalone bundle test harness independently exercises exported CLI;
    # here verify actual raw-input predictions outside the training data frame.
    import subprocess, sys
    result = subprocess.run([sys.executable, '-c', 'import inference; print(len(inference.predict(__import__("json").load(open("examples_input.json")))))'],
        cwd=bundle, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_preserved_authority_open_is_read_only_and_cannot_create_allowance(tmp_path):
    with pytest.raises(FileNotFoundError, match='never create'):
        CampaignBudgetAuthority.open_preserved(tmp_path / 'missing', POLICY)
    assert not (tmp_path / 'missing').exists()
    ledger = ROOT / 'docs/evidence/v2/robustness/20261021-robustness/campaign_budget_ledger.json'
    before = ledger.read_bytes()
    budget = CampaignBudgetAuthority.open_preserved(ledger.parent, POLICY)
    assert budget.get_status()['total_billed_usd'] == '0.050724020'
    assert ledger.read_bytes() == before


@pytest.mark.parametrize('name', ['run_kaggle_titanic_live', 'run_live_end_to_end_test'])
def test_legacy_paid_transport_has_no_dispatch_path(name, monkeypatch):
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'load_openrouter_key', lambda: pytest.fail('Read credentials from retired path'))
    with pytest.raises(RuntimeError, match='retired'): module.call_deepseek_flash([])


def test_changed_source_cannot_execute_under_old_session_binding(tmp_path, monkeypatch):
    loop, _ = make_loop(tmp_path, [])
    from mlrescurix_worker.scientific import execution_service
    monkeypatch.setattr(execution_service, 'source_identity', lambda root: 'changed-worker')
    with pytest.raises(ValueError, match='source changed'): loop._evaluate(loop.state['active_spec'])
    assert not (loop.workspace_root / 'evaluations').exists()
