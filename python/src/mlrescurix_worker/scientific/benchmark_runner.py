"""One contract-driven development benchmark lifecycle for K1–K5."""
import time
from decimal import Decimal
from .autonomous_loop import AutonomousAgentLoop
from .rust_gateway import RustGatewayAdapter
from .task_adapters import benchmark_adapter
from .benchmark_contracts import (verified_provider_pricing, locked_quality_threshold,
    accepted_candidate_evidence, package_accepted_candidate, score_meets_threshold,
    candidate_gate_failure_reason)


def run_benchmark(campaign, task_id):
    def blocked(reason):
        outcome = {'status': 'blocked', 'task': task_id, 'reason': str(reason)}
        campaign.summary['gates']['G_LIVE_DEV'] = outcome
        campaign.summary['status'] = 'blocked'
        campaign.save_summary()
        return outcome

    from .autonomous_loop import REPO_ROOT
    adapter = benchmark_adapter(campaign.policy, task_id, REPO_ROOT)
    # Scientific contracts reject before gateway initialization, reservation or upload.
    try: adapter.validate_contract()
    except (ValueError, KeyError, TypeError) as exc: return blocked(exc)
    pricing = verified_provider_pricing(campaign.policy)
    if pricing is None:
        return blocked('Provider pricing snapshot is stale or has not been explicitly revalidated for live use')
    try:
        path, frame, folds, topology, statement = adapter.prepare(campaign.run_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc: return blocked(exc)
    gateway_block = campaign._live_gateway_block()
    if gateway_block is not None: return gateway_block
    status = campaign.campaign_authority.get_status()
    if Decimal(status['remaining_campaign_usd']) < Decimal('.05'):
        return blocked('Insufficient remaining shared campaign authority for USD 0.05 run cap')
    p = adapter.problem
    project = f'{task_id.lower()}_robustness'
    gateway = RustGatewayAdapter(state_dir=campaign.run_dir / 'rust_harness_state', project_id=project,
        task_id=f'{task_id.lower()}_dev_solve', role_id='scientific-agent',
        model_name='deepseek/deepseek-v4.1-flash-1017', endpoint='inference-net/fp8', pricing=pricing)
    started = time.monotonic()
    controller = AutonomousAgentLoop(workspace_root=campaign.run_dir, project_id=project,
        session_id=campaign.run_id, campaign_authority=campaign.campaign_authority,
        dataset_path=path, target_column=p['target'], task_type=p['task'], metric_name=p['metric'],
        metric_direction=p['direction'], locked_folds=folds, require_locked_folds=True,
        max_turns=3, per_run_budget_usd=.05, pricing=pricing, topology=topology,
        problem_statement=statement, provider_caller=gateway,
        worker_memory_bytes=p['budget']['memory_mib'] * 1024**2)
    state = controller.run()
    if state.get('stop_reason') in ('human_pause', 'human_stop'):
        outcome = {'status': 'paused' if state['stop_reason'] == 'human_pause' else 'stopped',
                   'task': task_id, 'reason': state['stop_reason'], 'production_certified': False}
        campaign.summary['gates']['G_LIVE_DEV'] = outcome
        campaign.summary['status'] = outcome['status']
        campaign.save_summary()
        return outcome
    evidence_ok, score, evidence_reason = accepted_candidate_evidence(state)
    bundle_ok, identity_ok, package_reason = package_accepted_candidate(controller, state, frame,
        candidate_dir=campaign.run_dir / 'candidate', bundle_dir=campaign.run_dir / 'artifacts/bundle',
        dataset_path=path, test_access_id=f'gate_{campaign.run_id}', project_id=project,
        task_type='regression' if p['task'] == 'panel_forecasting' else p['task'],
        target_column=p['target'], primary_metric=p['metric'])
    threshold = locked_quality_threshold(campaign.policy, task_id)
    passed = evidence_ok and bundle_ok and identity_ok and score_meets_threshold(score, threshold, p['direction'])
    usage = campaign.campaign_authority.get_status()
    resources = controller.sampler.sample_now()
    outcome = {'status': 'passed' if passed else 'failed', 'task': task_id,
        'reason': candidate_gate_failure_reason(evidence_ok, evidence_reason, score, threshold,
            p['direction'], bundle_ok, identity_ok, package_reason),
        'best_cv_score': score, 'metric': p['metric'], 'holdout_score': None, 'holdout_accesses': 0,
        'bundle_verified': bundle_ok, 'bundle_identity_verified': identity_ok,
        'candidate_identity': state.get('candidate_identity'), 'accepted_candidate_id': state.get('accepted_candidate_id'),
        'provider_cost_usd': usage.get('runs', {}).get(campaign.run_id, {}).get('billed_usd', '0'),
        'provider_cost_kind': 'unverified_or_calculated_estimate', 'verified_provider_charge_usd': None,
        'duration_s': time.monotonic() - started, 'production_certified': False}
    campaign._atomic_write_json(campaign.metrics_path, outcome)
    campaign._atomic_write_json(campaign.provider_usage_path, {'campaign': usage, 'verified_provider_charge_usd': None,
        'total_billed_cost_usd': outcome['provider_cost_usd'], 'reserved_cost_usd': usage.get('runs', {}).get(campaign.run_id, {}).get('reserved_usd', '0'),
        'cost_kind': 'unverified_or_calculated_estimate'})
    campaign.record_resource('development', resources)
    campaign.record_trace('benchmark_completed', outcome)
    campaign.summary['gates']['G_LIVE_DEV'] = {'outcome': outcome['status'], 'details': outcome}
    campaign.summary['status'] = outcome['status']
    campaign.save_summary()
    return outcome
