"""Run a resumable development agent. Never treats fixtures as live evidence.

python -m mlrescurix_worker.scientific.agent_cli --help
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from .autonomous_loop import AutonomousAgentLoop, REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="MLRescurix development agent: diagnose, write/run isolated Python, evaluate, repair, hand off")
    parser.add_argument("--development-csv", required=True, type=Path, help="Development-only CSV; never supply a final/holdout file")
    parser.add_argument("--folds", required=True, type=Path, help="JSON list of {train_indices, val_indices} in development-local row space")
    parser.add_argument("--workspace", required=True, type=Path, help="Dedicated durable session directory")
    parser.add_argument("--project", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", choices=["binary_classification", "multiclass_classification", "regression", "panel_forecasting"], default="binary_classification")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--problem", default="Analyze this development dataset and build a validated candidate or hand off unresolved work.")
    parser.add_argument("--topology", type=Path, help="Authoritative topology, including cv_scheme/entity/temporal columns")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--sandbox-image", help="Locally available pinned scientific Docker image")
    parser.add_argument("--policy", type=Path, help="Campaign policy with today's verified route pricing")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    parser.add_argument("--provider-route", default="inference-net/fp8", help="OpenRouter provider identifier; fallback is disabled")
    parser.add_argument("--approval-wait-seconds", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture-actions", type=Path, help="Offline deterministic actions; cannot count as live AI evidence")
    mode.add_argument("--development-live", action="store_true", help="Governed Rust transport; requires separate outbound approval; not production certification")
    args = parser.parse_args(argv)
    from .governance import CampaignBudgetAuthority
    from .benchmark_contracts import verified_provider_pricing
    policy = json.loads((args.policy or REPO_ROOT / "docs/v2/BENCHMARK_SUITE.json").read_text())
    folds = json.loads(args.folds.read_text())
    locked = [(f["train_indices"], f["val_indices"]) for f in folds]
    topology = json.loads(args.topology.read_text()) if args.topology else {"cv_scheme": "caller_locked"}
    authority = (CampaignBudgetAuthority(args.workspace / 'offline_fixture_budget', policy) if args.fixture_actions else
        CampaignBudgetAuthority.open_preserved(REPO_ROOT / "docs/evidence/v2/robustness/20261021-robustness", policy))
    provider = None
    if args.fixture_actions:
        actions = iter(json.loads(args.fixture_actions.read_text()))
        def provider(messages, max_tokens): return {"content": json.dumps(next(actions)), "cost_usd": 0, "fixture": True}
    pricing = verified_provider_pricing(policy) if args.development_live else None
    if args.development_live and pricing is None:
        parser.error("Current verified provider price ceilings required in BENCHMARK_SUITE.json before live dispatch")
    if args.development_live:
        route = policy["provider"].get("provider_routing", {})
        if (policy["provider"].get("model_id") != args.model or
                route.get("only") != [args.provider_route] or
                route.get("order") != [args.provider_route] or
                route.get("allow_fallbacks") is not False):
            parser.error("Verified policy must match the requested model and single provider route")
        from .rust_gateway import RustGatewayAdapter
        provider = RustGatewayAdapter(args.workspace / "rust_harness_state", args.project,
                                      model_name=args.model, endpoint=args.provider_route,
                                      pricing=pricing, require_policy_approval=True,
                                      approval_wait_seconds=args.approval_wait_seconds)
    loop = AutonomousAgentLoop(args.workspace, args.project, args.session, authority, args.development_csv,
        args.target, task_type=args.task, metric_name=args.metric, metric_direction=args.direction,
        locked_folds=locked, require_locked_folds=True, topology=topology,
        problem_statement=args.problem, max_turns=args.max_turns, provider_caller=provider,
        model_name=args.model, endpoint=args.provider_route,
        per_run_budget_usd=policy["provider"]["usd_per_main_live_run_max"],
        sandbox_image=args.sandbox_image, pricing=pricing, provider_is_fixture=bool(args.fixture_actions),
        allow_development_provider=args.development_live)
    if not args.fixture_actions and not args.development_live:
        print(json.dumps({"mode": "preflight_only", "diagnostics": loop.build_diagnostic_dossier(),
            "budget": authority.get_status(), "production_certified": False}, default=str, indent=2))
        return 0
    state = loop.run()
    print(json.dumps({"mode": "offline_fixture" if args.fixture_actions else "development_live", "turns_completed": state["turns_completed"],
        "accepted_candidate_id": state["accepted_candidate_id"], "best_cv_score": state["best_cv_score"],
        "stop_reason": state["stop_reason"], "handoff": str(loop.workspace_root / "handoff.json"), "production_certified": False}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
