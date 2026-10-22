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
    parser.add_argument("--task", choices=["binary_classification", "multiclass_classification", "regression"], default="binary_classification")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--problem", default="Analyze this development dataset and build a validated candidate or hand off unresolved work.")
    parser.add_argument("--topology", type=Path, help="Authoritative topology, including cv_scheme/entity/temporal columns")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--sandbox-image", help="Locally available pinned scientific Docker image")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fixture-actions", type=Path, help="Offline deterministic actions; cannot count as live AI evidence")
    mode.add_argument("--development-live", action="store_true", help="Explicit development transport; NOT production Rust certification")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_robustness_campaign import CampaignBudgetAuthority, verified_provider_pricing
    policy = json.loads((REPO_ROOT / "docs/v2/BENCHMARK_SUITE.json").read_text())
    folds = json.loads(args.folds.read_text())
    locked = [(f["train_indices"], f["val_indices"]) for f in folds]
    topology = json.loads(args.topology.read_text()) if args.topology else {"cv_scheme": "caller_locked"}
    authority = CampaignBudgetAuthority(REPO_ROOT / "docs/evidence/v2/robustness/20261021-robustness", policy)
    provider = None
    if args.fixture_actions:
        actions = iter(json.loads(args.fixture_actions.read_text()))
        def provider(messages, max_tokens): return {"content": json.dumps(next(actions)), "cost_usd": 0, "fixture": True}
    pricing = verified_provider_pricing(policy) if args.development_live else None
    if args.development_live and pricing is None:
        parser.error("Current verified provider price ceilings required in BENCHMARK_SUITE.json before live dispatch")
    loop = AutonomousAgentLoop(args.workspace, args.project, args.session, authority, args.development_csv,
        args.target, task_type=args.task, metric_name=args.metric, metric_direction=args.direction,
        locked_folds=locked, require_locked_folds=True, topology=topology,
        problem_statement=args.problem, max_turns=args.max_turns, provider_caller=provider,
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
