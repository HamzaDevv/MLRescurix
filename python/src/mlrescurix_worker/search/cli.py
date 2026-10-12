"""Command line interface for the search sidecar adapter."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapter import get_optimizer


def main() -> int:
    parser = argparse.ArgumentParser(prog="mlrescurix-search")
    commands = parser.add_subparsers(dest="command", required=True)

    ask_p = commands.add_parser("ask")
    ask_p.add_argument("--dir", type=Path, required=True)
    ask_p.add_argument("--session", required=True)
    ask_p.add_argument("--recipe", required=True)
    ask_p.add_argument("--fidelity", type=float, required=True)
    ask_p.add_argument("--task", required=True)
    ask_p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    ask_p.add_argument("--seed", type=int, default=2026)

    tell_p = commands.add_parser("tell")
    tell_p.add_argument("--dir", type=Path, required=True)
    tell_p.add_argument("--session", required=True)
    tell_p.add_argument("--recipe", required=True)
    tell_p.add_argument("--fidelity", type=float, required=True)
    tell_p.add_argument("--trial-id", type=int, required=True)
    tell_p.add_argument("--value", type=float, default=None)
    tell_p.add_argument("--state", choices=["COMPLETE", "FAIL", "PRUNED"], default="COMPLETE")
    tell_p.add_argument("--direction", choices=["maximize", "minimize"], default="maximize")
    tell_p.add_argument("--seed", type=int, default=2026)

    args = parser.parse_args()
    try:
        opt = get_optimizer(args.dir, args.session, args.direction, seed=args.seed)
        if args.command == "ask":
            trial_id, params = opt.ask(args.recipe, args.fidelity, args.task)
            result = {
                "status": "ok",
                "optimizer_trial_id": trial_id,
                "recipe": args.recipe,
                "fidelity": args.fidelity,
                "parameters": params,
            }
        else:
            opt.tell(args.recipe, args.fidelity, args.trial_id, args.value, args.state)
            result = {
                "status": "ok",
                "optimizer_trial_id": args.trial_id,
                "state": args.state,
            }
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
