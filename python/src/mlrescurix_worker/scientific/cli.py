"""Command line entry point for the M2 local workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import (WorkflowError, audit_csv, finalize_candidate,
                       fit_trial_worker, predict_trial_worker, run_baselines)


def main() -> int:
    parser = argparse.ArgumentParser(prog="mlrescurix-scientific")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--csv", type=Path, required=True)
    audit.add_argument("--spec", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("baselines")
    run.add_argument("--output", type=Path, required=True)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--candidate", required=True)
    finalize.add_argument("--test-access-id", required=True)
    for name in ("fit-trial", "predict-trial"):
        worker = commands.add_parser(name, help=argparse.SUPPRESS)
        worker.add_argument("--spec", type=Path, required=True)
        worker.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "audit":
            result = audit_csv(args.csv, json.loads(args.spec.read_text()), args.output)
        elif args.command == "baselines":
            result = run_baselines(args.output)
        elif args.command == "finalize":
            result = finalize_candidate(args.output, args.candidate, test_access_id=args.test_access_id)
        elif args.command == "fit-trial":
            result = fit_trial_worker(args.spec, args.output)
        else:
            result = predict_trial_worker(args.spec, args.output)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except (WorkflowError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
