"""Command line entry point for the M2 local workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .workflow import (WorkflowError, audit_csv, finalize_candidate,
                       fit_trial_worker, predict_trial_worker, run_baselines, run_trial)


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
    trial = commands.add_parser("trial")
    trial.add_argument("--output", type=Path, required=True)
    trial.add_argument("--spec", type=Path, required=True)
    pipeline = commands.add_parser("pipeline")
    pipeline.add_argument("--output", type=Path, required=True)
    pipeline.add_argument("--spec", type=Path, required=True)
    cand_eval = commands.add_parser("candidate-evaluate")
    cand_eval.add_argument("--output", type=Path, required=True)
    cand_eval.add_argument("--spec", type=Path, required=True)
    cand_synth = commands.add_parser("candidate-synthesize")
    cand_synth.add_argument("--spec", type=Path, required=True)
    cand_synth.add_argument("--output-dir", type=Path, required=True)
    cand_synth.add_argument("--output", type=Path, required=False)
    diag_cmd = commands.add_parser("diagnose-failure")
    diag_cmd.add_argument("--exit-code", type=int, default=1)
    diag_cmd.add_argument("--stderr", type=str, default="")
    diag_cmd.add_argument("--stdout", type=str, default="")
    diag_cmd.add_argument("--message", type=str, default="")
    diag_cmd.add_argument("--candidate", type=str, default="candidate")
    diag_cmd.add_argument("--stage", type=str, default="evaluate")
    diag_cmd.add_argument("--output", type=Path, required=False)
    iter_screen = commands.add_parser("iteration-screen")
    iter_screen.add_argument("--spec", type=Path, required=True)
    iter_screen.add_argument("--output", type=Path, required=True)
    pareto_cmd = commands.add_parser("pareto-frontier")
    pareto_cmd.add_argument("--spec", type=Path, required=True)
    pareto_cmd.add_argument("--output", type=Path, required=True)
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
        elif args.command == "trial":
            result = run_trial(args.output, json.loads(args.spec.read_text()))
        elif args.command == "pipeline":
            from .scheduled_pipeline import run_pipeline
            result = run_pipeline(args.output, json.loads(args.spec.read_text()))
        elif args.command == "candidate-evaluate":
            from .candidate_runner import evaluate_candidate_cv
            from .diagnostic_bridge import DiagnosticBridge
            spec_data = json.loads(args.spec.read_text())
            dataset_path = Path(spec_data["dataset_path"])
            candidate_dir = Path(spec_data["candidate_dir"])
            manifest = spec_data["manifest"]
            cid = manifest.get("candidate_id", "candidate")
            folds = spec_data.get("folds", 3)
            seed = spec_data.get("seed", 2026)
            slices = spec_data.get("diagnostic_slices")
            try:
                result = evaluate_candidate_cv(
                    dataset_csv=dataset_path,
                    manifest=manifest,
                    candidate_dir=candidate_dir,
                    folds=folds,
                    seed=seed,
                    diagnostic_slices=slices,
                )
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
            except Exception as exc:
                diag_receipt = getattr(exc, "diagnostic_receipt", None)
                if diag_receipt is None:
                    diag_receipt = DiagnosticBridge.diagnose_exception(
                        exc,
                        candidate_id=cid,
                        stage="evaluate",
                        candidate_dir=str(candidate_dir),
                    )
                result = {
                    "status": "failed",
                    "candidate_id": cid,
                    "error": str(exc),
                    "diagnostic_receipt": diag_receipt.to_dict() if hasattr(diag_receipt, "to_dict") else diag_receipt,
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "candidate-synthesize":
            from .candidate_synthesis import CandidateSynthesizer
            spec_data = json.loads(args.spec.read_text())
            synthesized = CandidateSynthesizer.synthesize(spec_data, args.output_dir)
            result = synthesized.to_dict()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "diagnose-failure":
            from .diagnostic_bridge import DiagnosticBridge
            diag_res = DiagnosticBridge.diagnose_failure(
                exit_code=args.exit_code,
                stderr=args.stderr,
                stdout=args.stdout,
                message=args.message,
                candidate_id=args.candidate,
                stage=args.stage,
            )
            result = diag_res.to_dict()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "iteration-screen":
            from .iteration_loop import MultiFidelityIterationEngine
            spec_data = json.loads(args.spec.read_text())
            dataset_path = Path(spec_data["dataset_path"])
            candidate_dir = Path(spec_data["candidate_dir"])
            manifest = spec_data.get("manifest")
            if manifest is None:
                manifest = json.loads((candidate_dir / "manifest.json").read_text())
            hypothesis = spec_data.get("hypothesis")
            baseline_score = spec_data.get("baseline_score")
            skip_proxy = spec_data.get("skip_proxy", False)
            folds = spec_data.get("folds", 3)
            seed = spec_data.get("seed", 2026)
            result = MultiFidelityIterationEngine.run_multi_fidelity_iteration(
                dataset_csv=dataset_path,
                manifest=manifest,
                candidate_dir=candidate_dir,
                hypothesis_spec=hypothesis,
                baseline_score=baseline_score,
                skip_proxy=skip_proxy,
                folds=folds,
                seed=seed,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "pareto-frontier":
            from .iteration_loop import MultiFidelityIterationEngine
            spec_data = json.loads(args.spec.read_text())
            records = spec_data["records"]
            primary_metric = spec_data.get("primary_metric_name", "primary_metric")
            direction = spec_data.get("optimization_direction", spec_data.get("direction", "maximize"))
            project_id = spec_data.get("project_id", "default")
            frontier_id = spec_data.get("frontier_id")
            result = MultiFidelityIterationEngine.compute_pareto_frontier(
                records=records,
                primary_metric_name=primary_metric,
                direction=direction,
                project_id=project_id,
                frontier_id=frontier_id,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
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
