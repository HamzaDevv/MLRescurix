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
    prof_cmd = commands.add_parser("dataset-profile")
    prof_cmd.add_argument("--csv", type=Path, required=True)
    prof_cmd.add_argument("--target", type=str, required=False)
    prof_cmd.add_argument("--id-col", type=str, required=False)
    prof_cmd.add_argument("--output", type=Path, required=False)
    eda_cmd = commands.add_parser("eda-forensics")
    eda_cmd.add_argument("--dataset", type=Path, required=True)
    eda_cmd.add_argument("--target", type=str, required=False)
    eda_cmd.add_argument("--problem-statement", type=str, required=False)
    eda_cmd.add_argument("--output-dir", type=Path, default=Path(".m2/artifacts/eda"))
    eda_cmd.add_argument("--output", type=Path, required=False)
    eda_cmd.add_argument("--db-path", type=Path, required=False)
    eda_cmd.add_argument("--project-id", type=str, default="default")
    pkg_bundle = commands.add_parser("candidate-package-bundle")
    pkg_bundle.add_argument("--spec", type=Path, required=True)
    pkg_bundle.add_argument("--output", type=Path, required=False)
    rescue_cmd = commands.add_parser("rescue-orchestrate")
    rescue_cmd.add_argument("--spec", type=Path, required=True)
    rescue_cmd.add_argument("--output", type=Path, required=False)
    ens_cmd = commands.add_parser("pareto-ensemble")
    ens_cmd.add_argument("--spec", type=Path, required=True)
    ens_cmd.add_argument("--output", type=Path, required=False)
    bench_cmd = commands.add_parser("prospective-benchmark")
    bench_cmd.add_argument("--spec", type=Path, required=True)
    bench_cmd.add_argument("--output", type=Path, required=False)
    ablation_cmd = commands.add_parser("ablation-sprint")
    ablation_cmd.add_argument("--spec", type=Path, required=True)
    ablation_cmd.add_argument("--output", type=Path, required=False)
    portfolio_cmd = commands.add_parser("model-portfolio-eval")
    portfolio_cmd.add_argument("--spec", type=Path, required=True)
    portfolio_cmd.add_argument("--output", type=Path, required=False)
    critic_cmd = commands.add_parser("critic-audit")
    critic_cmd.add_argument("--candidate", type=Path, required=True)
    critic_cmd.add_argument("--topology", type=Path, required=True)
    critic_cmd.add_argument("--forensics", type=Path, required=False)
    critic_cmd.add_argument("--dossier", type=Path, required=False)
    critic_cmd.add_argument("--output", type=Path, required=False)
    critic_cmd.add_argument("--db-path", type=Path, required=False)
    critic_cmd.add_argument("--project-id", type=str, default="default")
    critic_cmd.add_argument("--session-id", type=str, default="session_001")
    critic_cmd.add_argument("--epoch", type=int, default=1)
    reflexive_cmd = commands.add_parser("reflexive-orchestrate")
    reflexive_cmd.add_argument("--spec", type=Path, required=True)
    reflexive_cmd.add_argument("--output", type=Path, required=False)
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
        elif args.command == "dataset-profile":
            from .autonomous_rescue import profile_dataset
            prof = profile_dataset(args.csv, target_column=args.target, id_column=args.id_col)
            result = prof.to_dict()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "eda-forensics":
            from .eda_agent import eda_forensics_suite
            dossier = eda_forensics_suite(
                args.dataset,
                problem_statement=args.problem_statement,
                explicit_target=args.target,
                output_dir=args.output_dir,
                dataset_id=args.dataset.stem,
            )
            if getattr(args, "db_path", None) and args.db_path:
                from .autonomous_rescue import persist_dataset_topology_contract
                persist_dataset_topology_contract(
                    db_path=args.db_path,
                    contract=dossier.topology.to_dict() if hasattr(dossier.topology, "to_dict") else dossier.topology,
                    project_id=getattr(args, "project_id", "default"),
                    dataset_id=args.dataset.stem,
                )
            result = dossier.to_dict()
            result["status"] = "completed"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "candidate-package-bundle":
            from .autonomous_rescue import bundle_candidate
            spec_data = json.loads(args.spec.read_text())
            result = bundle_candidate(
                candidate_dir=spec_data["candidate_dir"],
                output_bundle_dir=spec_data["output_bundle_dir"],
                dataset_path=spec_data["dataset_path"],
                test_access_id=spec_data["test_access_id"],
                project_id=spec_data["project_id"],
                candidate_id=spec_data["candidate_id"],
                task_type=spec_data.get("task_type", "binary_classification"),
                target_column=spec_data.get("target_column", "target"),
                primary_metric=spec_data.get("primary_metric", "balanced_accuracy"),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "rescue-orchestrate":
            from .autonomous_rescue import AutonomousRescueEngine
            spec_data = json.loads(args.spec.read_text())
            engine = AutonomousRescueEngine(
                workspace_root=spec_data["workspace_root"],
                project_id=spec_data["project_id"],
                session_id=spec_data["session_id"],
            )
            result = engine.execute_closed_loop(
                dataset_path=spec_data["dataset_path"],
                target_column=spec_data.get("target_column", "target"),
                simulate_failure_mode=spec_data.get("simulate_failure_mode", "shape_mismatch"),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "pareto-ensemble":
            from .pareto_ensemble import ParetoEnsembleSynthesizer, ParetoEnsembleSpec
            spec_data = json.loads(args.spec.read_text())
            spec = ParetoEnsembleSpec.from_dict(spec_data)
            synth = ParetoEnsembleSynthesizer(output_root=spec_data.get("output_root"))
            res = synth.synthesize_ensemble(spec)
            result = res.to_dict()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
        elif args.command == "prospective-benchmark":
            from .benchmarks import run_prospective_benchmark, save_benchmark_report
            spec_data = json.loads(args.spec.read_text())
            report = run_prospective_benchmark(
                task=spec_data.get("task", "binary_classification"),
                metric=spec_data.get("metric", "balanced_accuracy"),
                direction=spec_data.get("direction", "maximize"),
                seeds=spec_data.get("seeds"),
                max_trials_per_strategy=spec_data.get("max_trials_per_strategy", 6),
                wall_budget_ms=spec_data.get("wall_budget_ms", 15000.0),
                dataset_name=spec_data.get("dataset_name", "synthetic"),
                max_memory_mb=spec_data.get("max_memory_mb", 512.0),
                max_threads=spec_data.get("max_threads", 2),
                baselines=spec_data.get("baselines"),
                include_failing_trial=spec_data.get("include_failing_trial", True),
                offline_only=spec_data.get("offline_only", True),
            )
            if args.output:
                save_benchmark_report(report, args.output)
            result = json.loads(json.dumps(report, default=lambda o: getattr(o, "__dict__", str(o))))
        elif args.command == "ablation-sprint":
            import pandas as pd
            from .fe_ablation import ablate_feature_hypothesis, build_cv_splitter
            from .autonomous_rescue import persist_feature_ablation
            spec_data = json.loads(args.spec.read_text(encoding="utf-8"))
            dataset_path = Path(spec_data["dataset_path"])
            topology = spec_data.get("topology")
            if not topology and spec_data.get("topology_path"):
                topology = json.loads(Path(spec_data["topology_path"]).read_text(encoding="utf-8"))
            if not topology:
                from .dirty_sanitizer import sanitize_dataframe
                from .semantic_sieve import analyze_and_lock_topology
                df_raw = pd.read_csv(dataset_path)
                san_res = sanitize_dataframe(df_raw)
                clean_df = san_res.df
                coercions = san_res.coercions
                target_col = spec_data.get("target_column")
                topology = analyze_and_lock_topology(clean_df, explicit_target=target_col, coercions=coercions)
            cand_fe = spec_data.get("candidate_features_py")
            base_fe = spec_data.get("base_features_py")
            hyp_id = spec_data.get("hypothesis_id", "hyp_ablation_sprint")
            db_path = spec_data.get("db_path")
            project_id = spec_data.get("project_id", "default")
            session_id = spec_data.get("session_id", "session_001")
            epoch = spec_data.get("epoch", 1)

            receipt = ablate_feature_hypothesis(
                hypothesis_id=hyp_id,
                candidate_features_py=cand_fe,
                base_features_py=base_fe,
                topology=topology,
                dataset_path=dataset_path,
                session_id=session_id,
                project_id=project_id,
                epoch=epoch,
            )
            if db_path:
                persist_feature_ablation(
                    db_path=db_path,
                    ablation=receipt,
                    project_id=project_id,
                    session_id=session_id,
                    epoch=epoch,
                )
            result = receipt
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        elif args.command == "model-portfolio-eval":
            import numpy as np
            import pandas as pd
            from .fe_ablation import build_cv_splitter
            from .model_zoo import train_and_diagnose_portfolio, HAS_LIGHTGBM, HAS_CATBOOST
            spec_data = json.loads(args.spec.read_text(encoding="utf-8"))
            dataset_path = Path(spec_data["dataset_path"])
            topology = spec_data.get("topology")
            if not topology and spec_data.get("topology_path"):
                topology = json.loads(Path(spec_data["topology_path"]).read_text(encoding="utf-8"))
            df = pd.read_csv(dataset_path)
            target_col = spec_data.get("target_column") or (topology.get("target_column") if topology else None)
            if not target_col:
                from .semantic_sieve import resolve_target
                target_col = resolve_target(df)
            task_type = spec_data.get("task_type") or (topology.get("task_type") if topology else "binary_classification")
            if not topology:
                topology = {"target_column": target_col, "task_type": task_type, "cv_scheme": "stratified_kfold", "cv_splits": 5}

            splitter, group_col, _ = build_cv_splitter(topology, df=df)
            y = df[target_col].to_numpy()
            feature_cols = [c for c in df.columns if c != target_col and c not in topology.get("id_columns", [])]
            X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0.0)
            if X.shape[1] == 0:
                X = pd.DataFrame({"dummy": np.ones(len(df))})

            families = spec_data.get("families")
            if not families:
                families = ["lgbm" if HAS_LIGHTGBM else "hist_gbdt", "catboost" if HAS_CATBOOST else "hist_gbdt", "penalized_linear"]
            families = list(dict.fromkeys(families))

            groups = df[group_col].to_numpy() if group_col and group_col in df.columns else None
            portfolio_res = train_and_diagnose_portfolio(
                families=families,
                X=X,
                y=y,
                cv_splitter=splitter,
                task_type=task_type,
                groups=groups,
                feature_names=list(X.columns),
            )

            portfolio_dict = {}
            for fam, res in portfolio_res.items():
                portfolio_dict[fam] = {
                    "family": res.family,
                    "effective_family": res.effective_family,
                    "mean_cv_score": res.mean_cv_score,
                    "mean_train_score": res.mean_train_score,
                    "train_val_gap": res.train_val_gap,
                    "cv_scores": res.cv_scores,
                    "train_scores": res.train_scores,
                    "feature_importances": res.feature_importances.tolist() if hasattr(res.feature_importances, "tolist") else list(res.feature_importances),
                    "feature_names": res.feature_names,
                    "worst_slices": [s.to_dict() for s in getattr(res, "worst_slices", [])] if hasattr(res, "worst_slices") else [],
                }
            result = {
                "status": "evaluated",
                "families_evaluated": list(portfolio_dict.keys()),
                "portfolio": portfolio_dict,
            }
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        elif args.command == "critic-audit":
            from .critic_engine import critic_audit_candidate
            cand_data = json.loads(args.candidate.read_text(encoding="utf-8"))
            topo_data = json.loads(args.topology.read_text(encoding="utf-8"))
            forensics_data = json.loads(args.forensics.read_text(encoding="utf-8")) if args.forensics and args.forensics.is_file() else None
            dossier_data = json.loads(args.dossier.read_text(encoding="utf-8")) if args.dossier and args.dossier.is_file() else None
            critique = critic_audit_candidate(
                candidate=cand_data,
                topology=topo_data,
                error_forensics=forensics_data,
                eda_dossier=dossier_data,
                db_path=getattr(args, "db_path", None),
                project_id=getattr(args, "project_id", "default"),
                session_id=getattr(args, "session_id", "session_001"),
                epoch=getattr(args, "epoch", 1),
            )
            result = critique.to_dict()
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        elif args.command == "reflexive-orchestrate":
            from .autonomous_rescue import AutonomousRescueEngine
            spec_data = json.loads(args.spec.read_text(encoding="utf-8"))
            engine = AutonomousRescueEngine(
                workspace_root=spec_data.get("workspace_root"),
                project_id=spec_data.get("project_id", "default"),
                session_id=spec_data.get("session_id", "session_001"),
                spec=spec_data,
            )
            result = engine.execute_reflexive_loop(
                dataset_path=spec_data["dataset_path"],
                target_column=spec_data.get("target_column"),
                max_epochs=spec_data.get("max_epochs", 2),
                db_path=spec_data.get("db_path"),
                inject_naive_first_epoch=spec_data.get("inject_naive_first_epoch", False),
                bundle_output_dir=spec_data.get("bundle_output_dir"),
                project_id=spec_data.get("project_id", "default"),
                session_id=spec_data.get("session_id", "session_001"),
            )
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                clean_result = {k: v for k, v in result.items() if k != "pipeline"}
                args.output.write_text(json.dumps(clean_result, indent=2, sort_keys=True, default=str), encoding="utf-8")
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
