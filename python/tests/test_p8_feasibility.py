import json
import tempfile
import unittest
from pathlib import Path
import jsonschema
import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.feasibility import (
    HardwareProfile,
    DatasetMetadata,
    extract_dataset_metadata,
    estimate_model_complexity,
    run_capped_micro_pilot,
    suggest_local_alternatives,
    generate_feasibility_report,
    negotiate_execution_plan,
)

ROOT = Path(__file__).resolve().parents[2]


class TestP8ComputeFeasibility(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p8_test_"))
        # Load JSON Schemas Draft 2020-12
        self.hw_schema = json.loads((ROOT / "contracts/v2/hardware_profile.schema.json").read_text())
        self.meta_schema = json.loads((ROOT / "contracts/v2/dataset_metadata.schema.json").read_text())
        self.report_schema = json.loads((ROOT / "contracts/v2/feasibility_report.schema.json").read_text())
        self.plan_schema = json.loads((ROOT / "contracts/v2/execution_plan.schema.json").read_text())

    def test_p8_1_schemas_and_artifact_validation(self):
        """P8.1: Verify strict typed schemas for HardwareProfile, DatasetMetadata, FeasibilityReport, and ExecutionPlan."""
        hw = HardwareProfile(
            schema_version="v2",
            effective_cpus=4,
            effective_memory_bytes=8 * 1024 * 1024 * 1024,
            effective_disk_bytes=50 * 1024 * 1024 * 1024,
        )
        jsonschema.validate(instance=hw.to_dict(), schema=self.hw_schema)

        # Synthetic tabular dataframe with mix of numerical and high-cardinality categoricals
        df = pd.DataFrame({
            "num1": np.random.randn(200),
            "num2": np.random.randn(200),
            "cat1": ["cat_a", "cat_b", "cat_c", "cat_d"] * 50,
            "cat2": [f"val_{i % 10}" for i in range(200)],
            "target": [0, 1] * 100,
        })
        csv_path = self.tmp / "test_data.csv"
        df.to_csv(csv_path, index=False)

        meta = extract_dataset_metadata(csv_path, dataset_id="ds_p8_test", task_type="binary", target_column="target")
        self.assertEqual(meta.row_count, 200)
        self.assertEqual(meta.feature_count, 4)
        self.assertIn("cat1", meta.categorical_columns)
        self.assertIn("cat2", meta.categorical_columns)
        self.assertIn("num1", meta.numeric_columns)
        self.assertGreater(meta.transformed_expansion_factor, 1.0)
        jsonschema.validate(instance=meta.to_dict(), schema=self.meta_schema)

        report = generate_feasibility_report(
            project_id="proj_p8_01",
            hardware_profile=hw,
            dataset_metadata=meta,
            model_family="boosting",
            cv_folds=3,
            run_pilot=True,
            pilot_data=df,
        )
        jsonschema.validate(instance=report, schema=self.report_schema)
        self.assertIn(report["verdict"], ["locally_feasible", "feasible_with_adaptation", "requires_external_compute"])

        plan = negotiate_execution_plan(
            report=report,
            accepted_adaptations=["quantization", "progressive_fidelity"],
            custom_constraints={"time_budget_ms": 1800000},
            target_preset="STANDARD",
        )
        jsonschema.validate(instance=plan, schema=self.plan_schema)
        self.assertTrue(plan["locked_split_protocol_preserved"])
        self.assertEqual(plan["selected_preset"], "STANDARD")

    def test_p8_2_analytical_estimators_and_complexity_bounds(self):
        """P8.2: Verify model family-specific analytical complexity bounds (linear, boosting, tree, neural)."""
        meta = DatasetMetadata(
            schema_version="v2",
            dataset_id="ds_analytical",
            row_count=10000,
            feature_count=20,
            raw_bytes=2000000,
            task_type="binary",
            categorical_columns=["c1", "c2"],
            numeric_columns=[f"n{i}" for i in range(18)],
            estimated_cardinality={"c1": 50, "c2": 100},
            estimated_expanded_feature_count=168,
            memory_dense_bytes=10000 * 168 * 8,
            transformed_expansion_factor=8.4,
            cv_folds=5,
        )

        for family in ["linear", "boosting", "tree", "neural", "custom"]:
            runtime_bounds, mem_bounds, assumptions = estimate_model_complexity(
                model_family=family,
                meta=meta,
                cv_folds=5,
                include_final_refit=True,
                concurrent_worker_leases=2,
            )
            # Check range bounds
            self.assertGreater(runtime_bounds["expected"], runtime_bounds["min"])
            self.assertGreater(runtime_bounds["max"], runtime_bounds["expected"])
            self.assertGreater(mem_bounds["expected"], mem_bounds["min"])
            self.assertGreater(mem_bounds["max"], mem_bounds["expected"])

            # Check documented assumptions
            self.assertTrue(any("Cross-validation" in a for a in assumptions))
            self.assertTrue(any("Feature transformation" in a for a in assumptions))
            self.assertTrue(any("Concurrent worker" in a for a in assumptions))

    def test_p8_2_capped_micro_pilots_respect_safety_guards(self):
        """P8.2: Micro-pilots strictly respect tight resource and time caps without runaway execution."""
        df = pd.DataFrame({
            "x1": np.random.randn(500),
            "x2": np.random.randn(500),
            "y": np.random.randint(0, 2, size=500),
        })

        # Succeeded capped pilot
        res_ok = run_capped_micro_pilot(
            df,
            model_family="linear",
            task_type="binary",
            time_cap_ms=3000,
            mem_cap_bytes=256 * 1024 * 1024,
            pilot_n=100,
        )
        self.assertEqual(res_ok["status"], "succeeded")
        self.assertLessEqual(res_ok["rows_probed"], 100)
        self.assertGreater(res_ok["elapsed_ms"], 0)
        self.assertLess(res_ok["elapsed_ms"], 3000)

        # Micro-pilot time cap violation is safely intercepted
        res_timeout = run_capped_micro_pilot(
            df,
            model_family="tree",
            task_type="binary",
            time_cap_ms=0,  # 0 ms forced timeout
            mem_cap_bytes=256 * 1024 * 1024,
            pilot_n=100,
        )
        self.assertEqual(res_timeout["status"], "timed_out")
        self.assertIn("exceeded strict time cap", res_timeout["error"])

        # Micro-pilot memory cap violation is safely intercepted
        res_oom = run_capped_micro_pilot(
            df,
            model_family="boosting",
            task_type="binary",
            time_cap_ms=5000,
            mem_cap_bytes=100,  # Unrealistic low memory cap
            pilot_n=100,
        )
        self.assertEqual(res_oom["status"], "memory_exceeded")
        self.assertIn("exceeded memory cap", res_oom["error"])

    def test_p8_3_transparent_multi_tier_feasibility_verdicts(self):
        """P8.3: Verify explicit multi-tier verdicts (locally_feasible, feasible_with_adaptation, requires_external_compute)."""
        # Case 1: Small dataset on large hardware -> locally_feasible
        hw_large = HardwareProfile(
            available={"available_cpus": 8, "available_memory_bytes": 16 * 1024 * 1024 * 1024, "available_disk_bytes": 50 * 1024 * 1024 * 1024, "memory_certification_supported": True, "accelerators": []}
        )
        meta_small = DatasetMetadata(
            row_count=500, feature_count=5, estimated_expanded_feature_count=5, memory_dense_bytes=20000, transformed_expansion_factor=1.0
        )
        rep1 = generate_feasibility_report("p1", hw_large, meta_small, "linear", run_pilot=False)
        self.assertEqual(rep1["verdict"], "locally_feasible")

        # Case 2: Moderate dataset on tight memory -> feasible_with_adaptation
        hw_tight = HardwareProfile(
            available={"available_cpus": 2, "available_memory_bytes": 200 * 1024 * 1024, "available_disk_bytes": 2 * 1024 * 1024 * 1024, "memory_certification_supported": False, "accelerators": []}
        )
        meta_mod = DatasetMetadata(
            row_count=20000, feature_count=50, estimated_expanded_feature_count=100, memory_dense_bytes=16000000, transformed_expansion_factor=2.0
        )
        rep2 = generate_feasibility_report("p2", hw_tight, meta_mod, "tree", run_pilot=False)
        self.assertEqual(rep2["verdict"], "feasible_with_adaptation")
        self.assertGreater(len(rep2["adaptation_proposals"]), 0)

        # Case 3: Massive dataset on tiny memory -> requires_external_compute
        hw_tiny = HardwareProfile(
            available={"available_cpus": 1, "available_memory_bytes": 100 * 1024 * 1024, "available_disk_bytes": 500 * 1024 * 1024, "memory_certification_supported": False, "accelerators": []}
        )
        meta_huge = DatasetMetadata(
            row_count=2000000, feature_count=200, estimated_expanded_feature_count=1000, memory_dense_bytes=16000000000, transformed_expansion_factor=5.0
        )
        rep3 = generate_feasibility_report("p3", hw_tiny, meta_huge, "neural", run_pilot=False)
        self.assertEqual(rep3["verdict"], "requires_external_compute")

    def test_p8_4_evidence_backed_local_alternatives_no_silent_degradation(self):
        """P8.4: Concrete alternatives suggested without quiet semantic degradation."""
        meta = DatasetMetadata(
            row_count=30000,
            feature_count=25,
            categorical_columns=["region", "device"],
            estimated_cardinality={"region": 50, "device": 10},
            estimated_expanded_feature_count=85,
            memory_dense_bytes=20400000,
            transformed_expansion_factor=3.4,
        )
        alternatives = suggest_local_alternatives("boosting", meta, "feasible_with_adaptation")
        types = [a["adaptation_type"] for a in alternatives]

        self.assertIn("quantization", types)
        self.assertIn("sparse_linear", types)
        self.assertIn("progressive_fidelity", types)
        self.assertIn("downsampled_screening", types)
        self.assertIn("smaller_architecture", types)

        # Confirm documentation of quality impact on every alternative
        for alt in alternatives:
            self.assertIn("quality_impact", alt)
            self.assertGreater(len(alt["quality_impact"]), 10)
            self.assertGreaterEqual(alt["predicted_speedup_factor"], 1.0)
            self.assertGreaterEqual(alt["predicted_memory_reduction_factor"], 1.0)

    def test_p8_5_constraint_negotiation_and_locked_split_preservation(self):
        """P8.5: Deterministic constraint negotiation preserving locked evaluation splits."""
        hw = HardwareProfile()
        meta = DatasetMetadata(row_count=5000, feature_count=15)
        report = generate_feasibility_report("p5", hw, meta, "tree", run_pilot=False)

        plan = negotiate_execution_plan(
            report,
            accepted_adaptations=["quantization", "smaller_architecture"],
            custom_constraints={"max_latency_p95_ms": 50, "target_f1": 0.85},
            target_preset="MICRO",
        )
        self.assertEqual(plan["selected_preset"], "MICRO")
        self.assertEqual(plan["allocated_threads"], 1)
        self.assertEqual(plan["allocated_memory_bytes"], 512 * 1024 * 1024)
        self.assertTrue(plan["locked_split_protocol_preserved"], "Evaluation splits must remain strictly locked!")
        self.assertEqual(len(plan["downscaling_applied"]), 2)
        applied_techs = [d["technique"] for d in plan["downscaling_applied"]]
        self.assertIn("quantization", applied_techs)
        self.assertIn("smaller_architecture", applied_techs)


if __name__ == "__main__":
    unittest.main()
