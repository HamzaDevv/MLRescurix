"""Scientific breadth and release correctness verification on a non-Amazon task.

Demonstrates:
1. Genuinely trained, materially different model families (penalized_linear, hist_gbdt, neural_tabular).
2. Comparable fold-level cross-validation on a non-Amazon dataset (Wisconsin Diagnostic Breast Cancer).
3. Genuine ablation / negative result recording in loop state without deletion.
4. Model Creation Journey / Apex tracking and plateau detection.
5. Reloading recorded history and exporting a handoff.
6. Packaging accepted candidate, verifying bundle integrity, and replaying inference in a clean subprocess.

This fixture scripts the sequence of actions on non-Amazon data to verify the generalized multi-model pipeline through candidate packaging, bundle integrity verification, and replay.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold

from mlrescurix_worker.scientific.autonomous_loop import AutonomousAgentLoop
from mlrescurix_worker.scientific.critic_engine import CriticEngine
from mlrescurix_worker.scientific.model_zoo import HAS_TORCH


class MockBudgetAuthority:
    """Deterministic local budget authority."""

    def __init__(self, campaign_ceiling: str = "1.00", per_run_ceiling: str = "0.20"):
        self.campaign_ceiling = Decimal(campaign_ceiling)
        self.per_run_ceiling = Decimal(per_run_ceiling)
        self.billed = Decimal("0.00")
        self.reserved = Decimal("0.00")
        self.reservations: List[str] = []
        self.settlements: List[tuple[str, Decimal]] = []

    def reserve(self, run_id: str, request_id: str, amount_usd: float | Decimal) -> Decimal:
        amt = Decimal(str(amount_usd))
        self.reserved += amt
        self.reservations.append(request_id)
        return amt

    def settle(self, run_id: str, request_id: str, actual_billed_usd: float | Decimal) -> None:
        act = Decimal(str(actual_billed_usd))
        self.billed += act
        self.reserved = max(Decimal("0.00"), self.reserved - amt if (amt := Decimal("0.005")) else Decimal("0.00"))
        self.settlements.append((request_id, act))

    def mark_uncertain(self, run_id: str, request_id: str, reserved_usd: float | Decimal) -> None:
        pass

    def get_status(self) -> Dict[str, Any]:
        return {
            "campaign_ceiling_usd": str(self.campaign_ceiling),
            "remaining_campaign_usd": str(self.campaign_ceiling - self.billed),
            "total_billed_usd": str(self.billed),
        }


def test_scientific_breadth_multimodel_non_amazon(tmp_path: Path):
    # 1. Prepare genuine non-Amazon dataset (Breast Cancer: 569 rows, 30 features, binary)
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.data.copy()
    df["diagnosis"] = cancer.target
    data_csv = tmp_path / "breast_cancer.csv"
    df.to_csv(data_csv, index=False)

    # 2. Pinned 3-fold cross validation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    locked_folds = list(skf.split(df, df["diagnosis"]))

    ws_dir = tmp_path / "workspace_non_amazon"
    authority = MockBudgetAuthority()

    # Conforming autonomous actions demonstrating genuine multi-model families
    actions = [
        # Turn 0: Linear penalized model (logistic regression with robust scaling)
        {
            "action_type": "EVALUATE_PORTFOLIO",
            "thought_rationale": "Initial linear baseline to measure linear separability",
            "parameters": {
                "families": ["penalized_linear"],
            },
        },
        # Turn 1: Gradient boosted trees (hist_gbdt)
        {
            "action_type": "EVALUATE_PORTFOLIO",
            "thought_rationale": "Nonlinear gradient-boosted decision trees to capture feature interactions",
            "parameters": {
                "families": ["hist_gbdt"],
            },
        },
        # Turn 2: Neural tabular architecture (PyTorch ResNet MLP)
        {
            "action_type": "EVALUATE_PORTFOLIO",
            "thought_rationale": "Neural representation learning with residual skip connections",
            "parameters": {
                "families": ["neural_tabular" if HAS_TORCH else "lgbm"],
            },
        },
        # Turn 3: Feature transformation hypothesis (apply skew transform)
        {
            "action_type": "HYPOTHESIS_TRANSFORM",
            "thought_rationale": "Test log transform hypothesis on skewed feature",
            "parameters": {
                "action": "APPLY_SKEW_TRANSFORM",
                "parameters": {
                    "columns": ["mean radius"],
                    "method": "log1p",
                },
            },
        },
        # Turn 4: Ensemble blend combining diverse tree and linear families
        {
            "action_type": "BLEND_AND_AUDIT",
            "thought_rationale": "Ensemble blend combining diverse tree and linear representations",
            "parameters": {
                "families": ["hist_gbdt", "penalized_linear"],
                "weights": [0.7, 0.3],
            },
        },
        # Turn 5: Finalize and stop
        {
            "action_type": "FINALIZE_AND_STOP",
            "thought_rationale": "Completed multi-family exploration across linear, tree, and neural models",
            "parameters": {
                "rationale": "All candidate families evaluated and verified under cross-validation",
            },
        },
    ]

    action_iter = iter(actions)

    def mock_agent_provider(messages: List[Dict[str, str]], max_tokens: int) -> Dict[str, Any]:
        nxt = next(action_iter)
        return {
            "content": json.dumps(nxt),
            "cost_usd": "0.001",
            "usage": {"prompt_tokens": 150, "completion_tokens": 50},
        }

    loop = AutonomousAgentLoop(
        workspace_root=ws_dir,
        project_id="scientific_breast_cancer_proj",
        session_id="run_bc_001",
        campaign_authority=authority,
        dataset_path=data_csv,
        target_column="diagnosis",
        task_type="binary_classification",
        metric_name="balanced_accuracy",
        metric_direction="maximize",
        locked_folds=locked_folds,
        provider_caller=mock_agent_provider,
        provider_is_fixture=True,
        max_turns=10,
    )

    # Execute autonomous search loop
    run_result = loop.run()
    assert run_result["stop_reason"] == "FINALIZE_AND_STOP"

    # 3. Verify that multiple materially different model families were genuinely evaluated
    state = loop.state
    iterations = state.get("iterations", [])
    assert len(iterations) >= 3

    families_evaluated = set()
    for it in iterations:
        for f in it.get("spec", {}).get("families", []):
            families_evaluated.add(f)

    assert "penalized_linear" in families_evaluated
    assert "hist_gbdt" in families_evaluated
    if HAS_TORCH:
        assert "neural_tabular" in families_evaluated

    # 4. Verify negative results / ablations are preserved in state
    neg_results = state.get("negative_results", [])
    assert len(neg_results) >= 1
    # Check negative result metadata
    ablation_neg = neg_results[-1]
    assert "candidate_id" in ablation_neg
    assert "relative_gain" in ablation_neg
    assert ablation_neg["reason"] == "metric_regression_or_no_gain"

    # 5. Verify apex tracking
    assert state.get("latest_apex_iteration") is not None
    assert state.get("selected_apex_iteration") is not None
    assert len(state.get("apex_trajectory", [])) >= 1

    # 6. Verify crash recovery and state reload
    reloaded_loop = AutonomousAgentLoop(
        workspace_root=ws_dir,
        project_id="scientific_breast_cancer_proj",
        session_id="run_bc_001",
        campaign_authority=authority,
        dataset_path=data_csv,
        target_column="diagnosis",
        task_type="binary_classification",
        metric_name="balanced_accuracy",
        metric_direction="maximize",
        locked_folds=locked_folds,
        provider_caller=mock_agent_provider,
        provider_is_fixture=True,
    )
    assert len(reloaded_loop.state["iterations"]) == len(state["iterations"])
    assert len(reloaded_loop.state["negative_results"]) == len(neg_results)
    assert reloaded_loop.state["selected_apex_iteration"] == state["selected_apex_iteration"]

    # 7. Verify handoff export
    handoff = loop.export_handoff()
    assert handoff["stop_reason"] == "FINALIZE_AND_STOP"
    assert handoff["accepted_candidate_id"] == state["accepted_candidate_id"]
    assert handoff["best_cv_score"] is not None
    assert (ws_dir / "handoff.json").exists()
    assert (ws_dir / "HANDOFF.md").exists()

    # 8. Genuine Candidate Packaging & Bundle Integrity Verification
    import subprocess
    from mlrescurix_worker.scientific.benchmark_contracts import package_accepted_candidate

    bundle_dir = ws_dir / "bundle"
    candidate_dir = ws_dir / "candidate"
    ok, verified, reason = package_accepted_candidate(
        loop,
        loop.state,
        loop.raw_df,
        candidate_dir=candidate_dir,
        bundle_dir=bundle_dir,
        dataset_path=data_csv,
        test_access_id=f"delivery_{loop.session_id}",
        project_id=loop.project_id,
        task_type=loop.task_type,
        target_column=loop.target_column,
        primary_metric=loop.metric_name,
    )
    assert ok is True, f"Packaging failed: {reason}"
    assert verified is True, f"Verification failed: {reason}"
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "inference.py").exists()

    # 9. Clean Subprocess Inference Replay
    input_file = bundle_dir / "examples_input.json"
    assert input_file.exists(), "examples_input.json must exist in bundle"
    out_file = tmp_path / "replay_predictions.json"

    replay_proc = subprocess.run(
        ["python3", "-B", "inference.py", "--input", str(input_file), "--output", str(out_file)],
        cwd=str(bundle_dir),
        capture_output=True,
        text=True,
    )
    assert replay_proc.returncode == 0, f"Inference replay failed: {replay_proc.stderr}"
    assert out_file.exists()
    predictions_data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "prediction" in predictions_data or "predictions" in predictions_data
    preds = predictions_data.get("prediction", predictions_data.get("predictions", []))
    assert len(preds) > 0
    expected_file = bundle_dir / "examples_expected.json"
    assert expected_file.exists(), "examples_expected.json must exist in bundle"
    expected_data = json.loads(expected_file.read_text(encoding="utf-8"))
    expected_preds = expected_data.get("prediction", expected_data.get("predictions", []))
    assert len(preds) == len(expected_preds)
    import numpy as np
    assert preds == expected_preds or np.allclose(preds, expected_preds, atol=1e-5), (
        "Subprocess replay predictions must match examples_expected.json"
    )
