"""Offline end-to-end checks for accepted candidate bundle portability."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.autonomous_rescue import bundle_candidate
from mlrescurix_worker.scientific.candidate_pipeline import CandidatePipeline, identity


@pytest.mark.parametrize("task_type", ["binary_classification", "regression"])
def test_candidate_bundle_runs_in_clean_subprocess_with_raw_categoricals(tmp_path, task_type):
    rng = np.random.default_rng(9)
    n = 80
    category = np.where(np.arange(n) % 2, "west", "east")
    x = rng.normal(size=n)
    if task_type == "regression":
        target = 2.5 * x + np.where(category == "west", 4.0, -4.0) + rng.normal(0, .1, n)
        metric = "rmse"
    else:
        target = ((x + np.where(category == "west", 2., -2.)) > 0).astype(int)
        metric = "balanced_accuracy"
    frame = pd.DataFrame({"x": x, "category": category, "target": target})
    csv = tmp_path / "dev.csv"
    frame.to_csv(csv, index=False)
    spec = {"families": ["penalized_linear"], "weights": [1.0], "transforms": [], "seed": 42}
    model = CandidatePipeline(spec, "target", task_type).fit(frame)
    accepted_eval = {"metric_name": metric, "cv_score": 0.9, "candidate_identity": identity(spec), "cv_scores": [0.9, 0.9], "fold_receipts": []}
    bundle = tmp_path / "bundle"
    receipt = bundle_candidate(
        candidate_dir=tmp_path / "candidate", output_bundle_dir=bundle,
        dataset_path=csv, test_access_id="dev-only", project_id="project",
        candidate_id="accepted-1", task_type=task_type, target_column="target",
        primary_metric=metric, pipeline=model,
        raw_holdout_records=frame.drop(columns="target").head(5).to_dict(orient="records"),
        accepted_evaluation=accepted_eval, accepted_candidate_identity=identity(spec),
    )

    manifest = json.loads((bundle / "manifest.json").read_text())
    from jsonschema import Draft202012Validator
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[2] / "contracts/v2/candidate_bundle.schema.json").read_text())
    Draft202012Validator(schema).validate(manifest)
    scorecard = json.loads((bundle / "scorecard.json").read_text())
    final = json.loads((bundle / "final.json").read_text())
    from importlib.metadata import version
    assert f"scikit-learn=={version('scikit-learn')}" in (bundle / "inference.lock").read_text()
    assert "scipy==" in (bundle / "inference.lock").read_text()
    assert manifest["candidate_identity"] == identity(spec)
    assert scorecard["candidate_identity"] == identity(spec)
    assert scorecard["evaluation_scope"] == "accepted_development_cross_validation"
    assert scorecard["primary_metric"] == pytest.approx(.9)
    assert scorecard["constraints"] == {}
    assert final["evaluation_scope"] == "accepted_development_cross_validation"
    assert final["development_metrics"][metric] == pytest.approx(.9)
    assert receipt["holdout_primary_metric"] is None
    assert (bundle / "mlrescurix_worker/scientific/__init__.py").is_file()

    input_path = bundle / "examples_input.json"
    output_path = bundle / "subprocess_output.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, str(bundle / "inference.py"), "--input", str(input_path), "--output", str(output_path)],
        cwd=tmp_path, env=env, check=True, capture_output=True, text=True,
    )
    output = json.loads(output_path.read_text())
    expected = json.loads((bundle / "examples_expected.json").read_text())
    assert output["prediction"] == pytest.approx(expected["prediction"])
    if task_type == "binary_classification":
        np.testing.assert_allclose(output["probabilities"], expected["probabilities"])
        assert output["classes"] == expected["classes"]
    else:
        assert "probabilities" not in output
