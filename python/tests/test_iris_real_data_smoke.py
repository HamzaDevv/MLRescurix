"""Local real-data smoke test; this is not supervisor or live-provider evidence."""
import json
import os
import subprocess
import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from mlrescurix_worker.custom.contracts import ResourceRequestContract
from mlrescurix_worker.custom.delivery import CustomBundlePackager
from mlrescurix_worker.custom.tabular_candidate import TabularCandidatePlugin


def test_iris_training_checkpoint_and_packaged_predictions(tmp_path):
    iris = load_iris(as_frame=True)
    train, validation = train_test_split(
        np.arange(len(iris.target)), test_size=0.2, random_state=42,
        stratify=iris.target,
    )
    x_train, x_validation = iris.data.iloc[train], iris.data.iloc[validation]
    y_train, y_validation = iris.target.iloc[train], iris.target.iloc[validation]
    candidate = TabularCandidatePlugin(max_iter=60, seed=42)
    candidate.prepare({"features": list(iris.feature_names)}, ResourceRequestContract())
    state = candidate.train((x_train, y_train))
    predicted = candidate.predict(x_validation)
    accuracy = accuracy_score(y_validation, predicted)
    assert accuracy >= 0.8

    candidate.save_checkpoint(tmp_path / "checkpoint", state.step, state)
    restored = TabularCandidatePlugin()
    restored.load_checkpoint(tmp_path / "checkpoint")
    np.testing.assert_array_equal(predicted, restored.predict(x_validation))

    weights = tmp_path / "weights.joblib"
    candidate.save_weights(weights)
    bundle = tmp_path / "bundle"
    CustomBundlePackager().package(
        candidate=candidate, output_bundle_dir=bundle, target_column="species",
        task_type="multiclass_classification", primary_metric="accuracy",
        primary_metric_score=float(accuracy), metric_direction="maximize",
        feature_names=list(iris.feature_names),
        raw_example_records=x_train.head(3).to_dict(orient="records"),
        source_weights_path=weights,
    )
    input_path, output_path = tmp_path / "input.json", tmp_path / "output.json"
    input_path.write_text(json.dumps(x_validation.to_dict(orient="records")))
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [sys.executable, str(bundle / "inference.py"), "--input", str(input_path),
         "--output", str(output_path)],
        env=env, capture_output=True, text=True, timeout=30, check=True,
    )
    output = json.loads(output_path.read_text())
    np.testing.assert_array_equal(predicted, output["prediction"])
    np.testing.assert_allclose(candidate.predict_proba(x_validation), output["probabilities"])
    print(f"Iris: train={len(train)}, validation={len(validation)}, accuracy={accuracy:.4f}; checkpoint and bundle parity passed")
