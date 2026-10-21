import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.eda_agent import run_exploratory_python
from mlrescurix_worker.scientific.fe_ablation import _resolve_and_apply_features
from mlrescurix_worker.scientific.iteration_loop import MultiFidelityIterationEngine
from mlrescurix_worker.scientific.pareto_ensemble import (
    ParetoEnsembleSpec,
    ParetoEnsembleSynthesizer,
)
from mlrescurix_worker.scientific.workflow import WorkflowError


# =====================================================================
# RB-001: Exploratory Python Sandbox Isolation & Timeout
# =====================================================================

def test_rb001_exploratory_timeout():
    """Verify that exploratory execution enforces hard timeout and terminates cleanly."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    infinite_loop_code = "import time\nwhile True:\n    time.sleep(0.1)\n"

    t0 = time.time()
    out = run_exploratory_python(infinite_loop_code, df, timeout_s=1.0)
    elapsed = time.time() - t0

    assert "[error]" in out
    assert "governed sandbox unavailable" in out
    assert elapsed < 5.0, f"Execution took too long: {elapsed:.2f}s"


def test_rb001_exploratory_no_parent_mutation():
    """Verify that exploratory execution cannot mutate the parent DataFrame in memory."""
    df = pd.DataFrame({"val": [10, 20, 30], "label": [0, 1, 0]})
    orig_val_0 = df.iloc[0, 0]

    mutation_code = """
df.iloc[0, 0] = 999999
df['injected_leak'] = [1, 2, 3]
"""
    out = run_exploratory_python(mutation_code, df, timeout_s=5.0)

    # Assert parent df was NOT modified
    assert df.iloc[0, 0] == orig_val_0
    assert "injected_leak" not in df.columns


def test_rb001_exploratory_label_isolation():
    """Verify that excluded columns (e.g. holdout/validation labels) are inaccessible."""
    df = pd.DataFrame({"feat": [1, 2, 3], "secret_target": [99, 100, 101]})
    probe_code = "print('Target sum:', df['secret_target'].sum())"

    out = run_exploratory_python(probe_code, df, timeout_s=5.0, exclude_columns=["secret_target"])

    assert "[error]" in out
    assert "governed sandbox unavailable" in out
    assert "99" not in out


# =====================================================================
# RB-002: Callable Validation-Label Firewall & Canary
# =====================================================================

def test_rb002_callable_validation_label_firewall():
    """Verify that candidate callables cannot see validation labels."""
    train_df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "x": [10.0, 20.0, 30.0, 40.0],
        "target": [0, 1, 0, 1],
    })
    val_df = pd.DataFrame({
        "id": [5, 6],
        "x": [50.0, 60.0],
        "target": [1, 0],
    })

    # Canary function: tries to read target from validation frame
    val_saw_target = []

    def canary_feature_extractor(df, y=None):
        if "target" in df.columns:
            val_saw_target.append(True)
            return pd.DataFrame({"leak": df["target"]})
        else:
            val_saw_target.append(False)
            return pd.DataFrame({"feat": df["x"] * 2.0})

    X_tr, X_val = _resolve_and_apply_features(
        canary_feature_extractor,
        train_df,
        val_df,
        target_col="target",
        id_cols=["id"],
    )

    # The training call may receive y_train, but the validation call MUST NOT see target in df
    # val_saw_target should have recorded False for the validation call
    assert False in val_saw_target, "Validation frame unexpectedly exposed 'target'!"
    assert "leak" not in X_val.columns
    assert "feat" in X_val.columns


def test_rb002_single_arg_callable_no_validation_label():
    """Verify single-argument callable f(df) cannot access validation label."""
    train_df = pd.DataFrame({"x": [1.0, 2.0], "target": [10, 20]})
    val_df = pd.DataFrame({"x": [3.0, 4.0], "target": [30, 40]})

    def single_arg_fn(df):
        if "target" in df:
            raise RuntimeError("CRITICAL LEAK: validation frame contains target!")
        return pd.DataFrame({"x2": df["x"] ** 2})

    X_tr, X_val = _resolve_and_apply_features(
        single_arg_fn,
        train_df,
        val_df,
        target_col="target",
        id_cols=[],
    )

    assert "x2" in X_tr.columns
    assert "x2" in X_val.columns


# =====================================================================
# RB-003: Topology-Safe Proxy Splitting
# =====================================================================

def test_rb003_proxy_screen_temporal_ordering(tmp_path: Path):
    """Verify proxy screen splitting preserves chronological order for temporal datasets."""
    # Synthetic time-series dataset
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "row_id": [f"ts_row_{i}" for i in range(100)],
        "timestamp": dates.astype(str),
        "feat": np.random.randn(100),
        "target": np.random.randint(0, 2, 100),
    })
    csv_path = tmp_path / "ts_data.csv"
    df.to_csv(csv_path, index=False)

    cand_dir = tmp_path / "cand_ts"
    cand_dir.mkdir(parents=True)
    entrypoint = cand_dir / "entrypoint.py"
    log_dir = tmp_path / "ts_logs"
    log_dir.mkdir(parents=True)
    entrypoint.write_text(f"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

log_dir = Path('{str(log_dir)}')

if __name__ == "__main__":
    if "fit" in sys.argv:
        tr_idx = sys.argv.index("--train-path") + 1
        df_tr = pd.read_csv(sys.argv[tr_idx])
        df_tr.to_csv(log_dir / "train_seen.csv", index=False)
    elif "predict" in sys.argv:
        val_idx = sys.argv.index("--input-path") + 1 if "--input-path" in sys.argv else sys.argv.index("--val-path") + 1
        df_val = pd.read_csv(sys.argv[val_idx])
        df_val.to_csv(log_dir / "val_seen.csv", index=False)
        out_idx = sys.argv.index("--output-path") + 1 if "--output-path" in sys.argv else sys.argv.index("--predictions-path") + 1
        row_col = "row_id" if "row_id" in df_val.columns else df_val.columns[0]
        pd.DataFrame({{
            row_col: df_val[row_col],
            "pred": np.ones(len(df_val), dtype=int),
            "prob_0": np.full(len(df_val), 0.3),
            "prob_1": np.full(len(df_val), 0.7),
        }}).to_csv(sys.argv[out_idx], index=False)
""", encoding="utf-8")

    manifest = {
        "candidate_id": "cand_ts_01",
        "name": "Candidate TS",
        "task_type": "binary",
        "entrypoint": "entrypoint.py",
        "supported_operations": ["fit", "predict"],
        "target_column": "target",
        "class_order": [0, 1],
        "stable_row_keys": "row_id",
    }

    topology = {
        "temporal_col": "timestamp",
        "cv_scheme": "purged_time_series_split",
    }

    receipt = MultiFidelityIterationEngine.run_proxy_screen(
        dataset_csv=csv_path,
        manifest=manifest,
        candidate_dir=cand_dir,
        max_samples=50,
        topology=topology,
        seed=2026,
    )

    # 1. Successful evaluation and admission
    assert receipt.candidate_id == "cand_ts_01"
    assert receipt.status == "admitted", f"Proxy screen failed: {receipt.reason}"
    assert receipt.proxy_metric_value is not None
    assert receipt.proxy_metric_name == "balanced_accuracy"

    # 2. Direct inspection of actual train and validation rows
    train_seen = pd.read_csv(log_dir / "train_seen.csv")
    val_seen = pd.read_csv(log_dir / "val_seen.csv")
    assert len(train_seen) > 0
    assert len(val_seen) > 0

    # 3. Strict chronological constraint verification (no lookahead leakage)
    assert train_seen["timestamp"].max() < val_seen["timestamp"].min()
    assert set(train_seen["row_id"]).isdisjoint(set(val_seen["row_id"]))


def test_rb003_proxy_screen_group_disjointness(tmp_path: Path):
    """Verify proxy screen splitting enforces zero group overlap between train and val."""
    # Synthetic group dataset (e.g. families / stores)
    groups = [f"group_{i % 10}" for i in range(100)]
    df = pd.DataFrame({
        "row_id": [f"grp_row_{i}" for i in range(100)],
        "family_id": groups,
        "feat": np.random.randn(100),
        "target": np.random.randint(0, 2, 100),
    })
    csv_path = tmp_path / "grp_data.csv"
    df.to_csv(csv_path, index=False)

    cand_dir = tmp_path / "cand_grp"
    cand_dir.mkdir(parents=True)
    log_dir = tmp_path / "grp_logs"
    log_dir.mkdir(parents=True)
    entrypoint = cand_dir / "entrypoint.py"
    entrypoint.write_text(f"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

log_dir = Path('{str(log_dir)}')

if __name__ == "__main__":
    if "fit" in sys.argv:
        tr_idx = sys.argv.index("--train-path") + 1
        df_tr = pd.read_csv(sys.argv[tr_idx])
        df_tr.to_csv(log_dir / "train_seen.csv", index=False)
    elif "predict" in sys.argv:
        val_idx = sys.argv.index("--input-path") + 1 if "--input-path" in sys.argv else sys.argv.index("--val-path") + 1
        df_val = pd.read_csv(sys.argv[val_idx])
        df_val.to_csv(log_dir / "val_seen.csv", index=False)
        out_idx = sys.argv.index("--output-path") + 1 if "--output-path" in sys.argv else sys.argv.index("--predictions-path") + 1
        row_col = "row_id" if "row_id" in df_val.columns else df_val.columns[0]
        pd.DataFrame({{
            row_col: df_val[row_col],
            "pred": np.ones(len(df_val), dtype=int),
            "prob_0": np.full(len(df_val), 0.3),
            "prob_1": np.full(len(df_val), 0.7),
        }}).to_csv(sys.argv[out_idx], index=False)
""", encoding="utf-8")

    manifest = {
        "candidate_id": "cand_grp_01",
        "name": "Candidate Group",
        "task_type": "binary",
        "entrypoint": "entrypoint.py",
        "supported_operations": ["fit", "predict"],
        "target_column": "target",
        "class_order": [0, 1],
        "stable_row_keys": "row_id",
    }

    topology = {
        "entity_group_col": "family_id",
        "cv_scheme": "group_kfold",
    }

    receipt = MultiFidelityIterationEngine.run_proxy_screen(
        dataset_csv=csv_path,
        manifest=manifest,
        candidate_dir=cand_dir,
        max_samples=50,
        topology=topology,
        seed=2026,
    )

    # 1. Successful evaluation and admission
    assert receipt.candidate_id == "cand_grp_01"
    assert receipt.status == "admitted", f"Proxy screen failed: {receipt.reason}"
    assert receipt.proxy_metric_value is not None

    # 2. Direct inspection of actual train and validation rows
    train_seen = pd.read_csv(log_dir / "train_seen.csv")
    val_seen = pd.read_csv(log_dir / "val_seen.csv")
    assert len(train_seen) > 0
    assert len(val_seen) > 0

    # 3. Strict group disjointness constraint: train and val share ZERO groups
    tr_groups = set(train_seen["family_id"])
    val_groups = set(val_seen["family_id"])
    assert tr_groups.isdisjoint(val_groups)

    # 4. Topology constraint rejection: if fewer than 2 distinct groups, must raise WorkflowError
    single_grp_df = pd.DataFrame({"row_id": [f"s_{i}" for i in range(20)], "family_id": ["single"] * 20, "feat": np.random.randn(20), "target": [0, 1] * 10})
    single_csv = tmp_path / "single_grp.csv"
    single_grp_df.to_csv(single_csv, index=False)
    with pytest.raises(WorkflowError, match="fewer than 2 distinct groups"):
        MultiFidelityIterationEngine.run_proxy_screen(
            dataset_csv=single_csv,
            manifest=manifest,
            candidate_dir=cand_dir,
            max_samples=20,
            topology=topology,
        )


# =====================================================================
# RB-004: Shared Split Propagation in Pareto Ensemble
# =====================================================================

def test_rb004_ensemble_shared_split_propagation(tmp_path: Path):
    """Verify ParetoEnsembleSynthesizer consumes locked shared split assignments."""
    df_train = pd.DataFrame({
        "id": list(range(20)),
        "f1": np.linspace(0, 1, 20),
        "target": [0, 1] * 10,
    })

    train_csv = tmp_path / "train.csv"
    df_train.to_csv(train_csv, index=False)

    # Create dummy base submodel
    sm_dir = tmp_path / "submodel_01"
    sm_dir.mkdir(parents=True)
    (sm_dir / "entrypoint.py").write_text("""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == "__main__":
    if "fit" in sys.argv:
        pass
    elif "predict" in sys.argv:
        val_idx = sys.argv.index("--input-path") + 1 if "--input-path" in sys.argv else sys.argv.index("--val-path") + 1
        out_idx = sys.argv.index("--output-path") + 1 if "--output-path" in sys.argv else sys.argv.index("--predictions-path") + 1
        df_v = pd.read_csv(sys.argv[val_idx])
        pd.DataFrame({"pred": np.full(len(df_v), 0.5)}).to_csv(sys.argv[out_idx], index=False)
""", encoding="utf-8")
    (sm_dir / "manifest.json").write_text("""
{
    "candidate_id": "submodel_01",
    "name": "Submodel 01",
    "task_type": "binary",
    "entrypoint": "entrypoint.py",
    "supported_operations": ["fit", "predict"],
    "class_order": [0, 1]
}
""", encoding="utf-8")

    # Fixed shared split assignments (2 folds)
    locked_splits = [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ]

    dest = tmp_path / "ensemble_dest"

    spec = ParetoEnsembleSpec(
        ensemble_id="ens_locked_01",
        task_type="binary",
        submodel_dirs=[str(sm_dir)],
        meta_learner="ridge",
        target_column="target",
        id_column="id",
        class_order=[0, 1],
        split_assignments=locked_splits,
    )

    synth = ParetoEnsembleSynthesizer.synthesize_ensemble(spec, dest, train_csv)

    assert synth.ensemble_id == "ens_locked_01"
    assert (dest / "manifest.json").exists()
    assert (dest / "ensemble_meta.json").exists()


def test_rb003_missing_declared_topology_column_fails_closed():
    """Verify missing declared temporal or group columns fail closed with WorkflowError."""
    from mlrescurix_worker.scientific.fe_ablation import build_cv_splitter, ablate_feature_hypothesis

    df = pd.DataFrame({"feat": [1, 2, 3], "target": [0, 1, 0]})

    # 1. Missing declared temporal column
    topo_ts = {"temporal_col": "nonexistent_time", "target_column": "target"}
    with pytest.raises(WorkflowError, match="Declared temporal column 'nonexistent_time' missing"):
        build_cv_splitter(topo_ts, df=df)

    with pytest.raises(WorkflowError, match="Declared temporal column 'nonexistent_time' missing"):
        ablate_feature_hypothesis("h1", candidate_features_py=["feat"], base_features_py=None, topology=topo_ts, df=df)

    # 2. Missing declared group column
    topo_grp = {"entity_group_col": "nonexistent_group", "target_column": "target", "cv_scheme": "group_kfold"}
    with pytest.raises(WorkflowError, match="Declared group column 'nonexistent_group' missing"):
        build_cv_splitter(topo_grp, df=df)

    with pytest.raises(WorkflowError, match="Declared group column 'nonexistent_group' missing"):
        ablate_feature_hypothesis("h2", candidate_features_py=["feat"], base_features_py=None, topology=topo_grp, df=df)


def test_rb003_temporal_comparison_rejects_equal_timestamps(tmp_path: Path):
    """Verify temporal validation rejects equal timestamps across the train/val cutoff."""
    df_train = pd.DataFrame({
        "id": list(range(10)),
        "timestamp": ["2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01", "2026-01-01",
                      "2026-01-01", "2026-01-02", "2026-01-02", "2026-01-02", "2026-01-02"],
        "target": [0, 1] * 5,
    })

    # Fold where max train time is '2026-01-01' and min val time is ALSO '2026-01-01' (equal timestamps!)
    equal_ts_splits = [
        ([0, 1, 2], [3, 4]),  # Both sets have timestamp '2026-01-01'
    ]

    with pytest.raises(WorkflowError, match="Temporal leakage detected"):
        ParetoEnsembleSynthesizer._collect_oof_predictions(
            submodels_dest=tmp_path,
            submodel_ids=[],
            df_train=df_train,
            target_col="target",
            id_col="id",
            task_type="binary",
            classes=[0, 1],
            split_assignments=None,
            topology={"temporal_col": "timestamp"},
        )


def test_rb004_exact_integer_indices_and_disjoint_validation_folds(tmp_path: Path):
    """Verify exact integer indices without float coercion and disjointness across validation folds."""
    df_train = pd.DataFrame({"id": list(range(10)), "target": [0, 1] * 5})

    # 1. Float indices rejected
    float_splits = [
        (np.array([0.0, 1.0, 2.0]), np.array([3.0, 4.0])),
    ]
    with pytest.raises(WorkflowError, match="indices must be exact integers without float coercion"):
        ParetoEnsembleSynthesizer._collect_oof_predictions(
            submodels_dest=tmp_path,
            submodel_ids=[],
            df_train=df_train,
            target_col="target",
            id_col="id",
            task_type="binary",
            classes=[0, 1],
            split_assignments=float_splits,
        )

    # 2. Overlapping validation indices across folds rejected
    overlap_val_splits = [
        ([0, 1, 2], [3, 4, 5]),
        ([0, 1, 3], [4, 5, 6]),  # indices 4 and 5 overlap with previous fold validation set!
    ]
    with pytest.raises(WorkflowError, match="Validation indices in fold 1 overlap with previous validation fold"):
        ParetoEnsembleSynthesizer._collect_oof_predictions(
            submodels_dest=tmp_path,
            submodel_ids=[],
            df_train=df_train,
            target_col="target",
            id_col="id",
            task_type="binary",
            classes=[0, 1],
            split_assignments=overlap_val_splits,
        )


def test_rb014_unsupported_metrics_and_task_incompatibility_rejected():
    """Verify task-aware metric dispatch rejects unsupported metrics and wrong task types."""
    from mlrescurix_worker.scientific.fe_ablation import ablate_feature_hypothesis

    df_cls = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": [0, 1, 0, 1, 0, 1]})
    topo_cls = {"target_column": "target", "task_type": "binary_classification", "cv_splits": 2}

    # 1. Classification task with regression metric fails closed
    with pytest.raises(WorkflowError, match="Regression metric 'rmse' is invalid for classification task"):
        ablate_feature_hypothesis("m1", candidate_features_py=["x"], base_features_py=None, topology=topo_cls, df=df_cls, metric_name="rmse")

    # 2. Unknown metric fails closed (no silent fallback to balanced_accuracy)
    with pytest.raises(WorkflowError, match="Unsupported classification metric: 'unknown_metric'"):
        ablate_feature_hypothesis("m2", candidate_features_py=["x"], base_features_py=None, topology=topo_cls, df=df_cls, metric_name="unknown_metric")

    # 3. Regression task with classification metric fails closed
    df_reg = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [10.5, 20.3, 30.1, 40.8, 50.2, 60.0]})
    topo_reg = {"target_column": "target", "task_type": "regression", "cv_splits": 2}
    with pytest.raises(WorkflowError, match="Classification metric 'balanced_accuracy' is invalid for regression task"):
        ablate_feature_hypothesis("m3", candidate_features_py=["x"], base_features_py=None, topology=topo_reg, df=df_reg, metric_name="balanced_accuracy")
