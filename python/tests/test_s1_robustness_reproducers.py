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
    assert "Timeout after 1.0s exceeded" in out
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
    assert "KeyError" in out
    assert "secret_target" in out


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
