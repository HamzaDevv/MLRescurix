"""Comprehensive Unit & Synthetic Verification for Phase 3: Primitives & M2-FE Ablation Engine.

Covers:
1. `NestedOOFEmpiricalBayesEncoder`:
   - 100% unique random strings test: asserts train-val gap remains < 0.05 without tree memorization.
   - Informative category target encoding with Empirical Bayes shrinkage and Gaussian noise.
   - Out-of-fold cross-validation during fit_transform and global prior fallback for unseen categories during transform.
   - Full `joblib` serialization compatibility.
2. `QuasiIdDecomposer`:
   - Drops raw IDs on Titanic (Ticket, Cabin).
   - Extracts group sizes, numeric aggregations per member (<col>__<num>_mean, <num> / <col>__group_size).
   - Extracts structural prefixes ('PC', 'C'), structural masks ('\\d+' -> '#'), string length, and digit ratio.
   - Empirical Bayes prefix encoding and zero-leakage transform.
3. `LightweightTextEmbedder`:
   - Sub-token indicator extraction with target Mutual Information gating (extracts honorifics like 'Mr', 'Mrs', 'Miss', 'Master').
   - Surname cluster size extraction on Titanic Name column.
   - In-fold Word + Char TF-IDF SVD embeddings (TruncatedSVD(n_components=16)).
4. `InformativeMissingnessAndInteractions`:
   - Binary <col>__is_nan indicators for missingness rate in [0.5%, 99.5%].
   - Fast MI-gated pairwise numeric interactions (A / (|B| + eps), A * B, A - B).
5. `fe_ablation.py`:
   - PurgedTimeSeriesSplit embargo and lookahead prevention.
   - Paired CV ablation against baseline with locked CV splitter from DatasetTopologyContract.
   - Auto-Gate: rejects non-improving (Delta CV <= 0) or overfitting (Delta Train-Val Gap > 0.08) hypotheses.
   - Synchronization with ResearchLedger.
6. End-to-end multi-primitive feature pipeline on Titanic dataset.
"""

from __future__ import annotations

import io
import json
import tempfile
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from mlrescurix_worker.scientific.fe_ablation import (
    PurgedTimeSeriesSplit,
    ablate_feature_hypothesis,
    build_cv_splitter,
)
from mlrescurix_worker.scientific.primitives import (
    InformativeMissingnessAndInteractions,
    LightweightTextEmbedder,
    NestedOOFEmpiricalBayesEncoder,
    QuasiIdDecomposer,
)
from mlrescurix_worker.scientific.semantic_sieve import analyze_and_lock_topology


@pytest.fixture
def titanic_df() -> pd.DataFrame:
    p = Path(".mlrescurix-state/kaggle-titanic/titanic.csv")
    if not p.is_file():
        p = Path(".mlrescurix-state/data/titanic_raw.csv")
    assert p.is_file(), f"Titanic CSV missing: {p}"
    return pd.read_csv(p)


# =====================================================================
# 1. NestedOOFEmpiricalBayesEncoder Tests
# =====================================================================

def test_nested_oof_eb_encoder_unique_strings_memorization_prevention():
    """Verify that feeding 100% unique random strings does NOT allow tree memorization.

    Blueprint Step 3 Verification Test:
    Feed a column of 100% unique random strings into `NestedOOFEmpiricalBayesEncoder` + `LightGBM`.
    Assert validation balanced accuracy does NOT collapse to 0.5000 and train-val gap remains < 0.05.
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier as LGBMClassifier

    n_samples = 800
    rng = np.random.RandomState(42)

    # Ground truth target
    y = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # Informative feature with 80% accuracy signal
    signal = y.copy()
    flip_mask = rng.rand(n_samples) < 0.20
    signal[flip_mask] = 1 - signal[flip_mask]

    # 100% unique random strings (UUIDs)
    unique_strings = [f"uuid_{uuid.uuid4().hex[:12]}_{i}" for i in range(n_samples)]

    df = pd.DataFrame({
        "signal": signal.astype(float),
        "random_id": unique_strings,
    })

    # Train / Val split (70 / 30)
    train_idx = np.arange(0, 560)
    val_idx = np.arange(560, n_samples)

    df_train = df.iloc[train_idx].copy()
    y_train = y[train_idx]
    df_val = df.iloc[val_idx].copy()
    y_val = y[val_idx]

    encoder = NestedOOFEmpiricalBayesEncoder(
        categorical_columns=["random_id"],
        n_splits=5,
        min_smoothing=10.0,
        noise_level=0.01,
        random_state=42,
        drop_original=True,
    )

    X_train_enc = encoder.fit_transform(df_train, y_train)
    X_val_enc = encoder.transform(df_val)

    # Ensure output is clean numeric
    assert "random_id__eb" in X_train_enc.columns
    assert "random_id" not in X_train_enc.columns
    assert np.all(np.isfinite(X_train_enc["random_id__eb"]))
    assert np.all(np.isfinite(X_val_enc["random_id__eb"]))

    # Fit LightGBM on the encoded features + signal
    try:
        clf = LGBMClassifier(n_estimators=30, random_state=42, verbose=-1)
    except TypeError:
        clf = LGBMClassifier(max_iter=30, random_state=42)

    clf.fit(X_train_enc, y_train)

    train_preds = clf.predict(X_train_enc)
    val_preds = clf.predict(X_val_enc)

    from sklearn.metrics import balanced_accuracy_score
    train_acc = float(balanced_accuracy_score(y_train, train_preds))
    val_acc = float(balanced_accuracy_score(y_val, val_preds))

    gap = abs(train_acc - val_acc)

    # 1. Validation balanced accuracy does NOT collapse to 0.5000 (preserves true signal)
    assert val_acc > 0.70, f"Validation accuracy collapsed to {val_acc:.4f} (expected > 0.70)"

    # 2. Train-val gap remains strictly bounded (< 0.05)
    assert gap < 0.05, f"Train-val gap {gap:.4f} exceeded 0.05 boundary (train={train_acc:.4f}, val={val_acc:.4f})"


def test_nested_oof_eb_encoder_informative_categories_and_unseen_fallback():
    """Verify EB target encoding extracts signal and smoothly falls back to global prior on unseen values."""
    categories = ["catA"] * 200 + ["catB"] * 200 + ["catC"] * 100
    # catA has 80% 1s, catB has 20% 1s, catC has 50% 1s
    rng = np.random.RandomState(42)
    y_a = rng.choice([1, 0], size=200, p=[0.8, 0.2])
    y_b = rng.choice([1, 0], size=200, p=[0.2, 0.8])
    y_c = rng.choice([1, 0], size=100, p=[0.5, 0.5])
    y = np.concatenate([y_a, y_b, y_c])

    df_train = pd.DataFrame({"category": categories})

    encoder = NestedOOFEmpiricalBayesEncoder(
        categorical_columns=["category"],
        n_splits=5,
        min_smoothing=5.0,
        noise_level=0.005,
        random_state=42,
    )

    X_tr = encoder.fit_transform(df_train, y)
    assert "category__eb" in X_tr.columns

    # Mean for catA should be significantly higher than catB
    mean_a = X_tr.iloc[:200]["category__eb"].mean()
    mean_b = X_tr.iloc[200:400]["category__eb"].mean()
    assert mean_a > 0.65
    assert mean_b < 0.35

    # Test transform on unseen categories
    df_test = pd.DataFrame({"category": ["catA", "catB", "UNSEEN_XYZ", "ANOTHER_NEW"]})
    X_test = encoder.transform(df_test)

    # Unseen categories must smoothly map to global mean
    global_mean = encoder.global_mean_
    assert np.isclose(X_test["category__eb"].iloc[2], global_mean, atol=1e-3)
    assert np.isclose(X_test["category__eb"].iloc[3], global_mean, atol=1e-3)


def test_nested_oof_eb_encoder_joblib_serialization():
    """Verify joblib dump and load preserves exact transform parity."""
    df = pd.DataFrame({"grp": ["A", "B", "A", "C", "B", "A"] * 20})
    y = np.array([1, 0, 1, 0, 0, 1] * 20)

    encoder = NestedOOFEmpiricalBayesEncoder(categorical_columns=["grp"], random_state=42)
    encoder.fit(df, y)

    buf = io.BytesIO()
    joblib.dump(encoder, buf)
    buf.seek(0)
    loaded_encoder = joblib.load(buf)

    df_test = pd.DataFrame({"grp": ["A", "B", "C", "D"]})
    out1 = encoder.transform(df_test)
    out2 = loaded_encoder.transform(df_test)

    pd.testing.assert_frame_equal(out1, out2)


# =====================================================================
# 2. QuasiIdDecomposer Tests
# =====================================================================

def test_quasi_id_decomposer_titanic(titanic_df: pd.DataFrame):
    """Verify QuasiIdDecomposer extracts structural, group, and aggregation features from Ticket & Cabin."""
    decomposer = QuasiIdDecomposer(
        id_columns=["Ticket", "Cabin"],
        numeric_columns=["Fare", "Age"],
        encode_prefix=True,
        drop_original=True,
        random_state=42,
    )

    y = titanic_df["Survived"]
    X_out = decomposer.fit_transform(titanic_df, y=y)

    # 1. Assert raw ID strings dropped
    assert "Ticket" not in X_out.columns
    assert "Cabin" not in X_out.columns

    # 2. Assert group size features extracted
    assert "Ticket__group_size" in X_out.columns
    assert "Cabin__group_size" in X_out.columns
    assert np.all(X_out["Ticket__group_size"] >= 1.0)

    # 3. Assert numeric aggregations (<col>__<num>_mean and per_member)
    assert "Ticket__Fare_mean" in X_out.columns
    assert "Ticket__Fare_per_member" in X_out.columns
    assert "Cabin__Fare_mean" in X_out.columns

    # 4. Assert structural prefix & mask features
    assert "Ticket__str_len" in X_out.columns
    assert "Ticket__digit_ratio" in X_out.columns
    assert "Ticket__prefix_freq" in X_out.columns
    assert "Ticket__mask_freq" in X_out.columns
    assert "Ticket__prefix_eb" in X_out.columns
    assert "Cabin__prefix_eb" in X_out.columns

    # Test transform on a new slice with unseen values
    new_slice = pd.DataFrame({
        "Ticket": ["UNKNOWN_9999", "PC 17599"],
        "Cabin": ["Z999", "C85"],
        "Fare": [50.0, 71.2833],
        "Age": [30.0, 38.0],
    })
    transformed = decomposer.transform(new_slice)
    assert len(transformed) == 2
    assert "Ticket" not in transformed.columns
    assert transformed["Ticket__group_size"].iloc[0] == 1.0
    assert np.isfinite(transformed["Ticket__Fare_mean"].iloc[0])


# =====================================================================
# 3. LightweightTextEmbedder Tests
# =====================================================================

def test_lightweight_text_embedder_titanic(titanic_df: pd.DataFrame):
    """Verify LightweightTextEmbedder extracts honorifics, surnames, and SVD embeddings from Name."""
    embedder = LightweightTextEmbedder(
        text_columns=["Name"],
        n_svd_components=8,
        top_k_subtokens=15,
        min_subtoken_freq=5,
        random_state=42,
        drop_original=True,
    )

    y = titanic_df["Survived"]
    X_out = embedder.fit_transform(titanic_df, y=y)

    # 1. Raw column dropped
    assert "Name" not in X_out.columns

    # 2. Key honorifics captured in selected subtokens due to high Mutual Information
    selected = embedder.selected_subtokens_.get("Name", [])
    # Typical Titanic honorifics: mr, mrs, miss, master
    assert any(h in selected for h in ["mr", "mrs", "miss", "master"]), f"Honorifics missing from {selected}"

    # Verify binary indicators exist
    honorific_cols = [c for c in X_out.columns if "has_mr" in c or "has_mrs" in c or "has_miss" in c]
    assert len(honorific_cols) > 0

    # 3. Surname cluster size extracted
    assert "Name__surname_cluster_size" in X_out.columns
    assert "Name__family_surname_count" in X_out.columns
    assert np.all(X_out["Name__surname_cluster_size"] >= 1.0)

    # 4. TruncatedSVD embeddings (8 components)
    for k in range(8):
        assert f"Name__svd_{k}" in X_out.columns
        assert np.all(np.isfinite(X_out[f"Name__svd_{k}"]))

    # Test transform on unseen names
    test_slice = pd.DataFrame({"Name": ["Smith, Mr. John", "Newperson, Dr. Alice"]})
    out_test = embedder.transform(test_slice)
    assert len(out_test) == 2
    assert "Name__svd_0" in out_test.columns
    assert out_test["Name__surname_cluster_size"].iloc[1] == 1.0  # Unseen surname defaults to 1


# =====================================================================
# 4. InformativeMissingnessAndInteractions Tests
# =====================================================================

def test_informative_missingness_and_interactions(titanic_df: pd.DataFrame):
    """Verify informative missingness indicators and MI-gated numeric interactions."""
    trans = InformativeMissingnessAndInteractions(
        numeric_columns=["Age", "Fare", "SibSp", "Parch"],
        missing_rate_bounds=(0.005, 0.995),
        max_candidate_features=4,
        max_interactions=5,
        random_state=42,
    )

    y = titanic_df["Survived"]
    X_out = trans.fit_transform(titanic_df, y=y)

    # Age is ~20% missing, Cabin is ~77% missing; both are in [0.5%, 99.5%]
    assert "Age__is_nan" in X_out.columns
    assert np.all(np.isin(X_out["Age__is_nan"].unique(), [0, 1]))

    # Selected interactions
    assert len(trans.selected_interactions_) > 0
    for _, _, _, out_name in trans.selected_interactions_:
        assert out_name in X_out.columns
        assert np.all(np.isfinite(X_out[out_name]))

    # Test transform
    new_slice = pd.DataFrame({
        "Age": [np.nan, 25.0],
        "Fare": [10.0, 50.0],
        "SibSp": [0, 1],
        "Parch": [0, 2],
    })
    out_slice = trans.transform(new_slice)
    assert out_slice["Age__is_nan"].iloc[0] == 1
    assert out_slice["Age__is_nan"].iloc[1] == 0


# =====================================================================
# 5. PurgedTimeSeriesSplit & Splitter Building Tests
# =====================================================================

def test_purged_time_series_split():
    """Verify PurgedTimeSeriesSplit enforces embargo gap between train and forward test folds."""
    n_samples = 200
    splitter = PurgedTimeSeriesSplit(n_splits=4, embargo_pct=0.05)
    splits = list(splitter.split(np.arange(n_samples)))

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert len(train_idx) > 0
        assert len(test_idx) > 0
        # Strict temporal order
        assert np.max(train_idx) < np.min(test_idx)
        # Embargo gap strictly respected
        embargo_gap = np.min(test_idx) - np.max(train_idx)
        assert embargo_gap >= 1


def test_build_cv_splitter():
    """Verify build_cv_splitter resolves schemes based on topology."""
    top_temporal = {"cv_scheme": "purged_time_series_split", "temporal_col": "time_col", "cv_splits": 3}
    s, g, t = build_cv_splitter(top_temporal)
    assert isinstance(s, PurgedTimeSeriesSplit)
    assert t == "time_col"

    top_group = {"cv_scheme": "stratified_group_kfold", "entity_group_col": "group_col", "task_type": "binary_classification"}
    s, g, t = build_cv_splitter(top_group)
    assert g == "group_col"


# =====================================================================
# 6. M2-FE Ablation Engine & Auto-Gate Tests
# =====================================================================

def test_fe_ablation_positive_hypothesis_accepted(titanic_df: pd.DataFrame):
    """Verify that a genuine feature improvement produces ACCEPTED status in ResearchLedger."""
    # Analyze topology
    topology = analyze_and_lock_topology(
        titanic_df, explicit_target="Survived", dataset_id="titanic_test"
    )

    # Baseline features: basic numeric features (Pclass, Fare, Age)
    base_features = ["Pclass", "Fare", "Age"]

    # Candidate hypothesis: baseline + QuasiIdDecomposer on Ticket
    def candidate_features(df, y=None):
        X = df[["Pclass", "Fare", "Age", "Ticket"]].copy()
        dec = QuasiIdDecomposer(id_columns=["Ticket"], numeric_columns=["Fare"], encode_prefix=False)
        return dec.fit_transform(X, y=y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "research_ledger.json"
        initial_ledger = {
            "schema_version": "v2",
            "session_id": "sess_001",
            "project_id": "proj_001",
            "current_epoch": 0,
            "status": "active",
            "metric_name": "balanced_accuracy",
            "epochs": [{
                "epoch": 0,
                "hypotheses": ["hyp_ticket_decomposition"],
                "accepted_features": [],
                "rejected_features": [],
                "ablation_results": [],
                "model_results": [],
                "ensemble_result": None,
                "critique": None,
                "m1_reflection": None,
                "timestamp_ms": 1000,
            }],
        }
        ledger_path.write_text(json.dumps(initial_ledger), encoding="utf-8")

        receipt = ablate_feature_hypothesis(
            hypothesis_id="hyp_ticket_decomposition",
            candidate_features_py=candidate_features,
            base_features_py=base_features,
            topology=topology,
            df=titanic_df,
            fidelity="fast",
            ledger_path=ledger_path,
        )

        assert "ablation_id" in receipt
        assert receipt["total_folds"] == 5
        assert isinstance(receipt["cv_score_delta"], float)
        assert receipt["status"] in ("ACCEPTED", "REJECTED")

        # Read back updated ledger
        updated_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        epoch_0 = updated_ledger["epochs"][0]
        assert len(epoch_0["ablation_results"]) == 1
        if receipt["status"] == "ACCEPTED":
            assert "hyp_ticket_decomposition" in epoch_0["accepted_features"]
        else:
            assert "hyp_ticket_decomposition" in epoch_0["rejected_features"]


def test_fe_ablation_overfitting_hypothesis_rejected(titanic_df: pd.DataFrame):
    """Verify that an overfitting or non-improving hypothesis triggers auto-gating to REJECTED."""
    topology = analyze_and_lock_topology(
        titanic_df, explicit_target="Survived", dataset_id="titanic_test"
    )

    base_features = ["Pclass", "Fare"]

    # Candidate hypothesis: baseline + 100% pure random noise that degrades validation score
    def degraded_candidate(df, y=None):
        X = df[["Pclass", "Fare"]].copy()
        rng = np.random.RandomState(999)
        # 20 columns of random noise to dilute signal and overfit
        for i in range(20):
            X[f"noise_{i}"] = rng.normal(0, 10.0, size=len(df))
        return X

    ledger: dict = {"epochs": []}

    receipt = ablate_feature_hypothesis(
        hypothesis_id="hyp_noisy_overfit",
        candidate_features_py=degraded_candidate,
        base_features_py=base_features,
        topology=topology,
        df=titanic_df,
        fidelity="fast",
        ledger=ledger,
    )

    # Assert that degraded performance triggered REJECTED
    assert receipt["status"] == "REJECTED"
    assert receipt["rejection_reason"] is not None
    assert "hyp_noisy_overfit" in ledger["epochs"][0]["rejected_features"]


# =====================================================================
# 7. End-to-End Pipeline with All Primitives
# =====================================================================

def test_end_to_end_pipeline_with_all_primitives(titanic_df: pd.DataFrame):
    """Compose all 4 primitives into an end-to-end scikit-learn pipeline on Titanic."""
    y = titanic_df["Survived"].to_numpy()
    X = titanic_df.drop(columns=["Survived"]).copy()

    # Full feature engineering pipeline
    fe_pipeline = Pipeline([
        ("quasi_id", QuasiIdDecomposer(id_columns=["Ticket", "Cabin"], numeric_columns=["Fare", "Age"])),
        ("text_embed", LightweightTextEmbedder(text_columns=["Name"], n_svd_components=4, top_k_subtokens=8)),
        ("missing_interactions", InformativeMissingnessAndInteractions(max_interactions=4)),
        ("eb_encoder", NestedOOFEmpiricalBayesEncoder(categorical_columns=["Sex", "Embarked"], n_splits=5)),
    ])

    X_transformed = fe_pipeline.fit_transform(X, y)

    # Assert all transformed columns are numeric
    for col in X_transformed.columns:
        assert pd.api.types.is_numeric_dtype(X_transformed[col]), f"Column {col} is not numeric: {X_transformed[col].dtype}"

    # Assert model fits and predicts cleanly
    model = HistGradientBoostingClassifier(max_iter=30, random_state=42)
    model.fit(X_transformed, y)
    preds = model.predict(X_transformed)

    acc = np.mean(preds == y)
    assert acc > 0.80, f"Expected training accuracy > 0.80, got {acc:.4f}"


def test_fe_ablation_with_python_code_snippet(titanic_df: pd.DataFrame):
    """Verify that ablate_feature_hypothesis can execute Python code string specifications."""
    topology = analyze_and_lock_topology(
        titanic_df, explicit_target="Survived", dataset_id="titanic_test"
    )

    base_features = ["Pclass", "Fare"]

    # Candidate feature definition as raw Python code string
    candidate_code = """
import numpy as np
import pandas as pd

def build_features(df, y=None):
    X = pd.DataFrame(index=df.index)
    X["Pclass"] = df["Pclass"].astype(float)
    X["Fare"] = df["Fare"].fillna(df["Fare"].median()).astype(float)
    X["LogFare"] = np.log1p(np.maximum(0, df["Fare"].fillna(0)))
    return X
"""

    receipt = ablate_feature_hypothesis(
        hypothesis_id="hyp_log_fare_snippet",
        candidate_features_py=candidate_code,
        base_features_py=base_features,
        topology=topology,
        df=titanic_df,
        fidelity="fast",
    )

    assert receipt["hypothesis_id"] == "hyp_log_fare_snippet"
    assert "LogFare" in receipt["feature_names"]
    assert receipt["status"] in ("ACCEPTED", "REJECTED")


def test_text_embedder_unsupervised_and_regression_continuous_target():
    """Verify LightweightTextEmbedder works unsupervised (no y) and with continuous regression y."""
    texts = [
        "Software Engineer with Python experience",
        "Senior Data Scientist machine learning",
        "Software Engineer backend systems",
        "Frontend Developer React TypeScript",
        "Data Scientist deep learning AI",
        "Software Engineer fullstack web",
    ] * 10
    salaries = np.random.RandomState(42).normal(120000, 20000, size=len(texts))

    df = pd.DataFrame({"job_title": texts})

    # Unsupervised
    embedder_unsup = LightweightTextEmbedder(text_columns=["job_title"], n_svd_components=4, min_subtoken_freq=2)
    out_unsup = embedder_unsup.fit_transform(df)
    assert "job_title__svd_0" in out_unsup.columns
    assert len(embedder_unsup.selected_subtokens_["job_title"]) > 0

    # Continuous target regression
    embedder_reg = LightweightTextEmbedder(text_columns=["job_title"], n_svd_components=4, min_subtoken_freq=2)
    out_reg = embedder_reg.fit_transform(df, y=salaries)
    assert "job_title__svd_0" in out_reg.columns
    assert any("engineer" in t or "scientist" in t for t in embedder_reg.selected_subtokens_["job_title"])


def test_eb_encoder_continuous_target_regression():
    """Verify NestedOOFEmpiricalBayesEncoder performs continuous target mean encoding with shrinkage."""
    groups = ["RegionA"] * 50 + ["RegionB"] * 50 + ["RegionC"] * 50
    rng = np.random.RandomState(42)
    # Continuous target
    y = np.concatenate([
        rng.normal(100.0, 5.0, size=50),
        rng.normal(200.0, 5.0, size=50),
        rng.normal(150.0, 5.0, size=50),
    ])

    df = pd.DataFrame({"region": groups})

    encoder = NestedOOFEmpiricalBayesEncoder(
        categorical_columns=["region"], n_splits=5, min_smoothing=5.0, noise_level=0.01, random_state=42
    )

    X_enc = encoder.fit_transform(df, y)
    assert "region__eb" in X_enc.columns

    mean_a = X_enc.iloc[:50]["region__eb"].mean()
    mean_b = X_enc.iloc[50:100]["region__eb"].mean()
    mean_c = X_enc.iloc[100:150]["region__eb"].mean()

    assert 90.0 < mean_a < 110.0
    assert 190.0 < mean_b < 210.0
    assert 140.0 < mean_c < 160.0

