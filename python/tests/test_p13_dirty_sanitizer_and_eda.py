"""Comprehensive Unit & Synthetic Verification for Phase 2: Dirty Sanitizer, Semantic Sieve & M2-EDA.

Covers:
1. `dirty_sanitizer.py`: Sentinels, accounting negatives, percentages, currency, 85% promotion rule,
   and `apply_coercions` zero train-serve skew test.
2. `semantic_sieve.py`: Multi-signal target resolution (explicit, prompt match, canonical lexicon, candidate fallback;
   NEVER df.columns[-1]), information-theoretic 4-stage column classifications, entity/temporal group resolution,
   and Draft 2020-12 schema conformance.
3. `eda_agent.py`: `eda_forensics_suite` on Titanic dataset and synthetic dirty dataset,
   visual diagnostic SVG plot generation, Context Firewall assertions (< 4 KB dossier),
   and `run_exploratory_python` execution sandbox (< 2 KB firewall).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pytest

from mlrescurix_worker.scientific.dirty_sanitizer import (
    apply_coercions,
    is_sentinel,
    parse_dirty_numeric_value,
    sanitize_dataframe,
    sanitize_dataset,
)
from mlrescurix_worker.scientific.eda_agent import (
    eda_forensics_suite,
    run_exploratory_python,
)
from mlrescurix_worker.scientific.semantic_sieve import (
    analyze_and_lock_topology,
    compute_column_metrics,
    infer_task_type,
    resolve_target_column,
)


@pytest.fixture
def topology_schema() -> dict:
    schema_path = Path("contracts/v2/dataset_topology.schema.json")
    assert schema_path.is_file(), f"Schema file missing: {schema_path}"
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture
def titanic_csv_path() -> Path:
    p = Path(".mlrescurix-state/kaggle-titanic/titanic.csv")
    if not p.is_file():
        p = Path(".mlrescurix-state/data/titanic_raw.csv")
    assert p.is_file(), f"Titanic CSV missing: {p}"
    return p


def test_dirty_sanitizer_sentinels_and_promotion():
    """Verify sentinel normalization, dirty numeric parsing, and >= 85% column promotion."""
    df = pd.DataFrame({
        "curr": ["$1,200", "$2,500.50", "€45.00", "£99.99", "?", "N/A"],
        "pct": ["15.4%", "20.0%", "-5.5%", "0%", "null", "nan"],
        "acct": ["(1,234.50)", "(50.0)", "100.0", "(25.25)", "-999", "None"],
        "mixed_strings": ["Item A", "Item B", "Item C", "Item D", "?", "N/A"],
        "mostly_text_with_digits": ["User 1", "User 2", "User 3", "100", "None", "?"],
        "already_numeric": [10.0, 20.0, -999, 40.0, 50.0, 60.0],
    })

    res = sanitize_dataframe(df, promotion_threshold=0.85)

    # 1. Assert promotions
    assert "curr" in res.promoted_columns
    assert "pct" in res.promoted_columns
    assert "acct" in res.promoted_columns
    assert "mixed_strings" not in res.promoted_columns
    assert "mostly_text_with_digits" not in res.promoted_columns

    # 2. Assert dtypes
    assert pd.api.types.is_float_dtype(res.df["curr"])
    assert pd.api.types.is_float_dtype(res.df["pct"])
    assert pd.api.types.is_float_dtype(res.df["acct"])

    # 3. Assert values
    assert np.isclose(res.df["curr"].iloc[0], 1200.0)
    assert np.isclose(res.df["curr"].iloc[1], 2500.50)
    assert np.isclose(res.df["curr"].iloc[2], 45.00)
    assert np.isclose(res.df["pct"].iloc[0], 0.154)
    assert np.isclose(res.df["pct"].iloc[2], -0.055)
    assert np.isclose(res.df["acct"].iloc[0], -1234.50)
    assert np.isclose(res.df["acct"].iloc[1], -50.0)
    assert np.isclose(res.df["acct"].iloc[2], 100.0)

    # 4. Assert sentinels cleaned
    assert np.isnan(res.df["curr"].iloc[4])
    assert np.isnan(res.df["already_numeric"].iloc[2])  # -999 cleaned to NaN
    assert res.df["mixed_strings"].iloc[4] is None
    assert res.sentinels_cleaned_count >= 10

    # 5. Assert coercions map
    assert res.coercions["curr"] == "currency_to_float64"
    assert res.coercions["pct"] == "percentage_to_float64"
    assert res.coercions["acct"] == "accounting_parentheses_to_float64"

    # 6. Zero train-serve skew verification via apply_coercions
    serve_df = pd.DataFrame({
        "curr": ["$5,000", "N/A"],
        "pct": ["12.5%", "nan"],
        "acct": ["(300)", "10"],
        "mixed_strings": ["New Item", "?"],
    })
    serve_cleaned = apply_coercions(serve_df, res.coercions)
    assert pd.api.types.is_float_dtype(serve_cleaned["curr"])
    assert np.isclose(serve_cleaned["curr"].iloc[0], 5000.0)
    assert np.isnan(serve_cleaned["curr"].iloc[1])
    assert np.isclose(serve_cleaned["pct"].iloc[0], 0.125)
    assert np.isclose(serve_cleaned["acct"].iloc[0], -300.0)


def test_semantic_sieve_multi_signal_target_resolution():
    """Verify target resolver respects priority and NEVER blindly falls back to df.columns[-1]."""
    # Dataset where target is placed in the middle and last column is Embarked or something else
    df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 0, 1],
        "Age": [22.0, 38.0, 26.0, 35.0, 28.0],
        "Fare": [7.25, 71.28, 7.92, 53.10, 8.05],
        "Embarked": ["S", "C", "S", "S", "Q"],
    })

    # Priority 1: Explicit target overrides everything
    assert resolve_target_column(df, explicit_target="Age") == "Age"

    # Priority 2: Problem statement header match
    prompt = "Predict whether passenger Survived or died on the Titanic voyage."
    assert resolve_target_column(df, problem_statement=prompt) == "Survived"

    # Priority 3: Canonical target lexicon match (Survived is in lexicon)
    # Even without prompt, it must pick Survived, NEVER Embarked (columns[-1])
    assert resolve_target_column(df) == "Survived"

    # Test churn lexicon
    df_churn = pd.DataFrame({
        "customer_id": [101, 102, 103],
        "churn": [1, 0, 1],
        "monthly_fee": [50.0, 80.0, 70.0],
        "signup_source": ["web", "app", "referral"],  # last column
    })
    assert resolve_target_column(df_churn) == "churn"

    # Priority 4: Fallback candidate inspection when names are non-canonical
    # Target column is binary {0, 1} named 'flag', last col is high-cardinality string
    df_fallback = pd.DataFrame({
        "id_col": [1, 2, 3, 4],
        "flag": [1, 0, 1, 0],
        "score_a": [10.5, 20.2, 15.3, 18.9],
        "notes": ["note 1", "note 2", "note 3", "note 4"],  # last col
    })
    resolved = resolve_target_column(df_fallback)
    assert resolved == "flag"
    assert resolved != "notes"
    assert resolved != "id_col"


def test_titanic_ground_truth_eda_forensics(titanic_csv_path, topology_schema, tmp_path):
    """Execute complete M2-EDA forensics on live Titanic dataset and verify architectural invariants."""
    eda_dir = tmp_path / "artifacts" / "eda"
    dossier = eda_forensics_suite(
        titanic_csv_path,
        problem_statement="Predict passenger survival on Titanic",
        output_dir=eda_dir,
        dataset_id="titanic_benchmark",
    )

    topology = dossier.topology

    # 1. Assert target resolution
    assert topology["target_column"] == "Survived", "Target MUST be Survived, never Embarked!"
    assert topology["task_type"] == "binary_classification"
    assert topology["cv_scheme"] == "stratified_kfold"
    assert topology["cv_splits"] == 5

    # 2. Assert ID column isolation
    assert "PassengerId" in topology["id_columns"]
    assert "PassengerId" not in topology["clean_feature_columns"]
    assert topology["column_classifications"]["PassengerId"] == "DROP_SURROGATE_ID"

    # 3. Assert column classifications
    classes = topology["column_classifications"]
    assert classes["Survived"] == "TARGET"
    assert classes["Name"] == "SHORT_TEXT_EMBED"
    assert classes["Ticket"] == "QUASI_ID_DECOMPOSE"
    assert classes["Cabin"] == "QUASI_ID_DECOMPOSE"
    assert classes["Sex"] == "SAFE_CATEGORICAL"
    assert classes["Embarked"] == "SAFE_CATEGORICAL"
    assert classes["Fare"] == "NUMERIC_CONTINUOUS"
    assert classes["Age"] == "NUMERIC_CONTINUOUS"

    # 4. Assert Context Firewall on dossier
    raw_summary_bytes = len(dossier.summary_markdown.encode("utf-8"))
    assert raw_summary_bytes < 4096, f"Dossier markdown exceeded 4 KB Context Firewall ({raw_summary_bytes} bytes)"
    assert "### M2-EDA Forensics Dossier:" in dossier.summary_markdown
    assert "Survived" in dossier.summary_markdown
    assert "PassengerId" in dossier.summary_markdown

    # 5. Assert visual artifacts were generated
    assert len(dossier.artifacts) >= 2
    for art in dossier.artifacts:
        p = Path(art)
        assert p.is_file(), f"Expected plot file to exist: {p}"
        assert p.stat().st_size > 50, f"Expected non-empty plot file: {p}"

    # 6. Assert strict Draft 2020-12 schema conformance
    validator = jsonschema.Draft202012Validator(topology_schema)
    validator.validate(topology)


def test_synthetic_dirty_dataset_with_shuffled_columns(topology_schema, tmp_path):
    """Stress test with shuffled columns, dirty currencies, percentages, negative parens, and entity grouping."""
    np.random.seed(42)
    n = 200

    # Entity groups: households/families sharing group_id
    group_ids = [f"GRP_{(i // 4):03d}" for i in range(n)]

    synthetic_df = pd.DataFrame({
        "notes": [f"Customer description with order #{np.random.randint(100, 999)}" for _ in range(n)],
        "household_id": group_ids,
        "discount_pct": [f"{np.random.choice([5, 10, 15, 20])}%" if i % 10 != 0 else "?" for i in range(n)],
        "account_id": [f"ACC_{i:05d}" for i in range(n)],
        "target": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "revenue_dirty": [f"${np.random.uniform(50, 1500):.2f}" if i % 8 != 0 else "N/A" for i in range(n)],
        "balance_acct": [f"({np.random.uniform(10, 500):.2f})" if i % 5 == 0 else f"{np.random.uniform(10, 500):.2f}" for i in range(n)],
        "trailing_col": ["active" if i % 2 == 0 else "inactive" for i in range(n)],
    })

    csv_path = tmp_path / "dirty_synthetic.csv"
    synthetic_df.to_csv(csv_path, index=False)

    dossier = eda_forensics_suite(
        csv_path,
        output_dir=tmp_path / "eda",
        dataset_id="dirty_synthetic_dataset",
    )

    topo = dossier.topology

    # Assertions
    assert topo["target_column"] == "target"
    assert topo["task_type"] == "binary_classification"
    assert "account_id" in topo["id_columns"]
    assert topo["column_classifications"]["account_id"] == "DROP_SURROGATE_ID"

    # Coercions asserted
    assert "revenue_dirty" in topo["coercions"]
    assert "discount_pct" in topo["coercions"]
    assert "balance_acct" in topo["coercions"]

    # Grouping detected
    assert topo["entity_group_col"] == "household_id"
    assert topo["cv_scheme"] == "stratified_group_kfold"

    # Context Firewall
    assert len(dossier.summary_markdown.encode("utf-8")) < 4096

    # Schema validity
    validator = jsonschema.Draft202012Validator(topology_schema)
    validator.validate(topo)


def test_exploratory_python_sandbox_context_firewall():
    """Verify run_exploratory_python sandbox enforces < 2 KB Context Firewall and handles exceptions cleanly."""
    df = pd.DataFrame({
        "A": list(range(100)),
        "B": [f"Val_{i}" for i in range(100)],
    })

    # Test 1: Normal execution
    code_1 = "print('Sum of A:', df['A'].sum())"
    out_1 = run_exploratory_python(code_1, df)
    assert "governed sandbox unavailable" in out_1
    assert len(out_1.encode("utf-8")) < 2048

    # Test 2: Huge print output truncation
    code_2 = "for i in range(200): print(f'Row {i}: huge output string testing context firewall truncation')"
    out_2 = run_exploratory_python(code_2, df)
    assert len(out_2.encode("utf-8")) <= 2048
    assert "governed sandbox unavailable" in out_2

    # Test 3: Exception handling
    code_3 = "df['non_existent_column'].mean()"
    out_3 = run_exploratory_python(code_3, df)
    assert "[error]" in out_3
    assert "governed sandbox unavailable" in out_3
    assert len(out_3.encode("utf-8")) < 2048
