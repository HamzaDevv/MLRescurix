"""Validation tests for Kaggle K1-K5 synthetic contract fixtures and metric schemas."""
import numpy as np
import pytest

from fixtures.kaggle_synthetic_contracts import (
    generate_k1_synthetic_titanic,
    generate_k2_synthetic_house_prices,
    generate_k3_synthetic_forest_cover,
    generate_k4_synthetic_fraud,
    generate_k5_synthetic_store_sales,
)


def test_k1_synthetic_titanic():
    df = generate_k1_synthetic_titanic(n_samples=100)
    assert len(df) == 100
    assert "Survived" in df.columns
    assert set(df["Survived"].unique()).issubset({0, 1})
    assert "FamilyGroup" in df.columns
    assert "Ticket" in df.columns


def test_k2_synthetic_house_prices():
    df = generate_k2_synthetic_house_prices(n_samples=100)
    assert len(df) == 100
    assert "SalePrice" in df.columns
    assert np.all(df["SalePrice"] > 0), "All SalePrice values must be positive"


def test_k3_synthetic_forest_cover():
    df = generate_k3_synthetic_forest_cover(n_samples=100)
    assert len(df) == 100
    assert "Cover_Type" in df.columns
    classes = set(df["Cover_Type"].unique())
    assert classes.issubset(set(range(1, 8))), f"Classes must be within 1-7, got {classes}"


def test_k4_synthetic_fraud():
    df = generate_k4_synthetic_fraud(n_samples=100)
    assert len(df) == 100
    assert "isFraud" in df.columns
    assert "TransactionDT" in df.columns
    # Check chronological ordering
    assert df["TransactionDT"].is_monotonic_increasing


def test_k5_synthetic_store_sales():
    df = generate_k5_synthetic_store_sales(n_dates=30)
    assert len(df) > 0
    assert "sales" in df.columns
    assert "date" in df.columns
    assert "store_nbr" in df.columns
    assert "family" in df.columns
    assert np.all(df["sales"] >= 0), "Sales must be non-negative"
