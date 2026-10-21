"""Synthetic dataset contract fixtures for Kaggle benchmarks K1-K5.

Provides schema-conforming synthetic datasets for deterministic development,
integration tests, and pipeline verification.

Note: Synthetic cases do NOT count as authentic Kaggle benchmark results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_k1_synthetic_titanic(n_samples: int = 200, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic Titanic dataset conforming to K1 schema and family/ticket groups."""
    rng = np.random.RandomState(seed)
    families = [f"Fam_{i:03d}" for i in range(n_samples // 4)]
    tickets = [f"PC_{rng.randint(1000, 9999)}" for _ in range(n_samples // 3)]

    pclasses = rng.choice([1, 2, 3], size=n_samples, p=[0.24, 0.21, 0.55])
    sexes = rng.choice(["male", "female"], size=n_samples, p=[0.65, 0.35])
    ages = rng.normal(loc=29.0, scale=14.0, size=n_samples)
    ages = np.where(ages < 0.5, 0.5, ages)
    # Inject missingness into age
    ages = np.where(rng.rand(n_samples) < 0.2, np.nan, ages)

    fares = np.where(pclasses == 1, rng.exponential(60.0, size=n_samples) + 25.0,
             np.where(pclasses == 2, rng.exponential(20.0, size=n_samples) + 10.0,
                      rng.exponential(10.0, size=n_samples) + 4.0))

    family_assignments = [families[i % len(families)] for i in range(n_samples)]
    ticket_assignments = [tickets[i % len(tickets)] for i in range(n_samples)]

    # Target: female and 1st class higher survival
    prob_surv = 0.38 + 0.35 * (sexes == "female") + 0.15 * (pclasses == 1) - 0.15 * (pclasses == 3)
    prob_surv = np.clip(prob_surv, 0.05, 0.95)
    survived = rng.binomial(1, prob_surv)

    names = [f"Passenger, {'Miss.' if s == 'female' else 'Mr.'} {fam} Name_{i}" for i, (s, fam) in enumerate(zip(sexes, family_assignments))]

    df = pd.DataFrame({
        "PassengerId": np.arange(1, n_samples + 1),
        "Survived": survived,
        "Pclass": pclasses,
        "Name": names,
        "Sex": sexes,
        "Age": ages,
        "SibSp": rng.randint(0, 4, size=n_samples),
        "Parch": rng.randint(0, 3, size=n_samples),
        "Ticket": ticket_assignments,
        "Fare": fares,
        "Cabin": [f"C{rng.randint(10, 150)}" if rng.rand() < 0.25 else np.nan for _ in range(n_samples)],
        "Embarked": rng.choice(["S", "C", "Q"], size=n_samples, p=[0.72, 0.19, 0.09]),
        "FamilyGroup": family_assignments,
    })
    return df


def generate_k2_synthetic_house_prices(n_samples: int = 200, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic House Prices dataset conforming to K2 schema."""
    rng = np.random.RandomState(seed)
    gr_liv_area = rng.normal(1500, 500, size=n_samples)
    gr_liv_area = np.maximum(400, gr_liv_area)
    overall_qual = rng.choice(np.arange(1, 11), size=n_samples)
    year_built = rng.randint(1920, 2020, size=n_samples)
    neighborhood = rng.choice(["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel"], size=n_samples)

    log_price = 10.5 + 0.0006 * gr_liv_area + 0.15 * overall_qual + 0.002 * (year_built - 1970) + rng.normal(0, 0.15, size=n_samples)
    sale_price = np.exp(log_price)

    df = pd.DataFrame({
        "Id": np.arange(1, n_samples + 1),
        "GrLivArea": gr_liv_area,
        "OverallQual": overall_qual,
        "YearBuilt": year_built,
        "Neighborhood": neighborhood,
        "TotalBsmtSF": rng.normal(1000, 300, size=n_samples).clip(0),
        "FullBath": rng.choice([1, 2, 3], size=n_samples),
        "SalePrice": sale_price,
    })
    return df


def generate_k3_synthetic_forest_cover(n_samples: int = 250, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic Forest Cover Type dataset conforming to K3 schema (7 classes)."""
    rng = np.random.RandomState(seed)
    elevation = rng.normal(2700, 300, size=n_samples)
    aspect = rng.randint(0, 360, size=n_samples)
    slope = rng.randint(0, 45, size=n_samples)

    # 7 classes: 1 to 7
    probs = [0.37, 0.49, 0.06, 0.01, 0.02, 0.03, 0.02]
    cover_type = rng.choice(np.arange(1, 8), size=n_samples, p=probs)

    df = pd.DataFrame({
        "Id": np.arange(1, n_samples + 1),
        "Elevation": elevation,
        "Aspect": aspect,
        "Slope": slope,
        "Horizontal_Distance_To_Hydrology": rng.normal(250, 150, size=n_samples).clip(0),
        "Vertical_Distance_To_Hydrology": rng.normal(45, 30, size=n_samples),
        "Horizontal_Distance_To_Roadways": rng.normal(1500, 800, size=n_samples).clip(0),
        "Cover_Type": cover_type,
    })
    return df


def generate_k4_synthetic_fraud(n_samples: int = 300, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic IEEE-CIS Fraud dataset conforming to K4 schema."""
    rng = np.random.RandomState(seed)
    # TransactionDT is seconds from an arbitrary start date
    start_dt = 86400
    transaction_dts = np.sort(rng.randint(start_dt, start_dt + 30 * 86400, size=n_samples))
    transaction_ids = np.arange(3000000, 3000000 + n_samples)
    amounts = rng.exponential(120.0, size=n_samples) + 5.0
    product_cd = rng.choice(["W", "C", "R", "H", "S"], size=n_samples)

    # Imbalanced fraud: ~3.5%
    is_fraud = rng.binomial(1, 0.035, size=n_samples)

    df = pd.DataFrame({
        "TransactionID": transaction_ids,
        "TransactionDT": transaction_dts,
        "TransactionAmt": amounts,
        "ProductCD": product_cd,
        "card1": rng.randint(1000, 9999, size=n_samples),
        "card2": rng.choice([100.0, 200.0, 300.0, np.nan], size=n_samples),
        "isFraud": is_fraud,
    })
    return df


def generate_k5_synthetic_store_sales(n_dates: int = 60, seed: int = 2026) -> pd.DataFrame:
    """Generate synthetic Store Sales dataset conforming to K5 panel forecasting schema."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2026-01-01", periods=n_dates, freq="D").astype(str)
    stores = [1, 2]
    families = ["GROCERY", "BEVERAGES"]

    rows = []
    idx = 1
    for d in dates:
        day_of_week = pd.to_datetime(d).dayofweek
        weekend_boost = 1.3 if day_of_week >= 5 else 1.0
        for s in stores:
            for f in families:
                base_sales = 200.0 if f == "GROCERY" else 80.0
                sales = base_sales * weekend_boost + rng.normal(0, 15.0)
                sales = max(0.0, sales)
                rows.append({
                    "id": idx,
                    "date": d,
                    "store_nbr": s,
                    "family": f,
                    "sales": sales,
                    "onpromotion": rng.binomial(1, 0.2),
                })
                idx += 1
    return pd.DataFrame(rows)
