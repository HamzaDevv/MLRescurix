import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlrescurix_worker.scientific.time_series import (
    FutureLookaheadLeakageError,
    InvalidTimestampError,
    LastValueBaseline,
    MovingAverageBaseline,
    SeasonalMeanBaseline,
    TemporalSplitProtocol,
    TemporalSplitter,
    TimeSeriesDatasetAdapter,
    TimeSeriesFeatureEngineer,
    TimeSeriesPredictor,
    compute_forecasting_metrics,
    evaluate_baselines_on_split,
    parse_and_validate_timestamps,
    verify_temporal_split_no_leakage,
)

ROOT = Path(__file__).resolve().parents[2]


class TestP9TimeSeriesAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p9_ts_"))
        self.schema = json.loads((ROOT / "contracts/v2/time_series_adapter.schema.json").read_text())

    def test_p9_2_schema_validation(self):
        """P9.2: Validate TimeSeriesDatasetAdapter contract under strict Draft 2020-12 schema."""
        adapter = TimeSeriesDatasetAdapter.synthetic_daily("ds_sales_daily")
        jsonschema.validate(instance=adapter.to_dict(), schema=self.schema)

    def test_p9_2_temporal_splitters_and_leakage_rejection(self):
        """P9.2: Strict temporal split contracts: entity-aware, rolling-window, and expanding-window."""
        n_days = 150
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
        df_single = pd.DataFrame({
            "timestamp": dates,
            "demand": np.sin(np.linspace(0, 10, n_days)) * 50 + 100 + np.random.randn(n_days) * 5,
        })

        # 1. Rolling-window splitter
        proto_rolling = TemporalSplitProtocol(
            protocol_type="rolling_window",
            train_window_steps=60,
            gap_steps=2,
            test_horizon_steps=14,
            step_stride=14,
            n_splits=3,
        )
        splitter_rolling = TemporalSplitter(proto_rolling, timestamp_column="timestamp")
        splits_rolling = splitter_rolling.split(df_single)
        self.assertEqual(len(splits_rolling), 3)

        for s in splits_rolling:
            self.assertEqual(len(s.train_indices), 60)
            self.assertEqual(len(s.gap_indices), 2)
            self.assertEqual(len(s.test_indices), 14)
            # Confirm strictly monotonic timestamps without overlap
            self.assertLess(s.train_end, s.test_start)

        # 2. Expanding-window splitter
        proto_expanding = TemporalSplitProtocol(
            protocol_type="expanding_window",
            train_window_steps=50,
            gap_steps=1,
            test_horizon_steps=10,
            step_stride=20,
            n_splits=3,
        )
        splitter_expanding = TemporalSplitter(proto_expanding, timestamp_column="timestamp")
        splits_expanding = splitter_expanding.split(df_single)
        self.assertEqual(len(splits_expanding), 3)
        self.assertLess(len(splits_expanding[0].train_indices), len(splits_expanding[1].train_indices))
        self.assertLess(len(splits_expanding[1].train_indices), len(splits_expanding[2].train_indices))

        # 3. Entity-aware temporal splitter
        df_entities = pd.concat([
            pd.DataFrame({
                "timestamp": dates,
                "store_id": "store_A",
                "demand": np.random.randn(n_days) * 10 + 100,
            }),
            pd.DataFrame({
                "timestamp": dates,
                "store_id": "store_B",
                "demand": np.random.randn(n_days) * 15 + 200,
            }),
        ], ignore_index=True)

        proto_entity = TemporalSplitProtocol(
            protocol_type="entity_aware_temporal",
            train_window_steps=50,
            gap_steps=2,
            test_horizon_steps=14,
            step_stride=14,
            n_splits=2,
        )
        splitter_entity = TemporalSplitter(
            proto_entity, timestamp_column="timestamp", entity_columns=["store_id"]
        )
        splits_entity = splitter_entity.split(df_entities)
        self.assertEqual(len(splits_entity), 2)
        # Verify 2 stores * 50 rows = 100 train rows per split
        self.assertEqual(len(splits_entity[0].train_indices), 100)
        self.assertEqual(len(splits_entity[0].test_indices), 28)

    def test_p9_2_negative_scientific_cases_leakage_and_ordering(self):
        """P9.2: Negative cases: reject future lookahead leakage, unordered timestamps, and negative lags."""
        dates = pd.date_range("2026-01-01", periods=10, freq="D")

        # 1. Lookahead leakage check: test min timestamp <= train max timestamp
        train_df = pd.DataFrame({"timestamp": dates[0:6], "y": range(6)})
        test_df_leaked = pd.DataFrame({"timestamp": dates[5:10], "y": range(5)})  # date 5 is in train!
        with self.assertRaises(FutureLookaheadLeakageError):
            verify_temporal_split_no_leakage(train_df, test_df_leaked, timestamp_column="timestamp")

        # 2. Out-of-order timestamps
        unordered_df = pd.DataFrame({
            "timestamp": [dates[0], dates[2], dates[1], dates[3]],
            "y": range(4),
        })
        with self.assertRaises(InvalidTimestampError):
            parse_and_validate_timestamps(unordered_df, timestamp_column="timestamp")

        # 3. Duplicate timestamp for same entity
        duplicate_df = pd.DataFrame({
            "timestamp": [dates[0], dates[0], dates[1]],
            "store_id": ["A", "A", "A"],
            "y": range(3),
        })
        with self.assertRaises(InvalidTimestampError):
            parse_and_validate_timestamps(duplicate_df, timestamp_column="timestamp", entity_columns=["store_id"])

        # 4. Negative lag (lookahead in feature engineering)
        with self.assertRaises(FutureLookaheadLeakageError):
            TimeSeriesFeatureEngineer(
                target_column="y",
                timestamp_column="timestamp",
                lags=[-1, 1],  # -1 is future leakage!
            )

    def test_p9_2_causal_features_and_invariance(self):
        """P9.2: Lagged features and rolling statistics are strictly causal; truncating future data produces identical features."""
        n_days = 60
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
        np.random.seed(42)
        df = pd.DataFrame({
            "timestamp": dates,
            "demand": np.arange(n_days, dtype=np.float64) * 2.0 + np.random.randn(n_days),
        })

        feat_eng = TimeSeriesFeatureEngineer(
            target_column="demand",
            timestamp_column="timestamp",
            lags=[1, 2, 7],
            rolling_windows=[{"window_size": 7, "metrics": ["mean", "std"]}],
        )

        full_features = feat_eng.fit_transform(df)

        # Truncate to first 30 days
        df_truncated = df.iloc[:30].copy()
        trunc_features = feat_eng.fit_transform(df_truncated)

        # Features at day 29 in both runs MUST be numerically identical
        cols_to_check = feat_eng.get_feature_names()
        row_full_29 = full_features.iloc[29][cols_to_check].values.astype(float)
        row_trunc_29 = trunc_features.iloc[29][cols_to_check].values.astype(float)
        np.testing.assert_allclose(row_full_29, row_trunc_29, rtol=1e-7, atol=1e-7)

    def test_p9_2_naive_and_seasonal_baselines(self):
        """P9.2: Naive and seasonal baselines provide benchmark targets against which ML models compete."""
        n_days = 80
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
        # Weekly seasonality (period 7)
        weekly_pattern = np.array([10.0, 12.0, 15.0, 18.0, 25.0, 30.0, 14.0])
        demand = np.tile(weekly_pattern, n_days // 7 + 1)[:n_days]

        train_df = pd.DataFrame({"timestamp": dates[:66], "demand": demand[:66]})
        test_df = pd.DataFrame({"timestamp": dates[66:], "demand": demand[66:]})

        # Evaluate baselines on split
        baselines = evaluate_baselines_on_split(
            train_df, test_df, target_column="demand", seasonal_period=7
        )
        self.assertEqual(len(baselines), 3)

        sm_baseline = next(b for b in baselines if b.baseline_name == "seasonal_mean")
        lv_baseline = next(b for b in baselines if b.baseline_name == "last_value")

        # Seasonal baseline should have very low MAE on periodic series
        self.assertLess(sm_baseline.metrics["mae"], 1.0)
        self.assertGreater(lv_baseline.metrics["mae"], sm_baseline.metrics["mae"])

    def test_p9_2_inference_delivery_contract(self):
        """P9.2: Time-series delivery inference contract: multi-step horizon prediction with timestamp validation."""
        n_days = 90
        dates = pd.date_range("2026-01-01", periods=n_days, freq="D")
        np.random.seed(42)
        trend = np.linspace(50, 100, n_days)
        noise = np.random.randn(n_days) * 2
        df = pd.DataFrame({"timestamp": dates, "demand": trend + noise})

        train_df = df.iloc[:76].copy()
        test_df = df.iloc[76:].copy()

        feat_eng = TimeSeriesFeatureEngineer(
            target_column="demand",
            timestamp_column="timestamp",
            lags=[1, 2, 7],
            rolling_windows=[{"window_size": 7, "metrics": ["mean"]}],
        )

        train_features = feat_eng.fit_transform(train_df).dropna()
        X_train = train_features[feat_eng.get_feature_names()].values
        y_train = train_features["demand"].values

        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        predictor = TimeSeriesPredictor(
            model=model,
            feature_engineer=feat_eng,
            target_column="demand",
            timestamp_column="timestamp",
            frequency="1D",
        )

        # 14-step horizon forecast
        forecasts = predictor.predict_horizon(history_df=train_df, horizon_steps=14)
        self.assertEqual(len(forecasts), 14)
        self.assertEqual(forecasts[0]["step"], 1)
        self.assertEqual(forecasts[13]["step"], 14)

        # Verify forecasts have increasing timestamps starting day after train end
        first_forecast_ts = pd.to_datetime(forecasts[0]["timestamp"])
        self.assertEqual(first_forecast_ts, train_df["timestamp"].iloc[-1] + pd.Timedelta(days=1))

        # Negative case: horizon <= 0 rejected
        with self.assertRaises(ValueError):
            predictor.predict_horizon(history_df=train_df, horizon_steps=0)

        # Negative case: empty history rejected
        with self.assertRaises(ValueError):
            predictor.predict_horizon(history_df=pd.DataFrame(), horizon_steps=7)


if __name__ == "__main__":
    unittest.main()
