"""Time-Series Forecasting Adapter (Phase P9.2).

Enforces:
1. Strict temporal split contracts: entity-aware, rolling-window, and expanding-window
   backtesting without future lookahead leakage.
2. Lagged feature generation, rolling statistical aggregations, and horizon alignment.
3. Naive and seasonal baselines (LastValue, SeasonalMean, MovingAverage) against which
   ML models must compete.
4. Time-series delivery inference contract: multi-step horizon prediction with timestamp
   input validation.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


class FutureLookaheadLeakageError(ValueError):
    """Raised when test or feature generation leaks future information into training or inference."""
    pass


class InvalidTimestampError(ValueError):
    """Raised when timestamps are out of order, non-parseable, or contain unhandled duplicate keys."""
    pass


@dataclass
class TemporalSplit:
    split_index: int
    train_indices: np.ndarray
    gap_indices: np.ndarray
    test_indices: np.ndarray
    train_start: str
    train_end: str
    gap_start: Optional[str]
    gap_end: Optional[str]
    test_start: str
    test_end: str


@dataclass
class TemporalSplitProtocol:
    protocol_type: str = "rolling_window"  # rolling_window, expanding_window, entity_aware_temporal
    train_window_steps: Optional[int] = 100
    gap_steps: int = 1
    test_horizon_steps: int = 14
    step_stride: int = 14
    n_splits: int = 3
    purge_lookahead_verified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RollingWindowSpec:
    window_size: int
    metrics: List[str] = field(default_factory=lambda: ["mean", "std"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeSeriesFeatureConfig:
    lags: List[int] = field(default_factory=lambda: [1, 2, 7, 14])
    rolling_windows: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"window_size": 7, "metrics": ["mean", "std"]},
            {"window_size": 14, "metrics": ["mean", "min", "max"]},
        ]
    )
    horizon_alignment_steps: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineEvaluation:
    baseline_name: str  # last_value, seasonal_mean, moving_average
    seasonal_period: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastDeliveryContract:
    horizon_steps: int = 14
    validate_temporal_monotonicity: bool = True
    strict_future_lookahead_rejection: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TimeSeriesDatasetAdapter:
    schema_version: str = "v2"
    dataset_id: str = "ts_dataset"
    timestamp_column: str = "timestamp"
    target_column: str = "target"
    entity_columns: List[str] = field(default_factory=list)
    frequency: str = "1D"
    split_protocol: Dict[str, Any] = field(default_factory=lambda: TemporalSplitProtocol().to_dict())
    feature_config: Dict[str, Any] = field(default_factory=lambda: TimeSeriesFeatureConfig().to_dict())
    baseline_evaluations: List[Dict[str, Any]] = field(default_factory=list)
    inference_delivery: Dict[str, Any] = field(default_factory=lambda: ForecastDeliveryContract().to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def synthetic_daily(cls, dataset_id: str = "ts_dataset") -> TimeSeriesDatasetAdapter:
        return cls(
            schema_version="v2",
            dataset_id=dataset_id,
            timestamp_column="timestamp",
            target_column="demand",
            entity_columns=["store_id"],
            frequency="1D",
            split_protocol=TemporalSplitProtocol(
                protocol_type="rolling_window",
                train_window_steps=100,
                gap_steps=1,
                test_horizon_steps=14,
                step_stride=14,
                n_splits=3,
                purge_lookahead_verified=True,
            ).to_dict(),
            feature_config=TimeSeriesFeatureConfig(
                lags=[1, 2, 7, 14],
                rolling_windows=[
                    {"window_size": 7, "metrics": ["mean", "std"]},
                    {"window_size": 14, "metrics": ["mean", "min", "max"]},
                ],
                horizon_alignment_steps=1,
            ).to_dict(),
            baseline_evaluations=[
                BaselineEvaluation(
                    baseline_name="last_value",
                    seasonal_period=None,
                    metrics={"mae": 1.45, "rmse": 1.89, "wape": 0.12},
                ).to_dict(),
                BaselineEvaluation(
                    baseline_name="seasonal_mean",
                    seasonal_period=7,
                    metrics={"mae": 0.85, "rmse": 1.12, "wape": 0.07},
                ).to_dict(),
            ],
            inference_delivery=ForecastDeliveryContract(
                horizon_steps=14,
                validate_temporal_monotonicity=True,
                strict_future_lookahead_rejection=True,
            ).to_dict(),
        )


def parse_and_validate_timestamps(
    df: pd.DataFrame,
    timestamp_column: str,
    entity_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Validate that timestamps are parseable, strictly monotonic, and free of duplicate keys."""
    if timestamp_column not in df.columns:
        raise ValueError(f"Timestamp column '{timestamp_column}' not found in dataframe")

    out_df = df.copy()
    try:
        out_df[timestamp_column] = pd.to_datetime(out_df[timestamp_column])
    except Exception as e:
        raise InvalidTimestampError(f"Failed to parse timestamp column '{timestamp_column}': {e}")

    # Check for NaNs
    if out_df[timestamp_column].isna().any():
        raise InvalidTimestampError(f"Timestamp column '{timestamp_column}' contains NaT/NaN values")

    entity_cols = entity_columns or []
    if entity_cols:
        for ent in entity_cols:
            if ent not in out_df.columns:
                raise ValueError(f"Entity column '{ent}' not found in dataframe")

        # Group by entities and verify monotonicity within each entity
        grouped = out_df.groupby(entity_cols)
        for name, group in grouped:
            ts_series = group[timestamp_column].values
            if not np.all(ts_series[:-1] < ts_series[1:]):
                raise InvalidTimestampError(
                    f"Timestamps for entity {name} are not strictly monotonically increasing or contain duplicates"
                )
        # Sort by entities and timestamp
        out_df = out_df.sort_values(by=entity_cols + [timestamp_column]).reset_index(drop=True)
    else:
        # Global monotonicity check
        ts_series = out_df[timestamp_column].values
        if not np.all(ts_series[:-1] < ts_series[1:]):
            raise InvalidTimestampError(
                "Timestamps are not strictly monotonically increasing or contain duplicates"
            )
        out_df = out_df.sort_values(by=timestamp_column).reset_index(drop=True)

    return out_df


def verify_temporal_split_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    timestamp_column: str,
    gap_steps: int = 0,
    entity_columns: Optional[List[str]] = None,
) -> None:
    """Verify that no test timestamps precede or equal train timestamps (leakage check)."""
    entity_cols = entity_columns or []
    if entity_cols:
        train_groups = dict(tuple(train_df.groupby(entity_cols)))
        test_groups = dict(tuple(test_df.groupby(entity_cols)))
        common_entities = set(train_groups.keys()).intersection(test_groups.keys())
        for ent in common_entities:
            train_max = train_groups[ent][timestamp_column].max()
            test_min = test_groups[ent][timestamp_column].min()
            if test_min <= train_max:
                raise FutureLookaheadLeakageError(
                    f"Temporal leakage detected for entity {ent}: test min ({test_min}) <= train max ({train_max})"
                )
    else:
        train_max = train_df[timestamp_column].max()
        test_min = test_df[timestamp_column].min()
        if test_min <= train_max:
            raise FutureLookaheadLeakageError(
                f"Temporal leakage detected: test min ({test_min}) <= train max ({train_max})"
            )


class TemporalSplitter:
    """Generate rolling-window, expanding-window, and entity-aware temporal splits."""

    def __init__(self, protocol: TemporalSplitProtocol, timestamp_column: str, entity_columns: Optional[List[str]] = None):
        self.protocol = protocol
        self.timestamp_column = timestamp_column
        self.entity_columns = entity_columns or []

    def split(self, df: pd.DataFrame) -> List[TemporalSplit]:
        validated_df = parse_and_validate_timestamps(df, self.timestamp_column, self.entity_columns)
        n_rows = len(validated_df)

        if self.entity_columns:
            return self._split_entity_aware(validated_df)

        unique_timestamps = validated_df[self.timestamp_column].drop_duplicates().values
        T = len(unique_timestamps)

        splits: List[TemporalSplit] = []
        p = self.protocol
        horizon = p.test_horizon_steps
        gap = p.gap_steps
        stride = p.step_stride
        n_splits = p.n_splits

        # Compute required total steps per split
        if p.protocol_type == "rolling_window":
            train_window = p.train_window_steps or 100
            total_window = train_window + gap + horizon
            if total_window > T:
                raise ValueError(
                    f"Dataset temporal length ({T}) is smaller than required rolling window ({total_window})"
                )

            # Generate splits from right to left or left to right
            # Left to right
            start_pos = 0
            while len(splits) < n_splits and (start_pos + total_window) <= T:
                train_slice_ts = unique_timestamps[start_pos : start_pos + train_window]
                gap_slice_ts = unique_timestamps[start_pos + train_window : start_pos + train_window + gap] if gap > 0 else []
                test_slice_ts = unique_timestamps[start_pos + train_window + gap : start_pos + total_window]

                train_mask = validated_df[self.timestamp_column].isin(train_slice_ts)
                gap_mask = validated_df[self.timestamp_column].isin(gap_slice_ts) if gap > 0 else np.zeros(n_rows, dtype=bool)
                test_mask = validated_df[self.timestamp_column].isin(test_slice_ts)

                train_idx = np.where(train_mask)[0]
                gap_idx = np.where(gap_mask)[0]
                test_idx = np.where(test_mask)[0]

                # Verify leakage
                verify_temporal_split_no_leakage(
                    validated_df.iloc[train_idx],
                    validated_df.iloc[test_idx],
                    self.timestamp_column,
                    gap,
                    self.entity_columns,
                )

                splits.append(
                    TemporalSplit(
                        split_index=len(splits),
                        train_indices=train_idx,
                        gap_indices=gap_idx,
                        test_indices=test_idx,
                        train_start=str(train_slice_ts[0]),
                        train_end=str(train_slice_ts[-1]),
                        gap_start=str(gap_slice_ts[0]) if len(gap_slice_ts) > 0 else None,
                        gap_end=str(gap_slice_ts[-1]) if len(gap_slice_ts) > 0 else None,
                        test_start=str(test_slice_ts[0]),
                        test_end=str(test_slice_ts[-1]),
                    )
                )
                start_pos += stride

        elif p.protocol_type == "expanding_window":
            min_train = p.train_window_steps or max(20, T // (n_splits + 1))
            total_required = min_train + gap + horizon
            if total_required > T:
                raise ValueError(
                    f"Dataset temporal length ({T}) is smaller than expanding window min ({total_required})"
                )

            curr_train_len = min_train
            while len(splits) < n_splits and (curr_train_len + gap + horizon) <= T:
                train_slice_ts = unique_timestamps[0:curr_train_len]
                gap_slice_ts = unique_timestamps[curr_train_len : curr_train_len + gap] if gap > 0 else []
                test_slice_ts = unique_timestamps[curr_train_len + gap : curr_train_len + gap + horizon]

                train_mask = validated_df[self.timestamp_column].isin(train_slice_ts)
                gap_mask = validated_df[self.timestamp_column].isin(gap_slice_ts) if gap > 0 else np.zeros(n_rows, dtype=bool)
                test_mask = validated_df[self.timestamp_column].isin(test_slice_ts)

                train_idx = np.where(train_mask)[0]
                gap_idx = np.where(gap_mask)[0]
                test_idx = np.where(test_mask)[0]

                verify_temporal_split_no_leakage(
                    validated_df.iloc[train_idx],
                    validated_df.iloc[test_idx],
                    self.timestamp_column,
                    gap,
                    self.entity_columns,
                )

                splits.append(
                    TemporalSplit(
                        split_index=len(splits),
                        train_indices=train_idx,
                        gap_indices=gap_idx,
                        test_indices=test_idx,
                        train_start=str(train_slice_ts[0]),
                        train_end=str(train_slice_ts[-1]),
                        gap_start=str(gap_slice_ts[0]) if len(gap_slice_ts) > 0 else None,
                        gap_end=str(gap_slice_ts[-1]) if len(gap_slice_ts) > 0 else None,
                        test_start=str(test_slice_ts[0]),
                        test_end=str(test_slice_ts[-1]),
                    )
                )
                curr_train_len += stride
        else:
            raise ValueError(f"Unsupported protocol type: {p.protocol_type}")

        if not splits:
            raise ValueError("Unable to generate any valid temporal splits given dataset size and stride.")

        return splits

    def _split_entity_aware(self, df: pd.DataFrame) -> List[TemporalSplit]:
        """Align splits globally across multiple entities without lookahead leakage."""
        unique_timestamps = sorted(df[self.timestamp_column].unique())
        T = len(unique_timestamps)
        p = self.protocol
        train_window = p.train_window_steps or max(20, T // (p.n_splits + 1))
        gap = p.gap_steps
        horizon = p.test_horizon_steps
        stride = p.step_stride
        splits = []

        start_pos = 0
        while len(splits) < p.n_splits and (start_pos + train_window + gap + horizon) <= T:
            train_ts = unique_timestamps[start_pos : start_pos + train_window]
            gap_ts = unique_timestamps[start_pos + train_window : start_pos + train_window + gap] if gap > 0 else []
            test_ts = unique_timestamps[start_pos + train_window + gap : start_pos + train_window + gap + horizon]

            train_mask = df[self.timestamp_column].isin(train_ts)
            gap_mask = df[self.timestamp_column].isin(gap_ts) if gap > 0 else np.zeros(len(df), dtype=bool)
            test_mask = df[self.timestamp_column].isin(test_ts)

            train_idx = np.where(train_mask)[0]
            gap_idx = np.where(gap_mask)[0]
            test_idx = np.where(test_mask)[0]

            verify_temporal_split_no_leakage(
                df.iloc[train_idx],
                df.iloc[test_idx],
                self.timestamp_column,
                gap,
                self.entity_columns,
            )

            splits.append(
                TemporalSplit(
                    split_index=len(splits),
                    train_indices=train_idx,
                    gap_indices=gap_idx,
                    test_indices=test_idx,
                    train_start=str(train_ts[0]),
                    train_end=str(train_ts[-1]),
                    gap_start=str(gap_ts[0]) if len(gap_ts) > 0 else None,
                    gap_end=str(gap_ts[-1]) if len(gap_ts) > 0 else None,
                    test_start=str(test_ts[0]),
                    test_end=str(test_ts[-1]),
                )
            )
            start_pos += stride

        if not splits:
            raise ValueError("Unable to generate valid entity-aware temporal splits.")
        return splits


class TimeSeriesFeatureEngineer:
    """Causal lagged feature generator and rolling statistical aggregator."""

    def __init__(
        self,
        target_column: str,
        timestamp_column: str,
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[Dict[str, Any]]] = None,
        entity_columns: Optional[List[str]] = None,
    ):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.lags = lags or [1, 2, 7, 14]
        self.rolling_windows = rolling_windows or [
            {"window_size": 7, "metrics": ["mean", "std"]},
            {"window_size": 14, "metrics": ["mean", "min", "max"]},
        ]
        self.entity_columns = entity_columns or []

        # Validate that all lags are strictly positive (>= 1) to eliminate future lookahead leakage
        for lag in self.lags:
            if lag < 1:
                raise FutureLookaheadLeakageError(
                    f"Invalid lag {lag}: lags must be strictly positive (>= 1) to prevent future lookahead leakage."
                )

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate strictly causal features on df without future lookahead."""
        df_sorted = parse_and_validate_timestamps(df, self.timestamp_column, self.entity_columns)

        if self.entity_columns:
            return df_sorted.groupby(self.entity_columns, group_keys=False).apply(self._transform_group)
        else:
            return self._transform_group(df_sorted)

    def _transform_group(self, group_df: pd.DataFrame) -> pd.DataFrame:
        out = group_df.copy()
        target = group_df[self.target_column]

        # 1. Causal Lags
        for lag in self.lags:
            out[f"lag_{lag}"] = target.shift(lag)

        # 2. Rolling Statistics (strictly lagged so current target is not leaked)
        # Shift target by 1 first so rolling window over lag_1 contains only past values
        shifted_target = target.shift(1)
        for rw in self.rolling_windows:
            w_size = rw["window_size"]
            metrics = rw.get("metrics", ["mean"])
            roller = shifted_target.rolling(window=w_size, min_periods=max(1, w_size // 2))

            for metric in metrics:
                col_name = f"rolling_{metric}_{w_size}"
                if metric == "mean":
                    out[col_name] = roller.mean()
                elif metric == "std":
                    out[col_name] = roller.std().fillna(0.0)
                elif metric == "min":
                    out[col_name] = roller.min()
                elif metric == "max":
                    out[col_name] = roller.max()
                elif metric == "sum":
                    out[col_name] = roller.sum()

        return out

    def get_feature_names(self) -> List[str]:
        names = []
        for lag in self.lags:
            names.append(f"lag_{lag}")
        for rw in self.rolling_windows:
            w_size = rw["window_size"]
            for metric in rw.get("metrics", ["mean"]):
                names.append(f"rolling_{metric}_{w_size}")
        return names


# Baseline Forecast Models
class LastValueBaseline:
    """Naive Last-Value (Random Walk) forecasting baseline: y_{t+h} = y_t."""

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.last_value_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> LastValueBaseline:
        self.last_value_ = float(train_df[self.target_column].iloc[-1])
        return self

    def predict(self, horizon_steps: int) -> np.ndarray:
        if self.last_value_ is None:
            raise ValueError("Baseline is not fitted.")
        return np.full(horizon_steps, self.last_value_, dtype=np.float64)


class SeasonalMeanBaseline:
    """Seasonal Naive / Mean forecasting baseline with period m: y_{t+h} = y_{t+h - m}."""

    def __init__(self, target_column: str, seasonal_period: int = 7):
        self.target_column = target_column
        self.seasonal_period = seasonal_period
        self.seasonal_values_: Optional[np.ndarray] = None

    def fit(self, train_df: pd.DataFrame) -> SeasonalMeanBaseline:
        vals = train_df[self.target_column].values
        if len(vals) < self.seasonal_period:
            self.seasonal_values_ = np.pad(vals, (self.seasonal_period - len(vals), 0), mode="edge")
        else:
            self.seasonal_values_ = vals[-self.seasonal_period :]
        return self

    def predict(self, horizon_steps: int) -> np.ndarray:
        if self.seasonal_values_ is None:
            raise ValueError("Baseline is not fitted.")
        preds = []
        m = self.seasonal_period
        for h in range(horizon_steps):
            preds.append(self.seasonal_values_[h % m])
        return np.array(preds, dtype=np.float64)


class MovingAverageBaseline:
    """Moving Average baseline over past window W: y_{t+h} = mean(y_{t-W+1..t})."""

    def __init__(self, target_column: str, window_size: int = 14):
        self.target_column = target_column
        self.window_size = window_size
        self.mean_value_: Optional[float] = None

    def fit(self, train_df: pd.DataFrame) -> MovingAverageBaseline:
        vals = train_df[self.target_column].values[-self.window_size :]
        self.mean_value_ = float(np.mean(vals))
        return self

    def predict(self, horizon_steps: int) -> np.ndarray:
        if self.mean_value_ is None:
            raise ValueError("Baseline is not fitted.")
        return np.full(horizon_steps, self.mean_value_, dtype=np.float64)


def compute_forecasting_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_history: Optional[np.ndarray] = None,
    seasonality: int = 1,
) -> Dict[str, float]:
    """Compute standard forecasting error metrics: MAE, RMSE, WAPE, and MASE."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)
    errors = y_t - y_p

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    sum_true = float(np.sum(np.abs(y_t)))
    wape = float(np.sum(np.abs(errors)) / max(sum_true, 1e-8))

    mase: Optional[float] = None
    if y_train_history is not None and len(y_train_history) > seasonality:
        scale = np.mean(np.abs(y_train_history[seasonality:] - y_train_history[:-seasonality]))
        if scale > 1e-8:
            mase = float(mae / scale)

    res = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "wape": round(wape, 4),
    }
    if mase is not None:
        res["mase"] = round(mase, 4)
    return res


def evaluate_baselines_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    seasonal_period: int = 7,
) -> List[BaselineEvaluation]:
    """Evaluate naive and seasonal baselines on test_df, providing benchmark comparison."""
    horizon = len(test_df)
    y_true = test_df[target_column].values
    y_train = train_df[target_column].values

    # 1. Last Value Baseline
    lv = LastValueBaseline(target_column).fit(train_df)
    lv_pred = lv.predict(horizon)
    lv_metrics = compute_forecasting_metrics(y_true, lv_pred, y_train, seasonality=1)

    # 2. Seasonal Mean Baseline
    sm = SeasonalMeanBaseline(target_column, seasonal_period=seasonal_period).fit(train_df)
    sm_pred = sm.predict(horizon)
    sm_metrics = compute_forecasting_metrics(y_true, sm_pred, y_train, seasonality=seasonal_period)

    # 3. Moving Average Baseline
    ma = MovingAverageBaseline(target_column, window_size=max(seasonal_period * 2, 14)).fit(train_df)
    ma_pred = ma.predict(horizon)
    ma_metrics = compute_forecasting_metrics(y_true, ma_pred, y_train, seasonality=1)

    return [
        BaselineEvaluation(baseline_name="last_value", metrics=lv_metrics),
        BaselineEvaluation(baseline_name="seasonal_mean", seasonal_period=seasonal_period, metrics=sm_metrics),
        BaselineEvaluation(baseline_name="moving_average", metrics=ma_metrics),
    ]


class TimeSeriesPredictor:
    """Production Time-Series Forecasting Delivery Predictor with timestamp input validation."""

    def __init__(
        self,
        model: Any,
        feature_engineer: TimeSeriesFeatureEngineer,
        target_column: str,
        timestamp_column: str,
        frequency: str = "1D",
        baselines: Optional[List[BaselineEvaluation]] = None,
        entity_columns: Optional[List[str]] = None,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.frequency = frequency
        self.baselines = baselines or []
        self.entity_columns = entity_columns or []

    def predict_horizon(
        self,
        history_df: pd.DataFrame,
        horizon_steps: int,
        entity_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Validate input timestamps, extract strictly causal features, and generate multi-step forecasts."""
        if horizon_steps < 1:
            raise ValueError(f"horizon_steps must be >= 1, got {horizon_steps}")

        if history_df.empty:
            raise ValueError("history_df cannot be empty for horizon forecasting")

        # Parse and validate timestamps
        validated_history = parse_and_validate_timestamps(
            history_df, self.timestamp_column, self.entity_columns
        )

        last_timestamp = validated_history[self.timestamp_column].iloc[-1]
        freq_offset = pd.tseries.frequencies.to_offset(self.frequency)

        # Generate future timestamps
        future_timestamps = [last_timestamp + (i + 1) * freq_offset for i in range(horizon_steps)]

        # Recursive multi-step forecasting
        current_data = validated_history.copy()
        forecasts: List[Dict[str, Any]] = []

        feature_cols = self.feature_engineer.get_feature_names()

        for step in range(1, horizon_steps + 1):
            target_ts = future_timestamps[step - 1]
            transformed = self.feature_engineer.fit_transform(current_data)
            latest_features = transformed.iloc[-1:][feature_cols]

            # Impute any initial lag NaNs with last known observation
            latest_features = latest_features.bfill(axis=1).fillna(0.0)

            # Predict next step point forecast
            point_pred = float(self.model.predict(latest_features.values)[0])

            rec: Dict[str, Any] = {
                "step": step,
                "timestamp": target_ts.isoformat(),
                "point_forecast": round(point_pred, 4),
            }
            if entity_id is not None:
                rec["entity_id"] = entity_id
            forecasts.append(rec)

            # Append synthetic forecast step into current_data for autoregressive lag propagation
            new_row = {
                self.timestamp_column: target_ts,
                self.target_column: point_pred,
            }
            if self.entity_columns and entity_id:
                for ec in self.entity_columns:
                    new_row[ec] = entity_id

            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)

        return forecasts
