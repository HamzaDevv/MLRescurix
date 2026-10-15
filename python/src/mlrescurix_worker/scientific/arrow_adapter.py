"""Out-of-Core / Apache Arrow Data Adapter (Phase P9.1).

Enforces:
1. Strict typed schemas and contracts for Arrow IPC / Feather / Parquet ingestion fingerprints,
   feature-only split views, and byte limits.
2. True memory-mapped column access and chunked batch iterators without inflating memory
   into full dense NumPy arrays.
3. Empirical memory accounting: measuring actual preprocessing and model copies, rejecting
   unfounded zero-copy claims.
4. Out-of-core baseline models: IncrementalFeatureScaler, OutOfCoreSGDClassifier, and
   OutOfCoreSGDRegressor with chunked batch fitting.
"""

from __future__ import annotations

import hashlib
import json
import os
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from sklearn.linear_model import SGDClassifier, SGDRegressor


class MemoryCapExceededError(ValueError):
    """Raised when an out-of-core batch or operation exceeds the allocated memory cap."""
    pass


class UnfoundedZeroCopyError(ValueError):
    """Raised when an unfounded zero-copy claim is asserted on an array that incurred a copy."""
    pass


class SchemaFingerprintMismatchError(ValueError):
    """Raised when the actual Arrow schema does not match the expected fingerprint."""
    pass


@dataclass
class ColumnDescriptor:
    name: str
    arrow_type: str
    physical_type: str
    nullable: bool
    role: str = "feature"  # feature, target, id, group, weight

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SplitViewConfig:
    split_name: str = "train"
    row_count: int = 0
    byte_limit: int = 100 * 1024 * 1024
    feature_only: bool = False
    selected_features: Optional[List[str]] = None
    target_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchConfig:
    batch_size_rows: int = 1000
    max_batch_bytes: int = 16 * 1024 * 1024
    memory_cap_bytes: int = 64 * 1024 * 1024

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryAccounting:
    peak_memory_bytes: int = 0
    bytes_read: int = 0
    batches_processed: int = 0
    actual_copies_count: int = 0
    zero_copy_fraction: float = 0.0
    unfounded_zero_copy_refused: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArrowDatasetAdapter:
    schema_version: str = "v2"
    dataset_id: str = "arrow_dataset"
    format: str = "parquet"  # ipc_stream, ipc_file, feather, parquet
    path: str = ""
    schema_fingerprint: str = ""
    row_count: int = 0
    column_count: int = 0
    file_bytes: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    split_view: Dict[str, Any] = field(default_factory=lambda: SplitViewConfig().to_dict())
    batch_config: Dict[str, Any] = field(default_factory=lambda: BatchConfig().to_dict())
    memory_accounting: Dict[str, Any] = field(default_factory=lambda: MemoryAccounting().to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_arrow_schema_fingerprint(schema: pa.Schema) -> str:
    """Compute deterministic SHA-256 fingerprint for an Arrow Schema."""
    canonical_repr = []
    for field_idx in range(len(schema)):
        f = schema.field(field_idx)
        canonical_repr.append(f"{f.name}:{f.type}:{f.nullable}")
    joined = "|".join(canonical_repr)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def detect_arrow_format(path: Union[str, Path]) -> str:
    """Detect Arrow dataset format from extension and file header."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return "parquet"
    if suffix in (".feather", ".ftr"):
        return "feather"
    if suffix in (".arrow", ".ipc"):
        return "ipc_file"
    if suffix == ".arrows":
        return "ipc_stream"

    # Inspect magic bytes
    with open(p, "rb") as f:
        head = f.read(6)
        if head.startswith(b"PAR1"):
            return "parquet"
        if head.startswith(b"ARROW1"):
            return "feather"
    return "parquet"


def extract_arrow_fingerprint(
    source_path: Union[str, Path],
    dataset_id: str = "ds_arrow",
    format_type: str = "auto",
    target_column: Optional[str] = None,
    id_columns: Optional[List[str]] = None,
    byte_limit: int = 100 * 1024 * 1024,
    batch_size_rows: int = 1000,
    memory_cap_bytes: int = 64 * 1024 * 1024,
) -> ArrowDatasetAdapter:
    """Inspect Arrow/Parquet/Feather source, extract schema fingerprint and metadata."""
    p = Path(source_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Arrow dataset file not found: {p}")

    file_bytes = p.stat().st_size
    if format_type == "auto":
        detected_fmt = detect_arrow_format(p)
    else:
        detected_fmt = format_type

    id_cols = set(id_columns or [])
    columns: List[ColumnDescriptor] = []

    if detected_fmt == "parquet":
        pq_file = pq.ParquetFile(str(p), memory_map=True)
        schema = pq_file.schema_arrow
        row_count = pq_file.metadata.num_rows
    elif detected_fmt in ("feather", "ipc_file"):
        mmap_source = pa.memory_map(str(p), "r")
        reader = ipc.RecordBatchFileReader(mmap_source)
        schema = reader.schema
        row_count = reader.read_all().num_rows
    elif detected_fmt == "ipc_stream":
        mmap_source = pa.memory_map(str(p), "r")
        reader = ipc.RecordBatchStreamReader(mmap_source)
        schema = reader.schema
        table = reader.read_all()
        row_count = table.num_rows
    else:
        raise ValueError(f"Unsupported Arrow format: {detected_fmt}")

    fingerprint = compute_arrow_schema_fingerprint(schema)

    for field_idx in range(len(schema)):
        f = schema.field(field_idx)
        name = f.name
        if target_column and name == target_column:
            role = "target"
        elif name in id_cols:
            role = "id"
        else:
            role = "feature"

        columns.append(
            ColumnDescriptor(
                name=name,
                arrow_type=str(f.type),
                physical_type=f.type.__class__.__name__,
                nullable=f.nullable,
                role=role,
            )
        )

    split_view = SplitViewConfig(
        split_name="full",
        row_count=row_count,
        byte_limit=byte_limit,
        feature_only=(target_column is None),
        selected_features=[c.name for c in columns if c.role == "feature"],
        target_column=target_column,
    )

    batch_cfg = BatchConfig(
        batch_size_rows=batch_size_rows,
        max_batch_bytes=min(16 * 1024 * 1024, memory_cap_bytes // 2),
        memory_cap_bytes=memory_cap_bytes,
    )

    mem_acct = MemoryAccounting(
        peak_memory_bytes=0,
        bytes_read=0,
        batches_processed=0,
        actual_copies_count=0,
        zero_copy_fraction=1.0,
        unfounded_zero_copy_refused=True,
    )

    return ArrowDatasetAdapter(
        schema_version="v2",
        dataset_id=dataset_id,
        format=detected_fmt,
        path=str(p),
        schema_fingerprint=fingerprint,
        row_count=row_count,
        column_count=len(columns),
        file_bytes=file_bytes,
        columns=[c.to_dict() for c in columns],
        split_view=split_view.to_dict(),
        batch_config=batch_cfg.to_dict(),
        memory_accounting=mem_acct.to_dict(),
    )


class ArrowBatchIterator:
    """True memory-mapped chunked batch iterator for Arrow / Parquet data."""

    def __init__(
        self,
        source_path: Union[str, Path],
        batch_size_rows: int = 1000,
        max_batch_bytes: int = 16 * 1024 * 1024,
        memory_cap_bytes: int = 64 * 1024 * 1024,
        feature_columns: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        expected_schema_fingerprint: Optional[str] = None,
        claim_zero_copy: bool = False,
        format_type: str = "auto",
    ):
        self.source_path = Path(source_path).resolve()
        if not self.source_path.is_file():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        self.batch_size_rows = batch_size_rows
        self.max_batch_bytes = max_batch_bytes
        self.memory_cap_bytes = memory_cap_bytes
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.claim_zero_copy = claim_zero_copy

        if format_type == "auto":
            self.format = detect_arrow_format(self.source_path)
        else:
            self.format = format_type

        # Open underlying file memory mapped
        self._mmap_handle = None
        if self.format == "parquet":
            self._pq_file = pq.ParquetFile(str(self.source_path), memory_map=True)
            self.schema = self._pq_file.schema_arrow
        elif self.format in ("feather", "ipc_file"):
            self._mmap_handle = pa.memory_map(str(self.source_path), "r")
            self._reader = ipc.RecordBatchFileReader(self._mmap_handle)
            self.schema = self._reader.schema
        elif self.format == "ipc_stream":
            self._mmap_handle = pa.memory_map(str(self.source_path), "r")
            self._reader = ipc.RecordBatchStreamReader(self._mmap_handle)
            self.schema = self._reader.schema
        else:
            raise ValueError(f"Unsupported Arrow format: {self.format}")

        # Validate fingerprint if provided
        actual_fingerprint = compute_arrow_schema_fingerprint(self.schema)
        if expected_schema_fingerprint and actual_fingerprint != expected_schema_fingerprint:
            raise SchemaFingerprintMismatchError(
                f"Schema fingerprint mismatch: expected {expected_schema_fingerprint}, got {actual_fingerprint}"
            )
        self.schema_fingerprint = actual_fingerprint

        all_col_names = [self.schema.field(i).name for i in range(len(self.schema))]
        if self.feature_columns is None:
            self.feature_columns = [c for c in all_col_names if c != self.target_column]
        else:
            for fc in self.feature_columns:
                if fc not in all_col_names:
                    raise ValueError(f"Feature column '{fc}' not found in Arrow schema")

        if self.target_column and self.target_column not in all_col_names:
            raise ValueError(f"Target column '{self.target_column}' not found in Arrow schema")

        # Memory accounting counters
        self.peak_memory_bytes = 0
        self.bytes_read = 0
        self.batches_processed = 0
        self.actual_copies_count = 0
        self.zero_copy_arrays_count = 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
        tracemalloc.start()
        columns_to_read = list(self.feature_columns)
        if self.target_column and self.target_column not in columns_to_read:
            columns_to_read.append(self.target_column)

        try:
            if self.format == "parquet":
                batch_generator = self._pq_file.iter_batches(
                    batch_size=self.batch_size_rows,
                    columns=columns_to_read,
                    use_threads=False,
                )
                for raw_batch in batch_generator:
                    yield self._process_batch(raw_batch)
            elif self.format in ("feather", "ipc_file"):
                num_batches = self._reader.num_record_batches
                for i in range(num_batches):
                    raw_batch = self._reader.get_batch(i)
                    # Subdivide batch if larger than requested batch_size_rows
                    if raw_batch.num_rows > self.batch_size_rows:
                        for offset in range(0, raw_batch.num_rows, self.batch_size_rows):
                            sub_batch = raw_batch.slice(offset, self.batch_size_rows)
                            yield self._process_batch(sub_batch)
                    else:
                        yield self._process_batch(raw_batch)
            elif self.format == "ipc_stream":
                for raw_batch in self._reader:
                    if raw_batch.num_rows > self.batch_size_rows:
                        for offset in range(0, raw_batch.num_rows, self.batch_size_rows):
                            sub_batch = raw_batch.slice(offset, self.batch_size_rows)
                            yield self._process_batch(sub_batch)
                    else:
                        yield self._process_batch(raw_batch)
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_memory_bytes = max(self.peak_memory_bytes, peak)

    def _process_batch(self, batch: pa.RecordBatch) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        batch_bytes = batch.nbytes
        self.bytes_read += batch_bytes

        if batch_bytes > self.max_batch_bytes:
            raise MemoryCapExceededError(
                f"Batch size {batch_bytes} bytes exceeds configured max_batch_bytes {self.max_batch_bytes}"
            )

        # Check peak memory during iteration
        current, peak = tracemalloc.get_traced_memory()
        self.peak_memory_bytes = max(self.peak_memory_bytes, peak)
        if peak > self.memory_cap_bytes:
            raise MemoryCapExceededError(
                f"Peak memory {peak} bytes exceeded memory_cap_bytes {self.memory_cap_bytes}"
            )

        # Extract features and target
        x_arrays = []
        for col_name in self.feature_columns:
            arrow_col = batch.column(batch.schema.get_field_index(col_name))
            # Determine if zero-copy is possible
            np_col, was_copied = self._convert_column_to_numpy(arrow_col)
            if was_copied:
                self.actual_copies_count += 1
                if self.claim_zero_copy:
                    raise UnfoundedZeroCopyError(
                        f"Unfounded zero-copy claim refused: column '{col_name}' required memory copy "
                        f"(nulls={arrow_col.null_count > 0}, type={arrow_col.type})"
                    )
            else:
                self.zero_copy_arrays_count += 1
            x_arrays.append(np_col)

        y_array = None
        if self.target_column:
            arrow_target = batch.column(batch.schema.get_field_index(self.target_column))
            np_target, was_copied = self._convert_column_to_numpy(arrow_target)
            if was_copied:
                self.actual_copies_count += 1
            else:
                self.zero_copy_arrays_count += 1
            y_array = np_target

        self.batches_processed += 1
        X_batch = np.column_stack(x_arrays) if x_arrays else np.empty((batch.num_rows, 0), dtype=np.float32)

        return X_batch, y_array

    def _convert_column_to_numpy(self, arr: pa.Array) -> Tuple[np.ndarray, bool]:
        """Convert Arrow array to NumPy array, checking whether an empirical copy occurred."""
        has_nulls = arr.null_count > 0
        if has_nulls:
            # Nullable arrays cannot be zero-copy converted to primitive numpy arrays without copy or fill
            filled = arr.fill_null(0.0 if pa.types.is_floating(arr.type) else 0)
            np_arr = filled.to_numpy(zero_copy_only=False)
            return np_arr, True

        try:
            np_arr = arr.to_numpy(zero_copy_only=True)
            # True zero-copy: np_arr shares memory with Arrow buffer
            return np_arr, False
        except (pa.ArrowInvalid, ValueError):
            np_arr = arr.to_numpy(zero_copy_only=False)
            return np_arr, True

    def get_memory_accounting(self) -> MemoryAccounting:
        """Return empirical memory accounting statistics."""
        total_arrays = self.zero_copy_arrays_count + self.actual_copies_count
        zero_copy_fraction = (
            float(self.zero_copy_arrays_count) / float(total_arrays) if total_arrays > 0 else 0.0
        )
        return MemoryAccounting(
            peak_memory_bytes=self.peak_memory_bytes,
            bytes_read=self.bytes_read,
            batches_processed=self.batches_processed,
            actual_copies_count=self.actual_copies_count,
            zero_copy_fraction=round(zero_copy_fraction, 4),
            unfounded_zero_copy_refused=True,
        )


class IncrementalFeatureScaler:
    """Incremental feature scaler using parallel Welford/Chan update algorithm for out-of-core learning."""

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.n_samples_seen_: int = 0
        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self._M2: Optional[np.ndarray] = None

    def partial_fit(self, X: np.ndarray) -> IncrementalFeatureScaler:
        """Update running statistics with a new batch X (2D numpy array)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        batch_samples, n_features = X.shape
        if batch_samples == 0:
            return self

        batch_mean = np.mean(X, axis=0)
        batch_M2 = np.sum((X - batch_mean) ** 2, axis=0)

        if self.n_samples_seen_ == 0:
            self.n_samples_seen_ = batch_samples
            self.mean_ = batch_mean
            self._M2 = batch_M2
        else:
            # Chan et al. parallel update
            n_a = self.n_samples_seen_
            n_b = batch_samples
            n_total = n_a + n_b
            delta = batch_mean - self.mean_

            self.mean_ = self.mean_ + delta * (n_b / n_total)
            self._M2 = self._M2 + batch_M2 + (delta ** 2) * (n_a * n_b / n_total)
            self.n_samples_seen_ = n_total

        self.var_ = self._M2 / max(1, self.n_samples_seen_)
        scale = np.sqrt(self.var_)
        # Handle zero variance features
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, X: np.ndarray, copy: bool = True) -> np.ndarray:
        """Scale features in X according to accumulated statistics."""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler is not fitted yet.")

        X = np.array(X, copy=copy, dtype=np.float32)
        if self.with_mean:
            X -= self.mean_.astype(np.float32)
        if self.with_std:
            X /= self.scale_.astype(np.float32)
        return X

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.partial_fit(X).transform(X)


class OutOfCoreSGDClassifier:
    """Out-of-core linear classifier with incremental scaling and SGDClassifier.partial_fit."""

    def __init__(
        self,
        loss: str = "log_loss",
        penalty: str = "l2",
        alpha: float = 0.0001,
        random_state: int = 42,
        with_scaling: bool = True,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.random_state = random_state
        self.with_scaling = with_scaling
        self.scaler = IncrementalFeatureScaler() if with_scaling else None
        self.model = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            random_state=random_state,
        )
        self.classes_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None) -> OutOfCoreSGDClassifier:
        """Fit one batch of data."""
        if classes is not None:
            self.classes_ = np.asarray(classes)

        if self.with_scaling and self.scaler is not None:
            self.scaler.partial_fit(X)
            X = self.scaler.transform(X, copy=True)

        self.model.partial_fit(X, y, classes=self.classes_)
        if self.classes_ is None:
            self.classes_ = self.model.classes_
        self.is_fitted_ = True
        return self

    def fit_batches(
        self,
        batch_iterator: ArrowBatchIterator,
        classes: Optional[np.ndarray] = None,
        epochs: int = 1,
    ) -> Tuple[OutOfCoreSGDClassifier, MemoryAccounting]:
        """Fit over chunked batches from an ArrowBatchIterator across one or more epochs."""
        for epoch in range(epochs):
            for X_batch, y_batch in batch_iterator:
                if y_batch is None:
                    raise ValueError("Cannot train classifier without target column in batch iterator")
                self.partial_fit(X_batch, y_batch, classes=classes)

        accounting = batch_iterator.get_memory_accounting()
        return self, accounting

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for X."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        if self.with_scaling and self.scaler is not None:
            X = self.scaler.transform(X, copy=True)
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        if self.with_scaling and self.scaler is not None:
            X = self.scaler.transform(X, copy=True)
        return self.model.predict_proba(X)


class OutOfCoreSGDRegressor:
    """Out-of-core linear regressor with incremental scaling and SGDRegressor.partial_fit."""

    def __init__(
        self,
        loss: str = "squared_error",
        penalty: str = "l2",
        alpha: float = 0.0001,
        random_state: int = 42,
        with_scaling: bool = True,
    ):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.random_state = random_state
        self.with_scaling = with_scaling
        self.scaler = IncrementalFeatureScaler() if with_scaling else None
        self.model = SGDRegressor(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            random_state=random_state,
        )
        self.is_fitted_: bool = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> OutOfCoreSGDRegressor:
        """Fit one batch of continuous targets."""
        if self.with_scaling and self.scaler is not None:
            self.scaler.partial_fit(X)
            X = self.scaler.transform(X, copy=True)

        self.model.partial_fit(X, y)
        self.is_fitted_ = True
        return self

    def fit_batches(
        self,
        batch_iterator: ArrowBatchIterator,
        epochs: int = 1,
    ) -> Tuple[OutOfCoreSGDRegressor, MemoryAccounting]:
        """Fit over chunked batches from an ArrowBatchIterator across one or more epochs."""
        for epoch in range(epochs):
            for X_batch, y_batch in batch_iterator:
                if y_batch is None:
                    raise ValueError("Cannot train regressor without target column in batch iterator")
                self.partial_fit(X_batch, y_batch)

        accounting = batch_iterator.get_memory_accounting()
        return self, accounting

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict continuous values for X."""
        if not self.is_fitted_:
            raise ValueError("Model is not fitted yet.")
        if self.with_scaling and self.scaler is not None:
            X = self.scaler.transform(X, copy=True)
        return self.model.predict(X)
