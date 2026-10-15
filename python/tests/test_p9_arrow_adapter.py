import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from mlrescurix_worker.scientific.arrow_adapter import (
    ArrowBatchIterator,
    ArrowDatasetAdapter,
    IncrementalFeatureScaler,
    MemoryAccounting,
    MemoryCapExceededError,
    OutOfCoreSGDClassifier,
    OutOfCoreSGDRegressor,
    SchemaFingerprintMismatchError,
    UnfoundedZeroCopyError,
    compute_arrow_schema_fingerprint,
    extract_arrow_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]


class TestP9ArrowAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p9_arrow_"))
        self.schema = json.loads((ROOT / "contracts/v2/arrow_dataset_adapter.schema.json").read_text())

    def test_p9_1_schema_and_fingerprint_extraction(self):
        """P9.1: Ingestion fingerprinting on Parquet and Feather datasets, strictly validated under Draft 2020-12 schema."""
        # 1. Create synthetic dataset with numeric features, nullable column, and binary target
        n_rows = 500
        df = pd.DataFrame({
            "f1": np.linspace(0.0, 10.0, n_rows, dtype=np.float64),
            "f2": np.random.randn(n_rows).astype(np.float64),
            "f3_nullable": [None if i % 10 == 0 else float(i) for i in range(n_rows)],
            "target": (np.linspace(0.0, 10.0, n_rows) > 5.0).astype(np.int64),
        })

        table = pa.Table.from_pandas(df)
        parquet_path = self.tmp / "test_data.parquet"
        pq.write_table(table, str(parquet_path))

        feather_path = self.tmp / "test_data.feather"
        feather.write_feather(table, str(feather_path))

        # Inspect parquet
        adapter_pq = extract_arrow_fingerprint(
            source_path=parquet_path,
            dataset_id="ds_pq_test",
            target_column="target",
        )
        self.assertEqual(adapter_pq.format, "parquet")
        self.assertEqual(adapter_pq.row_count, n_rows)
        self.assertEqual(adapter_pq.column_count, 4)
        self.assertEqual(len(adapter_pq.schema_fingerprint), 64)
        jsonschema.validate(instance=adapter_pq.to_dict(), schema=self.schema)

        # Inspect feather
        adapter_ftr = extract_arrow_fingerprint(
            source_path=feather_path,
            dataset_id="ds_ftr_test",
            target_column="target",
        )
        self.assertEqual(adapter_ftr.format, "feather")
        self.assertEqual(adapter_ftr.row_count, n_rows)
        jsonschema.validate(instance=adapter_ftr.to_dict(), schema=self.schema)

    def test_p9_1_chunked_batch_iterator_and_feature_only_projection(self):
        """P9.1: True memory-mapped column access and chunked batch iterators without inflating memory into dense NumPy arrays."""
        n_rows = 1000
        df = pd.DataFrame({
            "f1": np.arange(n_rows, dtype=np.float64),
            "f2": np.arange(n_rows, dtype=np.float64) * 2.0,
            "f3": np.arange(n_rows, dtype=np.float64) * 3.0,
            "target": (np.arange(n_rows) % 2).astype(np.int64),
        })
        table = pa.Table.from_pandas(df)
        pq_path = self.tmp / "large_data.parquet"
        pq.write_table(table, str(pq_path))

        # Batch iterator with batch size = 200 rows
        it = ArrowBatchIterator(
            source_path=pq_path,
            batch_size_rows=200,
            feature_columns=["f1", "f2"],
            target_column="target",
        )

        batches = list(it)
        self.assertEqual(len(batches), 5)
        for X_batch, y_batch in batches:
            self.assertEqual(X_batch.shape, (200, 2))
            self.assertEqual(y_batch.shape, (200,))
            self.assertFalse(np.isnan(X_batch).any())

        accounting = it.get_memory_accounting()
        self.assertEqual(accounting.batches_processed, 5)
        self.assertGreater(accounting.bytes_read, 0)
        self.assertTrue(accounting.unfounded_zero_copy_refused)

    def test_p9_1_incremental_feature_scaler_parity(self):
        """P9.1: IncrementalFeatureScaler running Welford/Chan update matches scikit-learn StandardScaler."""
        np.random.seed(42)
        X = np.random.randn(1000, 5) * np.array([2.0, 10.0, 0.5, 50.0, 1.0]) + np.array([5.0, -20.0, 1.0, 100.0, 0.0])

        # Standard scikit-learn scaler
        sklearn_scaler = StandardScaler()
        X_sklearn = sklearn_scaler.fit_transform(X)

        # IncrementalFeatureScaler fitted in 10 chunks of 100
        inc_scaler = IncrementalFeatureScaler()
        for i in range(10):
            chunk = X[i * 100 : (i + 1) * 100]
            inc_scaler.partial_fit(chunk)

        X_inc = inc_scaler.transform(X)

        np.testing.assert_allclose(inc_scaler.mean_, sklearn_scaler.mean_, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(inc_scaler.var_, sklearn_scaler.var_, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(X_inc, X_sklearn, rtol=1e-5, atol=1e-6)

    def test_p9_1_out_of_core_baseline_models(self):
        """P9.1: Out-of-core SGDClassifier and SGDRegressor training across chunked batch iterators."""
        n_rows = 800
        np.random.seed(42)
        X = np.random.randn(n_rows, 4)
        y_clf = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
        y_reg = (X[:, 0] * 3.0 + X[:, 1] * 2.0 - X[:, 2]).astype(np.float64)

        df = pd.DataFrame({
            "f0": X[:, 0],
            "f1": X[:, 1],
            "f2": X[:, 2],
            "f3": X[:, 3],
            "target_clf": y_clf,
            "target_reg": y_reg,
        })
        pq_path = self.tmp / "model_data.parquet"
        pq.write_table(pa.Table.from_pandas(df), str(pq_path))

        # 1. Classifier
        it_clf = ArrowBatchIterator(
            source_path=pq_path,
            batch_size_rows=200,
            feature_columns=["f0", "f1", "f2", "f3"],
            target_column="target_clf",
        )
        clf = OutOfCoreSGDClassifier(random_state=42)
        clf.fit_batches(it_clf, classes=np.array([0, 1]), epochs=3)

        preds_clf = clf.predict(X[:100])
        self.assertEqual(len(preds_clf), 100)
        proba_clf = clf.predict_proba(X[:100])
        self.assertEqual(proba_clf.shape, (100, 2))
        accuracy = np.mean(preds_clf == y_clf[:100])
        self.assertGreater(accuracy, 0.75)

        # 2. Regressor
        it_reg = ArrowBatchIterator(
            source_path=pq_path,
            batch_size_rows=200,
            feature_columns=["f0", "f1", "f2", "f3"],
            target_column="target_reg",
        )
        reg = OutOfCoreSGDRegressor(random_state=42)
        reg.fit_batches(it_reg, epochs=3)

        preds_reg = reg.predict(X[:100])
        self.assertEqual(len(preds_reg), 100)
        mae = np.mean(np.abs(preds_reg - y_reg[:100]))
        self.assertLess(mae, 1.0)

    def test_p9_1_negative_scientific_cases(self):
        """P9.1: Negative cases: reject memory cap violations, corrupt schema fingerprint, and unfounded zero copy."""
        n_rows = 500
        df = pd.DataFrame({
            "f1": np.arange(n_rows, dtype=np.float64),
            "f2_nullable": [None if i % 5 == 0 else float(i) for i in range(n_rows)],
            "target": np.ones(n_rows, dtype=np.int64),
        })
        pq_path = self.tmp / "negative_test.parquet"
        pq.write_table(pa.Table.from_pandas(df), str(pq_path))

        # 1. Exceeding max_batch_bytes cap
        it_cap = ArrowBatchIterator(
            source_path=pq_path,
            batch_size_rows=500,
            max_batch_bytes=512,  # deliberately tiny 512 bytes limit
            feature_columns=["f1"],
            target_column="target",
        )
        with self.assertRaises(MemoryCapExceededError):
            list(it_cap)

        # 2. Mismatched schema fingerprint
        with self.assertRaises(SchemaFingerprintMismatchError):
            ArrowBatchIterator(
                source_path=pq_path,
                expected_schema_fingerprint="0000000000000000000000000000000000000000000000000000000000000000",
                feature_columns=["f1"],
            )

        # 3. Refusal of unfounded zero-copy claim on nullable column
        it_zero_copy_claim = ArrowBatchIterator(
            source_path=pq_path,
            batch_size_rows=100,
            feature_columns=["f2_nullable"],
            claim_zero_copy=True,
        )
        with self.assertRaises(UnfoundedZeroCopyError):
            list(it_zero_copy_claim)


if __name__ == "__main__":
    unittest.main()
