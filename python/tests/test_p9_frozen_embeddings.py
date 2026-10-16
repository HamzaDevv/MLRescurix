import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.frozen_embeddings import (
    CacheQuotaExceededError,
    DeterministicEmbeddingCache,
    DimensionFingerprint,
    EmbeddingCacheProtocol,
    FrozenEmbeddingAdapter,
    FrozenEmbeddingPredictor,
    FrozenTextEncoder,
    FrozenVisionEncoder,
    ImageDimensionError,
    InvalidEmbeddingConfigError,
    ModelArchitectureConfig,
    NonFiniteEmbeddingError,
    PreprocessingLimits,
    PrivacyLicensingProfile,
    SequenceLengthExceededError,
    UncertifiedRemoteDownloadError,
    extract_text_embeddings,
    extract_vision_embeddings,
    validate_privacy_and_remote_policy,
)

ROOT = Path(__file__).resolve().parents[2]


class TestP9FrozenEmbeddings(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p9_emb_"))
        self.schema = json.loads((ROOT / "contracts/v2/frozen_embeddings_adapter.schema.json").read_text())

    def test_p9_4_schema_validation(self):
        """P9.4: Validate FrozenEmbeddingAdapter under strict Draft 2020-12 schema."""
        jsonschema.Draft202012Validator.check_schema(self.schema)

        # 1. Text adapter schema validation
        text_adapter = FrozenEmbeddingAdapter.synthetic_text("ds_reviews", embedding_dim=64)
        jsonschema.validate(instance=text_adapter.to_dict(), schema=self.schema)

        # 2. Vision adapter schema validation
        vision_adapter = FrozenEmbeddingAdapter.synthetic_vision("ds_medical_scans", embedding_dim=128)
        jsonschema.validate(instance=vision_adapter.to_dict(), schema=self.schema)

    def test_p9_4_text_embeddings_pooling_and_truncation(self):
        """P9.4: Text embeddings with pooling strategies and sequence truncation."""
        texts = [
            "Clinical notes: patient presents with symptoms of acute hypertension.",
            "Normal follow-up electrocardiogram showing sinus rhythm.",
            "Emergency referral due to elevated troponin levels.",
        ]

        # 1. Mean pooling
        encoder_mean = FrozenTextEncoder(embedding_dim=64, pooling_strategy="mean", model_name="test-enc")
        emb_mean = encoder_mean.encode_batch(texts)
        self.assertEqual(emb_mean.shape, (3, 64))
        self.assertEqual(emb_mean.dtype, np.float32)

        # 2. CLS pooling
        encoder_cls = FrozenTextEncoder(embedding_dim=64, pooling_strategy="cls", model_name="test-enc")
        emb_cls = encoder_cls.encode_batch(texts)
        self.assertEqual(emb_cls.shape, (3, 64))
        # Ensure CLS and mean pooling produce distinct representations
        self.assertFalse(np.allclose(emb_mean[0], emb_cls[0]))

        # 3. Max pooling
        encoder_max = FrozenTextEncoder(embedding_dim=64, pooling_strategy="max", model_name="test-enc")
        emb_max = encoder_max.encode_batch(texts)
        self.assertEqual(emb_max.shape, (3, 64))

        # 4. Truncation strategy: truncate_tail
        long_text = "word " * 200
        encoder_trunc = FrozenTextEncoder(
            embedding_dim=32,
            max_sequence_length=10,
            truncation_strategy="truncate_tail",
        )
        emb_trunc = encoder_trunc.encode_single(long_text)
        self.assertEqual(emb_trunc.shape, (32,))

        # 5. Bag of embeddings family
        encoder_boe = FrozenTextEncoder(
            embedding_dim=32,
            family="bag_of_embeddings",
            pooling_strategy="mean",
        )
        emb_boe = encoder_boe.encode_batch(texts)
        self.assertEqual(emb_boe.shape, (3, 32))

    def test_p9_4_vision_embeddings_normalization_and_patching(self):
        """P9.4: Vision embeddings with patch projection and image normalization presets."""
        rng = np.random.RandomState(42)
        # Create synthetic images: shape (4, 3, 32, 32)
        images = rng.randint(0, 256, size=(4, 3, 32, 32)).astype(np.float32)

        # 1. ImageNet normalization with mean pooling
        encoder_vis = FrozenVisionEncoder(
            embedding_dim=64,
            patch_size=8,
            image_channels=3,
            normalization_preset="imagenet",
            pooling_strategy="mean",
        )
        emb_vis = encoder_vis.encode_batch(images)
        self.assertEqual(emb_vis.shape, (4, 64))
        self.assertEqual(emb_vis.dtype, np.float32)

        # 2. Unit interval normalization with CLS pooling
        encoder_cls = FrozenVisionEncoder(
            embedding_dim=64,
            patch_size=8,
            image_channels=3,
            normalization_preset="unit_interval",
            pooling_strategy="cls",
        )
        emb_cls = encoder_cls.encode_batch(images)
        self.assertEqual(emb_cls.shape, (4, 64))
        self.assertFalse(np.allclose(emb_vis[0], emb_cls[0]))

        # 3. Standard score normalization
        encoder_std = FrozenVisionEncoder(
            embedding_dim=48,
            patch_size=16,
            image_channels=3,
            normalization_preset="standard_score",
            pooling_strategy="max",
        )
        emb_std = encoder_std.encode_batch(images)
        self.assertEqual(emb_std.shape, (4, 48))

    def test_p9_4_content_addressed_embedding_cache(self):
        """P9.4: Deterministic content-addressed cache, hit/miss counting, and LRU eviction."""
        cache = DeterministicEmbeddingCache(
            storage_budget_bytes=1048576,  # 1 MiB
            max_entries=3,
            eviction_policy="lru",
        )

        texts = ["Patient record alpha", "Patient record beta", "Patient record gamma"]
        encoder = FrozenTextEncoder(embedding_dim=32, model_name="cache-test")

        # First pass: all misses
        embs1 = encoder.encode_batch(texts, cache=cache)
        stats1 = cache.get_stats()
        self.assertEqual(stats1["miss_count"], 3)
        self.assertEqual(stats1["hit_count"], 0)
        self.assertEqual(stats1["entry_count"], 3)

        # Second pass: all hits!
        embs2 = encoder.encode_batch(texts, cache=cache)
        stats2 = cache.get_stats()
        self.assertEqual(stats2["miss_count"], 3)
        self.assertEqual(stats2["hit_count"], 3)
        self.assertTrue(np.allclose(embs1, embs2))

        # Add 4th item -> causes eviction of least recently used entry
        encoder.encode_batch(["Patient record delta"], cache=cache)
        stats3 = cache.get_stats()
        self.assertEqual(stats3["entry_count"], 3)
        self.assertEqual(stats3["miss_count"], 4)

    def test_p9_4_delivery_inference_contract(self):
        """P9.4: Delivery inference contract producing calibrated tabular features."""
        adapter = FrozenEmbeddingAdapter.synthetic_text("ds_notes", embedding_dim=16)
        predictor = FrozenEmbeddingPredictor(adapter)

        notes = [
            "Cardiology consultation recommended.",
            "Discharged in stable condition.",
        ]

        tabular_features = predictor.transform_text_to_tabular(notes)
        self.assertIsInstance(tabular_features, pd.DataFrame)
        self.assertEqual(tabular_features.shape, (2, 16))
        self.assertEqual(list(tabular_features.columns), [f"text_emb_{i}" for i in range(16)])

        # Can concatenate with base tabular features
        tabular_base = pd.DataFrame({"patient_id": [101, 102], "age": [45, 62]})
        combined_df = pd.concat([tabular_base, tabular_features], axis=1)
        self.assertEqual(combined_df.shape, (2, 18))
        self.assertIn("text_emb_0", combined_df.columns)

    def test_p9_4_negative_scientific_cases(self):
        """P9.4: Negative scientific tests: sequence/image caps, non-finite features, remote downloads, cache quotas."""
        # 1. Reject uncertified remote model download request
        with self.assertRaises(UncertifiedRemoteDownloadError):
            validate_privacy_and_remote_policy({
                "remote_download_allowed": True,
                "uncertified_remote_download_refused": False,
            })

        adapter_bad = FrozenEmbeddingAdapter.synthetic_text("ds_bad")
        adapter_bad.privacy_licensing.remote_download_allowed = True
        with self.assertRaises(UncertifiedRemoteDownloadError):
            validate_privacy_and_remote_policy(adapter_bad)

        with self.assertRaises(UncertifiedRemoteDownloadError):
            FrozenEmbeddingPredictor(adapter_bad)

        # 2. Sequence length cap exceeded when truncation is None
        encoder_strict = FrozenTextEncoder(
            embedding_dim=16,
            max_sequence_length=5,
            truncation_strategy=None,
        )
        with self.assertRaises(SequenceLengthExceededError):
            encoder_strict.encode_single("one two three four five six seven eight")

        # 3. Image dimension cap violation
        encoder_vis_strict = FrozenVisionEncoder(
            embedding_dim=16,
            max_image_dimension=(64, 64),
        )
        huge_image = np.zeros((3, 128, 128), dtype=np.float32)
        with self.assertRaises(ImageDimensionError):
            encoder_vis_strict.encode_single(huge_image)

        # 4. Invalid image channels
        bad_channel_image = np.zeros((7, 32, 32), dtype=np.float32)
        with self.assertRaises(ImageDimensionError):
            encoder_vis_strict.encode_single(bad_channel_image)

        # 5. Non-finite feature representation (NaN or Inf)
        nan_image = np.zeros((3, 32, 32), dtype=np.float32)
        nan_image[0, 5, 5] = np.nan
        with self.assertRaises(NonFiniteEmbeddingError):
            encoder_vis_strict.encode_single(nan_image)

        inf_image = np.zeros((3, 32, 32), dtype=np.float32)
        inf_image[0, 5, 5] = np.inf
        with self.assertRaises(NonFiniteEmbeddingError):
            encoder_vis_strict.encode_single(inf_image)

        # 6. Cache quota violation with refuse policy
        tiny_cache = DeterministicEmbeddingCache(
            storage_budget_bytes=1048576,  # 1 MiB minimum
            max_entries=2,
            eviction_policy="refuse",
        )
        encoder_tiny = FrozenTextEncoder(embedding_dim=16)
        encoder_tiny.encode_batch(["item1", "item2"], cache=tiny_cache)
        # 3rd item exceeds max_entries with refuse policy
        with self.assertRaises(CacheQuotaExceededError):
            encoder_tiny.encode_batch(["item3"], cache=tiny_cache)

        # 7. Single entry exceeds total cache storage budget
        single_exceed_cache = DeterministicEmbeddingCache(storage_budget_bytes=1048576)
        large_emb = np.zeros((300, 1024), dtype=np.float32)  # ~1.2 MiB
        with self.assertRaises(CacheQuotaExceededError):
            single_exceed_cache.put("large_key", large_emb)


if __name__ == "__main__":
    unittest.main()
