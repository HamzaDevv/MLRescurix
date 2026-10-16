"""Frozen-Embedding Adapters (Text & Vision) for MLRescurix (Phase P9.4).

Provides:
- Lightweight frozen transformer and bag-of-embeddings encoders for text data.
- Lightweight frozen vision backbones (patch projector, normalized tensor representations).
- Deterministic content-addressed embedding cache (SHA-256 of raw inputs + model config)
  with local cache storage budget management and eviction bounds.
- Delivery inference contracts: extract_text_embeddings and extract_vision_embeddings
  producing calibrated downstream feature matrices compatible with tabular models.
- Negative scientific guards rejecting sequence/image dimension cap violations,
  non-finite feature representations, uncertified remote download requests,
  and cache quota violations.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Custom Scientific Exceptions
# -----------------------------------------------------------------------------


class FrozenEmbeddingError(Exception):
    """Base exception for frozen embedding errors."""
    pass


class SequenceLengthExceededError(FrozenEmbeddingError):
    """Raised when text inputs exceed sequence length limits without permitted truncation."""
    pass


class ImageDimensionError(FrozenEmbeddingError):
    """Raised when image inputs violate dimension, channel, or size caps."""
    pass


class NonFiniteEmbeddingError(FrozenEmbeddingError):
    """Raised when inputs or extracted embeddings contain NaN or +/-Inf values."""
    pass


class UncertifiedRemoteDownloadError(FrozenEmbeddingError):
    """Raised when uncertified remote model download requests are made."""
    pass


class CacheQuotaExceededError(FrozenEmbeddingError):
    """Raised when embedding cache storage exceeds the configured budget."""
    pass


class InvalidEmbeddingConfigError(FrozenEmbeddingError):
    """Raised when adapter or encoder configuration is invalid."""
    pass


# -----------------------------------------------------------------------------
# Data Models and Configurations
# -----------------------------------------------------------------------------


@dataclass
class ModelArchitectureConfig:
    family: Literal["transformer", "bag_of_embeddings", "patch_projector", "convnet", "multimodal_projection"] = "transformer"
    encoder_type: str = "lightweight_frozen_transformer"
    weights_status: Literal["frozen_embedded", "frozen_local_checkpoint", "deterministic_synthetic"] = "frozen_embedded"
    backbone_name: str = "frozen-transformer-base"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DimensionFingerprint:
    input_dim: List[int] = field(default_factory=lambda: [128])
    embedding_dim: int = 64
    pooling_strategy: Literal["mean", "cls", "max", "flatten_project"] = "mean"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingCacheProtocol:
    cache_enabled: bool = True
    content_addressed: bool = True
    hasher: Literal["sha256", "sha512"] = "sha256"
    storage_budget_bytes: int = 33554432  # 32 MiB
    max_entries: Optional[int] = 10000
    eviction_policy: Literal["lru", "fifo", "refuse"] = "lru"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessingLimits:
    max_batch_size: int = 1000
    max_sequence_length: Optional[int] = 128
    truncation_strategy: Optional[Literal["truncate_tail", "truncate_head"]] = "truncate_tail"
    max_image_dimension: Optional[List[int]] = field(default_factory=lambda: [1024, 1024])
    image_channels: Optional[int] = 3
    normalization_preset: Optional[Literal["imagenet", "unit_interval", "standard_score"]] = "imagenet"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyLicensingProfile:
    license_type: Literal["MIT", "Apache-2.0", "BSD-3-Clause", "permissive", "proprietary_restricted"] = "MIT"
    remote_download_allowed: bool = False
    uncertified_remote_download_refused: bool = True
    offline_inference_certified: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeliveryInferenceContract:
    downstream_feature_dim: int = 64
    output_representation: Literal["dense_float32", "dense_float64"] = "dense_float32"
    calibrated_tabular_prefix: str = "emb"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FrozenEmbeddingAdapter:
    schema_version: str = "v2"
    adapter_id: str = "frozen_emb_default"
    dataset_id: str = "ds_default"
    modality: Literal["text", "vision", "multimodal"] = "text"
    model_architecture: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    dimension_fingerprint: DimensionFingerprint = field(default_factory=DimensionFingerprint)
    cache_protocol: EmbeddingCacheProtocol = field(default_factory=EmbeddingCacheProtocol)
    preprocessing_limits: PreprocessingLimits = field(default_factory=PreprocessingLimits)
    privacy_licensing: PrivacyLicensingProfile = field(default_factory=PrivacyLicensingProfile)
    delivery_inference: DeliveryInferenceContract = field(default_factory=DeliveryInferenceContract)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "dataset_id": self.dataset_id,
            "modality": self.modality,
            "model_architecture": self.model_architecture.to_dict(),
            "dimension_fingerprint": self.dimension_fingerprint.to_dict(),
            "cache_protocol": self.cache_protocol.to_dict(),
            "preprocessing_limits": self.preprocessing_limits.to_dict(),
            "privacy_licensing": self.privacy_licensing.to_dict(),
            "delivery_inference": self.delivery_inference.to_dict(),
        }

    @classmethod
    def synthetic_text(cls, dataset_id: str, embedding_dim: int = 64) -> FrozenEmbeddingAdapter:
        return cls(
            adapter_id=f"text_emb_{dataset_id}",
            dataset_id=dataset_id,
            modality="text",
            model_architecture=ModelArchitectureConfig(
                family="transformer",
                encoder_type="lightweight_frozen_transformer",
                weights_status="frozen_embedded",
                backbone_name="frozen-transformer-text-v1",
            ),
            dimension_fingerprint=DimensionFingerprint(
                input_dim=[128],
                embedding_dim=embedding_dim,
                pooling_strategy="mean",
            ),
            delivery_inference=DeliveryInferenceContract(
                downstream_feature_dim=embedding_dim,
                output_representation="dense_float32",
                calibrated_tabular_prefix="text_emb",
            ),
        )

    @classmethod
    def synthetic_vision(cls, dataset_id: str, embedding_dim: int = 64) -> FrozenEmbeddingAdapter:
        return cls(
            adapter_id=f"vision_emb_{dataset_id}",
            dataset_id=dataset_id,
            modality="vision",
            model_architecture=ModelArchitectureConfig(
                family="patch_projector",
                encoder_type="lightweight_patch_projector",
                weights_status="frozen_embedded",
                backbone_name="frozen-patch-projector-v1",
            ),
            dimension_fingerprint=DimensionFingerprint(
                input_dim=[3, 64, 64],
                embedding_dim=embedding_dim,
                pooling_strategy="mean",
            ),
            delivery_inference=DeliveryInferenceContract(
                downstream_feature_dim=embedding_dim,
                output_representation="dense_float32",
                calibrated_tabular_prefix="vis_emb",
            ),
        )


# -----------------------------------------------------------------------------
# Policy and Bounds Validators
# -----------------------------------------------------------------------------


def validate_privacy_and_remote_policy(config: Union[FrozenEmbeddingAdapter, PrivacyLicensingProfile, Dict[str, Any]]) -> None:
    """Enforces certified offline execution by rejecting uncertified remote download requests."""
    if isinstance(config, FrozenEmbeddingAdapter):
        profile = config.privacy_licensing
        remote_allowed = profile.remote_download_allowed
        refused = profile.uncertified_remote_download_refused
    elif isinstance(config, PrivacyLicensingProfile):
        remote_allowed = config.remote_download_allowed
        refused = config.uncertified_remote_download_refused
    elif isinstance(config, dict):
        remote_allowed = config.get("remote_download_allowed", False)
        refused = config.get("uncertified_remote_download_refused", True)
    else:
        raise TypeError(f"Unsupported config type: {type(config)}")

    if remote_allowed or not refused:
        raise UncertifiedRemoteDownloadError(
            "Uncertified remote model downloads are strictly rejected in certified offline execution."
        )


def validate_finite_matrix(matrix: np.ndarray, name: str = "embeddings") -> None:
    """Checks that array contains no NaN or infinite values."""
    if not np.all(np.isfinite(matrix)):
        raise NonFiniteEmbeddingError(f"{name} contains non-finite representation (NaN, Inf, or -Inf).")


# -----------------------------------------------------------------------------
# Deterministic Embedding Cache (Content-Addressed)
# -----------------------------------------------------------------------------


class DeterministicEmbeddingCache:
    """Content-addressed cache for frozen embeddings with storage budget bounds."""

    def __init__(
        self,
        storage_budget_bytes: int = 33554432,  # 32 MiB
        max_entries: Optional[int] = 10000,
        eviction_policy: Literal["lru", "fifo", "refuse"] = "lru",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        if storage_budget_bytes < 1024 * 1024:
            raise InvalidEmbeddingConfigError(f"Storage budget must be at least 1 MiB, got {storage_budget_bytes}")
        self.storage_budget_bytes = storage_budget_bytes
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.content_addressed: bool = True
        self.hasher: str = "sha256"

        self._entries: OrderedDict[str, np.ndarray] = OrderedDict()
        self._entry_bytes: Dict[str, int] = {}
        self._current_bytes: int = 0
        self.hit_count: int = 0
        self.miss_count: int = 0

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    def compute_text_key(self, text: str, model_fingerprint: str) -> str:
        h = hashlib.sha256()
        h.update(b"text:")
        h.update(model_fingerprint.encode("utf-8"))
        h.update(b":")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def compute_vision_key(self, image_bytes: bytes, model_fingerprint: str) -> str:
        h = hashlib.sha256()
        h.update(b"vision:")
        h.update(model_fingerprint.encode("utf-8"))
        h.update(b":")
        h.update(image_bytes)
        return h.hexdigest()

    def get(self, key: str) -> Optional[np.ndarray]:
        if key in self._entries:
            self.hit_count += 1
            self._entries.move_to_end(key)
            return self._entries[key].copy()
        self.miss_count += 1
        return None

    def put(self, key: str, embedding: np.ndarray) -> None:
        validate_finite_matrix(embedding, name="cached_embedding")
        arr = np.ascontiguousarray(embedding)
        byte_size = arr.nbytes

        if byte_size > self.storage_budget_bytes:
            raise CacheQuotaExceededError(
                f"Single embedding entry ({byte_size} bytes) exceeds total cache storage budget ({self.storage_budget_bytes} bytes)."
            )

        # If already exists, update
        if key in self._entries:
            old_size = self._entry_bytes[key]
            self._current_bytes = self._current_bytes - old_size + byte_size
            self._entries[key] = arr
            self._entry_bytes[key] = byte_size
            self._entries.move_to_end(key)
            return

        # Check if eviction is necessary
        while (self._current_bytes + byte_size > self.storage_budget_bytes) or (
            self.max_entries is not None and len(self._entries) >= self.max_entries
        ):
            if self.eviction_policy == "refuse":
                raise CacheQuotaExceededError(
                    f"Embedding cache budget exceeded ({self._current_bytes + byte_size} > {self.storage_budget_bytes} bytes) with refusal policy."
                )
            if not self._entries:
                break
            # Pop oldest
            evicted_key, _ = self._entries.popitem(last=False)
            evicted_bytes = self._entry_bytes.pop(evicted_key)
            self._current_bytes -= evicted_bytes

        if self._current_bytes + byte_size > self.storage_budget_bytes:
            raise CacheQuotaExceededError(
                f"Unable to allocate {byte_size} bytes within budget {self.storage_budget_bytes} bytes."
            )

        self._entries[key] = arr
        self._entry_bytes[key] = byte_size
        self._current_bytes += byte_size

    def get_stats(self) -> Dict[str, Any]:
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "entry_count": len(self._entries),
            "current_bytes": self._current_bytes,
            "storage_budget_bytes": self.storage_budget_bytes,
            "eviction_policy": self.eviction_policy,
        }


# -----------------------------------------------------------------------------
# Lightweight Frozen Text Encoder (Transformer & Bag-of-Embeddings)
# -----------------------------------------------------------------------------


class FrozenTextEncoder:
    """Lightweight frozen text encoder supporting bag-of-embeddings and transformer modes."""

    def __init__(
        self,
        embedding_dim: int = 64,
        max_sequence_length: Optional[int] = 128,
        truncation_strategy: Optional[Literal["truncate_tail", "truncate_head"]] = "truncate_tail",
        pooling_strategy: Literal["mean", "cls", "max"] = "mean",
        family: Literal["transformer", "bag_of_embeddings"] = "transformer",
        model_name: str = "frozen-transformer-text-v1",
        vocab_size: int = 10007,
        random_seed: int = 42,
    ):
        if embedding_dim <= 0 or embedding_dim > 4096:
            raise InvalidEmbeddingConfigError(f"embedding_dim must be in [1, 4096], got {embedding_dim}")
        if max_sequence_length is not None and (max_sequence_length <= 0 or max_sequence_length > 4096):
            raise InvalidEmbeddingConfigError(f"max_sequence_length must be in [1, 4096], got {max_sequence_length}")

        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.truncation_strategy = truncation_strategy
        self.pooling_strategy = pooling_strategy
        self.family = family
        self.model_name = model_name
        self.vocab_size = vocab_size
        self.random_seed = random_seed

        # Initialize deterministic frozen weights using seed derived from model name & seed
        seed_hash = int(hashlib.sha256(f"{model_name}:{random_seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_hash)

        # Token embedding matrix: (vocab_size, embedding_dim)
        scale = 1.0 / math.sqrt(embedding_dim)
        self.token_embeddings = (rng.randn(vocab_size, embedding_dim) * scale).astype(np.float32)

        # CLS embedding vector (1, embedding_dim)
        self.cls_embedding = (rng.randn(1, embedding_dim) * scale).astype(np.float32)

        # Frozen projection weights for lightweight self-attention if transformer
        if family == "transformer":
            self.w_q = (rng.randn(embedding_dim, embedding_dim) * scale).astype(np.float32)
            self.w_k = (rng.randn(embedding_dim, embedding_dim) * scale).astype(np.float32)
            self.w_v = (rng.randn(embedding_dim, embedding_dim) * scale).astype(np.float32)
            self.w_o = (rng.randn(embedding_dim, embedding_dim) * scale).astype(np.float32)

    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace and subword tokenization mapping tokens to vocab indices."""
        raw_tokens = text.strip().lower().split()
        if not raw_tokens:
            return [0]

        token_ids: List[int] = []
        for word in raw_tokens:
            # Hash word into vocab index
            token_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:6], 16)
            token_id = token_hash % (self.vocab_size - 1) + 1  # 0 reserved for padding/empty
            token_ids.append(token_id)
        return token_ids

    def encode_single(self, text: str) -> np.ndarray:
        """Encodes a single text string into a 1D embedding vector of shape (embedding_dim,)."""
        if not isinstance(text, str):
            raise InvalidEmbeddingConfigError(f"Expected text input to be str, got {type(text)}")

        tokens = self._tokenize(text)

        # Sequence length validation & truncation
        if self.max_sequence_length is not None:
            if len(tokens) > self.max_sequence_length:
                if self.truncation_strategy == "truncate_tail":
                    tokens = tokens[: self.max_sequence_length]
                elif self.truncation_strategy == "truncate_head":
                    tokens = tokens[-self.max_sequence_length :]
                else:
                    raise SequenceLengthExceededError(
                        f"Text sequence length ({len(tokens)}) exceeds max_sequence_length ({self.max_sequence_length}) without truncation."
                    )

        # Look up token vectors: (seq_len, embedding_dim)
        token_vecs = self.token_embeddings[tokens]  # (T, D)

        # Prepend [CLS] token vector
        seq_vecs = np.vstack([self.cls_embedding, token_vecs])  # (T + 1, D)

        # If transformer, apply frozen multi-head attention step
        if self.family == "transformer":
            Q = np.dot(seq_vecs, self.w_q)
            K = np.dot(seq_vecs, self.w_k)
            V = np.dot(seq_vecs, self.w_v)

            # Scaled dot-product attention
            d_k = float(self.embedding_dim)
            scores = np.dot(Q, K.T) / math.sqrt(d_k)
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            attn_out = np.dot(attn_weights, V)
            # Residual connection & layer norm
            seq_vecs = seq_vecs + np.dot(attn_out, self.w_o)
            mean = np.mean(seq_vecs, axis=-1, keepdims=True)
            std = np.std(seq_vecs, axis=-1, keepdims=True) + 1e-6
            seq_vecs = (seq_vecs - mean) / std

        # Pooling
        if self.pooling_strategy == "cls":
            pooled = seq_vecs[0]  # CLS token at index 0
        elif self.pooling_strategy == "max":
            pooled = np.max(seq_vecs, axis=0)
        else:  # mean pooling
            pooled = np.mean(seq_vecs, axis=0)

        validate_finite_matrix(pooled, name="text_embedding")
        return pooled.astype(np.float32)

    def encode_batch(
        self,
        texts: Sequence[str],
        cache: Optional[DeterministicEmbeddingCache] = None,
    ) -> np.ndarray:
        """Encodes a batch of texts into a 2D embedding matrix (N, embedding_dim)."""
        if len(texts) == 0:
            raise InvalidEmbeddingConfigError("Input batch of texts cannot be empty.")

        embeddings: List[np.ndarray] = []
        for text in texts:
            if cache is not None and cache.content_addressed:
                cache_key = cache.compute_text_key(text, f"{self.model_name}:{self.embedding_dim}:{self.pooling_strategy}")
                cached = cache.get(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    continue

                emb = self.encode_single(text)
                cache.put(cache_key, emb)
                embeddings.append(emb)
            else:
                emb = self.encode_single(text)
                embeddings.append(emb)

        matrix = np.vstack(embeddings).astype(np.float32)
        validate_finite_matrix(matrix, name="text_embeddings_batch")
        return matrix


# -----------------------------------------------------------------------------
# Lightweight Frozen Vision Encoder (Patch Projector / Feature Extractor)
# -----------------------------------------------------------------------------


class FrozenVisionEncoder:
    """Lightweight frozen vision backbone with patch projection and standard image normalization."""

    def __init__(
        self,
        embedding_dim: int = 64,
        patch_size: int = 16,
        image_channels: int = 3,
        max_image_dimension: Tuple[int, int] = (1024, 1024),
        normalization_preset: Literal["imagenet", "unit_interval", "standard_score"] = "imagenet",
        pooling_strategy: Literal["mean", "cls", "max", "flatten_project"] = "mean",
        model_name: str = "frozen-patch-projector-v1",
        random_seed: int = 42,
    ):
        if embedding_dim <= 0 or embedding_dim > 4096:
            raise InvalidEmbeddingConfigError(f"embedding_dim must be in [1, 4096], got {embedding_dim}")
        if patch_size <= 0:
            raise InvalidEmbeddingConfigError(f"patch_size must be positive, got {patch_size}")
        if image_channels not in (1, 3, 4):
            raise ImageDimensionError(f"image_channels must be 1, 3, or 4, got {image_channels}")

        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_channels = image_channels
        self.max_image_dimension = max_image_dimension
        self.normalization_preset = normalization_preset
        self.pooling_strategy = pooling_strategy
        self.model_name = model_name

        # Patch projection weights: (channels * patch_size * patch_size, embedding_dim)
        patch_dim = image_channels * patch_size * patch_size
        seed_hash = int(hashlib.sha256(f"{model_name}:{random_seed}".encode()).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed_hash)

        scale = 1.0 / math.sqrt(patch_dim)
        self.patch_projection = (rng.randn(patch_dim, embedding_dim) * scale).astype(np.float32)
        self.cls_token = (rng.randn(1, embedding_dim) * scale).astype(np.float32)

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Applies configured image normalization."""
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

        if self.normalization_preset == "imagenet":
            # ImageNet mean & std for 3 channels:
            # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            if arr.shape[0] == 3:
                arr = (arr - mean) / std
            else:
                arr = (arr - 0.5) / 0.5
        elif self.normalization_preset == "standard_score":
            mean = np.mean(arr)
            std = np.std(arr) + 1e-6
            arr = (arr - mean) / std
        elif self.normalization_preset == "unit_interval":
            arr = np.clip(arr, 0.0, 1.0)
        return arr

    def encode_single(self, image: np.ndarray) -> np.ndarray:
        """Encodes a single image array (C, H, W) or (H, W, C) into (embedding_dim,)."""
        validate_finite_matrix(image, name="image_pixels")

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]  # (1, H, W)
        elif arr.ndim == 3:
            # If shape is (H, W, C), transpose to (C, H, W)
            if arr.shape[2] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                arr = np.transpose(arr, (2, 0, 1))
        else:
            raise ImageDimensionError(f"Expected 2D or 3D image tensor, got shape {arr.shape}")

        c, h, w = arr.shape
        if c != self.image_channels:
            raise ImageDimensionError(
                f"Image channels mismatch: expected {self.image_channels}, got {c}"
            )

        max_h, max_w = self.max_image_dimension
        if h > max_h or w > max_w:
            raise ImageDimensionError(
                f"Image dimension ({h}, {w}) exceeds maximum allowable cap ({max_h}, {max_w})"
            )

        # Normalize
        norm_img = self._normalize(arr)

        # Divide into non-overlapping patches
        p = self.patch_size
        # Pad image if h or w not divisible by p
        pad_h = (p - (h % p)) % p
        pad_w = (p - (w % p)) % p
        if pad_h > 0 or pad_w > 0:
            norm_img = np.pad(norm_img, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            c, h, w = norm_img.shape

        num_patches_h = h // p
        num_patches_w = w // p

        patches: List[np.ndarray] = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = norm_img[:, i * p : (i + 1) * p, j * p : (j + 1) * p]
                patches.append(patch.reshape(-1))

        if not patches:
            raise ImageDimensionError("Image has 0 patches; check patch size and image resolution.")

        patch_matrix = np.vstack(patches)  # (num_patches, patch_dim)
        # Project patches to embedding dim
        patch_embeddings = np.dot(patch_matrix, self.patch_projection)  # (num_patches, embedding_dim)

        # Prepend CLS token
        all_embeddings = np.vstack([self.cls_token, patch_embeddings])

        if self.pooling_strategy == "cls":
            pooled = all_embeddings[0]
        elif self.pooling_strategy == "max":
            pooled = np.max(all_embeddings, axis=0)
        else:  # mean pooling
            pooled = np.mean(all_embeddings, axis=0)

        validate_finite_matrix(pooled, name="vision_embedding")
        return pooled.astype(np.float32)

    def encode_batch(
        self,
        images: Union[np.ndarray, Sequence[np.ndarray]],
        cache: Optional[DeterministicEmbeddingCache] = None,
    ) -> np.ndarray:
        """Encodes a batch of images into (N, embedding_dim)."""
        if len(images) == 0:
            raise InvalidEmbeddingConfigError("Input images sequence cannot be empty.")

        embeddings: List[np.ndarray] = []
        for img in images:
            if cache is not None and cache.content_addressed:
                img_bytes = np.ascontiguousarray(img).tobytes()
                cache_key = cache.compute_vision_key(
                    img_bytes, f"{self.model_name}:{self.embedding_dim}:{self.pooling_strategy}"
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    embeddings.append(cached)
                    continue

                emb = self.encode_single(img)
                cache.put(cache_key, emb)
                embeddings.append(emb)
            else:
                emb = self.encode_single(img)
                embeddings.append(emb)

        matrix = np.vstack(embeddings).astype(np.float32)
        validate_finite_matrix(matrix, name="vision_embeddings_batch")
        return matrix


# -----------------------------------------------------------------------------
# Delivery Inference Contract Functions & Predictor
# -----------------------------------------------------------------------------


def extract_text_embeddings(
    texts: Sequence[str],
    adapter_or_config: Optional[Union[FrozenEmbeddingAdapter, Dict[str, Any]]] = None,
    cache: Optional[DeterministicEmbeddingCache] = None,
    embedding_dim: int = 64,
    pooling_strategy: Literal["mean", "cls", "max"] = "mean",
    max_sequence_length: Optional[int] = 128,
    truncation_strategy: Optional[Literal["truncate_tail", "truncate_head"]] = "truncate_tail",
    model_name: str = "frozen-transformer-text-v1",
) -> np.ndarray:
    """High-level delivery interface extracting calibrated text embeddings."""
    if adapter_or_config is not None:
        validate_privacy_and_remote_policy(adapter_or_config)
        if isinstance(adapter_or_config, FrozenEmbeddingAdapter):
            embedding_dim = adapter_or_config.dimension_fingerprint.embedding_dim
            pooling_strategy = adapter_or_config.dimension_fingerprint.pooling_strategy  # type: ignore
            max_sequence_length = adapter_or_config.preprocessing_limits.max_sequence_length
            truncation_strategy = adapter_or_config.preprocessing_limits.truncation_strategy
            model_name = adapter_or_config.model_architecture.backbone_name
        elif isinstance(adapter_or_config, dict):
            embedding_dim = adapter_or_config.get("dimension_fingerprint", {}).get("embedding_dim", embedding_dim)
            pooling_strategy = adapter_or_config.get("dimension_fingerprint", {}).get("pooling_strategy", pooling_strategy)
            max_sequence_length = adapter_or_config.get("preprocessing_limits", {}).get("max_sequence_length", max_sequence_length)
            truncation_strategy = adapter_or_config.get("preprocessing_limits", {}).get("truncation_strategy", truncation_strategy)
            model_name = adapter_or_config.get("model_architecture", {}).get("backbone_name", model_name)

    encoder = FrozenTextEncoder(
        embedding_dim=embedding_dim,
        max_sequence_length=max_sequence_length,
        truncation_strategy=truncation_strategy,
        pooling_strategy=pooling_strategy,
        model_name=model_name,
    )
    return encoder.encode_batch(texts, cache=cache)


def extract_vision_embeddings(
    images: Union[np.ndarray, Sequence[np.ndarray]],
    adapter_or_config: Optional[Union[FrozenEmbeddingAdapter, Dict[str, Any]]] = None,
    cache: Optional[DeterministicEmbeddingCache] = None,
    embedding_dim: int = 64,
    patch_size: int = 16,
    image_channels: int = 3,
    max_image_dimension: Tuple[int, int] = (1024, 1024),
    normalization_preset: Literal["imagenet", "unit_interval", "standard_score"] = "imagenet",
    pooling_strategy: Literal["mean", "cls", "max", "flatten_project"] = "mean",
    model_name: str = "frozen-patch-projector-v1",
) -> np.ndarray:
    """High-level delivery interface extracting calibrated vision embeddings."""
    if adapter_or_config is not None:
        validate_privacy_and_remote_policy(adapter_or_config)
        if isinstance(adapter_or_config, FrozenEmbeddingAdapter):
            embedding_dim = adapter_or_config.dimension_fingerprint.embedding_dim
            pooling_strategy = adapter_or_config.dimension_fingerprint.pooling_strategy
            max_dim = adapter_or_config.preprocessing_limits.max_image_dimension
            if max_dim:
                max_image_dimension = (max_dim[0], max_dim[1])
            image_channels = adapter_or_config.preprocessing_limits.image_channels or 3
            normalization_preset = adapter_or_config.preprocessing_limits.normalization_preset or "imagenet"
            model_name = adapter_or_config.model_architecture.backbone_name
        elif isinstance(adapter_or_config, dict):
            embedding_dim = adapter_or_config.get("dimension_fingerprint", {}).get("embedding_dim", embedding_dim)
            pooling_strategy = adapter_or_config.get("dimension_fingerprint", {}).get("pooling_strategy", pooling_strategy)
            max_dim = adapter_or_config.get("preprocessing_limits", {}).get("max_image_dimension")
            if max_dim:
                max_image_dimension = (max_dim[0], max_dim[1])
            image_channels = adapter_or_config.get("preprocessing_limits", {}).get("image_channels", 3)
            normalization_preset = adapter_or_config.get("preprocessing_limits", {}).get("normalization_preset", "imagenet")
            model_name = adapter_or_config.get("model_architecture", {}).get("backbone_name", model_name)

    encoder = FrozenVisionEncoder(
        embedding_dim=embedding_dim,
        patch_size=patch_size,
        image_channels=image_channels,
        max_image_dimension=max_image_dimension,
        normalization_preset=normalization_preset,
        pooling_strategy=pooling_strategy,
        model_name=model_name,
    )
    return encoder.encode_batch(images, cache=cache)


class FrozenEmbeddingPredictor:
    """Delivery inference engine transforming raw unstructured inputs into calibrated tabular feature matrices."""

    def __init__(
        self,
        adapter: FrozenEmbeddingAdapter,
        cache: Optional[DeterministicEmbeddingCache] = None,
    ):
        validate_privacy_and_remote_policy(adapter)
        self.adapter = adapter
        self.cache = cache

        if adapter.modality == "text":
            self.text_encoder: Optional[FrozenTextEncoder] = FrozenTextEncoder(
                embedding_dim=adapter.dimension_fingerprint.embedding_dim,
                max_sequence_length=adapter.preprocessing_limits.max_sequence_length,
                truncation_strategy=adapter.preprocessing_limits.truncation_strategy,
                pooling_strategy=adapter.dimension_fingerprint.pooling_strategy,  # type: ignore
                model_name=adapter.model_architecture.backbone_name,
            )
            self.vision_encoder = None
        elif adapter.modality == "vision":
            max_dim = adapter.preprocessing_limits.max_image_dimension
            max_tup = (max_dim[0], max_dim[1]) if max_dim else (1024, 1024)
            self.vision_encoder: Optional[FrozenVisionEncoder] = FrozenVisionEncoder(
                embedding_dim=adapter.dimension_fingerprint.embedding_dim,
                image_channels=adapter.preprocessing_limits.image_channels or 3,
                max_image_dimension=max_tup,
                normalization_preset=adapter.preprocessing_limits.normalization_preset or "imagenet",
                pooling_strategy=adapter.dimension_fingerprint.pooling_strategy,
                model_name=adapter.model_architecture.backbone_name,
            )
            self.text_encoder = None
        else:
            self.text_encoder = None
            self.vision_encoder = None

    def transform_text_to_tabular(self, texts: Sequence[str]) -> pd.DataFrame:
        """Transforms text sequence into a calibrated DataFrame with named feature columns."""
        if self.text_encoder is None:
            raise InvalidEmbeddingConfigError("Predictor was not initialized with text modality.")
        matrix = self.text_encoder.encode_batch(texts, cache=self.cache)
        prefix = self.adapter.delivery_inference.calibrated_tabular_prefix
        col_names = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix, columns=col_names)

    def transform_vision_to_tabular(self, images: Union[np.ndarray, Sequence[np.ndarray]]) -> pd.DataFrame:
        """Transforms image tensors into a calibrated DataFrame with named feature columns."""
        if self.vision_encoder is None:
            raise InvalidEmbeddingConfigError("Predictor was not initialized with vision modality.")
        matrix = self.vision_encoder.encode_batch(images, cache=self.cache)
        prefix = self.adapter.delivery_inference.calibrated_tabular_prefix
        col_names = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
        return pd.DataFrame(matrix, columns=col_names)
