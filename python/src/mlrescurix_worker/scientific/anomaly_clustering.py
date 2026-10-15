"""Unsupervised Anomaly Detection & Clustering Adapter (Phase P9.3).

Enforces:
1. Strict Draft 2020-12 typed contracts: unsupervised contamination assumptions,
   score calibrations, clustering partition protocols, and evaluation profiles.
2. Outlier detection: IsolationForest and LocalOutlierFactor with empirical threshold
   calibration and bounded score distribution receipts.
3. Unsupervised clustering: KMeans (with silhouette and inertia heuristics) and
   density-based clustering (HDBSCAN/DBSCAN).
4. Metric suites: Silhouette score, Davies-Bouldin index, Calinski-Harabasz score,
   and contamination-aware ROC-AUC/PR-AUC when synthetic or verified labels are available.
5. Delivery inference contract: predict_anomaly_score and assign_cluster with out-of-sample
   projection guarantees.
6. Deterministic scientific negative guards: reject contamination priors (<=0 or >=0.5),
   non-finite input features, and clustering with invalid k or metric configs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import LocalOutlierFactor


class InvalidContaminationError(ValueError):
    """Raised when contamination prior is outside the valid open interval (0.0, 0.5)."""
    pass


class NonFiniteFeatureError(ValueError):
    """Raised when input features contain NaN, Inf, or -Inf values."""
    pass


class InvalidClusteringConfigError(ValueError):
    """Raised when clustering parameters (e.g. k < 2, invalid metric, eps <= 0) are invalid."""
    pass


class OutOfSampleProjectionError(RuntimeError):
    """Raised when out-of-sample projection fails or receives incompatible dimensions."""
    pass


def validate_input_features(X: Union[np.ndarray, pd.DataFrame, List[List[float]]]) -> np.ndarray:
    """Validates input features for 2D shape, finite values, and conversion to float64."""
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=np.float64)
    elif isinstance(X, (list, tuple)):
        arr = np.asarray(X, dtype=np.float64)
    elif isinstance(X, np.ndarray):
        if X.dtype != np.float64:
            arr = X.astype(np.float64)
        else:
            arr = X
    else:
        raise NonFiniteFeatureError(f"Unsupported feature type: {type(X)}")

    if arr.ndim != 2:
        raise NonFiniteFeatureError(f"Input features must be a 2D matrix, got shape {arr.shape}")

    if arr.shape[0] == 0:
        raise NonFiniteFeatureError("Input feature matrix cannot be empty (0 rows)")

    if arr.shape[1] == 0:
        raise NonFiniteFeatureError("Input feature matrix must contain at least one feature (0 columns)")

    if not np.isfinite(arr).all():
        raise NonFiniteFeatureError("Input features contain non-finite values (NaN, Inf, or -Inf)")

    return np.ascontiguousarray(arr)


def validate_contamination_prior(contamination: float) -> float:
    """Strictly validates that contamination prior is in the open interval (0.0, 0.5)."""
    if not isinstance(contamination, (int, float)):
        raise InvalidContaminationError(f"Contamination must be a float, got {type(contamination)}")

    if math.isnan(contamination) or math.isinf(contamination):
        raise InvalidContaminationError(f"Contamination must be finite, got {contamination}")

    if contamination <= 0.0 or contamination >= 0.5:
        raise InvalidContaminationError(
            f"Contamination prior must be strictly in the open interval (0.0, 0.5), got {contamination}"
        )

    return float(contamination)


@dataclass
class ScorePercentiles:
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ScoreCalibrationReceipt:
    method: str  # empirical_percentile, min_max, z_score
    threshold: float
    score_min: float
    score_max: float
    score_mean: float
    score_std: float
    percentiles: ScorePercentiles

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "threshold": self.threshold,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "percentiles": self.percentiles.to_dict(),
        }


@dataclass
class ClusteringPartitionProtocol:
    algorithm: str  # kmeans, hdbscan, dbscan
    metric: str = "euclidean"
    n_clusters: Optional[int] = 3
    eps: Optional[float] = None
    min_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "metric": self.metric,
            "n_clusters": self.n_clusters,
            "eps": self.eps,
            "min_samples": self.min_samples,
        }


@dataclass
class EvaluationProfile:
    silhouette_score: Optional[float]
    davies_bouldin_index: Optional[float]
    calinski_harabasz_score: Optional[float]
    num_noise_points: int
    num_clusters_found: int
    inertia: Optional[float] = None
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "silhouette_score": self.silhouette_score,
            "davies_bouldin_index": self.davies_bouldin_index,
            "calinski_harabasz_score": self.calinski_harabasz_score,
            "inertia": self.inertia,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "num_noise_points": self.num_noise_points,
            "num_clusters_found": self.num_clusters_found,
        }


@dataclass
class AnomalyClusteringDeliveryContract:
    supports_out_of_sample_scoring: bool = True
    supports_out_of_sample_assignment: bool = True
    anomaly_score_bounded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyClusteringAdapter:
    schema_version: str = "v2"
    adapter_id: str = "anomaly_cluster_default"
    dataset_id: str = "dataset_default"
    adapter_type: str = "combined"  # anomaly_detection, clustering, combined
    feature_columns: List[str] = field(default_factory=lambda: ["f1", "f2"])
    contamination_prior: float = 0.05
    score_calibration: Dict[str, Any] = field(default_factory=dict)
    clustering_protocol: Dict[str, Any] = field(default_factory=dict)
    evaluation_profile: Dict[str, Any] = field(default_factory=dict)
    delivery_inference: Dict[str, Any] = field(default_factory=lambda: AnomalyClusteringDeliveryContract().to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adapter_id": self.adapter_id,
            "dataset_id": self.dataset_id,
            "adapter_type": self.adapter_type,
            "feature_columns": self.feature_columns,
            "contamination_prior": self.contamination_prior,
            "score_calibration": self.score_calibration,
            "clustering_protocol": self.clustering_protocol,
            "evaluation_profile": self.evaluation_profile,
            "delivery_inference": self.delivery_inference,
        }

    @classmethod
    def synthetic(cls, dataset_id: str = "synth_anomaly_ds") -> AnomalyClusteringAdapter:
        calib = ScoreCalibrationReceipt(
            method="empirical_percentile",
            threshold=0.78,
            score_min=0.0,
            score_max=1.0,
            score_mean=0.25,
            score_std=0.18,
            percentiles=ScorePercentiles(p05=0.02, p25=0.12, p50=0.22, p75=0.35, p95=0.72),
        )
        proto = ClusteringPartitionProtocol(
            algorithm="kmeans",
            metric="euclidean",
            n_clusters=3,
            eps=None,
            min_samples=None,
        )
        eval_prof = EvaluationProfile(
            silhouette_score=0.62,
            davies_bouldin_index=0.55,
            calinski_harabasz_score=312.4,
            num_noise_points=0,
            num_clusters_found=3,
            inertia=45.2,
            roc_auc=0.94,
            pr_auc=0.88,
        )
        return cls(
            schema_version="v2",
            adapter_id=f"adapter_{dataset_id}",
            dataset_id=dataset_id,
            adapter_type="combined",
            feature_columns=["feature_0", "feature_1", "feature_2"],
            contamination_prior=0.05,
            score_calibration=calib.to_dict(),
            clustering_protocol=proto.to_dict(),
            evaluation_profile=eval_prof.to_dict(),
            delivery_inference=AnomalyClusteringDeliveryContract().to_dict(),
        )


def calibrate_scores(raw_scores: np.ndarray, contamination: float) -> Tuple[np.ndarray, ScoreCalibrationReceipt, float, float]:
    """Calibrates raw anomaly scores (where higher means more anomalous) into bounded [0, 1] range.

    Returns:
        calibrated_scores: bounded array in [0, 1]
        receipt: ScoreCalibrationReceipt
        train_min: minimum raw score from training
        train_max: maximum raw score from training
    """
    validate_contamination_prior(contamination)
    train_min = float(np.min(raw_scores))
    train_max = float(np.max(raw_scores))
    span = train_max - train_min
    if span <= 1e-12:
        span = 1.0

    calibrated = np.clip((raw_scores - train_min) / span, 0.0, 1.0)
    # The threshold corresponds to the (100 * (1 - contamination)) percentile of calibrated scores
    threshold = float(np.percentile(calibrated, 100.0 * (1.0 - contamination)))

    percentiles = ScorePercentiles(
        p05=float(np.percentile(calibrated, 5)),
        p25=float(np.percentile(calibrated, 25)),
        p50=float(np.percentile(calibrated, 50)),
        p75=float(np.percentile(calibrated, 75)),
        p95=float(np.percentile(calibrated, 95)),
    )

    receipt = ScoreCalibrationReceipt(
        method="empirical_percentile",
        threshold=threshold,
        score_min=float(np.min(calibrated)),
        score_max=float(np.max(calibrated)),
        score_mean=float(np.mean(calibrated)),
        score_std=float(np.std(calibrated)),
        percentiles=percentiles,
    )
    return calibrated, receipt, train_min, train_max


class IsolationForestDetector:
    """Outlier detector using Isolation Forest with empirical score calibration."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ):
        self.contamination = validate_contamination_prior(contamination)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_: Optional[IsolationForest] = None
        self.calibration_: Optional[ScoreCalibrationReceipt] = None
        self.train_min_: float = 0.0
        self.train_max_: float = 1.0
        self.n_features_in_: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> IsolationForestDetector:
        arr = validate_input_features(X)
        self.n_features_in_ = arr.shape[1]

        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self.model_.fit(arr)

        # Scikit-learn's decision_function returns negative values for outliers, positive for inliers.
        # Negate it so higher means more anomalous.
        raw_scores = -self.model_.decision_function(arr)
        _, receipt, t_min, t_max = calibrate_scores(raw_scores, self.contamination)
        self.calibration_ = receipt
        self.train_min_ = t_min
        self.train_max_ = t_max
        return self

    def predict_anomaly_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None or self.calibration_ is None:
            raise RuntimeError("IsolationForestDetector must be fitted before predict_anomaly_score")
        arr = validate_input_features(X)
        if arr.shape[1] != self.n_features_in_:
            raise OutOfSampleProjectionError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {arr.shape[1]}"
            )

        raw = -self.model_.decision_function(arr)
        span = self.train_max_ - self.train_min_
        if span <= 1e-12:
            span = 1.0
        calibrated = np.clip((raw - self.train_min_) / span, 0.0, 1.0)
        return calibrated

    def predict_is_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        scores = self.predict_anomaly_score(X)
        return scores >= self.calibration_.threshold


class LocalOutlierFactorDetector:
    """Outlier detector using Local Outlier Factor with novelty detection and score calibration."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_neighbors: int = 20,
    ):
        self.contamination = validate_contamination_prior(contamination)
        self.n_neighbors = n_neighbors
        self.model_: Optional[LocalOutlierFactor] = None
        self.calibration_: Optional[ScoreCalibrationReceipt] = None
        self.train_min_: float = 0.0
        self.train_max_: float = 1.0
        self.n_features_in_: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> LocalOutlierFactorDetector:
        arr = validate_input_features(X)
        self.n_features_in_ = arr.shape[1]

        k = min(self.n_neighbors, max(1, arr.shape[0] - 1))
        self.model_ = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=k,
            novelty=True,
        )
        self.model_.fit(arr)

        raw_scores = -self.model_.decision_function(arr)
        _, receipt, t_min, t_max = calibrate_scores(raw_scores, self.contamination)
        self.calibration_ = receipt
        self.train_min_ = t_min
        self.train_max_ = t_max
        return self

    def predict_anomaly_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None or self.calibration_ is None:
            raise RuntimeError("LocalOutlierFactorDetector must be fitted before predict_anomaly_score")
        arr = validate_input_features(X)
        if arr.shape[1] != self.n_features_in_:
            raise OutOfSampleProjectionError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {arr.shape[1]}"
            )

        raw = -self.model_.decision_function(arr)
        span = self.train_max_ - self.train_min_
        if span <= 1e-12:
            span = 1.0
        calibrated = np.clip((raw - self.train_min_) / span, 0.0, 1.0)
        return calibrated

    def predict_is_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        scores = self.predict_anomaly_score(X)
        return scores >= self.calibration_.threshold


def compute_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = "euclidean",
    inertia: Optional[float] = None,
) -> EvaluationProfile:
    """Computes silhouette, Davies-Bouldin, and Calinski-Harabasz metrics on clustered samples."""
    num_noise = int(np.sum(labels == -1))
    valid_mask = labels >= 0
    num_valid = int(np.sum(valid_mask))
    unique_clusters = np.unique(labels[valid_mask]) if num_valid > 0 else np.array([])
    num_clusters = len(unique_clusters)

    if num_clusters < 2 or num_valid < 2:
        return EvaluationProfile(
            silhouette_score=None,
            davies_bouldin_index=None,
            calinski_harabasz_score=None,
            num_noise_points=num_noise,
            num_clusters_found=max(1, num_clusters),
            inertia=inertia,
        )

    X_valid = X[valid_mask]
    y_valid = labels[valid_mask]

    try:
        sil = float(silhouette_score(X_valid, y_valid, metric=metric))
    except Exception:
        sil = None

    try:
        db = float(davies_bouldin_score(X_valid, y_valid))
    except Exception:
        db = None

    try:
        ch = float(calinski_harabasz_score(X_valid, y_valid))
    except Exception:
        ch = None

    return EvaluationProfile(
        silhouette_score=sil,
        davies_bouldin_index=db,
        calinski_harabasz_score=ch,
        num_noise_points=num_noise,
        num_clusters_found=num_clusters,
        inertia=inertia,
    )


class KMeansClusterer:
    """Unsupervised KMeans clustering with silhouette, inertia, and out-of-sample projection."""

    def __init__(
        self,
        n_clusters: int = 3,
        metric: str = "euclidean",
        random_state: int = 42,
    ):
        if n_clusters < 2:
            raise InvalidClusteringConfigError(f"KMeans requires n_clusters >= 2, got {n_clusters}")
        if metric not in ["euclidean", "manhattan", "cosine"]:
            raise InvalidClusteringConfigError(f"Unsupported clustering metric '{metric}'")

        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.model_: Optional[KMeans] = None
        self.evaluation_: Optional[EvaluationProfile] = None
        self.n_features_in_: int = 0

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> KMeansClusterer:
        arr = validate_input_features(X)
        if arr.shape[0] < self.n_clusters:
            raise InvalidClusteringConfigError(
                f"Sample count ({arr.shape[0]}) must be >= n_clusters ({self.n_clusters})"
            )

        self.n_features_in_ = arr.shape[1]
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = self.model_.fit_predict(arr)
        inertia = float(self.model_.inertia_)
        self.evaluation_ = compute_clustering_metrics(arr, labels, metric=self.metric, inertia=inertia)
        return self

    def assign_cluster(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("KMeansClusterer must be fitted before assign_cluster")
        arr = validate_input_features(X)
        if arr.shape[1] != self.n_features_in_:
            raise OutOfSampleProjectionError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {arr.shape[1]}"
            )
        return self.model_.predict(arr)

    def compute_cluster_distances(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("KMeansClusterer must be fitted before compute_cluster_distances")
        arr = validate_input_features(X)
        return self.model_.transform(arr)


class DensityClusterer:
    """Density-based clustering (HDBSCAN / DBSCAN) with out-of-sample projection guarantees."""

    def __init__(
        self,
        algorithm: str = "hdbscan",
        eps: Optional[float] = 0.5,
        min_samples: int = 5,
        min_cluster_size: int = 5,
        metric: str = "euclidean",
    ):
        if algorithm not in ["hdbscan", "dbscan"]:
            raise InvalidClusteringConfigError(f"Unsupported density algorithm '{algorithm}', expected 'hdbscan' or 'dbscan'")
        if metric not in ["euclidean", "manhattan", "cosine"]:
            raise InvalidClusteringConfigError(f"Unsupported metric '{metric}'")
        if algorithm == "dbscan":
            if eps is None or eps <= 0.0:
                raise InvalidClusteringConfigError(f"DBSCAN requires eps > 0, got {eps}")
        if min_samples < 1:
            raise InvalidClusteringConfigError(f"min_samples must be >= 1, got {min_samples}")
        if min_cluster_size < 2:
            raise InvalidClusteringConfigError(f"min_cluster_size must be >= 2, got {min_cluster_size}")

        self.algorithm = algorithm
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.model_: Any = None
        self.evaluation_: Optional[EvaluationProfile] = None
        self.n_features_in_: int = 0
        self.cluster_medoids_: Dict[int, np.ndarray] = {}
        self.cluster_max_radii_: Dict[int, float] = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> DensityClusterer:
        arr = validate_input_features(X)
        self.n_features_in_ = arr.shape[1]

        if self.algorithm == "hdbscan":
            self.model_ = HDBSCAN(
                min_cluster_size=min(self.min_cluster_size, max(2, arr.shape[0] // 2)),
                min_samples=min(self.min_samples, max(1, arr.shape[0] - 1)),
                metric=self.metric,
                copy=True,
            )
            labels = self.model_.fit_predict(arr)
        else:
            self.model_ = DBSCAN(
                eps=self.eps if self.eps is not None else 0.5,
                min_samples=min(self.min_samples, max(1, arr.shape[0] - 1)),
                metric=self.metric,
            )
            labels = self.model_.fit_predict(arr)

        self.evaluation_ = compute_clustering_metrics(arr, labels, metric=self.metric)

        # Precompute cluster medoids and radii for out-of-sample projection
        self.cluster_medoids_.clear()
        self.cluster_max_radii_.clear()
        for cid in np.unique(labels):
            if cid < 0:
                continue
            c_points = arr[labels == cid]
            centroid = np.mean(c_points, axis=0)
            dists = np.linalg.norm(c_points - centroid, axis=1)
            medoid_idx = int(np.argmin(dists))
            medoid = c_points[medoid_idx]
            max_r = float(np.max(dists)) if len(dists) > 0 else 1.0
            self.cluster_medoids_[int(cid)] = medoid
            # Tolerance radius for assigning an out-of-sample point
            self.cluster_max_radii_[int(cid)] = max(max_r * 1.5, (self.eps or 0.5) * 1.5)

        return self

    def assign_cluster(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Projects out-of-sample observations to nearest cluster medoid within tolerance, or -1 (noise)."""
        if self.model_ is None:
            raise RuntimeError("DensityClusterer must be fitted before assign_cluster")
        arr = validate_input_features(X)
        if arr.shape[1] != self.n_features_in_:
            raise OutOfSampleProjectionError(
                f"Feature dimension mismatch: expected {self.n_features_in_}, got {arr.shape[1]}"
            )

        if not self.cluster_medoids_:
            return np.full(arr.shape[0], -1, dtype=np.int64)

        assignments = np.full(arr.shape[0], -1, dtype=np.int64)
        cids = list(self.cluster_medoids_.keys())
        medoids = np.stack([self.cluster_medoids_[cid] for cid in cids], axis=0)
        radii = np.array([self.cluster_max_radii_[cid] for cid in cids])

        for i in range(arr.shape[0]):
            dists = np.linalg.norm(medoids - arr[i], axis=1)
            best_idx = int(np.argmin(dists))
            if dists[best_idx] <= radii[best_idx]:
                assignments[i] = cids[best_idx]
            else:
                assignments[i] = -1

        return assignments


def evaluate_supervised_anomaly_metrics(
    scores: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[float, float]:
    """Computes ROC-AUC and PR-AUC when ground-truth anomaly labels are available."""
    if len(scores) != len(true_labels):
        raise ValueError(f"Length mismatch: scores ({len(scores)}) vs true_labels ({len(true_labels)})")

    y = np.asarray(true_labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return 0.5, 0.0

    roc = float(roc_auc_score(y, scores))
    pr = float(average_precision_score(y, scores))
    return roc, pr


class AnomalyClusteringPredictor:
    """Production Delivery Inference Contract for Anomaly Detection & Clustering.

    Provides out-of-sample score predictions in [0, 1] and deterministic cluster assignments.
    """

    def __init__(
        self,
        detector: Optional[Union[IsolationForestDetector, LocalOutlierFactorDetector]] = None,
        clusterer: Optional[Union[KMeansClusterer, DensityClusterer]] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.detector = detector
        self.clusterer = clusterer
        self.feature_names = feature_names or []

    def predict_anomaly_score(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.detector is None:
            raise RuntimeError("AnomalyClusteringPredictor has no anomaly detector configured")
        return self.detector.predict_anomaly_score(X)

    def predict_is_anomaly(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.detector is None:
            raise RuntimeError("AnomalyClusteringPredictor has no anomaly detector configured")
        return self.detector.predict_is_anomaly(X)

    def assign_cluster(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.clusterer is None:
            raise RuntimeError("AnomalyClusteringPredictor has no clusterer configured")
        return self.clusterer.assign_cluster(X)

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        if self.detector is not None:
            result["anomaly_score"] = self.predict_anomaly_score(X)
            result["is_anomaly"] = self.predict_is_anomaly(X)
        if self.clusterer is not None:
            result["cluster_label"] = self.assign_cluster(X)
        return result
