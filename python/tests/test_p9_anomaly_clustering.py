import json
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

from mlrescurix_worker.scientific.anomaly_clustering import (
    AnomalyClusteringAdapter,
    AnomalyClusteringPredictor,
    ClusteringPartitionProtocol,
    DensityClusterer,
    EvaluationProfile,
    InvalidClusteringConfigError,
    InvalidContaminationError,
    IsolationForestDetector,
    KMeansClusterer,
    LocalOutlierFactorDetector,
    NonFiniteFeatureError,
    OutOfSampleProjectionError,
    ScoreCalibrationReceipt,
    compute_clustering_metrics,
    evaluate_supervised_anomaly_metrics,
    validate_contamination_prior,
    validate_input_features,
)

ROOT = Path(__file__).resolve().parents[2]


class TestP9AnomalyClustering(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="mlrescurix_p9_ac_"))
        self.schema = json.loads((ROOT / "contracts/v2/anomaly_clustering_adapter.schema.json").read_text())

    def test_p9_3_schema_validation(self):
        """P9.3: Validate AnomalyClusteringAdapter contract under strict Draft 2020-12 schema."""
        adapter = AnomalyClusteringAdapter.synthetic("ds_cyber_traffic")
        jsonschema.Draft202012Validator.check_schema(self.schema)
        jsonschema.validate(instance=adapter.to_dict(), schema=self.schema)

    def test_p9_3_isolation_forest_outlier_detection(self):
        """P9.3: IsolationForest with empirical threshold calibration and bounded score receipts."""
        rng = np.random.RandomState(42)
        # 200 normal points from N(0, 1) and 10 outliers from N(10, 1)
        X_normal = rng.randn(200, 3)
        X_outliers = rng.randn(10, 3) + 10.0
        X_train = np.vstack([X_normal, X_outliers])

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        detector.fit(X_train)

        # Receipt checks
        calib = detector.calibration_
        self.assertIsNotNone(calib)
        self.assertEqual(calib.method, "empirical_percentile")
        self.assertGreater(calib.threshold, 0.0)
        self.assertLessEqual(calib.threshold, 1.0)
        self.assertEqual(calib.score_min, 0.0)
        self.assertLessEqual(calib.score_max, 1.0)
        self.assertGreater(calib.score_std, 0.0)

        # Out-of-sample predictions
        X_test_normal = rng.randn(20, 3)
        X_test_outliers = rng.randn(5, 3) + 12.0
        scores_normal = detector.predict_anomaly_score(X_test_normal)
        scores_outliers = detector.predict_anomaly_score(X_test_outliers)

        # Boundedness
        self.assertTrue((scores_normal >= 0.0).all() and (scores_normal <= 1.0).all())
        self.assertTrue((scores_outliers >= 0.0).all() and (scores_outliers <= 1.0).all())

        # Anomaly scores on outliers must be significantly higher than on normal points
        self.assertGreater(np.mean(scores_outliers), np.mean(scores_normal))
        self.assertTrue((detector.predict_is_anomaly(X_test_outliers)).all())

    def test_p9_3_local_outlier_factor_detector(self):
        """P9.3: LocalOutlierFactor with novelty detection and calibrated score distributions."""
        rng = np.random.RandomState(101)
        X_train = rng.randn(150, 4)
        detector = LocalOutlierFactorDetector(contamination=0.08, n_neighbors=15)
        detector.fit(X_train)

        calib = detector.calibration_
        self.assertIsNotNone(calib)
        self.assertGreater(calib.threshold, 0.0)
        self.assertLessEqual(calib.threshold, 1.0)

        # Out of sample projection
        X_query = rng.randn(25, 4)
        scores = detector.predict_anomaly_score(X_query)
        self.assertEqual(scores.shape, (25,))
        self.assertTrue((scores >= 0.0).all() and (scores <= 1.0).all())

        is_anomaly = detector.predict_is_anomaly(X_query)
        self.assertEqual(is_anomaly.shape, (25,))
        self.assertEqual(is_anomaly.dtype, bool)

    def test_p9_3_kmeans_unsupervised_clustering(self):
        """P9.3: KMeans clustering with silhouette, inertia, and out-of-sample centroid projection."""
        rng = np.random.RandomState(2026)
        # 3 well-separated isotropic clusters
        c1 = rng.randn(60, 2) + np.array([0.0, 0.0])
        c2 = rng.randn(60, 2) + np.array([10.0, 0.0])
        c3 = rng.randn(60, 2) + np.array([5.0, 10.0])
        X = np.vstack([c1, c2, c3])

        clusterer = KMeansClusterer(n_clusters=3, metric="euclidean", random_state=42)
        clusterer.fit(X)

        eval_prof = clusterer.evaluation_
        self.assertIsNotNone(eval_prof)
        self.assertEqual(eval_prof.num_clusters_found, 3)
        self.assertEqual(eval_prof.num_noise_points, 0)
        self.assertIsNotNone(eval_prof.silhouette_score)
        self.assertGreater(eval_prof.silhouette_score, 0.5)  # Well-separated clusters
        self.assertIsNotNone(eval_prof.davies_bouldin_index)
        self.assertLess(eval_prof.davies_bouldin_index, 1.0)
        self.assertIsNotNone(eval_prof.calinski_harabasz_score)
        self.assertGreater(eval_prof.calinski_harabasz_score, 100.0)
        self.assertIsNotNone(eval_prof.inertia)

        # Out of sample assignment
        X_test = np.array([[0.1, 0.2], [10.2, -0.1], [5.1, 9.8]])
        labels = clusterer.assign_cluster(X_test)
        self.assertEqual(len(np.unique(labels)), 3)

        distances = clusterer.compute_cluster_distances(X_test)
        self.assertEqual(distances.shape, (3, 3))

    def test_p9_3_density_based_clustering_hdbscan_and_dbscan(self):
        """P9.3: Density-based clustering (HDBSCAN / DBSCAN) with out-of-sample projection."""
        rng = np.random.RandomState(42)
        # Dense cluster 1 around (0, 0), dense cluster 2 around (10, 10), plus random noise
        c1 = rng.randn(50, 2) * 0.4
        c2 = rng.randn(50, 2) * 0.4 + 10.0
        noise = rng.uniform(-15.0, 25.0, size=(15, 2))
        X = np.vstack([c1, c2, noise])

        # 1. HDBSCAN
        hdb = DensityClusterer(algorithm="hdbscan", min_cluster_size=10, min_samples=5)
        hdb.fit(X)
        eval_hdb = hdb.evaluation_
        self.assertGreaterEqual(eval_hdb.num_clusters_found, 2)
        self.assertGreater(eval_hdb.num_noise_points, 0)

        # Out of sample projection for HDBSCAN
        X_probe = np.array([
            [0.1, 0.1],    # inside cluster 1
            [10.1, 9.9],   # inside cluster 2
            [50.0, 50.0],  # far away noise
        ])
        labels_hdb = hdb.assign_cluster(X_probe)
        self.assertEqual(labels_hdb[2], -1)  # Far point projected to noise

        # 2. DBSCAN
        db = DensityClusterer(algorithm="dbscan", eps=1.2, min_samples=5)
        db.fit(X)
        eval_db = db.evaluation_
        self.assertGreaterEqual(eval_db.num_clusters_found, 2)

        labels_db = db.assign_cluster(X_probe)
        self.assertEqual(labels_db[2], -1)  # Far point projected to noise

    def test_p9_3_metric_suites_and_supervised_evaluation(self):
        """P9.3: Metric suites (Silhouette, Davies-Bouldin, Calinski-Harabasz, ROC-AUC/PR-AUC)."""
        rng = np.random.RandomState(7)
        X_clean = rng.randn(100, 2)
        labels_clean = np.array([0] * 50 + [1] * 50)
        prof = compute_clustering_metrics(X_clean, labels_clean)
        self.assertIsNotNone(prof.silhouette_score)
        self.assertIsNotNone(prof.davies_bouldin_index)
        self.assertIsNotNone(prof.calinski_harabasz_score)
        self.assertEqual(prof.num_noise_points, 0)

        # Supervised anomaly metrics (ROC-AUC / PR-AUC)
        true_labels = np.array([0] * 90 + [1] * 10)
        # Synthetic anomaly scores: high on true positives, low on true negatives
        scores = np.concatenate([rng.uniform(0.0, 0.3, size=90), rng.uniform(0.7, 1.0, size=10)])
        roc, pr = evaluate_supervised_anomaly_metrics(scores, true_labels)
        self.assertGreater(roc, 0.95)
        self.assertGreater(pr, 0.90)

    def test_p9_3_delivery_inference_contract(self):
        """P9.3: Delivery inference contract with AnomalyClusteringPredictor."""
        rng = np.random.RandomState(42)
        X = rng.randn(80, 3)

        detector = IsolationForestDetector(contamination=0.05, random_state=42).fit(X)
        clusterer = KMeansClusterer(n_clusters=3, random_state=42).fit(X)

        predictor = AnomalyClusteringPredictor(
            detector=detector,
            clusterer=clusterer,
            feature_names=["x0", "x1", "x2"],
        )

        X_new = rng.randn(10, 3)
        res = predictor.predict(X_new)
        self.assertIn("anomaly_score", res)
        self.assertIn("is_anomaly", res)
        self.assertIn("cluster_label", res)

        self.assertEqual(len(res["anomaly_score"]), 10)
        self.assertEqual(len(res["is_anomaly"]), 10)
        self.assertEqual(len(res["cluster_label"]), 10)
        self.assertTrue((res["anomaly_score"] >= 0.0).all() and (res["anomaly_score"] <= 1.0).all())

    def test_p9_3_negative_scientific_cases(self):
        """P9.3: Negative scientific test cases: reject invalid contamination priors, non-finite features, invalid k/metrics."""
        # 1. Contamination prior <= 0.0 or >= 0.5
        with self.assertRaises(InvalidContaminationError):
            validate_contamination_prior(0.0)

        with self.assertRaises(InvalidContaminationError):
            validate_contamination_prior(-0.05)

        with self.assertRaises(InvalidContaminationError):
            validate_contamination_prior(0.5)

        with self.assertRaises(InvalidContaminationError):
            validate_contamination_prior(0.85)

        with self.assertRaises(InvalidContaminationError):
            IsolationForestDetector(contamination=0.55)

        with self.assertRaises(InvalidContaminationError):
            LocalOutlierFactorDetector(contamination=-0.1)

        # 2. Non-finite input features (NaN, Inf, -Inf)
        with self.assertRaises(NonFiniteFeatureError):
            validate_input_features(np.array([[1.0, 2.0], [np.nan, 4.0]]))

        with self.assertRaises(NonFiniteFeatureError):
            validate_input_features(np.array([[1.0, np.inf], [3.0, 4.0]]))

        with self.assertRaises(NonFiniteFeatureError):
            validate_input_features(np.array([[1.0, -np.inf], [3.0, 4.0]]))

        detector = IsolationForestDetector(contamination=0.05)
        with self.assertRaises(NonFiniteFeatureError):
            detector.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))

        # 3. Invalid clustering config: k < 2, invalid metric, eps <= 0
        with self.assertRaises(InvalidClusteringConfigError):
            KMeansClusterer(n_clusters=1)

        with self.assertRaises(InvalidClusteringConfigError):
            KMeansClusterer(n_clusters=3, metric="chebyshev_unsupported")

        with self.assertRaises(InvalidClusteringConfigError):
            DensityClusterer(algorithm="dbscan", eps=0.0)

        with self.assertRaises(InvalidClusteringConfigError):
            DensityClusterer(algorithm="dbscan", eps=-0.5)

        with self.assertRaises(InvalidClusteringConfigError):
            DensityClusterer(algorithm="unknown_algo")

        # 4. Out-of-sample dimension mismatch
        X_train = np.random.randn(50, 3)
        detector.fit(X_train)
        with self.assertRaises(OutOfSampleProjectionError):
            detector.predict_anomaly_score(np.random.randn(5, 4))


if __name__ == "__main__":
    unittest.main()
