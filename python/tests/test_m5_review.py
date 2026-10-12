"""Regression checks using trusted static data; no custom candidate execution."""
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from mlrescurix_worker.scientific.features import OutOfFoldTargetEncoder, FeatureTransformSpec, SplitAwareFeaturePipeline
from mlrescurix_worker.scientific.ensemble import VotingEnsemble
from mlrescurix_worker.scientific.calibration import SplitAwareCalibrator

class M5ReviewTests(unittest.TestCase):
    def test_unknown_category_uses_fold_mean_and_ignores_index_labels(self):
        X = pd.DataFrame({'cat': list('abcdef')}, index=[10, 20, 30, 40, 50, 60])
        y = np.array([0., 1., 2., 5., 9., 10.])
        enc = OutOfFoldTargetEncoder(['cat'], n_splits=3)
        out = enc.fit_transform(X, y)
        for train, held in KFold(3, shuffle=True, random_state=2026).split(X):
            np.testing.assert_allclose(out.cat_te.iloc[held], y[train].mean())
        enc.fit(X, y)  # Non-positional indices must also work on refit.

    def test_imputation_without_scaling_and_new_missing_columns(self):
        X = pd.DataFrame({'a': [1., 2., 3.], 'empty': [np.nan]*3})
        pipe = SplitAwareFeaturePipeline(FeatureTransformSpec('test', numeric_scaler='none'))
        train = pipe.fit_transform_training(X, np.array([0, 1, 0]))
        pred = pipe.transform(pd.DataFrame({'a': [np.nan], 'empty': [np.nan]}))
        self.assertEqual(list(train.columns), list(pred.columns))
        self.assertTrue(np.isfinite(pred.to_numpy()).all())
        self.assertEqual(pred.a.iloc[0], 2.)

    def test_invalid_voting_weights_rejected(self):
        for weights in ([1.], [-1., 2.], [0., 0.], [np.nan, 1.]):
            with self.assertRaises(ValueError):
                VotingEnsemble(['a', 'b'], weights)
        with self.assertRaises(ValueError):
            VotingEnsemble(['a']).load_parent_models([])

    def test_calibration_rejects_wrong_class_order_and_invalid_probabilities(self):
        c = SplitAwareCalibrator().fit_from_oof(np.array([[.8,.2],[.2,.8]]), np.array([0,1]), [0,1])
        with self.assertRaises(ValueError):
            c.calibrate_predictions({'probabilities': [[.2,.8]], 'classes': [1,0]})
        with self.assertRaises(ValueError):
            c.transform(np.array([[.8,.8]]))

    def test_voting_aligns_parent_probability_columns(self):
        class Parent:
            def __init__(self, classes, probabilities):
                self.classes_ = np.array(classes)
                self.probabilities = probabilities
            def predict_proba(self, X):
                return np.tile(self.probabilities, (len(X), 1))
        voting = VotingEnsemble(['a', 'b'])
        voting.classes_ = ['no', 'yes']
        voting.models_ = [Parent(['no', 'yes'], [.1, .9]), Parent(['yes', 'no'], [.9, .1])]
        result = voting.predict(pd.DataFrame({'x': [1]}))
        self.assertEqual(result['prediction'], ['yes'])
        np.testing.assert_allclose(result['probabilities'], [[.1, .9]])
