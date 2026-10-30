import numpy as np
import pandas as pd
import pytest
from mlrescurix_worker.scientific.candidate_pipeline import CandidatePipeline, evaluate_candidate, score_metric
from mlrescurix_worker.scientific.autonomous_loop import bounded_evidence


def test_smape_percentage_and_edge_cases():
    assert score_metric('smape', [100], [120]) == pytest.approx(18.18181818)
    assert score_metric('smape', [0, 0], [0, 1]) == 100
    for y, pred in [([1], [1, 2]), ([1], [float('nan')]), ([], [])]:
        with pytest.raises(ValueError):
            score_metric('smape', y, pred)


def test_text_transform_fits_only_training_vocabulary_and_survives_prediction():
    frame = pd.DataFrame({'text': [f'product pack size {i % 4} common' for i in range(40)] + ['secretvalidationtoken'] * 20,
                          'price': [float(i % 4 + 1) for i in range(60)]})
    spec = {'families': ['penalized_linear'], 'weights': [1], 'seed': 42,
            'transforms': [{'action': 'TEXT_TFIDF_SVD', 'parameters': {'columns': ['text']}}]}
    model = CandidatePipeline(spec, 'price', 'regression').fit(frame.iloc[:40])
    assert 'secretvalidationtoken' not in model.text_models_['text'][0].vocabulary_
    assert np.isfinite(model.predict(frame.iloc[40:])).all()
    result = evaluate_candidate(spec, frame, 'price', 'regression', [(list(range(40)), list(range(40, 60)))], 'smape')
    assert 0 <= result['cv_score'] <= 200


def test_outbound_context_compacts_binding_without_mutating_receipt():
    receipt = {'cv_score': 64.87, 'execution_binding': {'data_sha256': 'a' * 64},
               'receipt_identity': 'b' * 64}
    compact = bounded_evidence(receipt)
    assert compact['cv_score'] == 64.87 and compact['receipt_identity'] == 'b' * 64
    assert compact['execution_binding'] == '[execution binding retained in local receipt]'
    assert receipt['execution_binding']['data_sha256'] == 'a' * 64
