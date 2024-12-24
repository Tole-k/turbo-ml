from datasets import get_iris, get_breast_cancer
from turbo_ml.algorithms import XGBoostClassifier, SCIKIT_MODELS
from turbo_ml.hpo import HyperTuner


def test_HyperTuner():
    tuner = HyperTuner()
    dataset = get_iris()
    model = SCIKIT_MODELS['GradientBoostingClassifier']
    task = 'classification'
    parameters = tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3)

    assert isinstance(parameters, dict)
    assert len(parameters) > 0
    assert 'ccp_alpha' in parameters
    assert 'learning_rate' in parameters

# def test_HyperTuner_2():
#     tuner = HyperTuner()
#     dataset = get_breast_cancer()
#     model = XGBoostClassifier
#     task = 'classification'
#     parameters = tuner.optimize_hyperparameters(
#         model, dataset, task, no_classes=2)
    
#     assert not parameters
    