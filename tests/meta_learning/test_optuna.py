from datasets import get_iris, get_breast_cancer
from turbo_ml.algorithms import AdaBoostClassifier, XGBoostClassifier
from turbo_ml.meta_learning import HyperTuner


def test_HyperTuner():
    tuner = HyperTuner()
    dataset = get_iris()
    model = AdaBoostClassifier
    task = 'classification'
    parameters = tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3)

    assert isinstance(parameters, dict)
    assert len(parameters) > 0
    assert 'algorithm' in parameters
    assert 'learning_rate' in parameters

# def test_HyperTuner_2():
#     tuner = HyperTuner()
#     dataset = get_breast_cancer()
#     model = XGBoostClassifier
#     task = 'classification'
#     parameters = tuner.optimize_hyperparameters(
#         model, dataset, task, no_classes=2)
    
#     assert not parameters
    