from quick_ai.forecast.hyperparameter_tuning import HyperTuner
from quick_ai.algorithms import DecisionTreeClassifier
from datasets import get_iris


def test_hpo():
    tuner = HyperTuner()
    dataset = get_iris()
    model = DecisionTreeClassifier
    task = 'classification'
    parameters = tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, trials=10)
    assert parameters is not None
    assert len(parameters) > 0
    model = DecisionTreeClassifier(**parameters)

    model.train(dataset[0], dataset[1])
    assert model is not None
    assert model.predict(dataset[0]) is not None
