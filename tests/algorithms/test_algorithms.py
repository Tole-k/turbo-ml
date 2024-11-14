from turbo_ml.base.model import Model, get_models_list
from turbo_ml.algorithms import *
from datasets import get_iris


class AlgorithmTesting:
    data, target = get_iris()
    models_list = get_models_list()

    def _baseline_test(model: Model) -> None:
        input_data, input_target = AlgorithmTesting.data.copy(), AlgorithmTesting.target.copy()

        model.train(AlgorithmTesting.data, AlgorithmTesting.target)
        res = model.predict(AlgorithmTesting.data[:10])
        assert input_data.equals(AlgorithmTesting.data), f'''{
            model.__class__.__name__} failed, the data was changed'''
        assert input_target.equals(AlgorithmTesting.target), f'''{
            model.__class__.__name__} failed, the target data was changed'''
        assert res is not None, f'''{
            model.__class__.__name__} failed, the result is None'''
        assert all(result in AlgorithmTesting.target for result in res), f'''{
            model.__class__.__name__} failed, the result is not in target'''

    def _existence_test(model: Model) -> None:
        assert model is not None, f'''{
            model.__class__.__name__} failed, model is None'''
        assert model.__class__ in AlgorithmTesting.models_list, f'''{
            model.__class__.__name__} failed, model is not in models list'''


def test_adaboost():
    model = AdaBoostClassifier()
    AlgorithmTesting._baseline_test(model)
    AlgorithmTesting._existence_test(model)


def test_decision_tree():
    model = DecisionTreeClassifier()
    AlgorithmTesting._baseline_test(model)
    AlgorithmTesting._existence_test(model)


def test_gboost():
    model = GradientBoostingClassifier()
    AlgorithmTesting._baseline_test(model)
    AlgorithmTesting._existence_test(model)


def test_xgboost():
    model = XGBoostClassifier()
    AlgorithmTesting._baseline_test(model)
    AlgorithmTesting._existence_test(model)


def test_random_guesser():
    model = RandomGuesser()
    AlgorithmTesting._baseline_test(model)
    AlgorithmTesting._existence_test(model)
