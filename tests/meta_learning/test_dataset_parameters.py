from turbo_ml.meta_learning.dataset_parameters import sota_dataset_parameters, SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from datasets import get_iris, get_adult
import numpy as np


def test_sota_np():
    dataset, target = get_iris()
    parameters = sota_dataset_parameters(dataset, target)
    assert isinstance(parameters, np.ndarray)
    assert parameters.dtype == 'float64'


def test_sota_dict():
    dataset, target = get_iris()
    parameters = sota_dataset_parameters(dataset, target, as_dict=True)
    assert isinstance(parameters, dict)


def test_simple_parameter_extraction():
    dataset, target = get_iris()
    parameters = SimpleMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    dataset, target = get_adult()
    parameters = SimpleMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    parameters = SimpleMetaFeatures()(dataset, target, as_dict=True)
    assert isinstance(parameters, dict)


def test_statistical_parameter_extraction():
    dataset, target = get_iris()
    parameters = StatisticalMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    dataset, target = get_adult()
    parameters = StatisticalMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    parameters = StatisticalMetaFeatures()(dataset, target, as_dict=True)
    assert isinstance(parameters, dict)


def test_pca_parameter_extraction():
    dataset, target = get_iris()
    parameters = PCAMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    dataset, target = get_adult()
    parameters = PCAMetaFeatures()(dataset, target)
    assert isinstance(parameters, np.ndarray)
    parameters = PCAMetaFeatures()(dataset, target, as_dict=True)
    assert isinstance(parameters, dict)
