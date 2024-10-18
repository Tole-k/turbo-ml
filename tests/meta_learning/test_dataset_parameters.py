from turbo_ml.meta_learning.dataset_parameters import sota_dataset_parameters
from datasets import get_iris
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
