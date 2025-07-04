import pandas as pd
import numpy as np
from sageml.preprocessing import Normalizer

BASE_DATAFRAME = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [10, 20, 30, 40],
    'C': ["a", "b", "c", "d"],
    'D': [False, True, False, True],
    'E': [1, 0.2, "c", 0],
    'target': [45.0, 22.0, 69.0, 18.0],
})
BASIC_DATA = BASE_DATAFRAME.drop(columns=["target"])
BASIC_TARGET = BASE_DATAFRAME["target"]


def test_data_transform():
    normalizer = Normalizer()
    transformed_data = normalizer.fit_transform(BASIC_DATA)

    data_true = pd.DataFrame(
        {
            "A": [0.0, 1/3, 2/3, 1.0],
            "B": [0.0, 1/3, 2/3, 1.0],
            "C": ["a", "b", "c", "d"],
            "D": [False, True, False, True],
            "E": [1, 0.2, "c", 0],
        }
    )
    assert np.allclose(transformed_data["A"], data_true["A"]) and np.allclose(transformed_data["B"],
                                                                              data_true["B"]) and transformed_data[['C', 'D', 'E']].equals(data_true[['C', 'D', 'E']])


def test_target_transform():
    normalizer = Normalizer()
    transformed_target = normalizer.fit_transform_target(BASIC_TARGET)
    target_true = pd.Series([27/51, 4/51, 1.0, 0.0])
    assert np.allclose(transformed_target, target_true)
    target_back = normalizer.inverse_transform_target(transformed_target)
    assert np.allclose(target_back, BASIC_TARGET)
