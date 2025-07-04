import pandas as pd
import numpy as np
from sageml.preprocessing import NanImputer

BASE_DATAFRAME = pd.DataFrame(
    {
        "A": [1, np.nan, 3, 4],
        "B": [10, np.nan, 30, np.nan],
        "C": ["a", "a", np.nan, "b"],
        "D": [0, np.nan, np.nan, 0],
        "E": [1, np.nan, "c", 1],
        "F": [True, False, np.nan, True],
        "target": [45, np.nan, 69, np.nan],
    }
)
BASIC_DATA = BASE_DATAFRAME.drop(columns=["target"])
BASIC_TARGET = BASE_DATAFRAME["target"]


def test_data_transform():
    nan_imputer = NanImputer()
    transformed_data = nan_imputer.fit_transform(BASIC_DATA)

    data_true = pd.DataFrame(
        {
            "A": [1.0, 8/3, 3.0, 4.0],
            "B": [10.0, 20.0, 30.0, 20.0],
            "C": ["a", "a", "a", "b"],
            "D": [0.0, 0.0, 0.0, 0.0],
            "E": ["1", "1", "c", "1"],
            "F": [True, False, True, True],
        }
    )

    data2 = pd.DataFrame(
        {
            "A": [np.nan, 2.5, np.nan, 4.5],
            "B": [11, 21, np.nan, 41],
            "C": [np.nan, "b", "c", np.nan],
            "D": [np.nan, 1, np.nan, 1],
            "E": [1, 0.2, "c", np.nan],
            "F": [True, False, False, np.nan],
        }
    )
    transformed_data2 = nan_imputer.transform(data2)

    data2_true = pd.DataFrame(
        {
            "A": [8/3, 2.5, 8/3, 4.5],
            "B": [11.0, 21.0, 20.0, 41.0],
            "C": ["a", "b", "c", "a"],
            "D": [0.0, 1.0, 0.0, 1.0],
            "E": ["1", "0.2", "c", "1"],
            "F": [True, False, False, True],
        }
    )

    assert transformed_data.equals(data_true) and transformed_data2.equals(data2_true)
