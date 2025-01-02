import pandas as pd
from turbo_ml.preprocessing.encoder import Encoder
import warnings

BASE_DATAFRAME = pd.DataFrame(
    {
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40],
        "C": ["a", "b", "c", "d"],
        "D": [0, 1, 0, 1],
        "E": [1, 0.2, "c", 0.2],
        "target": ["frog", "duck", "hen", "frog"],
    }
)


def test_one_hot_encoder():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    ohe = Encoder()
    data = ohe.fit_transform(data)
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "a", "a", "b"],
            "D": [0, 1, 0, 1],
            "E": [1, 1, 1, 1],
        }
    )
    data2 = ohe.transform(data2)


def test_new_categories():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    ohe = Encoder()
    data = ohe.fit_transform(data)
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "a", "a", "x"],
            "D": [0, 1, 0, 1],
            "E": [1, 1, 1, 0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data2 = ohe.transform(data2)
    assert data2 is not None


def test_one_hot_encoder_inverse():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    ohe = Encoder()
    data_cp = data.copy()
    data2 = ohe.fit_transform(data)
    assert data_cp.equals(data)
    # assert data.equals(ohe.inverse_transform(data2))  # TODO fix this
