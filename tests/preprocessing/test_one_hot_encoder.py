import warnings
import pandas as pd
from sageml.preprocessing.encoder import Encoder

BASE_DATAFRAME = pd.DataFrame(
    {
        "A": [1, 2, 3, 4],
        "B": [10, 20, 30, 40],
        "C": ["a", "b", "c", "d"],
        "D": [False, True, False, True],
        "E": [1, 0.2, "c", 0.2],
        "target": ["frog", "duck", "hen", "frog"],
    }
)


def test_one_hot_encoder():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    encoder = Encoder()
    data = encoder.fit_transform(data)
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "a", "a", "b"],
            "D": [False, True, False, True],
            "E": [1, 1, 1, 1],
        }
    )
    data2 = encoder.transform(data2)

    data_true = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "D": [False, True, False, True],
            "C_a": [True, False, False, False],
            "C_b": [False, True, False, False],
            "C_c": [False, False, True, False],
            "C_d": [False, False, False, True],
            "E_0.2": [False, True, False, True],
            "E_1": [True, False, False, False],
            "E_c": [False, False, True, False],
        }
    )

    data2_true = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "D": [False, True, False, True],
            "C_a": [True, True, True, False],
            "C_b": [False, False, False, True],
            "C_c": [False, False, False, False],
            "C_d": [False, False, False, False],
            "E_0.2": [False, False, False, False],
            "E_1": [True, True, True, True],
            "E_c": [False, False, False, False],
        }
    )

    assert data.equals(data_true) and data2.equals(data2_true)


def test_new_categories():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    encoder = Encoder()
    data = encoder.fit_transform(data)
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "a", "a", "x"],
            "D": [False, True, False, True],
            "E": [1, 1, 1, 0],
        }
    )

    data2_true = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "D": [False, True, False, True],
            "C_a": [True, True, True, False],
            "C_b": [False, False, False, False],
            "C_c": [False, False, False, False],
            "C_d": [False, False, False, False],
            "E_0.2": [False, False, False, False],
            "E_1": [True, True, True, False],
            "E_c": [False, False, False, False],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data2 = encoder.transform(data2)

    assert data2.equals(data2_true)


def test_target():
    target = BASE_DATAFRAME["target"]
    encoder = Encoder()
    target = encoder.fit_transform_target(target)
    assert all(target == pd.Series([1, 0, 2, 1]))


def test_one_hot_encoder_inverse():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    encoder = Encoder()
    data2 = encoder.fit_transform(data)
    data_back = encoder.inverse_transform(data2)
    data['E'] = data['E'].astype(str)
    assert data.equals(data_back)


def test_target_inverse():
    target = BASE_DATAFRAME["target"]
    encoder = Encoder()
    target_tr = encoder.fit_transform_target(target)
    target_back = encoder.inverse_transform_target(target_tr)
    assert all(target == target_back)
