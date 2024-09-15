import pandas as pd
from quick_ai.preprocessing.one_hot_encoder import OneHotEncoder

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
    ohe = OneHotEncoder()
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
    ohe = OneHotEncoder()
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
    data2 = ohe.transform(data2)


def test_one_hot_encoder_inverse():
    dataset = BASE_DATAFRAME.copy()
    data = dataset.drop(columns=["target"])
    ohe = OneHotEncoder()
    data_cp = data.copy()
    data2 = ohe.fit_transform(data)
    assert data_cp.equals(data)
    assert data.equals(ohe.inverse_transform(data2))


def main():
    dataset = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "b", "c", "d"],
            "D": [0, 1, 0, 1],
            "E": [1, 0.2, "c", 0.2],
            "target": ["frog", "duck", "hen", "frog"],
        }
    )
    data = dataset.drop(columns=["target"])
    target = dataset["target"]
    print("Before OHE:")
    print(data)
    print(target)
    print()
    ohe = OneHotEncoder()
    data3 = ohe.fit_transform(data)
    print("After OHE:")
    print(data3)

    print()
    print("After OHE:")
    print(data)

    print()
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["a", "a", "a", "a"],
            "D": [0, 1, 0, 1],
            "E": [1, 1, 1, 1],
        }
    )
    data2 = ohe.transform(data2)
    print("Encoding more data:")
    print(data2)
    print()
    data = ohe.inverse_transform(data)
    print("Inverse Data:")
    print(data)
    print()
    target = ohe.fit_transform_target(target)
    print("After OHE Target:")
    print(target)
    target = ohe.inverse_transform_target(target)
    print("Inverse Target:")
    print(target)
    assert all(target == ["frog", "duck", "hen", "frog"])


if __name__ == "__main__":
    main()
