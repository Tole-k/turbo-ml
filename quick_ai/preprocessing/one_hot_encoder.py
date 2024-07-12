from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import OneHotEncoder


class OneHotEnc():
    def __init__(self) -> None:
        super().__init__()
        self.encoder = OneHotEncoder(dtype=bool)
        self.target_encoder = OneHotEncoder(dtype=bool)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = data.select_dtypes(include=["category",object,"string"]).map(lambda x: str(x))
        self.categorical_cols = categorical_cols.columns
        encoded_data = pd.DataFrame(self.encoder.fit_transform(categorical_cols).toarray(), columns=self.encoder.get_feature_names_out())
        self.encoded_cols = encoded_data.columns
        data.drop(columns=categorical_cols.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
        return data
    
    def fit_transform_target(self, target: pd.Series) -> pd.DataFrame|pd.Series:
        if not np.isin(target.dtype, [np.number, bool]):
            target = pd.DataFrame(self.target_encoder.fit_transform(np.transpose([target])).toarray(),columns=self.target_encoder.get_feature_names_out())
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = data[self.categorical_cols].map(lambda x: str(x))
        encoded_data = pd.DataFrame(self.encoder.transform(categorical_cols).toarray(), columns=self.encoder.get_feature_names_out())
        data.drop(columns=categorical_cols.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        to_decode = data[self.encoded_cols]
        encoded_data = pd.DataFrame(self.encoder.inverse_transform(to_decode), columns=self.categorical_cols)
        data.drop(columns=to_decode.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
        return data

    def transform_target(self, target: pd.Series) -> pd.Series|pd.DataFrame:
        if not np.isin(target.dtype, [np.number, bool]):
            target = pd.DataFrame(self.target_encoder.transform(np.transpose([target])).toarray(),columns=self.target_encoder.get_feature_names_out())
        return target

    def inverse_transform_target(self, target: pd.DataFrame) -> pd.Series:
        ohe_columns = target.select_dtypes(include=[bool])
        target = pd.DataFrame(self.target_encoder.inverse_transform(target))
        return target[0]


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
    ohe = OneHotEnc()
    data = ohe.fit_transform(data)
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
    print("Equal to the original: ", all(target == ["frog", "duck", "hen", "frog"]))


if __name__ == "__main__":
    main()
