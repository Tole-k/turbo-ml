import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ..base.preprocess import Preprocessor
from .type_inferer import TypeInferer


class Normalizer(Preprocessor):
    scaler: MinMaxScaler
    target_scaler: MinMaxScaler
    type_inferer: TypeInferer

    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.type_inferer = TypeInferer()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        x_dtypes = self.type_inferer.infer(data)
        numeric_cols = x_dtypes[x_dtypes.isin(["floating", "integer", "mixed-integer-float"])].index
        if not numeric_cols.size:
            return data
        numerical_frame = data[numeric_cols]
        normalized_numeric_frame = pd.DataFrame(self.scaler.fit_transform(
            numerical_frame), columns=self.scaler.get_feature_names_out()).astype(float)
        data[numeric_cols] = normalized_numeric_frame
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.infer_target(target)
        if y_dtype in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.target_scaler.fit_transform(np.transpose([target]))[:, 0]).astype(float)
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        x_dtypes = self.type_inferer.recall()
        numeric_cols = x_dtypes[x_dtypes.isin(["floating", "integer", "mixed-integer-float"])].index
        if not numeric_cols.size:
            return data
        numerical_frame = data[numeric_cols]
        normalized_numeric_frame = pd.DataFrame(self.scaler.transform(
            numerical_frame), columns=self.scaler.get_feature_names_out()).astype(float)
        data[numeric_cols] = normalized_numeric_frame
        return data

    def transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.recall_target()
        if y_dtype in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.target_scaler.transform(np.transpose([target]))[:, 0]).astype(float)
        return target

    def inverse_transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.recall_target()
        if y_dtype in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.target_scaler.inverse_transform(np.transpose([target]))[:, 0]).astype(float)
        return target


def main():

    dataset = pd.DataFrame({
        'A': [1, 2, 3, 4],
        'B': [10, 20, 30, 40],
        'C': ["a", "b", "c", "d"],
        'D': [0, 1, 0, 1],
        'E': [1, 0.2, "c", 0],
        'target': [45, 22, 69, 18],
    })
    data = dataset.drop(columns=["target"])
    target = dataset["target"]
    print("Before Normalization:")
    print(data)
    print(target)
    print()
    normalizer = Normalizer()
    data = normalizer.fit_transform(data)
    print("After Normalization:")
    print(data)
    print()
    data = pd.DataFrame({
        'A': [1.5, 2.5, 3.5, 4.5],
        'B': [11, 21, 31, 41],
        'C': ["a", "b", "c", "d"],
        'D': [0, 1, 0, 1],
        'E': [1, 0.2, "c", 0],
    })
    data = normalizer.transform(data)
    print("Normalizing more data:")
    print(data)
    print()
    target = normalizer.fit_transform_target(target)
    print("After Normalizing Target:")
    print(target)
    target = normalizer.inverse_transform_target(target)
    print("Inverse Target:")
    print(target)
    assert all(target == [45, 22, 69, 18])


if __name__ == '__main__':
    main()
