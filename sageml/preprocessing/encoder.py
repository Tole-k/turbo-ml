import pandas as pd
from sklearn.preprocessing import (
    OneHotEncoder,
    LabelEncoder,
)
from ..base.preprocess import Preprocessor
from .type_inferer import TypeInferer


class Encoder(Preprocessor):
    encoder = OneHotEncoder
    target_encoder = LabelEncoder
    type_inferer = TypeInferer
    column_order: pd.Index
    categorical_cols: pd.Index
    numerical_cols: pd.Index
    encoded_cols: pd.Index

    def __init__(self) -> None:
        super().__init__()
        self.encoder = OneHotEncoder(drop="if_binary", handle_unknown="ignore")
        self.target_encoder = LabelEncoder()
        self.type_inferer = TypeInferer()

    def fit_transform(self, data: pd.DataFrame, unique_values_cap=20) -> pd.DataFrame:
        x_dtypes = self.type_inferer.infer(data)
        self.column_order = data.columns
        self.categorical_cols = x_dtypes[x_dtypes.isin(["string", "mixed-integer", "categorical", "mixed"])].index
        numerical_frame = data.drop(columns=self.categorical_cols)
        self.numerical_cols = numerical_frame.columns
        categorical_frame = data[self.categorical_cols].astype(str)
        categorical_frame.drop(columns=categorical_frame.loc[:, categorical_frame.nunique() > unique_values_cap].columns, inplace=True)
        self.categorical_cols = categorical_frame.columns
        encoded_data = pd.DataFrame(self.encoder.fit_transform(categorical_frame).toarray(),
                                    columns=self.encoder.get_feature_names_out()).astype(bool)
        self.encoded_cols = encoded_data.columns
        data = pd.concat([numerical_frame, encoded_data], axis=1)
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.infer_target(target)
        if y_dtype not in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.target_encoder.fit_transform(target), name=target.name).astype(int)
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_frame = data[self.categorical_cols].astype(str)
        numerical_frame = data[self.numerical_cols]
        encoded_data = pd.DataFrame(self.encoder.transform(categorical_frame).toarray(),
                                    columns=self.encoder.get_feature_names_out()).astype(bool)
        data = pd.concat([numerical_frame, encoded_data], axis=1)
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        to_decode = data[self.encoded_cols]
        encoded_data = pd.DataFrame(self.encoder.inverse_transform(to_decode.astype(bool)), columns=self.categorical_cols)
        data.drop(columns=to_decode.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
        data = data[self.column_order]
        return data

    def transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.recall_target()
        if y_dtype not in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.target_encoder.transform(target), name=target.name).astype(int)
        return target

    def inverse_transform_target(self, target: pd.Series) -> pd.Series:
        return self.target_encoder.inverse_transform(target.astype(int))


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
    encoder = Encoder()
    result = encoder.fit_transform(data)
    print("Before OHE:")
    print(data)
    print(target)
    print()

    print("After OHE:")
    print(result)

    print()
    data2 = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [10, 20, 30, 40],
            "C": ["b", "a", "d", "x"],
            "D": [0, 1, 0, 1],
            "E": [1, 1, 1, 1],
        }
    )
    data2 = encoder.transform(data2)
    print("Encoding more data:")
    print(data2)
    print()
    data = encoder.inverse_transform(result)
    print("Inverse Data:")
    print(data)
    print()
    target = encoder.fit_transform_target(target)
    print("After OHE Target:")
    print(target)
    target = encoder.inverse_transform_target(target)
    print("Inverse Target:")
    print(target)
    assert all(target == ["frog", "duck", "hen", "frog"])

    data3 = pd.DataFrame(
        {
            "A": [2],
            "B": [20],
            "C": ["b"],
            "D": [0],
            "E": [1],
            "target": ["duck"],
        }
    )
    target3 = data3["target"]
    data3 = encoder.transform(data3.drop(columns=["target"]))
    print("Encoding single data:")
    print(data3)
    target3 = encoder.transform_target(target3)
    print("Encoding single target:")
    print(target3)


if __name__ == "__main__":
    main()
