import pandas as pd
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import (
    OneHotEncoder as sklearnOneHotEncoder,
    LabelEncoder as sklearnLabelEncoder,
)


class Encoder(Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = sklearnOneHotEncoder(drop="if_binary", handle_unknown="ignore")
        self.target_encoder = sklearnLabelEncoder()

    def fit_transform(self, data: pd.DataFrame, unique_values_cap=20) -> pd.DataFrame:
        self.column_order = data.columns
        categorical_cols = data.select_dtypes(
            include=["category", object, "string"]
        ).map(lambda x: str(x))
        self.cols_to_drop = categorical_cols.columns
        num_cols = data.drop(columns=self.cols_to_drop)
        categorical_cols.drop(
            columns=categorical_cols.loc[
                :, categorical_cols.nunique() > unique_values_cap
            ].columns,
            inplace=True,
        )
        self.categorical_cols = categorical_cols.columns
        encoded_data = pd.DataFrame(
            self.encoder.fit_transform(categorical_cols).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        self.encoded_cols = encoded_data.columns
        data = pd.concat([num_cols, encoded_data], axis=1)
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        if not pd.api.types.is_float_dtype(target):
            target = pd.Series(
                self.target_encoder.fit_transform(target), name=target.name
            )
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = data[self.categorical_cols].map(lambda x: str(x))
        encoded_data = pd.DataFrame(
            self.encoder.transform(categorical_cols).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        num_cols = data.drop(columns=self.cols_to_drop)
        data = pd.concat([num_cols, encoded_data], axis=1)
        return data

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        to_decode = data[self.encoded_cols]
        encoded_data = pd.DataFrame(
            self.encoder.inverse_transform(to_decode), columns=self.categorical_cols
        )
        data.drop(columns=to_decode.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
        data = data[self.column_order]
        return data

    def transform_target(self, target: pd.Series) -> pd.Series | pd.DataFrame:
        if not pd.api.types.is_float_dtype(target):
            target = pd.Series(
                self.target_encoder.fit_transform(target), name=target.name
            )
        return target

    def inverse_transform_target(self, target: pd.Series) -> pd.Series:
        return self.target_encoder.inverse_transform(target)


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
    ohe = Encoder()
    result = ohe.fit_transform(data)
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
    data2 = ohe.transform(data2)
    print("Encoding more data:")
    print(data2)
    print()
    data = ohe.inverse_transform(result)
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
