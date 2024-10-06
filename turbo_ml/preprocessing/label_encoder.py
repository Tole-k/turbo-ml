from typing import Iterable, Tuple
import numpy as np
import pandas as pd
from ..base.preprocess import Preprocessor
from sklearn import preprocessing


class LabelEncoder(Preprocessor):
    def __init__(self):
        super().__init__()
        self.encoder = preprocessing.LabelEncoder()
        self.target_encoder = preprocessing.LabelEncoder()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.column_order = data.columns
        categorical_cols = data.select_dtypes(
            include=["category", object, "string"]
        )
        self.categorical_cols = categorical_cols.columns
        encoded_data = pd.DataFrame(
            self.encoder.fit_transform(categorical_cols).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        self.encoded_cols = encoded_data.columns
        result = data.drop(columns=categorical_cols.columns)
        result[encoded_data.columns] = encoded_data
        return result

    def fit_transform_target(self, target: pd.Series) -> pd.Series:
        if not pd.api.types.is_float_dtype(target):
            target = pd.Series(self.target_encoder.fit_transform(
                target), name=target.name)
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = data[self.categorical_cols]
        encoded_data = pd.DataFrame(
            self.encoder.transform(categorical_cols).toarray(),
            columns=self.encoder.get_feature_names_out(),
        )
        data.drop(columns=categorical_cols.columns, inplace=True)
        data[encoded_data.columns] = encoded_data
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

    def transform_target(self, target: pd.Series) -> pd.Series:
        if not pd.api.types.is_float_dtype(target):
            target = pd.Series(self.target_encoder.fit_transform(
                target), name=target.name)
        return target

    def inverse_transform_target(self, target: pd.Series) -> pd.Series:
        return self.target_encoder.inverse_transform(target)
