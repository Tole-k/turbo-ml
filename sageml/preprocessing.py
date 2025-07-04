""" Module with different preprocessing techniques """
import numpy as np

import pandas as pd
from pandas.api.types import infer_dtype

from sklearn.preprocessing import (
    OneHotEncoder as sklearnOneHotEncoder,
    LabelEncoder as sklearnLabelEncoder,
    StandardScaler,
    MinMaxScaler,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

from .base import Preprocessor


class CombinedPreprocessor(Preprocessor):
    """ Combines few preprocessing techniques into one """

    def __init__(self, *args) -> None:
        super().__init__()
        self.models: list[Preprocessor] = list(args)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for model in self.models:
            data = model.fit_transform(data)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for model in self.models:
            data = model.transform(data)
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        for model in self.models:
            target = model.fit_transform_target(target)
        return target

    def transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        for model in self.models:
            target = model.transform_target(target)
        return target


class Standardizer(Preprocessor):
    """ Sklearn standardizer """

    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.apply(lambda x: x.astype(
            bool) if x.isin([0, 1]).all() else x)
        normalized_numeric_cols = pd.DataFrame(
            self.scaler.fit_transform(data.select_dtypes(include=[np.number])),
            columns=self.scaler.get_feature_names_out(),
        )
        data = data.apply(
            lambda x: (
                normalized_numeric_cols[x.name]
                if x.name in normalized_numeric_cols.columns
                else x
            )
        )
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.Series:
        if target.isin([0, 1]).all():
            target = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = pd.Series(
                self.target_scaler.fit_transform(np.transpose([target]))[:, 0]
            )
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.apply(lambda x: x.astype(
            bool) if x.isin([0, 1]).all() else x)
        normalized_numeric_cols = pd.DataFrame(
            self.scaler.transform(data.select_dtypes(include=[np.number])),
            columns=self.scaler.get_feature_names_out(),
        )
        return data.apply(
            lambda x: (
                normalized_numeric_cols[x.name]
                if x.name in normalized_numeric_cols.columns
                else x
            )
        )

    def transform_target(self, target: pd.Series) -> pd.Series:
        if target.isin([0, 1]).all():
            target = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = self.target_scaler.transform(np.transpose([target]))
        return target

    def inverse_transform_target(self, target: pd.Series) -> pd.Series:
        if target.isin([0, 1]).all():
            target = target.astype(bool)
        if np.issubdtype(target.dtype, np.number):
            target = self.target_scaler.inverse_transform(
                np.transpose([target]))
        return np.transpose(target)[0]


class Normalizer(Preprocessor):
    """ Sklearn Normalizer """

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


class VarianceSelector(Preprocessor):
    """ Variance Threshold preprocessing """

    def fit_transform(
        self, data: pd.DataFrame, threshold1: float = 0.05, threshold2: float = 0.9
    ) -> pd.DataFrame:
        data = data.apply(lambda x: x.astype(
            bool) if x.isin([0, 1]).all() else x)
        self.variance_threshold = VarianceThreshold(threshold=threshold1)
        cols = data.columns
        numeric_cols = data.select_dtypes(include=[np.number])
        categorical_cols = data.select_dtypes(
            include=["category", object, "string", np.bool_]
        ).map(lambda x: str(x))
        cat_cols_left = [col for col in categorical_cols if (
            categorical_cols[col].value_counts(sort=1)[0] / len(categorical_cols[col])) < threshold2]
        self.cols_to_rem = set(cols) - set(cat_cols_left)
        if not numeric_cols.empty:
            try:
                self.variance_threshold.fit(numeric_cols)
                num_cols_left = self.variance_threshold.get_feature_names_out()
                self.cols_to_rem -= set(num_cols_left)
            except ValueError:
                pass
        data.drop(columns=self.cols_to_rem, inplace=True)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.drop(columns=self.cols_to_rem)
        return data

    def fit_transform_target(self, target):
        return target

    def transform_target(self, target):
        return target


class NanImputer(Preprocessor):
    """ Sklearn SimpleImputer """

    def __init__(self) -> None:
        super().__init__()
        self.numerical_imputer = SimpleImputer(strategy="mean")
        self.numerical_target_imputer = SimpleImputer(strategy="mean")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.categorical_target_imputer = SimpleImputer(strategy="most_frequent")
        self.type_inferer = TypeInferer()

    def fit_transform(self, data: pd.DataFrame, nan_threshold: float = 1.0) -> pd.DataFrame:
        nans = data.isna().sum() / len(data)
        self.cols_to_drop = nans[nans >= nan_threshold]
        data.drop(columns=self.cols_to_drop.index, inplace=True)
        og_cols = data.columns
        x_dtypes = self.type_inferer.infer(data)
        numerical_cols = x_dtypes[x_dtypes.isin(["floating", "integer", "mixed-integer-float"])].index
        categorical_cols = x_dtypes[x_dtypes.isin(["string", "mixed-integer", "categorical", "mixed"])].index
        boolean_cols = x_dtypes[x_dtypes.isin(["boolean"])].index
        numerical_frame = data[numerical_cols]
        categorical_frame = data[categorical_cols]
        if not boolean_cols.empty:
            boolean_frame = data[boolean_cols].map(lambda x: x if pd.isna(x) else str(x))
            categorical_frame = pd.concat([categorical_frame, boolean_frame], axis=1)
        if not numerical_cols.empty:
            numerical_frame = pd.DataFrame(self.numerical_imputer.fit_transform(numerical_frame),
                                           columns=numerical_cols).astype(data[numerical_cols].dtypes.to_dict())
        if not categorical_cols.empty:
            categorical_frame = pd.DataFrame(self.categorical_imputer.fit_transform(categorical_frame.map(lambda x: x if pd.isna(x) else str(x))),
                                             columns=categorical_frame.columns).astype(str)
        if not boolean_cols.empty:
            categorical_frame[boolean_cols] = categorical_frame[boolean_cols].map(lambda x: True if x == "True" else False)

        data = pd.concat([numerical_frame if not numerical_cols.empty else None,
                          categorical_frame if not categorical_cols.empty else None], axis=1)
        return data[og_cols]

    def fit_transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.infer_target(target)
        if y_dtype in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.numerical_target_imputer.fit_transform(target.to_numpy().reshape(-1, 1))[:, 0]).astype(float)
        elif y_dtype in ["integer"]:
            target = pd.Series(self.categorical_target_imputer.fit_transform(target.to_numpy().reshape(-1, 1))[:, 0]).astype(int)
        elif y_dtype in ["string", "mixed-integer", "categorical", "mixed"]:
            target = pd.Series(self.categorical_target_imputer.fit_transform(target.map(
                lambda x: x if pd.isna(x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).astype(str)
        elif y_dtype in ["boolean"]:
            target = pd.Series(self.categorical_target_imputer.fit_transform(target.map(lambda x: x if pd.isna(
                x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).map(lambda x: True if x == "True" else False)
        else:
            target = pd.Series(self.categorical_target_imputer.fit_transform(target.map(
                lambda x: x if pd.isna(x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).astype(str)
        return target

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(columns=self.cols_to_drop.index, inplace=True)
        og_cols = data.columns
        x_dtypes = self.type_inferer.recall()
        numerical_cols = x_dtypes[x_dtypes.isin(["floating", "integer", "mixed-integer-float"])].index
        categorical_cols = x_dtypes[x_dtypes.isin(["string", "mixed-integer", "categorical", "mixed"])].index
        boolean_cols = x_dtypes[x_dtypes.isin(["boolean"])].index
        numerical_frame = data[numerical_cols]
        categorical_frame = data[categorical_cols]
        if not boolean_cols.empty:
            boolean_frame = data[boolean_cols].map(lambda x: x if pd.isna(x) else str(x))
            categorical_frame = pd.concat([categorical_frame, boolean_frame], axis=1)
        if not numerical_cols.empty:
            numerical_frame = pd.DataFrame(self.numerical_imputer.transform(numerical_frame),
                                           columns=numerical_cols).astype(data[numerical_cols].dtypes.to_dict())
        if not categorical_cols.empty:
            categorical_frame = pd.DataFrame(self.categorical_imputer.transform(categorical_frame.map(lambda x: x if pd.isna(x) else str(x))),
                                             columns=categorical_frame.columns).astype(str)
        if not boolean_cols.empty:
            categorical_frame[boolean_cols] = categorical_frame[boolean_cols].map(lambda x: True if x == "True" else False)
        data = pd.concat([numerical_frame if not numerical_cols.empty else None,
                          categorical_frame if not categorical_cols.empty else None], axis=1)
        return data[og_cols]

    def transform_target(self, target: pd.Series) -> pd.Series:
        y_dtype = self.type_inferer.recall_target()
        if y_dtype in ["floating", "mixed-integer-float"]:
            target = pd.Series(self.numerical_target_imputer.transform(target.to_numpy().reshape(-1, 1))[:, 0]).astype(float)
        elif y_dtype in ["integer"]:
            target = pd.Series(self.categorical_target_imputer.transform(target.to_numpy().reshape(-1, 1))[:, 0]).astype(int)
        elif y_dtype in ["string", "mixed-integer", "categorical", "mixed"]:
            target = pd.Series(self.categorical_target_imputer.transform(target.map(
                lambda x: x if pd.isna(x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).astype(str)
        elif y_dtype in ["boolean"]:
            target = pd.Series(self.categorical_target_imputer.transform(target.map(lambda x: x if pd.isna(
                x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).map(lambda x: True if x == "True" else False)
        else:
            target = pd.Series(self.categorical_target_imputer.transform(target.map(
                lambda x: x if pd.isna(x) else str(x)).to_numpy().reshape(-1, 1))[:, 0]).astype(str)
        return target


class TypeInferer:
    def __init__(self) -> None:
        super().__init__()

    def infer(self, data: pd.DataFrame) -> pd.Series:
        self.x_dtypes = data.apply(lambda x: infer_dtype(x, skipna=True))
        return self.x_dtypes

    def infer_target(self, target: pd.Series) -> str:
        self.y_dtype = infer_dtype(target, skipna=True)
        return self.y_dtype

    def recall(self) -> pd.Series:
        return self.x_dtypes

    def recall_target(self) -> str:
        return self.y_dtype


class Encoder(Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = sklearnOneHotEncoder(drop="if_binary", handle_unknown="ignore")
        self.target_encoder = sklearnLabelEncoder()
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


def sota_preprocessor() -> Preprocessor:
    """
    Preprocess data using state-of-the-art techniques implemented in libraries.

    Args:
        data (pd.DataFrame): Data to preprocess.
        target (str): Target column.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    return CombinedPreprocessor(NanImputer(), Encoder(), Normalizer())
