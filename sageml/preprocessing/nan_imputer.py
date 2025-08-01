import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ..base.preprocess import Preprocessor
from .type_inferer import TypeInferer


class NanImputer(Preprocessor):
    numerical_imputer: SimpleImputer
    numerical_target_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    categorical_target_imputer: SimpleImputer
    type_inferer: TypeInferer
    cols_to_drop: pd.Index

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


def main():

    dataset = pd.DataFrame(
        {
            "A": [np.nan, np.nan, np.nan, np.nan],
            "B": [10, np.nan, 30, np.nan],
            "C": ["a", "b", np.nan, "d"],
            "D": [0, np.nan, np.nan, 0],
            "E": [1, np.nan, "c", 1],
            "F": [True, False, np.nan, True],
            "target": [45, np.nan, 69, np.nan],
        }
    )
    data = dataset.drop(columns=["target"])
    target = dataset["target"]
    print("Before NanHandling:")
    print(data)
    print(target)
    print()
    nan_imputer = NanImputer()
    data = nan_imputer.fit_transform(data)
    print("After data NanHandling:")
    print(data)
    print()
    data = pd.DataFrame(
        {
            "A": [np.nan, 2.5, np.nan, 4.5],
            "B": [11, 21, np.nan, 41],
            "C": [np.nan, "b", "c", np.nan],
            "D": [np.nan, 1, np.nan, 1],
            "E": [1, 0.2, "c", np.nan],
            "F": [True, False, True, np.nan],
        }
    )
    data = nan_imputer.transform(data)
    print("NanHandling more data:")
    print(data)
    print()
    target = nan_imputer.fit_transform_target(target)
    print("After target NanHandling:")
    print(target)


if __name__ == "__main__":
    main()
