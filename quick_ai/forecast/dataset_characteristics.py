import numpy as np
import pandas as pd
from datasets import *
from quick_ai.preprocessing.normalizer import Normalizer
from quick_ai.preprocessing.one_hot_encoder import OneHotEncoder
from quick_ai.preprocessing.nan_imputer import NanImputer


class StatisticalParametersExtractor:
    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target
        self.task = None

    def detect_task(self, one_hot_encoded: bool = False):
        num_target_features = 1 if len(
            self.target.shape) == 1 else self.target.shape[1]
        if num_target_features == 1:
            if pd.api.types.is_float_dtype(self.target):
                self.task = 'regression'
            else:
                if self.target.nunique() == 2:
                    self.task = 'binary_classification'
                else:
                    self.task = 'multiclass_classification'
        else:
            if all(map(pd.api.types.is_float_dtype, self.target.dtypes)):
                self.task = 'regression'
            else:
                if not one_hot_encoded:
                    self.task = 'multilabel_classification'
                else:
                    if any(self.target.sum(axis=1)) > 1:
                        self.task = 'multilabel_classification'
                    else:
                        self.task = 'multiclass_classification'
        return {'target_features': num_target_features, 'task': self.task}

    def describe_plus_plus(self, column: pd.Series, continuous: bool = True):
        if continuous:
            description = column.describe()
            description.at['median'] = column.median()
            description.at['var'] = column.var()
            description.at['skew'] = column.skew()
            description.at['nans'] = column.isna().sum()
            return description
        else:
            description = column.astype('object').describe()
            counts = column.value_counts().to_dict()
            description.at['biggest_class'] = description.at['top']
            del description['top']
            description.at['biggest_class_freq'] = description.at['freq']
            del description['freq']
            description.at['smallest_class'] = column.value_counts().idxmin()
            description.at['smallest_class_freq'] = column.value_counts().min()
            for index, value in counts.items():
                description.at[str(index)] = value
            description.at['nans'] = column.isna().sum()
            return description

    def target_description(self):
        return {'task': self.task, 'description': self.describe_plus_plus(self.target, self.task == 'regression')}

    def feature_description(self):
        params = {col: {'type': "continuous" if pd.api.types.is_float_dtype(self.data[col]) else "categorical", 'description': self.describe_plus_plus(
            self.data[col], pd.api.types.is_float_dtype(self.data[col]))} for col in self.data.columns} | {'num_columns': len(self.data.columns), 'num_rows': len(self.data)}
        continous_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[continous_cols].corr()
        removed_diag = corr_matrix.mask(
            np.eye(len(corr_matrix), dtype=bool)).abs()
        params['number_of_highly_correlated_features'] = int((
            removed_diag > 0.8).sum().sum())
        params['highest_correlation'] = float(removed_diag.max().max())
        params['number_of_lowly_correlated_features'] = int((
            removed_diag < 0.2).sum().sum())
        params['lowest_correlation'] = float(removed_diag.min().min())
        params['highest_eigenvalue'] = float(
            np.linalg.eigvals(corr_matrix).max())
        params['lowest_eigenvalue'] = float(
            np.linalg.eigvals(corr_matrix).min())
        return params


if __name__ == "__main__":
    data, target = get_titanic()
    extractor = StatisticalParametersExtractor(data, target)

    print(extractor.detect_task())
    print(extractor.target_description())
    print(extractor.feature_description())
