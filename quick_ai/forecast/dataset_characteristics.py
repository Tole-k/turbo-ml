import numpy as np
import pandas as pd
from datasets import *


class TaskDescription:
    target_features: int = None
    task: str = None

    def __str__(self):
        return f"Task: {self.task}\nTarget Features: {self.target_features}"


class TargetDescription:
    task: str = None
    num_classes: int = None
    biggest_class: str = None
    biggest_class_freq: int = None
    smallest_class: str = None
    smallest_class_freq: int = None
    target_mean: float = None
    target_std: float = None
    target_min: float = None
    target_25: float = None
    target_50: float = None
    target_75: float = None
    target_max: float = None
    target_median: float = None
    target_var: float = None
    target_skew: float = None
    target_nans: int = None

    def __str__(self):
        if self.task == 'regression':
            return f"Mean: {self.target_mean}\nStd: {self.target_std}\nMin: {self.target_min}\n25%: {self.target_25}\n50%: {self.target_50}\n75%: {self.target_75}\nMax: {self.target_max}\nMedian: {self.target_median}\nVariance: {self.target_var}\nSkew: {self.target_skew}\nNans: {self.target_nans}"
        else:
            return f"Number of Classes: {self.num_classes}\nBiggest Class: {self.biggest_class}\nBiggest Class Frequency: {self.biggest_class_freq}\nSmallest Class: {self.smallest_class}\nSmallest Class Frequency: {self.smallest_class_freq}\nNans: {self.target_nans}"


class DataSetDescription:
    num_columns: int = None
    num_rows: int = None
    number_of_highly_correlated_features: int = None
    highest_correlation: float = None
    number_of_lowly_correlated_features: int = None
    lowest_correlation: float = None
    highest_eigenvalue: float = None
    lowest_eigenvalue: float = None

    def __str__(self) -> str:
        return f"Number of Columns: {self.num_columns}\nNumber of Rows: {self.num_rows}\nNumber of Highly Correlated Features: {self.number_of_highly_correlated_features}\nHighest Correlation: {self.highest_correlation}\nNumber of Lowly Correlated Features: {self.number_of_lowly_correlated_features}\nLowest Correlation: {self.lowest_correlation}\nHighest Eigenvalue: {self.highest_eigenvalue}\nLowest Eigenvalue: {self.lowest_eigenvalue}"


class StatisticalParametersExtractor:
    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target
        self.task_description = TaskDescription()
        self.target_description = TargetDescription()
        self.dataset_description = DataSetDescription()
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
        self.task_description.target_features = num_target_features
        self.task_description.task = self.task
        return self.task_description

    def describe_target(self):
        self.target_description.task = self.task
        if self.task == 'regression':
            description = self.target.describe()
            self.target_description.target_mean = description['mean']
            self.target_description.target_std = description['std']
            self.target_description.target_min = description['min']
            self.target_description.target_25 = description['25%']
            self.target_description.target_50 = description['50%']
            self.target_description.target_75 = description['75%']
            self.target_description.target_max = description['max']
            self.target_description.target_median = self.target.median()
            self.target_description.target_var = self.target.var()
            self.target_description.target_skew = self.target.skew()
        else:
            description = self.target.astype('object').describe()
            self.target_description.num_classes = description['unique']
            self.target_description.biggest_class = description['top']
            self.target_description.biggest_class_freq = description['freq']
            self.target_description.smallest_class = self.target.value_counts().idxmin()
            self.target_description.smallest_class_freq = self.target.value_counts().min()
        self.target_description.target_nans = self.target.isna().sum()
        return self.target_description

    def describe_dataset(self):
        self.dataset_description.num_columns = len(self.data.columns)
        self.dataset_description.num_rows = len(self.data)
        continous_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[continous_cols].corr()
        removed_diag = corr_matrix.mask(
            np.eye(len(corr_matrix), dtype=bool)).abs()
        self.dataset_description.number_of_highly_correlated_features = int((
            removed_diag > 0.8).sum().sum())
        self.dataset_description.highest_correlation = float(
            removed_diag.max().max())
        self.dataset_description.number_of_lowly_correlated_features = int((
            removed_diag < 0.2).sum().sum())
        self.dataset_description.lowest_correlation = float(
            removed_diag.min().min())
        self.dataset_description.highest_eigenvalue = float(
            np.linalg.eigvals(corr_matrix).max())
        self.dataset_description.lowest_eigenvalue = float(
            np.linalg.eigvals(corr_matrix).min())
        return self.dataset_description


if __name__ == "__main__":
    data, target = get_titanic()
    extractor = StatisticalParametersExtractor(data, target)

    print(extractor.detect_task())
    print(extractor.describe_target())
    print(extractor.describe_dataset())
