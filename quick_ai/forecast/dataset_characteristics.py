from typing import Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class TaskDescription:
    target_features: int
    task: str

    def __str__(self):
        return f"Task: {self.task}\nTarget Features: {self.target_features}"


@dataclass
class TargetDescription:
    task: str
    target_nans: int


@dataclass
class TargetDescriptionClassification(TargetDescription):
    num_classes: int
    biggest_class: str
    biggest_class_freq: int
    smallest_class: str
    smallest_class_freq: int

    def __str__(self):
        return f"Number of Classes: {self.num_classes}\nBiggest Class: {self.biggest_class}\nBiggest Class Frequency: {self.biggest_class_freq}\nSmallest Class: {self.smallest_class}\nSmallest Class Frequency: {self.smallest_class_freq}\nNans: {self.target_nans}"


@dataclass
class TargetDescriptionRegression(TargetDescription):
    target_min: float
    target_25: float
    target_50: float
    target_75: float
    target_max: float
    target_mean: float
    target_std: float
    target_median: float
    target_var: float
    target_skew: float

    def __str__(self):
        return f"Mean: {self.target_mean}\nStd: {self.target_std}\nMin: {self.target_min}\n25%: {self.target_25}\n50%: {self.target_50}\n75%: {self.target_75}\nMax: {self.target_max}\nMedian: {self.target_median}\nVariance: {self.target_var}\nSkew: {self.target_skew}\nNans: {self.target_nans}"


@dataclass
class DataSetDescription:
    num_columns: int
    num_rows: int
    number_of_highly_correlated_features: int
    highest_correlation: float
    number_of_lowly_correlated_features: int
    lowest_correlation: float
    highest_eigenvalue: float
    lowest_eigenvalue: float

    def __str__(self) -> str:
        return f"Number of Columns: {self.num_columns}\nNumber of Rows: {self.num_rows}\nNumber of Highly Correlated Features: {self.number_of_highly_correlated_features}\nHighest Correlation: {self.highest_correlation}\nNumber of Lowly Correlated Features: {self.number_of_lowly_correlated_features}\nLowest Correlation: {self.lowest_correlation}\nHighest Eigenvalue: {self.highest_eigenvalue}\nLowest Eigenvalue: {self.lowest_eigenvalue}"


class StatisticalParametersExtractor:
    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data = data
        self.target = target
        self.task_description: Optional[TaskDescription] = None
        self.target_description: Optional[TargetDescription] = None
        self.dataset_description: Optional[DataSetDescription] = None
        self.task: Optional[str] = None

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
        self.task_description = TaskDescription(
            target_features=num_target_features, task=self.task)
        return self.task_description

    def describe_target(self) -> TargetDescriptionClassification | TargetDescriptionRegression:
        target_nans = self.target.isna().sum()
        if self.task == 'regression':
            description = self.target.describe()
            target_mean = description['mean']
            target_std = description['std']
            target_min = description['min']
            target_25 = description['25%']
            target_50 = description['50%']
            target_75 = description['75%']
            target_max = description['max']
            target_median = self.target.median()
            target_var = self.target.var()
            target_skew = self.target.skew()
            self.target_description = TargetDescriptionRegression(self.task, target_mean=target_mean, target_std=target_std, target_min=target_min,
                                                                  target_25=target_25, target_50=target_50, target_75=target_75, target_max=target_max, target_median=target_median, target_var=target_var, target_skew=target_skew, target_nans=target_nans)
        else:
            description = self.target.astype('object').describe()
            num_classes = description['unique']
            biggest_class = description['top']
            biggest_class_freq = description['freq']
            smallest_class = self.target.value_counts().idxmin()
            smallest_class_freq = self.target.value_counts().min()
            self.target_description = TargetDescriptionClassification(
                self.task, num_classes=num_classes, biggest_class=biggest_class, biggest_class_freq=biggest_class_freq, smallest_class=smallest_class, smallest_class_freq=smallest_class_freq, target_nans=target_nans)
        return self.target_description

    def describe_dataset(self) -> DataSetDescription:
        continous_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[continous_cols].corr()
        removed_diag = corr_matrix.mask(
            np.eye(len(corr_matrix), dtype=bool)).abs()

        self.dataset_description = DataSetDescription(
            num_columns=len(self.data.columns),
            num_rows=len(self.data),
            number_of_highly_correlated_features=int(
                (removed_diag > 0.8).sum().sum()),
            highest_correlation=float(removed_diag.max().max()),
            number_of_lowly_correlated_features=int(
                (removed_diag < 0.2).sum().sum()),
            lowest_correlation=float(removed_diag.min().min()),
            highest_eigenvalue=float(np.linalg.eigvals(corr_matrix).max()),
            lowest_eigenvalue=float(np.linalg.eigvals(corr_matrix).min())
        )
        return self.dataset_description


if __name__ == "__main__":
    from datasets import *
    data, target = get_titanic()
    extractor = StatisticalParametersExtractor(data, target)

    print(extractor.detect_task())
    print(extractor.describe_target())
    print(extractor.describe_dataset())
