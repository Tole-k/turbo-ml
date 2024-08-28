import pandas as pd
from datasets import *
import numpy as np


class StatisticalParametersExtractor:
    def __init__(self, data: pd.DataFrame, target: pd.DataFrame):
        self.data = data
        self.target = target

    def detect_task(self):
        num_target_features = 1 if len(
            self.target.shape) == 1 else self.target.shape[1]
        if num_target_features == 1:
            if pd.api.types.is_float_dtype(self.target):
                task = 'regression'
            else:
                print(self.target.nunique())
                if self.target.nunique() == 2:
                    task = 'binary_classification'
                else:
                    task = 'multiclass_classification'
        else:
            if all(map(pd.api.types.is_float_dtype, self.target.dtypes)):
                task = 'regression'
            else:
                print(self.target.nunique())
                if all(self.target.nunique() == 2):
                    task = 'binary_classification'
                else:
                    task = 'multiclass_classification'
        return {'target_features': num_target_features, 'task': task}


if __name__ == "__main__":
    data, target = get_heart_disease()
    print(data.head())
    print(target.head())
    extractor = StatisticalParametersExtractor(data, target)
    print(extractor.detect_task())
