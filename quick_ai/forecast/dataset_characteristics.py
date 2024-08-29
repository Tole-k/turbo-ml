import pandas as pd
from datasets import *
from quick_ai.preprocessing.normalizer import Normalizer
from quick_ai.preprocessing.one_hot_encoder import OneHotEnc
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
                print(type(self.target))
                print(type(self.target.nunique()))
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
            for index, value in counts.items():
                description.at[str(index)] = value
            description.at['nans'] = column.isna().sum()
            return description

    def target_description(self):
        return {'task': self.task, 'description': self.describe_plus_plus(self.target, self.task == 'regression')}

    def feature_description(self):
        return {col: {'type': "continuous" if pd.api.types.is_float_dtype(self.data[col]) else "categorical", 'description': self.describe_plus_plus(self.data[col], pd.api.types.is_float_dtype(self.data[col]))} for col in self.data.columns}


if __name__ == "__main__":
    data, target = get_titanic()
    extractor = StatisticalParametersExtractor(data, target)

    print(extractor.detect_task())
    print(extractor.target_description())
    print(extractor.feature_description())

    normalizer = Normalizer()
    ohe = OneHotEnc()
    nan_imputer = NanImputer()

    data = nan_imputer.fit_transform(data)
    data = normalizer.fit_transform(data)
    data = ohe.fit_transform(data)

    target = nan_imputer.fit_transform_target(target)
    target = normalizer.fit_transform_target(target)
    target = ohe.fit_transform_target(target)

    extractor = StatisticalParametersExtractor(data, target)
    print(extractor.detect_task(one_hot_encoded=True))
    print(extractor.target_description())
    print(extractor.feature_description())
