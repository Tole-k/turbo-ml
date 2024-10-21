from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from .dataset_characteristics import StatisticalParametersExtractor


class MetaFeature(ABC):
    @abstractmethod
    def __call__(self, dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
        pass


class SimpleMetaFeatures(MetaFeature):
    def __call__(self, dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
        types = dataset.dtypes
        num_features = len(dataset.dtypes)
        log_num_cols = np.log2(num_features)
        num_patters = len(dataset)
        log_num_rows = np.log2(num_patters)
        num_categorical = sum(1 for i in types if i == 'O')
        num_numerical = num_features - num_categorical
        assert num_features > 0 and num_patters > 0
        if num_categorical == 0:
            ratio_num_cat = 1
        else:
            ratio_num_cat = num_numerical / num_categorical
        if num_numerical == 0:
            ratio_cat_num = 1
        else:
            ratio_cat_num = num_categorical / num_numerical
        # Missing values
        missing_values_frame = dataset.isna()
        number_of_features_with_missing_values = missing_values_frame.any().sum()
        perc_of_features_with_missing_values = number_of_features_with_missing_values / num_features
        number_of_patterns_with_missing_values = missing_values_frame.any(
            axis=1).sum()
        perc_of_patterns_with_missing_values = number_of_patterns_with_missing_values / num_patters
        number_of_missing_values = missing_values_frame.sum().sum()
        perc_of_missing_values = number_of_missing_values / num_features / num_patters
        # Classes
        target_data = target_data.astype('category')
        num_classes = target_data.nunique()
        class_distribution = target_data.value_counts(normalize=True)
        class_prob_min = class_distribution.min()
        class_prob_max = class_distribution.max()
        class_prob_mean = class_distribution.mean()
        class_prob_std = class_distribution.std()
        class_entropy = -np.sum(class_distribution*np.log2(class_distribution))
        # Dimensionality
        dataset_dimensionality = num_features
        dataset_dimensionality_log = log_num_cols
        inverse_dataset_dimensionality = 1 / num_features
        inverse_dataset_dimensionality_log = 1 / log_num_cols
        results = {
            'num_patterns': num_patters,
            'log_patterns': log_num_rows,
            'num_classes': num_classes,
            'num_features': num_features,
            'log_features': log_num_cols,
            'num_patterns_with_missing_values': number_of_patterns_with_missing_values,
            'perc_patterns_with_missing_values': perc_of_patterns_with_missing_values,
            'num_features_with_missing_values': number_of_features_with_missing_values,
            'perc_features_with_missing_values': perc_of_features_with_missing_values,
            'num_missing_values': number_of_missing_values,
            'perc_missing_values': perc_of_missing_values,
            'num_numerical': num_numerical,
            'num_categorical': num_categorical,
            'ratio_num_cat': ratio_num_cat,
            'ratio_cat_num': ratio_cat_num,
            'dataset_dimensionality': dataset_dimensionality,
            'dataset_dimensionality_log': dataset_dimensionality_log,
            'inverse_dataset_dimensionality': inverse_dataset_dimensionality,
            'inverse_dataset_dimensionality_log': inverse_dataset_dimensionality_log,
            'class_prob_min': class_prob_min,
            'class_prob_max': class_prob_max,
            'class_prob_mean': class_prob_mean,
            'class_prob_std': class_prob_std,
            'class_entropy': class_entropy
        }
        if as_dict:
            return results
        return np.array(list(results.values()))


def sota_dataset_parameters(dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
    extractor = StatisticalParametersExtractor(dataset, target=target_data)
    description = extractor.describe_dataset()
    print(description)
    dictionary = description.dict()
    if as_dict:
        return dictionary
    variables = list(dictionary.values())[2:]
    # skipping task describing variables (strings)
    return np.array(variables)
