"""
This modules implements Meta Feautures for dataset parameters extraction.

Module contains:
    - MetaFeature: Abstract class for meta features
    - CombinedMetaFeatures: Class for combining multiple meta features
    - SimpleMetaFeatures: Class for simple meta features
    - StatisticalMetaFeatures: Class for statistical meta features
    - PCAMetaFeatures: Class for PCA meta features
    - sota_dataset_parameters: Function for extracting dataset parameters

SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures are the implementations of meta features mentioned in the paper: Initializing bayesian hyperparameter optimization via meta-learning

@inproceedings{10.5555/2887007.2887164,
author = {Feurer, Matthias and Springenberg, Jost Tobias and Hutter, Frank},
title = {Initializing bayesian hyperparameter optimization via meta-learning},
year = {2015},
isbn = {0262511290},
publisher = {AAAI Press},
abstract = {Model selection and hyperparameter optimization is crucial in applying machine learning to a novel dataset. Recently, a sub-community of machine learning has focused on solving this problem with Sequential Model-based Bayesian Optimization (SMBO), demonstrating substantial successes in many applications. However, for computationally expensive algorithms the overhead of hyperparameter optimization can still be prohibitive. In this paper we mimic a strategy human domain experts use: speed up optimization by starting from promising configurations that performed well on similar datasets. The resulting initialization technique integrates naturally into the generic SMBO framework and can be trivially applied to any SMBO method. To validate our approach, we perform extensive experiments with two established SMBO frameworks (Spearmint and SMAC) with complementary strengths; optimizing two machine learning frameworks on 57 datasets. Our initialization procedure yields mild improvements for low-dimensional hyperparameter optimization and substantially improves the state of the art for the more complex combined algorithm selection and hyperparameter optimization problem.},
booktitle = {Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence},
pages = {1128–1135},
numpages = {8},
location = {Austin, Texas},
series = {AAAI'15}
}

"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from .dataset_characteristics import StatisticalParametersExtractor
from sklearn.decomposition import PCA
import warnings


class MetaFeature(ABC):
    @abstractmethod
    def __call__(self, dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
        pass


class CombinedMetaFeatures(MetaFeature):
    def __init__(self, meta_features: list[MetaFeature]):
        self.meta_features = meta_features

    def __call__(self, dataset, target_data, as_dict=False):
        results = []

        for meta_feature in self.meta_features:
            results.append(meta_feature(dataset, target_data, as_dict=as_dict))
        if as_dict:
            results_dict = {}
            for i in results:
                results_dict.update(i)
            return results_dict
        return np.concatenate(results)


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


class StatisticalMetaFeatures(MetaFeature):
    def __call__(self, dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
        categorical = pd.Series([dataset[i].astype('category').nunique()
                                 for i in dataset if dataset[i].dtype == 'O'] or [0, 0])
        kurtosis = dataset.kurtosis(numeric_only=True)
        skewness = dataset.skew(numeric_only=True)
        results = {
            'categorical_min': categorical.min(),
            'categorical_max': categorical.max(),
            'categorical_mean': categorical.mean(),
            'categorical_std': categorical.std(),
            'categorical_total': categorical.sum(),
            'kurtosis_min': kurtosis.min() if len(kurtosis) > 0 else 0,
            'kurtoisis_max': kurtosis.max() if len(kurtosis) > 0 else 0,
            'kurtoisis_mean': kurtosis.mean() if len(kurtosis) > 0 else 0,
            'kurtoisis_std': kurtosis.std() if len(kurtosis) > 0 else 0,
            'skewness_min': skewness.min() if len(skewness) > 0 else 0,
            'skewness_max': skewness.max() if len(skewness) > 0 else 0,
            'skewness_mean': skewness.mean() if len(skewness) > 0 else 0,
            'skewness_std': skewness.std() if len(skewness) > 0 else 0
        }
        if as_dict:
            return results
        return np.array(list(results.values()))


class PCAMetaFeatures(MetaFeature):
    def __call__(self, dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
        pca = PCA()
        dataset = dataset.select_dtypes(include=[np.number])
        if dataset.shape[1] == 0:
            if as_dict:
                return {
                    'pca_95_index': 0,
                    'pca_skewness': 0,
                    'pca_kurtosis': 0
                }
            return np.zeros(3)
        new_set = pca.fit_transform(dataset)
        pca_95_index = np.argmax(
            np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        pc1_scores = new_set[:, 0]
        pca_skewness = pd.Series(pc1_scores).skew()
        pca_kurtosis = pd.Series(pc1_scores).kurtosis()
        results = {
            'pca_95_index': pca_95_index,
            'pca_skewness': pca_skewness,
            'pca_kurtosis': pca_kurtosis
        }
        if as_dict:
            return results
        return np.array(list(results.values()))


def sota_dataset_parameters(dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False, old: bool = False) -> np.ndarray | dict:
    if old:
        warnings.warn(
            "The 'old' parameter is deprecated and will be removed in a future version.", DeprecationWarning)
        extractor = StatisticalParametersExtractor(dataset, target=target_data)
        description = extractor.describe_dataset()
        dictionary = description.dict()
        if as_dict:
            return dictionary
        variables = list(dictionary.values())[2:]
        # skipping task describing variables (strings)
        return np.array(variables)
    return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])(dataset, target_data, as_dict=as_dict)
