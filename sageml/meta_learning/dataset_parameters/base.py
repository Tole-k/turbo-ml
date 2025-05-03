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
