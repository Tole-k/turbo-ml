from typing import Literal
import numpy as np
import pandas as pd
from .dataset_characteristics import StatisticalParametersExtractor
import warnings
from .base import CombinedMetaFeatures, SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from .topological import BallMapperFeatures, RipserFeatures


def sota_dataset_parameters(dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False, parameter_type: Literal['statistical', 'old', 'topological'] = 'statistical') -> np.ndarray | dict:
    if parameter_type == 'old':
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
    if parameter_type == 'topological':
        return RipserFeatures()(dataset, target_data, as_dict=as_dict)
    return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])(dataset, target_data, as_dict=as_dict)
