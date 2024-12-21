from typing import Literal
import numpy as np
import pandas as pd
from .base import CombinedMetaFeatures
from .statistical import SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from .topological import BallMapperFeatures


def sota_dataset_parameters(dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False, parameter_type: Literal['statistical', 'topological'] = 'statistical') -> np.ndarray | dict:
    if parameter_type == 'topological':
        return BallMapperFeatures()(dataset, target_data, as_dict=as_dict)
    elif parameter_type == 'statistical':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])(dataset, target_data, as_dict=as_dict)
    raise ValueError(f'Parameter type {parameter_type} not found.')
