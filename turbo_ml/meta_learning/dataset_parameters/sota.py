from typing import Literal
from .base import CombinedMetaFeatures, MetaFeature
from .statistical import SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from .topological import BallMapperFeatures


def get_sota_meta_features(parameter_type: Literal['statistical', 'topological'] = 'statistical') -> MetaFeature:
    if parameter_type == 'topological':
        return BallMapperFeatures()
    elif parameter_type == 'statistical':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])
    raise ValueError(f'Parameter type {parameter_type} not found.')
