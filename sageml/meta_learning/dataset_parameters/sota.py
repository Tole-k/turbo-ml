from typing import Literal
from .base import CombinedMetaFeatures, MetaFeature
from .statistical import SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from .topological import BallMapperFeatures


def get_sota_meta_features(parameter_type: Literal['statistical', 'topological', 'all'] = 'statistical') -> MetaFeature:
    if parameter_type == 'topological':
        return BallMapperFeatures()
    elif parameter_type == 'statistical':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])
    elif parameter_type == 'all':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures(), BallMapperFeatures()])
    raise ValueError(f'Parameter type {parameter_type} not found.')
