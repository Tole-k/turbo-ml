from typing import Literal
from .base import CombinedMetaFeatures, MetaFeature
from .statistical import SimpleMetaFeatures, StatisticalMetaFeatures, PCAMetaFeatures
from .topological import BallMapperFeatures


def sota_meta_features(parameter_type: Literal['statistical', 'topological', 'all'] = 'all') -> MetaFeature:
    """Returns currently the best meta feature extractor

    Args:
        parameter_type (Literal[&#39;statistical&#39;, &#39;topological&#39;, &#39;all&#39;], optional): Mode of extracted meta features. 
        Defaults to 'all'.

    Raises:
        ValueError: If mode is incorrect.

    Returns:
        MetaFeature: Meta feature extractor.
    """
    if parameter_type == 'topological':
        return BallMapperFeatures()
    elif parameter_type == 'statistical':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures()])
    elif parameter_type == 'all':
        return CombinedMetaFeatures([SimpleMetaFeatures(), StatisticalMetaFeatures(), PCAMetaFeatures(), BallMapperFeatures()])
    raise ValueError(f'Parameter type {parameter_type} not found.')
