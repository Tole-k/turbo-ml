from typing import TYPE_CHECKING
from .encoder import Encoder
from .nan_imputer import NanImputer
from .combined import CombinedPreprocessor
from .normalizer import Normalizer

# if TYPE_CHECKING:
from ..base import Preprocessor
import pandas as pd


def sota_preprocessor() -> Preprocessor:
    """
    Preprocess data using state-of-the-art techniques implemented in libraries.

    Args:
        data (pd.DataFrame): Data to preprocess.
        target (str): Target column.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    return CombinedPreprocessor(NanImputer(), Encoder(), Normalizer())
