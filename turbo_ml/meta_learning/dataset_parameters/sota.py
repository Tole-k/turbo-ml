import numpy as np
import pandas as pd
from .dataset_characteristics import StatisticalParametersExtractor


def sota_dataset_parameters(dataset: pd.DataFrame, target_data: pd.Series, as_dict: bool = False) -> np.ndarray | dict:
    extractor = StatisticalParametersExtractor(
        dataset, target=target_data, one_hot_encoded=True)
    description = extractor.describe_dataset()
    print(description)
    dictionary = description.dict()
    if as_dict:
        return dictionary
    variables = list(dictionary.values())[2:]
    # skipping task describing variables (strings)
    return np.array(variables)
