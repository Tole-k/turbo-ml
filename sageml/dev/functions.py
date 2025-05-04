""" Functions for easier CLI operations """
import os
from typing import Callable
import pandas as pd
from tqdm import tqdm
from pydataset import data as pydata
from sageml.preprocessing import sota_preprocessor
from sageml.meta_learning.dataset_parameters import sota_meta_features
from sageml.workflow.utils import list_dataset_files, read_data_file
from sageml.workflow.algorithms_evaluations import evaluate_algorithm


def _extract(dataset: pd.DataFrame, dataset_name: str) -> pd.Series:
    preprocessor = sota_preprocessor()
    extractor = sota_meta_features()
    train_x = dataset.drop(dataset.columns[-1], axis=1)
    train_y = dataset[dataset.columns[-1]]

    train_x = preprocessor.fit_transform(train_x)
    train_y = preprocessor.fit_transform_target(train_y)

    parameters = extractor(train_x, train_y, as_dict=True)
    parameters["name"] = dataset_name
    return pd.Series(parameters)


def create_datasets(path: str, max_datasets: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs all datasets found in path on all algorithms

    Args:
        path (str): path to directory with datasets
        max_datasets (int): maximum number of datasets to process.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: combined dataset scores and parameters
    """
    dataset_paths = list_dataset_files(path)[:max_datasets]
    score_dataset = pd.DataFrame()
    param_dataset = pd.DataFrame()
    for path in tqdm(dataset_paths):
        dataset = read_data_file(path)
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        score_dataset = pd.concat([score_dataset, evaluate_algorithm(dataset, dataset_name).to_frame().T], ignore_index=True)
        param_dataset = pd.concat([param_dataset, _extract(dataset, dataset_name).to_frame().T], ignore_index=True)

    return score_dataset, param_dataset


def create_pydataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs all datasets found in path on all algorithms

    Args:
        path (str): path to directory with datasets
        max_datasets (int): maximum number of datasets to process.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: combined dataset scores and parameters
    """
    score_dataset = pd.DataFrame()
    param_dataset = pd.DataFrame()
    for (dataset_id, dataset_name) in tqdm(pydata().iloc, total=757):
        dataset = pydata(dataset_id)
        score_dataset = pd.concat([score_dataset, evaluate_algorithm(dataset, dataset_name).to_frame().T], ignore_index=True)
        param_dataset = pd.concat([param_dataset, _extract(dataset, dataset_name).to_frame().T], ignore_index=True)

    return score_dataset, param_dataset
