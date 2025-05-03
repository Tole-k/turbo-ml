""" Functions for easier CLI operations """
import os
import pandas as pd
from sageml.workflow.utils import list_dataset_files, read_data_file
from sageml.workflow.algorithms_evaluations import evaluate_algorithm


def _read_dataset(path: str) -> pd.Series:
    dataset = read_data_file(path)
    entry = evaluate_algorithm(dataset, os.path.splitext(os.path.basename(path))[0])
    return entry


def create_dataset(path: str) -> pd.DataFrame:
    """Runs all datasets found in path on all algorithms

    Args:
        path (str): path to directory with datasets

    Returns:
        pd.DataFrame: combined dataset
    """
    dataset = pd.DataFrame()
    for path in list_dataset_files(path):
        dataset = pd.concat([dataset, _read_dataset(path).to_frame().T], ignore_index=True)
    return dataset
