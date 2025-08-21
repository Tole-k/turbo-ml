import os
from logging import getLogger
import re

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pydataset import data

from sageml.base import get_models_list
from sageml.workflow.utils import list_dataset_files, read_data_file
from sageml.base.model import Model
from sageml.preprocessing import sota_preprocessor
logger = getLogger(__name__)


def calculate_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_diff = y_true - y_pred
    n = len(y_diff)
    return sum(y_true == y_pred) / n


def evaluate_from_pydataset(dataset_name: str) -> pd.Series:
    return evaluate_algorithm(data(dataset_name), dataset_name)


def evaluate_from_file(dataset_path: str) -> pd.Series:
    dataset = read_data_file(dataset_path)
    return evaluate_algorithm(dataset, re.split(r' |\.', dataset_path))


def evaluate_algorithm(dataset: pd.DataFrame, dataset_name: str) -> pd.Series:
    """Evaluates given algorithm and creates meta-dataset entry

    Args:
        dataset (pd.DataFrame): Dataset
        dataset_name (str): Name of given dataset.

    Returns:
        pd.Series: Meta-Dataset entry.
    """
    y = dataset.iloc[:, -1]
    x = dataset.iloc[:, :-1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    preprocessor = sota_preprocessor()
    x_train = preprocessor.fit_transform(x_train)
    x_test = preprocessor.transform(x_test)
    y_train = preprocessor.fit_transform_target(y_train)
    y_test = preprocessor.transform_target(y_test)
    frame = {'name': dataset_name}
    frame.update({model.__name__: np.nan for model in get_models_list()})
    for model_cls in get_models_list():
        if model_cls.__name__ in ['XGBoostRegressor']:  # ignore regressors in classification tasks
            continue
        try:
            model: Model = model_cls()
            model.train(x_train, y_train)
            y_pred = model.predict(x_test)
            score = calculate_score(y_test.to_numpy(), np.asarray(y_pred))
            frame[model_cls.__name__] = score
        except Exception as e:
            frame[model_cls.__name__] = np.nan
            logger.error(f'Error while evaluating model {model_cls.__name__}: {e}')
    return pd.Series(frame)


def load_algorithms_evaluations(path: str = os.path.join('datasets', 'results_algorithms.csv')):
    return pd.read_csv(path)


def evaluate_datasets(datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'),
                      output_path='results_algorithms.csv', slice_index: int | None = None) -> pd.DataFrame:
    if slice_index is not None:
        names = list_dataset_files(datasets_dir)[
            slice_index*10:(slice_index+1)*10]
    else:
        names = list_dataset_files(datasets_dir)
    evaluations = []
    for dataset_name, path in names:
        evaluations.append(evaluate_algorithm(read_data_file(path), dataset_name))
    dataframe = pd.concat(evaluations, axis=1).T

    if dataframe is not None and output_path is not None:
        dataframe.to_csv(str(slice_index) + '_' + output_path, index=False)
    return dataframe
