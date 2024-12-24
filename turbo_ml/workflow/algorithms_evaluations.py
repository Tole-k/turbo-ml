from sklearn.model_selection import train_test_split
from turbo_ml.base import get_models_list
from turbo_ml.workflow.utils import list_dataset_files, read_data_file
from turbo_ml.base.model import Model
from turbo_ml.preprocessing import sota_preprocessor
from prefect_dask import DaskTaskRunner
from prefect import flow, task
import os
import pandas as pd
import numpy as np
from logging import getLogger
from typing import Optional
import re
from pydataset import data
logger = getLogger(__name__)


def calculate_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_diff = y_true - y_pred
    n = len(y_diff)
    return sum(y_true == y_pred) / n

@task(name='Evaluate dataset from pydataset')
def evaluate_from_pydataset(dataset_name:str) -> pd.Series:
    return evaluate_algorithms(data(dataset_name), dataset_name)


@task(name='Evaluate algorithms from file')
def evaluate_from_file(dataset_path: str) -> pd.Series:
    dataset = read_data_file(dataset_path)
    return evaluate_algorithms(dataset, re.split(r' |\.', dataset_path))
    

def evaluate_algorithms(dataset: pd.DataFrame, dataset_name: str) -> pd.Series:
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    preprocessor = sota_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    frame = {'name': dataset_name}
    frame.update({model.__name__: np.nan for model in get_models_list()})
    for model_cls in get_models_list():
        try:
            model: Model = model_cls()
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            score = calculate_score(y_test, y_pred)
            frame[model_cls.__name__] = score
        except Exception as e:
            logger.error(f'Error while evaluating model {
                         model_cls.__name__}: {e}')
    return pd.Series(frame)


@task(name='Load Algorithm Evaluations')
def load_algorithms_evaluations(path: str = os.path.join('datasets', 'results_algorithms.csv')):
    return pd.read_csv(path)


@flow(name='Evaluate Models for every dataset', task_runner=DaskTaskRunner())
def evaluate_datasets(datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'),
                      output_path='results_algorithms.csv', slice_index: Optional[int] = None) -> pd.DataFrame:
    if slice_index is not None:
        names = list_dataset_files(datasets_dir)[
            slice_index*10:(slice_index+1)*10]
    else:
        names = list_dataset_files(datasets_dir)
    evaluations = []
    for dataset_name, path in names:
        evaluations.append(evaluate_algorithms.submit(path, dataset_name))
    evaluations_results = [evaluation.result() for evaluation in evaluations]
    dataframe = pd.concat(evaluations_results, axis=1).T

    if dataframe is not None and output_path is not None:
        dataframe.to_csv(str(slice_index) + '_' + output_path, index=False)
    return dataframe
