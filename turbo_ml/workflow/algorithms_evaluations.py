from decorify import mute
from sklearn.model_selection import train_test_split
from turbo_ml.base import __ALL_MODELS__
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
from time import sleep
logger = getLogger(__name__)


def calculate_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_diff = y_true - y_pred
    n = len(y_diff)
    return sum(y_true == y_pred) / n


@task(name='Evaluate Models')
def evaluate_models(dataset_path: str, dataset_name: Optional[str] = None) -> pd.Series:
    print(dataset_name)
    dataset = read_data_file(dataset_path)
    y = dataset.iloc[:, -1]
    X = dataset.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    preprocessor = sota_preprocessor()
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    frame = {'name': dataset_name or dataset_path}
    frame.update({model.__name__: np.nan for model in __ALL_MODELS__})
    for model_cls in __ALL_MODELS__:
        try:
            model:Model = model_cls()
            model.train(X_train, y_train)
            y_pred = model.predict(X_test)
            score = calculate_score(y_test, y_pred)
            frame[model_cls.__name__] = score
        except Exception as e:
            logger.error(f'Error while evaluating model {model_cls.__name__}: {e}')
    return pd.Series(frame)


@task(name='Load Algorithm Evaluations')
def load_algorithms_evaluations():
    return pd.read_csv(os.path.join('datasets', 'results_algorithms.csv'))


@flow(name='Evaluate Models for every dataset', task_runner=DaskTaskRunner())
def evaluate_datasets(datasets_dir: str = os.path.join('datasets', 'AutoIRAD-datasets'),
                      output_path='results_algorithms.csv', slice_index:Optional[int]=None) -> pd.DataFrame:
    if slice_index is not None:
        names = list_dataset_files(datasets_dir)[slice_index*10:(slice_index+1)*10]
    else:
        names = list_dataset_files(datasets_dir)
    evaluations = []
    for dataset_name, path in names:
        evaluations.append(evaluate_models.submit(path, dataset_name))
    evaluations_results = [evaluation.result() for evaluation in evaluations]
    dataframe = pd.concat(evaluations_results, axis=1).T

    if dataframe is not None and output_path is not None:
        dataframe.to_csv(str(slice_index)+ '_' + output_path, index=False)
    return dataframe
