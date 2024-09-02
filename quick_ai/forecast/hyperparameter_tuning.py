import numpy as np
from quick_ai.algorithms import NeuralNetworkModel
import optuna as opt
from datasets import get_iris, get_diabetes, get_breast_cancer, get_linnerud
from sklearn.model_selection import train_test_split
from quick_ai.base import Model
from typing import Tuple
import pandas as pd
from quick_ai.algorithms import AdaBoostClassifier, AdaBoostRegressor, XGBoostClassifier, XGBoostRegressor
from typing import Literal
import json


class HyperTuner:

    def __init__(self) -> None:
        self.hyperparameters = json.load(
            open('quick_ai/forecast/hyperparameters.json'))

    @staticmethod
    def process_conditions(hyper_param: dict, no_classes: int, no_variables: int, device: str) -> dict:
        if 'conditional' in hyper_param:
            match hyper_param['condition']:
                case "binary/multi":
                    return hyper_param['variants'][not no_classes == 2]
                case 'single/multi':
                    return hyper_param['variants'][not no_variables == 1]
                case 'cpu/cuda':
                    return hyper_param['variants'][not device == 'cpu']
                # more cases to be added if needed
        return hyper_param

    def get_model_hyperparameters(self, model: Model) -> list:
        return self.hyperparameters[model.__name__]

    def objective(self, trial: opt.Trial, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu') -> float:
        x_train, x_test, y_train, y_test = train_test_split(
            *dataset, test_size=0.2)
        hyperparams: list = self.get_model_hyperparameters(model)
        params = {}
        for hyper_param in hyperparams:
            hyper_param = self.process_conditions(
                hyper_param, no_classes, no_variables, device)
            if hyper_param['optional'] and trial.suggest_categorical(f"{hyper_param['name']}=None", [True, False]):
                params[hyper_param['name']] = None
                continue
            if hyper_param['type'] == 'no_choice':
                params[hyper_param['name']] = hyper_param['choices'][0]
            if hyper_param['type'] == 'int':
                params[hyper_param['name']] = trial.suggest_int(
                    hyper_param['name'], hyper_param['min'], hyper_param['max'])
            elif hyper_param['type'] == 'float':
                params[hyper_param['name']] = trial.suggest_float(
                    hyper_param['name'], hyper_param['min'], hyper_param['max'])
            elif hyper_param['type'] == 'categorical':
                params[hyper_param['name']] = trial.suggest_categorical(
                    hyper_param['name'], hyper_param['choices'])
        model = model(**params)
        model.train(x_train, y_train)
        if task == 'classification':
            return sum(model.predict(x_test) == y_test)/len(y_test)
        else:
            return np.sum((model.predict(x_test)-y_test).values**2)/len(y_test)

    def optimize_hyperparameters(self, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu', trials: int = 10) -> dict:
        if model == NeuralNetworkModel:  # Neural Network requires a more specific approach, infeasible to adapt the general function do it's been implemented separately
            return NeuralNetworkModel.optimize_hyperparameters(dataset, task, no_classes, no_variables, device, trials)
        study = opt.create_study(
            direction='maximize' if task == 'classification' else 'minimize')
        study.optimize(lambda trial: self.objective(
            trial, model, dataset, task, no_classes, no_variables, device), n_trials=trials)
        return study.best_params


if __name__ == '__main__':
    tuner = HyperTuner()
    dataset = get_iris()
    model = AdaBoostClassifier
    task = 'classification'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, trials=10))
    model = NeuralNetworkModel
    task = 'classification'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, device=device, trials=10))
    dataset = get_diabetes()
    model = AdaBoostRegressor
    task = 'regression'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, trials=10))
    model = NeuralNetworkModel
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, device=device, trials=10))

    dataset = get_breast_cancer()
    model = XGBoostClassifier
    task = 'classification'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, trials=10))

    model = NeuralNetworkModel
    task = 'classification'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, trials=10))
    dataset = get_linnerud()
    model = XGBoostRegressor
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, trials=10))

    model = NeuralNetworkModel
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, trials=10))