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


class HyperTuner:
    @staticmethod
    def objective(trial: opt.Trial, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu') -> float:
        x_train, x_test, y_train, y_test = train_test_split(
            *dataset, test_size=0.2)
        hyperparams: list = model.hyperparameters
        params = {}
        for hyper_param in hyperparams:
            if 'conditional' in hyper_param:
                match hyper_param['condition']:
                    case "binary/multi":  # Depends on the number of classes in target feature
                        if no_classes == 2:
                            hyper_param = hyper_param['variants'][0]
                        else:
                            hyper_param = hyper_param['variants'][1]
                    case 'single/multi':  # Depends on the number of target features to be predicted
                        if no_variables == 1:
                            hyper_param = hyper_param['variants'][0]
                        else:
                            hyper_param = hyper_param['variants'][1]
                    case 'cpu/cuda':  # Depends on the availability of GPU
                        if device == 'cpu':
                            hyper_param = hyper_param['variants'][0]
                        else:
                            hyper_param = hyper_param['variants'][1]
            # more cases to be added if needed
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
        if model == NeuralNetworkModel: # Neural Network requires a more specific approach, infeasible to adapt the general function do it's been implemented separately
            return NeuralNetworkModel.optimize_hyperparameters(dataset, task, no_classes, no_variables, device, trials)
        study = opt.create_study(
            direction='maximize' if task == 'classification' else 'minimize')
        study.optimize(lambda trial: self.objective(
            trial, model, dataset, task, no_classes, no_variables, device), n_trials=trials)
        return study.best_params


if __name__ == '__main__':
    tuner = HyperTuner()
    dataset = get_iris()
    print(type(dataset[1]))
    model = AdaBoostClassifier
    task = 'classification'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, trials=100))
    model = NeuralNetworkModel
    task = 'classification'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, device=device, trials=100))
    dataset = get_diabetes()
    model = AdaBoostRegressor
    task = 'regression'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, trials=100))
    model = NeuralNetworkModel
    task = 'regression'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, device=device, trials=100))

    dataset = get_breast_cancer()
    model = XGBoostClassifier
    task = 'classification'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, trials=100))

    model = NeuralNetworkModel
    task = 'classification'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, trials=100))
    dataset = get_linnerud()
    model = XGBoostRegressor
    task = 'regression'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, trials=100))

    model = NeuralNetworkModel
    task = 'regression'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, trials=100))
