from time import sleep
import numpy as np
from quick_ai.algorithms import NeuralNetworkModel
import optuna as opt
from datasets import get_iris, get_diabetes, get_breast_cancer, get_linnerud
from sklearn.model_selection import train_test_split
from quick_ai.base import Model
from typing import Tuple
import pandas as pd
from typing import Literal
import json
from quick_ai.utils import option


class HyperTuner:

    def __init__(self) -> None:
        # opt.logging.set_verbosity(opt.logging.WARNING)
        self.sklearn_hyperparameters = json.load(
            open('quick_ai/forecast/sklearn_hyperparameters.json'))
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
        if option.hyperparameters_declaration_priority == 'sklearn' and model.__name__ in self.sklearn_hyperparameters:
            return self.sklearn_hyperparameters[model.__name__]
        elif option.hyperparameters_declaration_priority == 'sklearn' and model.__name__ in self.hyperparameters:
            return self.hyperparameters[model.__name__]
        elif option.hyperparameters_declaration_priority == 'custom' and model.__name__ in self.hyperparameters:
            return self.hyperparameters[model.__name__]
        elif option.hyperparameters_declaration_priority == 'custom' and model.__name__ in self.sklearn_hyperparameters:
            return self.sklearn_hyperparameters[model.__name__]
        else:
            raise ValueError(
                f"Model {model} not found in hyperparameters database")

    def objective(self, trial: opt.Trial, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu') -> float:
        x_train, x_test, y_train, y_test = train_test_split(
            *dataset, test_size=0.2)
        hyperparams: list = self.get_model_hyperparameters(model)
        params = {}
        for hyper_param in hyperparams:
            hyper_param = self.process_conditions(
                hyper_param, no_classes, no_variables, device)
            if hyper_param['optional'] and trial.suggest_categorical(f"{hyper_param['name']}=None", [True, False]):
                continue
            if hyper_param['type'] == 'no_choice':
                params[hyper_param['name']] = hyper_param['choices'][0]
                trial.set_user_attr(
                    hyper_param['name'], hyper_param['choices'][0])
            if hyper_param['type'] == 'int':
                params[hyper_param['name']] = trial.suggest_int(
                    hyper_param['name'], hyper_param['min'] if hyper_param['min'] is not None else 0, hyper_param['max'] if hyper_param['max'] is not None else 100)
            elif hyper_param['type'] == 'float':
                params[hyper_param['name']] = trial.suggest_float(
                    hyper_param['name'], hyper_param['min'] if hyper_param['min'] is not None else 0, hyper_param['max'] if hyper_param['max'] is not None else 100)
            elif hyper_param['type'] == 'categorical':
                params[hyper_param['name']] = trial.suggest_categorical(
                    hyper_param['name'], hyper_param['choices'])
            elif hyper_param['type'] == 'bool':
                params[hyper_param['name']] = trial.suggest_categorical(
                    hyper_param['name'], [True, False])
        model = model(**params)
        model.train(x_train, y_train)
        if task == 'classification':
            return sum(model.predict(x_test) == y_test)/len(y_test)
        else:
            return np.sum((model.predict(x_test)-y_test).values**2)/len(y_test)

    def filter_nones(self, best_params: dict) -> dict:
        return {k: v for k, v in best_params.items() if k[-5:] != '=None'}

    def optimize_hyperparameters(self, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu', trials: int = 10) -> dict:
        if model == NeuralNetworkModel:  # Neural Network requires a more specific approach, infeasible to adapt the general function do it's been implemented separately
            return NeuralNetworkModel.optimize_hyperparameters(dataset, task, no_classes, no_variables, device, trials)
        study = opt.create_study(
            direction='maximize' if task == 'classification' else 'minimize', study_name=model.__name__ + " Hyperparameter Optimization")
        study.optimize(lambda trial: self.objective(
            trial, model, dataset, task, no_classes, no_variables, device), n_trials=trials)
        return self.filter_nones(study.best_params) | study.best_trial.user_attrs


if __name__ == '__main__':

    from quick_ai.algorithms import AdaBoostClassifier, AdaBoostRegressor, XGBoostClassifier, XGBoostRegressor
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
