import numpy as np
import optuna as opt
from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
from typing import Literal
import json
from turbo_ml.algorithms import NeuralNetworkModel
from turbo_ml.base import Model
from turbo_ml.utils import options


class HyperTuner:

    def __init__(self) -> None:
        opt.logging.set_verbosity(
            verbosity=options.dev_mode_logging if options.dev_mode else options.user_mode_logging)
        self.sklearn_hyperparameters = json.load(
            open('turbo_ml/meta_learning/hpo/sklearn_hyperparameters.json'))
        self.hyperparameters = json.load(
            open('turbo_ml/meta_learning/hpo/hyperparameters.json'))

    @staticmethod
    def process_conditions(hyper_param: dict, no_classes: int, no_variables: int, device: str) -> dict:
        if 'conditional' in hyper_param:
            match hyper_param['condition']:
                case "binary/multi":
                    return hyper_param['variants'][not no_classes == 2]
                case 'single/multi':
                    return hyper_param['variants'][not no_variables == 1]
                case 'cpu/cuda':
                    return hyper_param['variants'][device == 'cuda']
                # more cases to be added if needed
        return hyper_param

    def get_model_hyperparameters(self, model: Model) -> list:
        if options.hyperparameters_declaration_priority == 'sklearn' and model.__name__ in self.sklearn_hyperparameters:
            return self.sklearn_hyperparameters[model.__name__]
        elif options.hyperparameters_declaration_priority == 'sklearn' and model.__name__ in self.hyperparameters:
            return self.hyperparameters[model.__name__]
        elif options.hyperparameters_declaration_priority == 'custom' and model.__name__ in self.hyperparameters:
            return self.hyperparameters[model.__name__]
        elif options.hyperparameters_declaration_priority == 'custom' and model.__name__ in self.sklearn_hyperparameters:
            return self.sklearn_hyperparameters[model.__name__]
        else:
            raise ValueError(
                f"Model {model} not found in hyperparameters database")

    def objective(self, trial: opt.Trial, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu', thread_num=1) -> float:
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
            elif hyper_param['type'] == 'thread_num':
                trial.set_user_attr(
                    hyper_param['name'], thread_num)
                params[hyper_param['name']
                       ] = trial.user_attrs[hyper_param['name']]
        model = model(**params)
        try:
            model.train(x_train, y_train)
            if task == 'classification':
                return sum(model.predict(x_test) == y_test)/len(y_test)
            else:
                return np.sum((model.predict(x_test)-y_test).values**2)/len(y_test)
        except Exception as e:
            print(e)
            raise opt.TrialPruned()

    def filter_nones(self, best_params: dict) -> dict:
        return {k: v for k, v in best_params.items() if k[-5:] != '=None'}

    def optimize_hyperparameters(self, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None, device='cpu', trials: int = 10, thread_num=1) -> dict:
        if model.__name__ in options.blacklist:
            return {}
        if model == NeuralNetworkModel:  # Neural Network requires a more specific approach, infeasible to adapt the general function do it's been implemented separately
            return NeuralNetworkModel.optimize_hyperparameters(dataset, task, no_classes, no_variables, device, trials*10)
        study = opt.create_study(
            direction='maximize' if task == 'classification' else 'minimize', study_name=model.__name__ + " Hyperparameter Optimization")
        study.optimize(lambda trial: self.objective(
            trial, model, dataset, task, no_classes, no_variables, device, thread_num=thread_num), n_trials=trials)
        return self.filter_nones(study.best_params) | study.best_trial.user_attrs


def __main_import__():
    from datasets import get_iris, get_diabetes, get_breast_cancer, get_linnerud
    return get_iris, get_diabetes, get_breast_cancer, get_linnerud


if __name__ == '__main__':
    get_iris, get_diabetes, get_breast_cancer, get_linnerud = __main_import__()
    from turbo_ml.algorithms import AdaBoostClassifier, AdaBoostRegressor, XGBoostClassifier, XGBoostRegressor
    tuner = HyperTuner()
    dataset = get_iris()
    model = AdaBoostClassifier
    task = 'classification'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, thread_num=options.threads))
    model = NeuralNetworkModel
    task = 'classification'
    device = 'cuda'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=3, device=device, thread_num=options.threads))
    dataset = get_diabetes()
    model = AdaBoostRegressor
    task = 'regression'
    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, thread_num=options.threads))
    model = NeuralNetworkModel
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=1, device=device, thread_num=options.threads))

    dataset = get_breast_cancer()
    model = XGBoostClassifier
    task = 'classification'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, thread_num=options.threads))

    model = NeuralNetworkModel
    task = 'classification'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_classes=2, device=device, thread_num=options.threads))
    dataset = get_linnerud()
    model = XGBoostRegressor
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, thread_num=options.threads))

    model = NeuralNetworkModel
    task = 'regression'

    print(tuner.optimize_hyperparameters(
        model, dataset, task, no_variables=3, device=device, thread_num=options.threads))
