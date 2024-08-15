
from quick_ai.algorithms import DecisionTreeClassifier
import optuna as opt
from datasets import get_iris, get_diabetes
from sklearn.model_selection import train_test_split
from quick_ai.base import Model
from typing import Tuple
import pandas as pd
from quick_ai.algorithms import AdaBoostClassifier, AdaBoostRegressor, CatBoostClassifier, CatBoostRegressor
from math import inf
from typing import Literal


def objective(trial: opt.Trial, model: Model, dataset: Tuple[pd.DataFrame, pd.DataFrame], task: Literal['classification', 'regression'], no_classes: int = None, no_variables: int = None) -> float:
    x_train, x_test, y_train, y_test = train_test_split(
        *dataset, test_size=0.2)
    hyperparams: list = model.hyperparameters
    params = {}
    for hyper_param in hyperparams:
        if 'conditional' in hyper_param:
            match hyper_param['condition']:
                case "binary/multi":
                    if no_classes == 2:
                        hyper_param = hyper_param['variants'][0]
                    else:
                        hyper_param = hyper_param['variants'][1]
                case 'single/multi':
                    if no_variables == 1:
                        hyper_param = hyper_param['variants'][0]
                    else:
                        hyper_param = hyper_param['variants'][1]
        # TODO: more cases to be added if needed
        if hyper_param['optional'] and trial.suggest_categorical(f"{hyper_param['name']}=None", [True, False]):
            params[hyper_param['name']] = None
            continue
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
        return -sum((model.predict(x_test)-y_test)**2)/len(y_test)


if __name__ == '__main__':
    #study = opt.create_study(
    #    direction='maximize', study_name='catboost')
    #study.optimize(lambda trial: objective(
    #    trial, CatBoostClassifier, get_iris(), task='classification', no_classes=3), n_trials=10)
    #print(study.best_params)

    study = opt.create_study(
        direction='maximize', study_name='catboost')
    study.optimize(lambda trial: objective(
        trial, CatBoostRegressor, get_diabetes(), task='regression', no_variables=1), n_trials=10)

    print(study.best_params)
