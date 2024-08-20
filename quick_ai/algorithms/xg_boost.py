import numpy as np
import xgboost as xgb
from ..base import Model
from typing import Optional, List, Literal
from sklearn.model_selection import train_test_split
from collections.abc import Iterable
from datasets import get_iris, get_diabetes


class XGBoostClassifier(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {list[int] | list[bool]}

    def __init__(
        self,
        booster: Literal['gbtree', 'dart'] = 'gbtree',
        device: Literal['cpu', 'cuda'] = 'cpu',
        learning_rate: float = 0.3,
        gamma: float = 0,
        max_depth: int = 6,
        subsample: float = 1,
        sampling_method: Literal['uniform', 'gradient_based'] = 'uniform',
        colsample_bytree: float = 1,
        colsample_bynode: float = 1,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        grow_policy: Literal['depthwise', 'lossguide'] = 'depthwise',
        early_stopping_rounds: Optional[int] = None,
        early_stopping_validation_fraction: float = 0.2,
        objective: Literal['binary:hinge', 'multi:softmax'] = 'binary:hinge',
        eval_metric: Literal['logloss', 'error',
                             'mlogloss', 'merror'] = 'error',
        **options
    ) -> None:

        self.device = device
        if early_stopping_rounds is not None:
            self.early_stop = True
            self.early_validation_fraction = early_stopping_validation_fraction
        else:
            self.early_stop = False

        self.clf = xgb.XGBClassifier(
            booster=booster,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            max_depth=max_depth,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            grow_policy=grow_policy,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_validation_fraction=early_stopping_validation_fraction,
            objective=objective,
            eval_metric=eval_metric,
            **options
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        self.mapping = {i: j for j, i in enumerate(set(target))}
        target = target.map(lambda x: self.mapping[x])
        if self.device == 'cuda':
            import cupy as cp
            data = cp.array(data)
            target = cp.array(target)
        if self.early_stop:
            x_train, x_test, y_train, y_test = train_test_split(
                data, target, test_size=self.early_validation_fraction)
            self.clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> List[int] | List[bool]:
        if self.device == 'cuda':
            import cupy as cp
            guess = cp.array(guess)
        return np.vectorize(self.mapping.get)(self.clf.predict(guess))


class XGBoostRegressor(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {list[float]}

    def __init__(
        self,
        booster: Literal['gbtree', 'gblinear', 'dart'] = 'gbtree',
        device: Literal['cpu', 'cuda'] = 'cpu',
        learning_rate: float = 0.3,
        gamma: float = 0,
        max_depth: int = 6,
        subsample: float = 1,
        sampling_method: Literal['uniform', 'gradient_based'] = 'uniform',
        colsample_bytree: float = 1,
        colsample_bynode: float = 1,
        reg_lambda: float = 1,
        reg_alpha: float = 0,
        grow_policy: Literal['depthwise', 'lossguide'] = 'depthwise',
        early_stopping_rounds: Optional[int] = None,
        early_stopping_validation_fraction: float = 0.2,
        objective: Literal['reg:squarederror', 'reg:squaredlogerror',
                           'reg:pseudohubererror', 'reg:absoluteerror'] = 'reg:squarederror',
        eval_metric: Literal['rmse', 'rmsle', 'mae', 'mape', 'mphe'] = 'rmse',
        **options
    ) -> None:

        self.device = device
        if early_stopping_rounds is not None:
            self.early_stop = True
            self.early_validation_fraction = early_stopping_validation_fraction
        else:
            self.early_stop = False

        if self.device == 'cuda':
            options['updater'] = 'grow_gpu_hist'

        self.clf = xgb.XGBRegressor(
            booster=booster,
            device=device,
            learning_rate=learning_rate,
            gamma=gamma,
            max_depth=max_depth,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            grow_policy=grow_policy,
            early_stopping_rounds=early_stopping_rounds,
            early_stopping_validation_fraction=early_stopping_validation_fraction,
            objective=objective,
            eval_metric=eval_metric,
            **options
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        if self.device == 'cuda':
            import cupy as cp
            data = cp.array(data)
            target = cp.array(target)
        if self.early_stop:
            x_train, x_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2)
            self.clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> list[float]:
        if self.device == 'cuda':
            import cupy as cp
            guess = cp.array(guess)
        return self.clf.predict(guess)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = train_test_split(
        *get_iris(), test_size=0.2)
    clf = XGBoostClassifier(booster='dart', device='cuda', learning_rate=0.1, max_depth=3, subsample=0.5, sampling_method='gradient_based', colsample_bytree=0.5, colsample_bynode=0.5,
                            reg_lambda=0.5, reg_alpha=0.5, grow_policy='lossguide', early_stopping_rounds=2, early_stopping_validation_fraction=0.2, objective='multi:softmax', eval_metric='mlogloss')
    clf.train(x_train, y_train)
    print(clf.predict(x_test) == y_test)

    x_train, x_test, y_train, y_test = train_test_split(
        *get_diabetes(), test_size=0.2)
    reg = XGBoostRegressor(booster='dart', device='cuda', learning_rate=0.1, max_depth=3, subsample=0.5, sampling_method='gradient_based', colsample_bytree=0.5, colsample_bynode=0.5,
                           reg_lambda=0.5, reg_alpha=0.5, grow_policy='lossguide', early_stopping_rounds=2, early_stopping_validation_fraction=0.2, objective='reg:squarederror', eval_metric='rmse')
    reg.train(x_train, y_train)
    print(np.mean((reg.predict(x_test)-y_test)**2))
