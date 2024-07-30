import xgboost as xgb
from ..base import Model
from typing import List, Iterable, Optional, Union, Sequence, Dict, Callable
import numpy as np
from sklearn.model_selection import train_test_split


class XGBoostClassifier(Model):
    _input_formats = {Iterable[int | float]}
    _output_formats = {List[int]}

    def __init__(self,
                 max_depth: Optional[int] = None,
                 max_leaves: Optional[int] = None,
                 max_bin: Optional[int] = None,
                 grow_policy: Optional[str] = None,
                 learning_rate: Optional[float] = None,
                 n_estimators: Optional[int] = None,
                 booster: Optional[str] = None,
                 tree_method: Optional[str] = None,
                 n_jobs: Optional[int] = None,
                 gamma: Optional[float] = None,
                 min_child_weight: Optional[float] = None,
                 max_delta_step: Optional[float] = None,
                 subsample: Optional[float] = None,
                 sampling_method: Optional[str] = None,
                 colsample_bytree: Optional[float] = None,
                 colsample_bylevel: Optional[float] = None,
                 colsample_bynode: Optional[float] = None,
                 reg_alpha: Optional[float] = None,
                 reg_lambda: Optional[float] = None,
                 scale_pos_weight: Optional[float] = None,
                 base_score: Optional[float] = None,
                 random_state: Optional[
                     Union[np.random.RandomState, np.random.Generator, int]
                 ] = None,
                 missing: float = np.nan,
                 num_parallel_tree: Optional[int] = None,
                 monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
                 interaction_constraints: Optional[Union[str,
                                                         Sequence[Sequence[str]]]] = None,
                 importance_type: Optional[str] = None,
                 device: Optional[str] = None,
                 validate_parameters: Optional[bool] = None,
                 enable_categorical: bool = False,
                 feature_types=None,
                 max_cat_to_onehot: Optional[int] = None,
                 max_cat_threshold: Optional[int] = None,
                 multi_strategy: Optional[str] = None,
                 eval_metric: Optional[Union[str, List[str], Callable]] = None,
                 early_stopping_rounds: Optional[int] = None,
                 early_stopping: bool = False) -> None:
        super().__init__()
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, metric_name='logloss', data_name='validation_0', save_best=True
        ) if early_stopping else None
        self.clf = xgb.XGBClassifier(
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            device=device,
            validate_parameters=validate_parameters,
            enable_categorical=enable_categorical,
            feature_types=feature_types,
            max_cat_to_onehot=max_cat_to_onehot,
            max_cat_threshold=max_cat_threshold,
            multi_strategy=multi_strategy,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[early_stop],
        )
        self.early_stopping = early_stopping

    def train(self, data: Iterable, target: Iterable) -> None:
        if self.early_stopping:
            x_train, x_val, y_train, y_val = train_test_split(
                data, target, test_size=0.20)
            self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable) -> List:
        return self.clf.predict(guess)


class XGBoostRegressor(Model):
    _input_formats = {Iterable[int | float]}
    _output_formats = {List[int | float]}

    def __init__(self,
                 max_depth: Optional[int] = None,
                 max_leaves: Optional[int] = None,
                 max_bin: Optional[int] = None,
                 grow_policy: Optional[str] = None,
                 learning_rate: Optional[float] = None,
                 n_estimators: Optional[int] = None,
                 booster: Optional[str] = None,
                 tree_method: Optional[str] = None,
                 n_jobs: Optional[int] = None,
                 gamma: Optional[float] = None,
                 min_child_weight: Optional[float] = None,
                 max_delta_step: Optional[float] = None,
                 subsample: Optional[float] = None,
                 sampling_method: Optional[str] = None,
                 colsample_bytree: Optional[float] = None,
                 colsample_bylevel: Optional[float] = None,
                 colsample_bynode: Optional[float] = None,
                 reg_alpha: Optional[float] = None,
                 reg_lambda: Optional[float] = None,
                 scale_pos_weight: Optional[float] = None,
                 base_score: Optional[float] = None,
                 random_state: Optional[
                     Union[np.random.RandomState, np.random.Generator, int]
                 ] = None,
                 missing: float = np.nan,
                 num_parallel_tree: Optional[int] = None,
                 monotone_constraints: Optional[Union[Dict[str, int], str]] = None,
                 interaction_constraints: Optional[Union[str,
                                                         Sequence[Sequence[str]]]] = None,
                 importance_type: Optional[str] = None,
                 device: Optional[str] = None,
                 validate_parameters: Optional[bool] = None,
                 enable_categorical: bool = False,
                 feature_types=None,
                 max_cat_to_onehot: Optional[int] = None,
                 max_cat_threshold: Optional[int] = None,
                 multi_strategy: Optional[str] = None,
                 eval_metric: Optional[Union[str, List[str], Callable]] = None,
                 early_stopping_rounds: Optional[int] = None,
                 early_stopping: bool = False) -> None:
        super().__init__()
        early_stop = xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, metric_name='logloss', data_name='validation_0', save_best=True
        ) if early_stopping else None
        self.clf = xgb.XGBRegressor(
            max_depth=max_depth,
            max_leaves=max_leaves,
            max_bin=max_bin,
            grow_policy=grow_policy,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            sampling_method=sampling_method,
            colsample_bytree=colsample_bytree,
            colsample_bynode=colsample_bynode,
            colsample_bylevel=colsample_bylevel,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            device=device,
            validate_parameters=validate_parameters,
            enable_categorical=enable_categorical,
            feature_types=feature_types,
            max_cat_to_onehot=max_cat_to_onehot,
            max_cat_threshold=max_cat_threshold,
            multi_strategy=multi_strategy,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[early_stop],
        )
        self.early_stopping = early_stopping

    def train(self, data: Iterable, target: Iterable) -> None:
        if self.early_stopping:
            x_train, x_val, y_train, y_val = train_test_split(
                data, target, test_size=0.20)
            self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable) -> List:
        return self.clf.predict(guess)


# X, y = load_breast_cancer(return_X_y=True)
# yp = np.array(["big" if i == 1 else "small" for i in y])
# clf = XGBoostClassifier(
#     tree_method="hist", early_stopping=True, early_stopping_rounds=2)
# clf.train(X, yp)
# print(clf.predict(X))
