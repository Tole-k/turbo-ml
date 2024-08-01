import xgboost as xgb
from ..base import Model
from typing import List, Iterable, Optional
from sklearn.model_selection import train_test_split


class XGBoostClassifier(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {List[int] | List[bool]}

    def __init__(self,
                 max_depth: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 n_estimators: Optional[int] = None,
                 booster: Optional[str] = None,
                 tree_method: Optional[str] = None,
                 gamma: Optional[float] = None,
                 subsample: Optional[float] = None,
                 sampling_method: Optional[str] = None,
                 reg_alpha: Optional[float] = None,
                 reg_lambda: Optional[float] = None,
                 early_stopping_rounds: Optional[int] = None,
                 **options) -> None:
        super().__init__()
        if early_stopping_rounds is not None:
            self.early_stop = True
        self.clf = xgb.XGBClassifier(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            booster=booster,
            tree_method=tree_method,
            gamma=gamma,
            subsample=subsample,
            sampling_method=sampling_method,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            early_stopping_rounds=early_stopping_rounds,
            **options
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        if self.early_stop:
            x_train, x_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2)
            self.clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> List[int] | List[bool]:
        return self.clf.predict(guess)


class XGBoostRegressor(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {List[float]}

    def __init__(self,
                 max_depth: Optional[int] = None,
                 learning_rate: Optional[float] = None,
                 n_estimators: Optional[int] = None,
                 booster: Optional[str] = None,
                 tree_method: Optional[str] = None,
                 gamma: Optional[float] = None,
                 subsample: Optional[float] = None,
                 reg_alpha: Optional[float] = None,
                 early_stopping_rounds: Optional[int] = None,
                 **options) -> None:
        super().__init__()
        if early_stopping_rounds is not None:
            self.early_stop = True
        self.clf = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            booster=booster,
            tree_method=tree_method,
            gamma=gamma,
            subsample=subsample,
            reg_alpha=reg_alpha,
            early_stopping_rounds=early_stopping_rounds,
            **options
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        if self.early_stop:
            x_train, x_test, y_train, y_test = train_test_split(
                data, target, test_size=0.2)
            self.clf.fit(x_train, y_train, eval_set=[(x_test, y_test)])
        else:
            self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> List[float]:
        return self.clf.predict(guess)


# X = np.random.choice([True, False], size=(100, 10))
# y = np.random.choice([True, False], size=(100))
# clf = XGBoostRegressor(
#     tree_method="hist", early_stopping_rounds=2)
# clf.train(X, y)
# print(clf.predict(X))
