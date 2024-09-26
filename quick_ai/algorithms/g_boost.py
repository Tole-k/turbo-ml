from sklearn import ensemble
from sklearn.datasets import make_hastie_10_2
from ..base import Model
from collections.abc import Iterable
from typing import Literal


class GradientBoostingClassifier(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {list[int], list[str]}

    def __init__(
        self, loss: Literal['log_loss', 'exponential'] = 'log_loss',
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: Literal['friedman_mse', 'squared_error'] = 'friedman_mse',
        min_samples_split: int = 2,
        max_depth: int | None = 3,
        max_features: Literal['sqrt', 'log2'] | None = None,
        n_iter_no_change: int | None = None,
        ccp_alpha: float = 0.0,
        **options
    ) -> None:
        super().__init__()
        self.clf = ensemble.GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            ccp_alpha=ccp_alpha,
            **options
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        self.clf = self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> list[int] | list[str]:
        return self.clf.predict(guess)


class GBoostRegressor(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {list[float]}

    def __init__(
        self, loss: Literal['squared_error', 'absolute_error', 'huber', 'quantile'] = 'squared_error',
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: Literal['friedman_mse', 'squared_error'] = 'friedman_mse',
        min_samples_split: int = 2,
        max_depth: int | None = 3,
        max_features: Literal['sqrt', 'log2'] | None = None,
        n_iter_no_change: int | None = None,
        ccp_alpha: float = 0.0,
        **options
    ) -> None:
        super().__init__()
        self.reg = ensemble.GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            ccp_alpha=ccp_alpha,
            **options
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        return self.reg.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> list[float]:
        return self.reg.predict(guess)


if __name__ == "__main__":
    X, y = make_hastie_10_2(random_state=0)
    X_train, X_test = X[:2000], X[2000:]
    y_train, y_test = y[:2000], y[2000:]
    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                     max_depth=1, random_state=0)
    clf.train(X_train, y_train)
    print(clf.predict(X_test))
