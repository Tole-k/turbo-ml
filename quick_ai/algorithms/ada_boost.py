from sklearn import ensemble
from ..base import Model
from collections.abc import Iterable
from sklearn.datasets import make_classification
from typing import Literal
from sklearn.base import BaseEstimator


class AdaBoostClassifier(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {list[int], list[str]}
    # TODO: ad estimator hyperparameter
    hyperparameters = [
        {
            "name": "n_estimators",
            "type": "int",
            "min": 1,
            "max": 1000,
            "optional": False
        },
        {
            "name": "learning_rate",
            "type": "float",
            "min": 0.0,
            "max": 10.0,
            "optional": False
        }
    ]

    def __init__(
        self,
        estimator: BaseEstimator = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state=None,
    ) -> None:
        super().__init__()
        self.ada_boost = ensemble.AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="SAMME",
            random_state=random_state,
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        self.ada_boost = self.ada_boost.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> list[int] | list[str]:
        return self.ada_boost.predict(guess)


class AdaBoostRegressor(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {list[float]}
    # TODO: ad estimator hyperparameter
    hyperparameters = [
        {
            "name": "n_estimators",
            "type": "int",
            "min": 1,
            "max": 1000,
            "optional": False
        },
        {
            "name": "learning_rate",
            "type": "float",
            "min": 0.0,
            "max": 10.0,
            "optional": False
        },
        {
            "name": "loss",
            "type": "categorical",
            "choices": ["linear", "square", "exponential"],
            "optional": False
        }
    ]

    def __init__(
        self,
        estimator: BaseEstimator = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: Literal["linear", "square", "exponential"] = 'linear',
        random_state=None,
    ) -> None:
        super().__init__()
        self.ada_boost = ensemble.AdaBoostRegressor(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state,
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        self.ada_boost = self.ada_boost.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> list[float]:
        return self.ada_boost.predict(guess)


if __name__ == "__main__":
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.train(X, y)
    print(clf.predict([[0, 0, 0, 0]]))
