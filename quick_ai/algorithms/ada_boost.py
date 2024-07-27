from sklearn import ensemble
from ..base import Model
from typing import List, Iterable


class AdaBoostClassifier(Model):
    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        learning_rate=1.0,
        algorithm='SAMME.R',
        random_state=None,
    ) -> None:
        super().__init__()
        self.ada_boost = ensemble.AdaBoostClassifier(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm=algorithm,
            random_state=random_state,
        )

    def train(self, data: Iterable, target: Iterable) -> None:
        self.ada_boost = self.ada_boost.fit(data, target)

    def predict(self, guess: Iterable) -> List:
        return self.ada_boost.predict(guess)


class AdaBoostRegressor(Model):
    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        learning_rate=1.0,
        loss='linear',
        random_state=None,
    ) -> None:
        super().__init__()
        self.ada_boost = ensemble.AdaBoostRegressor(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss,
            random_state=random_state,
        )

    def train(self, data: Iterable, target: Iterable) -> None:
        self.ada_boost = self.ada_boost.fit(data, target)

    def predict(self, guess: Iterable) -> List:
        return self.ada_boost.predict(guess)
