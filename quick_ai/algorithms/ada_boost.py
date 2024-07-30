from sklearn import ensemble
from ..base import Model
from typing import List, Iterable


class AdaBoostClassifier(Model):
    _input_formats = {Iterable[int | float]}
    _output_formats = {List[int | str]}

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
    _input_formats = {Iterable[int | float]}
    _output_formats = {List[float]}

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


# X, y = make_classification(n_samples=1000, n_features=4,
#                            n_informative=2, n_redundant=0,
#                            random_state=0, shuffle=False)
# clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
# clf.train(X, y)
# print(clf.predict([[0, 0, 0, 0]]))
