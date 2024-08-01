from sklearn import ensemble
from sklearn.datasets import make_hastie_10_2
from ..base import Model
from typing import List, Iterable


class GBoostClassifier(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {List[int], List[str]}

    def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, min_samples_split=2, max_depth=3, **options) -> None:
        super().__init__()
        self.clf = ensemble.GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            **options
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        self.clf = self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> List[int] | List[str]:
        return self.clf.predict(guess)


class GBoostRegressor(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {List[float]}

    def __init__(self, loss='squared_error', learning_rate=0.1,
                 n_estimators=100, subsample=1.0,
                 min_samples_split=2, max_depth=3, **options) -> None:
        super().__init__()
        self.reg = ensemble.GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            **options
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        return self.reg.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> List[float]:
        return self.reg.predict(guess)


X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]
clf = GBoostClassifier(n_estimators=100, learning_rate=1.0,
                       max_depth=1, random_state=0)
clf.train(X_train, y_train)
print(clf.predict(X_test))
