from sklearn import ensemble
from ..base import Model
from typing import List, Iterable


class GBoostClassifier(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {List[int], List[str]}

    def __init__(self, loss='log_loss', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                 min_impurity_decrease=0.0, init=None, random_state=None,
                 max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001,
                 ccp_alpha=0.0) -> None:
        super().__init__()
        self.clf = ensemble.GradientBoostingClassifier(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            ccp_alpha=ccp_alpha
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        self.clf = self.clf.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> List[int] | List[str]:
        return self.clf.predict(guess)


class GBoostRegressor(Model):
    input_formats = {Iterable[int | float]}
    output_formats = {List[float]}

    def __init__(self, loss='squared_error', learning_rate=0.1,
                 n_estimators=100, subsample=1.0, criterion='friedman_mse',
                 min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_depth=3,
                 min_impurity_decrease=0.0, init=None, random_state=None,
                 max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None,
                 warm_start=False, validation_fraction=0.1,
                 n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0) -> None:
        super().__init__()
        self.reg = ensemble.GradientBoostingRegressor(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=subsample,
            criterion=criterion,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_impurity_decrease,
            init=init,
            random_state=random_state,
            max_features=max_features,
            alpha=alpha,
            verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            ccp_alpha=ccp_alpha
        )

    def train(self, data: Iterable[int | float], target: Iterable) -> None:
        return self.reg.fit(data, target)

    def predict(self, guess: Iterable[int | float]) -> List[float]:
        return self.reg.predict(guess)


# X, y = make_hastie_10_2(random_state=0)
# X_train, X_test = X[:2000], X[2000:]
# y_train, y_test = y[:2000], y[2000:]
#
# clf = GBoostClassifier(n_estimators=100, learning_rate=1.0,
#                        max_depth=1, random_state=0)
# clf.train(X_train, y_train)
# print(clf.predict(X_test))
