import xg_boost as xgb
from ..base import Model
from typing import List, Iterable


class XGBoostClassifier(Model):
    def __init__(self, num_round=10) -> None:
        super().__init__()
        self.num_round = num_round

    def train(self, data: Iterable, target: Iterable) -> None:
        dtrain = xgb.DMatrix(data, label=target)
        self.bst = xgb.train({}, dtrain, self.num_round)

    def predict(self, guess: Iterable) -> List:
        dtest = xgb.DMatrix(guess)
        return self.bst.predict(dtest)