from typing import Literal, Mapping, Sequence
from numpy.random import RandomState
from ..base import Model
from sklearn import tree
from collections.abc import Iterable


class DecisionTreeClassifier(Model):
    input_formats = {Iterable[int | float | bool]}
    output_formats = {list[int], list[str] | list[bool]}
    hyperparameters = [
        {
            "name": "criterion", 
            "type": "categorical",
            "choices": ["gini", "entropy", "log_loss"],
            "optional": False
        },
        {
            "name": "splitter",
            "type": "categorical",
            "choices": ["best", "random"],
            "optional": False
        },
        {
            "name": "max_depth",
            "type": "int",
            "min": 1,
            "max": 100,
            "optional": True
        },
        {
            "name": "min_samples_split",
            "type": "int",
            "min": 2,
            "max": 10,
            "optional": False
        },
        {
            "name": "min_samples_leaf",
            "type": "int",
            "min": 1,
            "max": 10,
            "optional": False
        },
        {
            "name": "min_weight_fraction_leaf",
            "type": "float",
            "min": 0.0, "max": 0.5,
            "optional": False
        },
        {
            "name": "max_features",
            "type": "categorical",
            "choices": ["sqrt", "log2"],
            "optional": True
        },
        {
            "name": "max_leaf_nodes",
            "type": "int",
            "min": 2,
            "max": 100,
            "optional": True
        },
        {
            "name": "min_impurity_decrease",
            "type": "float",
            "min": 0.0, "max": 0.5,
            "optional": False
        },
        {
            "name": "ccp_alpha",
            "type": "float",
            "min": 0.0,
            "max": 0.5,
            "optional": False
        },
    ]

    def __init__(
        self,
        criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
        splitter: Literal['best', 'random'] = "best",
        max_depth: int | None = None,
        min_samples_split: float | int = 2,
        min_samples_leaf: float | int = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: float | int | Literal['sqrt', 'log2'] | None = None,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.tree = tree.DecisionTreeClassifier(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            ccp_alpha=ccp_alpha,
        )

    def train(self, data: Iterable[int | float | bool], target: Iterable) -> None:
        self.tree = self.tree.fit(data, target)

    def predict(self, guess: Iterable[int | float | bool]) -> list[int] | list[str] | list[bool]:
        return self.tree.predict(guess)
