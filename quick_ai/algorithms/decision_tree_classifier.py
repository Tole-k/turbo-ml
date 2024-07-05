from typing import List, Iterable, Literal,Mapping,Sequence
from numpy.random import RandomState
from ..base.model import Model
from sklearn import tree


class DecisionTreeClassifier(Model):

    def __init__(
        self,
        criterion: Literal['gini','entropy','log_loss'] = "gini",
        splitter: Literal['best','random'] = "best",
        max_depth: int|None = None,
        min_samples_split: float|int = 2,
        min_samples_leaf: float|int = 1,
        min_weight_fraction_leaf: float=0.0,
        max_features: float|int|Literal['auto','sqrt','log2']|None = None,
        random_state: int|RandomState|None = None,
        max_leaf_nodes: int|None = None,
        min_impurity_decrease: float = 0.0,
        class_weight: Mapping|str|Sequence[Mapping]|None = None,
        ccp_alpha:float=0.0,
        monotonic_cst=None,
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
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            monotonic_cst=monotonic_cst,
            ccp_alpha=ccp_alpha,
        )

    def train(self, data: Iterable, target: Iterable) -> None:
        self.tree = self.tree.fit(data, target)

    def predict(self, guess: Iterable) -> List:
        return self.tree.predict(guess)
