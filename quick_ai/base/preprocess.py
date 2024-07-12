from abc import ABC, ABCMeta, abstractmethod
from typing import Iterable, Tuple
import pandas as pd


class Preprocessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def preprocess(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        pass
