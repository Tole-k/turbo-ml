from abc import ABC, ABCMeta, abstractmethod
from typing import Iterable, Tuple
import pandas as pd


class Preprocessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass