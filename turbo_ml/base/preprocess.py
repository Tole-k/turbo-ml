from abc import ABC, abstractmethod
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

    @abstractmethod
    def fit_transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        pass

    @abstractmethod
    def transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        pass
