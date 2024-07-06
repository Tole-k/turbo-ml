from abc import ABC, ABCMeta, abstractmethod
from typing import Iterable, Tuple


class Preprocessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def preprocess(self, data: Iterable, target: Iterable) -> Tuple[Iterable]:
        pass
