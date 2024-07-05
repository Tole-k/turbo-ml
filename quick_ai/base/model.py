from abc import ABC, ABCMeta, abstractmethod
import pickle
from typing import List, Iterable


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def train(self, data: Iterable, target: Iterable) -> None:
        pass

    @abstractmethod
    def predict(self, guess: Iterable) -> List:
        pass

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)
