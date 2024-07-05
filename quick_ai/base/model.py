from abc import ABC, ABCMeta, abstractmethod
import pickle
from typing import List


class Model(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def train(self, data) -> None:
        pass

    @abstractmethod
    def predict(self, guess) -> List[int]:
        pass

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self.model, file)
