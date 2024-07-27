from abc import ABC, abstractmethod
import pickle
from typing import List, Iterable
from .process import Process


class Model(ABC):
    input_formats = {Iterable}
    output_formats = {List}

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


class ModelProcess(Process):
    def __init__(self, model: Model) -> None:
        self.model = model
        super().__init__()

    def pr(self, guess: Iterable) -> List:
        return self.model.predict(guess)

    def tr(self, data: Iterable, target: Iterable) -> None:
        self.model.train(data, target)

    def available_input_formats(self) -> set:
        return self.model.input_formats

    def available_output_formats(self) -> set:
        return self.model.output_formats
