from ..base import Model
import random
from collections.abc import Iterable


class RandomGuesser(Model):
    input_formats = {Iterable}
    output_formats = {list[int]}

    def __init__(self) -> None:
        super().__init__()
        self.possibilities: list[int] = list()
        self.mapping = dict()
        self.mapping_couter = 0

    def train(self, data: Iterable, target: Iterable) -> None:
        for value in target:
            if value not in self.mapping:
                self.mapping[value] = self.mapping_couter
                self.mapping_couter += 1
            self.possibilities.append(self.mapping[value])

    def predict(self, guess: Iterable) -> list[int]:
        return [random.choice(self.possibilities) for _ in guess]
