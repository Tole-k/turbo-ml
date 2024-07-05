from typing import List, Iterable
from ..base.model import Model
import random


class RandomGuesser(Model):
    def __init__(self) -> None:
        super().__init__()
        self.possibilities = list()

    def train(self, data: Iterable, target: Iterable) -> None:
        for i in target:
            self.possibilities.append(i)

    def predict(self, guess: Iterable) -> List[int]:
        return [random.choice(self.possibilities) for _ in guess]
