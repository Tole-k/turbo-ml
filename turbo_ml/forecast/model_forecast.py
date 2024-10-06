from abc import abstractmethod
from typing import *
from turbo_ml.base import Model
from turbo_ml.base.model import get_models_list
import random
import math


def evaluate(model: Type[Model], data: Any, target: Any) -> float:
    # TODO: Simple evaluation function, to be improved and moved somewhere else
    try:
        train_size = int(len(data) * 9/10)
        model = model()
        model.train(data[:train_size], target[:train_size])
        predictions = model.predict(data[train_size:])
        mse = sum((pred - targ) ** 2 for pred, targ in zip(predictions,
                                                           target[train_size:])) / len(predictions)
    except Exception as e:
        mse = float('inf')
    return -mse


class Forecast:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def predict(self, data, target) -> Model:
        pass


class ExhaustiveSearch(Forecast):
    """ Search for the best model by evaluating all models in the list and picking the best one based on the evaluation function
    This search ignores HPO steps and focus only on AS based on predefined hyper-parameters"""

    def __init__(self) -> None:
        self.counter = 0

    def predict(self, data, target) -> Model:
        best_model: Tuple = (None, -float('inf'))
        models = get_models_list().copy()
        random.shuffle(models)
        for model_cls in models:
            try:
                value = evaluate(model_cls, data, target)
                if math.isinf(value):  # Ignoring inf values
                    continue
                if value > best_model[1]:
                    best_model = (model_cls, value)
                self.counter += 1
            except:
                continue
        return best_model[0]()
