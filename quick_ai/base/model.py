from abc import ABC, ABCMeta, abstractmethod
import pickle
from typing import List, Iterable, Any, Type
from quick_ai.utils.error_tools.exceptions import NotTrainedException

__ALL_MODELS__: List[type] = []


class ModelMetaclass(type):

    def __new__(cls, name, bases, dct):
        new_class = super().__new__(cls, name, bases, dct)
        new_class._was_trained = False

        training = getattr(new_class, 'train', None)
        if training:
            def new_train(self, data: Iterable, target: Iterable):
                self._was_trained = True
                return training(self, data, target)
            setattr(new_class, 'train', new_train)

        prediction = getattr(new_class, 'predict', None)
        if prediction:
            def new_predict(self, guess: Any) -> List:
                if not self._was_trained:
                    raise NotTrainedException(
                        'Model must be trained before predicting')
                return prediction(self, guess)
            setattr(new_class, 'predict', new_predict)
        if training and prediction:
            __ALL_MODELS__.append(new_class)
        return new_class


class Model(metaclass=ModelMetaclass):
    def __init__(self) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def train(self, data: Iterable, target: Iterable) -> None:
        pass

    @abstractmethod
    def predict(self, guess: Any) -> List:
        pass

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)


def get_model_list() -> List[Type[Model]]:
    return __ALL_MODELS__
