from typing import Iterable, Tuple
import numpy as np
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import MinMaxScaler

class Normalizer(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, data: Iterable, target: Iterable) -> Tuple[Iterable]:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        target = scaler.fit_transform(np.transpose([target]))
        return data, target

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
target = [1, 2, 3, 4]
normalizer = Normalizer()
data, target = normalizer.preprocess(data, target)
print(data)
print(target)
