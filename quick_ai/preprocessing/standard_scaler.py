from typing import Iterable, Tuple
import numpy as np
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import StandardScaler

class Normalizer(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def preprocess(self, data: Iterable, target: Iterable) -> Tuple[Iterable]:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        target = scaler.fit_transform(np.transpose([target]))
        return data, target

data = [[0, 0], [0, 0], [1, 1], [1, 1]]
target = [1, 2, 3, 4]
normalizer = Normalizer()
data, target = normalizer.preprocess(data, target)
print(data)
print(target)
