from typing import Iterable, Tuple
import numpy as np
from ..base.preprocess import Preprocessor
from sklearn.preprocessing import MinMaxScaler


class Normalizer(Preprocessor):
    def __init__(self) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    def fit(self, data: Iterable, target: Iterable) -> Tuple[Iterable]:
        data = self.scaler.fit_transform(data)
        target = self.target_scaler.fit_transform(np.transpose([target]))
        return data, target

    def preprocess(self, data: Iterable) -> Tuple[Iterable]:
        return self.scaler.transform(data)

    def inverse(self, data: Iterable) -> Tuple[Iterable]:
        return self.scaler.inverse_transform(data)

    def preprocess_target(self, target: Iterable) -> Tuple[Iterable]:
        return self.target_scaler.transform(np.transpose([target]))

    def inverse_target(self, target: Iterable) -> Tuple[Iterable]:
        target = self.target_scaler.inverse_transform(target)
        return np.transpose(target)[0]


def main():

    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    target = [1, 2, 3, 4]
    normalizer = Normalizer()
    data, target = normalizer.fit(data, target)
    print(data)
    data = [[1, 2], [0.1, 13]]
    data = normalizer.preprocess(data)
    print(data)
    print(target)
    target = normalizer.inverse_target(target)
    print(target)
    print(all(target == [1, 2, 3, 4]))


if __name__ == '__main__':
    main()
