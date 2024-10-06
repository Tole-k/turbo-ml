from ..base.preprocess import Preprocessor
from typing import List
import pandas as pd


class CombinedPreprocessor(Preprocessor):
    def __init__(self, *args) -> None:
        super().__init__()
        self.models: List[Preprocessor] = list(args)

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for model in self.models:
            data = model.fit_transform(data)
        return data

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for model in self.models:
            data = model.transform(data)
        return data

    def fit_transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        for model in self.models:
            target = model.fit_transform_target(target)
        return target

    def transform_target(self, target: pd.Series) -> pd.DataFrame | pd.Series:
        for model in self.models:
            target = model.transform_target(target)
        return target


if __name__ == "__main__":
    from .nan_imputer import NanImputer
    from .one_hot_encoder import OneHotEncoder
    import pandas as pd
    dataframe = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [1, 2, 3, 2, 5],
        "c": ["a", "b", "c", "d", "e"],
        "d": ["a", "b", "c", "d", "e"],
        "e": [1, 2, 3, 4, 5],
        "f": [1, 2, 3, None, 5],
        "g": ["a", "b", "c", "d", "e"],
        "h": ["a", "b", "c", "d", "e"],
        "target": [1, 0, 1, 0, 1]
    })
    target = dataframe["target"]
    dataframe.drop(columns=["target"], inplace=True)
    preprocessor = CombinedPreprocessor(NanImputer(), OneHotEncoder())
    result = preprocessor.fit_transform(dataframe)
    print(result)
