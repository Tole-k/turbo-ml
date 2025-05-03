import pandas as pd
from pandas.api.types import infer_dtype


class TypeInferer():
    def __init__(self) -> None:
        super().__init__()

    def infer(self, data: pd.DataFrame) -> pd.Series:
        self.x_dtypes = data.apply(lambda x: infer_dtype(x, skipna=True))
        return self.x_dtypes

    def infer_target(self, target: pd.Series) -> str:
        self.y_dtype = infer_dtype(target, skipna=True)
        return self.y_dtype

    def recall(self) -> pd.Series:
        return self.x_dtypes

    def recall_target(self) -> str:
        return self.y_dtype
