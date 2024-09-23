from typing import Any, Optional
import pandas as pd
from .base import Model
from .forecast import StatisticalParametersExtractor


class QuickAI:
    def __init__(self, dataset: pd.DataFrame, target: Optional[str] = None):
        """ Documentation TODO """
        if target is None:
            # target = find_target()
            raise NotImplementedError(
                "Target automatic detection is not implemented yet, provide target column name")
        self._input_check(dataset, target)
        self.model: Model = None
        target_data = dataset[target]
        data = dataset.drop(columns=[target])
        try:
            extractor = StatisticalParametersExtractor(data, target_data)
            dataset_params = extractor.describe_dataset()
        except Exception:
            raise Exception("Dataset description failed")

        try:
            # = forecast.guess_best_model(dataset_params)
            self.model = None
        except Exception:
            raise Exception("Model optimization failed")

        try:
            # = forecast.find_best_model(dataset, target, dataset_params)
            self.model = self.model
        except Exception:
            print("Trying to find better model failed")

        try:
            self.model.train(data, target_data)
        except Exception:
            raise Exception("Model training failed")

    def _input_check(self, dataset: pd.DataFrame, target: str):
        assert dataset is not None and isinstance(dataset, pd.DataFrame)
        assert len(dataset) > 0
        assert target is not None and isinstance(target, str)
        assert target in dataset.columns

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return self.model.predict(X)

    def __call__(self, X: pd.DataFrame) -> pd.Series:
        return self.predict(X)
